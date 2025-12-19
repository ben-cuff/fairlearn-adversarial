import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_curve, auc


class FairnessMetrics:
    def __init__(self, predictions, targets, demographics):
        """
        Initializes the FairnessMetrics class with predictions, targets, and demographics.
        Standardizes the predictions and targets, and ensures demographics are binary integers.
        """
        self.S = np.array((predictions - np.mean(predictions)) / np.std(predictions))
        self.Y = np.array((targets - np.mean(targets)) / np.std(targets)).reshape(-1)
        self.A = np.array(demographics).astype(int).reshape(-1)

        unique_A = np.unique(self.A)
        if not np.all(np.isin(unique_A, [0, 1])):
            raise ValueError(f"Demographics A must be binary (0/1). Got values: {unique_A}")

    def train_classifiers(self, split_data=True):
        """
        Trains three classifiers:
        1) Predict A from S
        2) Predict A from Y
        3) Predict A from [S, Y] combined
        Returns classifiers and log-loss on validation data.
        """
        if split_data:
            S_train, S_test, Y_train, Y_test, A_train, A_test = train_test_split(
                self.S, self.Y, self.A, test_size=0.2, random_state=42
            )
        else:
            S_train, S_test = self.S, self.S
            Y_train, Y_test = self.Y, self.Y
            A_train, A_test = self.A, self.A

        independence = LogisticRegression(max_iter=1000)
        independence.fit(S_train.reshape(-1, 1), A_train)
        preds1 = independence.predict_proba(S_test.reshape(-1, 1))
        loss1 = log_loss(A_test, preds1)

        separation = LogisticRegression(max_iter=1000)
        separation.fit(Y_train.reshape(-1, 1), A_train)
        preds2 = separation.predict_proba(Y_test.reshape(-1, 1))
        loss2 = log_loss(A_test, preds2)

        combined_train = np.column_stack((S_train, Y_train))
        combined_test = np.column_stack((S_test, Y_test))

        sufficiency = LogisticRegression(max_iter=1000)
        sufficiency.fit(combined_train, A_train)
        preds3 = sufficiency.predict_proba(combined_test)
        loss3 = log_loss(A_test, preds3)

        self.classifiers = {
            "independence": (independence, loss1),
            "separation": (separation, loss2),
            "sufficiency": (sufficiency, loss3),
        }

    def calculate_base_statistics(self):
        """
        Calculates P(A=0), P(A=1) and entropy H[A].
        Works for binary demographics A ∈ {0,1}.
        """
        n = len(self.A)
        n_group_0 = np.sum(self.A == 0)
        n_group_1 = np.sum(self.A == 1)

        p0 = n_group_0 / n
        p1 = n_group_1 / n

        entropy = 0
        if p0 > 0:
            entropy -= p0 * np.log(p0)
        if p1 > 0:
            entropy -= p1 * np.log(p1)

        self.base_stats = {"P(A=0)": p0, "P(A=1)": p1, "H[A]": entropy}

    def calculate_independence_metric(self):
        independence = self.classifiers["independence"][0]
        probs = independence.predict_proba(self.S.reshape(-1, 1))
        n = len(self.A)

        class_to_idx = {c: i for i, c in enumerate(independence.classes_)}
        indices = np.array([class_to_idx[int(a)] for a in np.ravel(self.A)])

        p_conditional = probs[np.arange(n), indices]

        p0 = float(self.base_stats["P(A=0)"])
        p1 = float(self.base_stats["P(A=1)"])
        p_marginal = np.where(np.ravel(self.A) == 0, p0, p1)

        valid = (p_conditional > 0) & (p_marginal > 0)

        if not np.any(valid):
            return 0.0
        mutual_information = np.mean(np.log(p_conditional[valid] / p_marginal[valid]))
        H_A = self.base_stats["H[A]"]
        return 0.0 if np.isclose(H_A, 0.0) else mutual_information / H_A

    def calculate_separation_metric(self):
        classifier_2 = self.classifiers["separation"][0]
        classifier_3 = self.classifiers["sufficiency"][0]

        probs_marg = classifier_2.predict_proba(self.Y.reshape(-1, 1))
        probs_joint = classifier_3.predict_proba(np.column_stack((self.S, self.Y)))

        n = len(self.A)

        map2 = {c: i for i, c in enumerate(classifier_2.classes_)}
        map3 = {c: i for i, c in enumerate(classifier_3.classes_)}

        idx_m = np.array([map2[int(a)] for a in np.ravel(self.A)])
        idx_j = np.array([map3[int(a)] for a in np.ravel(self.A)])

        p_marginal = probs_marg[np.arange(n), idx_m]
        p_joint = probs_joint[np.arange(n), idx_j]

        conditional_entropy = (
            -np.mean(np.log(p_marginal[p_marginal > 0])) if np.any(p_marginal > 0) else 0.0
        )

        valid = (p_joint > 0) & (p_marginal > 0)
        if not np.any(valid):
            return 0.0

        conditional_mutual_info = np.mean(np.log(p_joint[valid] / p_marginal[valid]))
        return conditional_mutual_info / conditional_entropy

    def calculate_sufficiency_metric(self):
        classifier_1 = self.classifiers["independence"][0]
        classifier_3 = self.classifiers["sufficiency"][0]

        probs_marg = classifier_1.predict_proba(self.S.reshape(-1, 1))
        probs_joint = classifier_3.predict_proba(np.column_stack((self.S, self.Y)))

        n = len(self.A)

        map1 = {c: i for i, c in enumerate(classifier_1.classes_)}
        map3 = {c: i for i, c in enumerate(classifier_3.classes_)}

        idx_m = np.array([map1[int(a)] for a in np.ravel(self.A)])
        idx_j = np.array([map3[int(a)] for a in np.ravel(self.A)])

        p_marginal = probs_marg[np.arange(n), idx_m]
        p_joint = probs_joint[np.arange(n), idx_j]

        conditional_entropy = (
            -np.mean(np.log(p_marginal[p_marginal > 0])) if np.any(p_marginal > 0) else 0.0
        )

        valid = (p_joint > 0) & (p_marginal > 0)
        if not np.any(valid):
            return 0.0

        conditional_mutual_info = np.mean(np.log(p_joint[valid] / p_marginal[valid]))
        return conditional_mutual_info / conditional_entropy

    def calculate_metrics_with_cross_validation(self, k=10):
        """
        Calculates independence, separation, and sufficiency metrics using k-fold cross-validation.
        Returns mean and standard deviation for each metric.
        """
        results = []
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        for train_indices, test_indices in kf.split(self.S):
            S_train, Y_train, A_train = (
                self.S[train_indices],
                self.Y[train_indices],
                self.A[train_indices],
            )

            classifiers = FairnessMetrics(S_train, Y_train, A_train)
            classifiers.train_classifiers(split_data=False)

            classifiers.calculate_base_statistics()

            independence = classifiers.calculate_independence_metric()
            separation = classifiers.calculate_separation_metric()
            sufficiency = classifiers.calculate_sufficiency_metric()

            results.append([independence, separation, sufficiency])

        results = np.array(results)
        mean_metrics = dict(
            zip(["independence", "separation", "sufficiency"], results.mean(axis=0))
        )
        std_metrics = dict(zip(["independence", "separation", "sufficiency"], results.std(axis=0)))

        self.train_classifiers(split_data=False)

        return {"mean_metrics": mean_metrics, "std_metrics": std_metrics}

    def plot_roc_auc_curve(self, classifier_name):
        """
        Plots the ROC AUC curve for the specified classifier.
        classifier_name: str, one of ["independence", "separation", "sufficiency"]
        """
        import matplotlib.pyplot as plt

        if classifier_name not in self.classifiers:
            raise ValueError(
                f"Invalid classifier name. Choose from {list(self.classifiers.keys())}."
            )

        classifier = self.classifiers[classifier_name][0]
        if classifier_name == "sufficiency":
            probs = classifier.predict_proba(np.column_stack((self.S, self.Y)))[:, 1]
        elif classifier_name == "separation":
            probs = classifier.predict_proba(self.Y.reshape(-1, 1))[:, 1]
        else:
            probs = classifier.predict_proba(self.S.reshape(-1, 1))[:, 1]

        fpr, tpr, _ = roc_curve(self.A, probs)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve for {classifier_name.capitalize()} Classifier")
        plt.legend(loc="lower right")
        plt.show()

    def get_roc_curve_data(self, classifier_name):
        """
        Returns ROC curve data for the specified classifier without plotting.
        Output dict contains: fpr, tpr, thresholds, auc.
        classifier_name: str, one of ["independence", "separation", "sufficiency"]
        """
        if classifier_name not in self.classifiers:
            raise ValueError(
                f"Invalid classifier name. Choose from {list(self.classifiers.keys())}."
            )

        classifier = self.classifiers[classifier_name][0]
        if classifier_name == "sufficiency":
            probs = classifier.predict_proba(np.column_stack((self.S, self.Y)))[:, 1]
        elif classifier_name == "separation":
            probs = classifier.predict_proba(self.Y.reshape(-1, 1))[:, 1]
        else:
            probs = classifier.predict_proba(self.S.reshape(-1, 1))[:, 1]

        fpr, tpr, thresholds = roc_curve(self.A, probs)
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(auc(fpr, tpr)),
        }
