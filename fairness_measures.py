import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, KFold


class FairnessMetrics:
    def __init__(self, predictions, targets, demographics):
        """
        Initializes the FairnessMetrics class with predictions, targets, and demographics.
        Standardizes the predictions and targets.
        """
        self.S = (predictions - np.mean(predictions)) / np.std(predictions)
        self.Y = (targets - np.mean(targets)) / np.std(targets)
        self.A = demographics

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
        """
        Calculates the normalized mutual information between predictions (S)
        and demographics (A), using the independence classifier.
        """
        independence = self.classifiers["independence"][0]
        predicted_probs = independence.predict_proba(self.S.reshape(-1, 1))
        p0, p1 = self.base_stats["P(A=0)"], self.base_stats["P(A=1)"]
        H_A = self.base_stats["H[A]"]

        if np.isclose(H_A, 0.0):
            return 0.0

        mutual_information = 0.0
        n = len(self.S)

        for i in range(n):
            actual_group = self.A[i]
            p_conditional = predicted_probs[i, actual_group]

            p_marginal = p0 if actual_group == 0 else p1

            if p_conditional > 0 and p_marginal > 0:
                mutual_information += np.log(p_conditional / p_marginal)

        mutual_information /= n

        normalized_metric = mutual_information / H_A
        return normalized_metric

    def calculate_separation_metric(self):
        """
        Calculates the normalized conditional mutual information between predictions (S)
        and demographics (A), given targets (Y).
        """
        classifier_2 = self.classifiers["separation"][0]
        classifier_3 = self.classifiers["sufficiency"][0]

        n = len(self.S)
        conditional_entropy = 0.0
        for i in range(n):
            p = classifier_2.predict_proba(self.Y[i].reshape(1, -1))[0]
            conditional_entropy -= np.log(p[self.A[i]])
        conditional_entropy /= n

        if np.isclose(conditional_entropy, 0.0):
            return 0.0

        conditional_mutual_info = 0.0
        for i in range(n):
            p_joint = classifier_3.predict_proba(np.array([self.S[i], self.Y[i]]).reshape(1, -1))[
                0
            ]
            p_marginal = classifier_2.predict_proba(self.Y[i].reshape(1, -1))[0]
            conditional_mutual_info += np.log(p_joint[self.A[i]] / p_marginal[self.A[i]])
        conditional_mutual_info /= n

        normalized_metric = conditional_mutual_info / conditional_entropy
        return normalized_metric

    def calculate_sufficiency_metric(self):
        """
        Calculates the normalized conditional mutual information between targets (Y)
        and demographics (A), given predictions (S).
        """
        classifier_1 = self.classifiers["independence"][0]
        classifier_3 = self.classifiers["sufficiency"][0]

        n = len(self.S)
        conditional_entropy = 0.0
        for i in range(n):
            p = classifier_1.predict_proba(self.S[i].reshape(1, -1))[0]
            conditional_entropy -= np.log(p[self.A[i]])
        conditional_entropy /= n

        if np.isclose(conditional_entropy, 0.0):
            return 0.0

        conditional_mutual_info = 0.0
        for i in range(n):
            p_joint = classifier_3.predict_proba(np.array([self.S[i], self.Y[i]]).reshape(1, -1))[
                0
            ]
            p_marginal = classifier_1.predict_proba(self.S[i].reshape(1, -1))[0]
            conditional_mutual_info += np.log(p_joint[self.A[i]] / p_marginal[self.A[i]])
        conditional_mutual_info /= n

        normalized_metric = conditional_mutual_info / conditional_entropy
        return normalized_metric

    def calculate_metrics_with_cross_validation(self, k=10):
        """
        Calculates independence, separation, and sufficiency metrics using k-fold cross-validation.
        """
        results = []
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        for train_indices, test_indices in kf.split(self.S):
            S_train, S_test = self.S[train_indices], self.S[test_indices]
            Y_train, Y_test = self.Y[train_indices], self.Y[test_indices]
            A_train, A_test = self.A[train_indices], self.A[test_indices]

            classifiers = FairnessMetrics(S_train, Y_train, A_train)
            classifiers.train_classifiers(split_data=False)

            base_stats = classifiers.calculate_base_statistics()

            independence = classifiers.calculate_independence_metric()
            separation = classifiers.calculate_separation_metric()
            sufficiency = classifiers.calculate_sufficiency_metric()

            results.append([independence, separation, sufficiency])

        results = np.array(results)
        mean_metrics = results.mean(axis=0)
        std_metrics = results.std(axis=0)

        return mean_metrics, std_metrics
