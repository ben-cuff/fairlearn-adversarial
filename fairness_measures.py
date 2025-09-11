import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


def prepare_data(predictions, targets, demographics):
    """
    Standardizes the predictions and targets, and returns them along with the demographics.
    Parameters:
    -----------
    predictions : array-like
        The predicted values to be standardized.
    targets : array-like
        The target values to be standardized.
    demographics : array-like
        The demographic attributes associated with the data.
    Returns:
    --------
    tuple
        A tuple containing:
        - S_standardized (array-like): Standardized predictions.
        - Y_standardized (array-like): Standardized targets.
        - A (array-like): Demographic attributes (unchanged).
    """

    S_standardized = (predictions - np.mean(predictions)) / np.std(predictions)
    Y_standardized = (targets - np.mean(targets)) / np.std(targets)
    A = demographics

    return S_standardized, Y_standardized, A


def train_classifiers(S, Y, A, split_data=True):
    """
    Trains three classifiers:
    1) Predict A from S
    2) Predict A from Y
    3) Predict A from [S,Y] combined
    Returns classifiers and log-loss on validation data.
    Parameters:
    -----------
    S : array-like
        Standardized predictions.
    Y : array-like
        Standardized targets.
    A : array-like
        Demographic attributes.
    split_data : bool, optional (default=True)
        Whether to split the data into training and testing sets.
    Returns:
    --------
    tuple
        A tuple containing classifiers and their respective log-loss values.
    """
    if split_data:
        S_train, S_test, Y_train, Y_test, A_train, A_test = train_test_split(
            S, Y, A, test_size=0.2, random_state=42
        )
    else:
        S_train, S_test = S, S
        Y_train, Y_test = Y, Y
        A_train, A_test = A, A

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

    return (independence, loss1), (separation, loss2), (sufficiency, loss3)


def calculate_base_statistics(A):
    """
    Calculates P(A=0), P(A=1) and entropy H[A].
    Works for binary demographics A ∈ {0,1}.
    """
    n_total = len(A)
    n_group_0 = np.sum(A == 0)
    n_group_1 = np.sum(A == 1)

    p0 = n_group_0 / n_total
    p1 = n_group_1 / n_total

    entropy = 0
    if p0 > 0:
        entropy -= p0 * np.log(p0)
    if p1 > 0:
        entropy -= p1 * np.log(p1)

    return {"P(A=0)": p0, "P(A=1)": p1, "H[A]": entropy}


def calculate_independence_metric(S, A, independence, base_stats):
    """
    Calculates the normalized mutual information between predictions (S)
    and demographics (A), using independence's probability outputs.

    Parameters:
    - S: standardized predictions (numpy array)
    - A: demographics (numpy array, binary 0/1)
    - independence: trained logistic regression classifier (predicts A from S)
    - base_stats: dictionary from calculate_base_statistics (contains P(A) and H[A])

    Returns:
    - normalized_metric: value in [0, 1], higher = more demographic information in predictions
    """
    predicted_probs = independence.predict_proba(S.reshape(-1, 1))
    p0, p1 = base_stats["P(A=0)"], base_stats["P(A=1)"]
    H_A = base_stats["H[A]"]

    if np.isclose(H_A, 0.0):
        return 0.0

    mutual_information = 0.0
    n_samples = len(S)

    for i in range(n_samples):
        actual_group = A[i]
        p_conditional = predicted_probs[i, actual_group]

        p_marginal = p0 if actual_group == 0 else p1

        if p_conditional > 0 and p_marginal > 0:
            mutual_information += np.log(p_conditional / p_marginal)

    mutual_information /= n_samples

    normalized_metric = mutual_information / H_A
    return normalized_metric
