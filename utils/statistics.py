"""
Paired-seed statistical comparisons for HNAdam optimizer experiments.
"""

import itertools

import numpy as np
from scipy import stats


def exact_sign_flip_pvalue(differences):
    """
    Calculate the exact two-sided sign-flip permutation p-value.
    """
    differences = np.asarray(
        differences,
        dtype=float,
    ).reshape(-1)

    differences = differences[
        np.isfinite(differences)
    ]

    n = len(differences)

    if n == 0:
        return float("nan")

    observed_statistic = abs(np.mean(differences))

    permutation_statistics = []

    for signs in itertools.product([-1, 1], repeat=n):
        signs = np.asarray(signs, dtype=float)

        statistic = abs(
            np.mean(differences * signs)
        )

        permutation_statistics.append(statistic)

    permutation_statistics = np.asarray(
        permutation_statistics
    )

    p_value = np.mean(
        permutation_statistics
        >= observed_statistic - 1e-12
    )

    return float(p_value)


def holm_adjust(p_values):
    """
    Apply Holm-Bonferroni correction to multiple p-values.
    """
    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    if p_values.ndim != 1:
        raise ValueError("p_values must be one-dimensional.")

    if np.any(~np.isfinite(p_values)):
        raise ValueError("p_values must contain finite values.")

    if np.any((p_values < 0) | (p_values > 1)):
        raise ValueError("p_values must be between 0 and 1.")

    number_of_tests = len(p_values)

    if number_of_tests == 0:
        return []

    sorted_indices = np.argsort(p_values)

    adjusted_values = np.empty(
        number_of_tests,
        dtype=float,
    )

    running_maximum = 0.0

    for rank, index in enumerate(sorted_indices):
        adjusted_value = (
            number_of_tests - rank
        ) * p_values[index]

        running_maximum = max(
            running_maximum,
            adjusted_value,
        )

        adjusted_values[index] = min(
            running_maximum,
            1.0,
        )

    return adjusted_values.tolist()


def paired_comparison(reference, candidate):
    """
    Perform paired statistical comparison between two methods.

    Args:
        reference: Reference optimizer scores across random seeds.
        candidate: Candidate optimizer scores across the same seeds.

    Returns:
        Dictionary containing:
            n
            mean_difference
            ci95
            t_statistic
            p_value
            cohens_dz
            exact_sign_flip_p
    """
    reference = np.asarray(
        reference,
        dtype=float,
    )

    candidate = np.asarray(
        candidate,
        dtype=float,
    )

    if (
        reference.ndim != 1
        or candidate.ndim != 1
        or reference.shape != candidate.shape
    ):
        raise ValueError(
            "Inputs must be paired one-dimensional arrays "
            "of equal length."
        )

    if len(reference) < 2:
        raise ValueError(
            "At least two paired runs are required."
        )

    if (
        not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(candidate))
    ):
        raise ValueError(
            "Input scores must contain finite values."
        )

    differences = candidate - reference
    n = len(differences)

    mean_difference = float(
        np.mean(differences)
    )

    standard_deviation = np.std(
        differences,
        ddof=1,
    )

    standard_error = stats.sem(differences)

    if standard_deviation > 0:
        t_statistic, p_value = stats.ttest_rel(
            candidate,
            reference,
        )

        cohens_dz = (
            mean_difference / standard_deviation
        )

        critical_value = stats.t.ppf(
            0.975,
            df=n - 1,
        )

        confidence_interval = [
            mean_difference - critical_value * standard_error,
            mean_difference + critical_value * standard_error,
        ]

        t_statistic = float(t_statistic)
        p_value = float(p_value)
        cohens_dz = float(cohens_dz)

    else:
        if mean_difference == 0:
            t_statistic = 0.0
            p_value = 1.0
            cohens_dz = float("nan")
            confidence_interval = [0.0, 0.0]
        else:
            t_statistic = float(
                np.sign(mean_difference) * np.inf
            )
            p_value = 0.0
            cohens_dz = float("inf")
            confidence_interval = [
                mean_difference,
                mean_difference,
            ]

    return {
        "n": n,
        "mean_difference": mean_difference,
        "ci95": [
            float(confidence_interval[0]),
            float(confidence_interval[1]),
        ],
        "t_statistic": t_statistic,
        "p_value": p_value,
        "cohens_dz": cohens_dz,
        "exact_sign_flip_p": exact_sign_flip_pvalue(
            differences
        ),
    }