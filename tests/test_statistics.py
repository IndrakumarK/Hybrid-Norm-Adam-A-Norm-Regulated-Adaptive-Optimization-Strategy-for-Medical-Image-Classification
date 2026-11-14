from utils.statistics import exact_sign_flip_pvalue, paired_comparison


def test_sign_flip_minimum_for_five_same_sign():
    differences = [1, 1, 1, 1, 1]

    p_value = exact_sign_flip_pvalue(differences)

    assert p_value == 0.0625


def test_paired_comparison():
    hnadam_scores = [0.80, 0.81, 0.79, 0.82, 0.80]
    baseline_scores = [0.82, 0.83, 0.81, 0.84, 0.82]

    result = paired_comparison(
        hnadam_scores,
        baseline_scores,
    )

    assert result["n"] == 5