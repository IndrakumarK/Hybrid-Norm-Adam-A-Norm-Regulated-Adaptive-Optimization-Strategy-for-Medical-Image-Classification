"""
Optimizer comparison configuration for the HNAdam experiments.
"""

SUPPORTED_OPTIMIZERS = [
    "HNAdam",
    "Adam",
    "AMSGrad",
    "SGD",
    "AdamW",
]

PAPER_COMPARATORS = [
    "SGD",
    "Adam",
    "AMSGrad",
    "AdamW",
    "RAdam",
    "AdaBelief",
    "HNAdam",
]


def check_comparators(names):
    """
    Check whether the requested optimizers are supported.

    Args:
        names: List of optimizer names.

    Returns:
        True if all requested optimizers are supported.

    Raises:
        ValueError: If one or more optimizers are unsupported.
    """
    unsupported = [
        name for name in names
        if name not in SUPPORTED_OPTIMIZERS
    ]

    if unsupported:
        raise ValueError(
            f"Unsupported optimizer(s): {', '.join(unsupported)}. "
            f"Available optimizers: {', '.join(SUPPORTED_OPTIMIZERS)}"
        )

    return True