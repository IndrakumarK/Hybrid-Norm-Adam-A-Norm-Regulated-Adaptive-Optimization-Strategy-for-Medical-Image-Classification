"""
Ablation study definitions for the HNAdam manuscript.

The three configurations are:

A. Adam with adaptive norm adjustment.
B. Adam/AMSGrad switching without adaptive norm adjustment.
C. Full HNAdam with adaptive norm adjustment, hybrid switching,
   and norm clipping.

from typing import Any, Dict

ABLATION_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "adam_plus_adaptive_norm": {
        "description": "Adam with adaptive norm adjustment",
        "optimizer": None,
        "requires_equation_verification": True,
    },

    "hybrid_switch_without_adaptive_norm": {
        "description": (
            "Adam/AMSGrad switching without adaptive norm adjustment"
        ),
        "optimizer": None,
        "requires_equation_verification": True,
    },

    "full_hnadam": {
        "description": (
            "Complete HNAdam with adaptive norm adjustment, "
            "hybrid switching, and norm clipping"
        ),
        "optimizer": "HNAdam",
        "requires_equation_verification": False,
    },
}


def validate_ablation_name(name: str) -> Dict[str, Any]:
    """
    Validate an ablation configuration before execution.

    Args:
        name: Ablation configuration identifier.

    Returns:
        Dictionary containing the selected configuration.

    Raises:
        ValueError: If the configuration name is unknown.
        NotImplementedError: If its update equations have not been verified.
    """

    if name not in ABLATION_DEFINITIONS:
        available = ", ".join(ABLATION_DEFINITIONS.keys())
        raise ValueError(
            f"Unknown ablation '{name}'. "
            f"Available configurations: {available}"
        )

    definition = ABLATION_DEFINITIONS[name]

    if definition["requires_equation_verification"]:
        raise NotImplementedError(
            f"The '{name}' configuration cannot be executed yet. "
            "Verify its exact update equations against the final "
            "HNAdam manuscript before implementation."
        )

    return definition
