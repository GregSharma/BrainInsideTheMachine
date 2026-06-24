"""Dataset loaders. Each returns (positives, negatives) lists of strings."""
import sys
sys.path.insert(0, "/home/greg/Desktop/Projects/BrainInsideTheMachine")


def load_uncertainty():
    from uncertainty_dataset import GENUINE_UNCERTAINTY, PERFORMED_CONFIDENCE
    return list(GENUINE_UNCERTAINTY), list(PERFORMED_CONFIDENCE)


CUES = {
    "uncertainty": "genuinely uncertain exploring possibilities conditional reasoning diagnostic hedging",
}
