"""Define the expanding development folds independently of the holdout."""

from dataclasses import dataclass

from .model_features import Window
from .schema import DatasetError

DEV_STEPS = (1, 39)


@dataclass(frozen=True)
class Fold:
    name: str
    train: tuple[int, int]
    validation: tuple[int, int]

    @property
    def inner_early_stopping(self) -> tuple[int, int]:
        """Reserve the final three training steps when early stopping is used."""
        end = self.train[1]
        return end - 2, end


DEV_FOLDS = (
    Fold("F1", (1, 19), (20, 24)),
    Fold("F2", (1, 24), (25, 29)),
    Fold("F3", (1, 29), (30, 34)),
    Fold("F4", (1, 34), (35, 39)),
)


def development_folds(embargo: int = 0) -> tuple[Fold, ...]:
    """Drop the requested number of steps immediately before validation."""
    if type(embargo) is not int or embargo < 0:
        raise ValueError("Embargo must be a non-negative whole number of steps.")
    folds = tuple(
        Fold(fold.name, (fold.train[0], fold.train[1] - embargo), fold.validation)
        for fold in DEV_FOLDS
    )
    for fold in folds:
        first, last = fold.train
        start, end = fold.validation
        if not DEV_STEPS[0] <= first < last - 2 <= last < start <= end <= DEV_STEPS[1]:
            raise ValueError(
                "Fold must have an inner training split and stay in steps 1–39."
            )
    return folds


def validate_development(window: Window) -> None:
    """Refuse a development input whose declared bounds or rows include holdout steps."""
    if (
        (window.first, window.last) != DEV_STEPS
        or not window.steps.between(*DEV_STEPS).all()
        or not window.features.index.equals(window.steps.index)
        or not window.labels.index.equals(window.steps.index)
    ):
        raise DatasetError(
            "Development requires an aligned snapshot limited to steps 1–39."
        )


def slice_development(window: Window, bounds: tuple[int, int]) -> Window:
    """Take complete steps from a development-only snapshot."""
    validate_development(window)
    first, last = bounds
    if not DEV_STEPS[0] <= first <= last <= DEV_STEPS[1]:
        raise DatasetError("Requested development steps are outside 1–39.")
    steps = window.steps[window.steps.between(first, last)]
    if steps.empty:
        raise DatasetError("The requested development window has no transactions.")
    return Window(
        window.features.loc[steps.index],
        window.labels.loc[steps.index],
        steps,
        first,
        last,
    )
