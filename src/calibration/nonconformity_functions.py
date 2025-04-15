from typing import Callable, Dict


def _hinge_loss(y_true, y_pred):
    pass


def _margin_score(y_true, y_pred):
    pass


def _pip_score(y_true, y_pred):
    pass


NONCONFORMITY_FN_DIC: Dict[str, Callable[..., float]] = {
    "hinge": _hinge_loss,
    "margin": _margin_score,
    "pip": _pip_score,
}
