"""matplotlib style of the figures: rcParams and per-method marks, both read from the specification."""

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402

from .spec import FiguresSpec  # noqa: E402


def apply_style(spec: FiguresSpec, extra: dict | None = None) -> None:
    """Reset matplotlib, then apply the spec's rcParams and the figure's own overrides."""
    mpl.rcdefaults()
    mpl.rcParams.update(spec.style)
    if extra:
        mpl.rcParams.update(extra)


def method_style(spec: FiguresSpec, method: str, **overrides) -> dict:
    """Colour, line style and marker of a method; a hollow marker has a white face."""
    m = spec.methods[method]
    style = {
        "color": m.color,
        "linestyle": m.linestyle,
        "marker": m.marker,
        "markerfacecolor": "white" if m.hollow else m.color,
        "markeredgecolor": m.color,
        "markeredgewidth": 1.2,
    }
    style.update(overrides)
    return style
