"""gsipe_v1 — Sipe-lab publication matplotlib theme.

A self-contained style module in the same shape as :mod:`cold_field_v5`:
a :class:`Theme` class with :meth:`Theme.apply` that pushes rcParams
globally, plus a few axis helpers.  Register via ``style/__init__.py`` and
activate with :func:`databench.plotting.set_theme`::

    from databench.plotting import set_theme
    set_theme(style="gsipe_v1")

Design targets (publication standard)
-------------------------------------
* SVG output, 72 dpi.  Figures are sized relative to the US-Letter text
  block (6.5 x 9 in) via :func:`figure_size`, *not* to the full page —
  so they drop into a letter-size document and scale cleanly.
* White background; **all** text / axes / ticks true black ``#000000``.
* Type scale: suptitle 18, axes title 10, axis labels 8, ticks 8 pt.
* 0.75 pt axes border and tick width; 0.25 pt marker edges.
* Ticks point *out*; use :func:`nice_ticks` to keep 3-5 ticks per axis
  and force each axis to **end on a tick**.
* Declutter defaults: open two-spine axes (top/right hidden) and
  ``constrained_layout`` on, so multi-panel titles/labels never collide.

The colour palette is loaded from ``gsipe_colors.csv`` (extracted from the
lab's ``gsipe_colors.ase``) so colours live in data, not code.
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, cast

import matplotlib as mpl
from cycler import cycler
from matplotlib.ticker import MaxNLocator

# ════════════════════════════════════════════════════════
#  PALETTE  (loaded from CSV)
# ════════════════════════════════════════════════════════

_CSV_PATH = Path(__file__).with_name("gsipe_colors.csv")


def _load_palette(path: Path = _CSV_PATH) -> "OrderedDict[str, List[str]]":
    """Read ``family,shade,hex`` rows into ``{family: [hex, ...]}`` (shade order)."""
    rows: Dict[str, Dict[int, str]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            fam = row["family"].strip()
            rows.setdefault(fam, {})[int(row["shade"])] = row["hex"].strip()
    pal: "OrderedDict[str, List[str]]" = OrderedDict()
    for fam, shades in rows.items():
        pal[fam] = [shades[s] for s in sorted(shades)]
    return pal


# Full palette: family name -> list of 5 hex strings, darkest (1) to lightest (5).
PALETTE: "OrderedDict[str, List[str]]" = _load_palette()

# Flat name lookup, e.g. SWATCHES["Blue3"] -> "#005AC2".
SWATCHES: Dict[str, str] = {
    f"{fam}{i + 1}": hexv
    for fam, shades in PALETTE.items()
    for i, hexv in enumerate(shades)
}


def swatch(family: str, shade: int = 3) -> str:
    """Return one hex colour, e.g. ``swatch("Blue", 3)``.  Shade is 1 (dark)–5 (light)."""
    return PALETTE[family][shade - 1]


# Default qualitative cycle: distinct hues at the saturated mid-tone (shade 3),
# ordered blue/orange/green-first for colourblind legibility of the early entries.
_CYCLE_FAMILIES = (
    "Blue", "Orange", "Green", "Red", "Purple",
    "Teal", "Magenta", "Yellow", "Brown", "Gray",
)
DEFAULT_CYCLE: List[str] = [swatch(fam, 3) for fam in _CYCLE_FAMILIES]

_BLACK = "#000000"
_WHITE = "#FFFFFF"


# ════════════════════════════════════════════════════════
#  PAGE GEOMETRY  (size figures relative to US-Letter)
# ════════════════════════════════════════════════════════
# 8.5 x 11 in is the *paper*, not the figure.  Figures are sized to the
# text block (paper minus margins) so they drop into a letter-size
# document and scale cleanly — never set a figure to the full page.

PAGE_IN = (8.5, 11.0)        # US-Letter sheet
MARGIN_IN = 1.0              # assumed margin on every side
CONTENT_W = PAGE_IN[0] - 2 * MARGIN_IN   # 6.5 in — full text width
CONTENT_H = PAGE_IN[1] - 2 * MARGIN_IN   # 9.0 in — full text height

# Named fractions of the content width for common figure footprints.
_WIDTH_FRACTIONS = {
    "full": 1.0, "twothirds": 2 / 3, "half": 0.5, "third": 1 / 3,
}

_GOLDEN = 0.618              # default height:width when none is given


def figure_size(width="full", height=None, *, aspect=_GOLDEN):
    """Figure size (inches) relative to the US-Letter text block.

    Parameters
    ----------
    width : float or str
        Fraction of the 6.5 in content width (0–1), or a name:
        ``"full"`` (1.0), ``"twothirds"``, ``"half"``, ``"third"``.
    height : float, optional
        Fraction of the 9.0 in content height.  If omitted, the height
        is derived from the width via *aspect* (so the figure keeps a
        fixed shape regardless of how wide it is).
    aspect : float
        Height-to-width ratio used only when *height* is None
        (default golden ratio ≈ 0.618).

    Returns
    -------
    (w_in, h_in) : tuple[float, float]

    Examples
    --------
    >>> figure_size("full")            # full-width, golden-ratio tall
    (6.5, 4.017)
    >>> figure_size("half", aspect=1)  # half-width square
    (3.25, 3.25)
    >>> figure_size(1.0, height=0.78)  # near-full-page multi-panel
    (6.5, 7.02)
    """
    frac = _WIDTH_FRACTIONS.get(width, width) if isinstance(width, str) else width
    w = float(frac) * CONTENT_W
    h = float(height) * CONTENT_H if height is not None else w * aspect
    return (round(w, 4), round(h, 4))


# Default figure footprint: full text width, golden-ratio height.
DEFAULT_FIGSIZE = figure_size("full")


# ════════════════════════════════════════════════════════
#  THEME
# ════════════════════════════════════════════════════════

class Theme:
    """Publication theme.  ``apply()`` pushes the rcParams spec globally.

    A *mode* argument is accepted for API compatibility with the style
    registry, but this theme has a single (light, black-on-white)
    appearance — any mode resolves to it.
    """

    def __init__(self, mode: str = "light"):
        self.mode = mode

    # — palette accessors (mirror the cold_field_v5 interface) —
    @property
    def colors(self) -> List[str]:
        return list(DEFAULT_CYCLE)

    @property
    def palette(self) -> "OrderedDict[str, List[str]]":
        return PALETTE

    @property
    def bg(self) -> str:
        return _WHITE

    @property
    def surface(self) -> str:
        return _WHITE

    @property
    def fg(self) -> str:
        return _BLACK

    @property
    def shade(self) -> str:
        return swatch("Gray", 5)

    @property
    def lbl_bg(self) -> str:
        return _WHITE

    @property
    def bar_hi(self) -> str:
        return swatch("Gray", 2)

    @property
    def bar_lo(self) -> str:
        return swatch("Gray", 5)

    @property
    def dot_color(self) -> str:
        return _BLACK

    @property
    def dot_alpha(self) -> float:
        return 0.35

    def apply(self) -> "Theme":
        """Push this theme's colours and sizing into ``mpl.rcParams``."""
        mpl.rcParams.update({
            # ── backend & figure ──────────────────────────────
            "savefig.format":        "svg",
            "figure.figsize":        list(DEFAULT_FIGSIZE),
            "figure.dpi":            72,
            "savefig.dpi":           72,
            "figure.facecolor":      _WHITE,
            "axes.facecolor":        _WHITE,
            "savefig.facecolor":     _WHITE,
            "savefig.transparent":   False,
            "savefig.bbox":          "tight",

            # ── colours: true black everywhere ────────────────
            "text.color":            _BLACK,
            "axes.edgecolor":        _BLACK,
            "axes.labelcolor":       _BLACK,
            "xtick.color":           _BLACK,
            "ytick.color":           _BLACK,
            "xtick.labelcolor":      _BLACK,
            "ytick.labelcolor":      _BLACK,
            "axes.titlecolor":       _BLACK,
            "axes.prop_cycle":       cycler(color=DEFAULT_CYCLE),

            # ── font sizes ────────────────────────────────────
            "figure.titlesize":      18,
            "axes.titlesize":        10,
            "axes.labelsize":        8,
            "xtick.labelsize":       8,
            "ytick.labelsize":       8,
            "legend.fontsize":       8,
            "font.size":             8,

            # ── lines & strokes ───────────────────────────────
            "axes.linewidth":        0.75,
            "xtick.major.width":     0.75,
            "ytick.major.width":     0.75,
            "xtick.minor.width":     0.75,
            "ytick.minor.width":     0.75,
            "lines.linewidth":       1.2,
            "lines.markersize":      4,
            "lines.markeredgewidth": 0.25,

            # ── ticks ─────────────────────────────────────────
            "xtick.direction":       "out",
            "ytick.direction":       "out",
            "xtick.major.size":      3.0,
            "ytick.major.size":      3.0,

            # ── layout & spines (declutter defaults) ──────────
            "figure.constrained_layout.use":   True,
            "figure.constrained_layout.h_pad": 0.06,
            "figure.constrained_layout.w_pad": 0.06,
            "figure.constrained_layout.hspace": 0.05,
            "figure.constrained_layout.wspace": 0.05,
            "axes.spines.top":       False,
            "axes.spines.right":     False,

            # ── misc ──────────────────────────────────────────
            "axes.titleweight":      "normal",
            "axes.grid":             False,
            "legend.frameon":        False,
            "svg.fonttype":          "none",   # keep text selectable in SVG
            "pdf.fonttype":          42,
            "ps.fonttype":           42,
        })
        return self


# ════════════════════════════════════════════════════════
#  AXIS HELPERS
# ════════════════════════════════════════════════════════

def nice_ticks(ax, *, nbins: int = 4, axis: str = "both",
               integer: bool = False) -> None:
    """Place ~3–5 ticks per axis and make the axis **end exactly on a tick**.

    ``rcParams`` cannot express "end on a tick" — it depends on the data
    range, which only exists after plotting.  This helper is the mechanism:
    it picks a nice step with :class:`~matplotlib.ticker.MaxNLocator`, then
    **expands** the view limits *outward* to the nearest tick beyond the data
    (so edge points are never clipped) and pins the ticks across that span.

    Call **after** plotting all data, and after any manual ``set_xlim`` /
    ``set_ylim`` (the limits in force at call time define the data span to
    enclose), and before saving.

    Parameters
    ----------
    nbins : int
        Target number of intervals; tick count is roughly ``nbins`` (+1).
    axis : ``"x"``, ``"y"`` or ``"both"``
        Which axis/axes to adjust.
    integer : bool
        Force integer tick locations (e.g. for count data).
    """
    import math
    eps = 1e-9

    def _do(a, get_lim, set_lim):
        lo, hi = get_lim()
        if hi < lo:
            lo, hi = hi, lo
        if hi - lo < eps:                       # degenerate range; leave it
            return
        ticks = MaxNLocator(nbins=nbins, integer=integer).tick_values(lo, hi)
        if len(ticks) < 2:
            return
        step = ticks[1] - ticks[0]
        if step <= 0:
            return
        new_lo = step * math.floor(lo / step + eps)
        new_hi = step * math.ceil(hi / step - eps)
        n = int(round((new_hi - new_lo) / step))
        a.set_ticks([new_lo + i * step for i in range(n + 1)])
        set_lim(new_lo, new_hi)

    if axis in ("x", "both"):
        _do(ax.xaxis, ax.get_xlim, ax.set_xlim)
    if axis in ("y", "both"):
        _do(ax.yaxis, ax.get_ylim, ax.set_ylim)


def clean_ax(ax, offset: int = 0) -> None:
    """Ensure top/right spines are hidden (theme default) and optionally
    offset the remaining left/bottom spines outward by *offset* points.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if offset:
        ax.spines["left"].set_position(("outward", offset))
        ax.spines["bottom"].set_position(("outward", offset))


__all__ = [
    "Theme", "PALETTE", "SWATCHES", "DEFAULT_CYCLE",
    "swatch", "nice_ticks", "clean_ax",
    "figure_size", "DEFAULT_FIGSIZE",
    "PAGE_IN", "MARGIN_IN", "CONTENT_W", "CONTENT_H",
]
