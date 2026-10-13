"""Mesomap plotters and the functional grouping of mesomap regions.

Region names carry a hemisphere prefix (``L_``/``R_``) and an Allen acronym.
:data:`REGION_GROUPS` sorts those acronyms into the functional groups a
mesomap figure labels — visual, somatosensory, motor, auditory, retrosplenial,
frontal — and :func:`order_by_group` / :func:`group_blocks` turn a set of
region names into grouped, labelled rows for a heatmap axis.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import (Callable, Dict, Iterable, List, Mapping, Sequence, TYPE_CHECKING,
                    Tuple)
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    import pandas as pd

from databench.plotting import style_axes


# ── Functional groups ──────────────────────────────────────────────────────
# Acronyms are listed rather than matched by prefix: "VISC" (visceral) starts
# with "VIS" but is not a visual area, and "SSp-un" is somatosensory while
# "SSs" is a separate acronym.  Groups are drawn in this order.
REGION_GROUPS: Dict[str, Tuple[str, ...]] = {
    "Vis":   ("VISa", "VISal", "VISam", "VISl", "VISli", "VISp", "VISpl",
              "VISpm", "VISpor", "VISrl"),
    "SS":    ("SSp-bfd", "SSp-ll", "SSp-m", "SSp-n", "SSp-tr", "SSp-ul",
              "SSp-un", "SSs"),
    "Mo":    ("MOp", "MOs"),
    "Aud":   ("AUDd", "AUDp", "AUDpo", "AUDv"),
    "RSP":   ("RSPagl", "RSPd", "RSPv"),
    "Front": ("FRP", "ACAd", "PL"),
    "Other": ("TEa", "VISC"),
}
GROUP_ORDER: Tuple[str, ...] = tuple(REGION_GROUPS)
_GROUP_BY_ACRONYM: Dict[str, str] = {
    acronym: group for group, acronyms in REGION_GROUPS.items()
    for acronym in acronyms
}


def region_acronym(region: str) -> str:
    """Allen acronym of a mesomap region name, without its hemisphere prefix.

    ``"L_SSp-bfd"`` -> ``"SSp-bfd"``.  Names with no prefix pass through.
    """
    prefix, sep, rest = region.partition("_")
    return rest if sep and prefix in ("L", "R") else region


def region_group(region: str) -> str:
    """Functional group of a mesomap region — a key of :data:`REGION_GROUPS`.

    Acronyms absent from the table are grouped as ``"Other"``.
    """
    return _GROUP_BY_ACRONYM.get(region_acronym(region), "Other")


def order_by_group(regions: Iterable[str],
                   rank: Mapping[str, float] | Callable[[str], float] | None = None
                   ) -> List[str]:
    """*regions* sorted into :data:`GROUP_ORDER` blocks.

    Within a group, regions are ordered by *rank* — a mapping or callable from
    region name to a sort key — and by name when *rank* is omitted.
    """
    regions = list(regions)
    if rank is None:
        within = lambda region: region                      # noqa: E731
    elif callable(rank):
        within = rank
    else:
        within = lambda region: rank.get(region, 0.0)       # noqa: E731
    return sorted(regions,
                  key=lambda region: (GROUP_ORDER.index(region_group(region)),
                                      within(region)))


def group_blocks(regions: Sequence[str]) -> List[Tuple[str, int, int]]:
    """``(group, first_index, last_index)`` for each run of one group.

    Takes the output of :func:`order_by_group`; the indices are row positions
    in that sequence, so a heatmap can centre a label on each block and rule a
    line between them.
    """
    blocks: List[Tuple[str, int, int]] = []
    for index, region in enumerate(regions):
        group = region_group(region)
        if blocks and blocks[-1][0] == group:
            blocks[-1] = (group, blocks[-1][1], index)
        else:
            blocks.append((group, index, index))
    return blocks


def plot_stacked_envelopes(t: np.ndarray, envs: Dict[str, np.ndarray], keys: Iterable[str],
                           band_lo: float, band_hi: float, ds: int = 10):
    keys = list(keys)
    fig = plt.figure(figsize=(12, 7))
    for i, k in enumerate(keys, start=1):
        ax = plt.subplot(len(keys), 1, i)
        ax.plot(t[::ds], envs[k][::ds], linewidth=1.0)
        ax.set_ylabel(k)
        if i < len(keys):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("Time (s)")
        style_axes(ax)
    plt.suptitle(
        f"2–5 Hz envelope (Hilbert amplitude), bandpass {band_lo:.0f}–{band_hi:.0f} Hz",
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_spectrogram_panel(specs: Dict[str, tuple], keys: Iterable[str], fmax: float,
                           win_s: float, overlap_frac: float):
    keys = list(keys)
    fig = plt.figure(figsize=(12, 8))
    for i, k in enumerate(keys, start=1):
        f, tt, Sdb = specs[k]
        ax = plt.subplot(2, 2, i)
        ax.pcolormesh(tt, f, Sdb, shading="auto")
        ax.set_ylim(0, fmax)
        ax.set_title(k)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Hz")
        style_axes(ax)
    plt.suptitle(
        f"Spectrograms after 2–5 Hz bandpass (win={win_s:.1f}s, overlap={overlap_frac:.3f})",
        y=0.98,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    return fig

# ════════════════════════════════════════════════════════════════════════════
#  Atlas tables
# ════════════════════════════════════════════════════════════════════════════
# ``analysis_scripts/oscillations/build-mesomap-coords.py`` writes the two tables these
# loaders read.  They are looked for in $DATABENCH_ATLAS_DIR, then in this
# package's ``data/`` directory, then in the analysis_scripts checkout, so a
# checkout works untouched and an installed copy can carry its own.

_ATLAS_FILES = {"outlines": "mesomap_region_outlines.csv",
                "coords": "mesomap_region_coords.csv"}


def _atlas_dirs() -> List[Path]:
    import os
    here = Path(__file__).resolve()
    candidates = [os.environ.get("DATABENCH_ATLAS_DIR"),
                  here.parent / "data",
                  here.parents[2] / "analysis_scripts" / "oscillations"]
    return [Path(d) for d in candidates if d]


def atlas_table_path(kind: str) -> Path:
    """Path of the ``"outlines"`` or ``"coords"`` atlas table.

    Raises :class:`FileNotFoundError` naming every directory searched, so a
    missing table says how to fix itself.
    """
    if kind not in _ATLAS_FILES:
        raise KeyError(f"kind must be one of {sorted(_ATLAS_FILES)}, got {kind!r}")
    searched = _atlas_dirs()
    for directory in searched:
        candidate = directory / _ATLAS_FILES[kind]
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"{_ATLAS_FILES[kind]} not found in " +
        ", ".join(str(d) for d in searched) +
        ". Build it with analysis_scripts/oscillations/build-mesomap-coords.py, or set "
        "$DATABENCH_ATLAS_DIR.")


@lru_cache(maxsize=4)
def load_outlines(path: str | None = None) -> "pd.DataFrame":
    """Region boundary polygons — columns ``region, k, ml_mm, ap_mm``."""
    import pandas as pd
    return pd.read_csv(path or atlas_table_path("outlines"))


@lru_cache(maxsize=4)
def load_coords(path: str | None = None) -> "pd.DataFrame":
    """Region centroids — columns ``region, acronym, hemisphere, ap_mm, ml_mm``."""
    import pandas as pd
    return pd.read_csv(path or atlas_table_path("coords"))


# ════════════════════════════════════════════════════════════════════════════
#  Group colours
# ════════════════════════════════════════════════════════════════════════════

# One palette family per functional group, so every figure in the repo colours
# Vis the same way.  Resolved lazily: importing this module must not touch the
# style registry.
GROUP_FAMILIES: Dict[str, str] = {
    "Vis": "Magenta", "SS": "Teal", "Mo": "Orange", "Aud": "Green",
    "RSP": "Purple", "Front": "Blue", "Other": "Brown",
}


@lru_cache(maxsize=4)
def group_colors(shade: int = 3) -> Dict[str, str]:
    """Functional group -> hex colour, from the active style's palette."""
    from databench.plotting.style.gsipe_v1 import swatch
    return {group: swatch(family, shade)
            for group, family in GROUP_FAMILIES.items()}


def region_colors(regions: Iterable[str], shade: int = 3) -> Dict[str, str]:
    """Region name -> the colour of its functional group."""
    palette = group_colors(shade)
    return {name: palette[region_group(name)] for name in regions}


# ════════════════════════════════════════════════════════════════════════════
#  Correlation maps
# ════════════════════════════════════════════════════════════════════════════

def correlation_scale(*matrices: np.ndarray,
                      diverging: bool | None = None) -> Tuple[object, float, float]:
    """``(cmap, vmin, vmax)`` shared by *matrices*, off-diagonal entries only.

    A matrix with no negative entry gets a sequential scale anchored at zero;
    one with a negative entry gets a diverging scale centred on zero, so the
    midpoint of the colour bar always means "no correlation".  Pass *diverging*
    to force either choice — do that when several panels must share a scale and
    only some of them go negative.
    """
    import matplotlib.pyplot as plt
    values = []
    for matrix in matrices:
        off = np.array(matrix, dtype=float, copy=True)
        np.fill_diagonal(off, np.nan)
        values.append(off)
    stacked = np.concatenate([v.ravel() for v in values])
    lowest = float(np.nanmin(stacked))
    highest = float(np.nanmax(stacked))
    if diverging is None:
        diverging = lowest < 0
    if diverging:
        limit = max(abs(lowest), abs(highest))
        return plt.cm.RdBu_r, -limit, limit
    return plt.cm.magma, 0.0, highest


def plot_correlation_matrix(ax, matrix: np.ndarray, *,
                            blocks: Sequence[Tuple[str, int, int]] | None = None,
                            tick_labels: Sequence[str] | None = None,
                            tick_colors: Sequence[str] | None = None,
                            cmap=None, vmin: float | None = None,
                            vmax: float | None = None,
                            mask_diagonal: bool = True,
                            min_block_for_label: int = 1,
                            rule_color: str = "white", rule_lw: float = 0.6,
                            fontsize: float | None = None,
                            title: str | None = None,
                            bad_color: str = "0.92"):
    """Draw a region x region correlation *matrix* and return the image.

    *matrix* is drawn in the order it is given — order it first with
    :func:`order_by_group` (or by module, or by any other key) and pass the
    matching :func:`group_blocks` output as *blocks* to rule the boundaries and
    label each run.

    Labelling has two modes.  With *blocks* alone, one label sits at the centre
    of each run and takes that group's colour; runs shorter than
    *min_block_for_label* are ruled but left unlabelled, which is what keeps a
    40-region matrix readable at panel size.  With *tick_labels* (one entry per
    row) every region is labelled instead, optionally coloured by
    *tick_colors*.

    The diagonal is masked by default: a region's correlation with itself is 1
    by construction and only compresses the colour scale.
    """
    from databench.plotting import font_size as _font_size
    shown = np.array(matrix, dtype=float, copy=True)
    if mask_diagonal:
        np.fill_diagonal(shown, np.nan)
    if cmap is None or vmin is None or vmax is None:
        auto_cmap, auto_vmin, auto_vmax = correlation_scale(shown)
        cmap = auto_cmap if cmap is None else cmap
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax
    cmap = cmap.copy()
    cmap.set_bad(bad_color)
    image = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest")

    size = fontsize if fontsize is not None else _font_size("annotation") - 1
    if tick_labels is not None:
        positions = np.arange(len(tick_labels))
        ax.set_xticks(positions)
        ax.set_yticks(positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=size)
        ax.set_yticklabels(tick_labels, fontsize=size)
        if tick_colors is not None:
            for tick, color in zip(ax.get_xticklabels(), tick_colors):
                tick.set_color(color)
            for tick, color in zip(ax.get_yticklabels(), tick_colors):
                tick.set_color(color)
    elif blocks is not None:
        palette = group_colors()
        centres = [(first + last) / 2 for _, first, last in blocks]
        names = [name if last - first + 1 >= min_block_for_label else ""
                 for name, first, last in blocks]
        ax.set_xticks(centres)
        ax.set_yticks(centres)
        ax.set_xticklabels(names, rotation=90, fontsize=size)
        ax.set_yticklabels(names, fontsize=size)
        for tick, name in zip(ax.get_xticklabels() + ax.get_yticklabels(), names * 2):
            if name in palette:
                tick.set_color(palette[name])

    if blocks is not None:
        for _, _, last in list(blocks)[:-1]:
            ax.axhline(last + 0.5, color=rule_color, lw=rule_lw)
            ax.axvline(last + 0.5, color=rule_color, lw=rule_lw)

    ax.tick_params(length=0, pad=1.5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title, fontsize=size + 1.5, pad=3)
    return image


# ════════════════════════════════════════════════════════════════════════════
#  Flatmaps
# ════════════════════════════════════════════════════════════════════════════

def _polygon_area(xy: np.ndarray) -> float:
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def plot_flatmap(ax, face_of: Callable[[str], object] | Mapping[str, object], *,
                 outlines: "pd.DataFrame | None" = None,
                 label: bool = True, label_area_pctl: float = 45.0,
                 centroids: Mapping[str, Tuple[float, float]] | None = None,
                 empty_face: str = "white", edge_color: str = "0.25",
                 empty_edge_color: str = "0.8", linewidth: float = 0.6,
                 edge_of: Callable[[str], object] | None = None,
                 highlight_linewidth: float = 1.4,
                 fontsize: float | None = None, axis_labels: bool = True):
    """Fill every atlas region on a dorsal flatmap.

    *face_of* maps a region name to a face colour; a region it maps to ``None``
    is drawn as *empty_face*, which is how a figure shows "in the atlas, not
    measured here".  Regions above the *label_area_pctl* percentile of outline
    area are labelled with their acronym, so the small parcels do not collide.

    *edge_of* optionally gives a faced region its own outline colour, drawn at
    *highlight_linewidth* — that is how a map calls out a subset (seeds, a
    module, the regions a trace panel plots) without recolouring their fill.

    Returns the list of region names that were given a face.
    """
    import matplotlib.patheffects as pe
    from databench.plotting import font_size as _font_size

    table = load_outlines() if outlines is None else outlines
    polygons = {name: g.sort_values("k")[["ml_mm", "ap_mm"]].to_numpy()
                for name, g in table.groupby("region")}
    lookup = face_of if callable(face_of) else face_of.get
    areas = {name: _polygon_area(poly) for name, poly in polygons.items()}
    cut = float(np.percentile(list(areas.values()), label_area_pctl)) if areas else 0.0
    size = fontsize if fontsize is not None else _font_size("annotation") - 2.5

    faced: List[str] = []
    for name, poly in sorted(polygons.items()):
        face = lookup(name)
        if face is not None:
            faced.append(name)
        highlight = edge_of(name) if (edge_of is not None and face is not None) else None
        ax.fill(poly[:, 0], poly[:, 1],
                facecolor=face if face is not None else empty_face,
                edgecolor=(highlight if highlight is not None else
                           (edge_color if face is not None else empty_edge_color)),
                linewidth=highlight_linewidth if highlight is not None else linewidth,
                zorder=2 if face is not None else 1)
        if label and areas[name] >= cut:
            if centroids and name in centroids:
                cx, cy = centroids[name]
            else:
                cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
            ax.text(cx, cy, region_acronym(name), fontsize=size, ha="center",
                    va="center", zorder=5,
                    color="0.1" if face is not None else "0.6",
                    path_effects=[pe.withStroke(linewidth=1.3, foreground="white")])

    ax.invert_yaxis()
    ax.set_aspect("equal")
    if axis_labels:
        ax.set_xlabel("ML (mm, − left / + right)")
        ax.set_ylabel("AP (mm, anterior → posterior)")
    return faced


# ════════════════════════════════════════════════════════════════════════════
#  Parcellation in the recording's own frame
# ════════════════════════════════════════════════════════════════════════════

FRAME_SHAPE: Tuple[int, int] = (512, 512)
ATLAS_SHAPE_AP = 1320            # CCF dorsal projection at 10 µm
ATLAS_SHAPE_ML = 1140
ATLAS_RES_MM = 0.01


@dataclass
class Parcellation:
    """How one recording's cortical parcellation is drawn in its own frame.

    ``kind`` is ``"mask"`` when the mesomap label image was found on disk —
    already in frame pixels, and the most faithful source — ``"affine"`` when
    only the registration affine was available and the atlas outlines had to be
    warped in, and ``"none"`` when neither was.
    """
    kind: str
    mask: np.ndarray | None = None
    regions: "pd.DataFrame | None" = None
    cortex: np.ndarray | None = None
    frame: np.ndarray | None = None
    outlines: "pd.DataFrame | None" = None
    inside: set | None = None
    shape: Tuple[int, int] = FRAME_SHAPE
    source: str = ""

    @property
    def n_regions(self) -> int:
        if self.kind == "mask":
            return 0 if self.regions is None else len(self.regions)
        return 0 if self.inside is None else len(self.inside)


def read_image(path: "Path | str | None") -> np.ndarray | None:
    """Load *path* as an array, or return None when it is not on this machine."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        return np.asarray(tifffile.imread(path))
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"))


def resample_to(image: np.ndarray | None,
                shape: Tuple[int, int]) -> np.ndarray | None:
    """*image* on the pixel grid *shape*, by nearest neighbour.

    Panel layers are written by different tools at whatever size each chose;
    putting them on one grid before drawing is what makes them co-register.
    """
    if image is None or tuple(image.shape[:2]) == tuple(shape):
        return image
    rows = np.linspace(0, image.shape[0] - 1, shape[0]).round().astype(int)
    cols = np.linspace(0, image.shape[1] - 1, shape[1]).round().astype(int)
    return image[np.ix_(rows, cols)]


def atlas_in_frame(affine, *, outlines: "pd.DataFrame | None" = None,
                   frame_shape: Tuple[int, int] = FRAME_SHAPE) -> "pd.DataFrame":
    """Atlas outlines mapped into frame pixels, with ``xi``/``yi`` columns added.

    Mirrors ``mesomap.mapper.AllenCortexMapper.transform_segmentation``: the CCF
    dorsal projection is resized to the frame shape and only then warped by the
    registration affine, so inverting the outline table takes the same two steps
    in the same order.
    """
    table = (load_outlines() if outlines is None else outlines).copy()
    x = ((table["ml_mm"] / ATLAS_RES_MM + ATLAS_SHAPE_ML / 2.0)
         * frame_shape[1] / ATLAS_SHAPE_ML)
    y = (table["ap_mm"] / ATLAS_RES_MM) * frame_shape[0] / ATLAS_SHAPE_AP
    warped = np.column_stack([x, y, np.ones(len(x))]) @ np.asarray(affine, float).T
    return table.assign(xi=warped[:, 0], yi=warped[:, 1])


def load_parcellation(session, *, mask_dir=None, cortex_dir=None, frame_dir=None,
                      source: str = "mesomap",
                      frame_shape: Tuple[int, int] = FRAME_SHAPE) -> Parcellation:
    """Resolve the best parcellation available for *session*.

    Looks first for the mesomap label image written beside that recording's
    traces in *mask_dir* (``*_traces.mask.npy`` with its ``.regions.csv``),
    together with a cortex mask under ``*cortex_dir*/sub-<subject>_<session>/``
    and a reference frame in *frame_dir*.  Falls back to the label image the
    dataset carries in its own mesomap metadata (``mask_image`` with
    ``regions_table``, and ``frame_image`` unless *frame_dir* has a frame), and
    then to the registration affine stored there.

    Every directory is optional: pass the ones this machine has.
    """
    import pandas as pd

    if mask_dir is not None:
        pattern = f"*sub-{session.subject}_{session.session}_{session.task}_*"
        masks = sorted(Path(mask_dir).glob(pattern + "traces.mask.npy"))
        if masks:
            mask = np.load(masks[0])
            regions = pd.read_csv(str(masks[0]).replace(".mask.npy", ".regions.csv"))
            cortex = None
            if cortex_dir is not None:
                cortex = read_image(Path(cortex_dir) /
                                    f"sub-{session.subject}_{session.session}" /
                                    "cortical_mask.tif")
                cortex = resample_to(cortex, mask.shape)
            frame = None
            if frame_dir is not None:
                frame = read_image(Path(frame_dir) /
                                   f"sub-{session.subject}_{session.session}_"
                                   f"{session.task}_frame.png")
                frame = resample_to(frame, mask.shape)
            return Parcellation(kind="mask", mask=mask, regions=regions,
                                cortex=None if cortex is None else cortex > 0,
                                frame=frame, shape=tuple(mask.shape),
                                source=masks[0].name)

    meta = session._row.get((source, "meta"))
    if isinstance(meta, dict) and meta.get("mask_image") is not None:
        mask = np.asarray(meta["mask_image"])
        frame = None
        if frame_dir is not None:
            frame = resample_to(read_image(Path(frame_dir) /
                                           f"sub-{session.subject}_{session.session}_"
                                           f"{session.task}_frame.png"), mask.shape)
        if frame is None and meta.get("frame_image") is not None:
            frame = resample_to(np.asarray(meta["frame_image"]), mask.shape)
        return Parcellation(kind="mask", mask=mask, regions=meta["regions_table"],
                            frame=frame, shape=tuple(mask.shape),
                            source=f"{source}.meta mask_image")

    if isinstance(meta, dict) and "affine_matrix" in meta:
        frame = None
        if frame_dir is not None:
            frame = read_image(Path(frame_dir) /
                               f"sub-{session.subject}_{session.session}_"
                               f"{session.task}_frame.png")
            frame = resample_to(frame, frame_shape)
        return Parcellation(kind="affine",
                            outlines=atlas_in_frame(meta["affine_matrix"],
                                                    frame_shape=frame_shape),
                            inside=set(meta["regions_table"]["acronym"]),
                            frame=frame, shape=frame_shape,
                            source=f"{source}.meta affine")
    return Parcellation(kind="none", shape=frame_shape)


def plot_parcellation(ax, parcellation: Parcellation, *,
                      colors: Mapping[str, str] | None = None,
                      face_of: Callable[[str], object] | None = None,
                      over_frame_alpha: float = 0.55,
                      outside_alpha: float = 0.18,
                      frame_cmap: str = "gray",
                      boundary_color: Tuple[float, float, float, float] =
                      (1.0, 1.0, 1.0, 0.75),
                      cortex_line: str = "white",
                      frame_edge: str = "0.35") -> str:
    """Draw *parcellation* into *ax*; return a one-line description of it.

    Regions are filled by functional group, or by *face_of* when given — a
    callable from region name to a face colour, which is how a map shows a
    per-region value on a colour scale; a region it maps to ``None`` is left
    unfilled, with its boundary still drawn.  Anything the cortex mask excludes
    is drawn at *outside_alpha*, so the cranial window's edge reads against the
    parcellation instead of being cropped away.  With a reference frame present
    the parcellation is drawn over it at *over_frame_alpha*, otherwise opaque.

    The exclusion is pixel-wise where a cortex mask image was found, and
    per-region where the source was the registration affine — there, the regions
    absent from the recording's ``regions_table`` are the ones the mask dropped.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb

    palette = group_colors() if colors is None else colors
    if face_of is None:
        face_of = lambda name: palette[region_group(name)]  # noqa: E731
    frame = parcellation.frame
    if frame is not None:
        ax.imshow(frame, cmap=frame_cmap, zorder=0)
    fill_alpha = over_frame_alpha if frame is not None else 1.0

    if parcellation.kind == "mask":
        mask, regions = parcellation.mask, parcellation.regions
        cortex = parcellation.cortex
        rgba = np.zeros(mask.shape + (4,))
        for row in regions.itertuples():
            selected = mask == row.mask_label
            face = face_of(row.acronym)
            if face is None or not selected.any():
                continue
            rgba[selected, :3] = to_rgb(face)
            rgba[selected, 3] = (fill_alpha if cortex is None else
                                 np.where(cortex[selected], fill_alpha,
                                          outside_alpha))
        ax.imshow(rgba, zorder=2, interpolation="nearest")

        edges = np.zeros(mask.shape, dtype=bool)
        edges[:-1, :] |= mask[:-1, :] != mask[1:, :]
        edges[:, :-1] |= mask[:, :-1] != mask[:, 1:]
        boundary = np.zeros(mask.shape + (4,))
        boundary[edges] = boundary_color
        ax.imshow(boundary, zorder=3, interpolation="nearest")

        if cortex is not None:
            ax.contour(cortex.astype(float), [0.5], colors=cortex_line,
                       linewidths=0.8, zorder=4)
        ax.set_xlim(-0.5, mask.shape[1] - 0.5)
        ax.set_ylim(mask.shape[0] - 0.5, -0.5)
        ax.set_aspect("equal")
        return (f"{len(regions)} parcels" +
                ("" if cortex is None else ", cortex mask outlined"))

    if parcellation.kind == "affine":
        outlines, inside = parcellation.outlines, parcellation.inside
        for name, poly in outlines.groupby("region"):
            poly = poly.sort_values("k")
            kept = name in inside
            face = face_of(name)
            ax.fill(poly["xi"], poly["yi"], facecolor="none" if face is None else face,
                    alpha=fill_alpha if kept else outside_alpha,
                    edgecolor="white" if kept else "none", lw=0.35,
                    zorder=3 if kept else 2)
        height, width = parcellation.shape
        ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False,
                                   edgecolor=frame_edge, lw=0.7,
                                   ls=(0, (3, 2)), zorder=5))
        ax.set_xlim(-75, width + 75)
        ax.set_ylim(height + 75, -75)
        ax.set_aspect("equal")
        return f"{len(inside)} regions in mask"

    return "no parcellation on this machine"
