"""Internal signal-processing primitives."""

from databench.analysis._signal.remap import (  # noqa: F401 — public re-exports
    GAP_THRESHOLD_S,
    remap_previous_sample,
    remap_to_timebase,
    prepare_sparse_trace,
)

from databench.analysis._signal.epoching import (  # noqa: F401 — public re-exports
    # Column-name constants
    EPOCH_ID,
    START_S,
    END_S,
    DURATION_S,
    EVENT_TYPE,
    START_IDX,
    END_IDX,
    EPOCH_COLUMNS,
    REQUIRED_COLUMNS,
    # Segment primitives
    segments_from_mask,
    merge_gaps,
    apply_min_duration,
    merge_gaps_by_time,
    apply_min_duration_by_time,
    # EpochTable builders
    detect_epochs,
    epochs_around_events,
    # Point-event table
    make_events,
    # Peri-event extraction
    epoch_indices,
    extract_epoch_interpolated,
)
