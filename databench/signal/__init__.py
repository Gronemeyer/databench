"""Signal-processing primitives shared by analysis and plotting."""

from databench.signal.bandpass import (  # noqa: F401
    bandpass_envelope,
    robust_threshold,
)

from databench.signal.preproc import (  # noqa: F401
    MEDIAN_FILTER_SIZE,
    SAVGOL_WINDOW,
    SAVGOL_POLYORDER,
    OUTLIER_IQR_K,
    smooth_savgol,
    smooth_dense,
    smooth_median,
    remove_outliers_iqr,
    detrend_zscore_1d,
)

from databench.signal.remap import (  # noqa: F401 — public re-exports
    GAP_THRESHOLD_S,
    break_at_gaps,
    remap_previous_sample,
    remap_to_timebase,
    prepare_sparse_trace,
)

from databench.signal.epoching import (  # noqa: F401 — public re-exports
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
