from __future__ import annotations

POSE_COLUMNS_REQUIRED = (
    "FrameNumber",
    "TimeUS",
    "easting",
    "northing",
    "altamsl",
    "utm_zone",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
)

SCORE_COLUMNS = (
    "feature_count",
    "sharpness",
    "quality_score",
)

SELECTION_COLUMNS = (
    "cell_x",
    "cell_y",
    "selected",
    "reject_reason",
)
