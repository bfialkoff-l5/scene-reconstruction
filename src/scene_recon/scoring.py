from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

ORB_NFEATURES = 500
QUALITY_WEIGHT_FEATURES = 0.6
QUALITY_WEIGHT_SHARPNESS = 0.4


def score_image(bgr: np.ndarray) -> tuple[int, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=ORB_NFEATURES)
    keypoints = orb.detect(gray, None)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return len(keypoints), sharpness


def compute_quality_scores(
    feature_count: pd.Series,
    sharpness: pd.Series,
    *,
    weight_features: float,
    weight_sharpness: float,
) -> pd.Series:
    feat = feature_count.astype(float)
    sharp = sharpness.astype(float)
    feat_norm = (feat - feat.min()) / (feat.max() - feat.min() + 1e-9)
    sharp_norm = (sharp - sharp.min()) / (sharp.max() - sharp.min() + 1e-9)
    return weight_features * feat_norm + weight_sharpness * sharp_norm
