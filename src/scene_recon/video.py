from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from scene_recon.record import Record
from scene_recon.scoring import (
    QUALITY_WEIGHT_FEATURES,
    QUALITY_WEIGHT_SHARPNESS,
    compute_quality_scores,
    score_image,
)

SCORE_WORKERS = 4
MAX_PENDING_SCORES = SCORE_WORKERS * 4


def _flush_scores(
    pending: dict[int, Future[tuple[int, float]]],
    candidates: pd.DataFrame,
) -> None:
    for frame_number, future in pending.items():
        feature_count, sharpness = future.result()
        candidates.loc[frame_number, "feature_count"] = feature_count
        candidates.loc[frame_number, "sharpness"] = sharpness
    pending.clear()


def score_all_frames(record: Record, candidates: pd.DataFrame) -> pd.DataFrame:
    """Decode every pose frame sequentially; score in a thread pool; fill DataFrame."""
    targets = set(int(n) for n in candidates.index)
    pending: dict[int, Future[tuple[int, float]]] = {}

    cap = cv2.VideoCapture(str(record.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {record.video}")

    try:
        idx = 0
        with ThreadPoolExecutor(max_workers=SCORE_WORKERS) as pool:
            with tqdm(total=len(targets), desc="Scoring frames", unit="frame") as pbar:
                while True:
                    if not cap.grab():
                        break
                    if idx in targets:
                        ok, bgr = cap.retrieve()
                        if not ok:
                            raise RuntimeError(f"failed to retrieve frame {idx}")
                        pending[idx] = pool.submit(score_image, bgr.copy())
                        pbar.update(1)
                        if len(pending) >= MAX_PENDING_SCORES:
                            _flush_scores(pending, candidates)
                    idx += 1

                if pending:
                    _flush_scores(pending, candidates)
    finally:
        cap.release()

    scored = set(
        int(n)
        for n in candidates.index[candidates["feature_count"].notna()].tolist()
    )
    missing = targets - scored
    if missing:
        raise RuntimeError(
            f"video ended before {len(missing)} pose frames; first missing: {min(missing)}"
        )

    candidates["quality_score"] = compute_quality_scores(
        candidates["feature_count"],
        candidates["sharpness"],
        weight_features=QUALITY_WEIGHT_FEATURES,
        weight_sharpness=QUALITY_WEIGHT_SHARPNESS,
    )
    return candidates


def extract_frames(
    record: Record,
    frame_numbers: list[int],
    images_dir: Path,
) -> None:
    """Second forward pass: write PNGs for selected frames only."""
    images_dir.mkdir(parents=True, exist_ok=True)
    want = set(frame_numbers)
    if not want:
        raise ValueError("no frames to extract")

    cap = cv2.VideoCapture(str(record.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {record.video}")

    try:
        idx = 0
        with tqdm(total=len(want), desc="Exporting keyframes", unit="frame") as pbar:
            while want:
                if not cap.grab():
                    break
                if idx in want:
                    ok, bgr = cap.retrieve()
                    if not ok:
                        raise RuntimeError(f"failed to retrieve frame {idx}")
                    out_path = images_dir / frame_filename(idx)
                    if not cv2.imwrite(str(out_path), bgr):
                        raise RuntimeError(f"failed to write {out_path}")
                    want.remove(idx)
                    pbar.update(1)
                idx += 1
    finally:
        cap.release()

    if want:
        raise RuntimeError(f"video ended before frames: {sorted(want)[:5]}...")


def frame_filename(frame_number: int) -> str:
    return f"{frame_number:06d}.png"
