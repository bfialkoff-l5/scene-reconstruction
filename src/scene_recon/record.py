from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Record:
    path: Path
    slug: str
    video: Path
    poses_path: Path
    intrinsics: Path
    stream_id: str

    @classmethod
    def from_path(cls, abspath: str | Path) -> Record:
        path = Path(abspath).resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Record path not found: {path}")

        derived = path / "_derived"
        if not derived.is_dir():
            raise FileNotFoundError(f"Missing _derived/ in {path}")

        intrinsics = path / "intrinsicK.csv"
        if not intrinsics.is_file():
            raise FileNotFoundError(f"Missing intrinsicK.csv in {path}")

        candidates: list[tuple[Path, Path]] = []
        for video in sorted(path.glob("*.mp4")):
            poses_path = derived / f"gt_{video.stem}.csv"
            if poses_path.is_file():
                candidates.append((video, poses_path))

        if not candidates:
            raise FileNotFoundError(
                f"No mp4 with matching _derived/gt_<stem>.csv in {path}"
            )
        if len(candidates) > 1:
            names = ", ".join(v.name for v, _ in candidates)
            raise ValueError(
                f"Multiple mp4/gt pairs in {path}: {names}. Resolve ambiguity manually."
            )

        video, poses_path = candidates[0]
        return cls(
            path=path,
            slug=path.name,
            video=video,
            poses_path=poses_path,
            intrinsics=intrinsics,
            stream_id=video.stem,
        )

    @property
    def data_root(self) -> Path:
        if self.path.parent.name != "raw":
            raise ValueError(
                f"Expected record under .../raw/{{slug}}, got parent {self.path.parent.name!r}"
            )
        return self.path.parent.parent
