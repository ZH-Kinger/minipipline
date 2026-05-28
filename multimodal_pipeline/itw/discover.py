from __future__ import annotations

import json
from pathlib import Path

from ..config import ITWConfig
from .schemas import (
    ActorInfo,
    DiscoveryResult,
    FileMismatch,
    ITWManifest,
    ITWManifestItem,
    TaskInfo,
)


# Canonical names within a session for each modality we expect.
# Keys are stable across sessions; values come from `cfg` so users can override.
_KNOWN_MANIFEST_NAMES = {
    "rgb_head",
    "depth_head",
    "imu",
    "mic",
    "head_hands_sixdof",
}


def is_session_dir(path: Path, cfg: ITWConfig | None = None) -> bool:
    """Heuristically decide whether a directory is an ITW session.

    A directory counts as a session if it contains both the manifest file
    (``config.json``) and the reference RGB video (``rgb_head.mp4``). This
    pair is the minimum a session must have for Layer 1 to do anything
    useful, and it cleanly distinguishes from "parent of many sessions" or
    unrelated directories.
    """

    if not path.is_dir():
        return False
    cfg = cfg or ITWConfig()
    return (path / cfg.manifest_filename).is_file() and (
        path / cfg.rgb_video_filename
    ).is_file()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _parse_manifest(raw: dict) -> ITWManifest:
    items_raw = raw.get("items", []) or []
    items: list[ITWManifestItem] = []
    for entry in items_raw:
        items.append(
            ITWManifestItem(
                name=str(entry.get("name", "")),
                file=str(entry.get("file", "")),
                description=str(entry.get("description", "") or ""),
                fps=float(entry["fps"]) if "fps" in entry else None,
                count=int(entry["count"]) if "count" in entry else None,
                width=int(entry["width"]) if "width" in entry else None,
                height=int(entry["height"]) if "height" in entry else None,
                sample_rate=int(entry["sample_rate"]) if "sample_rate" in entry else None,
            )
        )
    return ITWManifest(
        version=str(raw.get("version", "")),
        type=str(raw.get("type", "")),
        duration_s=float(raw.get("duration", 0.0)),
        items=tuple(items),
        raw=raw,
    )


def _parse_task_info(raw: dict) -> TaskInfo:
    actor_raw = raw.get("actor") or {}
    actor = ActorInfo(
        gender=actor_raw.get("gender"),
        height_cm=float(actor_raw["height"]) if "height" in actor_raw else None,
        weight_kg=float(actor_raw["weight"]) if "weight" in actor_raw else None,
        age=int(actor_raw["age"]) if "age" in actor_raw else None,
    )
    return TaskInfo(
        task_id=str(raw.get("task_id", "")),
        name=str(raw.get("name", "")),
        scene=str(raw.get("scene", "")),
        task_scene=str(raw.get("task_scene", "")),
        success=int(raw.get("success", 0)),
        duration_s=float(raw.get("duration", 0.0)),
        modality=tuple(str(m) for m in raw.get("modality", []) or []),
        steps=tuple(str(s) for s in raw.get("steps", []) or []),
        tags=tuple(str(t) for t in raw.get("tags", []) or []),
        items=tuple(str(i) for i in raw.get("items", []) or []),
        hints=tuple(str(h) for h in raw.get("hints", []) or []),
        actor=actor,
        version=str(raw.get("version", "")),
        request_id=str(raw.get("request_id", "")),
    )


def _resolve_head_6dof(session_dir: Path, cfg: ITWConfig) -> tuple[Path | None, str | None, tuple[str, ...]]:
    """Locate the head-6DoF CSV among a list of candidate names.

    Returns (resolved_path, picked_name, all_existing_candidate_names).
    Matches the on-disk reality where config.json declares
    `head_hands_sixdof.csv` but the file is actually `head_hands_sixdof2.csv`.
    """
    found: list[str] = []
    for candidate in cfg.head_6dof_candidates:
        if (session_dir / candidate).exists():
            found.append(candidate)
    if not found:
        return None, None, ()
    # Prefer the alphabetically last variant (sixdof2 > sixdof). This matches
    # the empirical observation that producers rev the file name when re-cutting.
    picked = sorted(found)[-1]
    return session_dir / picked, picked, tuple(found)


def _scan_extras(session_dir: Path, expected: set[Path]) -> tuple[str, ...]:
    extras: list[str] = []
    expected_resolved = {p.resolve() for p in expected if p is not None}
    for path in sorted(session_dir.rglob("*")):
        if path.is_dir():
            continue
        if path.resolve() in expected_resolved:
            continue
        rel = path.relative_to(session_dir).as_posix()
        extras.append(rel)
    return tuple(extras)


def discover(session_dir: Path, cfg: ITWConfig) -> DiscoveryResult:
    session_dir = Path(session_dir).resolve()
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    manifest_path = session_dir / cfg.manifest_filename
    task_path = session_dir / cfg.task_info_filename
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not task_path.exists():
        raise FileNotFoundError(f"Task info not found: {task_path}")

    manifest = _parse_manifest(_read_json(manifest_path))
    task_info = _parse_task_info(_read_json(task_path))

    found: dict[str, Path] = {}
    mismatches: list[FileMismatch] = []

    # Resolve each declared manifest item to an actual file on disk.
    for item in manifest.items:
        declared = session_dir / item.file
        if declared.exists():
            found[item.name] = declared
            continue

        # Special case: head_hands_sixdof (config declares foo.csv but disk has foo2.csv).
        if item.name == "head_hands_sixdof":
            resolved, picked, all_found = _resolve_head_6dof(session_dir, cfg)
            if resolved is not None and picked is not None:
                found[item.name] = resolved
                if picked != item.file:
                    mismatches.append(
                        FileMismatch(
                            manifest_name=item.name,
                            declared_file=item.file,
                            actual_file=picked,
                            candidates=all_found,
                            note=(
                                f"manifest declares {item.file!r} but file is {picked!r}; "
                                "auto-resolved via head_6dof_candidates"
                            ),
                        )
                    )
                continue

        mismatches.append(
            FileMismatch(
                manifest_name=item.name,
                declared_file=item.file,
                actual_file=None,
                candidates=(),
                note=f"declared file not found on disk under {session_dir.name}/",
            )
        )

    # Always include the standalone files that may not be in the manifest.
    hand_kpts = session_dir / cfg.hand_keypoints_filename
    if hand_kpts.exists():
        found["hands_keypoint_3d"] = hand_kpts
    else:
        mismatches.append(
            FileMismatch(
                manifest_name="hands_keypoint_3d",
                declared_file=cfg.hand_keypoints_filename,
                actual_file=None,
                candidates=(),
                note="hand keypoint json missing; this is required for hand annotation",
            )
        )

    kalibr = session_dir / cfg.kalibr_filename
    if kalibr.exists():
        found["kalibr_parameters"] = kalibr
    else:
        mismatches.append(
            FileMismatch(
                manifest_name="kalibr_parameters",
                declared_file=cfg.kalibr_filename,
                actual_file=None,
                candidates=(),
                note="kalibr yaml missing; calibration will fall back to camera_params/head_param.json",
            )
        )

    head_param = session_dir / cfg.head_param_path
    has_cam_subdir = (session_dir / "camera_params").is_dir()
    if head_param.exists():
        found["head_param"] = head_param

    # Register the timestamp sidecars under distinct keys so they aren't
    # flagged as "extras". The manifest declares the videos as `rgb_head` /
    # `depth_head`; the per-frame timestamp csvs sit alongside them.
    timestamp_sidecars = (
        ("rgb_head_timestamps", cfg.rgb_timestamps_filename),
        ("depth_head_timestamps", cfg.depth_timestamps_filename),
    )
    for key, sidecar_name in timestamp_sidecars:
        sidecar = session_dir / sidecar_name
        if sidecar.exists():
            found[key] = sidecar
        else:
            mismatches.append(
                FileMismatch(
                    manifest_name=key,
                    declared_file=sidecar_name,
                    actual_file=None,
                    candidates=(),
                    note=f"timestamp sidecar {sidecar_name} missing; cannot align this stream",
                )
            )

    extras = _scan_extras(session_dir, set(found.values()) | {manifest_path, task_path, head_param})

    return DiscoveryResult(
        session_dir=session_dir,
        session_id=session_dir.name,
        manifest=manifest,
        task_info=task_info,
        found_files=found,
        extra_files=extras,
        mismatches=tuple(mismatches),
        has_camera_params_subdir=has_cam_subdir,
    )


__all__ = ["discover"]
