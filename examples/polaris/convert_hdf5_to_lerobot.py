"""Convert polaris_real2sim TAMP HDF5 episodes to a GR00T N1.7 LeRobot v2.1 dataset.

Targets GR00T's pretrained DROID embodiment (``oxe_droid_relative_eef_relative_joint``)
so the fine-tuned checkpoint is plug-compatible with the unchanged zero-shot eval
client: two letterboxed 180x320 views (``exterior_image_1_left`` <- sim
``external_cam``, ``wrist_image_left`` <- sim ``wrist_cam``), 17-D state/action
``[eef_9d, gripper_position, joint_position]`` with absolute action targets
(GR00T's processor handles the relative encoding), and the
``annotation.language.language_instruction`` language key. No custom modality
config is generated — the embodiment is already registered in GR00T.

Self-contained: writes the LeRobot v2.1 layout directly with pyarrow/imageio
(GR00T ships no dataset-creation API) and optionally runs GR00T's own
``stats.py``. Depends only on this repo's deps + the bundled ``polaris_hdf5``
reader; no lerobot package and no IsaacLab.

    cd third_party/gr00t
    uv sync --frozen          # gr00t + imageio/pyarrow/numpy
    uv pip install h5py       # gr00t's lock omits h5py and can't be re-locked on
                              # x86_64 (its required-environments pins an aarch64
                              # cp310 torchcodec wheel), so add h5py directly.
    uv run examples/polaris/convert_hdf5_to_lerobot.py \\
        --hdf5-root /path/to/tamp_hdf5_dataset \\
        --output-root /path/to/out_gr00t_dataset --compute-stats

Then finetune in this same env via
``bash examples/finetune.sh ... --embodiment-tag oxe_droid_relative_eef_relative_joint``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import imageio.v2 as imageio
import numpy as np
from polaris_hdf5 import CameraMapping, Hdf5DatasetReader, Hdf5EpisodeReader, droid_state_action
import pyarrow as pa
import pyarrow.parquet as pq


# Repo root is examples/polaris/<file> -> ../../ ; stats.py lives at gr00t/data/stats.py.
_GR00T_ROOT = Path(__file__).resolve().parents[2]

_CHUNK_SIZE = 1000
_VIDEO_CODEC = "h264"
_VIDEO_PIX_FMT = "yuv420p"

_DROID_EMBODIMENT_TAG = "oxe_droid_relative_eef_relative_joint"
_DROID_IMAGE_H = 180
_DROID_IMAGE_W = 320

_GR00T_VIDEO_NAMES: dict[str, str] = {
    "exterior_1": "exterior_image_1_left",
    "wrist": "wrist_image_left",
}

_MODALITY_SLICES: dict[str, dict[str, int]] = {
    "eef_9d": {"start": 0, "end": 9},
    "gripper_position": {"start": 9, "end": 10},
    "joint_position": {"start": 10, "end": 17},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--camera-map", default=None, help="JSON {role: hdf5_image_key} override")
    parser.add_argument("--robot-type", default="franka_robotiq")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--raw-gripper", action="store_true")
    parser.add_argument(
        "--compute-stats", action="store_true", help="run gr00t/data/stats.py afterwards"
    )
    return parser.parse_args()


def _letterbox(frames: np.ndarray) -> np.ndarray:
    """Pad-resize ``(T, H, W, 3)`` frames to DROID's 180x320, like the eval client."""
    num_frames, height, width = frames.shape[0], frames.shape[1], frames.shape[2]
    scale = min(_DROID_IMAGE_H / height, _DROID_IMAGE_W / width)
    new_h, new_w = int(round(height * scale)), int(round(width * scale))
    top = (_DROID_IMAGE_H - new_h) // 2
    left = (_DROID_IMAGE_W - new_w) // 2
    out = np.zeros((num_frames, _DROID_IMAGE_H, _DROID_IMAGE_W, 3), dtype=np.uint8)
    for i, frame in enumerate(frames):
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        out[i, top : top + new_h, left : left + new_w] = resized
    return out


def _write_video(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=int(fps),
        codec=_VIDEO_CODEC,
        quality=8,
        pixelformat=_VIDEO_PIX_FMT,
        macro_block_size=1,
    )
    for frame in frames:
        writer.append_data(np.asarray(frame, dtype=np.uint8))
    writer.close()


def _state_names(sample: Hdf5EpisodeReader) -> list[str]:
    return [
        "eef_x",
        "eef_y",
        "eef_z",
        *[f"eef_rot6d_{i}" for i in range(6)],
        "gripper",
        *sample.arm_joint_names,
    ]


def _build_features(
    state_dim: int,
    action_dim: int,
    state_names: list[str],
    camera_shapes: dict[str, tuple[int, int, int]],
    fps: int,
) -> dict[str, dict]:
    features: dict[str, dict] = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": state_names},
        "action": {"dtype": "float32", "shape": [action_dim], "names": state_names},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "next.reward": {"dtype": "float32", "shape": [1], "names": None},
        "next.done": {"dtype": "bool", "shape": [1], "names": None},
    }
    for name, (height, width, channels) in camera_shapes.items():
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": [height, width, channels],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": _VIDEO_CODEC,
                "video.pix_fmt": _VIDEO_PIX_FMT,
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": channels,
                "has_audio": False,
            },
        }
    return features


def main() -> None:
    args = _parse_args()
    normalize = not args.raw_gripper
    mapping = CameraMapping.from_json(args.camera_map)
    out = args.output_root
    if out.exists():
        raise FileExistsError(f"output already exists: {out}")
    meta_dir = out / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    episode_paths = Hdf5DatasetReader(args.hdf5_root).episode_paths()
    if not episode_paths:
        raise FileNotFoundError(f"no episodes under {args.hdf5_root}")

    sample = Hdf5EpisodeReader(episode_paths[0])
    fps = args.fps if args.fps is not None else sample.fps
    state_names = _state_names(sample)

    tasks: dict[str, int] = {}
    task_records: list[dict] = []
    episode_records: list[dict] = []
    camera_shapes: dict[str, tuple[int, int, int]] = {}
    state_dim = action_dim = 0
    global_index = 0

    for episode_index, path in enumerate(episode_paths):
        reader = Hdf5EpisodeReader(path)
        state, action = droid_state_action(reader, normalize_gripper_value=normalize)
        state_dim, action_dim = int(state.shape[1]), int(action.shape[1])
        length = reader.num_frames

        instruction = reader.instruction
        if instruction not in tasks:
            tasks[instruction] = len(tasks)
            task_records.append({"task_index": tasks[instruction], "task": instruction})
        task_index = tasks[instruction]

        chunk = episode_index // _CHUNK_SIZE
        for role, video_name in _GR00T_VIDEO_NAMES.items():
            frames = _letterbox(reader.read_video(mapping.key_for(role)))
            camera_shapes[video_name] = tuple(int(v) for v in frames.shape[1:])
            video_path = (
                out
                / f"videos/chunk-{chunk:03d}/observation.images.{video_name}/episode_{episode_index:06d}.mp4"
            )
            _write_video(video_path, frames, fps)

        timestamps = np.arange(length, dtype=np.float32) / float(fps)
        done = np.zeros(length, dtype=bool)
        done[-1] = True
        table = pa.table(
            {
                "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32())),
                "action": pa.array(action.tolist(), type=pa.list_(pa.float32())),
                "timestamp": pa.array(timestamps.tolist(), type=pa.float32()),
                "frame_index": pa.array(range(length), type=pa.int64()),
                "episode_index": pa.array([episode_index] * length, type=pa.int64()),
                "index": pa.array(range(global_index, global_index + length), type=pa.int64()),
                "task_index": pa.array([task_index] * length, type=pa.int64()),
                "next.reward": pa.array(
                    np.zeros(length, dtype=np.float32).tolist(), type=pa.float32()
                ),
                "next.done": pa.array(done.tolist(), type=pa.bool_()),
            }
        )
        parquet_path = out / f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)

        episode_records.append(
            {"episode_index": episode_index, "tasks": [instruction], "length": length}
        )
        global_index += length
        print(f"[gr00t-convert] wrote episode {episode_index:06d} ({length} frames)", flush=True)

    total_chunks = (len(episode_records) - 1) // _CHUNK_SIZE + 1
    info = {
        "codebase_version": "v2.1",
        "robot_type": args.robot_type,
        "total_episodes": len(episode_records),
        "total_frames": global_index,
        "total_tasks": len(task_records),
        "total_videos": len(episode_records) * len(camera_shapes),
        "total_chunks": total_chunks,
        "chunks_size": _CHUNK_SIZE,
        "fps": int(fps),
        "splits": {"train": f"0:{len(episode_records)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": _build_features(state_dim, action_dim, state_names, camera_shapes, int(fps)),
    }
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    with (meta_dir / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for record in episode_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with (meta_dir / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for record in task_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    modality = {
        "state": _MODALITY_SLICES,
        "action": _MODALITY_SLICES,
        "video": {name: {"original_key": f"observation.images.{name}"} for name in camera_shapes},
        "annotation": {"language.language_instruction": {"original_key": "task_index"}},
    }
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=2), encoding="utf-8")

    print(f"[gr00t-convert] done: {len(episode_records)} episodes -> {out}", flush=True)

    if args.compute_stats:
        stats_script = _GR00T_ROOT / "gr00t" / "data" / "stats.py"
        subprocess.run(
            [
                sys.executable,
                str(stats_script),
                "--dataset-path",
                str(out),
                "--embodiment-tag",
                _DROID_EMBODIMENT_TAG,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
