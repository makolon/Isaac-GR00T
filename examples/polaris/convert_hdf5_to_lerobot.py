"""Convert polaris_real2sim TAMP HDF5 episodes to a GR00T N1.7 LeRobot v2.1 dataset.

Self-contained: writes the LeRobot v2.1 layout directly with pyarrow/imageio
(GR00T ships no dataset-creation API), adds GR00T's ``meta/modality.json`` and a
``NEW_EMBODIMENT`` modality config, and optionally runs GR00T's own ``stats.py``.
Depends only on this repo's deps + the bundled ``polaris_hdf5`` reader; no lerobot
package and no IsaacLab.

    cd third_party/gr00t
    uv sync --frozen          # gr00t + imageio/pyarrow/numpy
    uv pip install h5py       # gr00t's lock omits h5py and can't be re-locked on
                              # x86_64 (its required-environments pins an aarch64
                              # cp310 torchcodec wheel), so add h5py directly.
    uv run examples/polaris/convert_hdf5_to_lerobot.py \\
        --hdf5-root /path/to/tamp_hdf5_dataset \\
        --output-root /path/to/out_gr00t_dataset \\
        --action-space joint --compute-stats

Then finetune in this same env via ``bash examples/finetune.sh ... --embodiment-tag new_embodiment``.
``--action-space joint`` -> arm joints + gripper (NON_EEF); ``ee`` -> EE pose
(xyz + 6D rot) + gripper as a RELATIVE EEF action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import imageio.v2 as imageio
import numpy as np
from polaris_hdf5 import CameraMapping, Hdf5DatasetReader, Hdf5EpisodeReader, build_state_action
import pyarrow as pa
import pyarrow.parquet as pq


# Repo root is examples/polaris/<file> -> ../../ ; stats.py lives at gr00t/data/stats.py.
_GR00T_ROOT = Path(__file__).resolve().parents[2]

_CHUNK_SIZE = 1000
_VIDEO_CODEC = "h264"
_VIDEO_PIX_FMT = "yuv420p"

_GR00T_VIDEO_NAMES: dict[str, str] = {
    "exterior_1": "front",
    "exterior_2": "exterior_2",
    "wrist": "wrist",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--action-space", choices=("joint", "ee"), default="joint")
    parser.add_argument("--camera-map", default=None, help="JSON {role: hdf5_image_key} override")
    parser.add_argument("--robot-type", default="franka_robotiq")
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--raw-gripper", action="store_true")
    parser.add_argument("--embodiment-tag", default="new_embodiment")
    parser.add_argument("--modality-config-path", type=Path, default=None)
    parser.add_argument(
        "--compute-stats", action="store_true", help="run gr00t/data/stats.py afterwards"
    )
    return parser.parse_args()


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


def _modality_slices(action_space: str, dim: int) -> dict[str, dict[str, int]]:
    if action_space == "joint":
        return {
            "single_arm": {"start": 0, "end": dim - 1},
            "gripper": {"start": dim - 1, "end": dim},
        }
    return {"eef": {"start": 0, "end": 9}, "gripper": {"start": 9, "end": 10}}


def _state_names(action_space: str, sample: Hdf5EpisodeReader) -> list[str]:
    if action_space == "joint":
        return [*sample.arm_joint_names, "gripper"]
    return ["x", "y", "z", *[f"rot6d_{i}" for i in range(6)], "gripper"]


def _modality_config_text(action_space: str, embodiment_tag: str) -> str:
    video_keys = list(_GR00T_VIDEO_NAMES.values())
    if action_space == "joint":
        modality_keys = ["single_arm", "gripper"]
        action_configs = (
            "        ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),\n"
            "        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),\n"
        )
    else:
        modality_keys = ["eef", "gripper"]
        action_configs = (
            "        ActionConfig(rep=ActionRepresentation.RELATIVE, type=ActionType.EEF, format=ActionFormat.XYZ_ROT6D),\n"
            "        ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT),\n"
        )
    return (
        "from gr00t.configs.data.embodiment_configs import register_modality_config\n"
        "from gr00t.data.embodiment_tags import EmbodimentTag\n"
        "from gr00t.data.types import (\n"
        "    ActionConfig,\n    ActionFormat,\n    ActionRepresentation,\n    ActionType,\n    ModalityConfig,\n)\n\n\n"
        "polaris_config = {\n"
        f'    "video": ModalityConfig(delta_indices=[0], modality_keys={video_keys!r}),\n'
        f'    "state": ModalityConfig(delta_indices=[0], modality_keys={modality_keys!r}),\n'
        '    "action": ModalityConfig(\n'
        "        delta_indices=list(range(0, 16)),\n"
        f"        modality_keys={modality_keys!r},\n"
        "        action_configs=[\n"
        f"{action_configs}"
        "        ],\n"
        "    ),\n"
        '    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),\n'
        "}\n\n"
        f"register_modality_config(polaris_config, embodiment_tag=EmbodimentTag.{embodiment_tag.upper()})\n"
    )


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
    state_names = _state_names(args.action_space, sample)

    tasks: dict[str, int] = {}
    task_records: list[dict] = []
    episode_records: list[dict] = []
    camera_shapes: dict[str, tuple[int, int, int]] = {}
    state_dim = action_dim = 0
    global_index = 0

    for episode_index, path in enumerate(episode_paths):
        reader = Hdf5EpisodeReader(path)
        state, action = build_state_action(
            reader, args.action_space, normalize_gripper_value=normalize
        )
        state_dim, action_dim = int(state.shape[1]), int(action.shape[1])
        length = reader.num_frames

        instruction = reader.instruction
        if instruction not in tasks:
            tasks[instruction] = len(tasks)
            task_records.append({"task_index": tasks[instruction], "task": instruction})
        task_index = tasks[instruction]

        chunk = episode_index // _CHUNK_SIZE
        for role, video_name in _GR00T_VIDEO_NAMES.items():
            frames = reader.read_video(mapping.key_for(role))
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
        "state": _modality_slices(args.action_space, state_dim),
        "action": _modality_slices(args.action_space, action_dim),
        "video": {name: {"original_key": f"observation.images.{name}"} for name in camera_shapes},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    (meta_dir / "modality.json").write_text(json.dumps(modality, indent=2), encoding="utf-8")

    config_path = args.modality_config_path or (out / "polaris_modality_config.py")
    config_path.write_text(
        _modality_config_text(args.action_space, args.embodiment_tag), encoding="utf-8"
    )

    print(f"[gr00t-convert] done: {len(episode_records)} episodes -> {out}", flush=True)
    print(f"[gr00t-convert] modality config: {config_path}", flush=True)

    if args.compute_stats:
        stats_script = _GR00T_ROOT / "gr00t" / "data" / "stats.py"
        subprocess.run(
            [
                sys.executable,
                str(stats_script),
                "--dataset-path",
                str(out),
                "--embodiment-tag",
                args.embodiment_tag,
                "--modality-config-path",
                str(config_path),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
