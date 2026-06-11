"""Self-contained reader for polaris_real2sim TAMP HDF5 episodes + state/action builders.

Bundled into this model repo so dataset conversion stays fully independent of the
polaris_real2sim root package (and its IsaacLab / gsplat dependencies). Only
``h5py`` / ``numpy`` / ``imageio`` are required. Mirrors the per-episode layout
written by ``polaris_tamp.utils.hdf5_utils.Hdf5EpisodeRecorder``.

State/action follow GR00T's pretrained DROID embodiment
(``oxe_droid_relative_eef_relative_joint``): ``[eef_9d (9), gripper_position
(1), joint_position (7)]``. Actions are stored absolute; GR00T's processor
relativizes the EEF/joint keys against the current state at train time and
un-relativizes at inference, so a checkpoint fine-tuned on this layout
evaluates through the unchanged zero-shot client.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np


# exterior_1 must be the camera the zero-shot eval client sends as
# ``exterior_image_1_left`` (sim ``external_cam``), so SFT checkpoints stay
# plug-compatible with the unchanged ``Gr00tN17FrankaDroidClient``.
DEFAULT_SPLAT_ROLES: dict[str, str] = {
    "exterior_1": "splat.external_cam",
    "exterior_2": "splat.external_cam_2",
    "wrist": "splat.wrist_cam",
}


@dataclass(frozen=True)
class CameraMapping:
    """Maps logical camera roles to HDF5 image keys."""

    roles: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SPLAT_ROLES))

    def key_for(self, role: str) -> str:
        return self.roles[role]

    @classmethod
    def from_json(cls, text: str | None) -> CameraMapping:
        if not text:
            return cls()
        return cls(roles={str(k): str(v) for k, v in json.loads(text).items()})


def _as_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_json_dataset(dataset: h5py.Dataset) -> object:
    raw = dataset[()]
    return json.loads(_as_str(raw))


class Hdf5EpisodeReader:
    """Read one ``episodes/episode_<NNNNNN>.h5`` file (+ its MP4 videos)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.root_dir = self.path.parent.parent
        with h5py.File(self.path, "r") as h5:
            self.attrs = {key: h5.attrs[key] for key in h5.attrs}
            obs = h5["obs"]
            self.arm_joint_position = np.asarray(obs["arm_joint_position"][()], dtype=np.float32)
            self.gripper_joint_position = np.asarray(
                obs["gripper_joint_position"][()], dtype=np.float32
            )
            self.ee_position_world = np.asarray(obs["ee_position_world"][()], dtype=np.float32)
            self.ee_quat_world_wxyz = np.asarray(obs["ee_quat_world_wxyz"][()], dtype=np.float32)
            self.action_full = np.asarray(
                h5["action"]["full_joint_position_target"][()], dtype=np.float32
            )
            self._image_meta: dict[str, dict[str, object]] = {}
            if "images" in obs:
                for key in obs["images"]:
                    group = obs["images"][key]
                    self._image_meta[str(key)] = {attr: group.attrs[attr] for attr in group.attrs}
            self._planning_problem: dict[str, object] = {}
            if "meta" in h5 and "planning_problem" in h5["meta"]:
                loaded = _read_json_dataset(h5["meta"]["planning_problem"])
                if isinstance(loaded, dict):
                    self._planning_problem = loaded

    @property
    def num_frames(self) -> int:
        return int(self.arm_joint_position.shape[0])

    @property
    def instruction(self) -> str:
        return _as_str(self.attrs.get("instruction", ""))

    @property
    def fps(self) -> int:
        return int(self.attrs.get("fps", 20))

    @property
    def robot_name(self) -> str:
        return _as_str(self.attrs.get("robot_name", ""))

    @property
    def gripper_name(self) -> str:
        return _as_str(self.attrs.get("gripper_name", ""))

    @property
    def arm_joint_names(self) -> list[str]:
        names = self._planning_problem.get("arm_joint_names")
        if isinstance(names, list):
            return [str(n) for n in names]
        joint_names = self.attrs.get("joint_names")
        if joint_names is not None:
            decoded = json.loads(_as_str(joint_names))
            return [str(n) for n in decoded][: self.arm_joint_position.shape[1]]
        return [f"joint_{i}" for i in range(self.arm_joint_position.shape[1])]

    @property
    def gripper_open_value(self) -> float | None:
        value = self._planning_problem.get("gripper_open_value")
        return None if value is None else float(value)

    @property
    def gripper_closed_value(self) -> float | None:
        value = self._planning_problem.get("gripper_closed_value")
        return None if value is None else float(value)

    @property
    def image_keys(self) -> list[str]:
        return list(self._image_meta)

    def has_image(self, image_key: str) -> bool:
        return image_key in self._image_meta

    def read_video(self, image_key: str) -> np.ndarray:
        """Decode an MP4 stream into an ``(T, H, W, 3)`` uint8 array."""
        meta = self._image_meta[image_key]
        path = self.root_dir / _as_str(meta["video_path"])
        reader = imageio.get_reader(str(path))
        frames = [np.asarray(frame, dtype=np.uint8) for frame in reader]
        reader.close()
        return np.stack(frames, axis=0)


class Hdf5DatasetReader:
    """Iterate the per-episode HDF5 files under a collection root."""

    def __init__(self, root: str | Path) -> None:
        root = Path(root)
        self.episodes_dir = root if root.name == "episodes" else root / "episodes"

    def episode_paths(self) -> list[Path]:
        return sorted(self.episodes_dir.glob("episode_*.h5"))

    def __len__(self) -> int:
        return len(self.episode_paths())

    def __iter__(self) -> Iterator[Hdf5EpisodeReader]:
        for path in self.episode_paths():
            yield Hdf5EpisodeReader(path)


# Egocentric EE-frame correction applied by GR00T's DROID reference inference
# (and the zero-shot eval client) before encoding eef_9d.
_DROID_EEF_ROTATION_CORRECT = np.array(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def normalize_gripper(
    values: np.ndarray, open_value: float | None, closed_value: float | None
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if open_value is None or closed_value is None or float(open_value) == float(closed_value):
        return values
    span = float(closed_value) - float(open_value)
    return ((values - float(open_value)) / span).astype(np.float32)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Convert ``(..., 4)`` wxyz quaternions to ``(..., 3, 3)`` rotation matrices."""
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    mat = np.empty((*quat.shape[:-1], 3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    mat[..., 0, 1] = 2.0 * (x * y - z * w)
    mat[..., 0, 2] = 2.0 * (x * z + y * w)
    mat[..., 1, 0] = 2.0 * (x * y + z * w)
    mat[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    mat[..., 1, 2] = 2.0 * (y * z - x * w)
    mat[..., 2, 0] = 2.0 * (x * z - y * w)
    mat[..., 2, 1] = 2.0 * (y * z + x * w)
    mat[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return mat


def droid_eef_9d(position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """``(T, 9)``: xyz + rot6d, where rot6d is the first two rows of
    (R @ correction) — the encoding GR00T-N1.7-DROID expects."""
    rot = quat_wxyz_to_matrix(quat_wxyz) @ _DROID_EEF_ROTATION_CORRECT
    rot6d = rot[:, :2, :].reshape(rot.shape[0], 6).astype(np.float32)
    return np.concatenate([np.asarray(position, dtype=np.float32), rot6d], axis=1)


def _gripper_column(
    reader: Hdf5EpisodeReader, values: np.ndarray, *, normalize: bool
) -> np.ndarray:
    if not normalize:
        return np.asarray(values, dtype=np.float32)
    return normalize_gripper(values, reader.gripper_open_value, reader.gripper_closed_value)


def droid_state_action(
    reader: Hdf5EpisodeReader, *, normalize_gripper_value: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """``(T, 17)`` state and action in the ``oxe_droid_relative_eef_relative_joint``
    layout: ``[eef_9d (9), gripper_position (1), joint_position (7)]``.

    The action stores absolute targets: next-frame measured EE pose, recorded
    gripper target, recorded joint position target.
    """
    eef = droid_eef_9d(reader.ee_position_world, reader.ee_quat_world_wxyz)
    gripper_state = _gripper_column(
        reader, reader.gripper_joint_position, normalize=normalize_gripper_value
    )
    state = np.concatenate(
        [eef, gripper_state, reader.arm_joint_position], axis=1
    ).astype(np.float32)

    eef_next = np.concatenate([eef[1:], eef[-1:]], axis=0)
    gripper_target = _gripper_column(
        reader, reader.action_full[:, -1:], normalize=normalize_gripper_value
    )
    action = np.concatenate(
        [eef_next, gripper_target, reader.action_full[:, :-1]], axis=1
    ).astype(np.float32)
    return state, action
