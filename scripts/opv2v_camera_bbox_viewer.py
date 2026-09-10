#!/usr/bin/env python3
"""Interactively verify OPV2V LiDAR-ground-truth projections in RGB cameras.

The OPV2V YAML annotations store vehicle cuboids in CARLA world coordinates,
as well as the world pose and intrinsics of each of the four RGB cameras.
This viewer transforms every 3-D cuboid from world coordinates into the
selected camera frame, projects its visible corners, and overlays its tight
2-D rectangle on the corresponding image.  It rejects distant/tiny labels and
boxes heavily covered by nearer vehicle projections.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml


WINDOW_NAME = "OPV2V 3D-to-2D ground-truth projection"
BOX_COLOR = (0, 220, 0)
WIRE_COLOR = (255, 210, 0)
TEXT_COLOR = (255, 255, 255)
EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/mnt/external/workspace/public/dataset/opv2v"),
    )
    parser.add_argument("--split", choices=("train", "validate", "test"), default="test")
    parser.add_argument("--scenario", help="Scenario directory name; default: first scenario")
    parser.add_argument("--agent", help="CAV directory name; default: first CAV")
    parser.add_argument("--frame", help="Frame key without .yaml; default: first frame")
    parser.add_argument("--camera", type=int, choices=range(4), default=0)
    parser.add_argument(
        "--max-distance", type=float, default=50.0,
        help="Ignore labels farther than this LiDAR-frame XY distance in metres.",
    )
    parser.add_argument(
        "--min-box-pixels", type=int, default=12,
        help="Ignore labels narrower or shorter than this many pixels.",
    )
    parser.add_argument(
        "--occlusion-threshold", type=float, default=0.65,
        help=(
            "Ignore a box when this fraction of its projected cuboid mask is "
            "covered by nearer projected vehicles; zero disables the filter."
        ),
    )
    parser.add_argument(
        "--min-lidar-points", type=int, default=3,
        help=(
            "Require at least this many selected-CAV LiDAR returns inside the "
            "3-D vehicle cuboid; zero disables LiDAR-supported visibility filtering."
        ),
    )
    parser.add_argument(
        "--lidar-box-margin", type=float, default=0.15,
        help="Metres added around each cuboid when counting LiDAR returns.",
    )
    parser.add_argument("--show-3d", action="store_true", help="Overlay projected 3-D cuboid edges")
    return parser.parse_args()


def pose_to_world(pose: list[float]) -> np.ndarray:
    """Return OPV2V/CARLA sensor-local to world transform for [x,y,z,r,y,p]."""
    x, y, z, roll, yaw, pitch = map(float, pose)
    cy, sy = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
    cr, sr = np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))
    cp, sp = np.cos(np.deg2rad(pitch)), np.sin(np.deg2rad(pitch))
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = (x, y, z)
    matrix[:3, :3] = np.array(
        [
            [cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr],
            [sp, -cp * sr, cp * cr],
        ]
    )
    return matrix


def vehicle_pose(vehicle: dict) -> list[float]:
    """Return the OPV2V world pose at the centre of a vehicle cuboid."""
    center = np.asarray(vehicle["center"], dtype=np.float64)
    location = np.asarray(vehicle["location"], dtype=np.float64)
    return [*(location + center), *vehicle["angle"]]


def vehicle_world_corners(vehicle: dict) -> np.ndarray:
    """Construct the eight world-coordinate corners of one OPV2V vehicle."""
    extent = np.asarray(vehicle["extent"], dtype=np.float64)
    local = np.array(
        [
            [extent[0], -extent[1], -extent[2]],
            [extent[0], extent[1], -extent[2]],
            [-extent[0], extent[1], -extent[2]],
            [-extent[0], -extent[1], -extent[2]],
            [extent[0], -extent[1], extent[2]],
            [extent[0], extent[1], extent[2]],
            [-extent[0], extent[1], extent[2]],
            [-extent[0], -extent[1], extent[2]],
        ],
        dtype=np.float64,
    )
    homogeneous = np.concatenate((local, np.ones((8, 1))), axis=1)
    return (pose_to_world(vehicle_pose(vehicle)) @ homogeneous.T).T[:, :3]


@lru_cache(maxsize=2)
def load_pcd_xyz(path: str) -> np.ndarray:
    """Load the XYZ columns from OPV2V's ASCII CARLA point-cloud files."""
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip().lower().startswith("data"):
                break
        else:
            raise ValueError(f"No PCD DATA header in {path}")
        points = np.loadtxt(stream, dtype=np.float32, usecols=(0, 1, 2))
    return np.atleast_2d(points)


def lidar_points_in_vehicle(
    points_world: np.ndarray, vehicle: dict, margin: float
) -> int:
    """Count real LiDAR returns within a ground-truth vehicle cuboid."""
    local_transform = np.linalg.inv(pose_to_world(vehicle_pose(vehicle)))
    homogeneous = np.concatenate(
        (points_world, np.ones((len(points_world), 1))), axis=1
    )
    local = (local_transform @ homogeneous.T).T[:, :3]
    extent = np.asarray(vehicle["extent"], dtype=np.float64) + margin
    return int(np.count_nonzero(np.all(np.abs(local) <= extent, axis=1)))


def load_opv2v_annotation(path: Path) -> dict:
    """Load OPV2V's trusted simulator annotation, including legacy NumPy tags.

    A small subset of OPV2V YAMLs serializes scalar values with PyYAML's old
    ``python/object/apply:numpy`` tags.  ``safe_load`` intentionally rejects
    these tags, while the dataset-owned files need them reconstructed.
    """
    with path.open("r", encoding="utf-8") as stream:
        return yaml.unsafe_load(stream)


def project_corners(
    corners_world: np.ndarray, camera: dict, image_shape: tuple[int, int, int]
) -> tuple[np.ndarray, tuple[int, int, int, int], float] | None:
    """Project a fully front-facing 3-D box into a clipped 2-D rectangle.

    CARLA sensor coordinates are (forward, right, up).  OpenCV's pinhole
    camera coordinates are (right, down, forward), hence [y, -z, x] below.
    """
    world_to_camera = np.linalg.inv(pose_to_world(camera["cords"]))
    homogeneous = np.concatenate((corners_world, np.ones((8, 1))), axis=1)
    sensor = (world_to_camera @ homogeneous.T).T[:, :3]
    camera_xyz = np.stack((sensor[:, 1], -sensor[:, 2], sensor[:, 0]), axis=1)
    depth = camera_xyz[:, 2]
    if np.any(depth <= 1e-3):
        return None
    intrinsic = np.asarray(camera["intrinsic"], dtype=np.float64)
    image_points = (intrinsic @ camera_xyz.T).T
    pixels = image_points[:, :2] / image_points[:, 2:3]

    height, width = image_shape[:2]
    x1 = max(0, int(np.floor(pixels[:, 0].min())))
    y1 = max(0, int(np.floor(pixels[:, 1].min())))
    x2 = min(width - 1, int(np.ceil(pixels[:, 0].max())))
    y2 = min(height - 1, int(np.ceil(pixels[:, 1].max())))
    if x1 >= x2 or y1 >= y2:
        return None
    return pixels, (x1, y1, x2, y2), float(depth.mean())


def lidar_distance(corners_world: np.ndarray, lidar_pose: list[float]) -> float:
    """Return a vehicle's horizontal distance from the selected CAV LiDAR."""
    center_world = corners_world.mean(axis=0)
    homogeneous = np.append(center_world, 1.0)
    center_lidar = np.linalg.inv(pose_to_world(lidar_pose)) @ homogeneous
    return float(np.linalg.norm(center_lidar[:2]))


def cuboid_mask(pixels: np.ndarray, image_shape: tuple[int, int, int]) -> np.ndarray:
    """Rasterize the visible cuboid silhouette for a conservative occlusion test."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(np.rint(pixels).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


@dataclass
class ProjectedVehicle:
    object_id: str
    pixels: np.ndarray
    rectangle: tuple[int, int, int, int]
    depth: float
    distance: float
    lidar_points: int | None


@dataclass
class ViewerState:
    scenarios: list[Path]
    scenario_index: int
    agent_index: int
    frame_index: int
    camera_index: int
    draw_3d: bool

    @property
    def scenario(self) -> Path:
        return self.scenarios[self.scenario_index]

    @property
    def agents(self) -> list[Path]:
        return sorted(path for path in self.scenario.iterdir() if path.is_dir())

    @property
    def agent(self) -> Path:
        return self.agents[self.agent_index]

    @property
    def frames(self) -> list[str]:
        return sorted(path.stem for path in self.agent.glob("*.yaml"))

    @property
    def frame(self) -> str:
        return self.frames[self.frame_index]

    def set_scenario(self, offset: int) -> None:
        self.scenario_index = (self.scenario_index + offset) % len(self.scenarios)
        self.agent_index = 0
        self.frame_index = 0

    def step_frame(self, offset: int) -> None:
        """Move through frames; crossing an end advances to the next scenario."""
        next_index = self.frame_index + offset
        if 0 <= next_index < len(self.frames):
            self.frame_index = next_index
            return
        previous_agent = self.agent.name
        self.scenario_index = (self.scenario_index + offset) % len(self.scenarios)
        agent_names = [agent.name for agent in self.agents]
        self.agent_index = agent_names.index(previous_agent) if previous_agent in agent_names else 0
        self.frame_index = 0 if offset > 0 else len(self.frames) - 1


def _initial_index(values: list, requested: str | None, key=lambda value: value.name) -> int:
    if requested is None:
        return 0
    for index, value in enumerate(values):
        if key(value) == requested:
            return index
    raise ValueError(f"Requested value {requested!r} is unavailable: {[key(v) for v in values]}")


def render(state: ViewerState, args: argparse.Namespace) -> tuple[np.ndarray, str]:
    yaml_path = state.agent / f"{state.frame}.yaml"
    image_path = state.agent / f"{state.frame}_camera{state.camera_index}.png"
    annotation = load_opv2v_annotation(yaml_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read {image_path}")
    camera = annotation[f"camera{state.camera_index}"]
    points_world = None
    pcd_path = state.agent / f"{state.frame}.pcd"
    if args.min_lidar_points > 0 and pcd_path.exists():
        points_lidar = load_pcd_xyz(str(pcd_path))
        homogeneous = np.concatenate(
            (points_lidar, np.ones((len(points_lidar), 1))), axis=1
        )
        points_world = (pose_to_world(annotation["lidar_pose"]) @ homogeneous.T).T[:, :3]
    candidates: list[ProjectedVehicle] = []
    for object_id, vehicle in annotation.get("vehicles", {}).items():
        corners_world = vehicle_world_corners(vehicle)
        projection = project_corners(corners_world, camera, image.shape)
        if projection is None:
            continue
        pixels, rectangle, depth = projection
        candidates.append(ProjectedVehicle(
            str(object_id), pixels, rectangle, depth,
            lidar_distance(corners_world, annotation["lidar_pose"]),
            (
                lidar_points_in_vehicle(points_world, vehicle, args.lidar_box_margin)
                if points_world is not None else None
            ),
        ))

    # Fill nearer vehicle silhouettes first.  This is a vehicle-only geometric
    # visibility test: OPV2V's RGB/YAML files do not supply a depth image or
    # semantic mask for static-structure occlusion.
    covered = np.zeros(image.shape[:2], dtype=np.uint8)
    visible = 0
    for candidate in sorted(candidates, key=lambda item: item.depth):
        x1, y1, x2, y2 = candidate.rectangle
        target_mask = cuboid_mask(candidate.pixels, image.shape)
        target_pixels = int(target_mask.sum())
        occluded_fraction = (
            float(np.count_nonzero((target_mask == 1) & (covered == 1)))
            / target_pixels if target_pixels else 1.0
        )
        small = (x2 - x1 < args.min_box_pixels or y2 - y1 < args.min_box_pixels)
        heavily_occluded = (
            args.occlusion_threshold > 0
            and occluded_fraction >= args.occlusion_threshold
        )
        keep = (
            candidate.distance <= args.max_distance
            and not small
            and not heavily_occluded
            and (
                candidate.lidar_points is None
                or candidate.lidar_points >= args.min_lidar_points
            )
        )
        if keep:
            visible += 1
            cv2.rectangle(image, (x1, y1), (x2, y2), BOX_COLOR, 2, cv2.LINE_AA)
            cv2.putText(image, candidate.object_id, (x1 + 3, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, BOX_COLOR, 2, cv2.LINE_AA)
            if state.draw_3d:
                points = np.rint(candidate.pixels).astype(np.int32)
                for start, end in EDGES:
                    cv2.line(image, tuple(points[start]), tuple(points[end]),
                             WIRE_COLOR, 1, cv2.LINE_AA)
        covered |= target_mask

    header = (
        f"split={state.scenario.parent.name}  scenario={state.scenario.name}  "
        f"CAV={state.agent.name}  frame={state.frame}  camera={state.camera_index}  "
        f"labels={visible}/{len(candidates)}  max distance={args.max_distance:g} m  "
        f"min LiDAR points={args.min_lidar_points if points_world is not None else 'off'}"
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 29), (22, 22, 22), -1)
    cv2.putText(image, header, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                TEXT_COLOR, 1, cv2.LINE_AA)
    return image, header


def main() -> None:
    args = parse_args()
    split_root = args.data_root / args.split
    scenarios = sorted(path for path in split_root.iterdir() if path.is_dir())
    if not scenarios:
        raise FileNotFoundError(f"No scenarios found in {split_root}")
    scenario_index = _initial_index(scenarios, args.scenario)
    provisional = ViewerState(scenarios, scenario_index, 0, 0, args.camera, args.show_3d)
    provisional.agent_index = _initial_index(provisional.agents, args.agent)
    provisional.frame_index = _initial_index(provisional.frames, args.frame, key=lambda value: value)
    state = provisional
    if args.max_distance <= 0:
        raise ValueError("--max-distance must be positive")
    if args.min_box_pixels < 1:
        raise ValueError("--min-box-pixels must be at least one")
    if args.min_lidar_points < 0:
        raise ValueError("--min-lidar-points cannot be negative")
    if args.lidar_box_margin < 0:
        raise ValueError("--lidar-box-margin cannot be negative")
    if not 0 <= args.occlusion_threshold <= 1:
        raise ValueError("--occlusion-threshold must be in [0, 1]")

    # waitKeyEx values vary between OpenCV/X11 builds; accept the common Linux
    # and Windows virtual-key values so the viewer remains arrow-key-only.
    left_keys = {81, 2424832, 65361}
    right_keys = {83, 2555904, 65363}
    up_keys = {82, 2490368, 65362}
    down_keys = {84, 2621440, 65364}
    print("Controls: ←/→ frame (crosses scenarios at the ends) | ↑/↓ camera | Q/Esc quit")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    while True:
        image, header = render(state, args)
        cv2.imshow(WINDOW_NAME, image)
        key = cv2.waitKeyEx(0)
        if key in (27, ord("q"), ord("Q")):
            break
        if key in left_keys:
            state.step_frame(-1)
        elif key in right_keys:
            state.step_frame(1)
        elif key in up_keys:
            state.camera_index = (state.camera_index - 1) % 4
        elif key in down_keys:
            state.camera_index = (state.camera_index + 1) % 4
        else:
            print(f"Ignored key code {key}; {header}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
