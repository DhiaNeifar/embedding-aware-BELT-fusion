# Scripts

## OPV2V RGB ground-truth viewer

`opv2v_camera_bbox_viewer.py` transforms each OPV2V vehicle cuboid from CARLA
world coordinates to the selected RGB camera, then draws its tight projected
2-D bounding box. It is intended to validate the camera-label geometry before
generating YOLO labels.

```bash
python scripts/opv2v_camera_bbox_viewer.py \
  --data-root /mnt/external/workspace/public/dataset/opv2v \
  --split test \
  --max-distance 50 \
  --min-box-pixels 12 \
  --occlusion-threshold 0.65 \
  --min-lidar-points 3 \
  --show-3d
```

Controls: left/right arrows change frame and cross to the previous/next scenario
at a boundary; up/down arrows select the camera; `Q` or Esc exits. Use
`--scenario`, `--agent`, `--frame`, and `--camera` to choose the initial view.

The viewer removes labels beyond `--max-distance`, labels below the requested
pixel size, labels whose projected cuboid is at least `--occlusion-threshold`
covered by nearer vehicle cuboids, and labels without enough real LiDAR returns
inside their cuboid. The last filter removes vehicles hidden by walls/buildings
in the selected CAV's LiDAR view. Set `--min-lidar-points 0` to disable it.
