"""Point-cloud registration (information beyond boxes) for every cached CooperScene ego<->CAV pair.
# Usage: python scripts/register_cooperscene_scans_gicp.py 0 0 outputs/calibration/cooperscene_gicp_sigma0.npz  (sigma 0 = from the true poses)

CAV scan (its own LiDAR frame) is registered to the ego scan with multi-scale Generalized ICP, initialised at
the CAV's REPORTED (noisy) pose -- the same starting point the box methods get. Output per pair: the 2-D
correction D = T_icp o N^-1 (psi, tx, ty), GICP fitness and inlier RMSE at the finest scale, point counts."""
import sys, glob, os, multiprocessing as mp
import numpy as np, torch, open3d as o3d
sys.path.insert(0, "scripts")
import check_verify_core as c
base = c.base
from embedding_aware_belt_fusion.alignformer_ext.propose_cache import noisy_poses

ROOT = "/home/dhia/cooperscene_preds/pp/TownCooperScene"
VOXEL, SCALES = 0.3, (3.0, 1.5, 0.6)
sigma, seed, out = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
records = c.load_frame_table(c.Path("outputs/cooperscene_test_frame_table.pt"))
keys = []
for s in sorted(glob.glob("outputs/cooperscene_geometry_test_cache/*.pt")):
    keys += [(fr["scenario_id"], fr["frame_id"]) for fr in torch.load(s, weights_only=False)]
cache = dict(np.load(f"outputs/propose_cache_cooperscene/sigma_{sigma:g}_seed_{seed}.npz"))


def cloud(scenario, agent, frame):
    pc = o3d.io.read_point_cloud(f"{ROOT}/{scenario}/{agent}/{frame}.pcd")
    p = np.asarray(pc.points)
    r = np.linalg.norm(p[:, :2], axis=1)
    pc = pc.select_by_index(np.flatnonzero((r > 3.0) & (r < 80.0)))
    pc = pc.voxel_down_sample(VOXEL)
    pc.estimate_covariances(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=20))
    return pc


def pose2d(m):
    return np.array([m[0, 3], m[1, 3], np.arctan2(m[1, 0], m[0, 0])])


def work(f):
    torch.set_num_threads(1)
    r = records[f]; scen, fid = keys[f]
    pairs = [p for p in np.flatnonzero(cache["frame"] == f)]
    if not pairs:
        return []
    ego = cloud(scen, "1", fid)
    rep = noisy_poses(base, r, f, sigma, seed)
    res = []
    for p in pairs:
        cid = sorted(r.cav_boxes)[int(cache["cav_rank"][p])]
        n = np.asarray(base.x1_to_x2(rep[cid], r.ego_pose), dtype=np.float64)
        src = cloud(scen, cid, fid)
        T = n.copy(); fit = rmse = 0.0
        for d in SCALES:
            reg = o3d.pipelines.registration.registration_generalized_icp(
                src, ego, d, T, o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
            T, fit, rmse = reg.transformation, reg.fitness, reg.inlier_rmse
        dcorr = pose2d(T @ np.linalg.inv(n))
        res.append((int(p), dcorr[2], dcorr[0], dcorr[1], fit, rmse, len(src.points), len(ego.points)))
    return res


if __name__ == "__main__":
    frames = sorted(set(int(x) for x in cache["frame"]))
    rows = []
    with mp.get_context("fork").Pool(16) as pool:
        for i, res in enumerate(pool.imap_unordered(work, frames, chunksize=4)):
            rows += res
            if i % 100 == 0:
                print(f"{i}/{len(frames)} frames", flush=True)
    a = np.array(sorted(rows))
    np.savez(out, pair=a[:, 0].astype(int), psi=a[:, 1], tx=a[:, 2], ty=a[:, 3], fitness=a[:, 4], rmse=a[:, 5],
             src_points=a[:, 6], ego_points=a[:, 7])
    print("saved", out, len(a), "pairs")
