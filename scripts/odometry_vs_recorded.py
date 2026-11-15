#!/usr/bin/env python3
"""Per-frame odometry against recorded motion (paper Sec. 5.1): per agent, recorded-pose step minus KISS-ICP step,
and the one-step inconsistency of the recorded relative pose against KISS-ICP propagation. Reads frame tables and
outputs/odometry/ only. Run with the opencood env: python scripts/odometry_vs_recorded.py"""
import sys, numpy as np
sys.path.insert(0, 'scripts')
import check_verify_core, reproduce_opv2v_sweep as base
from embedding_aware_belt_fusion.alignformer_ext import odometry
def p2(m):
    m=np.asarray(m,float); return np.array([m[0,3],m[1,3],np.arctan2(m[1,0],m[0,0])])
for name, table, odo, geo in [
    ('v2xreal test', 'outputs/v2xreal_test_frame_table.pt', 'outputs/odometry/v2xreal_test.npz', 'outputs/v2xreal_geometry_test_cache'),
    ('v2xreal val', 'outputs/v2xreal_val_frame_table.pt', 'outputs/odometry/v2xreal_val.npz', 'outputs/v2xreal_geometry_val_cache'),
    ('opv2v test', 'outputs/opv2v_frame_table.pt', 'outputs/odometry/opv2v_test_kissicp.npz', 'outputs/r140_geometry_test_cache')]:
    recs = check_verify_core.load_frame_table(table)
    tab = odometry.OdometryTable.for_records(odo, geo, recs)
    dt, dy, rel = [], [], []
    # relative-pose jitter: change of recorded relative pose minus change predicted by both agents' KISS-ICP motion
    for f in range(1, len(recs)):
        r, q = recs[f], recs[f-1]
        if tab.frames[f][0] != tab.frames[f-1][0]: continue
        for cav in r.cav_poses:
            if cav not in q.cav_poses: continue
            ko = tab.between(f, f-1, cav); ke = tab.between(f-1, f, odometry.EGO)
            if ko is None or ke is None: continue
            rec_step = p2(base.x1_to_x2(r.cav_poses[cav], q.cav_poses[cav]))  # CAV pose at f in its frame f-1
            ko_fwd = tab.between(f, f-1, cav)
            d = rec_step - ko_fwd; d[2] = np.arctan2(np.sin(d[2]), np.cos(d[2]))
            dt.append(np.hypot(d[0], d[1])); dy.append(abs(np.degrees(d[2])))
            # relative transform at f predicted from f-1 with KISS-ICP vs recorded at f
            Tprev = np.asarray(base.x1_to_x2(q.cav_poses[cav], q.ego_pose), float)
            Tnow = np.asarray(base.x1_to_x2(r.cav_poses[cav], r.ego_pose), float)
            def mat(x):
                c,s=np.cos(x[2]),np.sin(x[2]); M=np.eye(4); M[:2,:2]=[[c,-s],[s,c]]; M[0,3],M[1,3]=x[0],x[1]; return M
            # X_f = A^-1 X_{f-1} B with A = ego motion f-1->f (ego pose f in frame f-1), B = cav pose f in frame f-1
            A = mat(tab.between(f, f-1, odometry.EGO)); B = mat(ko_fwd)
            pred = np.linalg.inv(A) @ Tprev @ B
            e = np.linalg.inv(pred) @ Tnow
            rel.append(np.hypot(e[0,3], e[1,3]))
    dt, dy, rel = map(np.asarray, (dt, dy, rel))
    print(f"{name:13s} n={len(dt):5d} | per-agent step: recorded - KISS-ICP  median {np.median(dt)*100:5.1f} cm  p90 {np.percentile(dt,90)*100:6.1f} cm  yaw med {np.median(dy):.3f} deg"
          f" | relative-pose one-step inconsistency median {np.median(rel)*100:5.1f} cm p90 {np.percentile(rel,90)*100:6.1f} cm")
