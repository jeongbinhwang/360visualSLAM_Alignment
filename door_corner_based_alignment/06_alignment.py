#!/usr/bin/env python3
"""
Usage example.

python3 "06_alignment.py" \
  --keyframes ~/00_keyframes.csv \ 
  --slam_doors /home/jb/workspace/360video/graph_codes/00_final_bim_initialpose/1stfloor_door_corner_result/02_landmarks_door_mask.csv \
  --bim_doors /home/jb/workspace/360video/graph_codes/BIMcorners_door_LEVEL_1_46.csv \
  --out_dir /home/jb/workspace/360video/graph_codes/00_final_bim_initialpose/1stfloor_door_corner_result/99_test_noflatten_v3_paper/ \
  --scale_min 1.0 \
  --scale_max 2.0 \
  --z_output_frame absolute_bim

keyframes: ## "00_keyframes.csv" is in the "path" directory which is the same one with "path" of "03.py".
slam_doors: ## "02_landmarks_door_mask.csv" is in the "path" directory which is the same one with "path" of "03.py".
bim_doors: output of 00.py
out_dir: Where you want to store final result.
scale_min, scale_max: the range of scale factor
"""

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import time

# -----------------------------
# IO / utility
# -----------------------------

def read_xyz_csv(path, xcol=0, ycol=1, zcol=2, header=None):
    df = pd.read_csv(path, header=("infer" if header == "infer" else None))
    pts = df.iloc[:, [xcol, ycol, zcol]].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    good = np.isfinite(pts).all(axis=1)
    return pts[good], df, good


def voxel_downsample(points, voxel_size):
    if voxel_size <= 0 or len(points) == 0:
        return points
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    out = np.zeros((inv.max() + 1, points.shape[1]), dtype=float)
    counts = np.bincount(inv)
    for d in range(points.shape[1]):
        out[:, d] = np.bincount(inv, weights=points[:, d]) / counts
    return out


def sample_rows(points, n, rng):
    if len(points) <= n:
        return points
    idx = rng.choice(len(points), size=n, replace=False)
    return points[idx]


def rot2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def detect_vertical_axis_from_keyframes(kf):
    ranges = np.ptp(kf, axis=0)
    return int(np.argmin(ranges)), ranges


def detect_bim_floor_z(bim_doors, cluster_tol=0.15, percentile=1.0):
    """Detect lowest dominant BIM z cluster.

    BIM door corners can include both bottom and top corners. For upper floors,
    absolute z can be around 8~12 m. This function detects the lower door-corner
    cluster, which is usually the level/base elevation.
    """
    z = np.asarray(bim_doors[:, 2], dtype=float)
    z = z[np.isfinite(z)]
    if len(z) == 0:
        return 0.0
    z_min = float(np.min(z))
    low = z[z <= z_min + cluster_tol]
    if len(low) >= max(3, int(0.005 * len(z))):
        return float(np.median(low))
    return float(np.percentile(z, percentile))


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class AxisConfig:
    vertical_axis: int
    h_axes: Tuple[int, int]
    h_signs: Tuple[int, int]


@dataclass
class Candidate:
    score: float
    scale: float
    theta: float
    tx: float
    ty: float
    z_scale: float
    z_offset: float
    vertical_axis: int
    h_axes: Tuple[int, int]
    h_signs: Tuple[int, int]


def generate_axis_configs(vertical_axis):
    hs = [a for a in [0, 1, 2] if a != vertical_axis]
    configs = []
    for h_axes in [tuple(hs), tuple(hs[::-1])]:
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                configs.append(AxisConfig(vertical_axis, h_axes, (sx, sy)))
    return configs


def make_horizontal_xy(points, cfg: AxisConfig):
    xy = points[:, list(cfg.h_axes)].copy()
    xy[:, 0] *= cfg.h_signs[0]
    xy[:, 1] *= cfg.h_signs[1]
    return xy


def make_axis_matrix(cfg: AxisConfig):
    """Return A where aligned_col = A @ original_col.

    aligned = [signed_horizontal_1, signed_horizontal_2, vertical]
    """
    A = np.zeros((3, 3), dtype=float)
    A[0, cfg.h_axes[0]] = cfg.h_signs[0]
    A[1, cfg.h_axes[1]] = cfg.h_signs[1]
    A[2, cfg.vertical_axis] = 1.0
    return A


def transform_xy(xy, scale, theta, tx, ty):
    return scale * (xy @ rot2(theta).T) + np.array([tx, ty], dtype=float)


# -----------------------------
# XY score/search/refine: keep v1 behavior
# -----------------------------

def trimmed_nn_score_xy(pxy, bim_tree, trim_ratio=0.70, clip=5.0):
    d, _ = bim_tree.query(pxy, k=1)
    if clip is not None:
        d = np.minimum(d, clip)
    d = np.sort(d)
    n = max(5, int(len(d) * trim_ratio))
    n = min(n, len(d))
    return float(np.mean(d[:n]))


def coarse_search_one_axis_config(
    slam_xy,
    bim_xy,
    scale_min,
    scale_max,
    scale_steps,
    theta_step_deg,
    rng,
    num_slam_samples=160,
    num_bim_samples=700,
    num_translation_votes=5000,
    vote_voxel=0.5,
    keep_top=5,
):
    slam_s = sample_rows(slam_xy, num_slam_samples, rng)
    bim_s = sample_rows(bim_xy, num_bim_samples, rng)
    bim_tree = cKDTree(bim_xy)
    scales = np.linspace(scale_min, scale_max, scale_steps)
    thetas = np.deg2rad(np.arange(0.0, 360.0, theta_step_deg))
    out = []

    for scale in scales:
        for theta in thetas:
            R = rot2(theta)
            slam_rt_s = scale * (slam_s @ R.T)
            n_votes = min(num_translation_votes, len(slam_rt_s) * len(bim_s))
            si = rng.integers(0, len(slam_rt_s), size=n_votes)
            bi = rng.integers(0, len(bim_s), size=n_votes)
            votes = bim_s[bi] - slam_rt_s[si]

            votes3 = np.c_[votes, np.zeros(len(votes))]
            votes_ds = voxel_downsample(votes3, vote_voxel)[:, :2]
            votes_ds = sample_rows(votes_ds, 600, rng)

            for tx, ty in votes_ds:
                p = transform_xy(slam_xy, scale, theta, tx, ty)
                score = trimmed_nn_score_xy(p, bim_tree, trim_ratio=0.70, clip=5.0)
                out.append((score, scale, theta, float(tx), float(ty)))

    out.sort(key=lambda x: x[0])
    return out[:keep_top]


def refine_xy(
    cand,
    slam_xy,
    bim_xy,
    scale_min,
    scale_max,
    scale_prior=None,
    scale_prior_weight=0.01,
    clip=3.0,
):
    """Refine XY transform while enforcing the user-given scale bounds."""
    _, scale0, theta0, tx0, ty0 = cand
    tree = cKDTree(bim_xy)

    if scale_min <= 0 or scale_max <= 0 or scale_min > scale_max:
        raise ValueError(f"Invalid scale bounds: scale_min={scale_min}, scale_max={scale_max}")

    scale0 = float(np.clip(scale0, scale_min, scale_max))
    if scale_prior is None:
        scale_prior = math.sqrt(scale_min * scale_max)

    def residual(x):
        log_s, theta, tx, ty = x
        scale = math.exp(log_s)
        p = transform_xy(slam_xy, scale, theta, tx, ty)
        d, idx = tree.query(p, k=1)
        q = bim_xy[idx]
        r = np.clip(p - q, -clip, clip).reshape(-1)
        r_scale = np.array([scale_prior_weight * math.log(scale / scale_prior)])
        return np.r_[r, r_scale]

    x0 = np.array([math.log(scale0), theta0, tx0, ty0], dtype=float)
    lower = np.array([math.log(scale_min), -np.inf, -np.inf, -np.inf], dtype=float)
    upper = np.array([math.log(scale_max),  np.inf,  np.inf,  np.inf], dtype=float)

    res = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=500,
    )

    log_s, theta, tx, ty = res.x
    scale = float(np.clip(math.exp(log_s), scale_min, scale_max))
    p = transform_xy(slam_xy, scale, theta, tx, ty)
    score = trimmed_nn_score_xy(p, tree, trim_ratio=0.70, clip=None)
    return score, scale, theta, float(tx), float(ty)


# -----------------------------
# Z mapping: use v2 floor/local logic
# -----------------------------

def estimate_z_mapping(
    slam_doors,
    bim_doors,
    cfg: AxisConfig,
    scale,
    z_mode="translation_only",
    z_anchor="bim_floor",
    bim_floor_cluster_tol=0.15,
):
    """Estimate absolute BIM z = z_scale * SLAM_vertical + z_offset.

    No flattening is performed.

    z_mode:
      - affine_with_xy_scale: z_scale = estimated XY scale. This makes XYZ use one uniform scale.
      - translation_only: z_scale = 1.0. This only shifts z without scaling it.

    z_anchor:
      - bim_floor: align low SLAM vertical reference to detected BIM floor/base z.
      - bim_median: align median SLAM vertical to median BIM z.
    """
    v = np.asarray(slam_doors[:, cfg.vertical_axis], dtype=float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        raise ValueError("No valid SLAM vertical values.")

    if z_mode == "affine_with_xy_scale":
        z_scale = float(scale)
    elif z_mode == "translation_only":
        z_scale = 1.0
    else:
        raise ValueError(z_mode)

    if z_anchor == "bim_floor":
        bim_z_ref = detect_bim_floor_z(bim_doors, cluster_tol=bim_floor_cluster_tol)
        # Use lower percentile of SLAM vertical values to match the lower BIM door-corner/base cluster.
        slam_v_ref = float(np.percentile(v, 5.0))
    elif z_anchor == "bim_median":
        bim_z_ref = float(np.median(bim_doors[:, 2]))
        slam_v_ref = float(np.median(v))
    else:
        raise ValueError(z_anchor)

    z_offset = float(bim_z_ref - z_scale * slam_v_ref)
    return float(z_scale), z_offset


# -----------------------------
# Estimate full candidate
# -----------------------------

def estimate_transform_axis_search(
    slam_doors,
    bim_doors,
    vertical_axis,
    scale_min,
    scale_max,
    scale_steps,
    theta_step_deg,
    slam_voxel,
    bim_voxel,
    keep_top_per_axis,
    random_seed,
    z_mode,
    z_anchor,
    bim_floor_cluster_tol,
):
    rng = np.random.default_rng(random_seed)
    bim_ds = voxel_downsample(bim_doors, bim_voxel)
    bim_xy = bim_ds[:, :2]
    all_refined = []

    print(f"[INFO] BIM doors: {len(bim_doors)} raw -> {len(bim_ds)} downsampled")
    print("[INFO] Testing horizontal axis order/sign candidates:")

    for cfg in generate_axis_configs(vertical_axis):
        slam_xy_raw = make_horizontal_xy(slam_doors, cfg)
        slam_xy_ds = voxel_downsample(slam_xy_raw, slam_voxel)

        coarse = coarse_search_one_axis_config(
            slam_xy_ds,
            bim_xy,
            scale_min,
            scale_max,
            scale_steps,
            theta_step_deg,
            rng,
            keep_top=keep_top_per_axis,
        )

        refined = [
            refine_xy(
                c,
                slam_xy_ds,
                bim_xy,
                scale_min=scale_min,
                scale_max=scale_max,
            )
            for c in coarse
        ]
        refined.sort(key=lambda x: x[0])
        best_local = refined[0]

        z_scale, z_offset = estimate_z_mapping(
            slam_doors,
            bim_doors,
            cfg,
            best_local[1],
            z_mode=z_mode,
            z_anchor=z_anchor,
            bim_floor_cluster_tol=bim_floor_cluster_tol,
        )

        cand = Candidate(
            score=best_local[0],
            scale=best_local[1],
            theta=best_local[2],
            tx=best_local[3],
            ty=best_local[4],
            z_scale=z_scale,
            z_offset=z_offset,
            vertical_axis=cfg.vertical_axis,
            h_axes=cfg.h_axes,
            h_signs=cfg.h_signs,
        )
        all_refined.append(cand)

        print(
            f"  v={cfg.vertical_axis}, h_axes={cfg.h_axes}, signs={cfg.h_signs} -> "
            f"score={cand.score:.4f}, scale={cand.scale:.6f}, "
            f"theta={math.degrees(cand.theta):.3f}, tx={cand.tx:.3f}, ty={cand.ty:.3f}, "
            f"z_scale={cand.z_scale:.3f}, z_offset={cand.z_offset:.3f}"
        )

    all_refined.sort(key=lambda c: c.score)
    print("[INFO] Top candidates overall:")
    for i, c in enumerate(all_refined[:10], 1):
        print(
            f"  #{i}: score={c.score:.4f}, v={c.vertical_axis}, h_axes={c.h_axes}, signs={c.h_signs}, "
            f"scale={c.scale:.6f}, theta_deg={math.degrees(c.theta):.3f}, "
            f"tx={c.tx:.3f}, ty={c.ty:.3f}, z_scale={c.z_scale:.6f}, z_offset={c.z_offset:.3f}"
        )
    return all_refined[0]


# -----------------------------
# Transform / output
# -----------------------------

def transform_points_absolute(points, cand: Candidate):
    cfg = AxisConfig(cand.vertical_axis, cand.h_axes, cand.h_signs)
    xy = make_horizontal_xy(points, cfg)
    out = np.zeros((len(points), 3), dtype=float)
    out[:, :2] = transform_xy(xy, cand.scale, cand.theta, cand.tx, cand.ty)
    out[:, 2] = cand.z_scale * points[:, cand.vertical_axis] + cand.z_offset
    return out


def make_matrix(cand: Candidate, z_translation_delta=0.0):
    """Affine 4x4 for original SLAM xyz -> BIM xyz.

    z_translation_delta = -bim_floor_z gives a level-local output matrix.
    """
    cfg = AxisConfig(cand.vertical_axis, cand.h_axes, cand.h_signs)
    A = make_axis_matrix(cfg)

    R3 = np.eye(3)
    R3[:2, :2] = rot2(cand.theta)

    S = np.diag([cand.scale, cand.scale, cand.z_scale])
    M3 = S @ R3 @ A

    T = np.eye(4)
    T[:3, :3] = M3
    T[:3, 3] = [cand.tx, cand.ty, cand.z_offset + z_translation_delta]
    return T, A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyframes", required=True)
    ap.add_argument("--slam_doors", required=True)
    ap.add_argument("--bim_doors", required=True)
    ap.add_argument("--out_dir", default="result_final_xy_v1_z_v2")
    ap.add_argument("--header", default=None, choices=[None, "infer"])

    ap.add_argument("--kf_xcol", type=int, default=0)
    ap.add_argument("--kf_ycol", type=int, default=1)
    ap.add_argument("--kf_zcol", type=int, default=2)
    ap.add_argument("--slam_xcol", type=int, default=0)
    ap.add_argument("--slam_ycol", type=int, default=1)
    ap.add_argument("--slam_zcol", type=int, default=2)
    ap.add_argument("--bim_xcol", type=int, default=0)
    ap.add_argument("--bim_ycol", type=int, default=1)
    ap.add_argument("--bim_zcol", type=int, default=2)

    ap.add_argument("--slam_vertical_axis", type=int, default=None, choices=[0, 1, 2],
                    help="Manual SLAM vertical axis among selected xyz columns. If omitted, smallest keyframe range is used.")
    ap.add_argument("--scale_min", type=float, default=0.6)
    ap.add_argument("--scale_max", type=float, default=3.5)
    ap.add_argument("--scale_steps", type=int, default=40)
    ap.add_argument("--theta_step_deg", type=float, default=3.0)
    ap.add_argument("--slam_voxel", type=float, default=0.10)
    ap.add_argument("--bim_voxel", type=float, default=0.05)
    ap.add_argument("--keep_top_per_axis", type=int, default=5)
    ap.add_argument("--random_seed", type=int, default=7)

    # Z options: default applies the estimated XY scale to Z as well.
    ap.add_argument("--door_z_mode", default="affine_with_xy_scale",
                    choices=["translation_only", "affine_with_xy_scale"],
                    help="Z mapping. affine_with_xy_scale applies the estimated XY scale to z; translation_only only shifts z.")
    ap.add_argument("--z_anchor", default="bim_floor", choices=["bim_floor", "bim_median"],
                    help="BIM z reference used for z offset. bim_floor uses lowest BIM door-corner cluster.")
    ap.add_argument("--z_output_frame", default="level_local", choices=["level_local", "absolute_bim"],
                    help="CSV bim_z output. level_local subtracts detected BIM floor/base z; absolute_bim keeps BIM elevation.")
    ap.add_argument("--bim_floor_cluster_tol", type=float, default=0.15)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    keyframes, _, _ = read_xyz_csv(args.keyframes, args.kf_xcol, args.kf_ycol, args.kf_zcol, args.header)
    slam_doors, _, _ = read_xyz_csv(args.slam_doors, args.slam_xcol, args.slam_ycol, args.slam_zcol, args.header)
    bim_doors, _, _ = read_xyz_csv(args.bim_doors, args.bim_xcol, args.bim_ycol, args.bim_zcol, args.header)
    now = time.time()
    if len(keyframes) < 3:
        raise ValueError("Not enough keyframes loaded.")
    if len(slam_doors) < 3:
        raise ValueError("Not enough SLAM door candidates loaded.")
    if len(bim_doors) < 3:
        raise ValueError("Not enough BIM door corners loaded.")

    auto_v, ranges = detect_vertical_axis_from_keyframes(keyframes)
    vertical_axis = args.slam_vertical_axis if args.slam_vertical_axis is not None else auto_v
    bim_floor_z = detect_bim_floor_z(bim_doors, cluster_tol=args.bim_floor_cluster_tol)

    print(f"[INFO] keyframe coordinate ranges selected columns [0,1,2] = {ranges}")
    print(f"[INFO] detected vertical axis by min range = {auto_v}; used vertical axis = {vertical_axis}")
    print(f"[INFO] BIM z range = {float(np.min(bim_doors[:, 2])):.6f} ~ {float(np.max(bim_doors[:, 2])):.6f}")
    print(f"[INFO] detected BIM floor/base z = {bim_floor_z:.6f}")
    print(f"[INFO] XY method = v1 trimmed NN search/refine")
    print(f"[INFO] Z method = {args.door_z_mode}, anchor={args.z_anchor}, output_frame={args.z_output_frame}")

    best = estimate_transform_axis_search(
        slam_doors=slam_doors,
        bim_doors=bim_doors,
        vertical_axis=vertical_axis,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        scale_steps=args.scale_steps,
        theta_step_deg=args.theta_step_deg,
        slam_voxel=args.slam_voxel,
        bim_voxel=args.bim_voxel,
        keep_top_per_axis=args.keep_top_per_axis,
        random_seed=args.random_seed,
        z_mode=args.door_z_mode,
        z_anchor=args.z_anchor,
        bim_floor_cluster_tol=args.bim_floor_cluster_tol,
    )

    keyframes_abs = transform_points_absolute(keyframes, best)
    slam_doors_abs = transform_points_absolute(slam_doors, best)

    if args.z_output_frame == "level_local":
        keyframes_out = keyframes_abs.copy()
        slam_doors_out = slam_doors_abs.copy()
        keyframes_out[:, 2] -= bim_floor_z
        slam_doors_out[:, 2] -= bim_floor_z
        z_translation_delta = -bim_floor_z
    else:
        keyframes_out = keyframes_abs.copy()
        slam_doors_out = slam_doors_abs.copy()
        z_translation_delta = 0.0

    T_abs, A = make_matrix(best, z_translation_delta=0.0)
    T_out, _ = make_matrix(best, z_translation_delta=z_translation_delta)

    out_kf = pd.DataFrame({
        "slam_original_0": keyframes[:, 0],
        "slam_original_1": keyframes[:, 1],
        "slam_original_2": keyframes[:, 2],
        "bim_x": keyframes_out[:, 0],
        "bim_y": keyframes_out[:, 1],
        "bim_z": keyframes_out[:, 2],
        "bim_z_absolute": keyframes_abs[:, 2],
        "bim_z_level_local": keyframes_abs[:, 2] - bim_floor_z,
    })
    out_doors = pd.DataFrame({
        "slam_original_0": slam_doors[:, 0],
        "slam_original_1": slam_doors[:, 1],
        "slam_original_2": slam_doors[:, 2],
        "bim_x": slam_doors_out[:, 0],
        "bim_y": slam_doors_out[:, 1],
        "bim_z": slam_doors_out[:, 2],
        "bim_z_absolute": slam_doors_abs[:, 2],
        "bim_z_level_local": slam_doors_abs[:, 2] - bim_floor_z,
    })

    # Diagnostics for XY nearest match.
    tree = cKDTree(bim_doors[:, :2])
    d_xy, idx = tree.query(slam_doors_abs[:, :2], k=1)
    nearest = bim_doors[idx]
    out_diag = pd.DataFrame({
        "slam_original_0": slam_doors[:, 0],
        "slam_original_1": slam_doors[:, 1],
        "slam_original_2": slam_doors[:, 2],
        "slam_to_bim_x_abs": slam_doors_abs[:, 0],
        "slam_to_bim_y_abs": slam_doors_abs[:, 1],
        "slam_to_bim_z_abs": slam_doors_abs[:, 2],
        "nearest_bim_x": nearest[:, 0],
        "nearest_bim_y": nearest[:, 1],
        "nearest_bim_z": nearest[:, 2],
        "nearest_xy_dist": d_xy,
    })

    out_kf_path = os.path.join(args.out_dir, "keyframes_transformed_to_BIM.csv")
    out_doors_path = os.path.join(args.out_dir, "slam_doors_transformed_to_BIM.csv")
    out_diag_path = os.path.join(args.out_dir, "slam_doors_match_diagnostics.csv")
    out_T_abs_path = os.path.join(args.out_dir, "slam_to_bim_transform_4x4_absolute_BIM.csv")
    out_T_out_path = os.path.join(args.out_dir, "slam_to_bim_transform_4x4_output_frame.csv")
    out_A_path = os.path.join(args.out_dir, "slam_axis_matrix_3x3.csv")
    out_json_path = os.path.join(args.out_dir, "slam_to_bim_transform_params.json")

    out_kf.to_csv(out_kf_path, index=False)
    out_doors.to_csv(out_doors_path, index=False)
    out_diag.to_csv(out_diag_path, index=False)
    np.savetxt(out_T_abs_path, T_abs, delimiter=",")
    np.savetxt(out_T_out_path, T_out, delimiter=",")
    np.savetxt(out_A_path, A, delimiter=",")

    params = asdict(best)
    params.update({
        "theta_deg": math.degrees(best.theta),
        "keyframe_ranges": ranges.tolist(),
        "detected_vertical_axis": auto_v,
        "used_vertical_axis": vertical_axis,
        "door_z_mode": args.door_z_mode,
        "z_anchor": args.z_anchor,
        "z_output_frame": args.z_output_frame,
        "detected_bim_floor_z": bim_floor_z,
        "bim_z_min": float(np.min(bim_doors[:, 2])),
        "bim_z_max": float(np.max(bim_doors[:, 2])),
        "matrix_4x4_absolute_BIM": T_abs.tolist(),
        "matrix_4x4_output_frame": T_out.tolist(),
        "axis_matrix_3x3_aligned_col_equals_A_times_original_col": A.tolist(),
        "note": "Final version: XY uses v1 trimmed nearest-neighbor search/refine; Z uses the same estimated XY scale by default plus BIM floor/local output logic. No z flattening and no constant z.",
    })
    with open(out_json_path, "w") as f:
        json.dump(params, f, indent=2)

    # Optional XY plot
    try:
        import matplotlib.pyplot as plt
        out_plot_path = os.path.join(args.out_dir, "alignment_result.png")
        plt.figure(figsize=(9, 9))
        plt.scatter(bim_doors[:, 0], bim_doors[:, 1], s=8, label="BIM door corners")
        plt.scatter(slam_doors_out[:, 0], slam_doors_out[:, 1], s=8, label="Transformed SLAM door candidates")
        plt.plot(keyframes_out[:, 0], keyframes_out[:, 1], linewidth=1.5, label="Transformed keyframes")
        plt.axis("equal")
        plt.xlabel("BIM X")
        plt.ylabel("BIM Y")
        plt.title("SLAM-to-BIM alignment: XY v1 + Z v2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_plot_path, dpi=200)
        plt.close()
    except Exception as e:
        out_plot_path = None
        print(f"[WARN] Failed to save alignment plot: {e}")

    print("\n[RESULT]")
    print(f"score                 = {best.score:.6f}")
    print(f"scale                 = {best.scale:.9f}")
    print(f"theta_deg             = {math.degrees(best.theta):.6f}")
    print(f"tx, ty                = {best.tx:.6f}, {best.ty:.6f}")
    print(f"z_scale, z_offset_abs = {best.z_scale:.6f}, {best.z_offset:.6f}")
    print(f"BIM floor/base z      = {bim_floor_z:.6f}")
    print(f"output z frame        = {args.z_output_frame}")
    print(f"axis config           = vertical={best.vertical_axis}, h_axes={best.h_axes}, h_signs={best.h_signs}")
    print("\n[SAVED]")
    print(out_kf_path)
    print(out_doors_path)
    print(out_diag_path)
    print(out_T_abs_path)
    print(out_T_out_path)
    print(out_A_path)
    print(out_json_path)
    print("DURATION: ",time.time()-now)
    if out_plot_path:
        print(out_plot_path)


if __name__ == "__main__":
    main()
