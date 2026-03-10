import numpy as np
import os
import nrrd
import time
import open3d as o3d

from abus_io.nrrd_io import read_nrrd
from features.dino_dense_feature import DenseDINO

from geometry.pointcloud import (
    volume_to_pointcloud,
    compute_fpfh,
    sample_dino_feature
)

from registration.ransac_registration import ransac_register
from registration.icp_refinement import icp_refine
from fusion.gpu_fusion import gpu_fuse


# ------------------------------------------------
# voxel downsample
# ------------------------------------------------

def voxel_downsample(points, voxel=5):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd = pcd.voxel_down_sample(voxel)

    return np.asarray(pcd.points)


# ------------------------------------------------
# feature fusion
# ------------------------------------------------

def fuse_features(f_geo, f_dino):

    f_geo = f_geo / (np.linalg.norm(f_geo, axis=1, keepdims=True) + 1e-8)
    f_dino = f_dino / (np.linalg.norm(f_dino, axis=1, keepdims=True) + 1e-8)

    fused = np.concatenate([f_geo, f_dino], axis=1)

    fused = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)

    return fused


# ------------------------------------------------
# direction initialization
# ------------------------------------------------

def direction_init(ref_vol, mov_vol, direction):

    T = np.eye(4)

    ref_shape = np.array(ref_vol.data.shape)
    mov_shape = np.array(mov_vol.data.shape)

    if direction == "left":

        T[2, 3] = -mov_shape[2] * 0.8

    elif direction == "right":

        T[2, 3] = ref_shape[2] * 0.8

    elif direction == "up":

        T[1, 3] = ref_shape[1] * 0.8

    elif direction == "down":

        T[1, 3] = -mov_shape[1] * 0.8

    return T


# ------------------------------------------------
# DINO cache
# ------------------------------------------------

def load_or_extract_dino(vol, extractor, cache_dir, name):

    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{name}_dino.npy")

    if os.path.exists(cache_path):

        print("Loading cached DINO:", name)

        return np.load(cache_path)

    print("Extracting DINO feature:", name)

    feat = extractor.extract_volume(vol.data)

    np.save(cache_path, feat)

    return feat


# ------------------------------------------------
# two view registration
# ------------------------------------------------

def register_two_views(vol_ref, vol_mov, extractor, direction=None, cache_dir=None):

    # -------------------------------
    # DINO feature
    # -------------------------------

    ref_name = getattr(vol_ref, "name", "ref")
    mov_name = getattr(vol_mov, "name", "mov")

    ref_dino = load_or_extract_dino(
        vol_ref,
        extractor,
        cache_dir,
        ref_name
    )

    mov_dino = load_or_extract_dino(
        vol_mov,
        extractor,
        cache_dir,
        mov_name
    )

    # -------------------------------
    # PCD extraction
    # -------------------------------

    print("Extract PCD")

    ref_pts = voxel_downsample(volume_to_pointcloud(vol_ref))
    mov_pts = voxel_downsample(volume_to_pointcloud(vol_mov))

    ref_pts = ref_pts[::5]
    mov_pts = mov_pts[::5]

    # -------------------------------
    # FPFH
    # -------------------------------

    print("Compute FPFH")

    ref_geo = compute_fpfh(ref_pts)
    mov_geo = compute_fpfh(mov_pts)

    # -------------------------------
    # DINO descriptor sampling
    # -------------------------------

    print("Sample DINO descriptors")

    ref_dino_desc = sample_dino_feature(
        ref_dino,
        ref_pts,
        vol_ref.data.shape
    )

    mov_dino_desc = sample_dino_feature(
        mov_dino,
        mov_pts,
        vol_mov.data.shape
    )

    # -------------------------------
    # feature fusion
    # -------------------------------

    print("Feature fusion")

    ref_feat = fuse_features(ref_geo, ref_dino_desc)
    mov_feat = fuse_features(mov_geo, mov_dino_desc)

    # -------------------------------
    # random limit points
    # -------------------------------

    max_points = 5000

    if ref_pts.shape[0] > max_points:

        idx = np.random.choice(ref_pts.shape[0], max_points, replace=False)

        ref_pts = ref_pts[idx]
        ref_feat = ref_feat[idx]

    if mov_pts.shape[0] > max_points:

        idx = np.random.choice(mov_pts.shape[0], max_points, replace=False)

        mov_pts = mov_pts[idx]
        mov_feat = mov_feat[idx]

    # -------------------------------
    # RANSAC
    # -------------------------------

    print("RANSAC registration")

    T_ransac = ransac_register(
        mov_pts,
        ref_pts,
        mov_feat,
        ref_feat
    )

    # -------------------------------
    # direction constraint
    # -------------------------------

    if direction is not None:

        print("Applying direction constraint:", direction)

        T_init = direction_init(vol_ref, vol_mov, direction)

        T_ransac = T_init @ T_ransac

    # -------------------------------
    # ICP
    # -------------------------------

    print("ICP refinement")

    T_icp = icp_refine(
        mov_pts,
        ref_pts,
        T_ransac,
        direction
    )

    print("Estimated translation:", T_icp[:3, 3])

    return T_icp


# ------------------------------------------------
# pipeline
# ------------------------------------------------

def run_pipeline(case_path):

    t0 = time.time()

    view1 = read_nrrd(f"{case_path}/view_1.nrrd")
    view2 = read_nrrd(f"{case_path}/view_2.nrrd")
    view3 = read_nrrd(f"{case_path}/view_3.nrrd")

    # 给 volume 加名字（用于 cache）
    view1.name = "view1"
    view2.name = "view2"
    view3.name = "view3"

    extractor = DenseDINO()

    case_name = os.path.basename(case_path)

    os.makedirs("results", exist_ok=True)

    cache_dir = f"{case_path}/cache"

    # ------------------------------------------------
    # STEP1
    # ------------------------------------------------

    print("\n===== STEP1 view2 to view1 =====")

    T12 = register_two_views(
        view1,
        view2,
        extractor,
        direction="left",
        cache_dir=cache_dir
    )

    fused1 = gpu_fuse(
        [view1, view2],
        [np.eye(4), T12]
    )

    fused1_path = f"results/{case_name}_fused1.nrrd"

    nrrd.write(fused1_path, fused1)

    print("Saved fused1:", fused1_path)

    # ------------------------------------------------
    # STEP2
    # ------------------------------------------------

    print("\n===== STEP2 view3 to fused1 =====")

    fused1_vol = type(view1)(
        fused1,
        view1.spacing,
        view1.origin
    )

    fused1_vol.name = "fused1"

    T23 = register_two_views(
        fused1_vol,
        view3,
        extractor,
        direction="right",
        cache_dir=cache_dir
    )

    fused_final = gpu_fuse(
        [fused1_vol, view3],
        [np.eye(4), T23]
    )

    final_path = f"results/{case_name}_final.nrrd"

    nrrd.write(final_path, fused_final)

    print("Saved final:", final_path)

    print("\nPipeline finished")

    print("Total time:", time.time() - t0)