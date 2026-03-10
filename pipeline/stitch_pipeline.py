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

from initialization.triview_init import initialize_triview
from registration.teaser_registration import teaser_register
from registration.ransac_registration import ransac_register
from registration.icp_refinement import icp_refine
from fusion.gpu_fusion import gpu_fuse

def voxel_downsample(points, voxel=3):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd = pcd.voxel_down_sample(voxel)

    return np.asarray(pcd.points)

def fuse_features(f_geo, f_dino):

    f_geo = f_geo / (np.linalg.norm(f_geo, axis=1, keepdims=True) + 1e-8)

    f_dino = f_dino / (np.linalg.norm(f_dino, axis=1, keepdims=True) + 1e-8)

    fused = np.concatenate([f_geo, f_dino], axis=1)

    fused = fused / (np.linalg.norm(fused, axis=1, keepdims=True) + 1e-8)

    return fused


def run_pipeline(case_path):
    
    t0=time.time()


    # -------------------------
    # load volumes
    # -------------------------

    lat = read_nrrd(f"{case_path}/LAT.nrrd")
    ap  = read_nrrd(f"{case_path}/AP.nrrd")
    med = read_nrrd(f"{case_path}/MED.nrrd")


    extractor = DenseDINO()


    # -------------------------
    # 1 DINO semantic features
    # -------------------------

    print("Extracting DINO features")

    lat_dino = extractor.extract_volume(lat.data)
    ap_dino  = extractor.extract_volume(ap.data)
    med_dino = extractor.extract_volume(med.data)
    print("DINO time:",time.time()-t0)



    # -------------------------
    # 2 Edge-based PCD extraction
    # -------------------------

    print("Extract PCD")

    lat_pts = voxel_downsample(volume_to_pointcloud(lat))
    ap_pts  = voxel_downsample(volume_to_pointcloud(ap))
    med_pts = voxel_downsample(volume_to_pointcloud(med))



    # optional downsample (0.2)

    lat_pts = lat_pts[::5]
    ap_pts  = ap_pts[::5]
    med_pts  = med_pts[::5]



    # -------------------------
    # 3 Geometric descriptor
    # -------------------------

    print("Compute FPFH")

    lat_geo = compute_fpfh(lat_pts)
    ap_geo  = compute_fpfh(ap_pts)
    med_geo  = compute_fpfh(med_pts)



    # -------------------------
    # 4 DINO descriptor sampling
    # -------------------------

    print("Sample DINO descriptors")

    #lat_dino_desc = sample_dino_feature(lat_dino, lat_pts)
    #ap_dino_desc  = sample_dino_feature(ap_dino, ap_pts)
    lat_dino_desc = sample_dino_feature(
        lat_dino,
        lat_pts,
        lat.data.shape
    )
    
    ap_dino_desc = sample_dino_feature(
        ap_dino,
        ap_pts,
        ap.data.shape
    )

    med_dino_desc = sample_dino_feature(
        med_dino,
        med_pts,
        med.data.shape
    )

    # -------------------------
    # 5 Feature fusion
    # -------------------------

    print("Feature fusion")

    lat_feat = fuse_features(lat_geo, lat_dino_desc)
    ap_feat  = fuse_features(ap_geo, ap_dino_desc)
    med_feat = fuse_features(med_geo, med_dino_desc)
    
    max_points = 5000

    if lat_pts.shape[0] > max_points:
    
        idx = np.random.choice(lat_pts.shape[0], max_points, replace=False)
    
        lat_pts = lat_pts[idx]
        lat_feat = lat_feat[idx]
    
    if ap_pts.shape[0] > max_points:
    
        idx = np.random.choice(ap_pts.shape[0], max_points, replace=False)
    
        ap_pts = ap_pts[idx]
        ap_feat = ap_feat[idx]
        
    if med_pts.shape[0] > max_points:
    
        idx = np.random.choice(med_pts.shape[0], max_points, replace=False)
    
        med_pts = med_pts[idx]
        med_feat = med_feat[idx]



    # -------------------------
    # 6 Tri-view initialization
    # -------------------------

    print("Auto tri-view initialization")
    

    T_lat, T_ap, T_med = initialize_triview(
        lat_pts,
        ap_pts,
        med_pts
    )



    # -------------------------
    # 7 Robust registration
    # -------------------------
    print("RANSAC coarse registration")

    T_lat_ransac = ransac_register(
        lat_pts,
        ap_pts,
        lat_feat,
        ap_feat
    )
    
    T_med_ransac = ransac_register(
        med_pts,
        ap_pts,
        med_feat,
        ap_feat
    )
    
    #print("TEASER++ robust registration")
    
    #T_teaser = teaser_register(
     #   lat_pts,
      #  ap_pts,
       # lat_feat,
        #ap_feat
        #init_T=T_ransac
    #)
    
    print("ICP refinement")
    
    T_lat_icp = icp_refine(
        lat_pts,
        ap_pts,
        T_lat_ransac
    )
    
    T_med_icp = icp_refine(
        med_pts,
        ap_pts,
        T_med_ransac
    )


    # -------------------------
    # 8 Volume fusion
    # -------------------------

    print("GPU fusion")

    fused = gpu_fuse(
        [lat, ap, med],
        [T_lat_icp, np.eye(4), T_med_icp]
    )



    # -------------------------
    # 9 Save result
    # -------------------------

    print("Saving result")

    os.makedirs("results", exist_ok=True)
    
    case_name = os.path.basename(case_path)
    
    output_path = f"results/{case_name}_stitched.nrrd"
    
    nrrd.write(output_path, fused)
    
    print("Saved:", output_path)


    print("Stitching finished")
