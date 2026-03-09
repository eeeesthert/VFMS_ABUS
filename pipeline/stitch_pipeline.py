import os
import nrrd
import numpy as np
from abus_io.nrrd_io import read_nrrd
from features.dino_dense_feature import DenseDINO
from geometry.pointcloud import volume_to_pointcloud
from initialization.triview_init import initialize_triview
from registration.teaser_registration import teaser_register
from fusion.gpu_fusion import gpu_fuse

def run_pipeline(case_path):

    case_name = os.path.basename(case_path)

    lat = read_nrrd(f"{case_path}/LAT.nrrd")
    ap = read_nrrd(f"{case_path}/AP.nrrd")
    med = read_nrrd(f"{case_path}/MED.nrrd")

    extractor = DenseDINO()

    print("Extracting features")
    lat_feat = extractor.extract_volume(lat.data)
    ap_feat = extractor.extract_volume(ap.data)

    lat_pts = volume_to_pointcloud(lat)
    ap_pts = volume_to_pointcloud(ap)

    print("Auto tri-view initialization")
    T_lat, T_ap, T_med = initialize_triview(lat_pts, ap_pts, lat_pts)

    print("TEASER++ registration")
    T = teaser_register(lat_pts, ap_pts)

    print("GPU fusion")
    fused = gpu_fuse([lat, ap, med], [T_lat, T_ap, T_med])

    print("Stitching finished")

    # save_result
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, case_name + "_stitched.nrrd")

    nrrd.write(output_file, fused)

    print("Saved:", output_file)
