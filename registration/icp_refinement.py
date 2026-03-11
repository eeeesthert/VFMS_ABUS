import open3d as o3d
import numpy as np

def icp_refine(src_pts, tgt_pts, T_init, direction=None):

    # -----------------------------
    # build point cloud
    # -----------------------------

    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()

    src.points = o3d.utility.Vector3dVector(src_pts)
    tgt.points = o3d.utility.Vector3dVector(tgt_pts)

    # -----------------------------
    # ICP
    # -----------------------------

    result = o3d.pipelines.registration.registration_icp(

        src,
        tgt,

        max_correspondence_distance=5.0,

        init=T_init,

        estimation_method=
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),

        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=50
        )
    )

    # read only
    T = result.transformation.copy()

    # -----------------------------
    # direction constraint
    # -----------------------------

    if direction == "left":

        T[0, 3] = -abs(T[0, 3])

    elif direction == "right":

        T[0, 3] = abs(T[0, 3])

    elif direction == "up":

        T[1, 3] = abs(T[1, 3])

    elif direction == "down":

        T[1, 3] = -abs(T[1, 3])
    print("ICP translation:", T[:3, 3])
    return T
