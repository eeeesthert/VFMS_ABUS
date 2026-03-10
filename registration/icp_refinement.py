import open3d as o3d
import numpy as np


def icp_refine(src_pts, tgt_pts, init_T):

    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()

    src.points = o3d.utility.Vector3dVector(src_pts)
    tgt.points = o3d.utility.Vector3dVector(tgt_pts)

    result = o3d.pipelines.registration.registration_icp(

        src,
        tgt,

        max_correspondence_distance=2.0,

        init=init_T,

        estimation_method=
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),

        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=50
        )
    )

    return result.transformation