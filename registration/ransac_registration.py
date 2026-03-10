import numpy as np
import open3d as o3d


def ransac_register(src_pts, tgt_pts, src_feat, tgt_feat):

    src = o3d.geometry.PointCloud()
    tgt = o3d.geometry.PointCloud()

    src.points = o3d.utility.Vector3dVector(src_pts)
    tgt.points = o3d.utility.Vector3dVector(tgt_pts)

    src_feature = o3d.pipelines.registration.Feature()
    tgt_feature = o3d.pipelines.registration.Feature()

    src_feature.data = src_feat.T
    tgt_feature.data = tgt_feat.T

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(

        src,
        tgt,
        src_feature,
        tgt_feature,
        mutual_filter=True,
        max_correspondence_distance=5.0,

        estimation_method=
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),

        ransac_n=4,

        checkers=[
            o3d.pipelines.registration.
            CorrespondenceCheckerBasedOnDistance(5.0)
        ],

        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            #400000,500
            20000,
            200
        )
    )

    return result.transformation