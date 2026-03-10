import numpy as np
import teaserpp_python
from sklearn.neighbors import NearestNeighbors


def find_correspondences(src_feat, tgt_feat):

    nbrs = NearestNeighbors(n_neighbors=1).fit(tgt_feat)

    dist, idx = nbrs.kneighbors(src_feat)

    src_idx = np.arange(src_feat.shape[0])
    tgt_idx = idx[:,0]

    return src_idx, tgt_idx


def teaser_register(src_pts, tgt_pts, src_feat, tgt_feat):

    src_idx, tgt_idx = find_correspondences(src_feat, tgt_feat)

    src_corr = src_pts[src_idx]
    tgt_corr = tgt_pts[tgt_idx]

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()

    solver_params.noise_bound = 0.01
    solver_params.cbar2 = 1
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    solver.solve(src_corr.T, tgt_corr.T)

    sol = solver.getSolution()

    T = np.eye(4)

    T[:3,:3] = sol.rotation
    T[:3,3] = sol.translation

    return T