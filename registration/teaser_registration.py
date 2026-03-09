
import numpy as np
import teaserpp_python

def teaser_register(src,tgt):

    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    solver.solve(src.T, tgt.T)
    sol = solver.getSolution()

    T = np.eye(4)
    T[:3,:3] = sol.rotation
    T[:3,3] = sol.translation

    return T
