
import numpy as np

def initialize_triview(lat_pts, ap_pts, med_pts):

    # assume AP center, LAT left, MED right
    lat_shift = np.array([-100,0,0])
    med_shift = np.array([100,0,0])

    T_lat = np.eye(4)
    T_med = np.eye(4)

    T_lat[:3,3] = lat_shift
    T_med[:3,3] = med_shift

    return T_lat, np.eye(4), T_med
