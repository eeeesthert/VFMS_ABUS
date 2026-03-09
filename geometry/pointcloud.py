
import numpy as np
from config.settings import POINT_THRESHOLD,POINT_SAMPLE

def volume_to_pointcloud(volume):

    img=volume.data

    mask=img>img.max()*POINT_THRESHOLD

    idx=np.argwhere(mask)

    if len(idx)>POINT_SAMPLE:
        idx=idx[np.random.choice(len(idx),POINT_SAMPLE,replace=False)]

    pts=idx.astype(float)

    pts[:,0]*=volume.spacing[0]
    pts[:,1]*=volume.spacing[1]
    pts[:,2]*=volume.spacing[2]

    return pts
