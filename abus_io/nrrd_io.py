
import nrrd
import numpy as np

class Volume:

    def __init__(self,data,spacing,origin):
        self.data=data.astype(np.float32)
        self.spacing=np.array(spacing)
        self.origin=np.array(origin)

def read_nrrd(path):

    data,header=nrrd.read(path)

    spacing=header.get("spacings",[1,1,1])
    origin=header.get("space origin",[0,0,0])

    return Volume(data,spacing,origin)
