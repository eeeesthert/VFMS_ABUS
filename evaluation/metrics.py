
import numpy as np

def rmse(src,tgt):
    return np.sqrt(np.mean((src-tgt)**2))

def tre(src,tgt):
    return np.mean(np.linalg.norm(src-tgt,axis=1))

def overlap(mask1,mask2):
    inter=(mask1&mask2).sum()
    union=(mask1|mask2).sum()
    return inter/union
