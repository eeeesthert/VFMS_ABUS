import numpy as np
import cv2
import open3d as o3d


def compute_edges(volume):

    edges=[]

    for slice in volume:

        slice=slice.astype(np.float32)

        sobel=cv2.Sobel(slice,cv2.CV_32F,1,1)

        log=cv2.Laplacian(slice,cv2.CV_32F)

        harris=cv2.cornerHarris(slice,2,3,0.04)

        edge=(np.abs(sobel)+np.abs(log)+np.abs(harris))/3

        edges.append(edge)

    return np.stack(edges)


def volume_to_pointcloud(vol):

    volume=vol.data

    edge_vol=compute_edges(volume)

    pts=np.argwhere(edge_vol>0.2)

    pts=pts.astype(np.float32)

    return pts


def compute_fpfh(points):

    pcd=o3d.geometry.PointCloud()

    pcd.points=o3d.utility.Vector3dVector(points)

    pcd.estimate_normals()

    fpfh=o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=25,
            max_nn=100
        )
    )

    return np.asarray(fpfh.data).T
    


def sample_dino_feature(dino_volume, points, vol_shape):

    Df, Hf, Wf, C = dino_volume.shape
    Dz, Hy, Wx = vol_shape

    scale_y = Hf / Hy
    scale_x = Wf / Wx

    pts = points.copy()

    z = np.clip(pts[:,0].astype(int),0,Df-1)

    y = np.clip((pts[:,1]*scale_y).astype(int),0,Hf-1)

    x = np.clip((pts[:,2]*scale_x).astype(int),0,Wf-1)

    descriptors = dino_volume[z,y,x]

    return descriptors