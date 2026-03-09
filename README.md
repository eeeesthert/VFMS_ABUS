
VFMStitch_ABUS_research

Research-grade ABUS 3-view stitching framework.

Modules included:
- Dense DINO feature extraction
- PCA compression
- Trilinear interpolation descriptors
- FPFH geometric descriptors
- Descriptor fusion and matching
- TEASER++ robust registration
- ICP refinement
- Automatic tri-view initialization
- Overlap detection
- GPU volume stitching
- Evaluation metrics (TRE, RMSE, overlap)

Dataset format:

dataset/
    case001/
        LAT.nrrd
        AP.nrrd
        MED.nrrd

Run:

pip install torch torchvision open3d pynrrd scipy scikit-learn teaserpp-python

python main.py
