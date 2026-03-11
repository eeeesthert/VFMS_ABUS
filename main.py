import os
from pipeline.stitch_pipeline import run_pipeline

DATASET_PATH="dataset"
DEVICE=os.getenv("VFMS_DEVICE", "cuda:1")

def main():

    cases=os.listdir(DATASET_PATH)

    for case in cases:

        case_path=os.path.join(DATASET_PATH,case)

        print("Processing:",case)

        run_pipeline(case_path, device=DEVICE)

if __name__=="__main__":
    main()
