import os
from pipeline.stitch_pipeline import run_pipeline

DATASET_PATH="dataset"

def main():

    cases=os.listdir(DATASET_PATH)

    for case in cases:

        case_path=os.path.join(DATASET_PATH,case)

        print("Processing:",case)

        run_pipeline(case_path)

if __name__=="__main__":
    main()