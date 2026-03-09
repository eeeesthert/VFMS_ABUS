
import os
from pipeline.stitch_pipeline import run_pipeline
from config.settings import DATASET_PATH

def main():
    cases = sorted(os.listdir(DATASET_PATH))
    for case in cases:
        print("Processing:", case)
        run_pipeline(os.path.join(DATASET_PATH, case))

if __name__ == "__main__":
    main()
