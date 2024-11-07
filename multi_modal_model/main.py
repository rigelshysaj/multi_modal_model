import pandas as pd
from step1_pipeline import Step1Pipeline
from step2_pipeline import Step2Pipeline

def main():
    # Update the paths according to your project structure
    CSV_PATH = 'data/candidates_data.csv'
    IMAGE_DIR = 'data/spacecraft_images'

    # Load data
    data = pd.read_csv(CSV_PATH)

    # Step 1: Training tabular model
    print("Step 1: Training tabular model...")
    step1 = Step1Pipeline()
    step1.train(data)
    print("Step 1 completed.\n")

    # Step 2: Training multimodal model
    print("Step 2: Training multimodal model...")
    step2 = Step2Pipeline()
    step2.train(data, IMAGE_DIR)
    print("Step 2 completed.")

if __name__ == "__main__":
    main()
