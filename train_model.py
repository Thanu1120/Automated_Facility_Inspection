from facility_inspection_model import train_facility_inspection_model

if __name__ == "__main__":
    data_directory = "dataset"  # Path to your dataset directory
    model_path = "facility_inspection_model.h5"  # Path to save the trained model

    train_facility_inspection_model(data_directory, model_path)
