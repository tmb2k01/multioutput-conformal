import os
import shutil
from sklearn.model_selection import train_test_split

data_dir = "data/UTKFaces"
part_dirs = [os.path.join(data_dir, part) for part in ["part1", "part2", "part3"]]
dataset_dir = os.path.join(data_dir, "merged")

def merge_parts():
    for part_dir in part_dirs:
        os.makedirs(dataset_dir, exist_ok=True)
        for filename in os.listdir(part_dir):
            src_path = os.path.join(part_dir, filename)
            dest_path = os.path.join(dataset_dir, filename)
            if os.path.isfile(src_path):
                shutil.copy(src_path, dest_path)
        shutil.rmtree(part_dir)


def save_split_data(X_split, y_split, split_name):
    for filename, labels in zip(X_split, y_split):
        src_path = os.path.join(dataset_dir, filename)
        dest_path = os.path.join(data_dir, split_name, "images", filename)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy(src_path, dest_path)

        label_file = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(data_dir, split_name, "labels", label_file)
        with open(label_path, "w") as f:
            f.write(f"{labels[0]} {labels[1]}")


def prepare_dataset():
    file_paths = [filename for filename in os.listdir(dataset_dir)]
    sex_labels = []
    race_labels = []

    filtered_files = []

    for path in file_paths:
        filename = os.path.basename(path)
        try:
            parts = filename.split("_")
            sex = parts[1]
            race = parts[2]
            sex_labels.append(sex)
            race_labels.append(race)
            filtered_files.append(path)
        except IndexError:
            print(f"Skipping invalid filename: {filename}")

    X = filtered_files
    y = list(zip(sex_labels, race_labels))

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test_cal, y_val, y_test_cal = train_test_split(
        X_temp, y_temp, test_size=0.75, random_state=42
    )
    X_test, X_calib, y_test, y_calib = train_test_split(
        X_test_cal, y_test_cal, test_size=0.5, random_state=42
    )

    for split in ["train", "valid", "test", "calib"]:
        os.makedirs(os.path.join(data_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, split, "labels"), exist_ok=True)

    save_split_data(X_train, y_train, "train")
    save_split_data(X_val, y_val, "valid")
    save_split_data(X_test, y_test, "test")
    save_split_data(X_calib, y_calib, "calib")
    
    shutil.rmtree(dataset_dir)


def main():
    merge_parts()
    prepare_dataset()


if __name__ == "__main__":
    main()
