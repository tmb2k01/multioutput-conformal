import argparse
import os
import shutil

from sklearn.model_selection import train_test_split

data_dir = "data/UTKFace"
part_dirs = [os.path.join(data_dir, part) for part in ["part1", "part2", "part3"]]
dataset_dir = os.path.join(data_dir, "merged")


def merge_parts() -> None:
    for part_dir in part_dirs:
        os.makedirs(dataset_dir, exist_ok=True)
        for filename in os.listdir(part_dir):
            src_path = os.path.join(part_dir, filename)
            dest_path = os.path.join(dataset_dir, filename)
            if os.path.isfile(src_path):
                shutil.copy(src_path, dest_path)
        shutil.rmtree(part_dir)


def save_split_data(X_split: list[str], y_split: list[tuple[str, str]], split_name: str) -> None:
    images_dir = os.path.join(data_dir, split_name, "images")
    labels_dir = os.path.join(data_dir, split_name, "labels")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for filename, labels in zip(X_split, y_split, strict=True):
        dest_path = os.path.join(images_dir, filename)
        if os.path.exists(dest_path):
            print(f"File already exists: {dest_path}")
            continue

        src_path = os.path.join(dataset_dir, filename)
        shutil.copy(src_path, images_dir)

        label_file = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(labels_dir, label_file), "w") as f:
            f.write(f"{labels[0]} {labels[1]}")


def prepare_dataset(experiment: bool) -> None:
    file_paths = list(os.listdir(dataset_dir))
    sex_labels = []
    race_labels = []

    filtered_files = []

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for path in file_paths:
        filename = os.path.basename(path)
        stem, ext = os.path.splitext(filename)

        if ext.lower() not in valid_exts:
            print(f"Skipping invalid image extension: {filename}")
            continue

        parts = stem.split("_")

        if len(parts) < 4:
            print(f"Skipping invalid filename: {filename}")
            continue

        if not parts[0].isdigit() or not parts[1].isdigit() or not parts[2].isdigit():
            print(f"Skipping invalid filename: {filename}")
            continue

        gender = parts[1]
        race = parts[2]

        sex_labels.append(gender)
        race_labels.append(race)
        filtered_files.append(filename)

    X = filtered_files
    y = list(zip(sex_labels, race_labels, strict=True))

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=[f"{g}_{r}" for g, r in y]
    )
    X_val, X_test_cal, y_val, y_test_cal = train_test_split(
        X_temp, y_temp, test_size=0.75, random_state=42, stratify=[f"{g}_{r}" for g, r in y_temp]
    )

    for split in ["train", "valid"]:
        os.makedirs(os.path.join(data_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, split, "labels"), exist_ok=True)

    save_split_data(X_train, y_train, "train")
    save_split_data(X_val, y_val, "valid")

    if experiment:
        for i in range(5):
            X_test, X_calib, y_test, y_calib = train_test_split(
                X_test_cal, y_test_cal, test_size=0.5, random_state=i, 
                stratify=[f"{g}_{r}" for g, r in y_test_cal]
            )

            for split in [f"test_{i}", f"calib_{i}"]:
                os.makedirs(os.path.join(data_dir, split, "images"), exist_ok=True)
                os.makedirs(os.path.join(data_dir, split, "labels"), exist_ok=True)
            save_split_data(X_test, y_test, f"test_{i}")
            save_split_data(X_calib, y_calib, f"calib_{i}")

    else:
        X_test, X_calib, y_test, y_calib = train_test_split(
            X_test_cal, y_test_cal, test_size=0.5, random_state=42, 
            stratify=[f"{g}_{r}" for g, r in y_test_cal]
        )
        for split in ["test", "calib"]:
            os.makedirs(os.path.join(data_dir, split, "images"), exist_ok=True)
            os.makedirs(os.path.join(data_dir, split, "labels"), exist_ok=True)
        save_split_data(X_test, y_test, "test")
        save_split_data(X_calib, y_calib, "calib")


    shutil.rmtree(dataset_dir)


def main(args: argparse.Namespace) -> None:
    experiment = args.experiment
    merge_parts()
    prepare_dataset(experiment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", action="store_true")
    args = parser.parse_args()
    main(args)
