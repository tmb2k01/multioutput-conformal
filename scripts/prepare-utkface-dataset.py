import argparse
import os
import re
import shutil

from sklearn.model_selection import train_test_split

data_dir = "data/UTKFace"
part_dirs = [os.path.join(data_dir, part) for part in ["part1", "part2", "part3"]]
dataset_dir = os.path.join(data_dir, "merged")
pattern = re.compile(
    r"^(\d{1,3})_(0|1)_(0|1|2|3|4)_[^.]+\.(jpg|jpeg|png|bmp|webp)$",
    re.IGNORECASE,
)

def age_to_bin(age: str) -> str:
    age_int = int(age)

    if age_int < 5:
        return "0_4"
    if age_int < 10:
        return "5_9"
    if age_int < 20:
        return "10_19"
    if age_int < 30:
        return "20_29"
    if age_int < 40:
        return "30_39"
    if age_int < 50:
        return "40_49"
    if age_int < 60:
        return "50_59"
    return "60_plus"

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

        src_path = os.path.join(dataset_dir, filename)
        if not os.path.exists(dest_path):
            shutil.copy(src_path, images_dir)
        else:
            print(f"File already exists: {dest_path}")

        label_file = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(labels_dir, label_file), "w") as f:
            f.write(f"{labels[0]} {labels[1]} {labels[2]}")


def prepare_dataset(experiment: bool) -> None:
    age_labels = []
    sex_labels = []
    race_labels = []
    filtered_files = []

    for filename in os.listdir(dataset_dir):
        if not pattern.match(filename):
            print(f"Skipping invalid filename: {filename}")
            continue

        stem, _ = os.path.splitext(filename)
        age, gender, race, _ = stem.split("_")

        age_labels.append(age)
        sex_labels.append(gender)
        race_labels.append(race)
        filtered_files.append(filename)

    X = filtered_files
    y = list(zip(sex_labels, race_labels, age_labels, strict=True))

    stratify_y = [f"{gender}_{race}_{age_to_bin(age)}" for gender, race, age in y]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=42,
        stratify=stratify_y,
    )

    stratify_temp = [f"{gender}_{race}_{age_to_bin(age)}" for gender, race, age in y_temp]

    X_val, X_test_cal, y_val, y_test_cal = train_test_split(
        X_temp,
        y_temp,
        test_size=0.75,
        random_state=42,
        stratify=stratify_temp,
    )

    save_split_data(X_train, y_train, "train")
    save_split_data(X_val, y_val, "valid")

    if experiment:
        for i in range(5):
            stratify_test_cal = [
                f"{gender}_{race}_{age_to_bin(age)}" for gender, race, age in y_test_cal
            ]

            X_test, X_calib, y_test, y_calib = train_test_split(
                X_test_cal,
                y_test_cal,
                test_size=0.5,
                random_state=i,
                stratify=stratify_test_cal,
            )

            save_split_data(X_test, y_test, f"test_{i}")
            save_split_data(X_calib, y_calib, f"calib_{i}")

    else:
        stratify_test_cal = [
            f"{gender}_{race}_{age_to_bin(age)}" for gender, race, age in y_test_cal
        ]

        X_test, X_calib, y_test, y_calib = train_test_split(
            X_test_cal,
            y_test_cal,
            test_size=0.5,
            random_state=42,
            stratify=stratify_test_cal,
        )

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
