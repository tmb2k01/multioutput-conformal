import copy
import yaml

from core.predictor import ConformalPredictor
from data.datamodule import MultiOutputDataModule

def deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)

    return result


def run(exp: dict):
    print(f"Running experiment: {exp['name']}")
    data_module_config = exp.get("data_module", {})
    predictor_config = exp.get("predictor", {})

    for i in range(data_module_config.get("max_iter", 1)):
        dm = MultiOutputDataModule(
            root_dir=data_module_config["data_root"],
            task_num_classes=data_module_config["task_num_classes"],
            batch_size=data_module_config["batch_size"],
            num_workers=data_module_config["num_workers"],
        )

        predictor = ConformalPredictor.build(
            model_cls=globals()[predictor_config["model_cls"]],
            calibrator_cls=globals()[predictor_config["calibrator_cls"]],
            task_num_classes=data_module_config["task_num_classes"],
            cp_type=predictor_config["cp_type"],
            artifacts_dir=predictor_config["artifact_root"],
            nonconformity_key=predictor_config["nonconformity_key"],
        )

        predictor.fit(
            data_module=dm,
            train_model=predictor_config["train_model"],
            calibrate_model=predictor_config["calibrate_model"],
        )



def run_experiments(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    defaults = config.get("defaults", {})
    experiments = config["experiments"]

    resolved_experiments = [
        deep_merge(defaults, exp)
        for exp in experiments
    ]

    for exp in resolved_experiments:
        run(exp)