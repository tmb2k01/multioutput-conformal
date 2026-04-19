from core.calibrators import HighLevelCalibrator, LowLevelCalibrator
from core.models import HighLevelModel, LowLevelModel
from core.predictor import ConformalPredictor
from data.datamodule import MultiOutputDataModule


def main() -> None:
    artifact_root = "./artifacts/SGVehicle/ll_model"
    data_root = "./data/SGVehicle"
    batch_size: int = 64
    num_workers: int = 8
    task_num_classes = [12, 11]

    dm = MultiOutputDataModule(
        root_dir=str(data_root),
        task_num_classes=task_num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
        iter='0'
    )
    predictor = ConformalPredictor.build(
        model_cls=LowLevelModel,
        calibrator_cls=LowLevelCalibrator,
        task_num_classes=task_num_classes,
        cp_type="scp_global_threshold",
        artifacts_dir=artifact_root)
    predictor.fit(data_module=dm, train_model=True, calibrate_model=False)

if __name__ == "__main__":
    main()
