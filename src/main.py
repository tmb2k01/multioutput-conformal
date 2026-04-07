

from core.calibrators import HighLevelCalibrator, LowLevelCalibrator
from core.models import HighLevelModel, LowLevelModel
from core.predictor import ConformalPredictor
from data.datamodule import MultiOutputDataModule


def main() -> None:
    data_root = "./data/UTKFace"
    batch_size: int = 64
    num_workers: int = 8
    task_num_classes = [2, 5]

    dm = MultiOutputDataModule(
        root_dir=str(data_root),
        task_num_classes=task_num_classes,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    predictor = ConformalPredictor.build(
        model_cls=LowLevelModel,
        calibrator_cls=HighLevelCalibrator,
        task_num_classes=[2, 5],
        cp_type="ccp_global_cluster_thresholds")
    predictor.fit(data_module=dm, max_epochs=1, train_model=False)

    predictor = ConformalPredictor.load(
        model_cls=LowLevelModel,
        calibrator_cls=HighLevelCalibrator,
        task_num_classes=task_num_classes,
        cp_type="ccp_global_cluster_thresholds")
    
    for X, y in dm.test_dataloader():
        pred_set = predictor.predict(X)
        print(pred_set)
        break


if __name__ == "__main__":
    main()
