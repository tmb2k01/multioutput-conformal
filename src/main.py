

from core.calibrators import HighLevelCalibrator
from core.models import HighLevelModel
from core.predictor import ConformalPredictor


def main() -> None:
    # predictor = ConformalPredictor.build(
    #     model_cls=HighLevelModel,
    #     calibrator_cls=HighLevelCalibrator,
    #     task_num_classes=[2, 5],
    #     cp_type="ccp_class_thresholds")
    # predictor.fit(data_root="./data/UTKFace", max_epochs=1)

    predictor = ConformalPredictor.load(
        model_cls=HighLevelModel,
        calibrator_cls=HighLevelCalibrator,
        task_num_classes=[2, 5],
        cp_type="ccp_class_thresholds")

if __name__ == "__main__":
    main()
