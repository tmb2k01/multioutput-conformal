from experiment import run_experiments
from data.datamodule import MultiOutputDataModule
from core.predictor import ConformalPredictor
from core.models import HighLevelModel, LowLevelModel
from core.calibrators import HighLevelCalibrator, LowLevelCalibrator

def main() -> None:
    # dm = MultiOutputDataModule(
    #     root_dir="./data/SGVehicle",
    #     task_num_classes=[12, 11],
    #     split_idx = 0
    # )

    # predictor = ConformalPredictor.build(
    #     model_cls=HighLevelModel,
    #     calibrator_cls=HighLevelCalibrator,
    #     task_num_classes=[12, 11],
    # )

    # predictor.fit(data_module=dm, calibrate_model = False)

    run_experiments('/home/marcs/projects/masters-thesis/experiments/sgvehicle/hinge/hl_hl_cal.yaml')

if __name__ == "__main__":
    main()
