from experiment import run_experiments


def main() -> None:
    # dm = MultiOutputDataModule(
    #     root_dir="./data/UTKFace",
    #     task_num_classes=[2, 5],
    #     split_idx = 0
    # )

    # predictor = ConformalPredictor.build(
    #     model_cls=LowLevelModel,
    #     calibrator_cls=LowLevelCalibrator,
    #     task_num_classes=[2, 5],
    #     artifacts_dir="./artifacts/UTKFace",
    # )

    # predictor.fit(data_module=dm, calibrate_model = False)

    run_experiments('/home/marcs/projects/masters-thesis/experiments/utkface/hinge/hl_hl_cal.yaml')
    run_experiments('/home/marcs/projects/masters-thesis/experiments/utkface/hinge/hl_ll_cal.yaml')
    run_experiments('/home/marcs/projects/masters-thesis/experiments/utkface/hinge/ll_hl_cal.yaml')
    run_experiments('/home/marcs/projects/masters-thesis/experiments/utkface/hinge/ll_ll_cal.yaml')

if __name__ == "__main__":
    main()
