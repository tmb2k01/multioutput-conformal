from src.models.high_level_model import HighLevelModel

SGVEHICLE_COLOR = 12
SGVEHICLE_TYPE = 11
SGVEHICLE_TASK_NUM_CLASSES = [SGVEHICLE_COLOR, SGVEHICLE_TYPE]


def load_models():
    MODEL_HL = HighLevelModel.load_from_checkpoint(
        "models/sgvehicle-high-level-model.ckpt",
        map_location="cpu",
        task_num_classes=SGVEHICLE_TASK_NUM_CLASSES,
    )
    MODEL_HL = MODEL_HL.eval()


def launch():
    pass
