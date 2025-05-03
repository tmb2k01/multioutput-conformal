from src.models.high_level_model import HighLevelModel

MDC_COLOR = 12
MDC_TYPE = 11
MDC_TASK_NUM_CLASSES = [MDC_COLOR, MDC_TYPE]


def load_models():
    MODEL_HL = HighLevelModel.load_from_checkpoint(
        "models/mdc-high-level-model.ckpt",
        map_location="cpu",
        task_num_classes=MDC_TASK_NUM_CLASSES,
    )
    MODEL_HL = MODEL_HL.eval()


def launch():
    pass
