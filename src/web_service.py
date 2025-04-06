from src.models.high_level_model import HighLevelModel


def load_models():
    MODEL_HL = HighLevelModel.load_from_checkpoint(
        "models/high-level-model.ckpt",
        map_location="cpu",
    )
    MODEL_HL = MODEL_HL.eval()


def launch():
    pass
