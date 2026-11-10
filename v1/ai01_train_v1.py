"""Train AI01 Swin V1 Tiny MAE. Run: python -m v1.ai01_train_v1 ..."""

import logging
import os

os.environ.setdefault("USE_TF", "0")

from utils.ai01_train import main as train_main, parser as train_parser
from .ai01_learning_v1 import StageLearningMonitor, optimizer_parameter_groups
from .ai01_model_v1 import AI01SwinMAE, TRAINING_FORMAT, checkpoint_model_config
from .ai01_transfer_v1 import export_encoder, load_ai01_encoder


def parser():
    result = train_parser()
    result.description = __doc__
    return result


def main(args):
    return train_main(
        args, model_factory=AI01SwinMAE, encoder_exporter=export_encoder,
        encoder_loader=load_ai01_encoder, training_format=TRAINING_FORMAT,
        saved_config_reader=checkpoint_model_config, monitor_factory=StageLearningMonitor,
        parameter_groups=optimizer_parameter_groups,
    )


if __name__ == "__main__":
    try:
        main(parser().parse_args())
    except Exception:
        logging.getLogger("ai01_ssl").exception("Swin V1 MAE training failed")
        raise
