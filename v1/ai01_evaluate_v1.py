"""Evaluate AI01 V1 reconstruction. Run: python -m v1.ai01_evaluate_v1 ..."""

import os

os.environ.setdefault("USE_TF", "0")

from utils.ai01_evaluate import main as evaluate_main, parser as evaluate_parser
from .ai01_model_v1 import AI01SwinMAE, TRAINING_FORMAT, checkpoint_model_config


def parser():
    result = evaluate_parser()
    result.description = __doc__
    return result


def main(args):
    return evaluate_main(args, model_factory=AI01SwinMAE,
                         training_format=TRAINING_FORMAT, saved_config_reader=checkpoint_model_config)


if __name__ == "__main__":
    main(parser().parse_args())
