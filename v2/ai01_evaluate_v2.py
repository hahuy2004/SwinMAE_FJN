"""Evaluate AI01 V2 masked reconstruction from a FULL V2 MAE checkpoint."""

import os

os.environ.setdefault("USE_TF", "0")

from utils.ai01_evaluate import main as evaluate_main, parser as evaluate_parser
from .ai01_model_v2 import AI01V2SwinMAE, TRAINING_FORMAT, checkpoint_model_config


def parser():
    result = evaluate_parser()
    result.description = __doc__
    return result


def main(args):
    return evaluate_main(args, model_factory=AI01V2SwinMAE,
                         training_format=TRAINING_FORMAT, saved_config_reader=checkpoint_model_config)


if __name__ == "__main__":
    main(parser().parse_args())
