"""Pretrain AI01 Swin V2 Tiny MAE on unlabelled 11-Z stacks, offline."""

import logging
import os
from pathlib import Path

os.environ.setdefault("USE_TF", "0")

from utils.ai01_train import main as train_main, parser as train_parser
from .ai01_learning_v2 import AI01V2StageLearningMonitor, optimizer_parameter_groups
from .ai01_model_v2 import AI01V2SwinMAE, TRAINING_FORMAT, ai01_v2_swin_config, checkpoint_model_config
from .ai01_transfer_v2 import export_encoder, load_ai01_v2_encoder


def parser():
    result = train_parser()
    result.description = __doc__
    result.set_defaults(output_dir="output_ai01_v2_ssl")
    result.add_argument("--window_size", type=int, default=8, help="V2 attention window (Tiny default: 8)")
    result.add_argument("--drop_path_rate", type=float, default=0.2)
    result.add_argument("--gradient_checkpointing", action="store_true",
                        help="Recompute encoder/decoder stages during backward to save activation memory")
    return result


def main(args):
    output = Path(args.output_dir)
    if not args.resume and output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new empty V2 output_dir or --resume; never overwrite another SSL run")

    def factory(**settings):
        model = AI01V2SwinMAE(encoder_config=ai01_v2_swin_config(
            image_size=args.image_size, window_size=args.window_size,
            drop_path_rate=args.drop_path_rate,
        ), **settings)
        if args.gradient_checkpointing:
            # Non-reentrant checkpointing supports trainable mask embeddings and
            # shared skip features. HF applies it to both encoder and V2 decoder.
            model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            checkpoint_function = model.backbone.encoder.layers[0]._gradient_checkpointing_func
            for stage in model.decoder.stages:
                stage.gradient_checkpointing = True
                stage._gradient_checkpointing_func = checkpoint_function
        return model

    return train_main(
        args, model_factory=factory, encoder_exporter=export_encoder,
        encoder_loader=load_ai01_v2_encoder, training_format=TRAINING_FORMAT,
        saved_config_reader=checkpoint_model_config, monitor_factory=AI01V2StageLearningMonitor,
        parameter_groups=optimizer_parameter_groups,
    )


if __name__ == "__main__":
    try:
        main(parser().parse_args())
    except Exception:
        logging.getLogger("ai01_ssl").exception("Swin V2 MAE training failed")
        raise
