"""V2 parameter groups and stage diagnostics; no V1 output-normalization layers."""

from utils.ai01_learning import StageLearningMonitor


def optimizer_parameter_groups(model, weight_decay):
    """Official V2 excludes logit scale and CPB MLP from weight decay."""
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        exempt = (parameter.ndim <= 1 or "logit_scale" in name
                  or "continuous_position_bias_mlp" in name or name == "mask_token")
        (no_decay if exempt else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


class AI01V2StageLearningMonitor(StageLearningMonitor):
    def __init__(self, backbone, output, logger, log_freq=20):
        groups = {f"stage{index}": list(layer.parameters())
                       for index, layer in enumerate(backbone.encoder.layers, start=1)}
        if len(groups) != 4:
            raise ValueError("V2 diagnostics require four stages")
        groups["stage1"] += list(backbone.embeddings.parameters())
        super().__init__(groups, output, logger, log_freq)
