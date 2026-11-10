"""Swin V1 optimizer grouping and four-stage diagnostic parameter selection."""

from utils.ai01_learning import StageLearningMonitor as BaseStageLearningMonitor


def stage_parameters(backbone):
    groups = {}
    for index, layer in enumerate(backbone.encoder.layers, start=1):
        parameters = list(layer.parameters())
        parameters += list(backbone.hidden_states_norms[f"stage{index}"].parameters())
        if index == 1:
            parameters += list(backbone.embeddings.parameters())
        groups[f"stage{index}"] = parameters
    if len(groups) != 4:
        raise ValueError("Learning diagnostics require four Swin stages")
    return groups


class StageLearningMonitor(BaseStageLearningMonitor):
    def __init__(self, backbone, output, logger, log_freq=20):
        super().__init__(stage_parameters(backbone), output, logger, log_freq)


def optimizer_parameter_groups(model, weight_decay):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (no_decay if parameter.ndim <= 1 else decay).append(parameter)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]
