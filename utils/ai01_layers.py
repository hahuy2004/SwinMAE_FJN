"""Architecture-neutral layers shared by both AI01 reconstruction decoders."""

from torch import nn


class PatchExpansion(nn.Module):
    """Double spatial resolution and halve channels."""

    def __init__(self, dim):
        super().__init__()
        self.expand = nn.Linear(dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(dim // 2)

    def forward(self, x):
        batch, height, width, channels = x.shape
        x = self.expand(x).reshape(batch, height, width, 2, 2, channels // 2)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(batch, 2 * height, 2 * width, channels // 2)
        return self.norm(x)
