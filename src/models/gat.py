import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

logger = logging.getLogger(__name__)


class GATEncoder(nn.Module):
    """Graph Attention Network encoder for capturing spectral-temporal relationships."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        heads: int = 4,
        dropout: float = 0.5
    ):
        """Initializes GAT Conv and Projection layers.

        Args:
            in_features: Number of input features per node.
            out_features: Number of output features per head.
            heads: Number of attention heads.
            dropout: Dropout rate inside GATConv.
        """
        super().__init__()
        # GAT convolution from PyTorch Geometric
        self.gat_conv = GATConv(
            in_channels=in_features,
            out_channels=out_features,
            heads=heads,
            concat=True,
            dropout=dropout
        )
        
        # Projection layer to compress multi-head concat outputs back to out_features
        self.proj = nn.Linear(out_features * heads, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms feature maps into graphs and applies GATConv.

        Args:
            x: Feature map tensor of shape (batch, channels, time).

        Returns:
            torch.Tensor: Attention-transformed features of shape (batch, out_features, time).
        """
        batch_size, channels, time_steps = x.shape
        device = x.device

        # 1. Create a fully connected graph topology along the time axis
        adj = torch.ones(time_steps, time_steps, device=device)
        edge_index = adj.nonzero().t().contiguous()  # Shape: (2, time_steps * time_steps)

        # 2. Reshape node features to (batch * time_steps, channels)
        # Transpose to shape (batch, time_steps, channels)
        x_nodes = x.transpose(1, 2).contiguous()
        flat_x = x_nodes.view(batch_size * time_steps, channels)

        # 3. Broadcast and offset edge_index for all batches in parallel
        # Offset shifts node IDs by time_steps for each batch index i
        offsets = torch.arange(batch_size, device=device).view(batch_size, 1, 1) * time_steps
        
        # Add offset to edge_index across batch dimension
        batch_edge_index = (edge_index.unsqueeze(0) + offsets).transpose(0, 1).contiguous()
        flat_edge_index = batch_edge_index.view(2, -1)  # Shape: (2, batch * time_steps * time_steps)

        # 4. GAT Convolution
        out = self.gat_conv(flat_x, flat_edge_index)
        out = F.elu(out)
        out = self.proj(out)

        # 5. Reshape and transpose back to (batch, out_features, time)
        return out.view(batch_size, time_steps, -1).transpose(1, 2).contiguous()
