import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SCVPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, residual=True):
        super().__init__()
        self.conv1 = self._make_conv(in_channels, out_channels)
        self.conv2 = self._make_conv(out_channels, out_channels)
        self.residual = residual
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    @staticmethod
    def _make_conv(in_channels, out_channels):
        layers = []
        for layer_idx in range(4):
            layers.append(CNNBlock(in_channels if layer_idx == 0 else out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        if self.residual:
            x2 = x2 + x1
        return self.relu(x2)


class ViewStatePositionEmbedding(nn.Module):
    """Bind each view state with its xyz position, then tile it as 3D context."""

    def __init__(self, view_positions, context_channels=4, hidden_dim=64, dropout=0.1):
        super().__init__()
        if view_positions.ndim != 2 or view_positions.shape[1] != 3:
            raise ValueError(f"view_positions must be (V, 3), got {tuple(view_positions.shape)}")

        self.num_views = int(view_positions.shape[0])
        self.context_channels = int(context_channels)
        self.register_buffer("view_positions", view_positions.float())

        self.view_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, self.context_channels),
        )

    def forward(self, view_state, spatial_size):
        if view_state.dim() == 3 and view_state.shape[1] == 1:
            view_state = view_state.squeeze(1)
        if view_state.dim() != 2:
            raise ValueError(f"view_state must be (B, V) or (B, 1, V), got {tuple(view_state.shape)}")

        batch_size, num_views = view_state.shape
        if num_views != self.num_views:
            raise ValueError(f"Expected {self.num_views} view states, got {num_views}")

        positions = self.view_positions.unsqueeze(0).expand(batch_size, -1, -1)
        view_input = torch.cat([view_state.float().unsqueeze(-1), positions], dim=-1)
        tokens = self.view_mlp(view_input).transpose(1, 2).contiguous()

        num_voxels = int(spatial_size) ** 3
        if num_voxels % self.num_views == 0:
            context = tokens.repeat_interleave(num_voxels // self.num_views, dim=2)
        else:
            context = F.interpolate(tokens, size=num_voxels, mode="linear", align_corners=False)
        return context.view(batch_size, self.context_channels, spatial_size, spatial_size, spatial_size)


class MASCVPPositionNet(nn.Module):
    """MA-SCVP for packed 64^3 grids with 128-view state+position context."""

    def __init__(
        self,
        view_positions,
        grid_size=64,
        output_views=128,
        context_channels=4,
        state_hidden_dim=64,
        residual=True,
        dropout=0.35,
    ):
        super().__init__()
        if grid_size % 16 != 0:
            raise ValueError(f"grid_size must be divisible by 16, got {grid_size}")

        self.grid_size = int(grid_size)
        self.output_views = int(output_views)
        self.context_channels = int(context_channels)
        self.dropout = float(dropout)

        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(2, 2)
        self.state_position_embedding = ViewStatePositionEmbedding(
            view_positions=view_positions,
            context_channels=context_channels,
            hidden_dim=state_hidden_dim,
            dropout=0.1,
        )

        first_channels = 32 + context_channels
        self.block1 = SCVPBlock(first_channels, 64, residual=residual)
        self.down1 = nn.Conv3d(64, 64, 2, 2)

        self.block2 = SCVPBlock(64, 128, residual=residual)
        self.down2 = nn.Conv3d(128, 128, 2, 2)

        self.block3 = SCVPBlock(128, 256, residual=residual)
        self.down3 = nn.Conv3d(256, 256, 2, 2)

        final_spatial = self.grid_size // 16
        final_channels = 480 + context_channels
        self.fc1 = nn.Linear(final_channels * final_spatial * final_spatial * final_spatial, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, self.output_views)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, grid, view_state):
        x1 = self.pool(self.conv1(grid))
        context = self.state_position_embedding(view_state, spatial_size=x1.shape[-1])
        x1 = torch.cat([x1, context], dim=1)

        x2 = self.down1(self.block1(x1))
        x3 = self.down2(self.block2(x2))
        x4 = self.down3(self.block3(x3))

        x2 = torch.cat([x2, self.pool(x1)], dim=1)
        x3 = torch.cat([x3, self.pool(x2)], dim=1)
        x4 = torch.cat([x4, self.pool(x3)], dim=1)

        x4 = torch.flatten(x4, 1)
        x4 = F.dropout(self.relu(self.fc1(x4)), p=self.dropout, training=self.training)
        x4 = F.dropout(self.relu(self.fc2(x4)), p=self.dropout, training=self.training)
        x4 = F.dropout(self.relu(self.fc3(x4)), p=self.dropout, training=self.training)
        x4 = F.dropout(self.relu(self.fc4(x4)), p=self.dropout, training=self.training)
        return self.fc5(x4)


if __name__ == "__main__":
    positions = torch.randn(128, 3)
    model = MASCVPPositionNet(positions, grid_size=64, output_views=128)
    grid = torch.randn(2, 1, 64, 64, 64)
    state = torch.zeros(2, 128)
    print(model(grid, state).shape)
