import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return x + self.net(x)


class PointCloudNet(nn.Module):
    """BENBV-Net architecture adapted without changing its tensor contract.

    Inputs:
        P: [B, 4096, 6] partial point cloud xyz + normal
        S: [B, 20, 6] candidate target xyz + view direction
        C: [B, 21, 1] 20 density values + step index
    Output:
        score: [B, 20, 1]
    """

    def __init__(self, dropout_rate=0.1):
        super().__init__()
        self.pos_feat = nn.Sequential(
            self._make_conv_block(3, 64),
            self._make_conv_block(64, 128),
            self._make_conv_block(128, 256),
            self._make_conv_block(256, 512),
            self._make_conv_block(512, 1024),
        )
        self.normal_feat = nn.Sequential(
            self._make_conv_block(3, 64),
            self._make_conv_block(64, 128),
            self._make_conv_block(128, 256),
            self._make_conv_block(256, 512),
        )

        self.feat_pos = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.feat_normal = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.feat_density = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.feat_view = nn.Sequential(
            nn.Linear(1, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.dv_fusion = nn.Sequential(nn.Linear(256, 128), nn.LayerNorm(128), nn.ReLU())
        self.global_fusion = nn.Sequential(nn.Linear(1536, 1536), nn.LayerNorm(1536), nn.ReLU())

        self.main_branch = nn.Sequential(
            self._make_fc_block(1792, 512),
            ResBlock(512),
            self._make_fc_block(512, 256),
            ResBlock(256),
            self._make_fc_block(256, 128),
            ResBlock(128),
        )

        self.pre_attention = nn.Sequential(nn.Linear(256, 256), nn.LayerNorm(256), nn.ReLU())
        self.self_attention = nn.MultiheadAttention(256, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(256)

        self.score_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _make_conv_block(in_channels, out_channels):
        return nn.Sequential(nn.Conv1d(in_channels, out_channels, 1), nn.BatchNorm1d(out_channels), nn.ReLU())

    @staticmethod
    def _make_fc_block(in_features, out_features):
        return nn.Sequential(nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features), nn.ReLU())

    def forward(self, P, S, C):
        batch_size = P.size(0)

        positions = P[:, :, :3].transpose(2, 1)
        normals = P[:, :, 3:].transpose(2, 1)
        pos_global = self.pos_feat(positions).max(dim=2)[0]
        normal_global = self.normal_feat(normals).max(dim=2)[0]

        density = C[:, :20, 0:1]
        view_order = C[:, 20:, 0:1]
        density_feat = self.feat_density(density.reshape(-1, 1)).view(batch_size, 20, -1)
        view_feat = self.feat_view(view_order.repeat(1, 20, 1).reshape(-1, 1)).view(batch_size, 20, -1)

        global_feat = torch.cat([pos_global, normal_global], dim=1)
        global_feat = self.global_fusion(global_feat).unsqueeze(1).expand(-1, 20, -1)

        pos = S[:, :, :3].reshape(-1, 3)
        normal = S[:, :, 3:].reshape(-1, 3)
        pos_feat = self.feat_pos(pos)
        normal_feat = self.feat_normal(normal)
        candidate_feat = torch.cat([pos_feat, normal_feat], dim=1).view(batch_size, 20, -1)

        combined = torch.cat([global_feat, candidate_feat], dim=2)
        features = self.main_branch(combined.reshape(-1, combined.size(2))).view(batch_size, 20, -1)

        dv_feat = self.dv_fusion(torch.cat([density_feat, view_feat], dim=2))
        pre_attention = self.pre_attention(torch.cat([features, dv_feat], dim=2))
        attended_features, _ = self.self_attention(pre_attention, pre_attention, pre_attention)
        features = self.norm1(attended_features).reshape(-1, 256)
        score = self.score_head(features).view(batch_size, 20, 1)
        return score
