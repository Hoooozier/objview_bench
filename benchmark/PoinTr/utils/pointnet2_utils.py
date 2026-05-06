import torch

try:
    from pytorch3d.ops import sample_farthest_points as _sample_farthest_points
except ModuleNotFoundError:
    _sample_farthest_points = None


def _fps_pure_torch(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Args:
        xyz: [B, N, 3]
        npoint: int
    Returns:
        indices: [B, npoint] (int64)
    """
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.full((B, N), 1e10, device=device, dtype=xyz.dtype)
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    return centroids


def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Args:
        xyz: [B, N, 3]
        npoint: int
    Returns:
        indices: [B, npoint] (int64)
    """
    if _sample_farthest_points is not None:
        _, idx = _sample_farthest_points(
            xyz,
            K=npoint,
            random_start_point=False,
        )
        return idx.to(dtype=torch.long)
    return _fps_pure_torch(xyz, npoint)


def gather_operation(features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Args:
        features: [B, C, N]
        idx: [B, S]
    Returns:
        gathered: [B, C, S]
    """
    if features.ndim != 3:
        raise ValueError(f"Expected features [B, C, N], got {features.shape}")
    if idx.ndim != 2:
        raise ValueError(f"Expected idx [B, S], got {idx.shape}")
    B, C, _ = features.shape
    idx_expanded = idx.unsqueeze(1).expand(B, C, idx.shape[1])
    return torch.gather(features, 2, idx_expanded)
