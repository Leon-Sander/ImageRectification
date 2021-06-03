import math

import torch
import torch.nn.functional as F
from torch.nn import ReplicationPad2d


# USAGE:
# 1) Create an angle map from given backwards map using calc_angles_torch()
# 2) Warp angles using UV file to allow per pixel comparison

def calc_angles_torch(bm: torch.Tensor) -> torch.Tensor:
    """
    Create the warped angle map from existing backward map.
    :param bm: Backward map for given sample with shape N, H, W, C=2
    :return: Returns the warped angle map with shape N, C=2 H, W
    """
    assert bm.dtype == torch.float
    assert len(bm.shape) == 4
    N, H, W, C = bm.shape
    assert H == W and C == 2

    def rotate(data: torch.Tensor, rotation):
        return ((data + math.pi + rotation) % (2 * math.pi)) - math.pi

    # calculate angles to warped x-axis
    eps_x = bm[:, :-1, :-1, :] - bm[:, 1:, :-1, :]
    eps_x = eps_x[..., 0] + 1j * eps_x[..., 1]
    angles_x = torch.angle(eps_x)
    angles_x = rotate(angles_x, -math.pi)

    # calculate angles to warped y-axis
    eps_y = bm[:, :-1, :-1, :] - bm[:, :-1, 1:, :]
    eps_y = eps_y[..., 0] + 1j * eps_y[..., 1]
    angles_y = torch.angle(eps_y)
    angles_y = rotate(angles_y, math.pi / 2)

    angles = torch.stack([angles_y, angles_x], dim=1)  # N, C=2 H, W
    angles = ReplicationPad2d(1)(angles)[:, :, 1:, 1:]
    return angles


def warp_grid_torch(grid: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """
    Warp a regular grid (like angles) using the UV mapping
    :param grid: Regular grid of floats (e.g. angles) with shape N, C=2, H, W
    :param uv: UV map for given sample with shape N, H, W, C=3. Note, the y-coord channel (1) needs to be unmodified (top=1, bottom=0)!
    :return: Returns the warped angle map with shape N, H, W, C=2
    """
    assert grid.dtype == torch.float
    assert len(grid.shape) == 4
    N, C, H, W = grid.shape
    assert H == W and C == 2

    assert uv.dtype == torch.float
    assert len(uv.shape) == 4
    N, H, W, C = uv.shape
    assert H == W and C == 3

    uv.requires_grad_(requires_grad=False)  # UV map (forward map) is considered a constant

    # split uv map channels in mask and data
    bg_mask = uv[..., 0] <= 0.5
    uv_grid = uv[..., 1:]  # N=1, H, W, C=2

    uv_grid[..., 0] = 1 - uv_grid[..., 0]  # invert y-values of uv map (forward map)
    uv_grid[bg_mask] = 5  # out of bounds value for background pixels
    uv_grid = uv_grid * 2 - 1  # adapt grid range required by grid_sample

    warped_grid = F.grid_sample(input=grid, grid=uv_grid, align_corners=True)  # N, C=2, H, W
    warped_grid = warped_grid.transpose(1, 2).transpose(2, 3)  # N, H, W, C=2

    return warped_grid  # N, H, W, C=2
