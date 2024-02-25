from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from tqdm.rich import tqdm

from tip import Tip, create_tip
from surface import Surface, create_surface
from utils import dilation_and_erosion

ReconstructionFunction = Callable[[Surface, Tip], Surface]


def loss_fn(
    surface_to_reconstruct: Surface,
    surface: Surface,
    tip: Tip,
    fn: ReconstructionFunction,
) -> torch.Tensor:
    # mse
    return F.mse_loss(surface_to_reconstruct.data, fn(surface, tip).data)


def optimize(
    image: Surface,
    surface: Surface,
    tip: Tip,
    fn: ReconstructionFunction,
    *,
    epochs: int = 1000,
    device: torch.device = "cpu",
    _tqdm = tqdm,
) -> Tuple[Tip, List[float]]:
    """
    Args:
        image: The image that was scanned using the unknown tip
        surface: The ground truth surface
    """
    tip = tip.clone()
    tip.data = tip.data.to(device)
    tip.data.requires_grad = True

    surface = surface.clone()
    surface.data = surface.data.to(device)

    image = image.clone()
    image.data = image.data.to(device)

    losses = []
    optimizer = torch.optim.Adam([tip.data], lr=0.1)

    pbar = _tqdm(range(epochs))
    for epoch in pbar:
        optimizer.zero_grad()

        loss = loss_fn(image, surface, tip, fn)
        loss.backward()

        optimizer.step()

        description = f"Epoch: {epoch}, Loss: {loss.item():.4f}"
        pbar.set_description(description)
        if epoch % (epochs // 10) == 0:
            print(description)

        losses.append(loss.item())

    tip.detach()
    return tip, losses

def batched_loss_fn(
    tip: Tip,
    surfaces: List[Surface],
    surface_prior: Surface,
    fn: ReconstructionFunction,
) -> torch.Tensor:
    # mse
    reconstructions = []
    for i in range(len(surfaces)):
        reconstructions.append(fn(surfaces[i], tip).data)
    reconstructions = torch.stack(reconstructions)

    return F.mse_loss(surface_prior.data, reconstructions).sum()

def batched_optimize(
    tip_prior: Tip,
    surfaces: List[Surface],
    surface_prior: Surface,
    fn: ReconstructionFunction,
    *,
    epochs: int = 1000,
    device: torch.device = "cpu",
    _tqdm = tqdm,
) -> Tuple[Tip, Surface, List[float]]:
    tip = tip_prior.clone()
    tip.data = tip.data.to(device)
    tip.data.requires_grad = True

    for i in range(len(surfaces)):
        surfaces[i] = surfaces[i].clone()
        surfaces[i].data = surfaces[i].data.to(device)

    surface = surface_prior.clone()
    surface.data = surface.data.to(device)
    surface.data.requires_grad = True

    losses = []
    optimizer = torch.optim.Adam([tip.data], lr=0.1)

    pbar = _tqdm(range(epochs))
    for epoch in pbar:
        optimizer.zero_grad()

        loss = batched_loss_fn(tip, surfaces, surface, fn)
        loss.backward()

        optimizer.step()

        description = f"Epoch: {epoch}, Loss: {loss.item():.4f}"
        pbar.set_description(description)
        if epoch % (epochs // 10) == 0:
            print(description)

        losses.append(loss.item())

    tip.detach()
    surface.detach()
    return tip, surface, losses


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    # tip = create_tip("PyramidTip")
    tip = create_tip("HollowPyramidTip", 20, 10, 5)
    # tip = create_tip("BigHollowPyramidTip")
    random_tip = create_tip("RandomTip", 20)
    # random_tip = create_tip("HollowPyramidTip")
    tip.add_noise(0.5)

    # surface = create_surface("SquareSurface")
    # surface = create_surface("SquareSurface", 160, 16, 50, 32)
    # surface = create_surface("WaveSurface")
    # surface = create_surface("TriangleSurface")
    # surface = create_surface("DeltaSurface")
    # surface = create_surface("DeltaSurface", 160, 4, 50, 32)
    # surface = create_surface("SpikedSurface")
    # surface = create_surface("PyramidSurface")
    surface = create_surface("ImageSurface")
    # surface.add_noise(2)

    # Perform dilation and erosion on the square surface with different tips
    image_surface = dilation_and_erosion(surface, tip)

    epochs = 1000
    random_tip_before = random_tip.data.clone()
    reconstructed_tip, losses = optimize(
        image_surface,
        surface,
        random_tip,
        dilation_and_erosion,
        epochs=epochs,
    )

    # Plot the loss curve
    plt.figure(figsize=(4, 4))
    plt.plot(losses)
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    fig_size = 5
    num_figures = 4
    fig = plt.figure(figsize=(fig_size * num_figures, fig_size))

    ax1 = fig.add_subplot(1, num_figures, 1, projection="3d")
    tip.plot(ax1, title="Original Tip")

    ax2 = fig.add_subplot(1, num_figures, 2, projection="3d")
    random_tip.plot(ax2, title="Random Tip")

    ax3 = fig.add_subplot(1, num_figures, 3, projection="3d")
    reconstructed_tip.plot(ax3, title="Reconstructed Tip")

    ax4 = fig.add_subplot(1, num_figures, 4, projection="3d")
    surface.plot(ax4, title="Surface")

    plt.show()
