import copy
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
) -> Tuple[Tip, List[float]]:
    """
    Args:
        image: The image that was scanned using the unknown tip
        surface: The ground truth surface
    """
    tip = copy.deepcopy(tip)
    tip.data.requires_grad = True

    losses = []
    optimizer = torch.optim.Adam([tip.data], lr=0.1)

    pbar = tqdm(range(epochs))
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


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    tip = create_tip("PyramidTip")
    # tip = create_tip("HollowPyramidTip")
    # tip = create_tip("BigHollowPyramidTip")
    random_tip = create_tip("RandomTip")
    # surface = create_surface("SquareSurface")
    # surface = create_surface("SquareSurface", 160, 16, 50, 32)
    # surface = create_surface("WaveSurface")
    # surface = create_surface("TriangleSurface")
    # surface = create_surface("DeltaSurface")
    # surface = create_surface("DeltaSurface", 160, 4, 50, 32)
    # surface = create_surface("SpikedSurface")
    surface = create_surface("PyramidSurface")

    # Perform dilation and erosion on the square surface with different tips
    image_surface = dilation_and_erosion(surface, tip)

    epochs = 500
    random_tip_before = random_tip.data.clone()
    reconstructed_tip, losses = optimize(
        image_surface,
        surface,
        random_tip,
        dilation_and_erosion,
        epochs=epochs,
    )

    # Plot the loss curve
    # plt.figure(figsize=(4, 4))
    # plt.plot(losses)
    # plt.title("Loss Curve")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")

    fig_size = 5
    fig = plt.figure(figsize=(fig_size * 3, fig_size))

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    tip.plot(ax1, title="Original Tip")

    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    random_tip.plot(ax2, title="Random Tip")

    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    reconstructed_tip.plot(ax3, title="Reconstructed Tip")

    plt.show()
