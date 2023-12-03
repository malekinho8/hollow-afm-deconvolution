import torch
import torch.nn.functional as F

from tip import Tip
from surface import Surface


def dilation(surface: Surface, tip: Tip) -> Surface:
    surface_data = surface.data.clone()
    tip_data = tip.data.clone()

    surf_xsiz, surf_ysiz = surface_data.shape
    tip_xsiz, tip_ysiz = tip_data.shape
    xc = tip_xsiz // 2
    yc = tip_ysiz // 2

    padded_surface = F.pad(
        surface_data.unsqueeze(0).unsqueeze(0),
        (yc, yc, xc, xc),
        mode="constant",
        value=0,
    )

    # Create a sliding window view of the padded surface
    window_shape = tip_data.shape
    strides = padded_surface.stride()[2:] * 2
    windows = torch.as_strided(
        padded_surface, (surf_xsiz, surf_ysiz) + window_shape, strides
    )

    # Perform the dilation operation
    dilated = torch.amax(windows + tip_data.view(1, 1, tip_xsiz, tip_ysiz), dim=(2, 3))

    return Surface(dilated.squeeze(0).squeeze(0))


def erosion(surface: Surface, tip: Tip) -> Surface:
    surface_data = surface.data.clone()
    tip_data = tip.data.clone()

    surf_xsiz, surf_ysiz = surface_data.shape
    tip_xsiz, tip_ysiz = tip_data.shape
    xc = tip_xsiz // 2
    yc = tip_ysiz // 2

    padded_surface = F.pad(
        surface_data.unsqueeze(0).unsqueeze(0),
        (yc, yc, xc, xc),
        mode="constant",
        value=0,
    )

    # Create a sliding window view of the padded surface
    window_shape = tip_data.shape
    strides = padded_surface.stride()[2:] * 2
    windows = torch.as_strided(
        padded_surface, (surf_xsiz, surf_ysiz) + window_shape, strides
    )

    # Perform the dilation operation
    eroded = torch.amin(windows - tip_data.view(1, 1, tip_xsiz, tip_ysiz), dim=(2, 3))

    return Surface(eroded.squeeze(0).squeeze(0))


def dilation_and_erosion(surface: Surface, tip: Tip) -> Surface:
    return erosion(dilation(surface, tip), tip)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from tip import create_tip
    from surface import create_surface

    pyramid_tip = create_tip("PyramidTip")
    hollow_pyramid_tip = create_tip("HollowPyramidTip")
    # surface = create_surface("SquareSurface")
    # surface = create_surface("WaveSurface")
    # surface = create_surface("TriangleSurface")
    # surface = create_surface("DeltaSurface")
    # surface = create_surface("SquareSurface", 160, 16, 50, 32)
    # surface = create_surface("SpikedSurface")
    surface = create_surface("PyramidSurface")

    # Dilation
    dilated_surface_pyramid = dilation(surface, pyramid_tip)
    dilated_surface_hollow_pyramid = dilation(surface, hollow_pyramid_tip)

    plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(1, 3, 1)
    surface.plot(ax1, title="Original Surface")
    ax2 = plt.subplot(1, 3, 2)
    dilated_surface_pyramid.plot(ax2, title="Dilated (Original Pyramid Tip)")
    ax3 = plt.subplot(1, 3, 3)
    dilated_surface_hollow_pyramid.plot(ax3, title="Dilated (Hollow Pyramid Tip)")

    # Erosion
    eroded_surface_pyramid = erosion(dilated_surface_pyramid, pyramid_tip)
    eroded_surface_hollow_pyramid = erosion(
        dilated_surface_hollow_pyramid, hollow_pyramid_tip
    )

    plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(1, 3, 1)
    surface.plot(ax1, title="Original Surface")
    ax2 = plt.subplot(1, 3, 2)
    eroded_surface_pyramid.plot(ax2, title="Eroded (Original Pyramid Tip)")
    ax3 = plt.subplot(1, 3, 3)
    eroded_surface_hollow_pyramid.plot(ax3, title="Eroded (Hollow Pyramid Tip)")

    # Plot slices
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    ax1 = axs[0, 0]
    dilated_surface_pyramid.plot_slice(ax1, 16, title="Dilated (Original Pyramid Tip)")
    surface.plot_slice(ax1, 16)
    ax2 = axs[1, 0]
    eroded_surface_pyramid.plot_slice(ax2, 16, title="Eroded (Original Pyramid Tip)")
    surface.plot_slice(ax2, 16)
    ax3 = axs[0, 1]
    dilated_surface_hollow_pyramid.plot_slice(ax3, 16, title="Dilated (Hollow Pyramid Tip)")
    surface.plot_slice(ax3, 16)
    ax4 = axs[1, 1]
    eroded_surface_hollow_pyramid.plot_slice(ax4, 16, title="Eroded (Hollow Pyramid Tip)")
    surface.plot_slice(ax4, 16)

    # Plot the tips in the same 3d plot as the surface
    pyramid_tip.data = -pyramid_tip.data + surface._surface_height
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    surface.plot(ax, title="Surface")
    pyramid_tip.plot(ax, title="PyramidTip")


    plt.show()

