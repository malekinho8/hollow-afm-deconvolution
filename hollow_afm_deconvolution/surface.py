from typing import Optional, Dict, Any

import torch

from tip import Tip

def unravel_index(index, shape):
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))

SURFACE_TYPES = {}

def register_surface(cls):
    SURFACE_TYPES[cls.__name__] = cls
    return cls

def create_surface(name: str, *args, **kwargs) -> "Surface":
    if name not in SURFACE_TYPES:
        raise ValueError(f"Surface {name} does not exist")
    return SURFACE_TYPES[name](*args, **kwargs)

class Surface:
    def __init__(self, data: torch.Tensor):
        assert data.shape[0] == data.shape[1], "Surface must be square"

        self._data = data
        self._surface_size = data.shape[0]

    def plot(self, ax, *, tip: Optional[Tip] = None, colorbar: bool = True, **kwargs):
        if hasattr(ax, "plot_surface"):
            arange = torch.arange(self._surface_size)
            x, y = torch.meshgrid(arange, arange)

            ax.plot_surface(
                x,
                y,
                self.data,
                cmap="Greys",
                rstride=1,
                cstride=1,
                alpha=0.8,
                antialiased=True,
            )

            if tip is not None:
                # Find the max point at the area within half the surface size in the center
                # of the surface.
                x = y = self._surface_size // 2
                delta = self._surface_size // 4
                idx = self.data[x - delta : x + delta, y - delta : y + delta].argmax()
                x, y = unravel_index(idx, (delta * 2, delta * 2))
                x += self._surface_size // 2 - delta
                y += self._surface_size // 2 - delta
                z = self.data[x, y]
                tip.plot(ax, position=(x, y, z))
        else:
            assert tip is None
            im = ax.imshow(self.data, cmap="Greys", interpolation="nearest")
            if colorbar:
                ax.get_figure().colorbar(im)

        ax.set(**kwargs)

    def plot_slice(self, ax, plot_kw: Dict[str, Any] = {}, **kwargs):
        slice = torch.diagonal(self.data)

        ax.plot(range(len(slice)), slice, **plot_kw)
        ax.set(**kwargs)

    def add_noise(self, std: float) -> None:
        self.data += torch.normal(0, std, size=self.data.shape)

    def with_noise(self, std: float) -> "Surface":
        surface = self.clone()
        surface.add_noise(std)
        return surface

    def clone(self) -> "Surface":
        return Surface(self.data.clone())

    def detach(self):
        self._data = self._data.detach().cpu()

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, data: torch.Tensor):
        self._data = data

    @property
    def size(self) -> int:
        return self._surface_size


@register_surface
class RectangleSurface(Surface):
    def __init__(
        self,
        surface_size: int = 160,
        feature_size: int = 32,
        surface_height: float = 50,
        periodicity: int = 64,
    ):
        self._surface_size = surface_size
        self._feature_size = feature_size
        self._surface_height = surface_height
        self._periodicity = periodicity

        data = self._create_rectangle(
            surface_size, feature_size, surface_height, periodicity
        )
        super().__init__(data)

    def _create_rectangle(self, size, feature_size, height, T) -> torch.Tensor:
        surface = torch.zeros((size, size), dtype=torch.float32)

        for _ in range(0, size, T):
            for j in range(0, size, T):
                surface[:, j : j + feature_size] = height

        return surface

@register_surface
class SquareSurface(Surface):
    def __init__(
        self,
        surface_size: int = 160,
        feature_size: int = 32,
        surface_height: float = 50,
        periodicity: int = 64,
    ):
        self._surface_size = surface_size
        self._feature_size = feature_size
        self._surface_height = surface_height
        self._periodicity = periodicity

        data = self._create_rectangle(
            surface_size, feature_size, surface_height, periodicity
        )
        super().__init__(data)

    def _create_pyramid(self, size, feature_size, height, T) -> torch.Tensor:
        surface = torch.zeros((size, size), dtype=torch.float32)

        for x in range(0, size, T):
            for y in range(0, size, T):
                for i in range(feature_size // 2):
                    for j in range(i, feature_size - i):
                        for h in range(i, feature_size - i):
                            surface[j + x, h + y] = i / (feature_size // 2) * height

    def _create_rectangle(self, size, feature_size, height, T) -> torch.Tensor:
        surface = torch.zeros((size, size), dtype=torch.float32)

        for x in range(0, size, T):
            for y in range(0, size, T):
                surface[x:x+feature_size, y:y+feature_size] = height

        return surface


# @register_surface
class WaveSurface(Surface):
    def __init__(
        self,
        surface_size: int = 160,
        feature_size: int = 32,
        surface_height: float = 50,
        periodicity: int = 64,
    ):
        self._surface_size = surface_size
        self._feature_size = feature_size
        self._surface_height = surface_height
        self._periodicity = periodicity

        data = self._create_wave(
            surface_size, feature_size, surface_height, periodicity
        )
        super().__init__(data)

    def _create_wave(self, size, feature_size, height, T) -> torch.Tensor:
        surface = torch.zeros((size, size), dtype=torch.float32)

        for i in range(0, size, T):
            for j in range(0, size, T):
                x = torch.linspace(-1, 1, feature_size)
                y = torch.sqrt(1 - x**2)
                surface[i : i + feature_size, j : j + feature_size] = height * y.view(
                    -1, 1
                )

        return surface

# @register_surface
class TriangleSurface(Surface):
    def __init__(
        self,
        surface_size: int = 160,
        feature_size: int = 32,
        surface_height: float = 50,
        periodicity: int = 64,
    ):
        self._surface_size = surface_size
        self._feature_size = feature_size
        self._surface_height = surface_height
        self._periodicity = periodicity

        data = self._create_triangle(
            surface_size, feature_size, surface_height, periodicity
        )
        super().__init__(data)

    def _create_triangle(self, size, feature_size, height, T) -> torch.Tensor:
        surface = torch.zeros((size, size), dtype=torch.float32)

        for i in range(0, size, T):
            for j in range(0, size, T):
                x = torch.linspace(-1, 1, feature_size)
                y = 1 - torch.abs(x)
                surface[i: i + feature_size, j : j + feature_size] = height * y.view(
                    -1, 1
                )

        return surface

# @register_surface
class DeltaSurface(Surface):
    """Delta function"""
    def __init__(
        self,
        surface_size: int = 160,
        feature_size: int = 32,
        surface_height: float = 50,
        periodicity: int = 64,
    ):
        self._surface_size = surface_size
        self._feature_size = feature_size
        self._surface_height = surface_height
        self._periodicity = periodicity

        data = self._create_delta(
            surface_size, feature_size, surface_height, periodicity
        )
        super().__init__(data)

    def _create_delta(self, size, feature_size, height, T) -> torch.Tensor:
        surface = torch.zeros((size, size), dtype=torch.float32)

        for i in range(0, size, T):
            for j in range(0, size, T):
                surface[i : i + feature_size, j : j + feature_size] = height

        return surface

# @register_surface
class SpikedSurface(Surface):
    def __init__(self,
                 surface_size: int = 160,
                 feature_size: int = 32,
                 surface_height: float = 50,
                 periodicity: int = 64,
                 ):
        self._surface_size = surface_size
        self._feature_size = feature_size
        self._surface_height = surface_height
        self._periodicity = periodicity

        data = self._create_spike(
            surface_size, feature_size, surface_height, periodicity
        )
        super().__init__(data)

    def _create_spike(self, size, feature_size, height, T) -> torch.Tensor:
        surface = torch.zeros((size, size), dtype=torch.float32)

        surface[::feature_size, ::feature_size] = height

        return surface

@register_surface
class PyramidSurface(Surface):
    def __init__(
        self,
        surface_size: int = 160,
        feature_size: int = 32,
        surface_height: float = 50,
        periodicity: int = 64,
    ):
        self._surface_size = surface_size
        self._feature_size = feature_size
        self._surface_height = surface_height
        self._periodicity = periodicity

        data = self._create_pyramid(
            surface_size, feature_size, surface_height, periodicity
        )
        super().__init__(data)

    def _create_pyramid(self, size, feature_size, height, T) -> torch.Tensor:
        surface = torch.zeros((size, size), dtype=torch.float32)

        for x in range(0, size, T):
            for y in range(0, size, T):
                for i in range(feature_size // 2):
                    for j in range(i, feature_size - i):
                        for h in range(i, feature_size - i):
                            if j + x < size and h + y < size:
                                surface[j + x, h + y] = i / (feature_size // 2) * height

        return surface

@register_surface
class ParaboloidSurface(Surface):
    def __init__(
        self,
        surface_size: int = 160,
        feature_size: int = 32,
        surface_height: float = 50,
        periodicity: int = 64,
    ):
        self._surface_size = surface_size
        self._feature_size = feature_size
        self._surface_height = surface_height
        self._periodicity = periodicity

        data = self._create_sine(
            surface_size, feature_size, surface_height, periodicity
        )
        super().__init__(data)

    def _create_sine(self, size, feature_size, height, T) -> torch.Tensor:
        surface = torch.zeros((size, size), dtype=torch.float32)

        for x in range(0, size, T):
            for y in range(0, size, T):
                for i in range(feature_size):
                    for j in range(i, feature_size - i):
                        for h in range(i, feature_size - i):
                            if j + x < size and h + y < size:
                                surface[j + x, h + y] = torch.sin(torch.tensor(i / (feature_size) * torch.pi)) * height

        return surface

@register_surface
class SineSurface(Surface):
    def __init__(
        self,
        surface_size: int = 160,
        feature_size: int = 32,
        surface_height: float = 50,
        periodicity: int = 64,
    ):
        self._surface_size = surface_size
        self._feature_size = feature_size
        self._surface_height = surface_height
        self._periodicity = periodicity

        data = self._create_sine(
            surface_size, feature_size, surface_height, periodicity
        )
        super().__init__(data)

    def _create_sine(self, size, feature_size, height, T) -> torch.Tensor:
        x = torch.linspace(-feature_size // 2, feature_size // 2, size)
        y = torch.linspace(-feature_size // 2, feature_size // 2, size)
        x, y = torch.meshgrid(x, y)
        z = torch.sin(torch.sqrt(x**2 + y**2)) * height // 2 + height // 2
        return z
        surface = torch.zeros((size, size), dtype=torch.float32)

        for x in range(0, size, T):
            for y in range(0, size, T):
                for i in range(feature_size):
                    for j in range(i, feature_size - i):
                        x = torch.linspace(-1, 1, feature_size)
                        y = torch.sqrt(1 - x**2)
                        surface[i : i + feature_size, j : j + feature_size] = height * y.view(
                            -1, 1
                        )

        return surface

@register_surface
class ImageSurface(Surface):
    def __init__(self, filename: str = "../dog.jpg", surface_size: int = 160, surface_height: float = 50, elevation: float = 80, azimuth: float = 0, roll: float = 0):
        self._elevation = elevation
        self._azimuth = azimuth
        self._roll = roll

        data = self._create_image(filename, surface_size, surface_height)
        super().__init__(data)

    def _create_image(self, filename: str, size: int, height: float) -> torch.Tensor:
        import os
        import cv2

        assert os.path.exists(filename), f"Image file does not exist: {filename}"
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (size, size))
        border = [size // 4] * 4
        img = cv2.copyMakeBorder(img, *border, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img = torch.from_numpy(img).float()
        img = img / 255.0
        img = height - img * height
        return img

    def plot(self, ax, *, tip: Optional[Tip] = None, view_init: bool = True, **kwargs):
        super().plot(ax, tip=tip, **kwargs)
        if view_init:
            if not hasattr(ax, "plot_surface") or tip is not None:
                print("Ignoring view_init for 2d plot and/or tip is not None")
                return
            ax.view_init(elev=self._elevation, azim=self._azimuth, roll=self._roll)

def main(subplots: bool = True):
    import matplotlib.pyplot as plt

    surfaces = [create_surface(surface) for surface in SURFACE_TYPES]
    surfaces.append(create_surface("ImageSurface", "../dog.jpg", elevation=45, azimuth=45))

    fig_size = 5
    if subplots:
        fig = plt.figure(figsize=(fig_size * len(surfaces), fig_size))
    for surface in surfaces:
        if subplots:
            ax = fig.add_subplot(1, len(surfaces), surfaces.index(surface) + 1, projection="3d")
        else:
            fig, ax = plt.subplots(figsize=(fig_size, fig_size), subplot_kw={"projection": "3d"})
        surface.plot(ax, title=surface.__class__.__name__, xlabel="x", ylabel="y", zlabel="z")

    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--no-subplots", action="store_true", help="Don't plot all surfaces in one figure")

    args = parser.parse_args()

    main(not args.no_subplots)