import torch
import numpy as np

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

    def plot(self, ax, **kwargs):
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
        else:
            im = ax.imshow(self.data, cmap="Greys", interpolation="nearest")
            ax.get_figure().colorbar(im)
        ax.set(**kwargs)

    def plot_slice(self, ax, size, **kwargs):
        slice = self.data[size, :]

        ax.plot(range(len(slice)), slice)
        ax.set(**kwargs)

    def astorch(self) -> torch.Tensor:
        return self.data

    def asnumpy(self) -> np.ndarray:
        return self.data.numpy()

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

        data = self._create_square(
            surface_size, feature_size, surface_height, periodicity
        )
        super().__init__(data)

    def _create_square(self, size, feature_size, height, T) -> torch.Tensor:
        surface = torch.zeros((size, size), dtype=torch.float32)

        for _ in range(0, size, T):
            for j in range(0, size, T):
                surface[:, j : j + feature_size] = height

        return surface


@register_surface
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

@register_surface
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

@register_surface
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

@register_surface
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
                            surface[j + x, h + y] = i / (feature_size // 2) * height
                    # surface[i : i + feature_size, j : j + feature_size] = height

        return surface

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    surfaces = [create_surface(surface) for surface in SURFACE_TYPES]
    surfaces.append(create_surface("SquareSurface", 160, 16, 50, 32))
    surfaces.append(create_surface("DeltaSurface", 160, 4, 50, 32))

    fig_size = 5
    fig = plt.figure(figsize=(fig_size * len(surfaces), fig_size))
    for surface in surfaces:
        ax = fig.add_subplot(
            1, len(surfaces), surfaces.index(surface) + 1, projection="3d"
        )
        surface.plot(ax, title=surface.__class__.__name__)

    plt.show()
