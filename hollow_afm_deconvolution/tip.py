from typing import Optional, Tuple, Dict, Any

import math

import torch

TIP_TYPES = {}


def register_tip(cls):
    TIP_TYPES[cls.__name__] = cls
    return cls


def create_tip(name: str, *args, **kwargs) -> "Tip":
    if name not in TIP_TYPES:
        raise ValueError(f"Tip {name} does not exist")
    return TIP_TYPES[name](*args, **kwargs)


class Tip:
    def __init__(self, data: torch.Tensor):
        assert data.shape[0] == data.shape[1], "Tip must be square"

        self._data = data
        self._tip_size = data.shape[0]

    def plot(
        self,
        ax,
        *,
        position: Optional[Tuple[float, float, float]] = None,
        plot_kw: Dict[str, Any] = {},
        **kwargs,
    ):
        arange = torch.arange(self._tip_size)
        x, y = torch.meshgrid(arange, arange)

        data = self.data.clone()
        if position is not None:
            tx, ty, tz = position

            # We assume that at_height is set because we're plotting the tip on the
            # surface. Flip it so it faces the surface.
            data = data.max() - data
            data += tz
            x = x.clone() + tx - self._tip_size // 2
            y = y.clone() + ty - self._tip_size // 2

        plot_kw.setdefault('cmap', 'viridis')
        plot_kw.setdefault('rstride', 1)
        plot_kw.setdefault('cstride', 1)
        plot_kw.setdefault('alpha', 0.8)
        plot_kw.setdefault('antialiased', True)
        ax.plot_surface(
            x,
            y,
            data,
            **plot_kw,
        )
        ax.set(**kwargs)

    def add_noise(self, std: float) -> None:
        self.data += torch.normal(0, std, size=self.data.shape)

    def with_noise(self, std: float) -> "Tip":
        tip = self.clone()
        tip.add_noise(std)
        return tip

    def detach(self):
        self._data = self._data.detach().cpu()

    def clone(self) -> "Tip":
        return Tip(self._data.clone())

    @property
    def data(self) -> torch.Tensor:
        return self._data

    @data.setter
    def data(self, data: torch.Tensor):
        self._data = data

    @property
    def requires_grad(self) -> bool:
        return self.data.requires_grad

    @requires_grad.setter
    def requires_grad(self, requires_grad: bool):
        self.data.requires_grad = requires_grad


@register_tip
class PyramidTip(Tip):
    def __init__(
        self, tip_size: int = 33, tip_height: float = 15, tip_size_top: float = 9
    ):
        self._tip_size = tip_size
        self._tip_height = tip_height
        self._tip_size_top = tip_size_top

        data = self._create_pyramid(tip_size, tip_size_top, tip_height)
        super().__init__(data)

    def _create_pyramid(self, size, size_top, height) -> torch.Tensor:
        tip = torch.zeros((size, size), dtype=torch.float32)
        x = tip.shape[0]

        for i in range(x // 2 - size_top // 2):
            for j in range(i, x - i):
                for h in range(i, x - i):
                    tip[j, h] = i / (x // 2 - size_top // 2) * height

        return tip - height

    @property
    def size(self) -> int:
        return self._tip_size


@register_tip
class HollowPyramidTip(PyramidTip):
    def __init__(
        self, tip_size: int = 33, tip_height: float = 15, tip_size_top: float = 9
    ):
        super().__init__(tip_size, tip_height, tip_size_top)

        # Set the points on the tip that are equal to the max to zero to simulate a
        # hollow tip
        for i in range(
            tip_size // 2 - tip_size_top // 2 + 1, tip_size // 2 + tip_size_top // 2
        ):
            for j in range(
                tip_size // 2 - tip_size_top // 2 + 1, tip_size // 2 + tip_size_top // 2
            ):
                self._data[i, j] = torch.min(self._data)


@register_tip
class CommercialPyramidTip(Tip):
    def __init__(self, tip_size: int = 33, tip_height: float = 15):
        self._tip_size = tip_size
        self._tip_height = tip_height

        data = self._create_pyramid(tip_size, tip_height)
        super().__init__(data)

    def _create_pyramid(self, size, height) -> torch.Tensor:
        tip = torch.zeros((size, size), dtype=torch.float32)
        x = tip.shape[0]

        for i in range(x // 2):
            for j in range(i, x - i):
                for h in range(i, x - i):
                    tip[j, h] = i / (x // 2) * height
        tip = tip - torch.max(tip)

        return tip


@register_tip
class CommercialSphereTip(Tip):
    def __init__(
        self, tip_size: int = 33, tip_height: float = 15, sphere_radius: float = 4
    ):
        self._tip_size = tip_size
        self._tip_height = tip_height
        self._sphere_radius = sphere_radius

        data = self._create_tip(tip_size, tip_height, sphere_radius)
        super().__init__(data)

    def _create_tip(self, size, height, sphere_radius) -> torch.Tensor:
        tip = torch.zeros((size, size), dtype=torch.float32)
        x = tip.shape[0]
        y = tip.shape[1]

        for i in range(x // 2 - sphere_radius // 2):
            for j in range(i, x - i):
                for h in range(i, x - i):
                    d = math.sqrt((j - x // 2) ** 2.0 + (h - x // 2) ** 2.0)
                    if d <= (x // 2):
                        tip[j, h] = (
                            (x // 2 - d)
                            * (height - sphere_radius)
                            / (x // 2 - sphere_radius)
                        )
        for i in range(x // 2 - sphere_radius // 2, x // 2 + 1):
            for j in range(i, x - i):
                for h in range(i, x - i):
                    d = math.sqrt((j - x // 2) ** 2.0 + (h - x // 2) ** 2.0)
                    if d <= sphere_radius:
                        tip[j, h] = (
                            height
                            - sphere_radius
                            + math.sqrt(sphere_radius**2.0 - d**2.0)
                        )

        tip = tip - torch.max(tip)

        return tip


@register_tip
class RandomTip(Tip):
    def __init__(self, tip_size: int = 33, tip_height: float = 15):
        self._tip_size = tip_size

        data = torch.rand((tip_size, tip_size), dtype=torch.float32) * tip_height
        super().__init__(data)


def main(subplots: bool = False):
    import matplotlib.pyplot as plt

    tips = [create_tip(tip) for tip in TIP_TYPES]
    tips.append(create_tip("HollowPyramidTip", 20, 10, 5))

    fig_size = 5
    if subplots:
        fig = plt.figure(figsize=(fig_size * len(tips), fig_size))
    for tip in tips:
        if subplots:
            ax = fig.add_subplot(1, len(tips), tips.index(tip) + 1, projection="3d")
        else:
            fig, ax = plt.subplots(
                figsize=(fig_size, fig_size), subplot_kw={"projection": "3d"}
            )
        tip.plot(ax, title=tip.__class__.__name__)

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--no-subplots", action="store_true", help="Don't plot all tips in one figure"
    )

    args = parser.parse_args()

    main(not args.no_subplots)
