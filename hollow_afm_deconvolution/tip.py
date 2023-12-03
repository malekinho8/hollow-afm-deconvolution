import torch
import numpy as np

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

    def plot(self, ax, **kwargs):
        arange = torch.arange(self._tip_size)
        x, y = torch.meshgrid(arange, arange)

        ax.plot_surface(
            x,
            y,
            self.data,
            cmap="viridis",
            rstride=1,
            cstride=1,
            alpha=0.8,
            antialiased=True,
        )
        ax.set(**kwargs)

    def astorch(self) -> torch.Tensor:
        return self.data

    def asnumpy(self) -> np.ndarray:
        return self.data.numpy()

    def detach(self):
        self._data = self._data.detach()

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
    def __init__(self, tip_size: int = 20, tip_height: float = 5, tip_size_top: float = 5):
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
    def __init__(self, tip_size: int = 20, tip_height: float = 5, tip_size_top: float = 5):
        super().__init__(tip_size, tip_height, tip_size_top)

        # Set the points on the tip that are equal to the max to zero to simulate a
        # hollow tip
        top = torch.max(self._data)
        self._data[self._data == top] = -tip_height

@register_tip
class BigHollowPyramidTip(PyramidTip):
    def __init__(self, tip_size: int = 40, tip_height: float = 20, tip_size_top: float = 10):
        super().__init__(tip_size, tip_height, tip_size_top)

        # Set the points on the tip that are equal to the max to zero to simulate a
        # hollow tip
        top = torch.max(self._data)
        self._data[self._data == top] = -tip_height


@register_tip
class RandomTip(Tip):
    def __init__(self, tip_size: int = 20):
        self._tip_size = tip_size

        data = torch.rand((tip_size, tip_size), dtype=torch.float32)
        super().__init__(data)


if __name__ == "__main__":
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    tips = []
    tips.append(create_tip("PyramidTip"))
    tips.append(create_tip("HollowPyramidTip"))
    tips.append(create_tip("RandomTip"))

    fig_size = 5
    fig = plt.figure(figsize=(fig_size * len(tips), fig_size))
    for tip in tips:
        ax = fig.add_subplot(1, len(tips), tips.index(tip) + 1, projection="3d")
        tip.plot(ax, title=tip.__class__.__name__)

    plt.show()
