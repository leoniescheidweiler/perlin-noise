import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools

from perlin_noise import perlin_noise


class World:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.x = range(width)
        self.y = range(height)
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="ij")
        self.zz = np.zeros(shape=(width, height))


width = 800
height = 600

world = World(width, height)

scale = 100
amplitude = 100
iterations = 4

fig, ax = plt.subplots(figsize=(10, 7))

world.zz += perlin_noise(world.x, world.y,
                         scale=scale,
                         amplitude=amplitude,
                         iterations=iterations
                         )

map = plt.pcolormesh(world.xx, world.yy, world.zz, cmap="coolwarm", vmin=-100, vmax=100)
plt.contour(world.xx, world.yy, world.zz, [0])
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(map, cax=cax, orientation='vertical')
# ax.set_axis_off()
plt.show()
