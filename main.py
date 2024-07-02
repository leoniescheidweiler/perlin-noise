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


width = 1600
height = 1200

world = World(width, height)

scale = 200
amplitude = 100
iterations = 6

fig, ax = plt.subplots(figsize=(10, 7))

world.zz += perlin_noise(world.x, world.y,
                         scale=scale,
                         amplitude=amplitude,
                         iterations=iterations
                         )

map = ax.pcolormesh(world.xx, world.yy, world.zz, cmap="coolwarm", vmin=-100, vmax=100)
ax.contour(world.xx, world.yy, world.zz, [0])
ax.set_aspect("equal")

divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(map, cax=cax, orientation='vertical')

# ax.set_axis_off()

plt.savefig("perlin_noise.png", dpi=600)
plt.show()
