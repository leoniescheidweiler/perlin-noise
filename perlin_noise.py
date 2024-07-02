import numpy as np


def smoothstep(x):
    return 3*x**2 - 2*x**3


def smootherstep(x):
    return 6*x**5 - 15*x**4 + 10*x**3


def perlin_noise(noise_x, noise_y, scale, amplitude, iterations):
    norm = 0
    for i in range(iterations):
        norm += 2^i

    noise = np.zeros(shape=(len(noise_x), len(noise_y)))

    for i in range(iterations):
        noise += perlin_iteration(noise_x, noise_y,
                                  scale=scale/2**i,
                                  amplitude=amplitude/2*(iterations-i-1)/norm
                                  )
    
    return noise


def perlin_iteration(noise_x, noise_y, scale, amplitude):
    # grid on which noise is calculated given by x, y

    # grid on which gradient vectors are calculated given by gradient_x, gradient_y
    x_offset = 0.5
    y_offset = 0.5
    perlin_xmin = min(noise_x) + x_offset - (scale + 1)
    perlin_xmax = max(noise_x) + x_offset + (scale + 1)
    perlin_ymin = min(noise_y) + y_offset - (scale - 1)
    perlin_ymax = max(noise_y) + y_offset + (scale - 1)
    gradient_x = np.arange(perlin_xmin, perlin_xmax, scale)
    gradient_y = np.arange(perlin_ymin, perlin_ymax, scale)

    # generate gradient vectors
    angle = np.random.rand(len(gradient_x), len(gradient_y))*360
    gradient_vectors = np.stack((np.cos(angle), np.sin(angle)), axis=2)

    # interpolate onto noise grid
    noise = np.zeros(shape=(len(noise_x), len(noise_y)))

    for xi in range(len(noise_x)):
        for yi in range(len(noise_y)):
            # coordinate of current candidate point
            x = noise_x[xi]
            y = noise_y[yi]

            # left closest gradient_x
            gradient_xi_left = np.max(np.where(gradient_x < x))
            gradient_x_left = gradient_x[gradient_xi_left]

            # right closest gradient_x
            gradient_xi_right = np.min(np.where(gradient_x >= x))
            gradient_x_right = gradient_x[gradient_xi_right]

            # bottom closest gradient_y
            gradient_yi_bottom = np.max(np.where(gradient_y < y))
            gradient_y_bottom = gradient_y[gradient_yi_bottom]

            # top closest gradient_y
            gradient_yi_top = np.min(np.where(gradient_y >= y))
            gradient_y_top = gradient_y[gradient_yi_top]

            # corner 1 (left, bottom)
            g1 = gradient_vectors[gradient_xi_left, gradient_yi_bottom]
            o1 = np.array((x - gradient_x_left, y - gradient_y_bottom))
            o1 = np.divide(o1, np.linalg.norm(o1))
            d1 = np.dot(g1, o1)

            # corner 2 (right, bottom)
            g2 = gradient_vectors[gradient_xi_right, gradient_yi_bottom]
            o2 = np.array((x - gradient_x_right, y - gradient_y_bottom))
            o2 = np.divide(o2, np.linalg.norm(o2))
            d2 = np.dot(g2, o2)

            # corner 3 (right, top)
            g3 = gradient_vectors[gradient_xi_right, gradient_yi_top]
            o3 = np.array((x - gradient_x_right, y - gradient_y_top))
            o3 = np.divide(o3, np.linalg.norm(o3))
            d3 = np.dot(g3, o3)

            # corner 4 (left, top)
            g4 = gradient_vectors[gradient_xi_left, gradient_yi_top]
            o4 = np.array((x - gradient_x_left, y - gradient_y_top))
            o4 = np.divide(o4, np.linalg.norm(o4))
            d4 = np.dot(g4, o4)

            # interpolation
            x_interp = smootherstep((x - gradient_x_left)/scale)
            y_interp = smootherstep((y - gradient_y_bottom)/scale)
            dtop = x_interp * d3 + (1 - x_interp) * d4
            dbot = x_interp * d2 + (1 - x_interp) * d1
            result = y_interp * dtop + (1 - y_interp) * dbot

            noise[x, y] = result * amplitude

    return noise
