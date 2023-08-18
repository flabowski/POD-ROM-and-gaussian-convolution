import numpy as np
from scipy.interpolate import griddata
from scipy.fft import fft2, ifft2, fftshift, ifftshift


def get_centers(points, triangles):
    return np.mean(points[triangles], axis=1)


def on_regular_grid(points, data, **kvargs):
    x = np.unique(np.round(points[:, 0], decimals=8))
    y = np.unique(np.round(points[:, 1], decimals=8))
    # xcenter = (x[:-1]+x[1:]) / 2
    # ycenter = (y[:-1]+y[1:]) / 2
    # assert len(xcenter) > 1, "not enough data"
    # assert len(ycenter) > 1, "not enough data"
    X, Y = np.meshgrid(x, y, indexing="ij")
    data_on_grid = griddata(points[:, :2], data,
                            (X, Y), **kvargs)
    return X, Y, data_on_grid


def is_odd(n):
    return n % 2 == 1


def to_frequency(subj, shape, mode='constant'):
    """
    subj might be image or kernel.
    """
    # TODO: make it work in nd
    msg = ("please make sure the number of samples in all dimensions is odd to"
           " avoid unexpected behaviour.")
    for n in subj.shape:
        assert is_odd(n), msg
    for n in shape:
        assert is_odd(n), msg
    pad_x, pad_y = shape
    padded = np.pad(subj, [(pad_x//2+1, pad_x//2+1), (pad_y//2+1, pad_y//2+1)],
                    mode=mode)
    return fft2(padded)


def to_space(subj):
    return ifftshift(ifft2(subj).real)


def _gauss_2d(sigma, truncate=4, size=False):
    if truncate and not size:
        s = int(truncate * sigma)
        print(s)
        x = np.arange(-s, s+1, 1)
        y = np.arange(-s, s+1, 1)
    elif not truncate and size:
        s = size//2
        if size % 2 == 1:
            x = np.arange(-s, s+1, 1)
            y = np.arange(-s, s+1, 1)
        else:
            x = np.arange(-s, s, 1)
            y = np.arange(-s, s, 1)
    else:
        raise ValueError("specify either truncate OR size.")
    x, y = np.meshgrid(x, y, indexing="ij")
    mu_x = mu_y = 0
    sigma_x = sigma_y = sigma
    exponent = ((x - mu_x)**2 / (2 * sigma_x**2)) + \
        ((y - mu_y)**2 / (2 * sigma_y**2))
    a = 1 / (2 * np.pi * sigma_x * sigma_y)
    res = a * np.exp(-exponent)
    res /= np.sum(res)
    return res
