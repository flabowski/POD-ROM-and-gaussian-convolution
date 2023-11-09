import numpy as np
from scipy.interpolate import griddata
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift


def get_centers(points, triangles):
    return np.mean(points[triangles], axis=1)


def on_regular_grid(points, data, case="", **kvargs):
    x = np.unique(np.round(points[:, 0], decimals=8))
    y = np.unique(np.round(points[:, 1], decimals=8))
    if case == "dam_break":
        x = y = np.linspace(0, 0.584, 256, endpoint=True)
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


def add_padding(subj, shape, mode='constant'):
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
    return padded


def remove_padding(subj, shape, mode='constant'):
    """
    subj might be image or kernel.
    """
    # TODO: make it work in nd
    pad_x, pad_y = subj.shape[0]-shape[0], subj.shape[1]-shape[1]
    return subj[pad_x//2:-pad_x//2, pad_y//2:-pad_y//2]


def to_frequency(padded, shift=False):
    if shift:
        return fft2(ifftshift(padded))
    else:
        return fft2(padded)


def to_space(subj):
    return ifft2(subj).real


def convolve_f(y, g_f):
    y_f = fft(y)
    y_f_smooth = y_f*g_f
    y_smooth = ifft(y_f_smooth)
    return y_smooth.real


def convolve_f2D(y, g_f):
    y_f = fft2(y)
    y_f_smooth = y_f*g_f
    y_smooth = ifft2(y_f_smooth)
    return y_smooth.real


def smoothen(data_on_grid, psf):
    psf_p = add_padding(psf, data_on_grid.shape)
    psf_f = to_frequency(psf_p, shift=True)
    data_on_grid_p = add_padding(data_on_grid, psf.shape, mode="reflect")
    data_on_grid_f = to_frequency(data_on_grid_p)
    data_on_grid_smooth_f = data_on_grid_f * psf_f
    data_on_grid_s = to_space(data_on_grid_smooth_f)
    return data_on_grid_s


def _gauss_2d(sigma, truncate=4, size=False):
    if truncate and not size:
        s = int(truncate * sigma)
        # print(s)
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


def gaussian(x, sigma, shift=True):
    m = len(x)
    if shift:
        x_kernel = fftshift(x)
        offset = x_kernel[0]
    else:
        x_kernel = x
        offset = np.min(np.abs(x))  # or 0?
    A = 1/(m*sigma*(2*np.pi)**.5)
    return A * np.exp(-(x_kernel-offset)**2 / (2 * sigma**2))


def gaussian_f(x, sigma):
    m = len(x)
    dx = x[1] - x[0]
    f = np.fft.fftfreq(m, d=dx)
    # Fourier Transform of the Gaussian by Konstantinos G. Derpanis
    omega = 2*np.pi*f
    g_f = np.exp(-omega**2*sigma**2/2)
    return g_f


def remove_wall(data, X, Y):
    data = data.copy()
    # [2, 0, 0],
    # [2, 0.32876, 0],
    # [2.16438, 0.32876, 0],
    # [2.16438, 0, 0],
    is_wall = (0.292 <= X) & (X <= 0.316) & (Y <= 0.048)
    # data[47:54, 0:16] = .5
    data[is_wall] = -1
    return data


def threshold(data):
    data = data.copy()
    data[data < 0.5] = 0
    data[data >= 0.5] = 1
    return data
