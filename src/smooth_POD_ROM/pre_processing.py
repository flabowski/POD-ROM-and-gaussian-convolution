import numpy as np
from scipy.interpolate import griddata
from scipy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import gaussian_filter


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
    data_on_grid = griddata(points[:, :2], data, (X, Y), **kvargs)
    return X, Y, data_on_grid


def is_odd(n):
    return n % 2 == 1


def add_padding(subj, shape, mode="constant"):
    """
    subj might be image or kernel.
    """
    # TODO: make it work in nd
    msg = "please make sure the number of samples in all dimensions is odd to" " avoid unexpected behaviour."
    for n in subj.shape:
        assert is_odd(n), msg
    for n in shape:
        assert is_odd(n), msg
    pad_x, pad_y = shape
    padded = np.pad(subj, [(pad_x // 2 + 1, pad_x // 2 + 1), (pad_y // 2 + 1, pad_y // 2 + 1)], mode=mode)
    return padded


def remove_padding(subj, shape, mode="constant"):
    """
    subj might be image or kernel.
    """
    # TODO: make it work in nd
    pad_x, pad_y = subj.shape[0] - shape[0], subj.shape[1] - shape[1]
    return subj[pad_x // 2 : -pad_x // 2, pad_y // 2 : -pad_y // 2]


def to_frequency(padded, shift=False):
    if shift:
        return fft2(ifftshift(padded))
    else:
        return fft2(padded)


def to_space(subj):
    return ifft2(subj).real


def convolve_f(y, g_f):
    y_f = fft(y)
    y_f_smooth = y_f * g_f
    y_smooth = ifft(y_f_smooth)
    return np.abs(y_smooth)


def convolve_f2D(y, g_f):
    y_f = fft2(y)
    y_f_smooth = y_f * g_f
    y_smooth = ifft2(y_f_smooth)
    return np.abs(y_smooth)  # y_smooth.real


def smoothen_(data_on_grid, psf):
    # old. 1 snapshot + padding
    psf_p = add_padding(psf, data_on_grid.shape)
    psf_f = to_frequency(psf_p, shift=True)
    data_on_grid_p = add_padding(data_on_grid, psf.shape, mode="reflect")
    data_on_grid_f = to_frequency(data_on_grid_p)
    data_on_grid_smooth_f = data_on_grid_f * psf_f
    data_on_grid_s = to_space(data_on_grid_smooth_f)
    return data_on_grid_s


def smoothen(X, sigma, shape, truncate=8, mode="wrap"):
    # sigma defined in terms of nodes not x!
    # consider passing sigma/dx to this function
    X_s = np.empty_like(X)
    for j in range(X.shape[1]):
        ss2D = X[:, j].reshape(shape)
        # add padding?
        X_s[:, j] = gaussian_filter(ss2D, sigma=sigma, truncate=truncate, mode=mode).ravel()
    return X_s


def smoothen_rowwise(X, sigma, shape, truncate=8, mode="wrap"):
    # sigma defined in terms of nodes not x!
    # consider passing sigma/dx to this function
    X_s = np.empty_like(X)
    for i in range(X.shape[0]):
        ss2D = X[i].reshape(shape)
        # add padding?
        X_s[i] = gaussian_filter(ss2D, sigma=sigma, truncate=truncate, mode=mode).ravel()
    return X_s


def _gauss_2d(sigma, truncate=4, size=False):
    if truncate and not size:
        s = int(truncate * sigma)
        # print(s)
        x = np.arange(-s, s + 1, 1)
        y = np.arange(-s, s + 1, 1)
    elif not truncate and size:
        s = size // 2
        if size % 2 == 1:
            x = np.arange(-s, s + 1, 1)
            y = np.arange(-s, s + 1, 1)
        else:
            x = np.arange(-s, s, 1)
            y = np.arange(-s, s, 1)
    else:
        raise ValueError("specify either truncate OR size.")
    x, y = np.meshgrid(x, y, indexing="ij")
    mu_x = mu_y = 0
    sigma_x = sigma_y = sigma
    exponent = ((x - mu_x) ** 2 / (2 * sigma_x**2)) + ((y - mu_y) ** 2 / (2 * sigma_y**2))
    a = 1 / (2 * np.pi * sigma_x * sigma_y)
    res = a * np.exp(-exponent)
    res /= np.sum(res)
    return res


def gaussian(x, sigma, shift=True):
    dx = x[1] - x[0]
    if shift:
        x_kernel = fftshift(x)
        offset = x_kernel[0]
    else:
        x_kernel = x
        offset = np.min(np.abs(x))  # or 0?
    A = dx / (sigma * (2 * np.pi) ** 0.5)
    return A * np.exp(-((x_kernel - offset) ** 2) / (2 * sigma**2))


def gaussian_f(x, sigma):
    m = len(x)
    dx = x[1] - x[0]
    f = np.fft.fftfreq(m, d=dx)
    # Fourier Transform of the Gaussian by Konstantinos G. Derpanis
    omega = 2 * np.pi * f
    g_f = np.exp(-(omega**2) * sigma**2 / 2)
    return g_f
