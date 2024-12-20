import numpy as np


def rectangular_pulse(x, mu, w=0.075 + 1e-6, periodic=True):
    y = np.zeros_like(x, dtype=np.float64)
    y[(0 - 1e-6 < (x - mu)) & ((x - mu) < w)] = 1.0
    if periodic:
        y[(0 - 1e-6 < ((x - 1) - mu)) & (((x - 1) - mu) < w)] = 1.0
        y[(0 - 1e-6 < ((x + 1) - mu)) & (((x + 1) - mu) < w)] = 1.0
    return y


def rect_pulse_sin(x, mu, w=0.075 + 1e-6, periodic=True):
    w = 1 / 14
    y = np.zeros_like(x, dtype=np.float64)
    y[(0 - 1e-6 < (x - mu)) & ((x - mu) < w)] = 0.9
    ys = (np.sin((x - mu) * 2 * np.pi / w / 2) + 1) / 2 * 0.1
    if periodic:
        ys = (np.sin(((x - 1) - mu) * 2 * np.pi / w / 2) + 1) / 2 * 0.1
        ys = (np.sin(((x + 1) - mu) * 2 * np.pi / w / 2) + 1) / 2 * 0.1
    return y + ys


def saw_tooth(x, mu, w=0.075 + 1e-6, periodic=True):
    w = 1 / 14
    y = np.zeros_like(x, dtype=np.float64)
    nonzero = (0 - 1e-6 < (x - mu)) & ((x - mu) < w)
    y[nonzero] = 1 / w * (x[nonzero] - mu)
    if periodic:
        nonzero = (0 - 1e-6 < ((x - 1) - mu)) & (((x - 1) - mu) < w)
        y[nonzero] = 1 / w * ((x - 1)[nonzero] - mu)
        nonzero = (0 - 1e-6 < ((x + 1) - mu)) & (((x + 1) - mu) < w)
        y[nonzero] = 1 / w * ((x + 1)[nonzero] - mu)
    return y


def triangle(x, mu, w=0.075 + 1e-6):
    w = 1 / 14
    y = np.zeros_like(x, dtype=np.float64)
    nonzero = (0 - 1e-6 < (x - mu)) & ((x - mu) < w / 2)
    y[nonzero] = 1 / (w / 2) * (x[nonzero] - mu)
    nonzero = (w / 2 < (x - mu)) & ((x - mu) < w)
    y[nonzero] = -1 / (w / 2) * (x[nonzero] - mu - w)
    return y
