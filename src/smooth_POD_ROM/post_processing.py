import warnings
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
from scipy.spatial import distance
from smooth_POD_ROM.reduced_order_model import train_ROM, L2_error
from smooth_POD_ROM.pre_processing import smoothen


def zielfunktion(params, x, mu_train, X_train, mu_test, X_test,
                 rank, shape, num_iter, clip=True):
    dx = x[1] - x[0]
    sigma, c = params[0], params[1]

    standard_rom = train_ROM(mu_train, X_train, rank=rank)
    X_test_ROM = standard_rom.predict(mu_test).T
    e_ROM = L2_error(X_test_ROM, X_test)

    X_train_s = smoothen(X_train, sigma/dx, shape)
    smooth_rom = train_ROM(mu_train, X_train_s, rank=rank)
    X_test_sROM = smooth_rom.predict(mu_test).T
    # e_sROM = L2_error(X_test_sROM, X_test)
    X_test_sROMs = post_process(x, X_test_sROM, sigma, c, mu_test, mu_train,
                                num_iter, shape=shape, clip=clip)
    e_sROMs = L2_error(X_test_sROMs, X_test)
    mean_sROMs = np.mean(e_sROMs)
    improvement = 100*mean_sROMs/np.mean(e_ROM)-100
    print("{:.6f}, {:.4f}, {:.8f}, {:.4f} %".format(
        sigma, c, mean_sROMs, improvement))
    return improvement, X_test_ROM, X_test_sROM, X_test_sROMs


def relative_distance_nearest_neighbour(single_point, points):
    distances = distance.cdist(points, single_point, 'euclidean').flatten()
    distances.sort()
    dim = points.shape[1]
    n = 2**dim  # 1D: 2NN; 2D: 4NN, 3D:8NN
    return distances[0] / (np.sum(distances[:n])/dim)


def get_sigma(sigma, mu, mu_train, c):
    d = relative_distance_nearest_neighbour(mu, mu_train)
    return sigma*(1 + c * d)


def richardson_lucy(x, im_blur, sgm, num_iter=50, truth=None, damping=2,
                    clip=False, mode="wrap"):
    if not len(im_blur.shape) == 2:
        warnings.warn("image needs to be 2D")
    dx = x[1]-x[0]
    im_deconv = im_blur.copy()
    eps = 1e-12  # regularization to avoid 0 division
    for _ in range(num_iter):
        #blurred = convolve2d(im_deconv.copy(), psf, boundary='symm', mode='same') + eps
        #blurred = convolve_f2D(im_deconv, psf_f2D) + eps
        blurred = gaussian_filter(
            im_deconv, sigma=sgm/dx, truncate=8, mode=mode)  # + eps
        blurred[np.abs(blurred) < eps] = eps
        relative_blur = im_blur / blurred
        #im_deconv *= convolve2d(relative_blur, psf_mirror, boundary='symm', mode='same')
        #im_deconv *= convolve_f2D(relative_blur, psf_f2D)
        error_estimate = gaussian_filter(
            relative_blur, sigma=sgm/dx, truncate=8, mode=mode)
        if damping:
            error_estimate[error_estimate > (1+damping)] = 1+damping
            error_estimate[error_estimate < (1-damping)] = 1-damping
        im_deconv *= error_estimate
        if clip:
            im_deconv[im_deconv < 0] = 0
            im_deconv[im_deconv > 1] = 1
        if isinstance(truth, np.ndarray):
            print(_, np.mean((truth-im_deconv.ravel())**2)**.5)
    return im_deconv


def post_process(x, data, sigma, c, mu_test, mu_train, num_iter,
                 shape, clip, progress=False):

    deconvolved = np.empty_like(data)
    for j in range(len(mu_test)):
        sgm_est = get_sigma(sigma, mu_test[j][None, ...], mu_train, c=c)
        # print(sgm_est)
        deconvolved[:, j] = richardson_lucy(x, data[:, j].reshape(shape),
                                            sgm_est, num_iter, truth=None,
                                            damping=2, clip=clip).ravel()
        if progress:
            print(j, end=", ")
    return deconvolved


def sharpen(data, s, ns):
    # assert s % 2 == 1, "must be odd"
    data_sharpened = data.copy()
    if s == 1:
        return data.ravel()
    elif s == 3:
        k = np.array([[0, -1, 0],
                      [-1,  5, -1],
                      [0, -1, 0]])
    elif s == 5:
        k = np.array([[0, 0, -1, 0, 0],
                      [0, 0, -1, 0, 0],
                      [-1, -1,  9, -1, -1],
                      [0, 0, -1, 0, 0],
                      [0, 0, -1, 0, 0]])
    elif s == 7:
        k = np.array([[0, 0, 0, -1, 0, 0, 0],
                      [0, 0, 0, -1, 0, 0, 0],
                      [0, 0, 0, -1, 0, 0, 0],
                      [-1, -1, -1,  13, -1, -1, -1],
                      [0, 0, 0, -1, 0, 0, 0],
                      [0, 0, 0, -1, 0, 0, 0],
                      [0, 0, 0, -1, 0, 0, 0]])
    elif s == 9:
        k = np.array([[0, 0, 0, 0, -1, 0, 0, 0, 0],
                      [0, 0, 0, 0, -1, 0, 0, 0, 0],
                      [0, 0, 0, 0, -1, 0, 0, 0, 0],
                      [0, 0, 0, 0, -1, 0, 0, 0, 0],
                      [-1, -1, -1, -1,  17, -1, -1, -1, -1],
                      [0, 0, 0, 0, -1, 0, 0, 0, 0],
                      [0, 0, 0, 0, -1, 0, 0, 0, 0],
                      [0, 0, 0, 0, -1, 0, 0, 0, 0],
                      [0, 0, 0, 0, -1, 0, 0, 0, 0]])
    elif s == 11:
        k = np.array([[0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                      [-1, -1, -1, -1, -1, 21, -1, -1, -1, -1, -1],
                      [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0]])
    else:
        raise ValueError("unknown size s ={:.0f}".format(s))
    for i in range(ns):
        data_sharpened = convolve2d(data_sharpened, k,
                                    boundary='symm', mode='same')
    return data_sharpened.ravel()


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
