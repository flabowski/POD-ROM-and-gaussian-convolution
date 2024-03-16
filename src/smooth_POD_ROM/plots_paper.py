import numpy as np
from scipy.interpolate._rgi_cython import evaluate_linear_2d, find_indices
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

page_width_pt = 455.24
pt2in = 0.01389
pt2cm = 0.0352777778
cm2in = 1/2.54
page_width_cm = 13.70499
# TODO: work with textwidth
plot_width_in = page_width_pt*pt2in/2
page_width_in = page_width_cm*cm2in
print(plot_width_in/cm2in)

fs = 10
fs_lbl = 6
plt.rcParams["figure.figsize"] = (plot_width_in, plot_width_in/1.61803398875)
plt.rcParams["figure.autolayout"] = True
plt.rcParams['font.size'] = fs
plt.rcParams['axes.titlesize'] = fs
plt.rcParams['axes.labelsize'] = fs
plt.rcParams['xtick.labelsize'] = fs
plt.rcParams['ytick.labelsize'] = fs
plt.rcParams['legend.labelspacing'] = 0.0
plt.rcParams['legend.fontsize'] = fs_lbl
plt.rcParams['legend.handlelength'] = 2.0

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif', size=fs, serif='Computer Modern Roman')
#plt.rcParams['font.serif'] = ['Times New Roman']
pth = "../Plots/"

markevery = 40


def Fig3(x, mu_train, X_train, X_train_s):
    fig = plt.figure(figsize=(page_width_in, page_width_in/2),
                     facecolor='white')
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    for i in range(len(mu_train)):
        ax1.plot(np.ones_like(x)*mu_train[i], x, X_train[:, i], "k-", lw=1)
        ax2.plot(np.ones_like(x)*mu_train[i], x, X_train_s[:, i], "k-", lw=1)
    for ax in (ax1, ax2):
        ax.xaxis.labelpad = -5
        ax.yaxis.labelpad = -5
        xticks = [0.0,  0.2, 0.4, 0.6, 0.8]
        xticks = [0.0, 0.5, 1.0, 1.5]
        yticks = [0.0, 0.5, 1.0]
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_zticks([0.5, 1.0])
        ax.tick_params(axis='x', which='major', pad=-5)
        ax.tick_params(axis='y', which='major', pad=-2)
        ax.tick_params(axis='z', which='major', pad=0)
        ax.set_yticklabels(yticks, va='baseline', ha='left')
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            for tick in axis.get_major_ticks():
                # for some reasons it does not match
                tick.label1.set_fontsize(8)
            axis.pane.set_edgecolor('white')

        ax.grid(False)
        ax.set_facecolor('white')
        ax.xaxis.pane.fill = False  # Remove the background pane
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.set_xlabel("$\mu$")
        ax.set_ylabel("$x$")
    plt.show()


def Fig4(x, i_c, mu_train, mu_test, X_train, X_train_s, X_test, X_test_ROM, X_test_sROM):
    # i_c = 0  # snapshots centered between 2 training points
    indices_l, norm_distances_l = find_indices(
        (mu_train.ravel(), ), mu_test[i_c, None])
    i_l = indices_l.ravel()  # index snapshot left
    for _X, _X_R, lbl, col in zip([X_train, X_train_s], [X_test_ROM, X_test_sROM],
                                  ["$u_{\mu,rb}$: standard ROM", "$u_{\mu, rb, S}$: regularized ROM"], ["C0-", "C1--"]):
        fig, ax = plt.subplots(figsize=(page_width_in/2, page_width_in/3))
        #plt.plot(x, pulse(x, mu_plot), "kx-", ms=4, markevery=markevery, label="$u_h$: true solution")
        plt.plot(x, X_test[:, i_c], "k-", ms=4,
                 markevery=markevery, label="$u_{\mu}$: true solution")
        #plt.plot(x, X_test_ROM[:, 0], "b.-", label="ROM")
        plt.plot(x, _X_R[:, i_c], col, ms=4, markevery=markevery, label=lbl)
        #plt.plot(x, X[:, 3], "r--", lw=1, label="closest snapshot (left)")
        #plt.plot(x, X[:, 4], "r--", lw=1, label="closest snapshot (right)")
        plt.plot(x, _X[:, i_l], "C3-", lw=.5, ms=4, markevery=(30,
                 markevery), label="$u_{2}$: snapshot \#3")
        plt.plot(x, _X[:, i_l+1], "C3-", lw=.5, ms=4, markevery=(30,
                 markevery), label="$u_{3}$: snapshot \#4")
        ax.set_xticks(np.linspace(0, 1, 11, endpoint=True), minor=True)
        ax.set_yticks(np.linspace(0, 1, 11, endpoint=True), minor=True)
        plt.grid(True, which='minor', linestyle='--', lw=.25)
        plt.grid(True, which='major', linestyle='-')
        plt.legend()
        e = np.mean((X_test[:, 0]-_X_R[:, 0])**2)**.5
        ax.text(0.35, 0.35, "$\|u_{\mu}-$"+lbl.split(":")
                [0]+"$\|_{L_2}"+"={:.2f}$".format(e), fontsize=8, va="center")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.05, 1.05)
        plt.show()


def Fig5(x, X_test, X_test_s, X_test_ROM, X_test_sROM, X_test_sROMs):
    for j in [0, 2, 5, 7]:
        markevery = 45
        fig, ax = plt.subplots(figsize=(page_width_in/2, page_width_in/3))
        plt.plot(x, X_test[:, j], "k-", ms=4, markevery=(0,
                 markevery), label="$u_{\mu}$")  # : true solution
        # : regularized true solution
        plt.plot(x, X_test_s[:, j], "k--", ms=4,
                 markevery=(20, markevery), label="$u_{\mu, S}$")
        plt.plot(x, X_test_ROM[:, j], "C0-", ms=4, markevery=(10,
                 markevery), label="$u_{\mu, rb}$")  # : standard ROM
        # : regularized ROM
        plt.plot(x, X_test_sROM[:, j], "C1--", ms=4,
                 markevery=markevery, label="$u_{\mu, rb, S}$")
        plt.plot(x, X_test_sROMs[:, j], "C2-", ms=4, markevery=(30,
                 markevery), label="$u_{\mu, rb, DS}$")  # : deconvolved reg. ROM
    #     plt.plot(x, X_test_sROMs2[:, j], "C3|-", ms=4, markevery=markevery, label="$u_{rb, D_{100}}$") # : deconvolved reg. ROM
        ax.set_xticks(np.linspace(0, 1, 11, endpoint=True), minor=True)
        ax.set_yticks(np.linspace(0, 2, 21, endpoint=True), minor=True)

        e = np.mean((X_test[:, j]-X_test_ROM[:, j])**2)**.5
        ax.text(0.35, 0.45, "$\|u_{\mu}{-u_{\mu, rb}}\|_{L_2}" +
                "={:.3f}$".format(e), fontsize=8, va="center")
        e = np.mean((X_test[:, j]-X_test_sROM[:, j])**2)**.5
        ax.text(0.35, 0.3, "$\|u_{\mu}{-u_{\mu, rb, S}}\|_{L_2}" +
                "={:.3f}$".format(e), fontsize=8, va="center")
        e = np.mean((X_test[:, j]-X_test_sROMs[:, j])**2)**.5
        ax.text(0.35, 0.15, "$\|u_{\mu}{-u_{\mu, rb, DS}}\|_{L_2}" +
                "={:.3f}$".format(e), fontsize=8, va="center")
    #     e = np.mean((X_test[:, j]-X_test_sROMs2[:, j])**2)**.5
    #     ax.text(0.35, 0.0, "$\|u_h{-u_{rb,D_{1000}}}\|_{L_2}"+"={:.3f}$".format(e), fontsize=8, va="center")

        plt.grid(True, which='minor', linestyle='--', lw=0.25)
        plt.grid(True, which='major', linestyle='-')
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.05, 1.05)
        plt.show()


def Fig6(mu_val, eROM, esROM, esROMs, esROMs2):
    fig, ax = plt.subplots(figsize=(page_width_in/2, page_width_in/3))
    ax.plot(mu_val, eROM, "C0-", label="$u_{\mu, rb}$")
    ax.plot(mu_val, esROM, "C1--", label="$u_{\mu, rb, S}$")
    ax.plot(mu_val, esROMs, "C2-", label="$u_{\mu, rb, SD_{10}}$")
    ax.plot(mu_val, esROMs2, "C3-", label="$u_{\mu, rb, SD_{250}}$")
    plt.ylabel("$\|u_{\mu}{-u_{\mu, rb}}\|_{L_2}$")
    plt.xlabel("$\mu$")
    plt.legend()
    ax.set_xticks(np.linspace(0, 1, 11, endpoint=True), minor=True)
    ax.set_yticks(np.linspace(0, .4, 19, endpoint=True), minor=True)
    plt.grid(True, which='minor', linestyle='--', lw=.25)
    plt.grid(True, which='major', linestyle='-')
    plt.xlim(0, 1)
    plt.show()


def Fig7(eROM, esROM, esROMs, c_all, sigmas):
    mean_eROM = np.mean(eROM)
    mean_esROM = np.mean(esROM, axis=0)
    mean_esROMs = np.mean(esROMs, axis=0)

    # print(mean_eROM)
    # print(np.min(mean_esROM), np.max(mean_esROM))
    # print(np.min(mean_esROMs), np.max(mean_esROMs))
    # print(np.min(mean_esROMs)/mean_eROM)
    fig, ax = plt.subplots(figsize=(page_width_in/2, page_width_in/3))
    dc = c_all[1]-c_all[0]
    ds = sigmas[1]-sigmas[0]
    ext = (c_all[0]-dc/2, c_all[-1]+dc/2, (sigmas[0]-ds/2), (sigmas[-1]+ds/2))
    Z = mean_esROMs/mean_eROM*100-100
    print(np.min(Z), "%")
    cs = ax.imshow(Z, interpolation="nearest", origin="lower",
                   vmin=np.min(Z), vmax=0, extent=ext)
    lvls = np.linspace(np.min(Z)//5*5, 0, 4)
    contours = ax.contour(
        Z, levels=lvls, colors=["g", "g", "g", "r"], extent=ext)
    #     cs.cmap.set_over('red')
    #     cs.cmap.set_under('blue')
    #np.save(pth+"mean_esROMs.npy", mean_esROMs)
    ax.set_xticks(c_all[::4])
    ax.set_yticks(sigmas[::4])
    ax.set_xlabel("$c$")
    ax.set_ylabel("$\sigma_S$")
    ax.set_aspect("auto")
    #     ax.ticklabel_format(axis='y', style='sci', useOffset=True)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(
        0, 0), useOffset=False, useMathText=True)
    ax.yaxis.get_offset_text().set_size(6)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(cs, cax=cax)
    cbar.set_ticks(lvls)
    # plt.savefig(pth+"2nd_ex_hyperparams.pdf")
    plt.show()
    # print(np.min(mean_esROMs[0, :]))
    # print(np.min(mean_esROMs[-1, :]))
    i, j = np.unravel_index(
        np.argmin(mean_esROMs, axis=None), mean_esROMs.shape)
    print("best sigma, best c:", i, j, sigmas[i], c_all[j])
    return


def Fig8(NN, dN_ROM, dN_sROMs, dN_sROMs2):
    fig, ax = plt.subplots(figsize=(page_width_in/2, page_width_in/3))
    plt.plot(NN, dN_ROM, "C0.-", ms=2, label="$u_{rb}$")
    #plt.plot(NN, esROM, "C1--", ms=2, label="$u_{rb,S}$")
    plt.plot(NN, dN_sROMs, "C2.-", ms=2, label="$u_{\mu,rb,D_{100}}$")
    plt.plot(NN, dN_sROMs2, "C3.-", ms=2, label="$u_{\mu,rb,D_{1000}}$")
    #plt.plot(mu_val, esROMs2, "C3<-", ms=4, markevery=(20, markevery), label="$u_{rb,D_{100}}$")
    plt.legend()
    plt.ylabel("$\|u_{\mu,rb}-u_{\mu}\|_{L_2}$")
    plt.xlabel("$N$")
    # plt.ylim(0, 0.4)
    plt.legend()
    ax.set_xticks(np.linspace(0, 250, 26, endpoint=True), minor=True)
    ax.set_yticks(np.linspace(0, .4, 19, endpoint=True), minor=True)
    plt.grid(True, which='minor', linestyle='--', lw=.25)
    plt.grid(True, which='major', linestyle='-')
    ax.set_yscale('log')
    plt.xlim(0, 150)
    plt.ylim(2e-2, .3)
    plt.show()


def Fig11(cases, improvements):
    fig, ax = plt.subplots(figsize=(page_width_in/2, page_width_in/3))
    for c, (sigma_S, sigma_D, case_name) in enumerate(cases):
        plt.plot(improvements[c], marker=".", label=case_name)
    plt.legend()
    plt.xlim(0, 3000)
    return
