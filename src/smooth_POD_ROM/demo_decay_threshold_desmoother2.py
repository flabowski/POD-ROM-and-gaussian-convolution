# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 15:02:44 2024

@author: florianma
"""

import numpy as np
import matplotlib.pyplot as plt
import time

NN = np.load("U:/POD-ROM-and-gaussian-convolution/src/smooth_POD_ROM/NN.npy")
dN_ROM = np.load(
    "U:/POD-ROM-and-gaussian-convolution/src/smooth_POD_ROM/dN_ROM.npy")
dN_sROMs = np.load(
    "U:/POD-ROM-and-gaussian-convolution/src/smooth_POD_ROM/dN_sROMs.npy")
dN_sig = np.load(
    "U:/POD-ROM-and-gaussian-convolution/src/smooth_POD_ROM/dN_sig.npy")

page_width_in = 5.395665354330708


fig, ax = plt.subplots(figsize=(page_width_in/2, page_width_in/3))
plt.scatter(NN, dN_sig, c=dN_sROMs/dN_ROM, vmin=-100, vmax=0)
plt.plot(NN, 4/NN, "k.")
plt.plot(NN, 4/NN*5, "r.")
plt.plot(NN, 0*NN+1e-7, "r.")
plt.show()


fig, ax = plt.subplots(figsize=(page_width_in/2, page_width_in/3))
plt.plot(NN, dN_ROM, "C0o-", ms=2, label="$u_{\mu,rb}$")
#plt.plot(NN, esROM, "C1--", ms=2, label="$u_{rb,S}$")
plt.plot(NN, dN_sROMs, "C2o-", ms=2, label="$u_{\mu,Srb,D}$")
plt.plot(NN, 0.5*1/NN**.5, "C0--", label=".5/sqrt(N)")
plt.plot(NN, 30/NN**2, "C2--", label="25/(N**2)")
plt.plot(NN, .9/NN, "C1--", label=".9/(N)")
# plt.plot(NN, dN_sROMs2, "C3.-", ms=2, label="$u_{\mu,rb,D_{1000}}$")
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
plt.xlim(0, 100)
plt.ylim(1e-3, .3)
plt.show()
