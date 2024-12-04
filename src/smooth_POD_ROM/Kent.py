#import scipy as sp
import matplotlib.pyplot as plt
import numpy as np


def gauss(x, sigma, x0): return (1/0.1772453850902798)*np.exp(-(x-x0)**2/sigma)


def step(x, sigma, x0):
    y = x.copy()*0
    for i in range(len(y)):
        if abs(x[i]-x0) < (sigma+1e-12):
            y[i] = 0.5/sigma
        else:
            y[i] = 0
    return y


N = 100
x = np.arange(0, 1, 0.001)
sigma = 0.05
points = x.copy()[::10]  # np.random.random(N)
gs = []
ss = []
for point in points:
    g = gauss(x, sigma, point)
    s = step(x, sigma, point)
    gs.append(g)
    ss.append(s)


# for g in gs:
#  plt.plot(g)
# for s in ss:
#  plt.plot(s)
# plt.show()


G = np.matrix(gs)
Ug, Sg, Vg = np.linalg.svd(G.T)

print(Sg)

S = np.matrix(ss)
Us, Ss, Vs = np.linalg.svd(S.T)

# plt.imshow(S)
print(Ss)

# plt.loglog(Sg[:-1])
# plt.loglog(Ss[:-1])
#plt.legend(["gauss", "step"])
# plt.show()


s_ = 6
for i in range(0, s_):
    # plt.plot(Vg[i, :], "b")
    plt.plot(x, Ug[:, i], "b.-")

for i in range(0, s_):
    # plt.plot(Vs[:, i], "r")
    plt.plot(x, Us[:, i], "r.-")


leg = ["gauss" for i in range(0, s_)]
leg2 = ["step" for i in range(0, s_)]
leg.extend(leg2)
plt.legend(leg)
plt.show()
