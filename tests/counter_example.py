# Author: Zhengying Liu
# Creation date: 2021 Oct 14

import numpy as np
import matplotlib.pyplot as plt


def get_Sn(n):
    """Return the list of n! permutations."""
    if n < 0:
        raise ValueError("`n` must be >= 0 but got {}.".format(n))
    elif n == 0:
        return [[]]
    elif n == 1:
        return [[0]]

    res = []
    Sn_1 = get_Sn(n - 1)

    for s in Sn_1:
        for i in range(n - 1, -1, -1):
            s_new = s[:i] + [n - 1] + s[i:]
            res.append(s_new)
    return res


def inv_perm(s):
    """Invert a permutation."""
    s_inv = [None] * len(s)
    for i in range(len(s)):
        s_inv[s[i]] = i
    return s_inv


if __name__ == "__main__":
    n = 4
    Sn = get_Sn(n)
    le = len(Sn)
    Sn_inv = [inv_perm(s) for s in Sn]
    prob = np.random.rand(len(Sn))
    prob = prob / prob.sum()
    Tr = []
    Te = []
    for k in range(n):
        Trk = 0
        Tek = 0
        for d in range(le):
            sd = Sn[d]
            for f in range(le):
                sf = Sn[f]
                sf_inv = Sn_inv[f]
                m = min([sf_inv[sd[x]] for x in range(k + 1)])
                sfm = sf[m]
                Trk += m * prob[d] * prob[f]
                Tek += sfm * prob[d] * prob[f]
        Tr.append(Trk)
        Te.append(Tek)
    print("Tr:", Tr)
    print("Te:", Te)
    print(prob)
    plt.plot(Tr, label="Tr", marker="o")
    plt.plot(Te, label="Te", marker="o")
    plt.legend()
    plt.show()




