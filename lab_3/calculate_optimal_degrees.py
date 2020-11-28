import numpy as np
from itertools import product


def determine_deg(a, p1, p2, p3):
    def _brute(args):
        i, j, k = args
        a.deg = [i + 1, j + 1, k + 1]
        a.built_A()
        a.lamb()
        a.psi()
        a.built_a()
        a.built_Fi()
        a.built_c()
        a.built_F()
        a.built_F_()
        res = np.linalg.norm(a.norm_error, np.inf)
        print (i, j, k), ':', np.linalg.norm(a.norm_error, np.inf)
        return (i, j, k), res, a.norm_error
    a.define_data()
    a.norm_data()
    a.define_norm_vectors()
    a.built_B()
    a.poly_func()
    d = map(_brute, product(p1, p2, p3))
    best = d[0]
    for i in d:
        if i[1] < best[1]:
            best = i
    return best
