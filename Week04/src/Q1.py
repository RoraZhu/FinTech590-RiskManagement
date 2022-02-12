import numpy as np


def brownianPrice(Pt_1, ret):
    return Pt_1 + ret


def arithmeticPrice(Pt_1, ret):
    return Pt_1 * (1 + ret)


def geometricBrownPrice(Pt_1, ret):
    return Pt_1 * np.exp(ret)