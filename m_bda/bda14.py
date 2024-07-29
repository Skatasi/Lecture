import numpy as np

def Fastmap(O, d, m):
    for i in 1 in range(m):
        A, B = SelectLine(O, d)
        for  C in O :
            x_{C, i} = (d_{AB}^2 + d_{AC}^2 - d_{BC}^2) / 2d_{AB}
        d = Deflate(O, d, x, i)
    return x

def SelectLine(O, d):
    B = RandomPoint(O)
    max_dist = 0
    for C in O:
        if d_{BC} > max_dist:
            A = C
            max_dist = d_{BC}
    max_dist = 0
    for C in O:
        if d_{AC} > max_dist:
            B = C, max_dist = d_{AC}
    return A, B

def Deflate(O, d, x, i):
    for C in O:
        for D in O:
            d_{CD} = np.sqrt(d_{CD}^2 - (x_{C, i} - x_{D, i})^2)
    return d