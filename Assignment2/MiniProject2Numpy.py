import matplotlib.pyplot as plt
import time
import numpy as np

def Mandelbrot(output, c, maxite):
    z = np.zeros_like(c)
    mask = np.ones_like(c, dtype=bool)

    for ii in range(maxite):
        z[mask] = z[mask]**2 + c[mask]
        mask = abs(z) <= 2
        output[mask] = ii / maxite

    return output



def complex_matrix(xmin, xmax, ymin, ymax, pxd):
    re = np.linspace(xmin, xmax, pxd)
    im = np.linspace(ymin, ymax, pxd)
    return re[np.newaxis, :] + im[:, np.newaxis]*1j

if __name__ == "__main__":
    pxd = 5000

    beginTimer = time.perf_counter()
    print("start: ", time.perf_counter()-beginTimer)

    c = complex_matrix(-2, 1, -1.5, 1.5, pxd)
    output = np.zeros((pxd, pxd))

    print("matrix done: ", time.perf_counter()-beginTimer)
    plt.imshow(Mandelbrot(output, c, maxite=100), cmap="hot", extent=[-2, 1, -1.5, 1.5])
    print("calculations done: ", time.perf_counter()-beginTimer)
    plt.show()
