import matplotlib.pyplot as plt
import time
import numpy as np

def Mandelbrot(c, maxite):
    """
    Calculates the Mandelbrot set, which will be passed on as a float matrix which will then be plotted.

    Parameters
    ----------
    c : complex128
        A complex matrix, used to compute the Mandelbrot set.
    maxite : integer
             Integer, used to set the maximum amount of iterations.

    Returns
    -------
    output : float64
             A matrix which is the result of the Mandelbrot set with a pixeldensity of pxd.
    """
    z = np.zeros_like(c)
    output = np.zeros_like(c, dtype=float)
    mask = np.ones_like(c, dtype=bool)

    for ii in range(maxite):
        z[mask] = z[mask]**2 + c[mask]
        mask = abs(z) <= 2
        output[mask] = ii / maxite

    return output


def complex_matrix(xmin, xmax, ymin, ymax, pxd):
    """
    Creates a complex matrix, which will be used to calculate the Mandelbrot set.

    Parameters
    ----------
    xmin : integer
           Integer, used to set the minimum value of the X axis.
    xmax : integer
           Integer, used to set the maximum value of the X axis.
    ymin : integer
           Integer, used to set the minimum value of the Y axis.
    ymax : integer
           Integer, used to set the maximum value of the Y axis.
    pxd : integer
          Integer, used to set the pixel density, which is the same as the size of the matrix.

    Returns
    -------
    re[np.newaxis, :] + im[:, np.newaxis]*1j : complex128
                                               A complex matrix, used to compute the Mandelbrot set.
    """
    re = np.linspace(xmin, xmax, pxd)
    im = np.linspace(ymin, ymax, pxd)
    return re[np.newaxis, :] + im[:, np.newaxis]*1j

if __name__ == "__main__":
    pxd = 5000

    beginTimer = time.perf_counter()
    print("start: ", time.perf_counter()-beginTimer)

    c = complex_matrix(-2, 1, -1.5, 1.5, pxd)

    print("matrix done: ", time.perf_counter()-beginTimer)
    plt.imshow(Mandelbrot(c, maxite=45), cmap="hot", extent=[-2, 1, -1.5, 1.5])
    print("calculations done: ", time.perf_counter()-beginTimer)
    plt.show()
