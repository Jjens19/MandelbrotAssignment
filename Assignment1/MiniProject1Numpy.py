import matplotlib.pyplot as plt
import time
import numpy as np

def Mandelbrot(c, maxite):
    z = np.zeros_like(c)
    output = np.zeros(c.shape)
    
    for ii in range(maxite):
        mask = (np.abs(z) <= 2)
        z[mask] = z[mask]**2 + c[mask]
        output[mask] = ii/maxite
    
    return output

    
def complex_matrix(xmin, xmax, ymin, ymax, pxd):
    re = np.linspace(xmin, xmax, int((xmax-xmin)*pxd))
    im = np.linspace(ymin, ymax, int((ymax-ymin)*pxd))
    return re[np.newaxis, :] + im[:, np.newaxis]*1j


if __name__ == "__main__":
    beginTimer = time.perf_counter()
    print("start: ", time.perf_counter()-beginTimer)
    c = complex_matrix(-2, 1, -1.5, 1.5, pxd=5000)
    print("matrix done: ", time.perf_counter()-beginTimer)
    plt.imshow(Mandelbrot(c, maxite=45), cmap="hot", extent=[-2, 1, -1.5, 1.5])
    print("calculations done: ", time.perf_counter()-beginTimer)
    plt.show()
