import matplotlib.pyplot as plt
import time
import dask.array as da
import dask
import numpy as np


def Mandelbrot(c, maxite):
    image = np.empty(c.shape)
    z = np.zeros_like(c)
    for i in range(maxite):
        mask = (np.abs((z)) < 2)
        image = np.where(mask, i, image)
        z = np.where(mask, np.square(z) + c, z)
    return image


def complex_matrix(xmin, xmax, ymin, ymax, pxd, chunksize=1000):
    re = da.linspace(xmin, xmax, pxd, chunks=chunksize)
    im = da.linspace(ymin, ymax, pxd, chunks=chunksize)
    c = re[:, None] + im[None, :]*1j
    image = da.map_blocks(Mandelbrot, c, maxite=45)
    #dask.visualize(image)
    return image.compute()


if __name__ == "__main__":
    pxd=5000
    chunks = 500

    while chunks <= pxd:
        beginTimer = time.perf_counter()
        c = complex_matrix(-2, 0.5, -1.5, 1.5, pxd, chunks)
        print("calculations for", chunks, "chunks: ", time.perf_counter()-beginTimer)
        chunks += 100
    
    plt.imshow(c, cmap="hot", extent=[-2, 1, -1.5, 1.5])
    
    plt.show()