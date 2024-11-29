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
    c = complex_matrix(-2, 1, -1.5, 1.5, pxd)
    b8 = np.bool8(complex_matrix(-2, 1, -1.5, 1.5, pxd))
    int8 = np.int8(complex_matrix(-2, 1, -1.5, 1.5, pxd))
    intp = np.intp(complex_matrix(-2, 1, -1.5, 1.5, pxd))
    uint8 = np.uint8(complex_matrix(-2, 1, -1.5, 1.5, pxd))
    float16 = np.float16(complex_matrix(-2, 1, -1.5, 1.5, pxd))
    c64 = np.csingle(complex_matrix(-2, 1, -1.5, 1.5, pxd))
    c128 = np.cdouble(complex_matrix(-2, 1, -1.5, 1.5, pxd))
    
    output = np.zeros((pxd, pxd))
    print("matrix done: ", time.perf_counter()-beginTimer)

    beginTimer = time.perf_counter()
    Mbregular = Mandelbrot(output, c, maxite=100)
    print("calculations for regular done: ", time.perf_counter()-beginTimer)
    
    beginTimer = time.perf_counter()
    Mbb8 = Mandelbrot(output, b8, maxite=100)
    print("calculations for b8 done: ", time.perf_counter()-beginTimer)

    beginTimer = time.perf_counter()
    Mbint8 = Mandelbrot(output, int8, maxite=100)
    print("calculations for int8 done: ", time.perf_counter()-beginTimer)

    beginTimer = time.perf_counter()
    Mbintp = Mandelbrot(output, intp, maxite=100)
    print("calculations for intp done: ", time.perf_counter()-beginTimer)

    beginTimer = time.perf_counter()
    Mbuint8 = Mandelbrot(output, uint8, maxite=100)
    print("calculations for uint8 done: ", time.perf_counter()-beginTimer)

    beginTimer = time.perf_counter()
    Mbfloat16 = Mandelbrot(output, float16, maxite=100)
    print("calculations for float16 done: ", time.perf_counter()-beginTimer)

    beginTimer = time.perf_counter()
    Mb64 = Mandelbrot(output, c64, maxite=100)
    print("calculations for c64 done: ", time.perf_counter()-beginTimer)

    beginTimer = time.perf_counter()
    Mb128 = Mandelbrot(output, c128, maxite=100)
    print("calculations for c128 done: ", time.perf_counter()-beginTimer)

    
    arrays = [Mbregular, Mbb8, Mbint8, Mbintp, Mbuint8, Mbfloat16, Mb64, Mb128]
    names = ['regular', 'b8', 'int8', 'intp', 'uint8', 'float16', 'c64', 'c128']
    counter = 0

    for i in range(len(arrays)):
        for j in range(i+1, len(arrays)):
            if arrays[i].all() == arrays[j].all():
                print(f"{names[i]} = {names[j]}")
                counter += 1

    print(f"In total, {counter} out of {len(names)*(len(names)-1)//2} comparisons are the same")


    fig = plt.figure(figsize=(12, 8))

    # create subplots and add images and titles to them
    plt.subplot(241)
    plt.imshow(Mbb8, cmap="hot", extent=[-2, 1, -1.5, 1.5])
    plt.title("Mbb8")

    plt.subplot(242)
    plt.imshow(Mbint8, cmap="hot", extent=[-2, 1, -1.5, 1.5])
    plt.title("Mbint8")

    plt.subplot(243)
    plt.imshow(Mbintp, cmap="hot", extent=[-2, 1, -1.5, 1.5])
    plt.title("Mbintp")

    plt.subplot(244)
    plt.imshow(Mbuint8, cmap="hot", extent=[-2, 1, -1.5, 1.5])
    plt.title("Mbuint8")

    plt.subplot(245)
    plt.imshow(Mbfloat16, cmap="hot", extent=[-2, 1, -1.5, 1.5])
    plt.title("Mbfloat16")

    plt.subplot(246)
    plt.imshow(Mb64, cmap="hot", extent=[-2, 1, -1.5, 1.5])
    plt.title("Mb64")

    plt.subplot(247)
    plt.imshow(Mb128, cmap="hot", extent=[-2, 1, -1.5, 1.5])
    plt.title("Mb128")

    plt.subplot(248)
    plt.imshow(Mbregular, cmap="hot", extent=[-2, 1, -1.5, 1.5])
    plt.title("Regular")

    plt.tight_layout()

    plt.show()
