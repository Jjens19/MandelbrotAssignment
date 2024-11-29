import matplotlib.pyplot as plt
import time


def Mandelbrot(c, maxite):
    z = 0
    for ii in range(maxite):
        z = z**2+c
        if abs(z) > 2:
            return ii/maxite
    return 1

    
def complex_matrix(xmin, xmax, ymin, ymax, pxd):
    re = []
    im = []
    stepx = (xmax - xmin) / pxd
    stepy = (ymax - ymin) / pxd
    for i in range(pxd):
        re.append(xmin + i * stepx)
        im.append(ymin + i * stepy)
    return [[r + i * 1j for r in re] for i in im]


def create(c):
    mb = []
    for row in c:
        mb_row = []
        for num in row:
            mb_row.append(Mandelbrot(num, maxite=45))
        mb.append(mb_row)
    return mb


if __name__ == "__main__":
    beginTimer = time.perf_counter()
    print("start: ", time.perf_counter()-beginTimer)
    c = complex_matrix(-2, 0.5, -1.5, 1.5, pxd=5000)
    print("matrix done: ", time.perf_counter()-beginTimer)
    mb = create(c)
    print("calculations done: ", time.perf_counter()-beginTimer)
    plt.imshow(mb, cmap="hot", extent=[-2, 1, -1.5, 1.5])
    plt.show()
