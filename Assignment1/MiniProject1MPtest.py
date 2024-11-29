import matplotlib.pyplot as plt
import time
import multiprocessing as mp


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


def create(row):
    mb_row = []
    for num in row:
        mb_row.append(Mandelbrot(num, maxite=45))
    return mb_row


if __name__ == "__main__":
    max_num_processes = mp.cpu_count()
    print("Num processes available: ", str(max_num_processes))
    
    chunk_amount = 1
    pxd = 200
    current_processes = 1

    while current_processes <= max_num_processes:
        pool = mp.Pool(current_processes)
        chunk_amount = 1
        while chunk_amount <= pxd:
            if pxd % chunk_amount == 0:
                beginTimer = time.perf_counter()
                #print("start: ", time.perf_counter()-beginTimer)
                c = complex_matrix(-2, 0.5, -1.5, 1.5, pxd)
                #print("matrix done: ", time.perf_counter()-beginTimer)
                
                mb= []
                chunk_size = int(pxd/chunk_amount)
                for chunk in range(chunk_amount):
                    results = [pool.apply_async(create, args=(row,)) for row in c[chunk*chunk_size:(chunk+1)*chunk_size]]
                    mb_row = [result.get() for result in results]
                    mb += mb_row
                
                
                print("calculations for ", current_processes, "processes with ", chunk_amount, "chunks: ", time.perf_counter()-beginTimer)
            chunk_amount += 1
        pool.close()
        current_processes += 1
        
    plt.imshow(mb, cmap="hot", extent=[-2, 1, -1.5, 1.5])
    plt.show()
