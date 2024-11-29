import matplotlib.pyplot as plt
import time
import numpy as np
import pyopencl as cl

def MandelCL(device, work_group, pxd):
    #------------------------EDIT---------------------------------------
    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx)
    #------------------------EDIT---------------------------------------
        
    c = complex_matrix(-2, 1, -1.5, 1.5, pxd)

    C_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=c.astype(np.complex64))
    O_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, size=c.nbytes)

    kernel_code = """
    #include <pyopencl-complex.h>

    __kernel void Mandelbrot(__global const cfloat_t *c, __global float *output){

        int i = get_global_id(0);
        int j = get_global_id(1);
        int pxd = get_global_size(0);
        int maxite = 100;
        cfloat_t sum = cfloat_new(0,0);
        

        for (int k = 0; k < maxite; k++) {
            sum = cfloat_add(cfloat_mul(sum,sum),c[i*pxd + j]); 
            if (cfloat_abs(sum) >= 2) {
                output[i*pxd + j] = (float) k/maxite;
                break;
            }           
        }
        if (cfloat_abs(sum) < 2) {
            output[i*pxd + j] = 1.0;
        }
    }
    """
    prg = cl.Program(ctx, kernel_code).build()
    Mandelbrot = prg.Mandelbrot
    
    Mandelbrot(queue, c.shape, work_group, C_buf, O_buf)
    
    output = np.zeros((pxd, pxd), dtype=np.float32)
    cl.enqueue_copy(queue, output, O_buf)

    return output




def complex_matrix(xmin, xmax, ymin, ymax, pxd):
    re = np.linspace(xmin, xmax, pxd)
    im = np.linspace(ymin, ymax, pxd)
    return re[np.newaxis, :] + im[:, np.newaxis]*1j


if __name__ == "__main__":
    platforms = cl.get_platforms()
    [NVIDIA, INTEL] = [platform.get_devices()[0] for platform in platforms]
    pixel_density = [1000, 2000, 6000, 8000, 10000, 12000] 
    work_groups = [(1, 1), (4, 4), (8, 8), (10, 10), (20, 20), (25, 25)]
    INTEL_results = []
    NVIDIA_results = []

    for device in [NVIDIA, INTEL]:
        print("Calculations for", device)
        for pxd in pixel_density:
            for work_group in work_groups:
                try:
                    time_start = time.perf_counter()
                    MandelCL(device, work_group, pxd)
                    time_end = time.perf_counter()
                    print("for:", pxd, work_group, "\n", "result:", time_end - time_start)
                    if device == NVIDIA:
                        NVIDIA_results.append(time_end - time_start)
                    if device == INTEL:
                        INTEL_results.append(time_end - time_start)
                except:
                    if device == NVIDIA:
                        NVIDIA_results.append(None)
                    if device == INTEL:
                        INTEL_results.append(None)
                    pass

for graph in range(len(pixel_density)):
    print(graph)
    x = range(len(work_groups))
    print(x)
    plt.plot(x, NVIDIA_results[graph*x[-1]+graph:(graph+1)*x[-1]+graph+1])
    plt.plot(x, INTEL_results[graph*x[-1]+graph:(graph+1)*x[-1]+graph+1])
    plt.ylabel('Time')
    plt.xlabel('Worker index')
    plt.title(pixel_density[graph])
    plt.show()
    

