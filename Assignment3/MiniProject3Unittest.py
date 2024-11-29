import unittest
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


class TestMandelbrot(unittest.TestCase):
    def test_mandelbrot_set(self):
        pxd = 5
        c = complex_matrix(-2, 1, -1.5, 1.5, pxd)
        output = np.zeros((pxd, pxd))
        expected_output = np.array([
            [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000],
            [0.00000000, 0.02222222, 0.08888889, 0.06666667, 0.00000000],
            [0.97777778, 0.97777778, 0.97777778, 0.97777778, 0.02222222],
            [0.00000000, 0.02222222, 0.08888889, 0.06666667, 0.00000000],
            [0.00000000, 0.00000000, 0.00000000, 0.00000000, 0.00000000]
        ])
        output = Mandelbrot(output, c, maxite=45)
        self.assertTrue(np.allclose(output, expected_output, rtol=1e-4))

    def test_complex_matrix(self):
        expected_output = np.array([
            [-2.00-1.50j, -1.25-1.50j, -0.50-1.50j, 0.25-1.50j, 1.00-1.50j],
            [-2.00-0.75j, -1.25-0.75j, -0.50-0.75j, 0.25-0.75j, 1.00-0.75j],
            [-2.00+0.00j, -1.25+0.00j, -0.50+0.00j, 0.25+0.00j, 1.00+0.00j],
            [-2.00+0.75j, -1.25+0.75j, -0.50+0.75j, 0.25+0.75j, 1.00+0.75j],
            [-2.00+1.50j, -1.25+1.50j, -0.50+1.50j, 0.25+1.50j, 1.00+1.50j]
            ])
        output = complex_matrix(-2, 1, -1.5, 1.5, pxd=5)
        self.assertTrue(np.allclose(output, expected_output, rtol=1e-4))

    def test_size(self):
        pxd = 5
        expected_output = True
        if complex_matrix(-2, 1, -1.5, 1.5, pxd).shape == Mandelbrot(np.zeros((pxd, pxd)), complex_matrix(-2, 1, -1.5, 1.5, pxd), maxite=45).shape: output = True 
        else: output = False
        self.assertTrue(output, expected_output)


if __name__ == '__main__':
    unittest.main()
