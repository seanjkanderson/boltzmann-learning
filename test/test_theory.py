import unittest
import numpy

import sys

sys.path.append("src")


class Test010Theory(unittest.TestCase):
    """Test010Theory Class to test theoretical results"""

    def test010_reverse_Jensen(self):
        print("test010_reverse_Jensen")

        def phi(x):
            return numpy.exp(-x)

        def dotphi(x):
            return -numpy.exp(-x)

        for xmin in [-2, -0.2, 0.2, 0.01]:
            for dx in [0.01, 0.1, 1, 0.02]:
                xmax = xmin + dx
                for x0 in [xmin / 2, xmin, (xmax + xmin) / 2, xmax]:
                    print(
                        "xmin = {0:8.3f}, xmax = {1:8.3f}, x0 = {2:8.3f},".format(
                            xmin, xmax, x0
                        )
                    )
                    delta = (phi(xmax) - phi(xmin)) / (xmax - xmin) / dotphi(x0)
                    delta0 = (phi(xmin) - phi(x0)) / dotphi(x0) + x0 - delta * xmin

                    N = 1000
                    xs = numpy.linspace(xmin, xmax, num=N)
                    xs = [xmin, xmax]
                    phiX = [phi(x) for x in xs]
                    Ex = sum(xs) / len(xs)
                    EphiX = sum(phiX) / len(xs)
                    bound = phi(delta * Ex + delta0)
                    print(
                        "  delta = {:10.6f} delta0 = {:10.6f} E[x] = {:10.6f} E[phi(x)] = {:10.6f} phi(dlt E[x]+dlt0) = {:13.6f} gap = {:13.6f}".format(
                            delta, delta0, Ex, EphiX, bound, bound - EphiX
                        )
                    )

                    self.assertLess(EphiX, bound)


if __name__ == "__main__":
    unittest.main(verbosity=2)
