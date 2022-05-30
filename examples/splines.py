import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    xr = np.linspace(0.0, 1.0, 100)

    qgl_1 = 1 - 3 * xr ** 2 + 2 * xr ** 3
    qgl_2 = (xr ** 3 - 2 * xr ** 2 + xr) * 1.
    qgl_3 = (- xr ** 3 + xr ** 2) * 1.
    qgl_4 = -2 * xr ** 3 + 3 * xr ** 2

    squad_1 = (1 - xr) * (1 - 2 * xr * (1 - xr))
    squad_2 = (1 - xr) * 2 * xr * (1 - xr)
    squad_3 = xr * 2 * xr * (1 - xr)
    squad_4 = xr * (1 - 2 * xr * (1 - xr))

    bezier_1 = (1 - xr) ** 3
    bezier_2 = 3 * (1 - xr) ** 2 * xr
    bezier_3 = 3 * (1 - xr) * xr ** 2
    bezier_4 = xr ** 3

    plt.subplot(131)
    plt.plot(xr, qgl_1)
    plt.plot(xr, qgl_2)
    plt.plot(xr, qgl_3)
    plt.plot(xr, qgl_4)
    plt.plot(xr, qgl_1 + qgl_2 + qgl_3 + qgl_4)
    plt.title('QGL T, absolutely nonsense')

    plt.subplot(132)
    plt.plot(xr, squad_1)
    plt.plot(xr, squad_2)
    plt.plot(xr, squad_3)
    plt.plot(xr, squad_4)
    plt.plot(xr, squad_1 + squad_2 + squad_3 + squad_4)
    plt.title('QGL Squad, only for quaternion')

    plt.subplot(133)
    plt.plot(xr, bezier_1)
    plt.plot(xr, bezier_2)
    plt.plot(xr, bezier_3)
    plt.plot(xr, bezier_4)
    plt.plot(xr, bezier_1 + bezier_2 + bezier_3 + bezier_4)
    plt.title('Linear bezier')

    plt.show()
