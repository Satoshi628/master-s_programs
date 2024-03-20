import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def cubic_spline(track1, track2):
    #track => [:, (t, x, y)] ndarray
    track = np.concatenate([track1, track2], axis=0)
    t = track[:, 0]
    xy = track[:, 1:]
    CS = CubicSpline(t, xy, axis=0)

    #2つの線を補間した部分
    start_t = track1[-1, 0]
    end_t = track2[0, 0]

    delta = 0.001
    times = np.arange(start_t, end_t, delta)
    line = CS(times)
    dxy_dt = np.gradient(line, times, axis=0)
    
    dxy_dt_2 = np.gradient(dxy_dt, times, axis=0)

    R = ((dxy_dt ** 2).sum(axis=-1)) ** 1.5 / np.clip(np.abs(dxy_dt[:, 0] * dxy_dt_2[:, 1] - dxy_dt[:, 1] * dxy_dt_2[:, 0]), 1e-7, None)
    
    return line, np.min(R)


track1 = np.array([[1, 2, 3, 4, 5, 6],[1, 2, 3, 4, 5, 6], [2, 3, 5, 6, 8, 9]]).T
track2 = np.array([[9, 10, 11, 12, 13, 14],[9, 10, 11, 12, 13, 14], [6, 4, 3, 2, 1, 0]]).T
# track2 = np.array([[9, 10, 11, 12, 13, 14], [9, 10, 11, 12, 13, 14], [12, 14, 15, 16, 18, 19]]).T


plt.plot(track1[:, 1], track1[:, 2], 'b')
plt.plot(track2[:, 1], track2[:, 2], 'r')
line, R = cubic_spline(track1, track2)
plt.plot(line[:, 0], line[:, 1], 'k')
print(R)

plt.savefig("spline.png")