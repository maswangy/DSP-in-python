import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def dtw(x, y, dist='euclidean'):
    """
    Function for Vanilla Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func or str dist: distance used as cost measure.
        it accepts any customized functions (ex lambda).
        Or you can use the strings of metric descirbed in scipy.spatial.distance.cdist
    """
    # sanity check
    r, c = len(x), len(y)
    assert r and c, "the input cannot be empty array"

    if np.ndim(x) == 1:
        x = np.array(x)[:, np.newaxis]
    if np.ndim(y) == 1:
        y = np.array(y)[:, np.newaxis]

    # initialization
    step = [(-1, -1), (-1, 0), (0, -1)]
    C = np.zeros((r + 1, c + 1))
    C[:, 0] = C[0, :] = np.inf

    # assign cost
    if isinstance(dist, str):
        C[1:, 1:] = cdist(x, y, dist)
    else:
        for i in range(1, r+1):
            for j in range(1, c+1):
                C[i, j] = dist(x[i-1], y[j-1])
    cost = C[1:, 1:].copy()

    # DP body
    for i in range(1, r+1):
        for j in range(1, c+1):
            if j == i == 1:
                continue
            C[i, j] += min([C[i+s[0], j+s[1]] for s in step])

    dtw_dist = C[-1, -1]/(r+c)
    acc_cost = C[1:, 1:]

    # trace back
    path = _traceback(C[1:, 1:], step)
    return dtw_dist, cost, acc_cost, path


def _traceback(C, step):
    i, j = np.array(C.shape) - 1
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        candidates = []
        for s in step:
            try:
                candidates += [C[i+s[0], j+s[1]]]
            except IndexError:
                continue
        idx = np.argmin(candidates)

        i += step[idx][0]
        j += step[idx][1]
        p.insert(0, i)
        q.insert(0, j)
    return np.array(p), np.array(q)


def plot_grid(acc, path, filename='grid.png'):
    plt.figure()
    plt.imshow(acc.T, origin='lower', cmap='gray')
    plt.plot(path[0], path[1], 'w')
    plt.title('colormap and optimal path')
    plt.savefig(filename)
    plt.close()


def plot_alignment(sig1, sig2, path, filename='alignment.png'):
    plt.figure()
    fig, ax = plt.subplots()
    x1 = np.linspace(0.0, len(sig1))
    x2 = np.linspace(0.0, len(sig2))
    sig1_y = 5
    sig2_y = 0
    y1 = [sig1_y] * len(x1)
    y2 = [sig2_y] * len(x2)
    plt.plot(x1, y1, 'b-')
    plt.plot(x2, y2, 'g-')
    plt.yticks([sig1_y, sig2_y])
    for i in range(0, len(path[0])):
        plt.plot([path[0][i], path[1][i]], [sig1_y, sig2_y], 'r*--')
    ax.set_yticklabels(['sig1', 'sig2'])
    plt.title('signal alignment')
    plt.savefig('alignment.png')
    plt.close()


if __name__ == '__main__':

    sig1 = [71, 73, 75, 80, 80, 80, 78, 76,
            75, 73, 71, 71, 71, 73, 75, 76,
            76, 68, 76, 76, 75, 73, 71, 70,
            70, 69, 68, 68, 72, 74, 78, 79,
            80, 80, 78]
    sig2 = [69, 69, 73, 75, 79, 80, 79, 78,
            76, 73, 72, 71, 70, 70, 69, 69,
            69, 71, 73, 75, 76, 76, 76, 76,
            76, 75, 73, 71, 70, 70, 71, 73,
            75, 80, 80, 80, 78]
    dist, cost_matrix, acc_cost_matrix, path = dtw(sig1, sig2)
    plot_grid(acc_cost_matrix, path)
    plot_alignment(sig1, sig2, path)
