import numpy as np
import copy
import matplotlib.pyplot as plt
import pickle

def gauss_erup(n_x, n_y, x_0, y_0, width=5, amp=1):
    '''
    :param n_x: number of points in x direction
    :param n_y: number of points in y direction
    :param x_0: x coordinate of the center of the eruption
    :param y_0: y coordinate of the center of the eruption
    :param width: width of the eruption
    :param amp: amplitude of the eruption
    '''
    g_erup = np.zeros((n_x, n_y))
    n_x, up_x, center_x = x_0 - int(width / 2), x_0 + int(width / 2), x_0
    down_y, up_y, center_y = y_0 - int(width / 2), y_0 + int(width / 2), y_0
    for k in range(n_x, up_x + 1):
        for j in range(down_y, up_y + 1):
            g_erup[k, j] = amp * \
                np.exp(-(np.power((k - center_x), 2) +
                       np.power((j - center_y), 2)) / 2)
    return g_erup


def wave_eq(n_x, n_y, x_0, y_0, t_steps):
    '''
    :param n_x: number of points in x direction
    :param n_y: number of points in y direction
    :param t_steps: number of time steps
    '''
    c_0 = 1484
    c_0sq = np.power(c_0, 2)
    d_x, d_y = (np.pi / (n_x - 1), np.pi / (n_y - 1))
    d_t = 1 / (c_0 * np.sqrt(2 * (1 / (np.power(d_x, 2)) + 1 / (np.power(d_y, 2)))))
    u_curr = gauss_erup(n_x, n_y, x_0, y_0, int(n_x / 32), 1)
    u_prev = copy.copy(u_curr)
    u_next = np.zeros((n_x, n_y))
    for _ in range(2, t_steps + 1):
        u_next[1:n_x - 1, 1:n_y - 1] = \
            2 * u_curr[1:n_x - 1, 1:n_y - 1] - u_prev[1:n_x - 1, 1:n_y - 1] + \
            c_0sq * np.power(d_t, 2) * (
            (u_curr[2:n_x, 1:n_y - 1] - 2 * u_curr[1:n_x - 1, 1:n_y - 1] +
             u_curr[0:n_x - 2, 1:n_y - 1]) / (np.power(d_x, 2))
            + (u_curr[1:n_x - 1, 2:n_y] - 2 * u_curr[1:n_x - 1, 1:n_y - 1] +
               u_curr[1:n_x - 1, 0:n_y - 2]) / (np.power(d_y, 2)))
        u_prev = copy.copy(u_curr)
        u_curr = copy.copy(u_next)
    return u_next


def get_location(n_x, n_y, x_0, y_0, bins_x, bins_y):
    '''
    :param n_x: number of points in x direction
    :param n_y: number of points in y direction
    :param x_0: x coordinate of the center of the eruption
    :param y_0: y coordinate of the center of the eruption
    :param bins_x: number of bins in x direction
    :param bins_y: number of bins in y direction
    '''
    one_hot = np.zeros((bins_x * bins_y))
    coarse_idx = int(x_0 * bins_x / n_x) * bins_y + int(y_0 * bins_y / n_y)
    one_hot[coarse_idx] = 1
    return one_hot



if __name__ == '__main__':
    n_x, n_y, t_steps = 64, 64, 100
    data_list = []
    label_list = []
    grid_interval = 3

    for x_idx in range(10, n_x - 10, 1):
        for y_idx in range(10, n_y - 10, 1):

            u = wave_eq(n_x, n_y,
                        x_idx, y_idx, t_steps)

            data_list.append(u)
            label_list.append(np.argmax(get_location(n_x, n_y,x_idx, y_idx, grid_interval, grid_interval)))


            # plt.imshow(u)
            # plt.xticks([])
            # plt.yticks([])
            # plt.xlabel('')
            # plt.ylabel('')
            # plt.title('Location: ' +
            #           str(get_location(n_x, n_y,
            #                            x_idx, y_idx, grid_interval, grid_interval)))
            # plt.savefig('data_gen/x{}y{}_label{}.png'.format(x_idx, y_idx,
            #           np.argmax(get_location(n_x, n_y,
            #                            x_idx, y_idx, grid_interval, grid_interval))))
            # exit()

    dataset = {'inputs': data_list, 'labels': label_list}
    print (len(data_list), len(label_list))
    with open('dataset_wave_{}.pkl'.format(grid_interval**2), 'wb') as f:
        pickle.dump(dataset, f)