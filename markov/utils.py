import numpy as np
from scipy.stats import multivariate_normal


def creer_trajectoire(F, Q, N, x_init):
    dim_x = x_init.shape[0]
    x = np.zeros((N, dim_x))
    x[0] = x_init
    for i in range(1, N):
        x[i] = F@x[i-1] + multivariate_normal.rvs(np.zeros((dim_x,)), Q)
    return x


def creer_observations(H, R, x):
    N = x.shape[0]
    dim_y = H.shape[0]
    y = np.zeros((x.shape[0], dim_y))
    for i in range(N):
        y[i] = H@x[i] + multivariate_normal.rvs(np.zeros((dim_y,)), R)
    return y


def filtre_de_kalman_iter(F, Q, H, R, y_n, x_kalm_prec, P_kalm_prec):
    x_kalm_predict = F@x_kalm_prec
    P_kalm_predict = Q + F@P_kalm_prec@F.transpose()

    y_n_moy = y_n - H@x_kalm_predict
    S = H@P_kalm_predict@H.transpose() + R
    K = P_kalm_predict@H.transpose()@np.linalg.inv(S)

    x_kalm_n = x_kalm_predict + K@y_n_moy
    P_kalm_n = P_kalm_predict - K@H@ P_kalm_predict
    return x_kalm_n, P_kalm_n


def filtre_de_kalman(F, Q, H, R, y, x_init):
    N = y.shape[0]
    dim_x = x_init.shape[0]
    x_est = np.zeros((N, dim_x))
    P_est = np.zeros((N, dim_x, dim_x))
    x_est[0] = x_init
    for i in range(1, N):
        x_est[i], P_est[i] = filtre_de_kalman_iter(F, Q, H, R, y[i], x_est[i-1], P_est[i-1])
    return x_est, P_est


def reg_lin_cache(x, y, reg = 10**-10):
    x_prev = x[:-1]
    x_post = x[1:]
    F_aux = x_prev.transpose() @ x_prev
    F = (np.linalg.inv(
        F_aux + reg * np.eye(F_aux.shape[0])) @ x_prev.transpose() @ x_post).transpose()
    Q = (1 / x_post.shape[0]) * np.einsum('...i,...j->...ij',
                                                                        x_post - np.einsum('ij, ...j->...i',
                                                                                                F, x_prev),
                                                                        x_post - np.einsum('ij, ...j->...i',
                                                                                                F,
                                                                                                x_prev)).sum(
        axis=0)

    mask_nan = np.logical_not(np.any(np.isnan(y), axis=1))
    x = x[mask_nan]
    y = y[mask_nan]
    H_aux = x.transpose() @ x
    H = (np.linalg.inv(H_aux + reg * np.eye(H_aux.shape[0])) @ x.transpose() @ y).transpose()
    R = (1 / y.shape[0]) * np.einsum('...i,...j->...ij', y - np.einsum('ij, ...j->...i', H, x),
                                          y - np.einsum('ij, ...j->...i', H, x)).sum(
            axis=0)
    return F, Q, H, R


def reg_lin_couple(x, y, reg = 10**-10):
    dim_x = x.shape[-1]
    dim_y = y.shape[-1]

    detect_nan = np.where(np.any(np.isnan(y), axis=1))[0].tolist()
    detect_nan = [-1] + detect_nan + [x.shape[0]]

    data_all = np.concatenate((x, y), axis=-1)
    data_all_prev = np.concatenate([data_all[(detect_nan[i]+1):(detect_nan[i+1]-1)] for i in range(len(detect_nan)-1)], axis=0)
    data_all_post = np.concatenate([data_all[(detect_nan[i] + 2):detect_nan[i + 1]] for i in range(len(detect_nan)-1)], axis=0)
    A_aux = data_all_prev.transpose() @ data_all_prev
    A = (np.linalg.inv(
        A_aux + reg * np.eye(A_aux.shape[0])) @ data_all_prev.transpose() @ data_all_post).transpose()
    sigma_A = np.linalg.cholesky((1 / data_all_post.shape[0]) * np.einsum('...i,...j->...ij',
                                                                          data_all_post - np.einsum(
                                                                              'ij, ...j->...i', A, data_all_prev),
                                                                          data_all_post - np.einsum(
                                                                              'ij, ...j->...i', A,
                                                                              data_all_prev)).sum(
        axis=0) + reg * np.eye(dim_x + dim_y))
    return A, sigma_A


