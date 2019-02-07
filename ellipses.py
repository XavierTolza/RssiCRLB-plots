import itertools
import os

import dill
import matplotlib.pyplot as plt
import numpy as np
from bokeh.io import show
from tqdm import tqdm

from plot_results import plot_results

cache_file = "/tmp/cache.pkl"
if os.path.isfile(cache_file):
    with open(cache_file,"rb") as fp:
        transmitters_coordinates, estimate, ap_coordinates, crlb = dill.load(fp)

else:
    room_shape = [11, 6]
    step_size = np.min(room_shape) / 100.0

    ap_coordinates = np.array([[0, 0], [0, room_shape[1]], room_shape, [room_shape[0], 0]])
    transmitters_coordinates = np.array([[i / 2.0 for i in room_shape], [2, 2]])
    gamma = 2
    receiver_gains = transmitters_gains = (0, 1)  # mu and variance
    a0_muvar = (51, .1)
    variance = 2

    J, I, N = len(ap_coordinates), len(transmitters_coordinates), 20
    n_configurations = [10]*50
    estimates = []
    for n_configuration in tqdm(n_configurations):
        (G, noise_G), (a0, noise_a0), (R, noise_R) = [
            (np.array([i[0]] * j), np.random.normal(0, np.sqrt(i[1]), (n_configuration, j)))
            for i, j in
            zip([transmitters_gains, a0_muvar, receiver_gains], (J, 1, I))]
        noise = np.random.normal(0, np.sqrt(variance), (n_configuration, J, I, N))
        noise = noise_G[:, :, None, None] - noise_R[:, None, :, None] + noise_a0[:, 0, None, None, None] + noise

        Gar = G[:, None] + a0[0] - R[None, :]
        vector = ap_coordinates[:, None, :] - transmitters_coordinates[None, :, :]
        d = np.linalg.norm(vector, axis=2)
        Yb = Gar - 10 * gamma * np.log(d) / np.log(10)
        Y = Yb[None, :, :, None] + noise

        # Calcul des estimes
        itertools_input = [np.arange(-i, 2 * i, step_size) for i in room_shape]
        theta = np.array(list(itertools.product(*itertools_input)))
        theta_len = np.prod([len(i) for i in itertools_input])


        dtheta = np.linalg.norm(ap_coordinates[:, None, :] - theta[None, :, :], axis=2)
        Ybtheta = Gar[:, :, None] - 10 * gamma * np.log(dtheta[:, None, :]) / np.log(10)
        error = Y[:, :, :, :, None] - Ybtheta[None, :, :, None, :]

        # On calcule le likelihood
        error = np.moveaxis(np.moveaxis(error, 4, 1), 3, 2).reshape((-1, I, J * N))
        receiver, _ = np.array(list(itertools.product(range(J), range(N)))).T
        cov = (receiver[:, None] == receiver[None, :]) * receiver_gains[1]
        cov = cov + a0_muvar[1] + transmitters_gains[1]
        cov = cov + np.eye(J * N) * variance

        covi = np.linalg.inv(cov)

        likelihood_list = []
        for i in range(I):
            likelihood = [e.T.dot(covi).dot(e) for e in error[:, i, :]]
            likelihood = np.reshape(likelihood, (n_configuration, theta_len))
            likelihood_list.append(likelihood)
        likelihood = np.array(likelihood_list)
        argmin = np.nanargmin(likelihood, axis=2).T
        estimates.append(theta[argmin, :])

    estimate = np.concatenate(estimates,axis=0)
    estimate_error = estimate - transmitters_coordinates[None, :, :]

    # Compute CRLB
    K = 10 * gamma / np.log(10)
    dYb = K * (vector / d[:, :, None] ** 2)
    dYb = np.repeat(np.moveaxis(dYb, 1, 0), N, axis=1)
    fisher = (i.T.dot(covi).dot(i) for i in dYb)
    crlb = np.array([np.linalg.inv(i) for i in fisher])
    with open(cache_file,"wb") as fp:
        dill.dump((transmitters_coordinates, estimate, ap_coordinates, crlb),fp)


if __name__ == '__main__':
    # Plotting results
    p = plot_results(transmitters_coordinates, estimate, ap_coordinates, crlb)
    show(p)
