import itertools

import numpy as np
import matplotlib.pyplot as plt

from plot_results import plot_results
from tools import plot_ellipse

room_size = 10
room_shape = [room_size, room_size]
step_size = room_size / 10.0

ap_coordinates = np.array([[0, 0], [0, room_shape[1]], room_shape, [room_shape[0], 0]])
transmitters_coordinates = np.array([[i / 2.0 for i in room_shape], [2, 2]])
gamma = 2
receiver_gains = transmitters_gains = (0, 1)  # mu and variance
a0_muvar = (51, .1)
variance = 2

J, I, N = len(ap_coordinates), len(transmitters_coordinates), 20
n_configuration = 100


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
theta = np.array(list(itertools.product(*[np.arange(-i, 2 * i, step_size) for i in room_shape])))
d_theta = np.linalg.norm(ap_coordinates[:, None, :] - theta[None, :, :], axis=2)
Yb_theta = Gar[:, :, None] - 10 * gamma * np.log(d_theta[:, None, :]) / np.log(10)
error = Y[:, :, :, :, None] - Yb_theta[None, :, :, None, :]

# On calcule le likelihood
error = np.moveaxis(np.moveaxis(error, 4, 1), 3, 2).reshape((-1, I, J * N))
receiver, _ = np.array(list(itertools.product(range(J), range(N)))).T
cov = (receiver[:, None] == receiver[None, :]) * receiver_gains[1]
cov = cov + a0_muvar[1] + transmitters_gains[1]
cov += np.eye(J * N) * variance

covi = np.linalg.inv(cov)

likelihood_list = []
for i in range(I):
    likelihood = [e.T.dot(covi).dot(e) for e in error[:, i, :]]
    likelihood = np.reshape(likelihood, (n_configuration, len(theta)))
    likelihood_list.append(likelihood)
likelihood = np.array(likelihood_list)
argmin = np.nanargmin(likelihood, axis=2).T
estimate = theta[argmin, :]

estimate_error = estimate - transmitters_coordinates[None, :, :]

# Compute CRLB
K = 10 * gamma / np.log(10)
dYb = K * (vector / d[:, :, None] ** 2)
dYb = np.repeat(np.moveaxis(dYb, 1, 0), N, axis=1)
fisher = (i.T.dot(covi).dot(i) for i in dYb)
crlb = np.array([np.linalg.inv(i) for i in fisher])

if __name__ == '__main__':
    # Plotting results
    plot_results(transmitters_coordinates,estimate,ap_coordinates,crlb)
    plt.show()
