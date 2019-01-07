import itertools

import numpy as np
import matplotlib.pyplot as plt

from tools import plot_ellipse

room_size = 10
room_shape = [room_size, room_size]
step_size = room_size / 100.0

ap_coordinates = np.array([[0, 0], [0, room_shape[1]], room_shape, [room_shape[0], 0]])
transmitters_coordinates = np.array([[i / 2.0 for i in room_shape], [0, room_shape[1] / 2.0]])
gamma = 2
a0_muvar = (51, .1)
receiver_gains = transmitters_gains = (10, 1)  # mu and variance
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
d = np.linalg.norm(ap_coordinates[:, None, :] - transmitters_coordinates[None, :, :], axis=2)
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

# Plotting results
axe = plt.figure().add_subplot(111)

# Drawing covariance ellipses
mean_cov = [(np.mean(i, axis=0), np.cov(i.T)) for i in np.moveaxis(estimate, 1, 0)]
for i in mean_cov:
    plot_ellipse(i[1], axe, i[0])

for t in range(I):
    axe.scatter(*estimate[:, t, :].T, label="Transmitter %i" % t,alpha=.2)
# axe.set_xlim(0,room_shape[0])
# axe.set_ylim(0,room_shape[1])
axe.legend()
plt.show()
pass
