import solver as mysolver
from Q1 import damp_oscillator
import numpy as np
import matplotlib.pyplot as plt


# Prepare the input arguments part(a)
# A = 1cm, omega0 = 1rad/s, gamma = 0.2s^-1, phi = -pi/2rad
t_span = (0, 20)
t_eval = np.linspace(0, 20, 2001)
omega0 = 1
gamma = 0.2
A = 1
phi = -np.pi/2
args = (omega0, gamma)
y0 = np.array([0, A])

# use the IVP solver
sol = mysolver.solve_ivp(damp_oscillator, t_span, y0, "RK4", t_eval, args=args)

# visualize the results
M = 1
K = 1
energy = 0.5*M*sol[1]**2 + 0.5*K*sol[0]**2
energy_loss = np.zeros(len(energy)-1)
for i in range(len(energy)-1):
    energy_loss[i] = (energy[i+1]-energy[i])/(t_eval[1]-t_eval[0])

plt.figure(1)
plt.plot(t_eval, energy)
plt.title(r'Total energy of Damped Oscillator versus time')
plt.xlabel("Time [s]")
plt.ylabel("Total Energy [J]")

plt.figure(2)
plt.plot(t_eval[:-1], energy_loss)
plt.title(r'Energy loss rate of Damped Oscillator versus time')
plt.xlabel("Time [s]")
plt.ylabel("Energy Loss Rate [W]")
plt.show()