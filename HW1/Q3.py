import solver as mysolver
import numpy as np
import matplotlib.pyplot as plt

def force_oscillator(t, y, K, M, L, F0, WF):
    """
    The derivate function for a forced oscillator
    In this example, we set

    y[0] = x
    y[1] = v

    yderive[0] = x' = v
    yderive[1] = v' = a

    :param t: the time
    :param y: the initial condition y
    :param K: the spring constant
    :param M: the mass of the oscillator
    :param L: the length of the spring
    :param F0: the amplitude of the external force
    :param WF: the frequency of the external force

    """

    omega0 = np.sqrt(K/M)
    gamma = L/2/M
    A = F0/M
    yderive = np.array([y[1], -omega0**2*y[0]-2*gamma*y[1]+A*np.cos(WF*t)])
 
    return yderive

# Prepare the input arguments part(a)
t_span = (0, 50)
t_eval = np.linspace(40, 50, 1001)
A = 1
K = 1
M = 1
L = [0.01, 0.1, 0.3]
F0 = 0.5
y0 = np.array([0, A])

# use the IVP solver
WF = np.linspace(0.5, 1.5, 21)
avgx = np.zeros(len(WF))

for l in L:
    Wmax = 0
    for i, W in enumerate(WF):
        args = (K, M, l, F0, W)
        sol = mysolver.solve_ivp(force_oscillator, t_span, y0, "RK4", t_eval, args=args)
        avgx[i] = np.mean(np.abs(sol[0]))
        if avgx[i] > avgx[Wmax]:
            Wmax = i
    print(WF[Wmax])
    plt.plot(WF, avgx, '--o', label=f'$\lambda$ = {l}')
plt.xlabel('Driving force frequency [rad/s]')
plt.ylabel('Average amplitude [m]')
plt.legend()
plt.show()