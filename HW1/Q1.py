import solver as mysolver
import numpy as np
import matplotlib.pyplot as plt

def damp_oscillator(t, y, omega0, gamma):
    """
    The derivate function for a damped oscillator
    In this example, we set

    y[0] = x
    y[1] = v

    yderive[0] = x' = v
    yderive[1] = v' = a

    :param t: the time
    :param y: the initial condition y
    :param omega0: the natural frequency of the oscillator
    :param gamma: the damping coefficient

    """

    yderive = np.array([y[1], -omega0**2*y[0]-2*gamma*y[1]])
 
    return yderive

if __name__=='__main__':
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
    plt.figure(1)
    plt.plot(t_eval, sol[0], label="position [m]")
    plt.plot(t_eval, sol[1], label="velocity [m/s]")
    plt.title(r'Damped Oscillator with $A = 1$ cm, $\omega_0 = 1$ rad/s, $\gamma = 0.2$ s$^{-1}$, $\phi = -\frac{\pi}{2}$ rad')
    plt.xlabel("Time [s]")
    plt.legend()

    # Prepare the input arguments part(b)
    # A = 1cm, omega0 = 1rad/s, gamma = 1.0s^-1, phi = -pi/2rad
    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 2001)
    omega0 = 1
    gamma = 1
    A = 1
    phi = -np.pi/2
    args = (omega0, gamma)
    y0 = np.array([0, A])

    # use the IVP solver
    sol = mysolver.solve_ivp(damp_oscillator, t_span, y0, "RK4", t_eval, args=args)

    # visualize the results
    plt.figure(2)
    plt.plot(t_eval, sol[0], label="position [m]")
    plt.plot(t_eval, sol[1], label="velocity [m/s]")
    plt.title(r'Damped Oscillator with $A = 1$ cm, $\omega_0 = 1$ rad/s, $\gamma = 1.0$ s$^{-1}$, $\phi = -\frac{\pi}{2}$ rad')
    plt.xlabel("Time [s]")
    plt.legend()

    # Prepare the input arguments part(c)
    # A = 1cm, omega0 = 1rad/s, gamma = 1.2s^-1, phi = -pi/2rad
    t_span = (0, 20)
    t_eval = np.linspace(0, 20, 2001)
    omega0 = 1
    gamma = 1.2
    A = 1
    phi = -np.pi/2
    args = (omega0, gamma)
    y0 = np.array([0, A])

    # use the IVP solver
    sol = mysolver.solve_ivp(damp_oscillator, t_span, y0, "RK4", t_eval, args=args)

    # visualize the results
    plt.figure(3)
    plt.plot(t_eval, sol[0], label="position [m]")
    plt.plot(t_eval, sol[1], label="velocity [m/s]")
    plt.title(r'Damped Oscillator with $A = 1$ cm, $\omega_0 = 1$ rad/s, $\gamma = 1.2$ s$^{-1}$, $\phi = -\frac{\pi}{2}$ rad')
    plt.xlabel("Time [s]")
    plt.legend()

    # Prepare the input arguments part(a)
    omega0 = 1
    gamma = 0.2
    args = (omega0, gamma)
    sol = mysolver.solve_ivp(damp_oscillator, t_span, y0, "RK4", t_eval, args=args)
    omega1 = np.sqrt(abs(omega0**2 - gamma**2))
    u = omega1 * sol[0]
    w = gamma*sol[0] + sol[1]

    # plot the phase diagram
    plt.figure(4)
    theta = np.arctan2(u, w)
    r = np.sqrt(u**2 + w**2)
    plt.polar(theta, r)
    plt.title(r'Phase Diagram with $A = 1$ cm, $\omega_0 = 1$ rad/s, $\gamma = 0.2$ s$^{-1}$, $\phi = -\frac{\pi}{2}$ rad')
    plt.gca().set_aspect('equal', adjustable='box')

    # Prepare the input arguments part(b)
    omega0 = 1
    gamma = 1
    args = (omega0, gamma)
    sol = mysolver.solve_ivp(damp_oscillator, t_span, y0, "RK4", t_eval, args=args)
    omega1 = np.sqrt(abs(omega0**2 - gamma**2))
    u = omega1 * sol[0]
    w = gamma*sol[0] + sol[1]

    # plot the phase diagram
    plt.figure(5)
    theta = np.arctan2(u, w)
    r = np.sqrt(u**2 + w**2)
    plt.polar(theta, r)
    plt.title(r'Phase Diagram with $A = 1$ cm, $\omega_0 = 1$ rad/s, $\gamma = 1.0$ s$^{-1}$, $\phi = -\frac{\pi}{2}$ rad')
    plt.gca().set_aspect('equal', adjustable='box')

    # Prepare the input arguments part(c)
    omega0 = 1
    gamma = 1.2
    args = (omega0, gamma)
    sol = mysolver.solve_ivp(damp_oscillator, t_span, y0, "RK4", t_eval, args=args)
    omega1 = np.sqrt(abs(omega0**2 - gamma**2))
    u = omega1 * sol[0]
    w = gamma*sol[0] + sol[1]

    # plot the phase diagram
    plt.figure(6)
    theta = np.arctan2(u, w)
    r = np.sqrt(u**2 + w**2)
    plt.polar(theta, r)
    plt.title(r'Phase Diagram with $A = 1$ cm, $\omega_0 = 1$ rad/s, $\gamma = 1.2$ s$^{-1}$, $\phi = -\frac{\pi}{2}$ rad')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
