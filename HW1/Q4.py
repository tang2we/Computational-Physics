import solver as mysolver
import numpy as np
import matplotlib.pyplot as plt

def rlc_circuit(t, y, R, L, C, E0, W):
    """
    Function to calculate the derivatives of the RLC circuit
    """
    # Unpack the state vector
    q, i = y

    # Calculate the derivatives
    dq = i
    di = (E0*np.sin(W*t) - R*i - q/C)/L

    return np.array([dq, di])

# Prepare the input arguments part(a)
t_span = (0, 20)
t_eval = np.linspace(0, 20, 2001)
L = 1
C = 1
E0 = 1
R = 0.8
W = 0.7
args = (R, L, C, E0, W)
q0 = (0, 0)

# use the IVP solver
sol = mysolver.solve_ivp(rlc_circuit, t_span, q0, "RK4", t_eval, args=args)
Vl = E0 * np.sin(W*t_eval) - sol[1] * R - sol[0] / C
Vr = sol[1] * R
Vc = sol[0] / C
# plt.plot(t_eval, sol[1])
plt.figure(1)
plt.plot(t_eval, sol[1], label='I [A]')
# plt.plot(t_eval, Vc, label='Vc')
plt.plot(t_eval, Vl, label=f'$V_L$ [V]')
# plt.plot(t_eval, Vr, label='Vr')
plt.xlabel('Time [s]')
plt.legend()


WF = np.linspace(0.3, 1.5, 13)
plt.figure(2)
for W in WF:
    args = (R, L, C, E0, W)
    sol = mysolver.solve_ivp(rlc_circuit, t_span, q0, "RK4", t_eval, args=args)
    Vl = E0 * np.sin(W*t_eval) - sol[1] * R - sol[0] / C
    plt.plot(t_eval, Vl, label=f'Vl, $\omega$={W:.1f}')
    plt.xlabel('Time [s]')
    plt.ylabel('Voltage across the inductor [V]')
plt.legend()

plt.figure(3)
for W in WF:
    args = (R, L, C, E0, W)
    sol = mysolver.solve_ivp(rlc_circuit, t_span, q0, "RK4", t_eval, args=args)
    plt.xlabel('Time [s]')
    plt.ylabel('Current [A]')
    plt.plot(t_eval, sol[1], label=f'I, $\omega$={W:.1f}')
plt.legend()
plt.show()
