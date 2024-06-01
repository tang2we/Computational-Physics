from model.finite import solve_laplace
import numpy as np
import matplotlib.pyplot as plt

rho = np.array([0,0,0,0,1,0,0,0,0]).flatten()
print(rho)
u = solve_laplace(N=3, rho=rho)
# Visualize the solution
print(u)
plt.imshow(u)
plt.colorbar()
plt.title("Solution of Poisson's Equation", fontsize=14)
plt.xlabel('x-axis', fontsize=14)
plt.ylabel('y-axis', fontsize=14)
plt.tight_layout()
plt.show()