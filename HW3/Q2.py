from model.finite import solve_laplace
from model.finite import generate_rho
import numpy as np
import matplotlib.pyplot as plt

# Visualize the density field
plt.figure(1)
rho = generate_rho(N=128, xmin=-5, xmax=5, ymin=-5, ymax=5)
plt.title("Color and contour plot of the source function", fontsize=14)
plt.xlabel('x-axis', fontsize=14)
plt.ylabel('y-axis', fontsize=14)
plt.tight_layout()
plt.imshow(rho)
plt.colorbar()
plt.contour(rho, colors="w")

# Visualize the solution
plt.figure(2)
dx = 10/128

rho = rho.flatten()

bc = np.array([1,0,1,0])*0.0001
u = solve_laplace(N=128, rho=rho, dx=dx, bc=bc)

plt.title("Solution of Poisson's Equation", fontsize=14)
plt.xlabel('x-axis', fontsize=14)
plt.ylabel('y-axis', fontsize=14)
plt.tight_layout()
plt.imshow(u.transpose())
plt.colorbar()
plt.contour(u.transpose(),colors="w")
plt.show()