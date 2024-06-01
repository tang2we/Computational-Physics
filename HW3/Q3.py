from model.iterate import generate_mesh, relax, generate_rho, update_bc
import matplotlib.pyplot as plt

nx = 128
ny = 128
buff = 1
xmin = -5
xmax = 5
ymin = -5
ymax = 5

method = ('jacobi', 'gauss_seidel', 'sor')
ws = (1.2, 1.5, 2.0)

# set the boundary conditions
bc = (0, 1, 0, 1)

# set the density field
rho = generate_rho(N=128, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

plt.figure(1)
for m in method:
    if m == 'sor':     
        for w in ws:
            u,x,y,dx,dy = generate_mesh(nx, ny, buff, xmin, xmax, ymin, ymax)
            u, itt, err = relax(u, dx, rho, method=m, tol=1e-6, maxiter=1_000_00, w=w)
            # ax[0].imshow(u, cmap='viridis')   
            plt.plot(itt, err, label=f"sor: w={w}", alpha=0.7, linewidth=3, linestyle='--')         
    else:
        u,x,y,dx,dy = generate_mesh(nx, ny, buff, xmin, xmax, ymin, ymax)
        u, itt, err = relax(u, dx, rho, method=m, tol=1e-6, maxiter=1_000_00)
        # ax[0].imshow(u, cmap='viridis')
        plt.plot(itt, err, label=m, alpha=0.7, linewidth=3)


plt.yscale('log') 
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('Error', fontsize=14)
plt.xlim(0, 40000)
plt.ylim(1e-8, 1e30)
plt.tight_layout()
plt.legend()
plt.show()

'''

u,x,y,dx,dy = generate_mesh(nx, ny, buff, xmin, xmax, ymin, ymax)
u, itt128, err128 = relax(u, dx, rho, method='jacobi', tol=1e-6, maxiter=1_000_00, w=1.5)

plt.figure(1)
plt.imshow(u, cmap='viridis')
plt.figure(2)
plt.plot(itt128, err128, label="128x128")
plt.yscale('log')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()
'''
