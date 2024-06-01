import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit, prange, set_num_threads

def generate_mesh(nx, ny, buff=1, 
                  xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0):
    """
    Generate 2D mesh grids for solving Laplace equation.

    Parameters
    ----------
    nx : int
        Number of grid points in x direction.
    ny : int
        Number of grid points in y direction.
    buff : int
        Number of ghost cells around the domain.
    xmin : float
        Minimum value of x.
    xmax : float
        Maximum value of x.
    ymin : float
        Minimum value of y.
    ymax : float
        Maximum value of y.

    Returns
    -------
    u : 2D numpy array
        Initial guess. 
        
    x : 2D numpy array
        Mesh grid for x.
    y : 2D numpy array
        Mesh grid for y.

    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
        
    """

    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    u = np.zeros((nx+2*buff, ny+2*buff))
    x = np.linspace(xmin - buff*dx, xmax + buff*dx, nx + 2*buff)
    y = np.linspace(ymin - buff*dy, ymax + buff*dy, ny + 2*buff)

    return u, x, y, dx, dy

def generate_rho(N, xmin=-1, xmax=1, ymin=-1,ymax=1):

    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(x, y)
    rho = np.zeros((N,N))
    r1 = (X+1.5)**2 + Y**2
    r2 = (X-1.5)**2 + Y**2
    rho = np.exp(-5/4*r1**2) + 3/2*np.exp(-r2**2)

    return rho

# method to solve the Laplace equation
# Jacobi method
@njit(parallel=True)
def jacobi(u, uold, nx, ny, dx, rho):
        
    for j in prange(1, ny-1):
        for i in prange(1, nx-1):
            u[i, j] = 0.25 * (uold[i-1, j] + uold[i+1, j] + uold[i, j-1] + uold[i, j+1] - rho[i-1, j-1]*dx**2)
    return u

# gauss_seidel method
@njit(parallel=True)
def gauss_seidel(u, nx, ny, dx, rho):

    for j in prange(1, ny-1):
        for i in prange(1, nx-1):
            u[i, j] = 0.25 * (u[i-1, j] + u[i+1, j] + u[i, j-1] + u[i, j+1] - rho[i-1, j-1]*dx**2)
    return u

# successive over-relaxation method
def successive_over_relax(xold, xgs, w):
    x = (1 - w) * xold + w * xgs

    return x

def update_bc(u, nx, ny, bc):
    
    u[-1, :] = bc[0] # right boundary
    u[0, :] = bc[2] # left boundary
    u[:, 0] = bc[3] # bottom boundary
    u[:, -1] = bc[1] # top boundary

    '''
    u[-1, :] = 0 # right boundary
    u[:, 0] = 0 # bottom boundary
    u[:, -1] = 0 # top boundary
    u[0, :] = 0 # left boundary
    '''

    return u

def jacobi_relax(u, dx, rho, tol=1e-8, maxiter=1e6, bc=np.array([0, 0, 0, 0])):
    """
    Relax the solution using Jacobi method.

    Parameters
    ----------
    u : 2D numpy array
        Initial guess.
    tolerance : float
        Tolerance for convergence.
    maxiter : int
        Maximum number of iterations.
    """

    nx, ny = u.shape
    u = update_bc(u, nx, ny, bc)
    
    errors = np.zeros(maxiter)
    iterations = np.zeros(maxiter)

    if rho is None:
        rho = np.zeros((nx, ny))

    for i in range(maxiter):
        uold = np.copy(u)
        u = jacobi(u, uold, nx, ny, dx, rho)
        err = np.linalg.norm(u - uold)
        errors[i] = err
        iterations[i] = i
        if err < tol:
            break    
    
    return u, iterations, errors

def gauss_seidel_relax(u, dx, rho, tol=1e-8, maxiter=1e6, bc=np.array([0, 0, 0, 0])):
    """
    Relax the solution using Jacobi method.

    Parameters
    ----------
    u : 2D numpy array
        Initial guess.
    tolerance : float
        Tolerance for convergence.
    maxiter : int
        Maximum number of iterations.
    """

    nx, ny = u.shape
    u = update_bc(u, nx, ny, bc)

    errors = np.zeros(maxiter)
    iterations = np.zeros(maxiter)

    if rho is None:
        rho = np.zeros((nx, ny))

    for i in range(maxiter):
        uold = np.copy(u)
        u = gauss_seidel(u, nx, ny, dx, rho)
        err = np.linalg.norm(u - uold)
        errors[i] = err
        iterations[i] = i
        if err < tol:
            break    
    
    return u, iterations, errors

def sor_relax(u, dx, rho, w, tol=1e-8, maxiter=1e6, bc=np.array([0, 0, 0, 0])):
    """
    Relax the solution using Jacobi method.

    Parameters
    ----------
    u : 2D numpy array
        Initial guess.
    tolerance : float
        Tolerance for convergence.
    maxiter : int
        Maximum number of iterations.
    """

    nx, ny = u.shape
    u = update_bc(u, nx, ny, bc)

    errors = np.zeros(maxiter)
    iterations = np.zeros(maxiter)

    if rho is None:
        rho = np.zeros((nx, ny))

    for i in range(maxiter):
        uold = np.copy(u)
        u = gauss_seidel(u, nx, ny, dx, rho)
        u = successive_over_relax(uold, u, w)
        err = np.linalg.norm(u - uold)
        errors[i] = err
        iterations[i] = i
        if err < tol:
            break    
    
    return u, iterations, errors

def relax(u, dx, rho=None, method='jacobi', tol=1e-8, maxiter=1e6, w=1.5, bc=np.array([0, 0, 0, 0])):
    """
    Relax the solution using Jacobi or Gauss-Seidel method.

    Parameters
    ----------
    method : str
        Relaxation method. Default is Jacobi.
    u : 2D numpy array
        Initial guess.
    tolerance : float
        Tolerance for convergence.
    maxiter : int
        Maximum number of iterations.
    """

    if method == "jacobi":
        u, itt, err = jacobi_relax(u, dx, rho, tol, maxiter, bc)
    elif method == "gauss_seidel":
        u, itt, err = gauss_seidel_relax(u, dx, rho, tol, maxiter, bc)
    elif method == "sor":
        u, itt, err = sor_relax(u, dx, rho, w, tol, maxiter, bc)
    else:
        raise ValueError("Invalid method. Choose either Jacobi, Gauss-Seidel or SOR.")

    return u, itt, err

if __name__ == "__main__":
    nx = 128
    ny = 128
    buff = 1
    xmin = -5
    xmax = 5
    ymin = -5
    ymax = 5

    # set the density field
    rho = generate_rho(N=128, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)

    u,x,y,dx,dy = generate_mesh(nx, ny, buff, xmin, xmax, ymin, ymax)
    u, itt128, err128 = relax(u, dx, rho, method='jacobi', tol=1e-7, maxiter=1_000_00, w=1.6)

    # plt.figure(1)
    # plt.imshow(u, cmap='viridis')
    plt.figure(2)
    plt.plot(itt128, err128, label="jacobi")
    u, itt128, err128 = relax(u, dx, rho, method='gauss_seidel', tol=1e-7, maxiter=1_000_00, w=1.6)
    plt.plot(itt128, err128, label="gauss_seidel")
    plt.yscale('log')
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Error')
    plt.show()
