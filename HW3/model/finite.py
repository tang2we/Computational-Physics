import numpy as np
from scipy.sparse import dia_array  # if dia_array is not able, use dia_matrix
from scipy.sparse import dia_matrix
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as splinalg
from numba import jit, njit, prange
import matplotlib.pyplot as plt

def convert_solution(x):
    usize = np.sqrt(len(x))
    u = x.reshape(int(usize),int(usize)).transpose()
    return u

# Copy your Laplace's Equation solver from demo1 here

def generate_the_laplace_matrix_with_size(N=3):
    """
    assume sqrt(N) is an integer.
    """

    nsq = N*N
    A   = np.zeros((nsq,nsq))
    ex = np.ones(N)
    data = np.array([-ex, 4*ex, -ex])
    offsets  = np.array([-1, 0, 1])
    d_matrix = dia_matrix((data, offsets), shape=(N, N)).toarray()
    o_matrix = -np.identity(N)
    init_matrix_kernel(A, d_matrix, o_matrix, N)
        
    return A

@njit(parallel=True)
def init_matrix_kernel(A, d_matrix, o_matrix, N):
    """
    A: the matrix to be initialized
    d_matrix: the diagonal matrix
    o_matrix: the off-diagonal matrix
    N: the size of the matrix
    """

    for i in prange(N):
        for j in prange(N):
            if i == j:
                A[i*N:(i+1)*N, j*N:(j+1)*N] = d_matrix
            elif i == j+1 or i == j-1:
                A[i*N:(i+1)*N, j*N:(j+1)*N] = o_matrix
    return A

def generate_rho(N, xmin=-1, xmax=1, ymin=-1,ymax=1):

    x = np.linspace(xmin, xmax, N)
    y = np.linspace(ymin, ymax, N)
    X, Y = np.meshgrid(x, y)
    rho = np.zeros((N,N))
    r1 = (X+1.5)**2 + Y**2
    r2 = (X-1.5)**2 + Y**2
    rho += np.exp(-5/4*r1**2) + 3/2*np.exp(-r2**2)
    return rho


def generate_the_rhs_vector_with_size(N=3, rho=None, dx=1, bc=np.array([0, 0, 0, 0])):
    """
    Generate the right-hand side vector with given parameters.

    Parameters
    ----------
    N : int
        Size of the grid (NxN).
    rho : 2D numpy array, optional
        Source term.
    dx : float
        Grid spacing.
    bc : numpy array of size 4
        Boundary conditions [top, left, bottom, right].

    Returns
    -------
    b : 1D numpy array
        Right-hand side vector.
    """

    b = np.zeros(N * N)
    b[:N] -= bc[2]  # left boundary
    b[-N:] -= bc[0]  # right boundary
    b[0::N] -= bc[1]  # top boundary
    b[N-1::N] -= bc[3]  # bottom boundary

    if rho is not None:
        rho = rho.flatten()
        b -= rho*dx**2

    return b

def solve_laplace(N=3, rho=None, dx=1, bc=np.array([0,0,0,0])):

    A = generate_the_laplace_matrix_with_size(N=N)
    if rho is None:
        rho = generate_rho(N=N)
    b = generate_the_rhs_vector_with_size(N=N, rho=rho, dx=dx/(N-1), bc=bc)
    #x = linalg.solve(A,b) # use scipy
    #x = lu_solve(A,b)      # use our solver
    x = splinalg.spsolve(A,b) # use scipy sparse solver
    u = convert_solution(x)
    return u

if __name__ == "__main__":
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
    # plt.contour(u,colors="w")
    plt.show()