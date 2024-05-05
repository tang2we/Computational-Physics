import numpy as np
import matplotlib.pyplot as plt
from nbody import * 
from numba import set_num_threads


# Set the number of threads to use for numba
nthreads = 8
set_num_threads(nthreads)

# Set the random seed
np.random.seed(0)

# initialize particles
num_particles = 1000
pts = Particles(N=num_particles)
pts.masses = np.ones((num_particles, 1)) * 20/num_particles
pts.positions = np.random.randn(num_particles,3) # 3 directions
pts.velocities = np.random.randn(num_particles,3)
pts.accelerations = np.random.randn(num_particles,3)
pts.tags = np.linspace(1,num_particles,num_particles)

# nbody simulation using leap_frog scheme
simulation = NBodySimulator(particles=pts)
simulation.setup(G=6.674e-11,rsoft=0.01,method='leap_frog', io_freq=200)
simulation.evolve(dt=0.01, tmax=10.0)

fns = load_files('nbody')
print(fns)

# plot the results
for fn in fns:
    data = np.loadtxt(fn)
    x = data[:, 2]
    y = data[:, 3]

    plt.scatter(x, y= y, s=4, alpha=0.5)
    plt.xlabel('x-axis [m]', fontsize=12)
    plt.ylabel('y-axis [m]', fontsize=12)
    plt.title('N-body simulation using leap_frog scheme', fontsize=14)
    plt.axis('equal')
    plt.show()
