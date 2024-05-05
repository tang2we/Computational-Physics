import numpy as np
import matplotlib.pyplot as plt
from nbody import * 
from numba import set_num_threads
import random

# Set the number of threads to use for numba
nthreads = 8
set_num_threads(nthreads)

method = ('Euler', 'RK2', 'RK4', 'leap_frog')

# plot the results

for m in method:

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
    
    # nbody simulation
    simulation = NBodySimulator(particles=pts)
    simulation.setup(G=1,rsoft=0.01,method=m, io_freq=10)
    simulation.evolve(dt=0.01, tmax=10, out_E=True)

    # load the data
    fns = load_files('nbody')

    times = np.linspace(0, 10, len(fns))
    KE = np.zeros(len(times))
    PE = np.zeros(len(times))
    E = np.zeros(len(times))

    for i, fn in enumerate(fns):
        data = np.loadtxt(fn, max_rows=1)
        KE[i] = data[0]
        PE[i] = data[1]
        E[i] = data[2]

    plt.figure(figsize=(8, 6))
    plt.plot(times, KE, label=f'kenetic energy', alpha=0.5)
    plt.plot(times, PE, label=f'potential energy')
    plt.plot(times, E, label=f'total energy', alpha=0.5)
    plt.legend()
    plt.xlabel('time [s]', fontsize=12)
    plt.ylabel('energy [J]', fontsize=12)
    plt.title(f'N-body simulation using {m} scheme', fontsize=14)
    plt.savefig(f'Q3_{m}.png')
