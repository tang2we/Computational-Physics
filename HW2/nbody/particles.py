import numpy as np
import matplotlib.pyplot as plt
from numba import jit, njit, set_num_threads

class Particles:
    """
    Particle class to store particle properties
    """
    def __init__(self, N:int=100):
        """
        :param N: Number of particles
        """
        self.time = 0.0 # start from t = 0
        self.nparticles = N
        self._tags = np.arange(N)
        self._masses = np.ones((N, 1))
        self._positions = np.zeros((N, 3))
        self._velocities = np.zeros((N, 3))
        self._accelerations = np.zeros((N, 3))
    
    @property
    def tags(self):
        return self._tags
    @tags.setter
    def tags(self, some_tags):
        if len(some_tags) != self.nparticles:
            raise ValueError("Number of particles doesn't match")
        self._tags = some_tags
        return
    
    @property
    def masses(self):
        return self._masses
    @masses.setter
    def masses(self, some_masses):
        if len(some_masses) != self.nparticles:
            raise ValueError("Number of particles doesn't match")
        self._masses = some_masses
        return
    
    @property
    def positions(self, dimension=3):
        return self._positions
    @positions.setter
    def positions(self, some_positions):
        if np.shape(some_positions) != np.shape(self._positions):
            raise ValueError("Shape of positions doesn't match")
        self._positions = some_positions
        return
    
    @property
    def velocities(self, dimension=3):
        return self._velocities
    @velocities.setter
    def velocities(self, some_velocities):
        if np.shape(some_velocities) != np.shape(self._velocities):
            raise ValueError("Shape of velocities doesn't match")
        self._velocities = some_velocities
        return
    
    @property
    def accelerations(self, dimension=3):
        return self._accelerations
    @accelerations.setter
    def accelerations(self, some_accelerations):
        if np.shape(some_accelerations) != np.shape(self._accelerations):
            raise ValueError("Shape of accelerations doesn't match")
        self._accelerations = some_accelerations
        return

    def add_particles(self, masses, positions, velocities, accelerations):
        """
        Add particles to the system
        """
        self.nparticles += len(masses)
        self.masses = np.vstack((self.masses, masses))
        self.positions = np.vstack((self.positions, positions))
        self.velocities = np.vstack((self.velocities, velocities))
        self.accelerations = np.vstack((self.accelerations, accelerations))
        self.tags = np.arange(self.nparticles)
        return
    
    def set_particles(self, positions, velocities, accelerations):
        """
        Set particle properties for the N-body simulation

        :param pos: positions of particles
        :param vel: velocities of particles
        :param acc: accelerations of particles
        """
        self.positions = positions
        self.velocities = velocities
        self.accelerations = accelerations
        return

    def compute_energy(self, G=1.0):
        """
        Compute the total energy of the system
        """
        # kinetic energy
        KE = 0.5 * np.sum(self.masses.flatten() * np.sum(self.velocities**2, axis=1))
        # potential energy
        PE = 0
        for i in range(self.nparticles):
            for j in range(i+1, self.nparticles):
                rij = self.positions[i] - self.positions[j]
                r = np.sqrt(np.sum(rij**2))
                PE -= G * self.masses[i][0] * self.masses[j][0] / r
        return (KE, PE)
    
    def output(self, filename='data.txt', out_E=False):
        """
        Write particle data to file
        """
        # numpy genformtxt to write data
        with open(filename, 'w') as f:
            # time KE PE
            f.write(f"# Time: {self.time:<12}\n")

            # energy KE PE E
            if out_E:
                KE, PE = self.compute_energy()
                f.write(f"{KE:<12} {PE:<12} {KE+PE:<12}\n")

            # tag mass position velocity acceleration
            f.write("# tag mass x y z vx vy vz ax ay az\n")
            for i in range(self.nparticles):
                f.write(f"{self.tags[i]:<4} {self.masses[i][0]:<4} ")
                f.write(f"{self.positions[i][0]:<4} {self.positions[i][1]:<4} {self.positions[i][2]:<4} ")
                f.write(f"{self.velocities[i][0]:<4} {self.velocities[i][1]:<4} {self.velocities[i][2]:<4} ")
                f.write(f"{self.accelerations[i][0]:<4} {self.accelerations[i][1]:<4} {self.accelerations[i][2]:<4}\n")
        return
    
    def draw(self, dim=3):
        """
        Draw particle positions
        """
        if dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.positions[:, 0], self.positions[:, 1], self.positions[:, 2])
            plt.show()
        elif dim == 2:
            plt.scatter(self.positions[:, 0], self.positions[:, 1])
            plt.show()
        else:
            raise ValueError("Invalid dimension")
        return

