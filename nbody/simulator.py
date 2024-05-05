import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .particles import Particles
from numba import jit, njit, prange, set_num_threads

"""
The N-Body Simulator class is responsible for simulating the motion of N bodies



"""

class NBodySimulator:

    def __init__(self, particles: Particles):
        
        self.particles = particles
        self.nparticles = particles.nparticles
        self.time = particles.time
        self.setup()

        return

    def setup(self, G=1,
                    rsoft=0.01,
                    method="RK4",
                    io_freq=10,
                    io_header="nbody",
                    io_screen=True,
                    visualization=False):
        """
        Customize the simulation enviroments.

        :param G: the graivtational constant
        :param rsoft: float, a soften length
        :param meothd: string, the numerical scheme
                       support "Euler", "RK2", and "RK4"

        :param io_freq: int, the frequency to outupt data.
                        io_freq <=0 for no output. 
        :param io_header: the output header
        :param io_screen: print message on screen or not.
        :param visualization: on the fly visualization or not. 
        """
        
        self.G = G
        self.rsoft = rsoft
        self.method = method
        self.io_freq = io_freq
        self.io_header = io_header
        self.io_screen = io_screen
        self.visualization = visualization

        return

    def _advance_particles(self, dt, particles):
        
        method = self.method.lower()
        if method == "euler": 
            particles = self._advance_particles_Euler(dt, particles)
        elif method == "rk2":
            particles = self._advance_particles_RK2(dt, particles)
        elif method == "rk4":
            particles = self._advance_particles_RK4(dt, particles)
        elif method == "leap_frog":
            particles = self._advance_particles_Leap_frog(dt, particles)
        else:
            raise ValueError("Method not supported!")
        return particles
    
    def evolve(self, dt:float, tmax:float, out_E=False):
        """
        Start to evolve the system

        :param dt: float, the time step
        :param tmax: float, the total time to evolve
        
        """

        time = self.time
        particles = self.particles
        t = int(np.ceil((tmax-time)/dt))

        # check output directory
        io_folder = 'data_' + self.io_header
        Path(io_folder).mkdir(parents=True, exist_ok=True)

        
        for i in range(t+1):

            # make sure the last step is correct
            if (time + dt) > tmax and dt*i != tmax:
                dt = tmax - time

            if i != 0:
                # updates
                particles = self._advance_particles(dt, particles)

            # check IO
            if i % self.io_freq == 0:
                
                # output data
                fn = self.io_header+"_"+str(i).zfill(6)+".dat"
                fn = io_folder+"/"+fn
                particles.output(fn, out_E)
                if self.visualization:
                    particles.draw()

                if self.io_screen:
                    print("n=",i , "Time: ", time, " dt: ", dt)

            # update the time
            time += dt

        print("Simulation is done!")
        return
        
    def _advance_particles_Euler(self, dt, particles):
        
        nparticles = particles.nparticles
        masses = particles.masses

        y0 = particles.positions
        v0 = particles.velocities
        a0 = _calculate_acceleration(nparticles, masses, y0)

        y1 = y0 + v0*dt
        v1 = v0 + a0*dt
        a1 = _calculate_acceleration(nparticles, masses, y1)

        # update the particles
        particles.set_particles(y1, v1, a1)

        return particles

    def _advance_particles_RK2(self, dt, particles):
        
        nparticles = particles.nparticles
        masses = particles.masses
        y0 = particles.positions
        v0 = particles.velocities
        a0 = _calculate_acceleration(nparticles, masses, y0)
        
        y1 = y0 + v0*dt
        v1 = v0 + a0*dt
        a1 = _calculate_acceleration(nparticles, masses, y1)

        y2 = y1 + v1*dt
        v2 = v0 + a1*dt
        
        y = (y0 + y2)/2
        v = (v0 + v1)/2
        
        a = _calculate_acceleration(nparticles, masses, y)

        particles.set_particles(y, v, a)

        return particles

    def _advance_particles_RK4(self, dt, particles):
        
        nparticles = particles.nparticles
        masses = particles.masses
        y0 = particles.positions
        v0 = particles.velocities
        a0 = _calculate_acceleration(nparticles, masses, y0)

        y1 = y0 + v0*dt/2
        v1 = v0 + a0*dt/2
        a1 = _calculate_acceleration(nparticles, masses, y1)

        y2 = y0 + v1*dt/2
        v2 = v0 + a1*dt/2
        a2 = _calculate_acceleration(nparticles, masses, y2)

        y3 = y0 + v2*dt
        v3 = v0 + a2*dt
        a3 = _calculate_acceleration(nparticles, masses, y3)

        y = y0 + (v0 + 2*v1 + 2*v2 + v3)*dt/6
        v = v0 + (a0 + 2*a1 + 2*a2 + a3)*dt/6
        a = _calculate_acceleration(nparticles, masses, y)

        particles.set_particles(y, v, a)

        return particles
    
    def _advance_particles_Leap_frog(self, dt, particles):
        
        nparticles = particles.nparticles
        masses = particles.masses
        y0 = particles.positions
        v0 = particles.velocities
        a0 = _calculate_acceleration(nparticles, masses, y0)

        v = v0 + a0*dt/2
        y = y0 + v*dt
        a = _calculate_acceleration(nparticles, masses, y)

        particles.set_particles(y, v, a)

        return particles

# accelerate the function
@njit(parallel=True)
def _calculate_acceleration(nparticles, masses, positions, rsoft=0.01, G=1):
    """
    Calculate the acceleration of the particles
    """

    accelerations = np.zeros_like(positions)

    for i in prange(nparticles):
        for j in prange(nparticles):
            if (j>i): 
                rij = positions[i,:] - positions[j,:]
                r = np.sqrt(np.sum(rij**2) + rsoft**2)
                force = - G * masses[i,0] * masses[j,0] * rij / r**3
                accelerations[i,:] += force[:] / masses[i,0]
                accelerations[j,:] -= force[:] / masses[j,0]

    return accelerations

if __name__ == "__main__":
    
    pass