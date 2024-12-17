import numpy as np
import torch

class metropolis_monte_carlo(object):
    """
    Implements the Metropolis Monte Carlo sampling algorithm.

    Parameters:
    - system: Object representing the physical system, providing energy computation and configuration details.
    - step_size: Maximum step size for particle movement.
    - n_cycles: Number of Metropolis sampling cycles to perform.
    - n_equilibration: Number of equilibration cycles (optional, defaults to 10 times n_cycles).
    - transform: Whether to transform sampled coordinates into internal coordinates (center particle 0).
    """

    def __init__(self, system, step_size, n_cycles, n_equilibration=None, transform=True):
        self.system = system
        self.dimensions = system.dimensions  # Number of spatial dimensions.
        self.n_particles = system.n_particles  # Number of particles in the system.
        self.dofs = system.dofs  # Degrees of freedom (n_particles * dimensions).
        self.device = system.device  # Torch device (CPU/GPU).

        self.step_size = step_size  # Maximum displacement per step.
        self.n_cycles = n_cycles  # Number of cycles for sampling.

        self.x0 = None  # Stores the initial configuration.
        self.equilibrated = False  # Flag indicating if the system has equilibrated.
        
        # Set equilibration cycles (default is 10 times n_cycles).
        self.n_equilibration = n_cycles * 10 if n_equilibration is None else n_equilibration

        self.transform = transform  # Enable/disable transformation into internal coordinates.


    def metropolis_cycle(self, x, u_x, beta, dx):
        """
        Perform one cycle of the Metropolis algorithm.

        Parameters:
        - x: Current configurations of the system.
        - u_x: Current energies of the configurations.
        - beta: Inverse temperature (1/kT).
        - dx: Maximum step size for displacements.

        Returns:
        - x: Updated configurations.
        - u_x: Updated energies.
        - acc: Fraction of accepted moves.
        """
        n_samples = x.shape[0]  # Number of configurations.

        # Propose random displacements.
        shift = torch.zeros([n_samples, self.dofs], device=self.device)
        selected_particles = torch.randint(self.n_particles, size=(n_samples,), device=self.device)
        selected_dofs = torch.stack([selected_particles * self.dimensions + i for i in range(self.dimensions)])
        shift[np.arange(len(shift)), selected_dofs] = (torch.rand(selected_dofs.shape, device=self.device) * 2 - 1) * dx
        xp = x + shift  # Propose new configurations.

        # Calculate energy of proposed configurations.
        u_xp = self.system.energy(xp).squeeze()

        # Metropolis acceptance criterion.
        mask = torch.rand(size=(n_samples,), device=self.device) < torch.exp(-beta * (u_xp - u_x))
        xp_mask = xp[mask].view(-1, self.n_particles, self.dimensions)
        
        # Apply periodic boundary conditions (PBC).
        if self.system.PBC:
            xp_mask -= self.system.box_length * torch.round(xp_mask / self.system.box_length)
        
        # Update accepted configurations and energies.
        x[mask] = xp_mask.view(-1, self.dofs)
        u_x[mask] = u_xp[mask]
        
        # Calculate acceptance ratio.
        acc = mask.sum() / len(mask)
        
        return x, u_x, acc


    def sample_space(self, N, beta):
        """
        Generate samples using the Metropolis Monte Carlo algorithm.

        Parameters:
        - N: Number of configurations to sample.
        - beta: Inverse temperature (1/kT).

        Returns:
        - x: Final configurations after sampling.
        - u_x: Energies of the sampled configurations.
        - acc: Fraction of accepted moves during sampling.
        """
        # Initialize configurations (either random or from the last sampled state).
        if self.x0 is None:
            x = torch.stack([self.system.init_conf() for i in range(N)])
        else:
            indx = np.random.choice(np.arange(0, self.x0.shape[0]), replace=(N > self.x0.shape[0]), size=N)
            x = self.x0[indx]

        # Compute initial energies.
        u_x = self.system.energy(x).squeeze()

        # Equilibration phase.
        if not self.equilibrated:
            for cycle in range(self.n_equilibration):
                x, u_x, acc = self.metropolis_cycle(x, u_x, beta, self.step_size)
            self.equilibrated = True

        # Sampling phase.
        for cycle in range(self.n_cycles):
            x, u_x, acc = self.metropolis_cycle(x, u_x, beta, self.step_size)

        # Optionally transform configurations into internal coordinates.
        if self.transform:
            xp = x.view(-1, self.n_particles, self.dimensions)
            c = xp[:, 0].clone()  # Center coordinates of particle 0.
            xp -= c.unsqueeze(1)
            if self.system.PBC:
                xp -= self.system.box_length * torch.round(xp / self.system.box_length)
            x = xp.view(-1, self.dofs)

        # Store the last sampled state for future use.
        self.x0 = x.clone()

        return x, u_x, acc
