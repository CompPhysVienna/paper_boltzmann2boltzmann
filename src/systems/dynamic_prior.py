import numpy as np
import torch

from systems.base import base_system

"""
This class is a wrapper used for live sampling, i.e. the sampling of novel 
configurations of a specific system (used as prior) along training of NFs.

It is initialized with a number of cached configurations, a test fraction and system. 

A sampler, some initial configurations and specific test data can be optionally provided.

The number of cached configuration must be a reasonably high number to allow for sampling
during network training.

The test fraction will be the fraction of cached configurations used for validation 
during training.

The system is the corresponding wrapped system class which is used as prior during 
network training.

If sampler is None, sampling is done randomly picking from initial configurations.

If init_conf is None, the system is used to generate a single configuration, 
which is replicated n_cached times. The sampler then takes care of decorrelating them.
This means that using sampler = None and init_conf = None is not allowed. 
"""
class dynamic_prior(base_system):

    def __init__(self, n_cached, test_fraction, system, sampler = None, init_confs = None, test_data = None):
        super().__init__(system.n_particles, system.dimensions, system.device) 

        self.name = system.name
        
        assert test_fraction > 0 and test_fraction < 1 or test_data is not None, \
                "test_fraction must be greater than 0 and smaller than 1 when test data are not explicitely provided."
        assert sampler is not None or init_confs is not None, \
                "Either sampler or a number of decorrelated intial configurations are required"
        
        self.n_cached = n_cached
        self.system = system
        self.sampler = sampler
        if sampler is None:
            print("Dummy sampling enabled")

        if init_confs is None:
            self.sampler.sample_space(N=int(n_cached*(1+test_fraction)), beta=1)
            init_confs = self.sampler.x0
        
        n_init_confs = init_confs.shape[0]

        if test_data is None:
            n_test = int(test_fraction*n_init_confs)
            indx = np.random.choice(np.arange(0, n_init_confs), replace=False, size=n_test)
            self.test_data = init_confs[indx].clone()
            indx_mask = np.ones(n_init_confs, dtype=bool)
            indx_mask[indx] = False
            train_data = init_confs[indx_mask].clone()
        else:
            self.test_data = test_data.clone()
            train_data = init_confs.clone()

        n_train = train_data.shape[0]
        indx = np.random.choice(np.arange(0, n_train), replace=(n_cached > n_train), size=n_cached)
        self.cache = train_data[indx].clone()

        print(f"Available initial configurations: {n_init_confs}")
        print(f"Train data made by {n_train} configurations")
        print(f"Test data made by {n_test} configurations")
        print(f"Configuration cache intialized with {n_cached} configurations")
        print(f"{n_cached - n_train} configurations have been repeated in cache")
        if sampler is None and (n_cached - n_train) > 0:
            raise Exception("Configurations in cache are no longer Boltzmann distributed")


    def sample(self, N, beta):
                
        if self.training:
    
            if N > self.n_cached:
                raise Exception("Cannot sample more than cached configurations")
    
            indx = np.random.choice(np.arange(0, self.n_cached), replace=False, size=N)
            if self.sampler is not None:
                self.sampler.x0 = self.cache[indx]
                self.sampler.sample_space(N=N, beta=beta)
                self.cache[indx] = self.sampler.x0

            return self.cache[indx]
        
        else:
            
            if N > self.test_data.shape[0]:
                raise Exception("Test Dataset is too small")
        
            indx = np.random.choice(np.arange(0, self.test_data.shape[0]), replace=False, size=N)
            return self.test_data[indx]


    def energy(self, x):
            
        return self.system.energy(x)