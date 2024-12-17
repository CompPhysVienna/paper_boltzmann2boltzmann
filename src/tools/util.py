import os
import numpy as np
import torch

from itertools import combinations


def transform_dataset(x, n_particles, dimensions, box_length, PBC=True):

    dofs = n_particles*dimensions
    
    # transform in internal coordinates (center particle 0)
    xp = x.view(-1, n_particles, dimensions)
    c = xp[:,0].clone()
    xp -= c.unsqueeze(1)
    if PBC:
        xp -= box_length*torch.round(xp/box_length)
    x = xp.view(-1, dofs)

    return x


def octahedral_transformation(dimensions, device):
        
        identity = torch.eye(dimensions, dtype=torch.float32, device=device)
        permuted_axes = identity[:, torch.randperm(dimensions)]            
        reflect = torch.randint(0, 2, size=(dimensions,), device=device) * 2 - 1
        
        return (reflect*permuted_axes).unsqueeze(0)
  

def get_targets(dimensions, n_blocks):
    targets = []

    A = range(dimensions)
    for r in range(1, len(A)):
        for i in combinations(A, r):
            targets.append(i)

    return targets*n_blocks


def get_target_indices(target, n_particles, dimensions):

    coordinate_indices =  np.arange(n_particles*dimensions)
    mask = np.ones(n_particles*dimensions, dtype=bool)
    for indx in target:
        mask[indx::dimensions] = 0

    return coordinate_indices[mask], coordinate_indices[~mask]


def ress(log_w):

    with torch.no_grad():

        sig = torch.nn.Softmax(dim=0)
        ress = 1/torch.sum(sig(log_w)**2)/log_w.shape[0]

    return ress


def searchsorted(bin_locations, inputs, eps=1e-6):

    bin_locations[..., -1] += eps

    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def generate_output_directory(run_id):

    output_dir = f"./output/{run_id}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Generated output directory: {output_dir}")
    
    return output_dir