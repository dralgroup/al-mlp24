

import mlatom as ml
import numpy as np

def stop_function(molecule):
    """
    Stop MD before maximum_propagation_time.
    The criterion: <1.6 Å for both bonds to be formed or reactants are separated by >5 Å.
    For product: get the length of both bonds by find the specific atoms.
    For reactant: get the length of specific bonds, compare with each other and find the shorter.
    """
    label = 'ts'
    # If the distance more than 5 Å, stop.
    dist_matrix = molecule.get_internuclear_distance_matrix() # 'mol' is the initial geometry, we can't use the new coordinate generated in the function.
    atomic_orders = [(0,10),(6,11)]
    distances = []
    for atoms_1 in atomic_orders:
        ii = atoms_1[0] ; jj = atoms_1[1]
        dist_1 = dist_matrix[ii][jj]
        distances.append(dist_1)
    min_dist = min(distances)
    if min_dist > 2.52: 
        label='reactant'
    max_dist = max(distances)
    if max_dist < 2.02:
        label = 'product'

    return label 

def bond_length(molecule):
    """
    Stop MD before maximum_propagation_time.
    The criterion: <1.6 Å for both bonds to be formed or reactants are separated by >5 Å.
    For product: get the length of both bonds by find the specific atoms.
    For reactant: get the length of specific bonds, compare with each other and find the shorter.
    """
    # label = 'ts'
    # If the distance more than 5 Å, stop.
    dist_matrix = molecule.get_internuclear_distance_matrix() # 'mol' is the initial geometry, we can't use the new coordinate generated in the function.
    atomic_orders = [(0,10),(6,11)]
    distances = []
    for atoms_1 in atomic_orders:
        ii = atoms_1[0] ; jj = atoms_1[1]
        dist_1 = dist_matrix[ii][jj]
        distances.append(dist_1)
    min_dist = min(distances)
    # if min_dist > 2.52: 
    #     label='reactant'
    max_dist = max(distances)
    # if max_dist < 2.02:
    #     label = 'product'

    return min_dist, max_dist, distances

def check_traj(forward_traj,backward_traj):
    forward_label = 'ts'
    backward_label = 'ts'
    forward_time = -1
    backward_time = -1
    forward_distances_list = [] 
    backward_distances_list = []
    for imol in range(len(forward_traj)):
        min_dist,max_dist,distances = bond_length(forward_traj[imol])
        forward_distances_list.append(distances)
        if min_dist > 5.0:
            forward_label = 'reactant'
            break
        if max_dist < 1.6:
            forward_label = 'product'
            break 
        if (min_dist > 2.52 or max_dist < 2.02) and forward_time < 0:
            forward_time = imol*0.5

    for imol in range(len(backward_traj)):
        min_dist,max_dist,distances = bond_length(backward_traj[imol])
        backward_distances_list.append(distances)
        if min_dist > 5.0:
            backward_label = 'reactant'
            break 
        if max_dist < 1.6:
            backward_label = 'product'
            break
        if (min_dist > 2.52 or max_dist < 2.02) and backward_time < 0:
            backward_time = imol*0.5

    # Check time gap
    if (forward_label=='product' and backward_label=='reactant') or (forward_label=='reactant' and backward_label=='product'):
        if forward_label == 'product':
            distances_list = forward_distances_list 
        else:
            distances_list = backward_distances_list 

        bond1_formed = False 
        bond2_formed = False

        for istep in range(len(distances_list)):
            distances = distances_list[istep]
            # print(distances)
            
            if min(distances) < 1.6 and not bond1_formed:
                gap = istep * 0.5 
                bond1_formed = True
            if max(distances) < 1.6 and not bond2_formed:
                gap = istep * 0.5 - gap
                bond2_formed = True 
                break
    else:
        gap = None
        
    return forward_label,backward_label,forward_time+backward_time,gap


reactive = 0 
nonreactive = 0
traverse_time_list = []
gap_list = []

for ii in range(1000):
    forward_traj = ml.data.molecular_database.from_xyz_file(f'forward_temp298_{ii+1}.xyz')
    backward_traj = ml.data.molecular_database.from_xyz_file(f'backward_temp298_{ii+1}.xyz')
    # forward_label = 'ts'
    # backward_label = 'ts'

    # for imol in range(len(forward_traj)):
    #     forward_label = stop_function(forward_traj[imol])
    #     if forward_label != 'ts':
    #         forward_time = imol*0.5
    #         break 

    # for imol in range(len(backward_traj)):
    #     backward_label = stop_function(backward_traj[imol])
    #     if backward_label != 'ts':
    #         backward_time = imol*0.5
    #         break 

    forward_label, backward_label, traverse_time, gap = check_traj(forward_traj,backward_traj)

    print(f'{forward_label} , {backward_label}, {traverse_time}, {gap}')
    
    if (forward_label == 'product' and backward_label == 'reactant') or (forward_label == 'reactant' and backward_label == 'product'):
        reactive += 1 
        # print(f'{forward_time} + {backward_time} = {forward_time+backward_time}')
        traverse_time_list.append(traverse_time)
        gap_list.append(gap)
    else:
        nonreactive += 1

print(reactive,nonreactive)
print(f"Average time: {np.mean(traverse_time_list)} fs")
print(f"Median time: {np.median(traverse_time_list)} fs")
print(f"Standard deviation: {np.std(traverse_time_list)} fs")
print(f"Average time gap: {np.mean(gap_list)} fs")
print(f"Median time gap: {np.median(gap_list)} fs")
print(f"Standard deviation: {np.std(gap_list)} fs")


