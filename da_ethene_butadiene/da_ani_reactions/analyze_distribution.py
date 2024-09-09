import mlatom as ml 
import matplotlib.pyplot as plt
import numpy as np

moldb = ml.data.molecular_database.load('incond_298.json',format='json')

bond_length = []

eqmol = ml.data.molecule() 
eqmol.load('../da_eqmol.json',format='json')
dist_matrix = eqmol.get_internuclear_distance_matrix()
print(dist_matrix[0][10])
print(dist_matrix[6][11])
for mol in moldb:
    dist_matrix = mol.get_internuclear_distance_matrix()
    bond_length.append(dist_matrix[0][10])
    bond_length.append(dist_matrix[6][11])

fig,ax = plt.subplots() 

ax.hist(bond_length,bins=30)

dev = abs(np.array(bond_length) - 2.27234)
dev.sort()
print(dev[980*2])

plt.savefig('bond_distribution.png',dpi=300)
