import mlatom as ml 
import numpy as np 
import os 
import sys 
import joblib
from joblib import Parallel, delayed

temperature = 298

init_cond_db = ml.data.molecular_database.load('incond_298.json',format='json')
re_init_cond_db = ml.data.molecular_database.load('re_incond_298.json',format='json')

ani = ml.models.ani(model_file='da_energy_iteration37.npz',device='cuda')

# kreg = ml.models.kreg(model_file='da_energy_iteration44.npz',ml_program='KREG_API')
# kreg.nthreads=1


# def run_md(moldb,job_name):
    

#     trajs = Parallel(n_jobs=joblib.cpu_count())(delayed(run_traj)(i) for i in range(len(moldb)))
#     sys.stdout.flush() 
#     # sequence_number = 1
#     # for traj in trajs:
#     #     traj.dump(filename=f'{job_name}_temp{temperature}_{sequence_number}', format='plain_text')
#     #     sequence_number += 1

def run_traj(imol,moldb,jobname):
    init_mol = moldb[imol]
    #nose_hoover = ml.md.Nose_Hoover_thermostat(temperature=temperature,molecule=init_mol,degrees_of_freedom=-6)
    dyn = ml.md(model=ani,
                molecule_with_initial_conditions = init_mol,
                ensemble='NVE',
                #thermostat=nose_hoover,
                time_step=0.5,
                maximum_propagation_time=150.0,
                # stop_function=stop_function
                )
    traj = dyn.molecular_trajectory
    traj.dump(filename=f'{jobname}_temp{temperature}_{imol+1}', format='plain_text')
    return traj

# print("Forward MD")
# run_md(init_cond_db,'forward')

# print("Backward MD")
# run_md(re_init_cond_db,'backward')
for ii in range(len(init_cond_db)):
    print(ii)
    run_traj(ii,init_cond_db,jobname='forward')
    run_traj(ii,re_init_cond_db,jobname='backward')



