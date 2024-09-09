import mlatom as ml 
import sys 
sys.path.append('~/scripts')
from al_general import active_learning,Sampler,stop_function,ml_model_trainer
import torch
import numpy as np

class my_al_init_sampler(Sampler):
    def __init__(self):
        pass 

    def sample(self,**kwargs):
        if 'al_object' in kwargs:
            al_object = kwargs['al_object']
        iteration = al_object.iteration
        if iteration == 0:
            nsample = al_object.maximum_number_of_sampled_points
            self.nsample = nsample
        else:
            # moldb = ml.data.molecular_database.load(f'labeled_db_iteration{iteration}.json',format='json')
            nsample = int(al_object.maximum_number_of_sampled_points*self.nsample/al_object.original_number_of_molecules_to_label)
            self.nsample = nsample
        print(f"Number of MD trajectories: {nsample}")

        kwargs['nsample'] = nsample 

        return super().harmonic_quantum_boltzmann(**kwargs)

if __name__ == '__main__':
    print(torch.cuda.is_available())
    ub3lyp = ml.models.methods(method='UB3LYP/6-31G*',program='gaussian') 
    eqmol = ml.data.molecule() 
    eqmol.load('da_eqmol.json',format='json')
    init_sampler = Sampler(sampler_function='harmonic-quantum-boltzmann')
    init_sampler_kwargs = {
        'eqmol':eqmol,
        'nsample':50,
        'initial_temperature':298
    }
    # al_init_sampler = Sampler(sampler_function='wigner')
    al_init_sampler = my_al_init_sampler()
    al_init_sampler_kwargs = {
        'eqmol':eqmol,
        # 'nsample':100,
        'initial_temperature':298,
    }
    al_sampler = Sampler(sampler_function='md')
    sampler_kwargs = {
        'stop_function':stop_function,
        'stop_function_kwargs':{},
        'maximum_propagation_time':150.0,
        'time_step':0.5,
        'md_parallel':True,
        'nthreads':6,
    }
    model_trainer = ml_model_trainer(ml_model_type='ani')
    active_learning(
        job_name='da',
        initial_points_sampler=init_sampler,
        initial_points_sampler_kwargs=init_sampler_kwargs,
        initial_points_refinement='cross_validation',
        init_ntrain_next=50,
        init_train_energies_only=True,
        minimum_number_of_fitting_points=5,
        label_nthreads=6,
        ml_model_trainer=model_trainer,
        # ml_model_type='KREG',
        device='cuda',
        property_to_learn=['energy','energy_gradients'],
        property_to_check=['energy'],
        initial_conditions_sampler=al_init_sampler,
        initial_conditions_sampler_kwargs=al_init_sampler_kwargs,
        sampler=al_sampler,
        sampler_kwargs=sampler_kwargs,
        maximum_number_of_sampled_points=100,
        minimum_number_of_sampled_points=0.05,
        reference_method=ub3lyp,
        # iteration=6
    )