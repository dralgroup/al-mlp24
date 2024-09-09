import mlatom as ml 
import sys 
sys.path.append('~/scripts')
from al_general import active_learning,Sampler,stop_function,ml_model_trainer
import torch

if __name__ == '__main__':
    print(torch.cuda.is_available())
    eqmol = ml.data.molecule() 
    eqmol.load('ethanol_eqmol.json',format='json')
    init_sampler = Sampler(sampler_function='wigner')
    init_sampler_kwargs = {
        'eqmol':eqmol,
        'nsample':50,
        'initial_temperature':300
    }
    al_init_sampler = Sampler(sampler_function='wigner')
    al_init_sampler_kwargs = {
        'eqmol':eqmol,
        'nsample':100,
        'initial_temperature':300,
    }
    al_sampler = Sampler(sampler_function='md')
    sampler_kwargs = {
        'stop_function':stop_function,
        'stop_function_kwargs':{},
        'maximum_propagation_time':5000.0,
        'time_step':0.5,
        'md_parallel':False,
        'nthreads':16,
    }
    model_trainer = ml_model_trainer(ml_model_type='ani')
    active_learning(
        job_name='ethanol',
        initial_points_sampler=init_sampler,
        initial_points_sampler_kwargs=init_sampler_kwargs,
        initial_points_refinement='cross_validation',
        init_ntrain_next=50,
        init_train_energies_only=True,
        minimum_number_of_fitting_points=5,
        label_nthreads=16,
        ml_model_trainer=model_trainer,
        # ml_model_type='KREG',
        device='cuda',
        property_to_learn=['energy','energy_gradients'],
        property_to_check=['energy'],
        initial_conditions_sampler=al_init_sampler,
        initial_conditions_sampler_kwargs=al_init_sampler_kwargs,
        sampler=al_sampler,
        sampler_kwargs=sampler_kwargs,
        maximum_number_of_sampled_points=50,
        minimum_number_of_sampled_points=5,
        reference_method=ml.models.methods(method='B3LYP/6-31G*',program='Gaussian',nthreads=1,save_files_in_current_directory=False),
        # iteration=6
    )