import mlatom as ml 
import sys 
sys.path.append('~/scripts')
from al_general import active_learning,Sampler,stop_function,ml_model_trainer
import torch
import numpy as np

class wigner_geomopt_sampler(Sampler):
    def __init__(self):
        pass 

    def sample(self,**kwargs):
        # Arguments 
        if 'al_object' in kwargs:
            al_object = kwargs['al_object']
        if 'initial_eqmol' in kwargs:
            initial_eqmol = kwargs['initial_eqmol']
        if 'nsample' in kwargs:
            nsample = kwargs['nsample']
        if 'initial_temperature' in kwargs:
            initial_temperature = kwargs['initial_temperature']
        if 'program' in kwargs:
            program = kwargs['program']
        else:
            program='ase'
        if 'reference_method' in kwargs:
            reference_method = kwargs['reference_method']

        if not 'eqmol_db' in self.__dict__:
            self.eqmol_db = ml.data.molecular_database()
            self.eqmol_db += initial_eqmol

        if al_object.iteration >= 1:
            db_to_opt = ml.data.molecular_database.load(f'db_to_label_iteration{al_object.iteration-1}.json',format='json')
            self.geomopt(method=al_object.collective_models,program=program,initial_molecular_database=db_to_opt,reference_method=reference_method)

        moldb = super().wigner(eqmol=self.eqmol_db,nsample=nsample,initial_temperature=initial_temperature)

        return moldb


    def geomopt(self,**kwargs):
        if 'method' in kwargs:
            method = kwargs['method']
        if 'program' in kwargs:
            program = kwargs['program']
        else:
            program='ase'
        if 'initial_molecular_database' in kwargs:
            initial_molecular_database = kwargs['initial_molecular_database']
        if 'reference_method' in kwargs:
            reference_method = kwargs['reference_method']
        
        itraj = 0
        opt_moldb = ml.data.molecular_database()
        for mol in initial_molecular_database:
            itraj += 1 
            opt = ml.optimize_geometry(model=method,program=program,initial_molecule=mol,maximum_number_of_steps=200)
            # Get optimized geometry
            print(f"Optimization trajctory length of molecule {itraj}: {len(opt.optimization_trajectory.steps)}")
            if len(opt.optimization_trajectory.steps) <= 200:
                opt_moldb += opt.optimized_molecule
        print(f"There are {len(opt_moldb)} successful optimizations out of {len(initial_molecular_database)}")
        opt_moldb.dump('optimized_moldb.json',format='json')

        for imol in range(len(opt_moldb)):
            if not self.check_similarity(opt_moldb[imol]):
                print(f"ML model: Optimized molecule {imol+1} is a new conformer")
                mol_to_check = opt_moldb[imol].copy(atomic_labels=['xyz_coordinates'])
                opt = ml.optimize_geometry(model=reference_method,program='Gaussian',initial_molecule=mol_to_check)
                opt_mol = opt.optimized_molecule 
                if len(opt.optimization_trajectory.steps) <= 200:
                    if not self.check_similarity(opt_mol):
                        print(f'Reference method: Optimized molecule {imol+1} is a new conformer')
                        print(f"Add to conformer database")
                        freq = ml.freq(model=reference_method,molecule=opt_mol,program='Gaussian')
                        self.eqmol_db += opt_mol
                    else:
                        print(f'Reference method: Optimized molecule {imol+1} is not a new conformer')
                else:
                    print(f'Reference method: Optimized molecule {imol+1} is not a new conformer')
            else:
                print(f"ML model: Optimized molecule {imol+1} is not a new conformer")
        self.eqmol_db.dump("conformer_db.json",format='json')
            

    def check_similarity(self,mol):
        similar = False
        for eqmol in self.eqmol_db:
            if self.similar(eqmol,mol):
                similar = True
                break
        # if not similar: 
        #     self.eqmol_db += mol
        return similar

    def similar(self,mol1,mol2):
        try:
            value = ml.xyz.rmsd_reorder_check_reflection(mol1.element_symbols,mol2.element_symbols,mol1.xyz_coordinates,mol2.xyz_coordinates)
            if value > 0.125:
                return False
        except:
            pass
        return True


            

        
if __name__ == '__main__':
    print(torch.cuda.is_available())
    eqmol = ml.data.molecule() 
    eqmol.load('glycine_eqmol.json',format='json')
    init_sampler = Sampler(sampler_function='wigner')
    init_sampler_kwargs = {
        'eqmol':eqmol,
        'nsample':50,
        'initial_temperature':300
    }
    # al_init_sampler = Sampler(sampler_function='wigner')
    al_init_sampler = wigner_geomopt_sampler()
    al_init_sampler_kwargs = {
        'initial_eqmol':eqmol,
        'nsample':100,
        'initial_temperature':300,
        'program':'ase',
        'reference_method':ml.models.methods(method='B3LYP/6-31G*',program='Gaussian',nthreads=8,save_files_in_current_directory=False),
    }
    al_sampler = Sampler(sampler_function='md')
    sampler_kwargs = {
        'stop_function':stop_function,
        'stop_function_kwargs':{},
        'maximum_propagation_time':2000.0,
        'time_step':0.5,
        'md_parallel':False,
        'nthreads':8,
    }
    model_trainer = ml_model_trainer(ml_model_type='ani')
    active_learning(
        job_name='glycine',
        initial_points_sampler=init_sampler,
        initial_points_sampler_kwargs=init_sampler_kwargs,
        initial_points_refinement='cross_validation',
        init_ntrain_next=50,
        init_train_energies_only=True,
        minimum_number_of_fitting_points=5,
        label_nthreads=8,
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
        minimum_number_of_sampled_points=5,
        reference_method=ml.models.methods(method='B3LYP/6-31G*',program='Gaussian',nthreads=1,save_files_in_current_directory=False),
        # iteration=6
    )