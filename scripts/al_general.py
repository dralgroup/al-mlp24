#!/usr/bin/env python
import sys 
import mlatom as ml 
import numpy as np 
import os 
import matplotlib.pyplot as plt 
import scipy
import random 
import joblib
from multiprocessing.pool import ThreadPool as Pool
from joblib import Parallel, delayed
import timeit
import json

class active_learning():
    '''
    Active learning procedure 

    Arguments:
        job_name (str): Job name. Output files will be named after it.
        initial_points_sampler (:class:`Sampler`): Initial points sampler.
        initial_points_sampler_kwargs (dict): Initial points sampler kwargs.
        initial_points_refinement (str, optional): Initial points refinement method. Options: 'cross_validation', 'validation', 'one_shot'(by default).
        init_ncvsplits (int, optional): Number of CV splits for cross validation initial points sampling. 5 by default.
        init_validation_set_fraction (float): Fraction of validation set in validation initial points sampling. 0.2 by default.
        init_RMSE_threshold (float): RMSE threshold for intial points sampling. Stop sampling if RMSE is smaller than this threshold.
        init_train_energies_only (bool): Train on energies only when sampling initial points. False by default.
        minimum_number_of_fitting_points (int):
        init_ntrain_next (int, optional): 
        label_nthreads (int, optional): Number of threads for labelling new points. CPU count by default. 
        ml_model_type (str, optional): ML model type. Options: 'KREG'(by default), 'ANI'.
        ml_model_trainer (:class:`ml_model_trainer`): ML model trainer.
        collective_ml_models (:class:`collective_ml_models`): Collective ML models.
        device (str, optional): Device to train ANI model. Options: 'cuda'(by default), 'cpu'.
        property_to_learn (List[str]): Properties to learn in active learning.
        property_to_check (List[str]): Properties to check in uncertainty quantification.
        validation_set_fraction (float): Validation set fraction. 0.1 by default.
        initial_conditions_sampler (:class:`Sampler`)
        initial_conditions_sampler_kwargs (dict)
        sampler (:class:`Sampler`): Active learning sampler.
        sampler_kwargs (Dict): Kwargs of active learning sampler.
        uncertainty_quantification (:class:`uq`): UQ class that calculate UQ thresholds and UQs.
        maximum_number_of_sampled_points (int): Maximum number of sampled points in each iteration. Same as number_of_trajectories by default.
        minimum_number_of_sampled_points (int): Minimum number of sampled points in each iteration. If number of sampled points is less than this value, active learning is considered as converged. 1 by default.
        reference_method (:class:`ml.models.methods`): Reference method.

        init_sampling_only (bool): Exit after sampling initial points
    '''
    def __init__(self,**kwargs):
        # Options of actice learning
        if 'job_name' in kwargs:
            self.job_name = kwargs['job_name'] 

        # .Initial points sampling 
        # ..initial_points_sampler: Initial points sampler, should be a "sampler" class or a class that inherits "sampler". 
        if 'initial_points_sampler' in kwargs:
            self.initial_points_sampler = kwargs['initial_points_sampler']
        else:
            stopper('Initial points sampler not provided')
        # ..initial_points_sampler_kwargs: Initial points sampler kwargs, a dict with all the arguments.
        if 'initial_points_sampler_kwargs' in kwargs:
            self.initial_points_sampler_kwargs = kwargs['initial_points_sampler_kwargs']
        else:
            stopper('Initial points sampler kwargs not provided')
        # ..initial_points_refinement: Initial points refinement method, a string which specifies the method.
        #   ...Options:
        #       "cross_validation": Check the cross validation error and fit the learning curve. Keep adding points until doubling Ntr improves accuracy by less than 10%.
        #       "validation": Check the validation error and fit the learning curve. Keep adding points until doubling Ntr improves accuracy by less than 10%.
        #       "one_shot": Sample points only once, without checking anything. [default]
        if 'initial_points_refinement' in kwargs:
            self.initial_points_refinement = kwargs['initial_points_refinement']
        else:
            self.initial_points_refinement = 'one_shot'
        # ..init_ncvsplits: Number of cross validation splits, only works for initial_points_refinement="cross_validation". 5 by default.
        if 'init_ncvsplits' in kwargs:
            self.init_ncvsplits = kwargs['init_ncvsplits']
        else:
            self.init_ncvsplits = 5 
        # ..init_validation_set_fraction: Fraction of validation set in validation initial points sampling. 0.2 by default.
        if 'init_validation_set_fraction' in kwargs:
            self.init_validation_set_fraction = kwargs['init_validation_set_fraction']
        else:
            self.init_validation_set_fraction = 0.2
        # ..init_RMSE_threshold: RMSE threshold for intial points sampling. Stop sampling if RMSE is smaller than this threshold.
        if 'init_RMSE_threshold' in kwargs:
            self.init_RMSE_threshold = kwargs['init_RMSE_threshold']
        else:
            self.init_RMSE_threshold = None
        # ..init_train_energies_only: Train on energies only when sampling initial points. False by default.
        if 'init_train_energies_only' in kwargs:
            self.init_train_energies_only = kwargs['init_train_energies_only']
        else:
            self.init_train_energies_only = True
        # ..minimum_number_of_fitting_points
        if 'minimum_number_of_fitting_points' in kwargs:
            self.minimum_number_of_fitting_points = kwargs['minimum_number_of_fitting_points']
        else:
            self.minimum_number_of_fitting_points = 5
        # ..init_ntrain_next: Number of additional training points to check the convergence of training set, only works for initial_points_refinement="cross_validation". 
        if 'init_ntrain_next' in kwargs:
            self.init_ntrain_next = kwargs['init_ntrain_next']

        # .Nthreads 
        # ..label_nthreads: Number of processes used for labeling
        if 'label_nthreads' in kwargs:
            self.label_nthreads = kwargs['label_nthreads']
        else:
            self.label_nthreads = joblib.cpu_count()
        
        if 'model_predict_kwargs' in kwargs:
            self.model_predict_kwargs = kwargs['model_predict_kwargs']
        else:
            self.model_predict_kwargs = {}

        # .ML models
        # ..ml_model_type: ML model type
        #   ...Options:
        #       "KREG": Kernel ridge regression (KRR) with Gaussian kernel function and relative-to-equilibrium (RE) descriptor
        #       "ANI": Accurate neural network engine for molecular energies (ANAKIN-ME)
        #       "MACE": Fast and accurate machine learning interatomic potentials with higher order equivariant message passing
        if 'ml_model_type' in kwargs:
            self.ml_model_type = kwargs['ml_model_type']
        else:
            self.ml_model_type = 'kreg' # KREG or ANI
        # ..ml_model_trainer: ML model trainer 
        if 'ml_model_trainer' in kwargs:
            self.ml_model_trainer = kwargs['ml_model_trainer']
        else:
            self.ml_model_trainer = ml_model_trainer
        # ..collective_ml_models: Collective 
        if 'collective_ml_models' in kwargs: 
            self.collective_ml_models = kwargs['collective_ml_models']
        else:
            self.collective_ml_models = collective_ml_models
        # ..device: Use what device to train ML model 
        #   ...Options:
        #       "cpu": Use CPU
        #       "cuda": Use GPU (cuda)
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = 'cuda'
        # ..property_to_learn: A list of properties to learn with ML
        #   ...Note:
        #       If you want to learn value property and its corresponding gradient property, name the gradient property by adding "_gradient" suffix to the value property name, e.g., "energy" and "energy_gradients". This will create one single model trained on both values and gradients.
        if 'property_to_learn' in kwargs:
            self.property_to_learn = kwargs['property_to_learn']
        else:
            self.property_to_learn = ['energy','energy_gradients']
        # ..property_to_check: A list of properties to check by uncentainty quantification ("uq" class)
        if 'property_to_check' in kwargs:
            self.property_to_check = kwargs['property_to_check']
        else:
            self.property_to_check = ['energy'] # Property to check in UQ
        # ..validation_set_fraction: Fraction of the whole training set to be used as validation set.
        if 'validation_set_fraction' in kwargs:
            self.validation_set_fraction = kwargs['validation_set_fraction']
        else:
            self.validation_set_fraction = 0.1

        # .Sampler 
        # ..initial_conditions_sampler: Initial conditions sampler for sampler in active learning iterations
        if 'initial_conditions_sampler' in kwargs:
            self.initial_conditions_sampler = kwargs['initial_conditions_sampler']
        else:
            stopper("Please provide initial conditions sampler")
        # ..initial_conditions_sampler_kwargs: Kwargs for initial conditions sampler 
        if 'initial_conditions_sampler_kwargs' in kwargs:
            self.initial_conditions_sampler_kwargs = kwargs['initial_conditions_sampler_kwargs']
        else:
            self.initial_points_sampler_kwargs = {}
        # ..sampler: Sampler used in active learning iterations
        if 'sampler' in kwargs:
            self.sampler = kwargs['sampler']
        else:
            self.sampler = Sampler(sampler_function='md')
        # ..sampler_kwargs: Kwargs for sampler used in active learning iterations
        if 'sampler_kwargs' in kwargs:
            self.sampler_kwargs = kwargs['sampler_kwargs']
        else:
            self.sampler_kwargs = {}

        # .Uncertainty quantification 
        if 'uncertainty_quantification' in kwargs:
            self.uncertainty_quantification = kwargs['uncertainty_quantification']
        else:
            self.uncertainty_quantification = uq
        
        if 'uq_kwargs' in kwargs:
            self.uq_kwargs = kwargs['uq_kwargs']
        else:
            self.uq_kwargs = {}

        # .Molecular dynamics
        # ..maximum_number_of_sampled_points: Maximum number of sampled points in each iteration
        if 'maximum_number_of_sampled_points' in kwargs:
            self.maximum_number_of_sampled_points = kwargs['maximum_number_of_sampled_points']
        else:
            self.maximum_number_of_sampled_points = self.number_of_trajectories 
        # ..minimum_number_of_sampled_points: Minimum number of sampled points in each iteration
        if 'minimum_number_of_sampled_points' in kwargs:
            self.minimum_number_of_sampled_points = kwargs['minimum_number_of_sampled_points']
        else:
            self.minimum_number_of_sampled_points = 1  # Active learning is considered converged if number of sampled points is less than this value
        
        # .Reference method
        if 'reference_method' in kwargs:
            self.reference_method = kwargs['reference_method']

        # .Options for debugging and testing
        if 'init_sampling_only' in kwargs: 
            self.init_sampling_only = kwargs['init_sampling_only']
        else:
            self.init_sampling_only = False

        # Others
        self.hyperparameters = {}
        for property in self.property_to_learn:
            if property[-10:] == '_gradients':
                continue 
            self.hyperparameters[property] = {}
            self.hyperparameters['aux_'+property] = {}

        self.mlmodel = {}
        self.aux_mlmodel = {}

        self.validation_error_list = []
        self.Ntrain_list = []

        self.iteration = -1 
        while True:
            if os.path.exists(f'db_to_label_iteration{self.iteration+1}.json'):
                self.iteration += 1
            else:
                break

        self.main()
    
    def main(self):
        self.converged = False
        if self.iteration == -1:
            time0 = timeit.default_timer()
            self.get_initial_data_pool()
            time1 = timeit.default_timer() 
            print(f"Initial points sampling time: {time1-time0} s")
        else:
            print(f"Restarting active learning from iteration {self.iteration+1}")
            self.labeled_database = ml.data.molecular_database() 
            for ii in range(self.iteration+1):
                self.labeled_database += ml.data.molecular_database.load(f'labeled_db_iteration{ii}.json',format='json')

        if self.init_sampling_only:
            return
        
        while not self.converged:
            self.iteration += 1
            time0 = timeit.default_timer()
            print(f'Active learning iteration {self.iteration}')
            self.label_points()
            self.create_ml_model()
            self.use_ml_model()
            time1 = timeit.default_timer() 
            print(f"Iteration {self.iteration} takes {time1-time0} s")
            # exit()
            print('\n\n\n')
        
    def get_initial_data_pool(self):
        # Sample initial points if previous sampling is not found
        if not os.path.exists('init_cond_db.json'):
            # Get initial data pool (eqmol is not included)
            if self.initial_points_refinement.casefold() == 'cross_validation'.casefold():
                print(" Initial points sampling: Use cross validation")
                print(f" Initial points sampling: Number of CV splits = {self.init_ncvsplits}")
                self.get_initial_data_pool_cross_validation(self.initial_points_sampler,self.initial_points_sampler_kwargs,option=self.initial_points_refinement)
            elif self.initial_points_refinement.casefold() == 'validation'.casefold():
                print(" Initial points sampling: Use validation")
                print(f" Initial points sampling: Fraction of validation set = {self.init_validation_set_fraction}")
                self.get_initial_data_pool_cross_validation(self.initial_points_sampler,self.initial_points_sampler_kwargs,option=self.initial_points_refinement)
            elif self.initial_points_refinement.casefold() == 'one_shot'.casefold():
                self.get_initial_data_pool_one_shot(self.initial_points_sampler,self.initial_points_sampler_kwargs)
            else:
                stopper('Unrecognized intial points refinement method')

            self.init_cond_db.dump(filename='init_cond_db.json', format='json')
            self.molecular_pool_to_label = self.init_cond_db

    # Increase the number of initial points until cross validation error does not improve much
    def get_initial_data_pool_cross_validation(self,sampler,sampler_kwargs,option):
        sample_initial_conditions = True 
        self.init_cond_db = ml.data.molecular_database()
        # self.init_cond_db.molecules = [self.eqmol]
        Ntrain_list = []
        eRMSE_list = []
        def linear_fit_error(slope,intercept,x):
            return np.exp(intercept)*x**slope
        print("Start initial points sampling...")

        if self.init_RMSE_threshold is None:
            print(" Initial points samplimg: initial points RMSE threshold not found, fit learning curve instead")
        else:
            print(" Initial points samplimg: initial points RMSE threshold found, stop sampling if RMSE is smaller than threshold")
            print(f" Initial points sampling: RMSE threshold = {self.init_RMSE_threshold}")

        if self.init_train_energies_only:
            print(" Initial points sampling: Train on energies only")
        else:
            print(" Initial points sampling: Train on both energies and gradients")

        while sample_initial_conditions:
            init_cond_db = sampler.sample(al_object=self,**sampler_kwargs)
            self.label_points_moldb(method=self.reference_method,model_predict_kwargs=self.model_predict_kwargs,moldb=init_cond_db,nthreads=self.label_nthreads)
            fail_count = 0
            for init_mol in init_cond_db:
                if 'energy' in init_mol.__dict__ and 'energy_gradients' in init_mol.atoms[0].__dict__:
                    self.init_cond_db += init_mol
                else:
                    fail_count += 1 
                if fail_count != 0:
                    print(f"{fail_count} molecules are abandoned due to failed calculation")

            Ntrain_list.append(len(self.init_cond_db))
            if option.casefold() == 'cross_validation'.casefold():
                eRMSE_list.append(self.init_cross_validation())
            elif option.casefold() == 'validation'.casefold():
                eRMSE_list.append(self.init_validation())
            print(f"    Number of points: {Ntrain_list[-1]}")
            print(f"    eRMSE = {eRMSE_list[-1]} Hartree")
            if self.init_RMSE_threshold is None:
                if len(Ntrain_list) > 1 and len(Ntrain_list) >= self.minimum_number_of_fitting_points:
                    x = np.log(Ntrain_list)
                    y = np.log(eRMSE_list)
                    linreg = scipy.stats.linregress(x,y)
                    slope = linreg.slope 
                    intercept = linreg.intercept 
                    rvalue = linreg.rvalue
                    print(f'        Linear regression: log(e) = {intercept} + {slope} log(Ntr)')
                    print(f'        Linear regression: Pearson correlation coefficient = {rvalue}')
                    if slope > 0:
                        # If slope is larger than 0, skip this iteration
                        print("        Linear regression: slope is larger than 0")
                        continue 
                    else:
                        eNtr_next = linear_fit_error(slope,intercept,Ntrain_list[-1]+self.init_ntrain_next)
                        eNtr = linear_fit_error(slope,intercept,Ntrain_list[-1])
                        value = (eNtr-eNtr_next) / eNtr 
                        print(f'        [e(Ntr)-e(Ntr+{self.init_ntrain_next})]/e(Ntr) = {value}')
                        if value >= 0.1:
                            print('        Improvement is large, continue sampling')
                        else:
                            print('        Improvement is small, initial points sampling done')
                            print(f'    Number of initial points: {Ntrain_list[-1]}')
                            sample_initial_conditions = False 
            else:
                rmse = eRMSE_list[-1]
                if rmse >= self.init_RMSE_threshold:
                    print('        RMSE is larger than threshold, continue sampling')
                else:
                    print('        RMSE is smaller than threshold, initial points sampling done')
                    print(f'    Number of initial points: {Ntrain_list[-1]}')
                    sample_initial_conditions = False
            sys.stdout.flush()


    def init_cross_validation(self):
        ncvsplits = self.init_ncvsplits 
        cvsplits = ml.data.sample(molecular_database_to_split=self.init_cond_db,number_of_splits=ncvsplits,split_equally=True)
        moldb = ml.data.molecular_database()
        for isplit in range(ncvsplits):
            validation_molDB = cvsplits[isplit]
            subtraining_molDB = ml.data.molecular_database()
            for ii in range(ncvsplits):
                if ii != isplit:
                    subtraining_molDB += cvsplits[ii]
            if os.path.exists(f'{self.job_name}_kreg_initial_points.npz'):
                os.remove(f'{self.job_name}_kreg_initial_points.npz')
            if self.init_train_energies_only:
                mlmodel = self.ml_model_trainer.aux_model_trainer(filename=f'{self.job_name}_kreg_initial_points.npz',
                                                                subtraining_molDB=subtraining_molDB,
                                                                validation_molDB=validation_molDB,
                                                                property_to_learn='energy',
                                                                device=self.device)
            else:
                mlmodel = self.ml_model_trainer.main_model_trainer(filename=f'{self.job_name}_kreg_initial_points.npz',
                                                                subtraining_molDB=subtraining_molDB,
                                                                validation_molDB=validation_molDB,
                                                                property_to_learn='energy',
                                                                xyz_derivative_property_to_learn='energy_gradients',
                                                                device=self.device)
            mlmodel.predict(molecular_database=validation_molDB, property_to_predict='estimated_energy',xyz_derivative_property_to_predict='estimated_energy_gradients')
            moldb += validation_molDB 
        energies = moldb.get_properties('energy')
        estimated_energies = moldb.get_properties('estimated_energy')
        eRMSE = ml.stats.rmse(energies,estimated_energies)
        return eRMSE
    
    def init_validation(self):
        fraction = self.init_validation_set_fraction
        subtraining_molDB, validation_molDB = self.init_cond_db.split(number_of_splits=2,fraction_of_points_in_splits=[1-fraction,fraction],sampling='random')
        if os.path.exists(f'{self.job_name}_kreg_initial_points.npz'):
            os.remove(f'{self.job_name}_kreg_initial_points.npz')
        if self.init_train_energies_only:
            mlmodel = self.ml_model_trainer.aux_model_trainer(filename=f'{self.job_name}_kreg_initial_points.npz',
                                                                subtraining_molDB=subtraining_molDB,
                                                                validation_molDB=validation_molDB,
                                                                property_to_learn='energy',
                                                                device=self.device)
        else:
            mlmodel = self.ml_model_trainer.main_model_trainer(filename=f'{self.job_name}_kreg_initial_points.npz',
                                                                subtraining_molDB=subtraining_molDB,
                                                                validation_molDB=validation_molDB,
                                                                property_to_learn='energy',
                                                                xyz_derivative_property_to_learn='energy_gradients',
                                                                device=self.device)
        mlmodel.predict(molecular_database=validation_molDB, property_to_predict='estimated_energy',xyz_derivative_property_to_predict='estimated_energy_gradients')
        energies = validation_molDB.get_properties('energy')
        estimated_energies = validation_molDB.get_properties('estimated_energy')
        eRMSE = ml.stats.rmse(energies,estimated_energies)
        return eRMSE

    # Sample intial points only once 
    def get_initial_data_pool_one_shot(self,sampler,sampler_kwargs):
        self.init_cond_db = ml.data.molecular_database()
        init_cond_db = sampler.sample(al_object=self,**sampler_kwargs)
        self.label_points_moldb(method=self.reference_method,model_predict_kwargs=self.model_predict_kwargs,moldb=init_cond_db,nthreads=self.label_nthreads)
        fail_count = 0
        for init_mol in init_cond_db:
            if 'energy' in init_mol.__dict__ and 'energy_gradients' in init_mol.atoms[0].__dict__:
                self.init_cond_db += init_mol
            else:
                fail_count += 1 
            if fail_count != 0:
                print(f"{fail_count} molecules are abandoned due to failed calculation")

    # Label points
    def label_points(self):
        if not 'labeled_database' in self.__dict__:
            self.labeled_database = ml.data.molecular_database()
        if not 'molecular_pool_to_label' in self.__dict__: 
            self.molecular_pool_to_label = ml.data.molecular_database()
            if self.iteration == 0:
                print("Loading existing initial condition database")
                self.molecular_pool_to_label = ml.data.molecular_database.load(filename=f'init_cond_db.json', format='json')
            elif os.path.exists(f'db_to_label_iteration{self.iteration-1}.json'): self.molecular_pool_to_label = ml.data.molecular_database.load(filename=f'db_to_label_iteration{self.iteration-1}.json', format='json')
    
        nmols = len(self.molecular_pool_to_label)
        labeled_database_iteration = ml.data.molecular_database() 
        if nmols > 0:
            self.label_points_moldb(method=self.reference_method,model_predict_kwargs=self.model_predict_kwargs,moldb=self.molecular_pool_to_label,nthreads=self.label_nthreads)
            for mol in self.molecular_pool_to_label:
                if 'energy' in mol.__dict__ and 'energy_gradients' in mol.atoms[0].__dict__:
                    self.labeled_database.molecules.append(mol)
                    labeled_database_iteration.molecules.append(mol)
            self.labeled_database.dump(filename='labeled_db.json', format='json')
            labeled_database_iteration.dump(filename=f'labeled_db_iteration{self.iteration}.json',format='json')
            print('Points to label:', len(self.molecular_pool_to_label.molecules))
            print('New labeled points:', len(labeled_database_iteration.molecules))
            print(f'{len(self.molecular_pool_to_label.molecules) - len(labeled_database_iteration.molecules)} points are abandoned due to failed calculation')
            print('Number of points in the labeled data set:', len(self.labeled_database.molecules))
    
    
    def create_ml_model(self):
        if not 'labeled_database' in self.__dict__:
            self.labeled_database = ml.data.molecular_database()
            if os.path.exists('labeled_db.json'): self.labeled_database.load(filename='labeled_db.json', format='json')
        if len(self.labeled_database.molecules) == 0: return

        # def train_models(nmember=None,sigma=None,lmbd=None):

        # Split training set into subtraining set and validation set
        if not os.path.exists(f'training_db_iteration{self.iteration}.json'):
            [self.subtraining_molDB, self.validation_molDB] = self.labeled_database.split(number_of_splits=2, fraction_of_points_in_splits=[1-self.validation_set_fraction, self.validation_set_fraction], sampling='random')
            # First Nsubtrain points in training_molDB are subtraining points
            self.training_molDB = ml.data.molecular_database() 
            self.training_molDB.molecules = self.subtraining_molDB.molecules + self.validation_molDB.molecules 
            # self.training_molDB.Nsubtrain = len(self.subtraining_molDB)
            # self.training_molDB.Nvalidate = len(self.validation_molDB)
            self.training_molDB.dump(filename=f'training_db_iteration{self.iteration}.json',format='json')
        else:
            print(f"Training set training_db_iteration{self.iteration}.json found")
            self.training_molDB = ml.data.molecular_database.load(f'training_db_iteration{self.iteration}.json',format='json')
            Nsubtrain = round(len(self.training_molDB)*(1-self.validation_set_fraction))
            self.subtraining_molDB = self.training_molDB[:Nsubtrain]
            self.validation_molDB = self.training_molDB[Nsubtrain:]

        self.Ntrain_list.append(len(self.training_molDB))

        for property in self.property_to_learn:
            if property[-10:] == '_gradients':
                continue
            if property+'_gradients' in self.property_to_learn:
                xyz_derivative_property = property+'_gradients'
            else:
                xyz_derivative_property = None
            
            # Training main model
            self.ml_model_trainer.hyperparameters = self.hyperparameters[property]
            mlmodel = self.ml_model_trainer.main_model_trainer(filename=f'{self.job_name}_{property}_iteration{self.iteration}.npz',
                                                     subtraining_molDB=self.subtraining_molDB,
                                                     validation_molDB=self.validation_molDB,
                                                     property_to_learn=property,
                                                     xyz_derivative_property_to_learn=xyz_derivative_property,
                                                     device=self.device)
            self.hyperparameters[property] = self.ml_model_trainer.hyperparameters
            # Training auxiliary model
            self.ml_model_trainer.hyperparameters = self.hyperparameters['aux_'+property]
            aux_mlmodel = self.ml_model_trainer.aux_model_trainer(filename=f'{self.job_name}_{property}_aux_iteration{self.iteration}.npz',
                                                     subtraining_molDB=self.subtraining_molDB,
                                                     validation_molDB=self.validation_molDB,
                                                     property_to_learn=property,
                                                     device=self.device)
            self.hyperparameters['aux_'+property] = self.ml_model_trainer.hyperparameters

            self.mlmodel[property] = mlmodel 
            self.aux_mlmodel[property] = aux_mlmodel

        # Get thresholds
        self.plotScatter()
        self.get_uq_thresholds(self.validation_molDB)


    def use_ml_model(self):
        # For parallel execution 
        for each in self.mlmodel.values():
            each.nthreads=1 
        for each in self.aux_mlmodel.values():
            each.nthreads=1

        self.collective_models = self.collective_ml_models(property_to_learn=self.property_to_learn,
                                      mlmodels=self.mlmodel,
                                      aux_mlmodels=self.aux_mlmodel)
        init_cond_db = self.initial_conditions_sampler.sample(al_object=self,**self.initial_conditions_sampler_kwargs)

        # Grab points to label from trajectories 
        self.molecular_pool_to_label = ml.data.molecular_database() 

        self.molecular_pool_to_label = self.sampler.sample(al_object=self,method=self.collective_models,
                                                           initial_molecular_database=init_cond_db,
                                                           uncertainty_quantification=self.uq,
                                                           iteration=self.iteration,
                                                           **self.sampler_kwargs)

        for imol in range(len(self.molecular_pool_to_label.molecules)):
            self.molecular_pool_to_label.molecules[imol] = self.molecular_pool_to_label.molecules[imol].copy(atomic_labels=['xyz_coordinates','xyz_velocities'],molecular_labels=[])
        self.original_number_of_molecules_to_label = len(self.molecular_pool_to_label.molecules)
        if self.original_number_of_molecules_to_label > self.maximum_number_of_sampled_points:
            self.molecular_pool_to_label.molecules = random.sample(self.molecular_pool_to_label.molecules, self.maximum_number_of_sampled_points)
            print(f'Number of points to be labeled is larger than {self.maximum_number_of_sampled_points}, sample {self.maximum_number_of_sampled_points} points from them')
        if self.minimum_number_of_sampled_points >= 1:
            if len(self.molecular_pool_to_label) < self.minimum_number_of_sampled_points:
                print(f'Number of points to be labeled is less than {self.minimum_number_of_sampled_points}, active learning converged')
                self.converged = True
        else:
            if len(self.molecular_pool_to_label) / len(init_cond_db) < self.minimum_number_of_sampled_points:
                print(f'Number of points to be labeled is less than {self.minimum_number_of_sampled_points*100}%, active learning converged')
                self.converged = True
        self.molecular_pool_to_label.dump(filename=f'db_to_label_iteration{self.iteration}.json', format='json')
        sys.stdout.flush()

    # Label points
    def label_points_moldb(self,method,model_predict_kwargs={},moldb=None,calculate_energy=True,calculate_energy_gradients=True,calculate_hessian=False,nthreads=1):
        '''
        function labeling points in molecular database

        Arguments:
            method (:class:`ml.models.model`): method that provides energies, energy gradients, etc.
            moldb (:class:`ml.data.molecular_database`): molecular database to label
            calculate_energy (bool): calculate energy
            calculate_energy_gradients (bool): calculate_energy_gradients 
            calculate_hessian (bool): calculate Hessian
            nthreads (int): number of threads
        '''
        def label(imol):
            mol2label = moldb[imol]
            if not ('energy' in mol2label.__dict__ and 'energy_gradients' in mol2label[0].__dict__):
                method.predict(molecule=mol2label,calculate_energy=calculate_energy,calculate_energy_gradients=calculate_energy_gradients,calculate_hessian=calculate_hessian,**model_predict_kwargs)
            return mol2label
        
        nmols = len(moldb)
        if nthreads > 1:
            pool = Pool(processes=nthreads)
            mols = pool.map(label,list(range(nmols)))
        else:
            method.predict(moldb,calculate_energy=calculate_energy,calculate_energy_gradients=calculate_energy_gradients,calculate_hessian=calculate_hessian,**model_predict_kwargs)
    
    def get_uq_thresholds(self,moldb):
        if not 'uq' in self.__dict__:
            self.uq = self.uncertainty_quantification(property_uqs=self.property_to_check,**self.uq_kwargs)
        for property in self.property_to_learn:
            if property[-10:] == '_gradients':
                continue
            if property+'_gradients' in self.property_to_learn:
                estimated_xyz_derivative_property = 'estimated_'+property+'_gradients'
                aux_estimated_xyz_derivative_property = 'aux_estimated_'+property+'_gradients'
            else:
                estimated_xyz_derivative_property = None
                aux_estimated_xyz_derivative_property = None
            
            # Predict 
            self.mlmodel[property].predict(molecular_database=moldb,property_to_predict='estimated_'+property,xyz_derivative_property_to_predict=estimated_xyz_derivative_property)
            self.aux_mlmodel[property].predict(molecular_database=moldb,property_to_predict='aux_estimated_'+property,xyz_derivative_property_to_predict=aux_estimated_xyz_derivative_property)

        self.uq.calculate_uq_thresholds(moldb)
        self.uq.update_threshold_settings(len(self.labeled_database))

    # Generate scatter plots of estimated values vs reference values 
    # Print eRMSE, fRMSE and correlation coefficients
    def plotScatter(self):
        training_molDB = ml.data.molecular_database.load(filename=f'training_db_iteration{self.iteration}.json',format='json')
        Ntrain = len(training_molDB)
        Nsubtrain = int(Ntrain*(1.0-self.validation_set_fraction))
        Nvalidate = Ntrain-Nsubtrain

        print(f'Iteration {self.iteration}:')
        print(f'    Number of training points: {Ntrain}')
        print(f'        Number of subtraining points: {Nsubtrain}')
        print(f'        Number of validation points: {Nvalidate}')

        for property in self.property_to_learn:
            if property[-10:] == '_gradients':
                continue 
            if property+'_gradients' in self.property_to_learn:
                xyz_derivative_property = property+'_gradients'
                estimated_xyz_derivative_property = 'estimated_'+property+'_gradients' 
            else:
                xyz_derivative_property = None
                estimated_xyz_derivative_property = None
            if self.device == 'cuda':
                predict_kwargs = {'batch_size':100}
            else:
                predict_kwargs = {}
            self.mlmodel[property].predict(molecular_database=training_molDB,
                                           property_to_predict='estimated_'+property,
                                           xyz_derivative_property_to_predict=estimated_xyz_derivative_property,
                                           **predict_kwargs)
            self.aux_mlmodel[property].predict(molecular_database=training_molDB,
                                               property_to_predict='aux_estimated_'+property,
                                               **predict_kwargs)
            
            values = training_molDB.get_properties(property)
            estimated_values = training_molDB.get_properties('estimated_'+property)
            aux_estimated_values = training_molDB.get_properties('aux_estimated_'+property)
            if not xyz_derivative_property is None:
                gradients = training_molDB.get_xyz_vectorial_properties(xyz_derivative_property)
                estimated_gradients = training_molDB.get_xyz_vectorial_properties(estimated_xyz_derivative_property)
            Natoms = len(training_molDB.molecules[0])

            # Evaluate main model performance 
            # .RMSE of values
            main_model_subtrain_vRMSE = ml.stats.rmse(estimated_values[:Nsubtrain],values[:Nsubtrain])
            main_model_validate_vRMSE = ml.stats.rmse(estimated_values[Nsubtrain:],values[Nsubtrain:])
            # .Pearson correlation coefficient of values
            main_model_subtrain_vPCC = ml.stats.correlation_coefficient(estimated_values[:Nsubtrain],values[:Nsubtrain])
            main_model_validate_vPCC = ml.stats.correlation_coefficient(estimated_values[Nsubtrain:],values[Nsubtrain:])
            if not xyz_derivative_property is None:
                # # .RMSE of gradients 
                # main_model_subtrain_gRMSE = ml.stats.rmse(estimated_gradients[:Nsubtrain].reshape(Nsubtrain*Natoms*3),gradients[:Nsubtrain].reshape(Nsubtrain*Natoms*3))
                # main_model_validate_gRMSE = ml.stats.rmse(estimated_gradients[Nsubtrain:].reshape(Nvalidate*Natoms*3),gradients[Nsubtrain:].reshape(Nvalidate*Natoms*3))
                # # .Pearson correlation coeffcient of gradients 
                # main_model_subtrain_gPCC = ml.stats.correlation_coefficient(estimated_gradients[:Nsubtrain].reshape(Nsubtrain*Natoms*3),gradients[:Nsubtrain].reshape(Nsubtrain*Natoms*3))
                # main_model_validate_gPCC = ml.stats.correlation_coefficient(estimated_gradients[Nsubtrain:].reshape(Nvalidate*Natoms*3),gradients[Nsubtrain:].reshape(Nvalidate*Natoms*3))
                # .RMSE of gradients 
                main_model_subtrain_gRMSE = ml.stats.rmse(estimated_gradients[:Nsubtrain].flatten(),gradients[:Nsubtrain].flatten())
                main_model_validate_gRMSE = ml.stats.rmse(estimated_gradients[Nsubtrain:].flatten(),gradients[Nsubtrain:].flatten())
                # .Pearson correlation coeffcient of gradients 
                main_model_subtrain_gPCC = ml.stats.correlation_coefficient(estimated_gradients[:Nsubtrain].flatten(),gradients[:Nsubtrain].flatten())
                main_model_validate_gPCC = ml.stats.correlation_coefficient(estimated_gradients[Nsubtrain:].flatten(),gradients[Nsubtrain:].flatten())
            # Evaluate auxiliary model performance
            # .RMSE of values 
            aux_model_subtrain_vRMSE = ml.stats.rmse(aux_estimated_values[:Nsubtrain],values[:Nsubtrain])
            aux_model_validate_vRMSE = ml.stats.rmse(aux_estimated_values[Nsubtrain:],values[Nsubtrain:])
            # .Pearson correlation coefficient of values
            aux_model_subtrain_vPCC = ml.stats.correlation_coefficient(aux_estimated_values[:Nsubtrain],values[:Nsubtrain])
            aux_model_validate_vPCC = ml.stats.correlation_coefficient(aux_estimated_values[Nsubtrain:],values[Nsubtrain:])

            print(f"    {property}")
            print("        Main model")
            print("            Subtraining set:")
            print(f"                RMSE of values = {main_model_subtrain_vRMSE}")
            print(f"                Correlation coefficient = {main_model_subtrain_vPCC}")
            if not xyz_derivative_property is None:
                print(f"                RMSE of gradients = {main_model_subtrain_gRMSE}")
                print(f"                Correlation coefficient = {main_model_subtrain_gPCC}")
            print("            Validation set:")
            print(f"                RMSE of values = {main_model_validate_vRMSE}")
            print(f"                Correlation coefficient = {main_model_validate_vPCC}")
            if not xyz_derivative_property is None:
                print(f"                RMSE of gradients = {main_model_validate_gRMSE}")
                print(f"                Correlation coefficient = {main_model_validate_gPCC}")
            print("        Auxiliary model")
            print("            Subtraining set:")
            print(f"                RMSE of values = {aux_model_subtrain_vRMSE}")
            print(f"                Correlation coefficient = {aux_model_subtrain_vPCC}")
            print("            Validation set:")
            print(f"                RMSE of values = {aux_model_validate_vRMSE}")
            print(f"                Correlation coefficient = {aux_model_validate_vPCC}")

            # Value scatter plot of the main model
            fig,ax = plt.subplots() 
            fig.set_size_inches(15,12)
            diagonal_line = [min([min(values),min(estimated_values)]),max([max(values),max(estimated_values)])]
            ax.plot(diagonal_line,diagonal_line,color='C3')
            ax.scatter(values[0:Nsubtrain],estimated_values[0:Nsubtrain],color='C0',label='subtraining points')
            ax.scatter(values[Nsubtrain:Ntrain],estimated_values[Nsubtrain:Ntrain],color='C1',label='validation points')
            ax.set_xlabel(f'{property} (Hartree)')
            ax.set_ylabel(f'Estimated {property} (Hartree)')
            plt.suptitle(f'Iteration {self.iteration} ({property})')
            plt.legend()
            plt.savefig(f'Iteration{self.iteration}_{property}.png',dpi=300)
            fig.clear()
            # Gradient scatter plot of the main model 
            fig,ax = plt.subplots()
            fig.set_size_inches(15,12)
            diagonal_line = [min([np.min(gradients),np.min(estimated_gradients)]),max([np.max(gradients),np.max(estimated_gradients)])]
            ax.plot(diagonal_line,diagonal_line,color='C3')
            ax.scatter(gradients[0:Nsubtrain].flatten(),estimated_gradients[0:Nsubtrain].flatten(),color='C0',label='subtraining points')
            ax.scatter(gradients[Nsubtrain:Ntrain].flatten(),estimated_gradients[Nsubtrain:Ntrain].flatten(),color='C1',label='validation points')
            ax.set_xlabel(f'{property} gradients (Hartree/Angstrom)')
            ax.set_ylabel(f'Estimated {property} gradients (Hartree/Angstrom)')
            ax.set_title(f'Iteration {self.iteration} ({property} gradients)')
            plt.legend()
            plt.savefig(f'Iteration{self.iteration}_{property}_gradients.png',dpi=300)
            fig.clear()

            # Value scatter plot of the auxiliary model
            fig,ax = plt.subplots() 
            fig.set_size_inches(15,12)
            diagonal_line = [min([min(values),min(aux_estimated_values)]),max([max(values),max(aux_estimated_values)])]
            ax.plot(diagonal_line,diagonal_line,color='C3')
            ax.scatter(values[0:Nsubtrain],aux_estimated_values[0:Nsubtrain],color='C0',label='subtraining points')
            ax.scatter(values[Nsubtrain:Ntrain],aux_estimated_values[Nsubtrain:Ntrain],color='C1',label='validation points')
            ax.set_xlabel(f'{property} (Hartree)')
            ax.set_ylabel(f'Estimated {property} from auxiliary model (Hartree)')
            ax.set_title(f'Iteration {self.iteration} ({property} of auxiliary model)')
            plt.legend()
            plt.savefig(f'Iteration{self.iteration}_aux_{property}.png',dpi=300)





class Sampler():
    def __init__(self,sampler_function=None):
        if type(sampler_function) == str:
            if sampler_function.casefold() == 'wigner'.casefold():
                self.sampler_function = self.wigner 
            elif sampler_function.casefold() == 'geomopt'.casefold():
                self.sampler_function = self.geometry_optimization 
            elif sampler_function.casefold() == 'md'.casefold():
                self.sampler_function = self.molecular_dynamics
            elif sampler_function.casefold() == 'random'.casefold():
                self.sampler_function = self.random
            elif sampler_function.casefold() == 'fixed'.casefold():
                self.sampler_function = self.fixed
            elif sampler_function.casefold() == 'harmonic-quantum-boltzmann'.casefold():
                self.sampler_function = self.harmonic_quantum_boltzmann
            else:
                stopper(f"Unsupported sampler function type: {sampler_function}")
        else:
            self.sampler_function = sampler_function

    def sample(self,**kwargs):
        return self.sampler_function(**kwargs)
    
    def fixed(self,**kwargs):
        if 'molecule' in kwargs:
            molecule = kwargs['molecule']
        else:
            molecule = None 
        if 'molecular_database' in kwargs:
            molecular_database = kwargs['molecular_database']
        else:
            molecular_database = None 
        if molecule is None and molecular_database is None:
            stopper("Sampler(fixed): Neither molecule nor molecular_database is provided")
        elif not molecule is None and not molecular_database is None:
            stopper("Sampler(fixed): Both molecule and molecular_database are provided")
        elif not molecule is None:
            molecular_database = ml.data.molecular_database()
            molecular_database.append(molecule)
        elif not molecular_database is None:
            pass 

        return molecular_database

    def random(self,**kwargs):
        if 'initial_molecule' in kwargs:
            initial_molecule = kwargs['initial_molecule']
        else:
            stopper("Sampler(random): Please specify intitial molecule for random sampling")
        if 'nsample' in kwargs:
            nsample = kwargs['nsample']
        else:
            stopper("Sampler(random): Please provide number of points to sample for random sampling")
        # if 'metric' in kwargs:
        #     metric = kwargs['metric']
        if 'scale' in kwargs:
            scale = kwargs['scale']
        else:
            scale = 0.1

        moldb = ml.data.molecular_database()
        rand = (np.random.rand(nsample,len(initial_molecule),3) - 0.5) * scale
        mol_xyz = initial_molecule.xyz_coordinates 
        rand_xyz = rand + mol_xyz 
        for each in rand_xyz:
            mol = initial_molecule.copy() 
            mol.xyz_coordinates = each 
            moldb.append(mol)
        return moldb


    def wigner(self,**kwargs):
        if 'eqmol' in kwargs:
            eqmol = kwargs['eqmol']
        else:
            stopper('Sampler(wigner): Please specify equiilibrium geometry for wigner sampling')
        if 'nsample' in kwargs:
            nsample = kwargs['nsample']
        else:
            stopper('Sampler(wigner): Please provide number of points to sample for wigner sampling')
        if 'initial_temperature' in kwargs:
            initial_temperature = kwargs['initial_temperature']
        else:
            initial_temperature = None
        if 'use_hessian' in kwargs:
            use_hessian = kwargs['use_hessian']
        else:
            use_hessian = True
        if isinstance(eqmol,ml.data.molecule):
            moldb = ml.generate_initial_conditions(molecule=eqmol,
                                                generation_method='wigner',
                                                number_of_initial_conditions=nsample,
                                                initial_temperature=initial_temperature,
                                                use_hessian=use_hessian)
        elif isinstance(eqmol,ml.data.molecular_database):
            nmols = len(eqmol)
            nsample_each = nsample // nmols 
            nremaining = nsample % nmols 
            nsample_list = [nsample_each for imol in range(nmols)]
            for ii in range(nremaining):
                nsample_list[ii] += 1 
            moldb = ml.data.molecular_database() 
            # print(nsample_list)
            for imol in range(nmols):
                mol = eqmol[imol]
                moldb_each = ml.generate_initial_conditions(molecule=mol,
                                                                    generation_method='wigner',
                                                                    number_of_initial_conditions=nsample_list[imol],
                                                                    initial_temperature=initial_temperature,
                                                                    use_hessian=use_hessian)
                moldb += moldb_each
                # print(len(moldb))
        return moldb 
    
    def geometry_optimization(self,**kwargs):
        if 'method' in kwargs:
            method = kwargs['method']
        else:
            stopper('Sampler(geomopt): Please provide method for geometry optimization sampling')
        if 'stop_function' in kwargs:
            stop_function = kwargs['stop_function']
        else:
            stop_function = None
        if 'stop_function_kwargs' in kwargs:
            stop_function_kwargs = kwargs['stop_function_kwargs']
        else:
            stop_function_kwargs = {}
        if 'initial_molecular_database' in kwargs:
            initial_molecular_database = kwargs['initial_molecular_database']
        else:
            stopper("Sampler(geomopt): Please provide initial molecular database")
        if 'program' in kwargs:
            program = kwargs['program']
        else:
            program = 'ase'
        if 'uncertainty_quantification' in kwargs:
            uncertainty_quantification = kwargs['uncertainty_quantification']
        else:
            stopper("Sampler(geomopt): Please provide uncertainty quantification")

        properties = uncertainty_quantification.property_uqs
        thresholds = uncertainty_quantification.uq_thresholds

        if not 'thresholds' in stop_function_kwargs:
            stop_function_kwargs['thresholds'] = thresholds
        if not 'properties' in stop_function_kwargs:
            stop_function_kwargs['properties'] = properties

        
        moldb = ml.data.molecular_database() 
        opt_moldb = ml.data.molecular_database()
        itraj = 0
        for mol in initial_molecular_database:
            itraj += 1
            opt = ml.optimize_geometry(model=method,program=program,initial_molecule=mol)
            opt_moldb.append(opt.optimized_molecule)
            if not stop_function is None:
                for istep in range(len(opt.optimization_trajectory.steps)):
                    step = opt.optimization_trajectory.steps[istep]
                    stop = stop_function(step.molecule,**stop_function_kwargs)
                    if stop:
                        if 'need_to_be_labeled' in step.molecule.__dict__:
                            print(f'Adding molecule from trajectory {itraj} at step {istep}')
                            moldb.append(step.molecule)
                            break
            else:
                moldb.append(opt.optimized_molecule)
        opt_moldb.dump(filename='geomopt_db.json',format='json')
        return moldb 
    
    def molecular_dynamics(self,**kwargs):
        if 'method' in kwargs:
            method = kwargs['method']
        else:
            stopper('Sampler(md): Please provide method for molecular dynamics sampling')
        if 'stop_function' in kwargs:
            stop_function = kwargs['stop_function']
        else:
            stop_function = None
        if 'stop_function_kwargs' in kwargs:
            stop_function_kwargs = kwargs['stop_function_kwargs']
        else:
            stop_function_kwargs = {}
        if 'initial_molecular_database':
            initial_molecular_database = kwargs['initial_molecular_database']
        else:
            stopper("Sampler(md): Please provide initial molecular database for MD")
        if 'maximum_propagation_time' in kwargs:
            maximum_propagation_time = kwargs['maximum_propagation_time']
        else:
            stopper("Sampler(md): Please provide maximum propagation time for MD")
        if 'time_step' in kwargs:
            time_step = kwargs['time_step']
        else:
            stopper("Sampler(md): Please provide time step for MD")
        if 'md_parallel' in kwargs:
            md_parallel = kwargs['md_parallel']
        else:
            md_parallel = True
        if 'nthreads' in kwargs:
            nthreads = kwargs['nthreads']
        else:
            nthreads = joblib.cpu_count()
        if 'uncertainty_quantification' in kwargs:
            uncertainty_quantification = kwargs['uncertainty_quantification']
        else:
            stopper("Sampler(md): Please provide uncertainty quantification")
        
        if not 'thresholds' in stop_function_kwargs:
            stop_function_kwargs['thresholds'] = uncertainty_quantification.uq_thresholds
        if not 'properties' in stop_function_kwargs:
            stop_function_kwargs['properties'] = uncertainty_quantification.property_uqs

        moldb = ml.data.molecular_database()
        if md_parallel:
            dyn = ml.md_parallel(model=method,
                                 molecular_database=initial_molecular_database,
                                 ensemble='NVE',
                                 time_step=time_step,
                                 maximum_propagation_time=maximum_propagation_time,
                                 dump_trajectory_interval=None,
                                 stop_function=stop_function,
                                 stop_function_kwargs=stop_function_kwargs)
            trajs = dyn.molecular_trajectory 
            for itraj in range(len(trajs.steps[0])):
                print(f"Trajectory {itraj} number of steps: {trajs.traj_len[itraj]}")
                if 'need_to_be_labeled' in trajs.steps[trajs.traj_len[itraj]][itraj].__dict__:
                    print(f'Adding molecule from trajectory {itraj} at time {trajs.traj_len[itraj]*time_step} fs')
                    moldb.molecules.append(trajs.steps[trajs.traj_len[itraj]][itraj])
        else:
            def run_traj(imol):
                initmol = initial_molecular_database.molecules[imol]

                dyn = ml.md(model=method,
                            molecule_with_initial_conditions=initmol,
                            ensemble='NVE',
                            time_step=time_step,
                            maximum_propagation_time=maximum_propagation_time,
                            dump_trajectory_interval=None,
                            stop_function=stop_function,
                            stop_function_kwargs=stop_function_kwargs)
                traj = dyn.molecular_trajectory 
                return traj 
            
            trajs = Parallel(n_jobs=nthreads)(delayed(run_traj)(i) for i in range(len(initial_molecular_database)))
            sys.stdout.flush() 

            itraj=0 
            for traj in trajs:
                itraj+=1 
                print(f"Trajectory {itraj} number of steps: {len(traj.steps)}")
                if 'need_to_be_labeled' in traj.steps[-1].molecule.__dict__:# and len(traj.steps) > 1:
                    print('Adding molecule from trajectory %d at time %.2f fs' % (itraj, traj.steps[-1].time))
                    moldb.molecules.append(traj.steps[-1].molecule)
        return moldb
    
    def harmonic_quantum_boltzmann(self,**kwargs):
        if 'eqmol' in kwargs:
            eqmol = kwargs['eqmol']
        else:
            stopper('Sampler(harmonic_quantum_boltzmann): Please specify equiilibrium geometry for harmonic quantum Boltzmann sampling')
        if 'nsample' in kwargs:
            nsample = kwargs['nsample']
        else:
            stopper('Sampler(harmonic_quantum_boltzmann): Please provide number of points to sample for harmonic quantum Boltzmann sampling')
        if 'initial_temperature' in kwargs:
            initial_temperature = kwargs['initial_temperature']
        else:
            stopper('Sampler(harmonic_quantum_boltzmann): Please provide initial temperature for harmonic quantum Boltzmann sampling')
        if 'use_hessian' in kwargs:
            use_hessian = kwargs['use_hessian']
        else:
            use_hessian = True
        
        if isinstance(eqmol,ml.data.molecule):
            moldb = ml.generate_initial_conditions(molecule=eqmol,
                                                   generation_method='harmonic-quantum-boltzmann',
                                                   number_of_initial_conditions=nsample,
                                                   initial_temperature=initial_temperature,
                                                   use_hessian=use_hessian)
        elif isinstance(eqmol,ml.data.molecular_database):
            nmols = len(eqmol)
            nsample_each = nsample // nmols 
            nremaining = nsample % nmols 
            nsample_list = [nsample_each for imol in range(nmols)]
            for ii in range(nremaining):
                nsample_list[ii] += 1 
            moldb = ml.data.molecular_database() 
            # print(nsample_list)
            for imol in range(nmols):
                mol = eqmol[imol]
                moldb_each = ml.generate_initial_conditions(molecule=mol,
                                                            generation_method='harmonic-quantum-boltzmann',
                                                            number_of_initial_conditions=nsample_list[imol],
                                                            initial_temperature=initial_temperature,
                                                            use_hessian=use_hessian)
                moldb += moldb_each
        return moldb 


class uq():
    '''
    Uncertainty quantification class which deals with UQ thresholds calculations, updation ... This is a template of UQ.

    Arguments:
        property_uqs (List[str]): List of properties to check for UQ

    '''
    class uq_metrics():
        pass 
    def __init__(self,property_uqs=[]) -> None:
        self.property_uqs = property_uqs
        self.min_threshold = {}
        self.fix_threshold = {}
        for property in self.property_uqs:
            self.fix_threshold[property] = False

    def calculate_uq(self, molecule):
        '''
        Calculate UQ of a molecule

        Arguments:
            molecule (:class:`ml.data.molecule`): A molecule object whose UQ needs to be calculated
        '''
        # Initialize
        molecule.uq = self.uq_metrics() 
        for each in self.property_uqs:
            molecule.uq.__dict__[each] = None 

        # Get UQs of properties 
        for each in self.property_uqs:
            try:
                molecule.uq.__dict__[each] = np.linalg.norm(molecule.__dict__['aux_estimated_'+each] - molecule.__dict__['estimated_'+each])
            except:
                molecule.uq.__dict__[each] = np.linalg.norm(molecule.get_xyz_vectorial_properties('aux_estimated_'+each) - molecule.get_xyz_vectorial_properties('estimated_'+each))

    def calculate_uq_thresholds(self,molecular_database):
        '''
        Calculate UQ thresholds 

        Arguments:
            molecular_database (:class:`ml.data.molecular_database`): Molecular database used to calculate UQ thresholds
        '''
        metric = 'M+3MAD'
        if not 'uq_thresholds' in self.__dict__:
            self.uq_thresholds = self.uq_metrics() 
        # Calculate UQs for each molecule in the molecular database
        for mol in molecular_database:
            self.calculate_uq(mol)
        # Calculate thresholds for property UQs 
        if os.path.exists('threshold_dict.json'):
            print("Loading previous threshold")
            jsonfile = open('threshold_dict.json','r') 
            threshold_dict = json.load(jsonfile)
        for property_uq in self.property_uqs:
            if not self.fix_threshold[property_uq]:
                if not os.path.exists('threshold_dict.json'):
                    absdevs = [mol.uq.__dict__[property_uq] for mol in molecular_database]
                    absdevs = np.array(absdevs).astype('float')
                    self.uq_thresholds.__dict__[property_uq] = self.threshold_metric(absdevs,metric)
                    print(f"New threshold for {property_uq}: {self.uq_thresholds.__dict__[property_uq]}")
                else:
                    self.uq_thresholds.__dict__[property_uq] = threshold_dict[property_uq]
                    print(f"Current threshold for {property_uq}: {self.uq_thresholds.__dict__[property_uq]}")
            else:
                print(f"Current threshold for {property_uq}: {self.uq_thresholds.__dict__[property_uq]}")
        if not os.path.exists('threshold_dict.json'):
            threshold_dict = {}
            for property_uq in self.property_uqs:
                threshold_dict[property_uq] = self.uq_thresholds.__dict__[property_uq]
            jsonfile=open('threshold_dict.json','w') 
            json.dump(threshold_dict,jsonfile,indent=4)
            jsonfile.close()


    def update_threshold_settings(self,Ntrain):
        '''
        Update threshold settings. 

        Arguments:
            Ntrain (int): Number of training points
        '''
        # if Ntrain > 100:
        # Update threshold only once
        for each in self.fix_threshold.keys():
            self.fix_threshold[each] = True


    def threshold_metric(self,absdevs,metric):
        '''
        Function that calculate thresholds

        Arguments:
            absdevs (List[float]): List of absolute deviations 
            metric (str): Threshold metric
        '''
        if metric.casefold() == 'max'.casefold():
            return np.max(absdevs)
        elif metric.casefold() =='M+3MAD'.casefold():
            if len(absdevs) >= 2:
                return np.median(absdevs) + 3*ml.stats.calc_median_absolute_deviation(absdevs) 
            else:
                return 0.0
# This class is a collection of all ML models, used for MD        
class collective_ml_models():
    '''
    Collection of main ML models and auxiliary ML models. It can be used for any kinds of simulations, e.g., MD. This is a template. User can use it as a parent class.

    Arguments:
        property_to_learn (List[str]): List of properties to learn in active learning.
        mlmodels (List[:class:`ml.models.model`]): List of main ML models.
        aux_mlmodels (list[:class:`ml.models.model`]): List of auxiliary ML models.
    '''
    def __init__(self,property_to_learn,mlmodels,aux_mlmodels):
        self.property_to_learn = property_to_learn 
        self.mlmodels = mlmodels 
        self.aux_mlmodels = aux_mlmodels

    def predict(self,molecule=None,molecular_database=None,calculate_energy=True,calculate_energy_gradients=True,calculate_hessian=False,**kwargs):
        '''
        Make predictions for molecular geometries with the model 

        Arguments:
            molecule (:class:`ml.data.molecule`, optional): A molecule object whose property needs to be predicted by the model.
            molecular_database (:class:`ml.data.molecular_database`, optional): A database contains the molecules whose properties need to be predicted by the model.
            calculate_energy (bool, optional): Use the model to calculate energy.
            calculate_energy_gradients (bool, optional): Use the model to calculate energy gradients.
        '''
        for property in self.property_to_learn:
            if property[-10:] == '_gradients':
                continue
            if property+'_gradients' in self.property_to_learn:
                xyz_derivative_property = property+'_gradients'
            else:
                xyz_derivative_property = None

            if calculate_hessian:
                if property == 'energy':
                    model_predict_kwargs = {'hessian_to_predict':'hessian'}
                else:
                    model_predict_kwargs = {'hessian_to_predict':property+'_hessian'}
            else:
                model_predict_kwargs = {}
            
            if not molecule is None:
                self.mlmodels[property].predict(molecule=molecule,property_to_predict=property,xyz_derivative_property_to_predict=xyz_derivative_property,**model_predict_kwargs)
                self.aux_mlmodels[property].predict(molecule=molecule,property_to_predict='aux_'+property,xyz_derivative_property_to_predict='aux_'+xyz_derivative_property)
            elif not molecular_database is None:
                self.mlmodels[property].predict(molecular_database=molecular_database,property_to_predict=property,xyz_derivative_property_to_predict=xyz_derivative_property,**model_predict_kwargs)
                self.aux_mlmodels[property].predict(molecular_database=molecular_database,property_to_predict='aux_'+property,xyz_derivative_property_to_predict='aux_'+xyz_derivative_property)
            # User-defined part: make sure that energy and energy_gradients are saved in molecule
                
class ml_model_trainer():
    def __init__(self,ml_model_type=None,hyperparameters={}):
        self.ml_model_type = ml_model_type 
        self.hyperparameters = hyperparameters
    
    def hyperparameters_setter(self):
        # if self.ml_model_type.casefold() == 'kreg':
        #     self.hyperparameters = {
        #         'sigma'
        #     }
        pass 

    # # Use holdout validation
    # def train(self,molecular_database,property_to_learn,xyz_derivative_property_to_learn):
    #     pass 

    def main_model_trainer(self,filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,**kwargs):
        if self.ml_model_type.casefold() == 'kreg':
            return self.main_model_trainer_kreg(filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,**kwargs)
        elif self.ml_model_type.casefold() == 'ani':
            return self.main_model_trainer_ani(filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,**kwargs)
        elif self.ml_model_type.casefold() == 'mace':
            return self.main_model_trainer_mace(filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,**kwargs)
         

    def aux_model_trainer(self,filename,subtraining_molDB,validation_molDB,property_to_learn,device,**kwargs):
        if self.ml_model_type.casefold() == 'kreg':
            return self.aux_model_trainer_kreg(filename,subtraining_molDB,validation_molDB,property_to_learn,device,**kwargs)
        elif self.ml_model_type.casefold() == 'ani':
            return self.aux_model_trainer_ani(filename,subtraining_molDB,validation_molDB,property_to_learn,device,**kwargs)
        elif self.ml_model_type.casefold() == 'mace':
            return self.aux_model_trainer_mace(filename,subtraining_molDB,validation_molDB,property_to_learn,device,**kwargs)
        pass

    def main_model_trainer_kreg(self,filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device):
        print("Training the main KREG model")
        if os.path.exists(filename):
            print(f"Model file {filename} found, skip training")
            main_model = ml.models.kreg(model_file=filename,ml_program='KREG_API')
        else:
            main_model = ml.models.kreg(model_file=filename,ml_program='KREG_API')
            main_model.hyperparameters['sigma'].minval = 2**-5
            # if not 'lmbd' in self.hyperparameters and not 'sigma' in self.hyperparameters:
            main_model.optimize_hyperparameters(subtraining_molecular_database=subtraining_molDB,
                                                validation_molecular_database=validation_molDB,
                                                optimization_algorithm='grid',
                                                hyperparameters=['lambda','sigma'],
                                                training_kwargs={'property_to_learn': property_to_learn, 'xyz_derivative_property_to_learn': xyz_derivative_property_to_learn, 'prior': 'mean'},
                                                prediction_kwargs={'property_to_predict': 'estimated_'+property_to_learn, 'xyz_derivative_property_to_predict': 'estimated_'+xyz_derivative_property_to_learn},
                                                validation_loss_function=None)
            lmbd_ = main_model.hyperparameters['lambda'].value ; sigma_ = main_model.hyperparameters['sigma'].value
            self.hyperparameters['lambda'] = lmbd_; self.hyperparameters['sigma'] = sigma_
            print(f"Optimized hyperparameters for {property_to_learn} main model: lambda={lmbd_}, sigma={sigma_}")
        return main_model 
    
    def aux_model_trainer_kreg(self,filename,subtraining_molDB,validation_molDB,property_to_learn,device):
        print("Training the main auxiliary model")
        if os.path.exists(filename):
            print(f"Model file {filename} found, skip training")
            aux_model = ml.models.kreg(model_file=filename,ml_program='KREG_API')
        else:
            aux_model = ml.models.kreg(model_file=filename,ml_program='KREG_API')
            aux_model.hyperparameters['sigma'].minval = 2**-5
            aux_model.optimize_hyperparameters(subtraining_molecular_database=subtraining_molDB,
                                                validation_molecular_database=validation_molDB,
                                            optimization_algorithm='grid',
                                            hyperparameters=['lambda', 'sigma'],
                                            training_kwargs={'property_to_learn': property_to_learn, 'prior': 'mean'},
                                            prediction_kwargs={'property_to_predict': 'estimated'+property_to_learn})
            lmbd_ = aux_model.hyperparameters['lambda'].value ; sigma_ = aux_model.hyperparameters['sigma'].value
            self.hyperparameters['aux_lambda'] = lmbd_ ; self.hyperparameters['sigma'] = sigma_
            print(f"Optimized hyperparameters for {property_to_learn} aux model: lambda={lmbd_}, sigma={sigma_}")
        return aux_model

    def main_model_trainer_ani(self,filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,**kwargs):
        print("Training the main ANI model")
        if os.path.exists(filename):
            print(f"Model file {filename} found, skip training")
            main_model = ml.models.ani(model_file=filename,device=device,verbose=False)
        else:
            main_model = ml.models.ani(model_file=filename,device=device,verbose=False)
            main_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn,xyz_derivative_property_to_learn=xyz_derivative_property_to_learn)
        return main_model 
    
    def aux_model_trainer_ani(self,filename,subtraining_molDB,validation_molDB,property_to_learn,device,**kwargs):
        print("Training the auxiliary ANI model")
        if os.path.exists(filename):
            print(f"Model file {filename} found, skip training")
            aux_model = ml.models.ani(model_file=filename,device=device,verbose=False)
        else:
            aux_model = ml.models.ani(model_file=filename,device=device,verbose=False)
            aux_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn)
        return aux_model 
    
    def main_model_trainer_mace(self,filename,subtraining_molDB,validation_molDB,property_to_learn,xyz_derivative_property_to_learn,device,**kwargs):
        print("Training the main MACE model")
        os.system("rm -rf MACE_*")
        if os.path.exists(filename):
            print(f"Model file {filename} found, skip training")
            main_model = ml.models.mace(model_file=filename,device=device,verbose=False)
        else:
            main_model = ml.models.mace(model_file=filename,device=device,verbose=False)
            main_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn,xyz_derivative_property_to_learn=xyz_derivative_property_to_learn)
        return main_model 
    
    def aux_model_trainer_mace(self,filename,subtraining_molDB,validation_molDB,property_to_learn,device,**kwargs):
        print("Training the auxiliary MACE model")
        os.system("rm -rf MACE_*")
        if os.path.exists(filename):
            print(f"Model file {filename} found, skip training")
            aux_model = ml.models.mace(model_file=filename,device=device,verbose=False)
        else:
            aux_model = ml.models.mace(model_file=filename,device=device,verbose=False)
            aux_model.train(molecular_database=subtraining_molDB,validation_molecular_database=validation_molDB,property_to_learn=property_to_learn)
        return aux_model
    

def stop_function(mol,properties,thresholds,bonds=[]):
    stop = False 

    # # Check bond lengths
    # dist_matrix = mol.get_internuclear_distance_matrix()
    # for bond in bonds:
    #     ii = bond[0] ; jj = bond[1]
    #     ian = mol.atoms[ii].atomic_number ; jan = mol.atoms[jj].atomic_number
    #     dist = dist_matrix[ii][jj]
    #     if (ian == 1 and (jan > 1 and jan < 10)) or (jan == 1 and (ian > 1 and ian < 10)):
    #         if dist > 1.5: stop = True
    #     if (ian > 1 and ian < 10) and (jan > 1 and jan < 10):
    #         if dist > 1.8: stop = True
    # # prevent too short bond lengths too
    # for ii in range(len(mol.atoms)):
    #     for jj in range(ii+1, len(mol.atoms)):
    #         ian = mol.atoms[ii].atomic_number ; jan = mol.atoms[jj].atomic_number
    #         dist = dist_matrix[ii][jj]
    #         if ian == 1 and jan == 1 and dist < 0.6: stop = True
    #         elif ((ian == 1 and (jan > 1 and jan < 10)) or (jan == 1 and (ian > 1 and ian < 10))) and dist < 0.85: stop = True
    #         if (ian > 1 and ian < 10) and (jan > 1 and jan < 10) and dist < 1.1: stop = True        
    # if stop: 
    #     return stop
    
    # Check UQs
    # User-defined parts: which properties do you want to check?
    for property in properties:
        try:
            abs_dev = np.linalg.norm(mol.__dict__[property] - mol.__dict__['aux_'+property])
        except:
            abs_dev = np.linalg.norm(mol.get_xyz_vectorial_properties(property) - mol.get_xyz_vectorial_properties('aux_'+property))
        if abs_dev > thresholds.__dict__[property]:
            stop = True 
            mol.need_to_be_labeled = True 
            break 
    return stop 

def stopper(errMsg):
    '''
    function printing error message
    '''
    print(f"<!> {errMsg} <!>")
    exit()

    
if __name__ == '__main__':
    active_learning()


