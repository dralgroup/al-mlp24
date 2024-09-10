# Active learning for machine learning potentials using physics-informed approach with MLatom

- You need to install [MLatom 3.7+](https://github.com/dralgroup/mlatom)
- Tutorial: https://xacs.xmu.edu.cn/docs/mlatom/tutorial_al.html

The theoretical background and examples of the use of this active learning procedure is described at:

- Yi-Fan Hou, Lina Zhang, Quanhao Zhang, Fuchun Ge, [Pavlo O. Dral](http://dr-dral.com). [Physics-informed active learning for accelerating quantum chemical simulations](https://doi.org/10.1021/acs.jctc.4c00821). *J. Chem. Theory Comput.* **2024**, *in press*. DOI: 10.1021/acs.jctc.4c00821.
Preprint on arXiv: https://arxiv.org/abs/2404.11811.

Please cite this work when using this implementation & methodology.

All the funtions of active learning are in `scripts/al_general.py`. In the other folders, you can find the active learning scripts named as `xx_al.py`, final main models `xx_energy_iterationxx.npz`, final auxiliary models `xx_energy_aux_iterationxx.npz`, molecules optimized with the reference method `xx_eqmol.json`, initial dataset of active learning `init_cond_db.json` and the final dataset `labeled_db.json`.

All the scripts are not going to be maintained as the users should use the released MLatom.


