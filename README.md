# Active learning for machine learning potentials using physics-informed approach with MLatom

- You need to install [MLatom 3.7+](https://github.com/dralgroup/mlatom)
- Tutorial: https://xacs.xmu.edu.cn/docs/mlatom/tutorial_al.html
- Brief Jupyter notebooks (on Google colab and XACS cloud): https://github.com/JakubMartinka/karlsruhe2024

The theoretical background and examples of the use of this active learning procedure is described at:

- Yi-Fan Hou, Lina Zhang, Quanhao Zhang, Fuchun Ge, [Pavlo O. Dral](http://dr-dral.com). [Physics-informed active learning for accelerating quantum chemical simulations](https://doi.org/10.1021/acs.jctc.4c00821). *J. Chem. Theory Comput.* **2024**, *ASAP*. DOI: 10.1021/acs.jctc.4c00821.
Preprint on arXiv: https://arxiv.org/abs/2404.11811.

Please cite this work when using this implementation & methodology.

## Data

Folder for each application contains molecules optimized with the reference method `xx_eqmol.json`, initial dataset of active learning `init_cond_db.json` and the final dataset `labeled_db.json`. The json format is using the MLatom's database format.

## Models

Final main models are saved in `xx_energy_iterationxx.npz` and final auxiliary models in `xx_energy_aux_iterationxx.npz`.

## Code

First of all, *this repository is only for the purpose of archiving the original scripts*. None of the scripts is going to be maintained as the users should use the released MLatom with more features. Models and data should still work with the new MLatom.

The development script used for the original study is saved in `scripts/al_general.py` (you still need to install MLatom for it to work!). This script is cleaned up and integrated into the official MLatom release, where many things changed.

In the other folders sorted by application, you can find the Python scripts to run AL with the above `al_general.py` script for each application and other necessary scripts to analyze the results.
