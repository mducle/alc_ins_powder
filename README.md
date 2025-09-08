# ALC project: Physics-informed machine learning for accelerating inelastic neutron powder data analysis

This private repository is to store scripts and data related to the Ada-Lovelace-Centre project:
"Physics-informed machine learning for accelerating inelastic neutron powder data analysis"

ML train: srcnn, unet, fno three types of models. Limit means how many samples we will choose in ML method. 
SrCNN model example: 

```sh
python gen_robustness.py --model srcnn --limit 60
```

# Environment

This will setup a virtualenv which can be used to run the above script.

```sh
mamba create -n janus311 -c mantid mantidworkbench=6.13 ase codecarbon phonopy pymatgen rich typer pytorch e3nn=0.4.4 opt_einsum torch-ema torchmetrics matscipy python-lmdb orjson scikit-learn mp-api
mamba activate janus311
python -m pip install --no-deps janus-core mace-torch
```
