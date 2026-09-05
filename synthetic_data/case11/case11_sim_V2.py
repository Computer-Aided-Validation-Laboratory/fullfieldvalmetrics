import time
from pathlib import Path
from typing import Any
import dataclasses
import shutil
import numpy as np
from scipy.spatial import KDTree
import pyvista as pv

#pyvale imports
import pyvale.dataset as dataset
import pyvale.sensorsim as sens
from pyvale.mooseherder import (MooseConfig,
                                MooseRunner,
                                ExodusReader)
from pyvale.mooseherder import (MooseHerd,
                                MooseRunner,
                                MooseConfig,
                                InputModifier,
                                DirectoryManager,
                                sweep_param_grid)
#%%
# Set file locations
case_name = "case11_sim"
mesh_name = "case11.msh"

# mesh_path = "./meshes/"
# output_path = f"./output/{case_name}/{case_name}_moose.vtu"
input_file_path = f"{case_name}.i"
moose_input = Path(input_file_path)
moose_modifier = InputModifier(moose_input,'#','')

# mesh_file_path = mesh_path + mesh_name

config = {'main_path': Path.home()/ 'moose',
          'app_path': Path.home() / 'proteus',
          'app_name': 'proteus-opt'}
moose_config = MooseConfig(config)

moose_runner = MooseRunner(moose_config)

moose_runner.set_run_opts(n_tasks = 1,
                          n_threads = 16,
                          redirect_out = False)

moose_runner.set_input_file(moose_input)

print(moose_runner.get_arg_list())
print()

#%%
# Number of epistemic simulation cases
epis_n = 50

# Number of aleatory experiments for each epistemic case
alea_n = 100

# Total number of experiments
num_exp = epis_n * alea_n
num_para_sims: int = num_exp
dir_manager = DirectoryManager(n_dirs=num_para_sims)
herd = MooseHerd([moose_runner],[moose_modifier],dir_manager)
# herd.set_num_para_sims(n_para=num_para_sims)
herd.set_num_para_sims(n_para=epis_n)

#%%
# Create output folder
output_path = Path.cwd() / "pyvale-output/sims"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

dir_manager.set_base_dir(output_path)
dir_manager.reset_dirs()

#%%

parent_folder = output_path
mesh_file = Path(mesh_name)

# Copy into each subfolder
for subfolder in parent_folder.iterdir():
    if subfolder.is_dir():
        destination = subfolder / mesh_file.name
        shutil.copy2(mesh_file, destination)
        print(f"Copied to: {destination}")

#%%

wThermCond = 127.0
surfHeatFlux = 10.0e6
wThermCond_vals = np.linspace(wThermCond * 0.95, wThermCond * 1.05, epis_n)
surfHeatFlux_vals = np.random.normal(loc=surfHeatFlux, scale=surfHeatFlux*0.03, size=alea_n)
# surfHeatFlux_vals = np.array([10.0e6, 10.0e6])

moose_params = {"wThermCond": wThermCond_vals,
                "surfHeatFlux": surfHeatFlux_vals}
params = [moose_params,]
sweep_params = sweep_param_grid(params)

print("\nParameter sweep variables by simulation:")
for ii,pp in enumerate(sweep_params):
    print(f"Sim: {ii}, Params [moose,]: {pp}")

print()
print(f"Total simulations = {len(sweep_params)}")
print()

#%%
num_para_runs: int = 1

if __name__ == '__main__':
    sweep_times = np.zeros((num_para_runs,),dtype=np.float64)
    for rr in range(num_para_runs):
        herd.run_para(sweep_params)
        sweep_times[rr] = herd.get_sweep_time()


print(80*"-")
for ii,ss in enumerate(sweep_times):
    print(f"Sweep {ii} took: {ss:.3f}seconds")

print(80*"-")
print(f"Average sweep time: {np.mean(sweep_times):.3f} seconds")
print(80*"-")
