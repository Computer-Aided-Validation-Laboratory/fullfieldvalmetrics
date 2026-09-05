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
#%%
# Set file locations
case_name = "case11_sim"
mesh_name = "case11.msh"

# mesh_path = "./meshes/"
# output_path = f"./output/{case_name}/{case_name}_moose.vtu"
input_file_path = f"{case_name}.i"

# mesh_file_path = mesh_path + mesh_name

config = {'main_path': Path.home()/ 'moose',
          'app_path': Path.home() / 'proteus',
          'app_name': 'proteus-opt'}
moose_config = MooseConfig(config)

moose_runner = MooseRunner(moose_config)

moose_runner.set_run_opts(n_tasks = 1,
                          n_threads = 1,
                          redirect_out = False)

moose_input = Path(input_file_path)

moose_runner.set_input_file(moose_input)

print(moose_runner.get_arg_list())
print()

#%%
# Run the simulation

start_time = time.perf_counter()
moose_runner.run()
run_time = time.perf_counter() - start_time

print()
print("-"*80)
print(f'MOOSE run time = {run_time:.3f} seconds')
print("-"*80)
print()

#%%
# Save the results

# input_path = Path(input_path)

# output_exodus = input_path / (moose_input.stem + "_out.e")
# exodus_reader = ExodusReader(output_exodus)

# print("\nReading exodus file with ExodusReader:")
# print(output_exodus.resolve())
# print()

# read_config = exodus_reader.get_read_config()
# sens.SimTools.print_dataclass_fields(read_config)

# read_config.time = False
# read_config.coords = True
# read_config.connect = False
# sim_data = exodus_reader.read_sim_data(read_config)

# sens.SimTools.print_sim_data(sim_data)

# sol_unordered = sim_data.node_vars['temperature'][:, 1]
# points_unordered = sim_data.coords

# res = pv.read(mesh_file_path)
# points = res.points
# x_coord = points[:, 0]
# y_coord = points[:, 1]

# tree2 = KDTree(points_unordered)

# _, indices = tree2.query(points)

# sol = sol_unordered[indices]

# sol_exact = 16*x_coord*(1-x_coord)*y_coord*(1-y_coord)
# abs_error = np.abs(sol - sol_exact)
# abs_error_max = np.max(abs_error)
# abs_error_mean = np.mean(abs_error)
# print(f"Max. absolute error between MOOSE analytical solution : {abs_error_max:.3e}.")
# print(f"Avg. absolute error between MOOSE and analytical solution : {abs_error_mean:.3e}.")

# res["sol"] = sol

# res.save(output_path)