
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# pyvale imports
import pyvale.sensorsim as sens
import pyvale.dataio as io
import pyvale.mooseherder as mh
from pyvale.sensorsim.experimentsimulator import ExpSimSaveKeys
from pyvale.sensorsim.errorsimulator import EErrDep
import json


# Number of epistemic simulation cases
epis_n = 50

# Number of aleatory experiments for each epistemic case
alea_n = 100

# Total number of experiments
num_exp = epis_n * alea_n

print("=" * 80)
print("Simulation sampling")
print("=" * 80)
print(f"Epistemic samples : {epis_n}")
print(f"Aleatory samples  : {alea_n}")
print(f"Total samples     : {num_exp}")
print("=" * 80)


# -------------------------------
# 1. Load data paths
# -------------------------------
# Exodus files, one for each epistemic sample

sim_json = Path.cwd() / "pyvale-output/sims/sim-workdir-1/output-key-1.json"

with open(sim_json, "r") as f:
    sim_files = [Path(p[0]) for p in json.load(f)]

print("Simulation files:")
for ii, path in enumerate(sim_files):
    print(f"  EP {ii:02d}: {path}")


# -------------------------------
# 2. Load epistemic data
# -------------------------------

sims: dict[str, io.SimData] = {}

for ee, data_path in enumerate(sim_files):

    print("-" * 80)
    print(f"Loading epistemic simulation {ee + 1}/{epis_n}")
    print(data_path)

    sim_data: io.SimData = (
        mh.ExodusLoader(data_path).load_all_sim_data()
    )

    sim_data: io.SimData = sens.scale_length_units(
        scale=1000.0,
        sim_data=sim_data,
        disp_keys=None,
    )

    sim_key = f"sim_ep_{ee:03d}"

    sims[sim_key] = sim_data


# -------------------------------
# 3. Build a virtual sensor array
# -------------------------------

# 7 TC locations to match the experiment
sens_pos: np.ndarray = sens.gen_pos_grid_inside(
    num_sensors=(1, 7, 1),
    x_lims=(12.5, 12.5),
    y_lims=(0.0, 33.0),
    z_lims=(0.0, 12.0),
)

sens_data = sens.SensorData(
    positions=sens_pos,
    sample_times=np.ndarray([1]),
)


# -------------------------------
# 4. Create sensor array
# -------------------------------

# The sensor array is based on one of the simulations.
# The sensor definition itself is independent of which epistemic
# simulation case is being used.

first_sim = sims["sim_ep_000"]

sens_array: sens.SensorsPoint = sens.SensorFactory.scalar_point(
    first_sim,
    sens_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.THREED,
    descriptor=sens.DescriptorFactory.temperature(),
)

# -------------------------------
# 5. Add simulated measurement errors
# -------------------------------

err_chain: list[sens.IErrSimulator] = [
    sens.ErrSysGen(sens.GenUniform(low=-1.0,high=1.0)),
    sens.ErrRandGen(sens.GenNormal(std=0.0), err_dep=EErrDep.DEPENDENT),
]
sens_array.set_error_chain(err_chain)


# -------------------------------
# 6. Create sensor dictionary
# -------------------------------

sensors: dict[str, sens.ISensorArray] = {"temp_sens": sens_array,}

# -------------------------------
# 7. Run aleatory experiments for each epistemic simulation
# -------------------------------

exp_sim_opts = sens.ExpSimOpts(
    workers=16,
    para=sens.EExpSimPara.ALL,
)

exp_sim = sens.ExperimentSimulator(
    sims,
    sensors,
    exp_sim_opts,
)

exp_data: dict[tuple[str, ...], np.ndarray] = (
    exp_sim.run_experiments(
        num_exp_per_sim=alea_n
    )
)


# -------------------------------
# 8. Calculate statistics
# -------------------------------

exp_stats: dict[tuple[str, ...], sens.ExpSimStats] = (
    sens.calc_exp_sim_stats(exp_data)
)

print(exp_stats)


# -------------------------------
# 9. Output directory
# -------------------------------

output_path = Path.cwd() / "pyvale-output"

if not output_path.is_dir():
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

# -------------------------------
# 10. Extract measurement data
# -------------------------------

exp_save_keys = ExpSimSaveKeys()

sens_key = "temp_sens"

# There are now 50 simulation keys
for ee in range(epis_n):

    sim_key = f"sim_ep_{ee:03d}"

    meas_key = (
        sim_key,
        sens_key,
        exp_save_keys.meas,
    )

    if meas_key not in exp_data:
        print(f"WARNING: {meas_key} not found")
        continue

    exp_arr = exp_data[meas_key]

    print("-" * 80)
    print(f"Epistemic sample {ee}")
    print(f"Raw shape: {exp_arr.shape}")


# -------------------------------
# 11. Save data in epistemic × aleatory format
# -------------------------------

headers = [
    f"TC{i}"
    for i in range(1, 8)
]


# Create one output array:
# epistemic × aleatory × sensor

sim_results = np.zeros(
    (epis_n, alea_n, len(headers)),
    dtype=float,
)


for ee in range(epis_n):

    sim_key = f"sim_ep_{ee:03d}"

    meas_key = (
        sim_key,
        sens_key,
        exp_save_keys.meas,
    )

    exp_arr = exp_data[meas_key]

    exp_arr = np.squeeze(exp_arr)

    print(
        f"EP {ee:02d}: "
        f"raw/squeezed shape = {exp_arr.shape}"
    )

    if exp_arr.shape != (alea_n, len(headers)):
        raise ValueError(
            f"Unexpected shape for {sim_key}: "
            f"{exp_arr.shape}. "
            f"Expected {(alea_n, len(headers))}."
        )

    sim_results[ee, :, :] = exp_arr


print("=" * 80)
print("FINAL SIMULATION ARRAY")
print(f"Shape: {sim_results.shape}")
print("=" * 80)


# -------------------------------
# 12. Save flattened CSV
# -------------------------------

sim_results_flat = sim_results.reshape(
    epis_n * alea_n,
    len(headers),
)

np.savetxt(
    output_path / "SamplingResultsOnlyPointSensors_sim.csv",
    sim_results_flat,
    delimiter=",",
    header=",".join(headers),
    comments="",
)

print(
    f"Saved CSV with shape: "
    f"{sim_results_flat.shape}"
)


