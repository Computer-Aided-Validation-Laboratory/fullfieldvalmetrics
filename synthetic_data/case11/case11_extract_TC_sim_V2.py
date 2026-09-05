
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


# -------------------------------
# 1. Load simulation paths and corresponding parameters
# -------------------------------

params_json = (
   Path.cwd() / "pyvale-output/sims/sim-workdir-1/sweep-vars-1.json"
)
files_json = (
    Path.cwd() / "pyvale-output/sims/sim-workdir-1/output-key-1.json"
)

with open(params_json, "r") as f:
    params = json.load(f)

with open(files_json, "r") as f:
    file_paths = json.load(f)

params = [p[0] for p in params]
file_paths = [p[0] for p in file_paths]

wThermCond_vals = sorted(
    set(p["wThermCond"] for p in params)
)

surfHeatFlux_vals = sorted(
    set(p["surfHeatFlux"] for p in params)
)

print("=" * 80)
print("PARAMETER GRID")
print("=" * 80)

print(f"wThermCond values : {len(wThermCond_vals)}")
print(f"surfHeatFlux values: {len(surfHeatFlux_vals)}")

print("wThermCond:")
print(wThermCond_vals)

print("surfHeatFlux:")
print(surfHeatFlux_vals)

sim_lookup = {(p["wThermCond"], p["surfHeatFlux"],): Path(path)
    for p, path in zip(params, file_paths)
}

expected_n = (len(wThermCond_vals) * len(surfHeatFlux_vals)
)

if len(sim_lookup) != expected_n:
    raise ValueError(
        f"Incomplete parameter grid. "
        f"Found {len(sim_lookup)} simulations, "
        f"expected {expected_n}."
    )


# -------------------------------
# 2. Create TC locations
# -------------------------------

sens_pos: np.ndarray = sens.gen_pos_grid_inside(
    num_sensors=(1, 7, 1),
    x_lims=(12.5, 12.5),
    y_lims=(0.0, 33.0),
    z_lims=(0.0, 12.0),
)

sens_data = sens.SensorData(
    positions=sens_pos,
    # None means use the simulation time steps
    sample_times=None,
)

# -------------------------------
# 3. Final simulation array
# -------------------------------
#
#   axis 0 = epistemic (wThermCond)
#   axis 1 = aleatory (surfHeatFlux)
#   axis 2 = TC
#   (n_epistemic, n_aleatory, n_TC)

sim_results = np.zeros(
    (
        len(wThermCond_vals),
        len(surfHeatFlux_vals),
        7,
    ),
    dtype=float,
)


# -------------------------------
# 4. Load sims and extract ground truth
# -------------------------------

for ii, w_cond in enumerate(wThermCond_vals):

    for jj, heat_flux in enumerate(surfHeatFlux_vals):

        sim_key = (w_cond, heat_flux)

        if sim_key not in sim_lookup:
            raise KeyError(
                f"No simulation found for "
                f"wThermCond={w_cond}, "
                f"surfHeatFlux={heat_flux}"
            )

        data_path = sim_lookup[sim_key]

        print("-" * 80)
        print(
            f"wThermCond = {w_cond:.8f} "
            f"({ii + 1}/{len(wThermCond_vals)})"
        )
        print(
            f"surfHeatFlux = {heat_flux:.8f} "
            f"({jj + 1}/{len(surfHeatFlux_vals)})"
        )
        print(f"Loading: {data_path}")


        sim_data: io.SimData = (
            mh.ExodusLoader(data_path)
            .load_all_sim_data()
        )

        sim_data: io.SimData = sens.scale_length_units(
            scale=1000.0,
            sim_data=sim_data,
            disp_keys=None,
        )

        sens_array: sens.SensorsPoint = (
            sens.SensorFactory.scalar_point(
                sim_data,
                sens_data,
                comp_key="temperature",
                spatial_dims=sens.EDim.THREED,
                descriptor=sens.DescriptorFactory.temperature(),
            )
        )

        truth = sens_array.get_truth()
        print(f"Truth shape: {truth.shape}")


        # (num_sensors, num_components, num_time_steps)

        truth = np.squeeze(truth)

        print(f"Squeezed shape: {truth.shape}")

        if truth.ndim == 1:
            # Already 7 values
            tc_values = truth

        elif truth.ndim == 2:
            # Take final time step
            tc_values = truth[:, -1]

        else:
            raise ValueError(
                f"Unexpected truth shape: {truth.shape}"
            )

        if tc_values.shape != (7,):
            raise ValueError(
                f"Unexpected TC shape for "
                f"wThermCond={w_cond}, "
                f"surfHeatFlux={heat_flux}: "
                f"{tc_values.shape}. "
                f"Expected (7,)."
            )

        # (epistemic, aleatory, sensor)
        sim_results[ii, jj, :] = tc_values


# -------------------------------
# 5. Save
# -------------------------------

print("=" * 80)
print("FINAL SIMULATION ARRAY")
print("=" * 80)

print(
    f"Shape = {sim_results.shape}"
)

print(
    f"Expected = "
    f"({len(wThermCond_vals)}, "
    f"{len(surfHeatFlux_vals)}, "
    f"7)"
)

output_path = Path.cwd() / "pyvale-output"

output_path.mkdir(
    parents=True,
    exist_ok=True,
)

np.save(
    output_path / "wThermCond_vals.npy",
    np.asarray(wThermCond_vals),
)

np.save(
    output_path / "surfHeatFlux_vals.npy",
    np.asarray(surfHeatFlux_vals),
)


#   wThermCond × surfHeatFlux × TC
np.save(
    output_path / "SamplingResultsOnlyPointSensors_sim.npy",
    sim_results,
)

# Headers:
#   wThermCond, surfHeatFlux, TC1, ..., TC7
headers = [
    "wThermCond",
    "surfHeatFlux",
    "TC1",
    "TC2",
    "TC3",
    "TC4",
    "TC5",
    "TC6",
    "TC7",
]

csv_results = []

for ii, w_cond in enumerate(wThermCond_vals):

    for jj, heat_flux in enumerate(surfHeatFlux_vals):

        csv_results.append(
            [
                w_cond,
                heat_flux,
                *sim_results[ii, jj, :],
            ]
        )


csv_results = np.asarray(
    csv_results,
    dtype=float,
)


np.savetxt(
    output_path / "SamplingResultsOnlyPointSensors_sim.csv",
    csv_results,
    delimiter=",",
    header=",".join(headers),
    comments="",
)

print("=" * 80)
print("Saved")
print("=" * 80)

print(
    f"3D array: "
    f"{output_path / 'SamplingResultsOnlyPointSensors_sim.npy'}"
)

print(
    f"CSV: "
    f"{output_path / 'SamplingResultsOnlyPointSensors_sim.csv'}"
)

print(
    f"CSV shape: {csv_results.shape}"
)
