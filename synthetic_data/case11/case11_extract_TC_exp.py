
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# pyvale imports
import pyvale.sensorsim as sens
import pyvale.dataio as io
import pyvale.mooseherder as mh
import pyvale.dataset as dataset
from pyvale.sensorsim.experimentsimulator import (ExperimentSimulator,
                                                  ExpSimSaveKeys)

# num_exp=434
num_exp=5000

# -------------------------------
# 1. Load physics simulation data
# -------------------------------
data_path: Path = Path("case11_exp_out.e")
sim_data: io.SimData = mh.ExodusLoader(data_path).load_all_sim_data()
sim_data: io.SimData = sens.scale_length_units(scale=1000.0,
                                               sim_data=sim_data,
                                               disp_keys=None)

# print(sim_data)
# loader = mh.ExodusLoader(data_path)
# loader.print_vars()

# -------------------------------
# 2. Build a virtual sensor array
# --------------------------------
# 7 TC locations to match the experiment
sens_pos: np.ndarray = sens.gen_pos_grid_inside(num_sensors=(1,7,1),
                                                    x_lims=(12.5,12.5),
                                                    y_lims=(0.0,33.0),
                                                    z_lims=(0.0,12.0))
# sens_data = sens.SensorData(positions=sens_pos)
sens_data = sens.SensorData(positions=sens_pos,
                            sample_times=np.ndarray([1]))

# Sample only at time step 1 as it's a steady state simulation
# sens_data = sens.SensorData(positions=sens_pos,
#                             sample_times=np.ndarray([0, 1]))

# sample_times: np.ndarray = np.linspace(0.0,np.max(sim_data.time),1)

# sens_data = sens.SensorData(positions=sens_pos,
#                             sample_times=sample_times)

sens_array: sens.SensorsPoint = sens.SensorFactory.scalar_point(
    sim_data,
    sens_data,
    comp_key="temperature",
    spatial_dims=sens.EDim.THREED,
    descriptor=sens.DescriptorFactory.temperature(),
)


# -------------------------------
# 2.1. Add simulated measurement errors
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# err_chain: list[sens.IErrSimulator] = [
#     sens.ErrSysGen(sens.GenUniform(low=-1.0,high=1.0)),
#     sens.ErrSysGenPercent(sens.GenUniform(low=-1.0,high=1.0)),
#     sens.ErrRandGen(sens.GenNormal(std=1.0)),
#     sens.ErrRandGenPercent(sens.GenNormal(std=2.0)),
#     sens.ErrSysDigitisation(bits_per_unit=2**16/100),
#     sens.ErrSysSaturation(meas_min=0.0,meas_max=450.0),
# ]

# err_chain: list[sens.IErrSimulator] = [
#     sens.ErrSysGen(sens.GenUniform(low=-10.0,high=10.0)),
#     sens.ErrRandGen(sens.GenNormal(std=5.0)),
# ]

err_chain: list[sens.IErrSimulator] = [
    sens.ErrSysGen(sens.GenUniform(low=-5.0,high=5.0)),
    sens.ErrRandGen(sens.GenNormal(std=5.0)),
]
sens_array.set_error_chain(err_chain)

# -------------------------------
# 3. Create & run simulated experiment
# ------------------------------------

sims: dict[str,io.SimData] = {"sim_nominal":sim_data,}
sensors: dict[str,sens.ISensorArray] = {"temp_sens":sens_array,}

exp_sim_opts = sens.ExpSimOpts(workers=4,para=sens.EExpSimPara.ALL)
exp_sim = sens.ExperimentSimulator(sims,sensors,exp_sim_opts)

exp_data: dict[tuple[str,...],np.ndarray] = (
    exp_sim.run_experiments(num_exp_per_sim=num_exp)
)

exp_stats: dict[tuple[str,...],sens.ExpSimStats] = (
    sens.calc_exp_sim_stats(exp_data)
)

print(exp_stats)

# -------------------------------
# 4. Analyse & visualise the results
# ----------------------------------

output_path = Path.cwd() / "pyvale-output"
if not output_path.is_dir():
    output_path.mkdir(parents=True, exist_ok=True)

pv_plot = sens.plot_point_sensors_on_sim(sens_array,"temperature")
pv_plot.camera_position = [(59.354, 43.428, 69.946),
                            (-2.858, 13.189, 4.523),
                            (-0.215, 0.948, -0.233)]

# Set to False to show an interactive plot instead of saving the figure
pv_plot.off_screen = False
if pv_plot.off_screen:
    pv_plot.screenshot(output_path/"locs_exp.png")
else:
    pv_plot.show()


trace_opts = sens.TraceOptsExperiment(plot_all_exp_points=True)
(fig,ax) = sens.plot_exp_traces(
    exp_data,
    comp_ind=0,
    sens_key="temp_sens",
    sim_key="sim_nominal",
    descriptor=sens.DescriptorFactory.temperature(),
    trace_opts=trace_opts,
)

fig.savefig(output_path/"traces_exp.png",dpi=300,bbox_inches="tight")

print(exp_data.keys())


exp_save_keys = ExpSimSaveKeys()
sens_key="temp_sens"
sim_key="sim_nominal"
meas_key = (sim_key,sens_key,exp_save_keys.meas)
sys_key = (sim_key,sens_key,exp_save_keys.sys)
rand_key = (sim_key,sens_key,exp_save_keys.rand)
time_key = (sim_key,sens_key,exp_save_keys.sens_times)
pos_key = (sim_key,sens_key,exp_save_keys.pert_sens_pos)

exp_arr = exp_data[meas_key]
samp_time = exp_data[time_key]

truth = exp_data[meas_key] - exp_data[sys_key] - exp_data[rand_key]
truth = truth[0,:,:,:]

print(type(exp_arr))
exp_arr = np.squeeze(exp_arr)
print(exp_arr.shape)

headers = [f"TC{i}" for i in range(1, 8)]
# headers = ["TC2", "TC3", "TC5", "TC6", "TC8", "TC9", "TC10"]
np.savetxt(
    "pyvale-output/SamplingResultsOnlyPointSensors_exp.csv",
    exp_arr,
    delimiter=",",
    header=",".join(headers),
    comments=""
)

pos = exp_data[pos_key][0, :, :]
headers = ["x", "y", "z"]
np.savetxt(
    "pyvale-output/PointSensorCoords.csv",
    pos,
    delimiter=",",
    header=",".join(headers),
    comments=""
)

print(pos.shape)
print(pos)



# Uncomment to show interactive figure
# plt.show()
