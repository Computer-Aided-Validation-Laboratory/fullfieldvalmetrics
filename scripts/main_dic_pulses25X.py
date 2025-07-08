import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import griddata
import valmetrics as vm

# NOTE
# - Can calculate difference maps for each epistemic sample so could have
# hundreds, need to limit this to find the limiting cdfs for each case.
# - Collapse full-field simulation data to limiting cdfs for each point?


def main() -> None:
    print(80*"=")
    print("MAVM Calc for DIC Data: Pulse 25X")
    print(80*"=")
    print()

    #===========================================================================
    EXP_IND: int = 2
    #===========================================================================

    #---------------------------------------------------------------------------
    # SIM: constants
    SIM_TAG = "simred"
    FE_DIR = Path.cwd()/ "STC_ProbSim_Reduced"
    conv_to_mm: float = 1000.0 # Simulation is in SI and exp is in mm

    # Reduced: 5000 = 100 aleatory x 50 epistemic
    # Full: 400 aleatory x 250 epistemic
    # exp_data = exp_data.reshape(samps_n,epis_n,alea_n)
    #samps_n: int = 5000
    SIM_EPIS_N: int = 50
    SIM_ALEA_N: int = 100


    #---------------------------------------------------------------------------
    # EXP: constants
    DIC_PULSES = ("253","254","255")
    DIC_DIRS = (
        Path.cwd() / "STC_Exp_Pulse253",
        Path.cwd() / "STC_Exp_Pulse254",
        Path.cwd() / "STC_Exp_Pulse255",
    )
    DIC_STEADY = [(297,694),
                  (302,694),
                  (293,694)]

    #---------------------------------------------------------------------------
    # Check directories exist and create output directories
    temp_path = Path.cwd() / f"temp_{SIM_TAG}"
    if not temp_path.is_dir():
        temp_path.mkdir()

    if not FE_DIR.is_dir():
        raise FileNotFoundError(f"{FE_DIR}: directory does not exist.")

    for dd in DIC_DIRS:
        if not dd.is_dir():
            raise FileNotFoundError(f"{dd}: directory does not exist.")


    save_path = Path.cwd() / "images_dic_pulse25X"
    if not save_path.is_dir():
        save_path.mkdir(exist_ok=True,parents=True)

    #---------------------------------------------------------------------------
    # Load simulation data
    print("LOAD SIM DATA")
    print(80*"-")
    sim_data = {}

    sim_coord_path = FE_DIR / "Mesh.csv"
    # sim_field_paths = {"disp_x": FE_DIR / "u (m)_All.npy",
    #                    "disp_y": FE_DIR / "v (m)_All.npy",
    #                    "disp_z": FE_DIR / "w (m)_All.npy",}

    sim_field_paths = {"disp_x": FE_DIR / "u (m)_All.npy",
                       "disp_y": FE_DIR / "v (m)_All.npy",
                       "disp_z": FE_DIR / "w (m)_All.npy",}

    # Load simulation nodal coords
    print(f"Loading sim coords from:\n    {sim_coord_path}")
    start_time = time.perf_counter()
    sim_coords_df = pd.read_csv(sim_coord_path)
    sim_coords = sim_coords_df.to_numpy()
    sim_coords = sim_coords*conv_to_mm
    end_time = time.perf_counter()
    print(f"Loading sim coords took: {end_time-start_time}s\n")
    del sim_coords_df

    sim_num_nodes = sim_coords.shape[0]

    # Load simulation field data
    print(f"Loading simulation data from:\n    {FE_DIR}")
    start_time = time.perf_counter()
    sim_temp = {}
    for ss in sim_field_paths:
        # shape=(n_pts,n_doe)
        sim_temp[ss] = np.load(sim_field_paths[ss])
        # shape=(n_doe,n_pts)
        sim_temp[ss] = sim_temp[ss].T
        print(f"{sim_temp[ss].shape=}")
        # shape=(n_epis,n_alea,n_pts)
        sim_temp[ss] = sim_temp[ss].reshape(SIM_EPIS_N,SIM_ALEA_N,sim_num_nodes)

        if "disp" in ss:
            sim_temp[ss] = sim_temp[ss]*conv_to_mm

    # Push the displacement data into a single matrix from the separate files
    # shape=(n_epis,n_alea,n_nodes,n_comps[x,y,z])
    sim_disp = np.zeros((SIM_EPIS_N,SIM_ALEA_N,sim_num_nodes,3),dtype=np.float64)

    print()
    print(f"{sim_coords.shape=}")
    print(f"{sim_temp['disp_x'].shape=}")
    print()

    sim_disp[:,:,:,0] = sim_temp["disp_x"]
    sim_disp[:,:,:,1] = sim_temp["disp_y"]
    sim_disp[:,:,:,2] = sim_temp["disp_z"]
    del sim_temp


    # First column of sim coords is the node number, remove it
    sim_coords = sim_coords[:,1:]
    # Add a column of zeros so we have a z coord of 0 as only x and y are given
    # in the coords file
    sim_coords = np.hstack((sim_coords,np.zeros((sim_coords.shape[0],1))))

    print()
    print("sim_coords=")
    print(sim_coords)
    print()

    #---------------------------------------------------------------------------
    # Load experiment data
    print("LOAD EXP DATA")
    print(80*"-")
    exp_coord_temp = temp_path / f"exp_coords_{EXP_IND}.npy"
    exp_disp_temp = temp_path / f"exp_disp_{EXP_IND}.npy"

    if not exp_coord_temp.is_file() and not exp_disp_temp.is_file():

        exp_field_slices = {"coords":slice(2,5),
                            "disp":slice(5,8),}

        exp_load_opts = vm.ExpDataLoadOpts(skip_header=1,
                                           threads_num=8)

        print(f"Loading exp data from:\n    {DIC_DIRS[EXP_IND]}")
        start_time = time.perf_counter()
        exp_data = vm.load_exp_data(DIC_DIRS[EXP_IND],
                                    exp_field_slices,
                                    slice(DIC_STEADY[EXP_IND][0],DIC_STEADY[EXP_IND][1]),
                                    exp_load_opts)
        end_time = time.perf_counter()
        print(f"Loading exp data took: {end_time-start_time}s\n")

        print()
        print("Saving data to numpy binary")
        np.save(exp_coord_temp,exp_data["coords"])
        np.save(exp_disp_temp,exp_data["disp"])

    else:
        exp_data = {}
        print("Loading numpy exp data from:"
              + f"\n    {exp_coord_temp}"
              + f"\n    {exp_disp_temp}")
        start_time = time.perf_counter()
        exp_data["coords"] = np.load(exp_coord_temp)
        exp_data["disp"] = np.load(exp_disp_temp)
        end_time = time.perf_counter()
        print(f"Loading numpy exp data took: {end_time-start_time}s\n")

    print(80*"-")
    print("SIM DATA: SHAPES")
    print("sim_coords.shape=(n_nodes,coord[x,y,z])")
    print(f"{sim_coords.shape=}")
    print("sim_field.shape=(n_nodes,n_doe,n_comp[x,y,z])")
    for ss in sim_data:
        if "coords" not in ss:
            print(f"sim_field[{ss}].shape={sim_data[ss].shape}")
    print()
    print("EXP DATA: SHAPES")
    for ee in exp_data:
        print(f"exp_data[{ee}].shape=(n_subsets,n_frames,n_comps[x,y,z])")
        print(f"{exp_data[ee].shape=}")
    print(80*"-")

    #---------------------------------------------------------------------------
    # EXP/SIM: Swap axes to be consistent with previous processing:
    # shape = (n_frames,n_pts,n_comps).
    # SIDE NOTE: should probably be (n_comps,n_frames,n_pts) for best C memory
    # layout - row major = last dimension is consistent in memory

    exp_coords = np.ascontiguousarray(
        np.swapaxes(exp_data["coords"],0,1))
    exp_disp = np.ascontiguousarray(
        np.swapaxes(exp_data["disp"],0,1))
    del exp_data

    print("SWAP AXES")
    print("shape=(n_frames/exps,n_space_pts,n_comps)")
    print()
    print("SIM DATA: Swap Axes")
    print(f"{sim_coords.shape=}")
    print(f"{sim_disp.shape=}")
    print()
    print("EXP DATA: Swap Axes")
    print(f"{exp_coords.shape=}")
    print(f"{exp_disp.shape=}")
    print(80*"-")


    #---------------------------------------------------------------------------
    # SIM: find limiting epistemic errors / CDFs
    sim_cdf_max_sum = 0.0
    sim_cdf_min_sum = 0.0
    sim_cdf_max = {}
    sim_cdf_min = {}

    # Which espistemic error causes the extreme cdfs?
    # For a fixed point and fixed component
    for ee in range(sim_disp.shape[0]): # loop over epistemic errors

        for pp in range(sim_disp.shape[2]): # loop over points
            for cc in range(sim_disp.shape[3]): # loop over components
                # Calculate the ecdf over aleatory errors
                this_cdf = stats.ecdf(sim_disp[ee,:,pp,cc]).cdf
                this_cdf_sum = np.sum(this_cdf.quantiles)

                # Check cdf limits
                if this_cdf_sum > sim_cdf_max_sum:
                    sim_cdf_max_sum = this_cdf_sum

                if this_cdf_sum < sim_cdf_min_sum:
                    sim_cdf_min_sum = this_cdf_sum



                # Calculate ecdf for each point in the field over aleatory errors

    return
    #---------------------------------------------------------------------------
    # Transform simulation coords
    # NOTE: field arrays have shape=(n_doe_samps,n_pts,n_comps)

    # Expects shape=(n_pts,coord[x,y,z]), outputs 4x4 transform matrix
    sim_to_world_mat = vm.fit_coord_matrix(sim_coords)
    world_to_sim_mat = np.linalg.inv(sim_to_world_mat)
    print("Sim to world matrix:")
    print(sim_to_world_mat)
    print()
    print("World to sim matrix:")
    print(world_to_sim_mat)
    print()

    print("Adding w coord and rotating sim coords")
    sim_with_w = np.hstack([sim_coords,
                            np.ones([sim_coords.shape[0],1])])
    print(f"{sim_with_w.shape=}")

    sim_coords = np.matmul(world_to_sim_mat,sim_with_w.T).T
    print(f"{sim_coords.shape=}")

    print("Returning sim coords by removing w coord:")
    sim_coords = sim_coords[:,:-1]
    print(f"{sim_coords.shape=}")
    del sim_with_w
    print()

    print(f"{sim_disp.shape=}")
    print()

    print("Transforming simulation coords...")
    sim_disp_t = np.zeros_like(sim_disp)
    for ee in range(0,sim_disp.shape[0]):
        for aa in range(0,sim_disp.shape[1]):
        sim_disp_t[ss,:,:] = np.matmul(world_to_sim_mat[:-1,:-1],
                                       sim_disp[ss,:,:].T).T

        rigid_disp = np.atleast_2d(np.mean(sim_disp_t[ss,:,:],axis=0)).T
        rigid_disp = np.tile(rigid_disp,sim_disp.shape[2]).T
        sim_disp_t[ss,:,:] -= rigid_disp

    sim_disp = sim_disp_t
    del sim_disp_t


    #---------------------------------------------------------------------------
    # Transform Exp Coords: required for each frame
    # NOTE: exp field arrays have shape=(n_frames,n_pts,n_comps)

    print("Transforming experimental coords...")
    print(f"{exp_coords.shape=}")

    exp_coord_t = np.zeros_like(exp_coords)
    exp_disp_t = np.zeros_like(exp_disp)

    for ff in range(0,exp_disp.shape[0]):
        exp_to_world_mat = vm.fit_coord_matrix(exp_coords[ff,:,:])
        world_to_exp_mat = np.linalg.inv(exp_to_world_mat)

        exp_with_w = np.hstack([exp_coords[ff,:,:],
                                np.ones([exp_coords.shape[1],1])])

        exp_coord_temp = np.matmul(world_to_exp_mat,exp_with_w.T).T
        exp_coord_t[ff,:,:] = exp_coord_temp[:,:-1]

        # Flip the y coord for the experiment?
        exp_coord_t[ff,:,1] = -exp_coord_t[ff,:,1]

        exp_disp_t[ff,:,:] = np.matmul(world_to_exp_mat[:-1,:-1],exp_disp[ff,:,:].T).T
        rigid_disp = np.atleast_2d(np.mean(exp_disp_t[ff,:,:],axis=0)).T
        rigid_disp = np.tile(rigid_disp,exp_disp.shape[1]).T
        exp_disp_t[ff,:,:] -= rigid_disp

    exp_coords = exp_coord_t
    exp_disp = exp_disp_t
    del exp_coord_t, exp_disp_t

    print("Coord transforms complete.")
    print()

    print("After transformation:")
    print(f"{exp_coords.shape=}")
    print(f"{exp_disp.shape=}")
    print()

    #---------------------------------------------------------------------------
    # Comparison of simulation and experimental coords
    PLOT_COORD_COMP = False

    if PLOT_COORD_COMP:
        down_samp: int = 5
        frame: int = int(round(exp_disp.shape[0]/2))

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.scatter(sim_coords[:,0],
                    sim_coords[:,1],
                    sim_coords[:,2],
                    label="sim")
        ax.scatter(exp_coords[frame,::down_samp,0],
                    exp_coords[frame,::down_samp,1],
                    exp_coords[frame,::down_samp,2],
                    label="exp")

        #ax.set_zlim(-1.0,1.0)
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.scatter(sim_coords[:,0],sim_coords[:,1],
                   label="sim")
        ax.scatter(exp_coords[frame,::down_samp,0],
                   exp_coords[frame,::down_samp,1],
                   label="exp")
        ax.legend()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.show()

    #---------------------------------------------------------------------------
    # Plot displacement fields on transformed coords
    print("Plotting displacement fields for sim and exp.")

    sim_disp = sim_disp[:,:,[1,2,0]]
    # Based on the figures:
    # exp_disp_0 = sim_disp_1 = X
    # exp_disp_1 = sim_disp_2 = Y
    # exp_disp_2 = sim_disp_0 = Z

    sim_x_min = np.min(sim_coords[:,0])
    sim_x_max = np.max(sim_coords[:,0])
    sim_y_min = np.min(sim_coords[:,1])
    sim_y_max = np.max(sim_coords[:,1])

    PLOT_DISP_SIMEXP = True

    if PLOT_DISP_SIMEXP:
        frame: int = int(round(exp_disp.shape[0]/2))
        div_n = 1000

        x_vec = np.linspace(sim_x_min,sim_x_max,div_n)
        y_vec = np.linspace(sim_y_min,sim_y_max,div_n)

        (x_grid,y_grid) = np.meshgrid(x_vec,y_vec)

        for aa in range(0,3):
            exp_disp_grid = griddata(exp_coords[frame,:,0:2],
                                     exp_disp[frame,:,aa],
                                     (x_grid,y_grid),
                                     method="linear")

            fig,ax = plt.subplots()
            image = ax.imshow(exp_disp_grid,
                              extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))
            #ax.scatter(exp_coords[frame,:,0],exp_coords[frame,:,1])
            plt.title(f"Exp Data: disp_{aa}")
            plt.colorbar(image)
            plt.savefig(
                save_path/f"exp{DIC_PULSES[EXP_IND]}_map_{SIM_TAG}_disp{aa}.png")


        for aa in range(0,3):
            sim_disp_grid = griddata(sim_coords[:,0:2],
                                     sim_disp[0,:,aa],
                                     (x_grid,y_grid),
                                     method="linear")

            fig,ax = plt.subplots()
            image = ax.imshow(sim_disp_grid,extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))
            #ax.scatter(sim_coords[:,0],sim_coords[:,1])
            plt.title(f"Sim Data: disp_{aa}")
            plt.colorbar(image)
            plt.savefig(save_path/f"sim_map_{SIM_TAG}_disp{aa}.png")

    plt.close("all")

    #---------------------------------------------------------------------------
    # Average fields from experiment and simulation to plot the difference
    print("\nAveraging experiment steady state and simulation for full-field comparison.")
    # exp_avg_start: int = 300
    # exp_avg_end: int = 650

    # No need to slice in this case as we have only loaded the steady state data
    # exp_coords = exp_coords[exp_avg_start:exp_avg_end,:,:]
    # exp_disp = exp_disp[exp_avg_start:exp_avg_end,:,:]

    # Had to change these to nanmean because of problems in experimental data
    # Again, no need to slice here as we only have steady state data
    exp_coords_avg = np.nanmean(exp_coords,axis=0)
    exp_disp_avg = np.nanmean(exp_disp,axis=0)
    sim_disp_avg = np.nanmean(sim_disp,axis=0)

    print(f"{exp_disp_avg.shape=}")
    print(f"{sim_disp_avg.shape=}")

    elem_size = np.min(np.sqrt(np.sum((sim_coords[1:,:] - sim_coords[0,:])**2,axis=1)))

    tol = 1e-6
    scale = 1/tol
    round_arr = np.round(sim_coords[:,0] * scale) / scale
    num_elem_x = np.unique(round_arr)
    round_arr = np.round(sim_coords[:,1]* scale) / scale
    num_elem_y = np.unique(round_arr)

    print(f"{elem_size=}")
    print()
    print(f"{sim_x_min=}")
    print(f"{sim_x_max=}")
    print(f"{sim_y_min=}")
    print(f"{sim_y_max=}")
    print(f"{(sim_x_max-sim_x_min)=}")
    print(f"{(sim_y_max-sim_y_min)=}")
    print()
    print(f"{num_elem_x.shape=}")
    print(f"{num_elem_y.shape=}")
    print()

    ax_inds = (0,1,2)
    ax_strs = ("x","y","z")

    PLOT_AVG_DISP_MAPS = True

    if PLOT_AVG_DISP_MAPS:
        for ii,ss in zip(ax_inds,ax_strs):
            (fig,ax) = vm.plot_avg_disp_maps_nosave(
                sim_coords,
                sim_disp_avg,
                exp_coords_avg,
                exp_disp_avg,
                ii,
                ss,
                scale_cbar=True,
            )

            save_fig_path = (save_path
                         / f"exp{DIC_PULSES[EXP_IND]}_{SIM_TAG}_disp_{ss}_comp.png")
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

            (fig,ax) = vm.plot_avg_disp_maps_nosave(
                sim_coords,
                sim_disp_avg,
                exp_coords_avg,
                exp_disp_avg,
                ii,
                ss,
                scale_cbar=False
            )

            save_fig_path = (save_path
                / f"exp{DIC_PULSES[EXP_IND]}_{SIM_TAG}_disp_{ss}_comp_cbarfree.png")
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")


    #---------------------------------------------------------------------------
    print(80*"-")
    print("COMPLETE.")
    plt.show()




if __name__ == "__main__":
    main()