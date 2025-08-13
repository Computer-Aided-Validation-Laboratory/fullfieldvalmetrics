import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import griddata
import pyvale
import valmetrics as vm

def main() -> None:
    print(80*"=")
    print("MAVM Calc for DIC Data: Pulse 25X")
    print(80*"=")
    print()

    PARA: int = 8

    #===========================================================================
    EXP_IND: int = 0
    #===========================================================================

    comps = (0,1,2)
    (xx,yy,zz) = (0,1,2)

    ax_inds = (0,1,2)
    ax_strs = ("x","y","z")

    plot_opts = pyvale.PlotOptsGeneral()
    fig_ind: int = 0
    exp_c: str = "tab:orange"
    sim_c: str = "tab:blue"
    mavm_c: str = "tab:green"

    DISP_COMP_STRS = ("x","y","z")

    #---------------------------------------------------------------------------
    # SIM: constants
    SIM_TAG = "red"
    FE_DIR = Path.cwd()/ "STC_ProbSim_FieldsReduced_25X"
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
        Path.cwd() / "STC_Exp_DIC_253",
        Path.cwd() / "STC_Exp_DIC_254",
        Path.cwd() / "STC_Exp_DIC_255",
    )
    # NOTE: first 100 frames are averaged to create the steady state reference
    # as frame 0000 the test data starts at frame 0100 and we need to then take
    # frames based on this frame number
    FRAME_OFFSET: int = 99
    DIC_STEADY = [(297-FRAME_OFFSET,694-FRAME_OFFSET+1),
                  (302-FRAME_OFFSET,694-FRAME_OFFSET+1),
                  (293-FRAME_OFFSET,694-FRAME_OFFSET+1)] # Need to add 1 to slice

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
    FORCE_EXP_LOAD = False
    exp_coord_temp = temp_path / f"exp{EXP_IND}_coords.npy"
    exp_disp_temp = temp_path / f"exp{EXP_IND}_disp.npy"

    if FORCE_EXP_LOAD or (
        not exp_coord_temp.is_file() and not exp_disp_temp.is_file()):

        exp_field_slices = {"coords":slice(2,5),
                            "disp":slice(5,8),}

        exp_load_opts = vm.ExpDataLoadOpts(skip_header=1,
                                           threads_num=PARA)

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
    # SIM: Transform simulation coords
    # NOTE: field arrays have shape=(n_doe_samps,n_pts,n_comps)
    print(80*"-")
    print("SIM: Fitting transformation matrix...")

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

    #---------------------------------------------------------------------------
    # SIM: Transform simulation displacements
    # NOTE: field arrays have shape=(n_doe_samps,n_pts,n_comps)
    print(80*"-")
    print("SIM: Transforming simulation displacements...")
    print(f"{sim_disp.shape=}")
    print()

    sim_disp_t = np.zeros_like(sim_disp)
    for ee in range(0,sim_disp.shape[0]):
        for aa in range(0,sim_disp.shape[1]):
            sim_disp_t[ee,aa,:,:] = np.matmul(world_to_sim_mat[:-1,:-1],
                                        sim_disp[ee,aa,:,:].T).T

            rigid_disp = np.atleast_2d(np.mean(sim_disp_t[ee,aa,:,:],axis=0)).T
            rigid_disp = np.tile(rigid_disp,sim_disp.shape[2]).T
            sim_disp_t[ee,aa,:,:] -= rigid_disp

    sim_disp = sim_disp_t
    del sim_disp_t

    #---------------------------------------------------------------------------
    # EXP: Transform coords, required for each frame
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
    # EXP-SIM Comparison of oords
    PLOT_COORD_COMP = False

    if PLOT_COORD_COMP:
        down_samp: int = 5
        exp_frame: int = int(round(exp_disp.shape[0]/2))

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.scatter(sim_coords[:,0],
                    sim_coords[:,1],
                    sim_coords[:,2],
                    label="sim")
        ax.scatter(exp_coords[exp_frame,::down_samp,0],
                    exp_coords[exp_frame,::down_samp,1],
                    exp_coords[exp_frame,::down_samp,2],
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
        ax.scatter(exp_coords[exp_frame,::down_samp,0],
                   exp_coords[exp_frame,::down_samp,1],
                   label="exp")
        ax.legend()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.show()


    #---------------------------------------------------------------------------
    # Plot displacement fields on transformed coords
    print("Plotting displacement fields for sim and exp.")

    sim_disp = sim_disp[:,:,:,[1,2,0]]
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
        # Frame to plot from the experiment
        exp_frame: int = int(round(exp_disp.shape[0]/2))
        # Sample from the DOE to plot
        sim_plot_epis: int = 0
        sim_plot_alea: int = 0
        div_n = 1000

        x_vec = np.linspace(sim_x_min,sim_x_max,div_n)
        y_vec = np.linspace(sim_y_min,sim_y_max,div_n)

        (x_grid,y_grid) = np.meshgrid(x_vec,y_vec)

        for aa in range(0,3):
            exp_disp_grid = griddata(exp_coords[exp_frame,:,0:2],
                                     exp_disp[exp_frame,:,aa],
                                     (x_grid,y_grid),
                                     method="linear")

            fig,ax = plt.subplots()
            image = ax.imshow(exp_disp_grid,
                              extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))
            #ax.scatter(exp_coords[frame,:,0],exp_coords[frame,:,1])
            plt.title(f"Exp Data: disp. {DISP_COMP_STRS[aa]}")
            plt.colorbar(image)
            save_fig_path = (save_path/f"exp{DIC_PULSES[EXP_IND]}_map_{SIM_TAG}_disp_{DISP_COMP_STRS[aa]}.png")
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

        for aa in range(0,3):
            sim_disp_grid = griddata(sim_coords[:,0:2],
                                     sim_disp[sim_plot_epis,sim_plot_alea,:,aa],
                                     (x_grid,y_grid),
                                     method="linear")

            fig,ax = plt.subplots()
            image = ax.imshow(sim_disp_grid,extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))
            #ax.scatter(sim_coords[:,0],sim_coords[:,1])
            plt.title(f"Sim Data: disp. {DISP_COMP_STRS[aa]}")
            plt.colorbar(image)

            save_fig_path = (save_path/f"sim_map_{SIM_TAG}_disp_{DISP_COMP_STRS[aa]}.png")
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

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
    # Average twice, once over epistemic uncertainty and once over aleatory
    sim_disp_avg = np.nanmean(sim_disp,axis=0)
    sim_disp_avg = np.nanmean(sim_disp_avg,axis=0)

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



    PLOT_AVG_DISP_MAPS = True

    if PLOT_AVG_DISP_MAPS:
        print("Plotting avg. disp. maps and sim-exp diff.")

        for ii,ss in zip(ax_inds,ax_strs):
            field_str = f"disp. {ss} [mm]"

            (fig,ax) = vm.plot_avg_field_maps_nosave(
                sim_coords,
                sim_disp_avg,
                exp_coords_avg,
                exp_disp_avg,
                ii,
                field_str,
                scale_cbar=True,
            )

            save_fig_path = (save_path
                         / f"exp{DIC_PULSES[EXP_IND]}_{SIM_TAG}_dispavg_{ss}_comp.png")
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

            (fig,ax) = vm.plot_avg_field_maps_nosave(
                sim_coords,
                sim_disp_avg,
                exp_coords_avg,
                exp_disp_avg,
                ii,
                field_str,
                scale_cbar=False
            )

            save_fig_path = (save_path
                / f"exp{DIC_PULSES[EXP_IND]}_{SIM_TAG}_dispavg_{ss}_comp_cbarfree.png")
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")


    #---------------------------------------------------------------------------
    # EXP-SIM: interpolate fields to a common grid
    print()
    print(80*"-")
    print("SIM-EXP: interpolating to common grid")

    sim_x_min = np.min(sim_coords[:,0])
    sim_x_max = np.max(sim_coords[:,0])
    sim_y_min = np.min(sim_coords[:,1])
    sim_y_max = np.max(sim_coords[:,1])

    step = 0.5
    x_vec = np.arange(sim_x_min,sim_x_max,step)
    y_vec = np.arange(sim_y_min,sim_y_max,step)
    (x_grid,y_grid) = np.meshgrid(x_vec,y_vec)

    FORCE_INTERP_COMMON = False
    sim_disp_common_path = temp_path / f"sim_disp_common_{SIM_TAG}.npy"
    exp_disp_common_path = temp_path / f"exp{EXP_IND}_disp_common.npy"

    # Need to reshape simulation data to collapse epis and alea errors then
    # interpolate and then reshape back.

    sim_shape = sim_disp.shape
    sim_disp = np.reshape(sim_disp,(SIM_EPIS_N*SIM_ALEA_N,sim_shape[2],sim_shape[3]))
    print(f"{sim_shape=}")
    print(f"{sim_disp.shape=}")
    print()

    if (FORCE_INTERP_COMMON or not sim_disp_common_path.is_file()):

        print("Interpolating simulation displacements to common grid.")
        start_time = time.perf_counter()
        sim_disp_common = vm.interp_sim_to_common_grid(sim_coords,
                                                       sim_disp,
                                                       x_grid,
                                                       y_grid,
                                                       run_para=PARA)
        end_time = time.perf_counter()
        print(f"Interpolating sim. displacements took: {end_time-start_time}s\n")
        print()
        print("Reshaping common simulation disp to split epis and alea errors.")
        print(f"{sim_disp_common.shape=}")
        sim_disp_common = np.reshape(sim_disp_common,sim_shape)
        print(f"{sim_disp_common.shape=}")
        print()

        print("Saving interpolated common grid data in npy format for speed.")
        np.save(sim_disp_common_path,sim_disp_common)

    else:
        print("Loading pre-interpolated sim disp data for speed.")
        sim_disp_common = np.load(sim_disp_common_path)


    if (FORCE_INTERP_COMMON or not exp_disp_common_path.is_file()):

        print("Interpolating experiment displacements to common grid.")
        start_time = time.perf_counter()
        exp_disp_common = vm.interp_exp_to_common_grid(exp_coords,
                                                       exp_disp,
                                                       x_grid,
                                                       y_grid,
                                                       run_para=PARA)
        end_time = time.perf_counter()
        print(f"Interpolating exp. displacements took: {end_time-start_time}s\n")

        print("Saving interpolated common grid data in npy format for speed.")
        np.save(exp_disp_common_path,exp_disp_common)

    else:
        print("Loading pre-interpolated exp disp data for speed.")
        exp_disp_common = np.load(exp_disp_common_path)


    coords_common = np.vstack((x_grid.flatten(),y_grid.flatten())).T

    print()
    print("SIM-EXP: Interpolated data shapes:")
    print(f"{sim_disp_common.shape=}")
    print(f"{exp_disp_common.shape=}")
    print(f"{coords_common.shape=}")
    print()

    coord_common_file = temp_path / "coord_common_for_disp.npy"
    if not coord_common_file.is_file():
        print("Saving common coords")
        np.save(coord_common_file,coords_common)


    # Remove coords and disp to prevent errors
    del exp_coords, exp_disp, sim_coords, sim_disp

    #---------------------------------------------------------------------------
    # SIM-EXP: Calculate mavm at a few key points
    print(80*"-")
    print("SIM-EXP: finding key points in fields and plotting cdfs")

    find_point_x = np.array([24.0,-16.0]) # mm
    find_point_yz = np.array([0.0,-16.0])  # mm

    mavm_inds = np.zeros((3,),dtype=np.uintp)
    mavm_inds[xx] = vm.find_nearest_points(coords_common,find_point_x,k=3)[0]
    mavm_inds[yy] = vm.find_nearest_points(coords_common,find_point_yz,k=3)[0]
    mavm_inds[zz] = mavm_inds[yy]


    print(80*"-")
    print(f"{mavm_inds=}")
    print()
    print(f"{coords_common[mavm_inds[xx],:]=}")
    print(f"{coords_common[mavm_inds[yy],:]=}")
    print(f"{coords_common[mavm_inds[zz],:]=}")
    print(80*"-")
    print()

    print("Summing along aleatory axis and finding max/min...")
    sim_limits = np.sum(sim_disp_common,axis=1)
    sim_cdf_eind = {}
    sim_cdf_eind['max'] = np.argmax(sim_limits,axis=0)
    sim_cdf_eind['min'] = np.argmin(sim_limits,axis=0)

    print(f"{sim_disp_common.shape=}")
    print(f"{sim_limits.shape=}")
    print(f"{sim_cdf_eind['max'].shape=}")
    print(f"{sim_cdf_eind['min'].shape=}")
    print()
    print(f"{sim_cdf_eind['max'][0,0]=}")
    print(f"{sim_cdf_eind['min'][0,0]=}")
    print()

    PLOT_COMMON_PT_CDFS = True

    if PLOT_COMMON_PT_CDFS:
        print("Plotting all sim cdfs and limit cdfs for key points on common coords...")
        for cc in comps:
            pp = mavm_inds[cc]
            fig, axs=plt.subplots(1,1,
                                figsize=plot_opts.single_fig_size_landscape,
                                layout="constrained")
            fig.set_dpi(plot_opts.resolution)

            for ee in range(sim_disp_common.shape[0]):
                axs.ecdf(sim_disp_common[ee,:,pp,cc]
                        ,color='tab:blue',linewidth=plot_opts.lw)

            e_ind = sim_cdf_eind['max'][pp,cc]
            axs.ecdf(sim_disp_common[e_ind,:,pp,cc]
                    ,ls="--",color='black',linewidth=plot_opts.lw)

            min_e = sim_cdf_eind['min'][pp,cc]
            axs.ecdf(sim_disp_common[min_e,:,pp,cc]
                    ,ls="--",color='black',linewidth=plot_opts.lw)

            this_coord = coords_common[mavm_inds[cc],:]
            title_str = f"(x,y)=({this_coord[0]:.2f},{-1*this_coord[1]:.2f})"
            ax_str = f"sim disp. {DISP_COMP_STRS[cc]} [mm]"
            axs.set_title(title_str,fontsize=plot_opts.font_head_size)
            axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
            axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
            #axs.legend(loc="upper left",fontsize=6)

            save_fig_path = (save_path/f"sim_dispcom_{DISP_COMP_STRS[cc]}_ptcdfsall_{SIM_TAG}.png")
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")


        print("Plotting all sim-exp comparison cdfs for key points on common coords...")
        for cc in comps:
            pp = mavm_inds[cc]
            fig, axs=plt.subplots(1,1,
                                figsize=plot_opts.single_fig_size_landscape,
                                layout="constrained")
            fig.set_dpi(plot_opts.resolution)

            # SIM CDFS
            max_e = sim_cdf_eind['max'][pp,cc]
            axs.ecdf(sim_disp_common[max_e,:,pp,cc]
                    ,ls="--",color=sim_c,linewidth=plot_opts.lw,
                    label="sim.")

            min_e = sim_cdf_eind['min'][pp,cc]
            axs.ecdf(sim_disp_common[min_e,:,pp,cc]
                    ,ls="--",color=sim_c,linewidth=plot_opts.lw)

            sim_cdf_high = stats.ecdf(sim_disp_common[max_e,:,pp,cc]).cdf
            sim_cdf_low = stats.ecdf(sim_disp_common[min_e,:,pp,cc]).cdf
            axs.fill_betweenx(sim_cdf_high.probabilities,
                            sim_cdf_low .quantiles,
                            sim_cdf_high.quantiles,
                            color=sim_c,
                            alpha=0.2)

            # EXP CDF
            axs.ecdf(exp_disp_common[:,pp,cc]
                    ,ls="-",color=exp_c,linewidth=plot_opts.lw,
                    label="exp.")

            this_coord = coords_common[mavm_inds[cc],:]
            title_str = f"(x,y)=({this_coord[0]:.2f},{-1*this_coord[1]:.2f})"
            ax_str = f"disp. {DISP_COMP_STRS[cc]} [mm]"
            axs.set_title(title_str,fontsize=plot_opts.font_head_size)
            axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
            axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
            axs.legend(loc="upper left",fontsize=6)

            save_fig_path = (save_path
                        /f"exp{DIC_PULSES[EXP_IND]}_dispcom_{DISP_COMP_STRS[cc]}_ptcdfs_{SIM_TAG}.png")
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

    # plt.close("all")


    #---------------------------------------------------------------------------
    # Calculate MAVM at key points

    print(80*"-")
    print("SIM-EXP: Calculating MAVM at key points")

    sim_lim_keys = ("min","max")
    mavm = {}
    mavm_lims = {}
    for cc,aa in enumerate(ax_strs):

        this_mavm = {}
        this_mavm_lim = {}

        pp = mavm_inds[cc]

        dplus_cdf_sum = None
        dminus_cdf_sum = None

        for kk in sim_lim_keys:
            e_ind = sim_cdf_eind[kk][pp,cc]
            this_mavm[kk] = vm.mavm(sim_disp_common[e_ind,:,pp,cc],
                                    exp_disp_common[:,pp,cc])

            check_upper = np.sum(this_mavm[kk]["F_"] + this_mavm[kk]["d+"])
            check_lower = np.sum(this_mavm[kk]["F_"] - this_mavm[kk]["d-"])

            if dplus_cdf_sum is None:
                dplus_cdf_sum = check_upper
                this_mavm_lim["max"] = this_mavm[kk]
            else:
                if check_upper > dplus_cdf_sum:
                    dplus_cdf_sum = check_upper
                    this_mavm_lim["max"] = this_mavm[kk]

            if dminus_cdf_sum is None:
                dminus_cdf_sum = check_lower
                this_mavm_lim["min"] = this_mavm[kk]
            else:
                if check_lower < dminus_cdf_sum:
                    dminus_cdf_sum = dminus_cdf_sum
                    this_mavm_lim["min"] = this_mavm[kk]

        mavm_lims[aa] = this_mavm_lim
        mavm[aa] = this_mavm


    #print(f"{mavm['x']['max']=}")
    # print()
    # print(mavm_lims.keys())
    # print(mavm_lims["x"].keys())
    # print(mavm_lims["x"]["max"].keys())
    plt.close("all")

    print("Plotting MAVM at key points")
    for cc,aa in enumerate(ax_strs):
        pp = mavm_inds[cc]

        fig,axs=plt.subplots(1,1,
                    figsize=plot_opts.single_fig_size_landscape,
                    layout="constrained")
        fig.set_dpi(plot_opts.resolution)

        # SIM CDFS
        max_e = sim_cdf_eind['max'][pp,cc]
        axs.ecdf(sim_disp_common[max_e,:,pp,cc]
                ,ls="--",color=sim_c,linewidth=plot_opts.lw,
                label="sim.")

        min_e = sim_cdf_eind['min'][pp,cc]
        axs.ecdf(sim_disp_common[min_e,:,pp,cc]
                ,ls="--",color=sim_c,linewidth=plot_opts.lw)

        sim_cdf_high = stats.ecdf(sim_disp_common[max_e,:,pp,cc]).cdf
        sim_cdf_low = stats.ecdf(sim_disp_common[min_e,:,pp,cc]).cdf
        axs.fill_betweenx(sim_cdf_high.probabilities,
                        sim_cdf_low .quantiles,
                        sim_cdf_high.quantiles,
                        color=sim_c,
                        alpha=0.2)

        # EXP CDF
        axs.ecdf(exp_disp_common[:,pp,cc]
                ,ls="-",color=exp_c,linewidth=plot_opts.lw,
                label="exp.")

        mavm_c = "tab:red"
        axs.plot(mavm[aa]["min"]["F_"] - mavm[aa]["min"]["d-"],
                 mavm[aa]["min"]["F_Y"], label="min, d-",
                 ls="--",color=mavm_c,linewidth=plot_opts.lw*1.2)
        axs.plot(mavm[aa]["min"]["F_"] + mavm[aa]["min"]["d+"],
                 mavm[aa]["min"]["F_Y"], label="min, d+",
                 ls="-",color=mavm_c,linewidth=plot_opts.lw*1.2)

        mavm_c = "tab:green"
        axs.plot(mavm[aa]["max"]["F_"] - mavm[aa]["max"]["d-"],
                 mavm[aa]["max"]["F_Y"], label="max, d-",
                 ls="--",color= mavm_c,linewidth=plot_opts.lw*1.2)

        axs.plot(mavm[aa]["max"]["F_"] + mavm[aa]["max"]["d+"],
                 mavm[aa]["max"]["F_Y"], label="max, d+",
                 ls="-",color= mavm_c,linewidth=plot_opts.lw*1.2)

        print()
        print(80*"=")
        print(f"{aa=}")
        print(f"{mavm[aa]['min']['d-']=}")
        print(f"{mavm[aa]['min']['d+']=}")
        print(f"{mavm[aa]['max']['d-']=}")
        print(f"{mavm[aa]['max']['d+']=}")
        print(80*"=")
        print()

        this_coord = coords_common[mavm_inds[cc],:]
        title_str = f"(x,y)=({this_coord[0]:.2f},{-1*this_coord[1]:.2f})"
        ax_str = f"disp. {DISP_COMP_STRS[cc]} [mm]"
        axs.set_title(title_str,fontsize=plot_opts.font_head_size)
        axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
        axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
        axs.legend(loc="upper left",fontsize=6)

        save_fig_path = (save_path
            /f"exp{DIC_PULSES[EXP_IND]}_dispcom_{DISP_COMP_STRS[cc]}_allmavm_{SIM_TAG}.png")
        fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")


    print("Plotting mavm limits...")
    for cc,aa in enumerate(ax_strs):
        pp = mavm_inds[cc]

        fig,axs=plt.subplots(1,1,
                    figsize=plot_opts.single_fig_size_landscape,
                    layout="constrained")
        fig.set_dpi(plot_opts.resolution)

        # SIM CDFS
        max_e = sim_cdf_eind['max'][pp,cc]
        axs.ecdf(sim_disp_common[max_e,:,pp,cc]
                ,ls="--",color=sim_c,linewidth=plot_opts.lw,
                label="sim.")

        min_e = sim_cdf_eind['min'][pp,cc]
        axs.ecdf(sim_disp_common[min_e,:,pp,cc]
                ,ls="--",color=sim_c,linewidth=plot_opts.lw)

        sim_cdf_high = stats.ecdf(sim_disp_common[max_e,:,pp,cc]).cdf
        sim_cdf_low = stats.ecdf(sim_disp_common[min_e,:,pp,cc]).cdf
        axs.fill_betweenx(sim_cdf_high.probabilities,
                        sim_cdf_low .quantiles,
                        sim_cdf_high.quantiles,
                        color=sim_c,
                        alpha=0.2)

        # MAVM
        mavm_c = "black"
        axs.plot(mavm_lims[aa]["min"]["F_"] - mavm_lims[aa]["min"]["d-"],
                 mavm_lims[aa]["min"]["F_Y"], label="d-",
                 ls="--",color=mavm_c,linewidth=plot_opts.lw*1.2)
        axs.plot(mavm_lims[aa]["max"]["F_"] + mavm_lims[aa]["max"]["d+"],
                 mavm_lims[aa]["max"]["F_Y"], label="d+",
                 ls="-",color=mavm_c,linewidth=plot_opts.lw*1.2)

        axs.fill_betweenx(mavm_lims[aa]["max"]["F_Y"],
                          mavm_lims[aa]["min"]["F_"] - mavm_lims[aa]["min"]["d-"],
                          mavm_lims[aa]["max"]["F_"] + mavm_lims[aa]["max"]["d+"],
                          color=mavm_c,
                          alpha=0.2)

        this_coord = coords_common[mavm_inds[cc],:]
        title_str = f"(x,y)=({this_coord[0]:.2f},{-1*this_coord[1]:.2f})"
        ax_str = f"disp. {DISP_COMP_STRS[cc]} [mm]"
        axs.set_title(title_str,fontsize=plot_opts.font_head_size)
        axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
        axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
        axs.legend(loc="upper left",fontsize=6)

        save_fig_path = (save_path
            / f"exp{DIC_PULSES[EXP_IND]}_dispcom_{DISP_COMP_STRS[cc]}_mavmlims_{SIM_TAG}.png")
        fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

    #---------------------------------------------------------------------------
    print(80*"-")
    print("COMPLETE.")
    #plt.show()




if __name__ == "__main__":
    main()