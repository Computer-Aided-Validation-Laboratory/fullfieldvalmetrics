import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import griddata
import pyvale
import valmetrics as vm

# NOTE
# - Can calculate difference maps for each epistemic sample so could have
# hundreds, need to limit this to find the limiting cdfs for each case.
# - Collapse full-field simulation data to limiting cdfs for each point?


def main() -> None:
    print(80*"=")
    print("MAVM Calc for DIC Data: Pulse 211")
    print(80*"=")
    print()

    PLOT_AVG_FIELDS_RETURN = False

    PARA: int = 32

    #===========================================================================
    EXP_IND: int = 2
    #===========================================================================

    comps = (0,1,2)
    (xx,yy,zz) = (0,1,2)
    xy = 2

    plot_opts = pyvale.sensorsim.PlotOptsGeneral()
    fig_ind: int = 0
    exp_c: str = "tab:orange"
    sim_c: str = "tab:blue"

    STRAIN_COMP_STRS = ("xx","yy","xy")

    FIELD_UNIT_CONV = 1e3
    FIELD_UNIT_STR = r"$m\epsilon$"
    FIELD_AX_STRS = (r"$e_{xx}$",r"$e_{yy}$",r"$e_{xy}$")

    #---------------------------------------------------------------------------
    # SIM: constants
    SIM_TAG = "full"

    FE_DIR = Path.cwd()/ "STC_ProbSim_FieldsFull_25X"
    conv_to_mm: float = 1000.0 # Simulation is in SI and exp is in mm

    # Reduced: 5000 = 100 aleatory x 50 epistemic
    # Full: 400 aleatory x 250 epistemic
    # exp_data = exp_data.reshape(samps_n,epis_n,alea_n)
    #samps_n: int = 5000
    SIM_EPIS_N: int = 250 #50
    SIM_ALEA_N: int = 400 #100

    #---------------------------------------------------------------------------
    # EXP: constants

    DIC_PULSES = ("253","254","255")
    DIC_DIRS = (
        Path.cwd() / "STC_Exp_DIC_253",
        Path.cwd() / "STC_Exp_DIC_254",
        Path.cwd() / "STC_Exp_DIC_255",
    )

    EXP_TAG = DIC_PULSES[EXP_IND]

    # NOTE: first 100 frames are averaged to create the steady state reference
    # as frame 0000 the test data starts at frame 0100 and we need to then take
    # frames based on this frame number
    FRAME_OFFSET: int = 99
    DIC_STEADY = [(297-FRAME_OFFSET,694-FRAME_OFFSET+1),
                  (302-FRAME_OFFSET,694-FRAME_OFFSET+1),
                  (293-FRAME_OFFSET,694-FRAME_OFFSET+1)] # Need to add 1 to slice

    #---------------------------------------------------------------------------
    # Check directories exist and create output directories
    temp_path = Path.cwd() / f"temp_exp{EXP_TAG}_sim{SIM_TAG}"
    if not temp_path.is_dir():
        temp_path.mkdir()

    if not FE_DIR.is_dir():
        raise FileNotFoundError(f"{FE_DIR}: directory does not exist.")

    for dd in DIC_DIRS:
        if not dd.is_dir():
            raise FileNotFoundError(f"{dd}: directory does not exist.")

    # SAVE PATH!
    save_path = Path.cwd() / f"images_dic_pulse{EXP_TAG}_sim{SIM_TAG}_strainv2"
    if not save_path.is_dir():
        save_path.mkdir(exist_ok=True,parents=True)

    #---------------------------------------------------------------------------
    # Load simulation data
    print("LOAD SIM DATA")
    print(80*"-")
    sim_data = {}

    sim_coord_path = FE_DIR / "Mesh.csv"
    # sim_field_paths = {"strain_xx": FE_DIR / "solid.el11 (1)_All.npy",
    #                    "strain_yy": FE_DIR / "solid.el22 (1)_All.npy",
    #                    "strain_xy": FE_DIR / "solid.el12 (1)_All.npy",}
    sim_field_paths = {"strain_xx": FE_DIR / "solid.el22 (1)_All.npy",
                       "strain_yy": FE_DIR / "solid.el33 (1)_All.npy",
                       "strain_xy": FE_DIR / "solid.el23 (1)_All.npy",}

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
    sim_strain = np.zeros((SIM_EPIS_N,SIM_ALEA_N,sim_num_nodes,3),dtype=np.float64)

    print()
    print(f"{sim_coords.shape=}")
    print(f"{sim_temp['strain_xx'].shape=}")
    print()

    sim_strain[:,:,:,xx] = sim_temp["strain_xx"]
    sim_strain[:,:,:,yy] = sim_temp["strain_yy"]
    # NOTE: see message from JHJ about making coords consistent
    sim_strain[:,:,:,xy] = sim_temp["strain_xy"] # *-1
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
    FORCE_EXP_LOAD = True
    exp_coord_temp = temp_path / f"exp{EXP_IND}_coords.npy"
    exp_strain_temp = temp_path / f"exp{EXP_IND}_strain.npy"

    if FORCE_EXP_LOAD or (
        not exp_coord_temp.is_file() and not exp_strain_temp.is_file()):

        exp_field_slices = {"coords":slice(2,5),
                            "strain":slice(18,21),}

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
        np.save(exp_strain_temp,exp_data["strain"])

    else:
        exp_data = {}
        print("Loading numpy exp data from:"
              + f"\n    {exp_coord_temp}"
              + f"\n    {exp_strain_temp}")
        start_time = time.perf_counter()
        exp_data["coords"] = np.load(exp_coord_temp)
        exp_data["strain"] = np.load(exp_strain_temp)
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
    exp_strain = np.ascontiguousarray(
        np.swapaxes(exp_data["strain"],0,1))
    del exp_data

    #===========================================================================
    # UNIT CONVERSION
    exp_strain = exp_strain*FIELD_UNIT_CONV
    sim_strain = sim_strain*FIELD_UNIT_CONV
    #===========================================================================

    print("SWAP AXES")
    print("shape=(n_frames/exps,n_space_pts,n_comps)")
    print()
    print("SIM DATA: Swap Axes")
    print(f"{sim_coords.shape=}")
    print(f"{sim_strain.shape=}")
    print()
    print("EXP DATA: Swap Axes")
    print(f"{exp_coords.shape=}")
    print(f"{exp_strain.shape=}")
    print(80*"-")

    print(exp_strain[0,0,:])

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

    # TODO: work out how to do the tensor transformation for the experimental

    #---------------------------------------------------------------------------
    # EXP-SIM Comparison of coords
    PLOT_COORD_COMP = False

    if PLOT_COORD_COMP:
        down_samp: int = 5
        exp_frame: int = int(round(exp_strain.shape[0]/2))

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
    # Plot strain fields on transformed coords
    print("Plotting strain fields for sim and exp.")

    sim_x_min = np.min(sim_coords[:,0])
    sim_x_max = np.max(sim_coords[:,0])
    sim_y_min = np.min(sim_coords[:,1])
    sim_y_max = np.max(sim_coords[:,1])

    PLOT_STRAIN_SIMEXP = False

    if PLOT_STRAIN_SIMEXP:
        # Frame to plot from the experiment
        exp_frame: int = int(round(exp_strain.shape[0]/2))
        # Sample from the DOE to plot
        sim_plot_epis: int = 0
        sim_plot_alea: int = 0
        div_n = 1000

        x_vec = np.linspace(sim_x_min,sim_x_max,div_n)
        y_vec = np.linspace(sim_y_min,sim_y_max,div_n)

        (x_grid,y_grid) = np.meshgrid(x_vec,y_vec)

        for aa in range(0,3):
            exp_strain_grid = griddata(exp_coords[exp_frame,:,0:2],
                                     exp_strain[exp_frame,:,aa],
                                     (x_grid,y_grid),
                                     method="linear")

            fig,ax = plt.subplots()
            image = ax.imshow(exp_strain_grid,
                              extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))
            #ax.scatter(exp_coords[frame,:,0],exp_coords[frame,:,1])
            plt.title(f"exp. strain, {FIELD_AX_STRS[aa]} [{FIELD_UNIT_STR}]")
            plt.colorbar(image)
            plt.savefig(
                save_path/f"exp{EXP_TAG}_map_{SIM_TAG}_strain_{STRAIN_COMP_STRS[aa]}.png")


        for aa in range(0,3):
            sim_strain_grid = griddata(sim_coords[:,0:2],
                                     sim_strain[sim_plot_epis,sim_plot_alea,:,aa],
                                     (x_grid,y_grid),
                                     method="linear")

            fig,ax = plt.subplots()
            image = ax.imshow(sim_strain_grid,extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))
            #ax.scatter(sim_coords[:,0],sim_coords[:,1])
            plt.title(f"sim. strain, {FIELD_AX_STRS[aa]} [{FIELD_UNIT_STR}]")
            plt.colorbar(image)
            save_fig_path = (save_path/f"sim_map_{SIM_TAG}_strain_{STRAIN_COMP_STRS[aa]}.png")
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")



    #---------------------------------------------------------------------------
    # Average fields from experiment and simulation to plot the difference
    print("\nAveraging experiment steady state and simulation for full-field comparison.")

    # Had to change these to nanmean because of problems in experimental data
    exp_coords_avg = np.nanmean(exp_coords,axis=0)
    exp_strain_avg = np.nanmean(exp_strain,axis=0)
    # Average twice, once over epistemic uncertainty and once over aleatory
    sim_strain_avg = np.nanmean(sim_strain,axis=0)
    sim_strain_avg = np.nanmean(sim_strain_avg,axis=0)

    print(f"{exp_strain_avg.shape=}")
    print(f"{sim_strain_avg.shape=}")

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
    ax_strs = ("xx","yy","xy")

    PLOT_AVG_STRAIN_MAPS = True

    if PLOT_AVG_STRAIN_MAPS:
        print("Plotting avg. strain maps and sim-exp diff.")

        for ii,ss in zip(ax_inds,ax_strs):
            field_str = f"strain {FIELD_AX_STRS[ii]} [{FIELD_UNIT_STR}]"

            (fig,ax) = vm.plot_avg_field_maps_nosave(
                sim_coords,
                sim_strain_avg,
                exp_coords_avg,
                exp_strain_avg,
                ii,
                field_str,
                scale_cbar=True,
            )

            save_fig_path = (save_path
                         / f"exp{EXP_TAG}_{SIM_TAG}_strain_{ss}_comp.png")
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

            field_str = f"strain {FIELD_AX_STRS[ii]} [{FIELD_UNIT_STR}]"
            (fig,ax) = vm.plot_avg_field_maps_nosave(
                sim_coords,
                sim_strain_avg,
                exp_coords_avg,
                exp_strain_avg,
                ii,
                field_str,
                scale_cbar=False
            )

            save_fig_path = (save_path
                / f"exp{EXP_TAG}_{SIM_TAG}_strain_{ss}_comp_cbarfree.png")
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

    if PLOT_AVG_FIELDS_RETURN:
        plt.show()
        return

    #---------------------------------------------------------------------------
    # EXP-SIM: interpolate fields to a common grid
    print()
    print(80*"-")
    print("SIM-EXP: interpolating to common grid")

    sim_x_min = np.min(sim_coords[:,0])
    sim_x_max = np.max(sim_coords[:,0])
    sim_y_min = np.min(sim_coords[:,1])
    sim_y_max = np.max(sim_coords[:,1])

    tol = 1e-6
    step = 0.5
    x_vec = np.arange(sim_x_min,sim_x_max+tol,step)
    y_vec = np.arange(sim_y_min,sim_y_max+tol,step)
    (x_grid,y_grid) = np.meshgrid(x_vec,y_vec)
    grid_shape = x_grid.shape
    grid_num_pts = x_grid.size

    FORCE_INTERP_COMMON = False

    sim_strain_common_path = temp_path / f"sim_strain_common_{SIM_TAG}.npy"
    exp_strain_common_path = temp_path / f"exp{EXP_IND}_strain_common.npy"

    # Need to reshape simulation data to collapse epis and alea errors then
    # interpolate and then reshape back.

    sim_shape = sim_strain.shape
    sim_strain = np.reshape(sim_strain,(SIM_EPIS_N*SIM_ALEA_N,sim_shape[2],sim_shape[3]))
    print(f"{sim_shape=}")
    print(f"{sim_strain.shape=}")
    print()

    if (FORCE_INTERP_COMMON or not sim_strain_common_path.is_file()):

        print("Interpolating simulation strain to common grid.")
        start_time = time.perf_counter()
        sim_strain_common = vm.interp_sim_to_common_grid(sim_coords,
                                                       sim_strain,
                                                       x_grid,
                                                       y_grid,
                                                       run_para=PARA)
        end_time = time.perf_counter()
        print(f"Interpolating sim. strain took: {end_time-start_time}s\n")
        print()
        print("Reshaping common simulation strain to split epis and alea errors.")
        print(f"{sim_strain_common.shape=}")
        sim_strain_common = np.reshape(sim_strain_common,sim_shape)
        print(f"{sim_strain_common.shape=}")
        print()

        print("Saving interpolated common grid data in npy format for speed.")
        np.save(sim_strain_common_path,sim_strain_common)

    else:
        print("Loading pre-interpolated sim strain data for speed.")
        sim_strain_common = np.load(sim_strain_common_path)


    if (FORCE_INTERP_COMMON or not exp_strain_common_path.is_file()):

        print("Interpolating experiment strain to common grid.")
        start_time = time.perf_counter()
        exp_strain_common = vm.interp_exp_to_common_grid(exp_coords,
                                                       exp_strain,
                                                       x_grid,
                                                       y_grid,
                                                       run_para=PARA)
        end_time = time.perf_counter()
        print(f"Interpolating exp. strain took: {end_time-start_time}s\n")

        print("Saving interpolated common grid data in npy format for speed.")
        np.save(exp_strain_common_path,exp_strain_common)

    else:
        print("Loading pre-interpolated exp strain data for speed.")
        exp_strain_common = np.load(exp_strain_common_path)


    coords_common = np.vstack((x_grid.flatten(),y_grid.flatten())).T

    print()
    print("SIM-EXP: Interpolated data shapes:")
    print(f"{sim_strain_common.shape=}")
    print(f"{exp_strain_common.shape=}")
    print(f"{coords_common.shape=}")
    print()

    # Remove coords and strain to prevent errors
    # del exp_coords, exp_strain, sim_coords, sim_strain


    coord_common_file = temp_path / "coord_common_for_strain.npy"
    if not coord_common_file.is_file():
        print("Saving common coords")
        np.save(coord_common_file,coords_common)

    #---------------------------------------------------------------------------
    # SIM-EXP: Calculate mavm at a few key points
    print(80*"-")
    print("SIM-EXP: finding key points in fields and plotting cdfs")

    #PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
    # xx,yy = x=0, y=15
    # xy: x= +/-13, y=10
    # find_point_xx = np.array([0.0,-15.0]) # mm
    # find_point_xy = np.array([-13.0,-10.0])  # mm
    find_point_xx = np.array([0.0,-15.0]) # mm
    find_point_xy = np.array([-13.0,-10.0])  # mm
    #PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP

    mavm_inds = np.zeros((3,),dtype=np.uintp)
    mavm_inds[xx] = vm.find_nearest_points(coords_common,find_point_xx,k=3)[0]
    mavm_inds[yy] = mavm_inds[xx]
    mavm_inds[xy] = vm.find_nearest_points(coords_common,find_point_xy,k=3)[0]


    print(80*"-")
    print(f"{mavm_inds=}")
    print()
    print(f"{coords_common[mavm_inds[xx],:]=}")
    print(f"{coords_common[mavm_inds[yy],:]=}")
    print(f"{coords_common[mavm_inds[xy],:]=}")
    print(80*"-")
    print()

    print("Summing along aleatory axis and finding max/min...")
    sim_limits = np.sum(sim_strain_common,axis=1)
    sim_cdf_eind = {}
    sim_cdf_eind['max'] = np.argmax(sim_limits,axis=0)
    sim_cdf_eind['min'] = np.argmin(sim_limits,axis=0)

    print(f"{sim_strain_common.shape=}")
    print(f"{sim_limits.shape=}")
    print(f"{sim_cdf_eind['max'].shape=}")
    print(f"{sim_cdf_eind['min'].shape=}")
    print()
    print(f"{sim_cdf_eind['max'][0,0]=}")
    print(f"{sim_cdf_eind['min'][0,0]=}")
    print()

    PLOT_COMMON_PT_EXP_TRACES = True
    PLOT_COMMON_PT_CDFS = True

    if PLOT_COMMON_PT_EXP_TRACES:
        pass

    if PLOT_COMMON_PT_CDFS:
        print("Plotting all sim cdfs and limit cdfs for key points on common coords...")
        for cc in comps:
            pp = mavm_inds[cc]
            fig, axs=plt.subplots(1,1,
                                figsize=plot_opts.single_fig_size_landscape,
                                layout="constrained")
            fig.set_dpi(plot_opts.resolution)

            for ee in range(sim_strain_common.shape[0]):
                axs.ecdf(sim_strain_common[ee,:,pp,cc]
                        ,color='tab:blue',linewidth=plot_opts.lw)

            e_ind = sim_cdf_eind['max'][pp,cc]
            axs.ecdf(sim_strain_common[e_ind,:,pp,cc]
                    ,ls="--",color='black',linewidth=plot_opts.lw)

            min_e = sim_cdf_eind['min'][pp,cc]
            axs.ecdf(sim_strain_common[min_e,:,pp,cc]
                    ,ls="--",color='black',linewidth=plot_opts.lw)

            this_coord = coords_common[mavm_inds[cc],:]
            title_str = f"(x,y)=({this_coord[0]:.2f},{-1*this_coord[1]:.2f})"
            ax_str = f"sim strain {FIELD_AX_STRS[cc]} [{FIELD_UNIT_STR}]"
            axs.set_title(title_str,fontsize=plot_opts.font_head_size)
            axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
            axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
            #axs.legend(loc="upper left",fontsize=6)

            save_fig_path = (save_path/f"sim_straincom_{STRAIN_COMP_STRS[cc]}_ptcdfsall_{SIM_TAG}.png")
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
            axs.ecdf(sim_strain_common[max_e,:,pp,cc]
                    ,ls="--",color=sim_c,linewidth=plot_opts.lw,
                    label="sim.")

            min_e = sim_cdf_eind['min'][pp,cc]
            axs.ecdf(sim_strain_common[min_e,:,pp,cc]
                    ,ls="--",color=sim_c,linewidth=plot_opts.lw)

            sim_cdf_high = stats.ecdf(sim_strain_common[max_e,:,pp,cc]).cdf
            sim_cdf_low = stats.ecdf(sim_strain_common[min_e,:,pp,cc]).cdf
            axs.fill_betweenx(sim_cdf_high.probabilities,
                            sim_cdf_low .quantiles,
                            sim_cdf_high.quantiles,
                            color=sim_c,
                            alpha=0.2)

            # EXP CDF
            axs.ecdf(exp_strain_common[:,pp,cc]
                    ,ls="-",color=exp_c,linewidth=plot_opts.lw,
                    label="exp.")

            this_coord = coords_common[mavm_inds[cc],:]
            title_str = f"(x,y)=({this_coord[0]:.2f},{-1*this_coord[1]:.2f})"
            ax_str = f"strain {FIELD_AX_STRS[cc]} [{FIELD_UNIT_STR}]"
            axs.set_title(title_str,fontsize=plot_opts.font_head_size)
            axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
            axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
            axs.legend(loc="upper left",fontsize=6)

            save_fig_path = (save_path
                        /f"exp{EXP_TAG}_straincom_{STRAIN_COMP_STRS[cc]}_ptcdfs_{SIM_TAG}.png")
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

    # plt.close("all")


    #---------------------------------------------------------------------------
    # Calculate MAVM at key points

    print(80*"-")
    print("SIM-EXP: Calculating MAVM at key points")

    sim_lim_keys = ("min","max")
    mavm = {}
    mavm_lims = {}

    # sim_strain_common.shape=[epis,alea,point,comp]
    # exp_strain_common.shape=[alea,point,comp]
    # e_ind: int    = epistemic sample index for limit CDF
    # pp: int       = point/coord
    # cc: int       = component index
    for cc,aa in enumerate(ax_strs):

        this_mavm = {}
        this_mavm_lim = {}

        pp = mavm_inds[cc] # pp = point/coord, cc = component index

        dplus_cdf_sum = None
        dminus_cdf_sum = None

        for kk in sim_lim_keys:
            e_ind = sim_cdf_eind[kk][pp,cc]
            this_mavm[kk] = vm.mavm(sim_strain_common[e_ind,:,pp,cc],
                                    exp_strain_common[:,pp,cc])

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
        axs.ecdf(sim_strain_common[max_e,:,pp,cc]
                ,ls="--",color=sim_c,linewidth=plot_opts.lw,
                label="sim.")

        min_e = sim_cdf_eind['min'][pp,cc]
        axs.ecdf(sim_strain_common[min_e,:,pp,cc]
                ,ls="--",color=sim_c,linewidth=plot_opts.lw)

        sim_cdf_high = stats.ecdf(sim_strain_common[max_e,:,pp,cc]).cdf
        sim_cdf_low = stats.ecdf(sim_strain_common[min_e,:,pp,cc]).cdf
        axs.fill_betweenx(sim_cdf_high.probabilities,
                        sim_cdf_low .quantiles,
                        sim_cdf_high.quantiles,
                        color=sim_c,
                        alpha=0.2)

        # EXP CDF
        axs.ecdf(exp_strain_common[:,pp,cc]
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
        ax_str = f"strain {FIELD_AX_STRS[cc]} [{FIELD_UNIT_STR}]"
        axs.set_title(title_str,fontsize=plot_opts.font_head_size)
        axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
        axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
        axs.legend(loc="upper left",fontsize=6)

        save_fig_path = (save_path
            /f"exp{EXP_TAG}_straincom_{STRAIN_COMP_STRS[cc]}_allmavm_{SIM_TAG}.png")
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
        axs.ecdf(sim_strain_common[max_e,:,pp,cc]
                ,ls="--",color=sim_c,linewidth=plot_opts.lw,
                label="sim.")

        min_e = sim_cdf_eind['min'][pp,cc]
        axs.ecdf(sim_strain_common[min_e,:,pp,cc]
                ,ls="--",color=sim_c,linewidth=plot_opts.lw)

        sim_cdf_high = stats.ecdf(sim_strain_common[max_e,:,pp,cc]).cdf
        sim_cdf_low = stats.ecdf(sim_strain_common[min_e,:,pp,cc]).cdf
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
        ax_str = f"strain {FIELD_AX_STRS[cc]} [{FIELD_UNIT_STR}]"
        axs.set_title(title_str,fontsize=plot_opts.font_head_size)
        axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
        axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
        axs.legend(loc="upper left",fontsize=6)

        save_fig_path = (save_path
            / f"exp{EXP_TAG}_straincom_{STRAIN_COMP_STRS[cc]}_mavmlims_{SIM_TAG}.png")
        fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

    plt.close("all")

    #--------------------------------------------------------------------------
    # MAVM FIELD CALCULATION
    FORCE_MAVM_MAP_CALC = False
    mavm_d_plus_path = temp_path / f"mavm_d_plus_exp{EXP_TAG}_sim{SIM_TAG}.npy"
    mavm_d_minus_path = temp_path / f"mavm_d_minus_exp{EXP_TAG}_sim{SIM_TAG}.npy"
    mavm_d_plus_cdf_pts_path = temp_path / f"mavm_d_plus_cdf_pts_exp{EXP_TAG}_sim{SIM_TAG}.npy"
    mavm_d_minus_cdf_pts_path = temp_path / f"mavm_d_minus_cdf_pts_exp{EXP_TAG}_sim{SIM_TAG}.npy"
    mavm_d_plus_cdf_prob_path = temp_path / f"mavm_d_plus_cdf_prob_exp{EXP_TAG}_sim{SIM_TAG}.npy"
    mavm_d_minus_cdf_prob_path = temp_path / f"mavm_d_minus_cdf_prob_exp{EXP_TAG}_sim{SIM_TAG}.npy"

    # NOTE:
    # - Only have aleatory for field data no epistemic
    # - Want to keep the lowest bound:  SIM_LOWER - d-
    # - Want to keep the highest bound: SIM_UPPER + d+
    # - Combinations are for MAVM are:
    #   - EXP -> SIM_UP    = d+,d-
    #   - EXP -> SIM_DOWN  = d+,d-

    if (FORCE_MAVM_MAP_CALC
        or (not mavm_d_plus_path.is_file() and not mavm_d_minus_path.is_file())):
        print("Calculating MAVM d+ and d- over all points for all strain comps.")

        mavm_d_plus_cdf_pts = np.zeros((SIM_ALEA_N,grid_num_pts,3))
        mavm_d_minus_cdf_pts = np.zeros((SIM_ALEA_N,grid_num_pts,3))
        mavm_d_plus_cdf_prob = np.zeros((SIM_ALEA_N,grid_num_pts,3))
        mavm_d_minus_cdf_prob = np.zeros((SIM_ALEA_N,grid_num_pts,3))
        mavm_d_plus = np.zeros((grid_num_pts,3))
        mavm_d_minus = np.zeros((grid_num_pts,3))

        # sim_strain_common.shape=[epis,alea,point,comp], need [ee,:,pp,cc]
        # exp_strain_common.shape=[alea,point,comp], need [:,pp,cc]
        # ee: int    = epistemic sample index for limit CDF
        # pp: int    = point/coord
        # cc: int    = component index


        sim_lim_keys = ("min","max")
        mavm_lims = {}

        analyse_pts = [1816,1716]
        analyse_cmps = [1,]

        for pp in range(0,grid_num_pts): #analyse_pts
            print(f"Calculating MAVM for {pp}/{grid_num_pts}.")

            for cc in range(0,3): #analyse_cmps

                # If the experimental data is nan then we set the mavm to nan
                if np.count_nonzero(np.isnan(exp_strain_common[:,pp,cc])) > 0:
                    mavm_d_plus[pp,cc] = np.nan
                    mavm_d_minus[pp,cc] = np.nan
                    continue

                this_mavm = {}
                # Use these to find the limit cases
                dplus_cdf_sum = None
                dminus_cdf_sum = None

                for kk in sim_lim_keys: # "max" | "min"
                    ee = sim_cdf_eind[kk][pp,cc]
                    # print(80*"-")
                    # print(f"{pp=}")
                    # print(f"{cc=}")
                    # print(f"{kk=}")
                    # print(f"{ee=}")
                    # print(80*"-")

                    this_mavm[kk] = vm.mavm(sim_strain_common[ee,:,pp,cc],
                                            exp_strain_common[:,pp,cc])

                    # NOTE: have to sum then add d!!! Otherwise round off error
                    check_upper = np.sum(this_mavm[kk]["F_"]) + this_mavm[kk]["d+"]
                    if dplus_cdf_sum is None:
                        #print("Set dplus cdf")
                        dplus_cdf_sum = check_upper
                        mavm_d_plus[pp,cc] = this_mavm[kk]["d+"]
                        mavm_d_plus_cdf_pts[:,pp,cc] = this_mavm[kk]["F_"]
                        mavm_d_plus_cdf_prob[:,pp,cc] = this_mavm[kk]["F_Y"]
                    else:
                        if check_upper > dplus_cdf_sum:
                            #print("Update dplus cdf")
                            dplus_cdf_sum = check_upper
                            mavm_d_plus[pp,cc] = this_mavm[kk]["d+"]
                            mavm_d_plus_cdf_pts[:,pp,cc] = this_mavm[kk]["F_"]
                            mavm_d_plus_cdf_prob[:,pp,cc] = this_mavm[kk]["F_Y"]

                    check_lower = np.sum(this_mavm[kk]["F_"]) - this_mavm[kk]["d-"]
                    if dminus_cdf_sum is None:
                        #print("Set dminus cdf")
                        dminus_cdf_sum = check_lower
                        mavm_d_minus[pp,cc] = this_mavm[kk]["d-"]
                        mavm_d_minus_cdf_pts[:,pp,cc] = this_mavm[kk]["F_"]
                        mavm_d_minus_cdf_prob[:,pp,cc] = this_mavm[kk]["F_Y"]
                    else:
                        if check_lower < dminus_cdf_sum:
                            #print("Update dplus cdf")
                            dminus_cdf_sum = dminus_cdf_sum
                            mavm_d_minus[pp,cc] = this_mavm[kk]["d-"]
                            mavm_d_minus_cdf_pts[:,pp,cc] = this_mavm[kk]["F_"]
                            mavm_d_minus_cdf_prob[:,pp,cc] = this_mavm[kk]["F_Y"]


        print("Saving MAVM calculation for faster loading.")
        np.save(mavm_d_plus_path,mavm_d_plus)
        np.save(mavm_d_minus_path,mavm_d_minus)
        np.save(mavm_d_plus_cdf_pts_path,mavm_d_plus_cdf_pts)
        np.save(mavm_d_minus_cdf_pts_path,mavm_d_minus_cdf_pts)
        np.save(mavm_d_plus_cdf_prob_path,mavm_d_plus_cdf_prob)
        np.save(mavm_d_minus_cdf_prob_path,mavm_d_minus_cdf_prob)
    else:
        print("Loading previous MAVM d+ and d- from npy.")
        mavm_d_plus = np.load(mavm_d_plus_path)
        mavm_d_minus = np.load(mavm_d_minus_path)
        mavm_d_plus_cdf_pts = np.load(mavm_d_plus_cdf_pts_path)
        mavm_d_minus_cdf_pts = np.load(mavm_d_minus_cdf_pts_path)
        mavm_d_plus_cdf_prob = np.load(mavm_d_plus_cdf_prob_path)
        mavm_d_minus_cdf_prob = np.load(mavm_d_minus_cdf_prob_path)

    mavm_d_max = np.maximum(mavm_d_minus,mavm_d_plus)


    #--------------------------------------------------------------------------
    # MAVM FIELD PLOTS
    print(80*"-")
    print(f"{mavm_d_plus.shape=}")
    print(f"{mavm_d_minus.shape=}")
    print(f"{mavm_d_plus_cdf_pts.shape=}")
    print(f"{mavm_d_minus_cdf_pts.shape=}")
    print(80*"-")

    ax_strs = ("xx","yy","xy")
    ax_inds = (0,1,2)
    extent = (sim_x_min,sim_x_max,sim_y_min,sim_y_max)
    for ii,ss in zip(ax_inds,ax_strs):
        vm.plot_mavm_map(mavm_d_plus,
                         mavm_d_minus,
                         ii,
                         ss,
                         grid_shape,
                         extent,
                         save_tag=SIM_TAG,
                         field_str="strain",
                         unit_str=FIELD_UNIT_STR,
                         save_path=save_path)

    #---------------------------------------------------------------------------
    # Plot MAVM at follow up points
    # NOTE: y coord should be -'ve here to get to the top of block
    # NOTE: coord is flipped in cdf plots to make it look consistent
    find_pts = {}
    find_pts["yy"] = np.array(((-20.0,-12.0),
                               (20.0,-12.0),
                               (0.0,-15.0)))
    find_pts["xx"] = np.array(((-20.0,-12.0),
                               (20.0,-12.0),
                               (0.0,-15.0)))
    find_pts["xy"] = np.array(((-15.0,0.0),
                               (15.0,0.0),
                               (-14.0,-8.0),
                               (14,-8.0)))


    mavm_pts = {}
    for cc in STRAIN_COMP_STRS:
        this_mavm_pts = np.zeros((find_pts[cc].shape[0],),dtype=np.uintp)
        for pp in range(find_pts[cc].shape[0]):
            this_mavm_pts[pp] = vm.find_nearest_points(coords_common,
                                                        find_pts[cc][pp,:],
                                                        k=3)[0]
        mavm_pts[cc] = this_mavm_pts

    # mavm_pts_yy = vm.find_nearest_points(coords_common,
    #                                      np.array((-16.79,-8.73)),
    #                                      k=2)

    print(80*"-")
    print(f"{mavm_pts['xx']=}")
    print(f"{mavm_pts['yy']=}")
    print(f"{mavm_pts['xy']=}")
    print(80*"-")


    #cc: int = 1 # xx strain component
    for cc,aa in enumerate(STRAIN_COMP_STRS):
        for ii,pp in enumerate(mavm_pts[aa]):
            #-------------------------------------------------------------------
            # CDF COMP
            fig,axs=plt.subplots(1,1,
                    figsize=plot_opts.single_fig_size_landscape,
                    layout="constrained")
            fig.set_dpi(plot_opts.resolution)

            # SIM CDFS
            max_e = sim_cdf_eind['max'][pp,cc]
            axs.ecdf(sim_strain_common[max_e,:,pp,cc]
                    ,ls="--",color=sim_c,linewidth=plot_opts.lw,
                    label="sim.")

            min_e = sim_cdf_eind['min'][pp,cc]
            axs.ecdf(sim_strain_common[min_e,:,pp,cc]
                    ,ls="--",color=sim_c,linewidth=plot_opts.lw)

            sim_cdf_high = stats.ecdf(sim_strain_common[max_e,:,pp,cc]).cdf
            sim_cdf_low = stats.ecdf(sim_strain_common[min_e,:,pp,cc]).cdf
            axs.fill_betweenx(sim_cdf_high.probabilities,
                            sim_cdf_low .quantiles,
                            sim_cdf_high.quantiles,
                            color=sim_c,
                            alpha=0.2)

            axs.ecdf(exp_strain_common[:,pp,cc]
                        ,ls="-",color=exp_c,linewidth=plot_opts.lw,
                        label="exp.")

            this_coord = coords_common[pp,:]
            title_str = f"(x,y)=({this_coord[0]:.2f},{np.abs(this_coord[1]):.2f})"
            ax_str = f"strain {FIELD_AX_STRS[cc]} [{FIELD_UNIT_STR}]"
            axs.set_title(title_str,fontsize=plot_opts.font_head_size)
            axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
            axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
            axs.legend(loc="upper left",fontsize=6)


            save_fig_path = (save_path
                / f"mavm_exp{EXP_TAG}_sim{SIM_TAG}_strain_{STRAIN_COMP_STRS[cc]}_pt{ii}_cdfsonly.png")
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

            #-------------------------------------------------------------------
            # MAVM
            fig,axs=plt.subplots(1,1,
            figsize=plot_opts.single_fig_size_landscape,
            layout="constrained")
            fig.set_dpi(plot_opts.resolution)

            # SIM CDFS
            max_e = sim_cdf_eind['max'][pp,cc]
            axs.ecdf(sim_strain_common[max_e,:,pp,cc]
                    ,ls="--",color=sim_c,linewidth=plot_opts.lw,
                    label="sim.")

            min_e = sim_cdf_eind['min'][pp,cc]
            axs.ecdf(sim_strain_common[min_e,:,pp,cc]
                    ,ls="--",color=sim_c,linewidth=plot_opts.lw)

            sim_cdf_high = stats.ecdf(sim_strain_common[max_e,:,pp,cc]).cdf
            sim_cdf_low = stats.ecdf(sim_strain_common[min_e,:,pp,cc]).cdf
            axs.fill_betweenx(sim_cdf_high.probabilities,
                            sim_cdf_low .quantiles,
                            sim_cdf_high.quantiles,
                            color=sim_c,
                            alpha=0.2)

            # MAVM
            mavm_c = "black"
            axs.plot(mavm_d_minus_cdf_pts[:,pp,cc]- mavm_d_minus[pp,cc],
                     mavm_d_minus_cdf_prob[:,pp,cc], label="d-",
                     ls="--",color=mavm_c,linewidth=plot_opts.lw*1.2)

            axs.plot(mavm_d_plus_cdf_pts[:,pp,cc] + mavm_d_plus[pp,cc],
                     mavm_d_plus_cdf_prob[:,pp,cc], label="d+",
                     ls="-",color=mavm_c,linewidth=plot_opts.lw*1.2)

            axs.fill_betweenx(mavm_d_plus_cdf_prob[:,pp,cc],
                              mavm_d_minus_cdf_pts[:,pp,cc]- mavm_d_minus[pp,cc],
                              mavm_d_plus_cdf_pts[:,pp,cc] + mavm_d_plus[pp,cc],
                              color=mavm_c,
                              alpha=0.2)

            this_coord = coords_common[pp,:]
            title_str = f"(x,y)=({this_coord[0]:.2f},{np.abs(this_coord[1]):.2f})"
            ax_str = f"strain {FIELD_AX_STRS[cc]} [{FIELD_UNIT_STR}]"
            axs.set_title(title_str,fontsize=plot_opts.font_head_size)
            axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
            axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
            axs.legend(loc="upper left",fontsize=6)

            save_fig_path = (save_path
                / f"mavm_exp{EXP_TAG}_sim{SIM_TAG}_strain_{STRAIN_COMP_STRS[cc]}_pt{ii}_simonly.png")
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

            #-------------------------------------------------------------------
            # CDF COMP AND MAVM: EVERYTHING
            fig,axs=plt.subplots(1,1,
            figsize=plot_opts.single_fig_size_landscape,
            layout="constrained")
            fig.set_dpi(plot_opts.resolution)

            # SIM CDFS
            max_e = sim_cdf_eind['max'][pp,cc]
            axs.ecdf(sim_strain_common[max_e,:,pp,cc]
                    ,ls="--",color=sim_c,linewidth=plot_opts.lw,
                    label="sim.")

            min_e = sim_cdf_eind['min'][pp,cc]
            axs.ecdf(sim_strain_common[min_e,:,pp,cc]
                    ,ls="--",color=sim_c,linewidth=plot_opts.lw)

            sim_cdf_high = stats.ecdf(sim_strain_common[max_e,:,pp,cc]).cdf
            sim_cdf_low = stats.ecdf(sim_strain_common[min_e,:,pp,cc]).cdf
            axs.fill_betweenx(sim_cdf_high.probabilities,
                            sim_cdf_low .quantiles,
                            sim_cdf_high.quantiles,
                            color=sim_c,
                            alpha=0.2)

            axs.ecdf(exp_strain_common[:,pp,cc]
                        ,ls="-",color=exp_c,linewidth=plot_opts.lw,
                        label="exp.")

            # MAVM
            mavm_c = "black"
            axs.plot(mavm_d_minus_cdf_pts[:,pp,cc]- mavm_d_minus[pp,cc],
                        mavm_d_minus_cdf_prob[:,pp,cc], label="d-",
                        ls="--",color=mavm_c,linewidth=plot_opts.lw*1.2)

            axs.plot(mavm_d_plus_cdf_pts[:,pp,cc] + mavm_d_plus[pp,cc],
                        mavm_d_plus_cdf_prob[:,pp,cc], label="d+",
                        ls="-",color=mavm_c,linewidth=plot_opts.lw*1.2)

            axs.fill_betweenx(mavm_d_plus_cdf_prob[:,pp,cc],
                              mavm_d_minus_cdf_pts[:,pp,cc]- mavm_d_minus[pp,cc],
                              mavm_d_plus_cdf_pts[:,pp,cc] + mavm_d_plus[pp,cc],
                              color=mavm_c,
                              alpha=0.2)

            this_coord = coords_common[pp,:]
            title_str = f"(x,y)=({this_coord[0]:.2f},{np.abs(this_coord[1]):.2f})"

            ax_str = f"strain {FIELD_AX_STRS[cc]} [{FIELD_UNIT_STR}]"
            axs.set_title(title_str,fontsize=plot_opts.font_head_size)
            axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
            axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
            axs.legend(loc="upper left",fontsize=6)

            save_fig_path = (save_path
                / f"mavm_exp{EXP_TAG}_sim{SIM_TAG}_strain_{STRAIN_COMP_STRS[cc]}_pt{ii}_ALL.png")
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")



    plt.close("all")

    #---------------------------------------------------------------------------
    # Paper figure
    ax_ind = yy
    scale_cbar = True

    for ax_ind,ax_str in enumerate(STRAIN_COMP_STRS):
        field_str = FIELD_AX_STRS[ax_ind]

        sim_x_min = np.min(sim_coords[:,0])
        sim_x_max = np.max(sim_coords[:,0])
        sim_y_min = np.min(sim_coords[:,1])
        sim_y_max = np.max(sim_coords[:,1])

        step = 0.5
        x_vec = np.arange(sim_x_min,sim_x_max,step)
        y_vec = np.arange(sim_y_min,sim_y_max,step)
        (x_grid,y_grid) = np.meshgrid(x_vec,y_vec)

        exp_strain_grid_avg = griddata(exp_coords_avg[:,0:2],
                                exp_strain_avg[:,ax_ind],
                                (x_grid,y_grid),
                                method="linear")

        # This will do minimal interpolation as the input points are the same as the sim
        sim_strain_grid_avg = griddata(sim_coords[:,0:2],
                                sim_strain_avg[:,ax_ind],
                                (x_grid,y_grid),
                                method="linear")

        strain_diff_avg = sim_strain_grid_avg - exp_strain_grid_avg

        color_max = np.nanmax((np.nanmax(sim_strain_grid_avg),np.nanmax(exp_strain_grid_avg)))
        color_min = np.nanmin((np.nanmin(sim_strain_grid_avg),np.nanmin(exp_strain_grid_avg)))

        cbar_font_size = 6.0

        plot_opts = pyvale.sensorsim.PlotOptsGeneral()
        fig_size = (plot_opts.a4_print_width,plot_opts.a4_print_width/(plot_opts.aspect_ratio*2.8))
        fig,ax = plt.subplots(1,4,figsize=fig_size,layout='constrained')
        fig.set_dpi(plot_opts.resolution)

        if scale_cbar:
            image = ax[0].imshow(exp_strain_grid_avg,
                                extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max),
                                vmin = color_min,
                                vmax = color_max)
        else:
            image = ax[0].imshow(exp_strain_grid_avg,
                                extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))

        ax[0].set_title(f"Exp. Avg. \n{field_str} [{FIELD_UNIT_STR}]",
                        fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
        cbar = plt.colorbar(image)


        if scale_cbar:
            image = ax[1].imshow(sim_strain_grid_avg,
                                extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max),
                                vmin = color_min,
                                vmax = color_max)
        else:
            image = ax[1].imshow(sim_strain_grid_avg,
                                extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))

        ax[1].set_title(f"Sim. Avg.\n{field_str} [{FIELD_UNIT_STR}]",
                        fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
        cbar = plt.colorbar(image)


        image = ax[2].imshow(strain_diff_avg,
                            extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max),
                            cmap="RdBu")
        ax[2].set_title(f"(Sim. - Exp.)\n{field_str} [{FIELD_UNIT_STR}]",
                        fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
        cbar = plt.colorbar(image)

        mavm_map = np.reshape(mavm_d_max[:,ax_ind],grid_shape)
        image = ax[3].imshow(mavm_map,
            extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max),
            cmap="plasma")
        d_max_str = r"$d_{max}$"
        ax[3].set_title(f"MAVM {d_max_str}\n{field_str} [{FIELD_UNIT_STR}]",
                        fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
        cbar = plt.colorbar(image)

        for aa in ax:
            aa.set_xticks([])
            aa.set_yticks([])
            for spine in aa.spines.values():
                spine.set_visible(False)

        save_fig_path = (save_path
                        / f"exp{EXP_TAG}_{SIM_TAG}_strain_{ax_str}_maps_and_mavm.png")
        fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

    print(80*"-")
    print("COMPLETE.")
    plt.show()

if __name__ == "__main__":
    main()