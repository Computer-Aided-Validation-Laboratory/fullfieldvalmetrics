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
    print("MAVM Calc for DIC Data with Image Def: Pulse 25X")
    print(80*"=")
    print()

    PARA: int = 8
    #===========================================================================
    EXP_IND: int = 1
    #===========================================================================

    #---------------------------------------------------------------------------
    # General Constants
    comps = (0,1,2)
    (x,y,z) = (0,1,2)
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

    conv_to_mm: float = 1000.0 # Simulation is in SI and exp is in mm

    #---------------------------------------------------------------------------
    # SIM: constants
    SIM_TAG = "imagedefv3"
    FE_DIR = Path.cwd()/ "STC_ProbSim_ImageDef_FieldsFull_25X_v3"

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
    save_path = Path.cwd() / f"images_dic_pulse{EXP_TAG}_sim_{SIM_TAG}_strain"
    if not save_path.is_dir():
        save_path.mkdir(exist_ok=True,parents=True)


    #---------------------------------------------------------------------------
    # Load simulation data
    print(80*"-")
    print("LOAD SIM DATA")
    image_def_sims = {"max":"ImageDef_Max_Disp_TimeStep_10.csv",
                      "med":"ImageDef_Mean_Disp_TimeStep_10.csv",
                      "min":"ImageDef_Min_Disp_TimeStep_10.csv",}

    image_def_slices = {"coords":slice(2,5),
                        "strain":slice(18,21),}

    sim_coords = {}
    sim_strain = {}
    for kk,ff in image_def_sims.items():
        data_path = FE_DIR / ff
        data = pd.read_csv(data_path)
        data = data.to_numpy()

        sim_coords[kk] = data[:,image_def_slices["coords"]]
        sim_strain[kk] = data[:,image_def_slices["strain"]]

    for kk in sim_coords:
        print(f"{kk=}, {sim_coords[kk].shape=}, {sim_strain[kk].shape=}")    

    #---------------------------------------------------------------------------
    # Load experiment data
    print(80*"-")
    print("LOAD EXP DATA")
    FORCE_EXP_LOAD = False
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
    # print("SIM DATA: SHAPES")
    # print("sim_coords.shape=(n_nodes,coord[x,y,z])")
    # print(f"{sim_coords.shape=}")
    # print("sim_field.shape=(n_nodes,n_doe,n_comp[x,y,z])")
    # for ss in sim_data:
    #     if "coords" not in ss:
    #         print(f"sim_field[{ss}].shape={sim_data[ss].shape}")
    # print()
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
    for kk in sim_strain:
        sim_strain[kk] = sim_strain[kk]*FIELD_UNIT_CONV
    #===========================================================================

    print(80*"-")
    print("SIM DATA SHAPE:")
    print("shape=(n_space_pts,n_comps)")
    for kk in sim_coords:
        print(f"{kk=}, {sim_coords[kk].shape=}")
        print(f"{kk=}, {sim_strain[kk].shape=}")
    print()
    print("EXP DATA SHAPE:")
    print("shape=(n_frames,n_space_pts,n_comps)")
    print(f"{exp_coords.shape=}")
    print(f"{exp_strain.shape=}")
    print(80*"-")

    #---------------------------------------------------------------------------
    # Interpolate SIM->EXP coords, average EXP over steady state
    print()
    print(80*"-")
    print("SIM-EXP: interpolating to common grid")
    # NOTE: common coords are EXP coords to minimise interpolations
    
    # Had to change these to nanmean because of problems in experimental data
    exp_coords_avg = np.nanmean(exp_coords,axis=0)
    exp_strain_common_avg = np.nanmean(exp_strain,axis=0)

    coords_common = exp_coords_avg[:,:-1]
    exp_strain_common = exp_strain
    coords_num = coords_common.shape[0]
    
    sim_strain_common = {}
    # NOTE: had to remove the z coord from the interpolation and assume 2D - 
    # otherwise you get a field of nans.
    for case_key,strain_array in sim_strain.items():
        strain_interp = np.zeros((coords_num,3),dtype=np.float64)
        for strain_comp in range(strain_array.shape[-1]):
            strain_interp[:,strain_comp] = griddata(sim_coords[case_key][:,:-1],
                                                     strain_array[:,strain_comp],
                                                     coords_common,
                                                     method='linear')

        sim_strain_common[case_key] = strain_interp


    print()
    print("SIM-EXP: Interpolated data shapes:")
    print(f"{coords_common.shape=}") 
    print(f"{exp_strain_common.shape=}")
    print(f"{exp_strain_common_avg.shape=}")
    print(f"{sim_strain_common['max'].shape=}")
    print(f"{sim_strain_common['med'].shape=}")
    print(f"{sim_strain_common['min'].shape=}")
    print()

    # Interpolate onto a grid 
    x_max = np.max(coords_common[:,x])
    x_min = np.min(coords_common[:,x])
    x_len = x_max - x_min
    y_max = np.max(coords_common[:,y])
    y_min = np.min(coords_common[:,y])
    y_len = y_max - y_min
    aspect_ratio = x_len/y_len
    # x * y = N, x = AR*y, x^2 / AR = N
    x_num_grid_pts = int(np.ceil(np.sqrt(coords_num*aspect_ratio)))
    y_num_grid_pts = int(np.ceil(x_num_grid_pts/aspect_ratio))
    x_grid_size = x_len / x_num_grid_pts
    y_grid_size = y_len / y_num_grid_pts
    x_vec = np.linspace(x_min,x_max,x_num_grid_pts)
    y_vec = np.linspace(y_min,y_max,y_num_grid_pts)
    (x_grid,y_grid) = np.meshgrid(x_vec,y_vec)

    print("COMMON COORD LIMITS:")
    print(f"{x_min=}, {x_max=}")
    print(f"{y_min=}, {y_max=}")
    print(f"{x_len=}, {y_len=}, {aspect_ratio=}")
    print(f"{x_num_grid_pts=}, {y_num_grid_pts=}")
    print(f"{x_grid_size=}, {y_grid_size=}")
    print()
    
    coords_grid = np.hstack((x_grid[:],y_grid[:]))

    exp_strain_grid = np.zeros((y_num_grid_pts,
                                x_num_grid_pts,
                                3,)
                                ,dtype=np.float64)
    for ss in range(0,exp_strain_common_avg.shape[-1]):
        exp_strain_grid[:,:,ss] = griddata(coords_common,
                                           exp_strain_common_avg[:,ss],
                                           (x_grid,y_grid),
                                           method="linear")
        #print(f"{ss=}, {exp_strain_grid_avg.shape=}")

    sim_strain_grid = {} 
    for key,strain_array in sim_strain_common.items():     
        sim_strain_temp = np.zeros((y_num_grid_pts,
                                    x_num_grid_pts,
                                    3,)
                                    ,dtype=np.float64)
        for ss in range(0,3):
            sim_strain_temp[:,:,ss] = griddata(coords_common,
                                               strain_array[:,ss],
                                               (x_grid,y_grid),
                                               method="linear")
        sim_strain_grid[key] = sim_strain_temp 

    print("EXP, SIM DATA ON GRID:")
    print(f"{x_grid.shape=}, {y_grid.shape=}")
    print(f"{exp_strain_grid.shape=}")
    print(f"{sim_strain_grid['max'].shape=}")
    print(f"{sim_strain_grid['med'].shape=}")
    print(f"{sim_strain_grid['min'].shape=}")
    print()

    #---------------------------------------------------------------------------


    # INPUTS
    
    sim_field_grid = sim_strain_grid["min"]
    exp_field_grid = exp_strain_grid
    cbar_field_same = True
    cbar_field_sym = False
    cbar_diff_sym = True
    
    extent = (x_min,x_max,y_min,y_max)
    cbar_font_size = 6.0
    plot_opts = pyvale.sensorsim.PlotOptsGeneral()
    fig_size = (plot_opts.a4_print_width,
                plot_opts.a4_print_width/(plot_opts.aspect_ratio*2.8))

    tag_dict = {"min":"Min.","max":"Max.","med":"Med."}

    for kk in sim_strain_grid:
        sim_field_grid = sim_strain_grid[kk]
                
        for ax_ind,ax_str in enumerate(STRAIN_COMP_STRS):
            field_str = FIELD_AX_STRS[ax_ind]

            field_diff_grid = (sim_field_grid[:,:,ax_ind] 
                              - exp_field_grid[:,:,ax_ind])

            field_color_max = np.nanmax((np.nanmax(sim_field_grid[:,:,ax_ind]),
                                         np.nanmax(exp_field_grid[:,:,ax_ind])))
            field_color_min = np.nanmin((np.nanmin(sim_field_grid[:,:,ax_ind]),
                                         np.nanmin(exp_field_grid[:,:,ax_ind])))
            if cbar_field_sym:
                field_color_lim = np.max(
                    (np.abs(field_color_max),np.abs(field_color_min))
                )
                field_color_max = field_color_lim
                field_color_min = -field_color_lim

            if cbar_diff_sym:
                diff_color_lim = np.nanmax(
                    (np.abs(np.nanmax(field_diff_grid[:])),
                     np.abs(np.nanmin(field_diff_grid[:])))
                )
                diff_color_max = diff_color_lim 
                diff_color_min = -diff_color_lim
                
            #-------------------------------------------------------------------
            # FIG setup
            fig,ax = plt.subplots(1,3,figsize=fig_size,layout='constrained')
            fig.set_dpi(plot_opts.resolution)

            #-------------------------------------------------------------------
            # Fig 0: EXP Field
            if cbar_field_same:
                image = ax[0].imshow(exp_field_grid[:,:,ax_ind],
                                     extent=extent,
                                     vmin=field_color_min,
                                     vmax=field_color_max)
            else: 
                image = ax[0].imshow(exp_field_grid[:,:,ax_ind],
                                     extent=extent)

            ax[0].set_title(
                f"Exp. Avg. \n{field_str} [{FIELD_UNIT_STR}]",
                fontsize=plot_opts.font_head_size, 
                fontname=plot_opts.font_name
            )
            cbar = plt.colorbar(image)
            
            #-------------------------------------------------------------------
            # Fig 1: SIM FIELD
            if cbar_field_same:
                image = ax[1].imshow(sim_field_grid[:,:,ax_ind],
                                     extent=extent,
                                     vmin = field_color_min,
                                     vmax = field_color_max)
            else: 
                image = ax[1].imshow(sim_field_grid[:,:,ax_ind],
                                     extent=extent)

            ax[1].set_title(
                f"Sim. Avg.\n{field_str} [{FIELD_UNIT_STR}], {tag_dict[kk]}",
                fontsize=plot_opts.font_head_size, 
                fontname=plot_opts.font_name
            )
            cbar = plt.colorbar(image)
            

            #-------------------------------------------------------------------
            # Fig 2: DIFF
            if cbar_diff_sym:
                image = ax[2].imshow(field_diff_grid,
                                     extent=extent,
                                     vmin = diff_color_min,
                                     vmax = diff_color_max,
                                     cmap="RdBu")
            else: 
                image = ax[2].imshow(field_diff_grid,
                                     extent=extent,
                                     cmap="RdBu")    

            ax[2].set_title(
                f"(Sim. - Exp.)\n{field_str} [{FIELD_UNIT_STR}], {tag_dict[kk]}",
                fontsize=plot_opts.font_head_size, 
                fontname=plot_opts.font_name
            )
            cbar = plt.colorbar(image)

            #-------------------------------------------------------------------
            # Turn of x,y axis ticks
            for aa in ax:
                aa.set_xticks([])
                aa.set_yticks([])
                for spine in aa.spines.values():
                    spine.set_visible(False)

            save_name = f"exp_{EXP_TAG}_sim_{kk}_{SIM_TAG}_strain_{ax_str}.png"
            save_fig_path = save_path / save_name

            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")
                  
    #---------------------------------------------------------------------------
    print(80*"-")
    print("COMPLETE.")
    plt.show()
        
 
if __name__ == "__main__":
    main()
