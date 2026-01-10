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
    print("Systematic Error Calc from Image Def.: Pulse 25X")
    print(80*"=")
    print()

    PARA: int = 8
    #===========================================================================
    EXP_IND: int = 2
    #===========================================================================

    #---------------------------------------------------------------------------
    # General Constants
    comps = (0,1,2)
    (x,y,z) = (0,1,2)
    (xx,yy,zz) = (0,1,2)
    xy = 2

    plot_opts = pyvale.sensorsim.PlotOptsGeneral()
    fig_ind: int = 0
    id_c: str = "tab:orange"
    fe_c: str = "tab:blue"

    STRAIN_COMP_STRS = ("xx","yy","xy")

    FIELD_UNIT_CONV = 1e3
    FIELD_UNIT_STR = r"$m\epsilon$"
    FIELD_AX_STRS = (r"$e_{xx}$",r"$e_{yy}$",r"$e_{xy}$")

    conv_to_mm: float = 1000.0 # Simulation is in SI and exp is in mm

    case_keys = {"max","med","min"}
    #---------------------------------------------------------------------------
    # SIM: constants
    SIM_TAG = "imagedefv3"
    FE_DIR = Path.cwd()/ "STC_ProbSim_FieldsFull_25X_v3"
    ID_DIR = Path.cwd()/ "STC_ProbSim_ImageDef_FieldsFull_25X_v3"
    
    #---------------------------------------------------------------------------
    # Check directories exist and create output directories
    if not FE_DIR.is_dir():
        raise FileNotFoundError(f"{FE_DIR}: directory does not exist.")

    if not ID_DIR.is_dir():
        raise FileNotFoundError(f"{ID_DIR}: directory does not exist.")

    # SAVE PATH!
    save_path = Path.cwd() / f"imagedef_fe_{SIM_TAG}_strain"
    if not save_path.is_dir():
        save_path.mkdir(exist_ok=True,parents=True)

    #---------------------------------------------------------------------------
    # Load image def data
    print(80*"-")
    print("LOAD IMAGE DEF. DATA")
    id_files = {"max":"ImageDef_Max_Disp_TimeStep_10.csv",
                "med":"ImageDef_Mean_Disp_TimeStep_10.csv",
                "min":"ImageDef_Min_Disp_TimeStep_10.csv",}

    id_slices = {"coords":slice(2,5),
                 "strain":slice(18,21),}

    id_coords = {}
    id_strain = {}
    for kk,ff in id_files.items():
        data_path = ID_DIR / ff
        data = pd.read_csv(data_path)
        data = data.to_numpy()

        id_coords[kk] = data[:,id_slices["coords"]]
        id_strain[kk] = data[:,id_slices["strain"]]

    print("ID DATA:")
    for kk in id_coords:
        print(f"{kk=}, {id_coords[kk].shape=}, {id_strain[kk].shape=}")    

    #---------------------------------------------------------------------------
    # Load simulation data
    print(80*"-")
    print("LOAD SIM DATA")
    fe_files = {"max":"sim_sample_max_strain",
                "med":"sim_sample_med_strain",
                "min":"sim_sample_min_strain",}

    fe_coord_path = FE_DIR / "Mesh.csv"

    # Load simulation nodal coords
    print(f"Loading FE coords from:\n    {fe_coord_path}")
    start_time = time.perf_counter()
    fe_coords = pd.read_csv(fe_coord_path)
    fe_coords = fe_coords.to_numpy()
    fe_coords = fe_coords*conv_to_mm
    end_time = time.perf_counter()
    print(f"Loading FE coords took: {end_time-start_time}s\n")

    # First column of sim coords is the node number, remove it
    fe_coords = fe_coords[:,1:]
    # Add a column of zeros so we have a z coord of 0 as only x and y are given
    # in the coords file
    fe_coords = np.hstack((fe_coords,np.zeros((fe_coords.shape[0],1))))

    fe_num_nodes = fe_coords.shape[0]
    fe_strain = {}
    for kk,ff in fe_files.items():
        fe_strain[kk] = np.zeros((fe_num_nodes,3),dtype=np.float64)
        for ii,cc in enumerate(STRAIN_COMP_STRS):
            strain_file = f"{ff}_{cc}.npy"
            fe_strain[kk][:,ii] = np.load(FE_DIR/strain_file)

    #===========================================================================
    # UNIT CONVERSION
    for kk in fe_strain:
        fe_strain[kk] = fe_strain[kk]*FIELD_UNIT_CONV
        id_strain[kk] = id_strain[kk]*FIELD_UNIT_CONV
    #===========================================================================
    
    print()
    print("ID DATA:")
    for kk in id_coords:
        print(f"{kk=}, {id_coords[kk].shape=}, {id_strain[kk].shape=}")
    print()
    print("FE DATA:")
    for kk in fe_strain:
        print(f"{kk=}, {fe_coords.shape=}, {fe_strain[kk].shape=}")    
    print()

    #---------------------------------------------------------------------------
    # SIM: Transform simulation coords
    # NOTE: field arrays have shape=(n_doe_samps,n_pts,n_comps)
    print(80*"-")
    print("fe: Fitting transformation matrix...")

    # Expects shape=(n_pts,coord[x,y,z]), outputs 4x4 transform matrix
    fe_to_world_mat = vm.fit_coord_matrix(fe_coords)
    world_to_fe_mat = np.linalg.inv(fe_to_world_mat)
    print("FE to world matrix:")
    print(fe_to_world_mat)
    print()
    print("World to sim matrix:")
    print(world_to_fe_mat)
    print()

    print("Adding w coord and rotating sim coords")
    fe_with_w = np.hstack([fe_coords,
                            np.ones([fe_coords.shape[0],1])])
    print(f"{fe_with_w.shape=}")

    fe_coords = np.matmul(world_to_fe_mat,fe_with_w.T).T
    print(f"{fe_coords.shape=}")

    print("Returning sim coords by removing w coord:")
    fe_coords = fe_coords[:,:-1]
    print(f"{fe_coords.shape=}")
    del fe_with_w
    print()

    
    #---------------------------------------------------------------------------
    # EXP-SIM Comparison of coords
    PLOT_COORD_COMP = False

    if PLOT_COORD_COMP:
        down_samp: int = 1
 
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.scatter(fe_coords[:,0],
                    fe_coords[:,1],
                    fe_coords[:,2],
                    label="FE")
        id_key = "med"
        ax.scatter(id_coords[id_key][::down_samp,0],
                   id_coords[id_key][::down_samp,1],
                   id_coords[id_key][::down_samp,2],
                   label="ID")

        #ax.set_zlim(-1.0,1.0)
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.scatter(fe_coords[:,0],fe_coords[:,1],
                   label="FE")
        ax.scatter(id_coords[id_key][::down_samp,0],
                   id_coords[id_key][::down_samp,1],
                   label="ID")
        ax.legend()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.show()

    #---------------------------------------------------------------------------
    # Interp to common grid
    
    fe_x_min = np.min(fe_coords[:,0])
    fe_x_max = np.max(fe_coords[:,0])
    fe_y_min = np.min(fe_coords[:,1])
    fe_y_max = np.max(fe_coords[:,1])

    tol = 1e-6
    step = 0.5
    x_vec = np.arange(fe_x_min,fe_x_max+tol,step)
    y_vec = np.arange(fe_y_min,fe_y_max+tol,step)
    (x_grid,y_grid) = np.meshgrid(x_vec,y_vec)
    grid_shape = x_grid.shape
    grid_num_pts = x_grid.size

    print("FE COORD LIMITS:")
    print(f"{fe_x_min=}, {fe_x_max=}")
    print(f"{fe_y_min=}, {fe_y_max=}")
    print(f"{grid_shape=}, {grid_num_pts=}")
    print() 

    shift_x = 0.0
    shift_y = 0.0
    fe_strain_grid = {}
    id_strain_grid = {}
    for kk in fe_strain:
        fe_strain_temp = np.zeros(grid_shape+(3,),dtype=np.float64)
        id_strain_temp = np.zeros(grid_shape+(3,),dtype=np.float64)
        
        for cc in range(0,3): 
            fe_strain_temp[:,:,cc] = griddata(fe_coords[:,0:2],
                                          fe_strain[kk][:,cc],
                                          (x_grid,y_grid),
                                          method="linear")
                                          
            id_strain_temp[:,:,cc] = griddata(id_coords[kk][:,0:2],
                                              id_strain[kk][:,cc],
                                              (x_grid+shift_x,y_grid+shift_y),
                                              method="linear")
        fe_strain_grid[kk] = fe_strain_temp
        id_strain_grid[kk] = id_strain_temp
        print(f"{kk=}, {fe_strain_grid[kk].shape=}, "
              +f"{id_strain_grid[kk].shape=}")

        
    #---------------------------------------------------------------------------
    # Figure comparing fields
    cbar_field_same = False
    cbar_field_sym = False
    cbar_diff_sym = True
    
    extent = (fe_x_min,fe_x_max,fe_y_min,fe_y_max)
    cbar_font_size = 6.0
    plot_opts = pyvale.sensorsim.PlotOptsGeneral()
    fig_size = (plot_opts.a4_print_width,
                plot_opts.a4_print_width/(plot_opts.aspect_ratio*2.8))

    tag_dict = {"min":"Min.","max":"Max.","med":"Med."}

    # err = ID - FE, FE = ID - err => TRUTH = EXP - err
    # err = FE - ID, FE = ID + err => TRUTH = EXP + err
    for kk in fe_strain_grid:
        fe_field_plot = fe_strain_grid[kk]
        id_field_plot = id_strain_grid[kk]
                
        for ax_ind,ax_str in enumerate(STRAIN_COMP_STRS):
            field_str = FIELD_AX_STRS[ax_ind]

            field_diff_plot = (fe_field_plot[:,:,ax_ind] 
                              - id_field_plot[:,:,ax_ind])

            field_color_max = np.nanmax((np.nanmax(fe_field_plot[:,:,ax_ind]),
                                         np.nanmax(id_field_plot[:,:,ax_ind])))
            field_color_min = np.nanmin((np.nanmin(fe_field_plot[:,:,ax_ind]),
                                         np.nanmin(id_field_plot[:,:,ax_ind])))
            if cbar_field_sym:
                field_color_lim = np.max(
                    (np.abs(field_color_max),np.abs(field_color_min))
                )
                field_color_max = field_color_lim
                field_color_min = -field_color_lim

            if cbar_diff_sym:
                diff_color_lim = np.nanmax(
                    (np.abs(np.nanmax(field_diff_plot[:])),
                     np.abs(np.nanmin(field_diff_plot[:])))
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
                image = ax[0].imshow(id_field_plot[:,:,ax_ind],
                                     extent=extent,
                                     vmin=field_color_min,
                                     vmax=field_color_max)
            else: 
                image = ax[0].imshow(id_field_plot[:,:,ax_ind],
                                     extent=extent)

            ax[0].set_title(
                f"ID Avg. \n{field_str} [{FIELD_UNIT_STR}]",
                fontsize=plot_opts.font_head_size, 
                fontname=plot_opts.font_name
            )
            cbar = plt.colorbar(image)
            
            #-------------------------------------------------------------------
            # Fig 1: SIM FIELD
            if cbar_field_same:
                image = ax[1].imshow(fe_field_plot[:,:,ax_ind],
                                     extent=extent,
                                     vmin = field_color_min,
                                     vmax = field_color_max)
            else: 
                image = ax[1].imshow(fe_field_plot[:,:,ax_ind],
                                     extent=extent)

            ax[1].set_title(
                f"FE Avg.\n{field_str} [{FIELD_UNIT_STR}], {tag_dict[kk]}",
                fontsize=plot_opts.font_head_size, 
                fontname=plot_opts.font_name
            )
            cbar = plt.colorbar(image)
            

            #-------------------------------------------------------------------
            # Fig 2: DIFF
            if cbar_diff_sym:
                image = ax[2].imshow(field_diff_plot,
                                     extent=extent,
                                     vmin = diff_color_min,
                                     vmax = diff_color_max,
                                     cmap="RdBu")
            else: 
                image = ax[2].imshow(field_diff_plot,
                                     extent=extent,
                                     cmap="RdBu")    

            ax[2].set_title(
                f"(FE - ID)\n{field_str} [{FIELD_UNIT_STR}], {tag_dict[kk]}",
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

            save_name = f"case_{kk}_id_vs_fe_{SIM_TAG}_strain_{ax_str}.png"
            save_fig_path = save_path / save_name

            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

    #---------------------------------------------------------------------------
    # Save Systematic Error Maps to Disk 
    # err = ID - FE, FE = ID - err => TRUTH = EXP - err
    # **err = FE - ID, FE = ID + err => TRUTH = EXP + err**
    print(80*"-")
    print("ERROR FIELD CALCULATION")
    print("ERR = FE - ID => FE = ID + ERR")
    print("TRUTH = EXP + ERR")
    
    err_field_shape = (2,) + grid_shape + (3,)
    err_field = np.zeros(err_field_shape,dtype=np.float64)
    kk = "max"
    err_field[1,:,:,:] = fe_strain_grid[kk] - id_strain_grid[kk] # MAX
    kk = "min"
    err_field[0,:,:,:] = fe_strain_grid[kk] - id_strain_grid[kk] # MIN

    print("shape=([min,max],grid_y,grid_x,strain[xx,yy,xy])")
    print(f"{err_field.shape=}")
    print()
    print("Saving FE-ID error field to file.")
    print()
    
    save_name = f"strain_err_field_fe_take_id_{SIM_TAG}.npy"
    np.save(FE_DIR / save_name,err_field)     
        
    #---------------------------------------------------------------------------
    print(80*"-")
    print("COMPLETE.")
    plt.show()
        
 
if __name__ == "__main__":
    main()
