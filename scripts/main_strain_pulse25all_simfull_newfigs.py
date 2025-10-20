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
    print("Aggregated experimental data on common grid")
    print()

    #---------------------------------------------------------------------------
    comps = (0,1,2)
    (xx,yy,zz) = (0,1,2)
    xy = 2

    ax_strs = ("x","y","z")

    plot_opts = pyvale.sensorsim.PlotOptsGeneral()
    exp_c: str = "tab:orange"
    sim_c: str = "tab:blue"
    mavm_c: str = "tab:green"

    STRAIN_COMP_STRS = ("xx","yy","xy")

    FIELD_UNIT_CONV = 1e3
    FIELD_UNIT_STR = r"$m\epsilon$"
    FIELD_AX_STRS = (r"$e_{xx}$",r"$e_{yy}$",r"$e_{xy}$")

    SIM_TAG = "redv2"
    EXP_TAG = "all"

    DIC_PULSES = ("253","254","255")
    DIC_DIRS = (
        Path.cwd() / "STC_Exp_DIC_253",
        Path.cwd() / "STC_Exp_DIC_254",
        Path.cwd() / "STC_Exp_DIC_255",
    )

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
    temp_path = Path.cwd() / f"temp_exp{EXP_TAG}_sim{SIM_TAG}"
    if not temp_path.is_dir():
        temp_path.mkdir(exist_ok=True,parents=True)

    save_path = Path.cwd() / f"images_dic_pulse25X_exp{EXP_TAG}_sim{SIM_TAG}_strainv2"
    if not save_path.is_dir():
        save_path.mkdir(exist_ok=True,parents=True)

    #---------------------------------------------------------------------------
    # SIM: Load common grid data for the simulation
    print(80*"-")
    print("SIM: loading sim data on common grid")
    this_sim_path = Path.cwd() / f"temp_exp{DIC_PULSES[0]}_sim{SIM_TAG}"
    sim_strain_common_path = this_sim_path / f"sim_strain_common_{SIM_TAG}.npy"
    sim_strain_common = np.load(sim_strain_common_path)
    print(f"{sim_strain_common.shape=}")

    #---------------------------------------------------------------------------
    # EXP: Load common grid data for all 3 experiments
    print(80*"-")
    print("EXP: loading exp data on common grid")

    exp_data = []
    for ee in range(3):
        this_exp_path = Path.cwd() / f"temp_exp{DIC_PULSES[ee]}_sim{SIM_TAG}"
        exp_common_path = this_exp_path / f"exp{ee}_strain_common.npy"
        exp_data.append(np.load(exp_common_path))

        print(f"{exp_common_path=}")
        print(f"{exp_data[ee].shape=}")
        print()


    exp_strain_common = np.concatenate(exp_data,axis=0)
    print("Concatenating experimental data into a single array:")
    print(f"{exp_strain_common.shape=}")

    del exp_data

    #---------------------------------------------------------------------------
    # SIM-EXP: Load common coords
    print(80*"-")
    print("SIM-EXP: loading common coords")

    coord_common_file = this_sim_path / "coord_common_for_strain.npy"
    coords_common = np.load(coord_common_file)
    print(f"{coords_common.shape=}")
    print()

    #---------------------------------------------------------------------------
    # Average fields from experiment and simulation to plot the difference
    print(80*"-")
    print("Averaging experiment steady state and simulation for full-field comparison.")

    # Had to change these to nanmean because of problems in experimental data
    exp_strain_avg = np.nanmean(exp_strain_common,axis=0)
    # Average twice, once over epistemic uncertainty and once over aleatory
    sim_strain_avg = np.nanmean(sim_strain_common,axis=0)
    sim_strain_avg = np.nanmean(sim_strain_avg,axis=0)

    print()
    print(f"{coords_common.shape=}")
    print(f"{exp_strain_avg.shape=}")
    print(f"{sim_strain_avg.shape=}")
    print()

    x_min = np.min(coords_common[:,0])
    x_max = np.max(coords_common[:,0])
    y_min = np.min(coords_common[:,1])
    y_max = np.max(coords_common[:,1])

    tol = 1e-6
    step = 0.5 # NOTE: watch this
    x_vec = np.arange(x_min,x_max+tol,step)
    y_vec = np.arange(y_min,y_max+tol,step)
    (x_grid,y_grid) = np.meshgrid(x_vec,y_vec)
    grid_shape = x_grid.shape
    grid_num_pts = x_grid.size

    print(f"{x_grid.shape=}")
    print(f"{y_grid.shape=}")
    print(f"{grid_num_pts=}")
    print()

    exp_strain_shape = x_grid.shape + (exp_strain_avg.shape[-1],)
    sim_strain_shape = x_grid.shape + (sim_strain_avg.shape[-1],)

    print(f"{exp_strain_shape}")
    print(f"{sim_strain_shape}")

    exp_strain_grid_avg = np.reshape(exp_strain_avg,exp_strain_shape)
    sim_strain_grid_avg = np.reshape(sim_strain_avg,sim_strain_shape)
    strain_diff_avg = sim_strain_grid_avg - exp_strain_grid_avg

    #---------------------------------------------------------------------------
    # SIM: Find limiting epistemic CDFs
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

    #---------------------------------------------------------------------------
    # SIM-EXP: Calculate and/or load MAVM
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

    #---------------------------------------------------------------------------
    # Paper Figures
    scale_cbar = True

    for ax_ind,ax_str in enumerate(STRAIN_COMP_STRS):
        field_str = FIELD_AX_STRS[ax_ind]

        color_max = np.nanmax((np.nanmax(sim_strain_grid_avg[:,:,ax_ind]),
                               np.nanmax(exp_strain_grid_avg[:,:,ax_ind])))
        color_min = np.nanmin((np.nanmin(sim_strain_grid_avg[:,:,ax_ind]),
                               np.nanmin(exp_strain_grid_avg[:,:,ax_ind])))

        plot_opts = pyvale.sensorsim.PlotOptsGeneral()
        fig_size = (plot_opts.a4_print_width,plot_opts.a4_print_width/(plot_opts.aspect_ratio*2.8))
        fig,ax = plt.subplots(1,4,figsize=fig_size,layout='constrained')
        fig.set_dpi(plot_opts.resolution)

        if scale_cbar:
            image = ax[0].imshow(exp_strain_grid_avg[:,:,ax_ind],
                                extent=(x_min,x_max,y_min,y_max),
                                vmin = color_min,
                                vmax = color_max)
        else:
            image = ax[0].imshow(exp_strain_grid_avg[:,:,ax_ind],
                                extent=(x_min,x_max,y_min,y_max))

        ax[0].set_title(f"Exp. Avg. \n{field_str} [{FIELD_UNIT_STR}]",
                        fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
        cbar = plt.colorbar(image)


        if scale_cbar:
            image = ax[1].imshow(sim_strain_grid_avg[:,:,ax_ind],
                                extent=(x_min,x_max,y_min,y_max),
                                vmin = color_min,
                                vmax = color_max)
        else:
            image = ax[1].imshow(sim_strain_grid_avg[:,:,ax_ind],
                                extent=(x_min,x_max,y_min,y_max))

        ax[1].set_title(f"Sim. Avg.\n{field_str} [{FIELD_UNIT_STR}]",
                        fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
        cbar = plt.colorbar(image)


        image = ax[2].imshow(strain_diff_avg[:,:,ax_ind],
                            extent=(x_min,x_max,y_min,y_max),
                            cmap="RdBu")
        ax[2].set_title(f"(Sim. - Exp.)\n{field_str} [{FIELD_UNIT_STR}]",
                        fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
        cbar = plt.colorbar(image)

        mavm_map = np.reshape(mavm_d_max[:,ax_ind],grid_shape)
        image = ax[3].imshow(mavm_map,
            extent=(x_min,x_max,y_min,y_max),
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


    #---------------------------------------------------------------------------
    print(80*"-")
    print("COMPLETE.")
    #plt.show()

if __name__ == "__main__":
    main()