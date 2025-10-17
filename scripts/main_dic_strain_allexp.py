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
    SIM_TAG = "redv2"

    FE_DIR = Path.cwd()/ "STC_ProbSim_FieldsReduced_25X"
    conv_to_mm: float = 1000.0 # Simulation is in SI and exp is in mm

    # Reduced: 5000 = 100 aleatory x 50 epistemic
    # Full: 400 aleatory x 250 epistemic
    # exp_data = exp_data.reshape(samps_n,epis_n,alea_n)
    #samps_n: int = 5000
    SIM_EPIS_N: int = 50
    SIM_ALEA_N: int = 100


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

    # OLD CODE:
    # #---------------------------------------------------------------------------
    # # SIM-EXP: Calculate mavm at a few key points
    # print(80*"-")
    # print("SIM-EXP: finding key points in fields and plotting cdfs")

    # find_point_x = np.array([24.0,-16.0]) # mm
    # find_point_yz = np.array([0.0,-16.0])  # mm

    # mavm_inds = np.zeros((3,),dtype=np.uintp)
    # mavm_inds[xx] = vm.find_nearest_points(coords_common,find_point_x,k=3)[0]
    # mavm_inds[yy] = vm.find_nearest_points(coords_common,find_point_yz,k=3)[0]
    # mavm_inds[zz] = mavm_inds[yy]

    # print(80*"-")
    # print(f"{mavm_inds=}")
    # print()
    # print(f"{coords_common[mavm_inds[xx],:]=}")
    # print(f"{coords_common[mavm_inds[yy],:]=}")
    # print(f"{coords_common[mavm_inds[zz],:]=}")
    # print(80*"-")
    # print()

    # print("Summing along aleatory axis and finding max/min...")
    # sim_limits = np.sum(sim_strain_common,axis=1)
    # sim_cdf_eind = {}
    # sim_cdf_eind['max'] = np.argmax(sim_limits,axis=0)
    # sim_cdf_eind['min'] = np.argmin(sim_limits,axis=0)

    # print(f"{sim_strain_common.shape=}")
    # print(f"{sim_limits.shape=}")
    # print(f"{sim_cdf_eind['max'].shape=}")
    # print(f"{sim_cdf_eind['min'].shape=}")
    # print()
    # print(f"{sim_cdf_eind['max'][0,0]=}")
    # print(f"{sim_cdf_eind['min'][0,0]=}")
    # print()

    # PLOT_COMMON_PT_CDFS = True

    # if PLOT_COMMON_PT_CDFS:
    #     print("Plotting all sim cdfs and limit cdfs for key points on common coords...")
    #     for cc in comps:
    #         pp = mavm_inds[cc]
    #         fig, axs=plt.subplots(1,1,
    #                             figsize=plot_opts.single_fig_size_landscape,
    #                             layout="constrained")
    #         fig.set_dpi(plot_opts.resolution)

    #         for ee in range(sim_strain_common.shape[0]):
    #             axs.ecdf(sim_strain_common[ee,:,pp,cc]
    #                     ,color='tab:blue',linewidth=plot_opts.lw)

    #         e_ind = sim_cdf_eind['max'][pp,cc]
    #         axs.ecdf(sim_strain_common[e_ind,:,pp,cc]
    #                 ,ls="--",color='black',linewidth=plot_opts.lw)

    #         min_e = sim_cdf_eind['min'][pp,cc]
    #         axs.ecdf(sim_strain_common[min_e,:,pp,cc]
    #                 ,ls="--",color='black',linewidth=plot_opts.lw)

    #         this_coord = coords_common[mavm_inds[cc],:]
    #         title_str = f"(x,y)=({this_coord[0]:.2f},{-1*this_coord[1]:.2f})"
    #         ax_str = f"sim strain e_{STRAIN_COMP_STRS[cc]} [-]"
    #         axs.set_title(title_str,fontsize=plot_opts.font_head_size)
    #         axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
    #         axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
    #         #axs.legend(loc="upper left",fontsize=6)

    #         save_fig_path = (save_path/f"sim_strainncom_{STRAIN_COMP_STRS[cc]}_ptcdfsall_{SIM_TAG}.png")
    #         fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")


    #     print("Plotting all sim-exp comparison cdfs for key points on common coords...")
    #     for cc in comps:
    #         pp = mavm_inds[cc]
    #         fig, axs=plt.subplots(1,1,
    #                             figsize=plot_opts.single_fig_size_landscape,
    #                             layout="constrained")
    #         fig.set_dpi(plot_opts.resolution)

    #         # SIM CDFS
    #         max_e = sim_cdf_eind['max'][pp,cc]
    #         axs.ecdf(sim_strain_common[max_e,:,pp,cc]
    #                 ,ls="--",color=sim_c,linewidth=plot_opts.lw,
    #                 label="sim.")

    #         min_e = sim_cdf_eind['min'][pp,cc]
    #         axs.ecdf(sim_strain_common[min_e,:,pp,cc]
    #                 ,ls="--",color=sim_c,linewidth=plot_opts.lw)

    #         sim_cdf_high = stats.ecdf(sim_strain_common[max_e,:,pp,cc]).cdf
    #         sim_cdf_low = stats.ecdf(sim_strain_common[min_e,:,pp,cc]).cdf
    #         axs.fill_betweenx(sim_cdf_high.probabilities,
    #                         sim_cdf_low .quantiles,
    #                         sim_cdf_high.quantiles,
    #                         color=sim_c,
    #                         alpha=0.2)

    #         # EXP CDF
    #         axs.ecdf(exp_strain_common[:,pp,cc]
    #                 ,ls="-",color=exp_c,linewidth=plot_opts.lw,
    #                 label="exp.")

    #         this_coord = coords_common[mavm_inds[cc],:]
    #         title_str = f"(x,y)=({this_coord[0]:.2f},{-1*this_coord[1]:.2f})"
    #         ax_str = f"strain e_{STRAIN_COMP_STRS[cc]} [-]"
    #         axs.set_title(title_str,fontsize=plot_opts.font_head_size)
    #         axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
    #         axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
    #         axs.legend(loc="upper left",fontsize=6)

    #         save_fig_path = (save_path
    #                     /f"exp{EXP_TAG}_straincom_{STRAIN_COMP_STRS[cc]}_ptcdfs_{SIM_TAG}.png")
    #         fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

    # # plt.close("all")


    # #---------------------------------------------------------------------------
    # # Calculate MAVM at key points

    # print(80*"-")
    # print("SIM-EXP: Calculating MAVM at key points")

    # sim_lim_keys = ("min","max")
    # mavm = {}
    # mavm_lims = {}
    # for cc,aa in enumerate(ax_strs):

    #     this_mavm = {}
    #     this_mavm_lim = {}

    #     pp = mavm_inds[cc]

    #     dplus_cdf_sum = None
    #     dminus_cdf_sum = None

    #     for kk in sim_lim_keys:
    #         e_ind = sim_cdf_eind[kk][pp,cc]
    #         this_mavm[kk] = vm.mavm(sim_strain_common[e_ind,:,pp,cc],
    #                                 exp_strain_common[:,pp,cc])

    #         check_upper = np.sum(this_mavm[kk]["F_"] + this_mavm[kk]["d+"])
    #         check_lower = np.sum(this_mavm[kk]["F_"] - this_mavm[kk]["d-"])

    #         if dplus_cdf_sum is None:
    #             dplus_cdf_sum = check_upper
    #             this_mavm_lim["max"] = this_mavm[kk]
    #         else:
    #             if check_upper > dplus_cdf_sum:
    #                 dplus_cdf_sum = check_upper
    #                 this_mavm_lim["max"] = this_mavm[kk]

    #         if dminus_cdf_sum is None:
    #             dminus_cdf_sum = check_lower
    #             this_mavm_lim["min"] = this_mavm[kk]
    #         else:
    #             if check_lower < dminus_cdf_sum:
    #                 dminus_cdf_sum = dminus_cdf_sum
    #                 this_mavm_lim["min"] = this_mavm[kk]

    #     mavm_lims[aa] = this_mavm_lim
    #     mavm[aa] = this_mavm


    # #print(f"{mavm['x']['max']=}")
    # # print()
    # # print(mavm_lims.keys())
    # # print(mavm_lims["x"].keys())
    # # print(mavm_lims["x"]["max"].keys())
    # plt.close("all")

    # print("Plotting MAVM at key points")
    # for cc,aa in enumerate(ax_strs):
    #     pp = mavm_inds[cc]

    #     fig,axs=plt.subplots(1,1,
    #                 figsize=plot_opts.single_fig_size_landscape,
    #                 layout="constrained")
    #     fig.set_dpi(plot_opts.resolution)

    #     # SIM CDFS
    #     max_e = sim_cdf_eind['max'][pp,cc]
    #     axs.ecdf(sim_strain_common[max_e,:,pp,cc]
    #             ,ls="--",color=sim_c,linewidth=plot_opts.lw,
    #             label="sim.")

    #     min_e = sim_cdf_eind['min'][pp,cc]
    #     axs.ecdf(sim_strain_common[min_e,:,pp,cc]
    #             ,ls="--",color=sim_c,linewidth=plot_opts.lw)

    #     sim_cdf_high = stats.ecdf(sim_strain_common[max_e,:,pp,cc]).cdf
    #     sim_cdf_low = stats.ecdf(sim_strain_common[min_e,:,pp,cc]).cdf
    #     axs.fill_betweenx(sim_cdf_high.probabilities,
    #                     sim_cdf_low .quantiles,
    #                     sim_cdf_high.quantiles,
    #                     color=sim_c,
    #                     alpha=0.2)

    #     # EXP CDF
    #     axs.ecdf(exp_strain_common[:,pp,cc]
    #             ,ls="-",color=exp_c,linewidth=plot_opts.lw,
    #             label="exp.")

    #     mavm_c = "tab:red"
    #     axs.plot(mavm[aa]["min"]["F_"] - mavm[aa]["min"]["d-"],
    #              mavm[aa]["min"]["F_Y"], label="min, d-",
    #              ls="--",color=mavm_c,linewidth=plot_opts.lw*1.2)
    #     axs.plot(mavm[aa]["min"]["F_"] + mavm[aa]["min"]["d+"],
    #              mavm[aa]["min"]["F_Y"], label="min, d+",
    #              ls="-",color=mavm_c,linewidth=plot_opts.lw*1.2)

    #     mavm_c = "tab:green"
    #     axs.plot(mavm[aa]["max"]["F_"] - mavm[aa]["max"]["d-"],
    #              mavm[aa]["max"]["F_Y"], label="max, d-",
    #              ls="--",color= mavm_c,linewidth=plot_opts.lw*1.2)

    #     axs.plot(mavm[aa]["max"]["F_"] + mavm[aa]["max"]["d+"],
    #              mavm[aa]["max"]["F_Y"], label="max, d+",
    #              ls="-",color= mavm_c,linewidth=plot_opts.lw*1.2)

    #     print()
    #     print(80*"=")
    #     print(f"{aa=}")
    #     print(f"{mavm[aa]['min']['d-']=}")
    #     print(f"{mavm[aa]['min']['d+']=}")
    #     print(f"{mavm[aa]['max']['d-']=}")
    #     print(f"{mavm[aa]['max']['d+']=}")
    #     print(80*"=")
    #     print()

    #     this_coord = coords_common[mavm_inds[cc],:]
    #     title_str = f"(x,y)=({this_coord[0]:.2f},{-1*this_coord[1]:.2f})"
    #     ax_str = f"strain e_{STRAIN_COMP_STRS[cc]} [-]"
    #     axs.set_title(title_str,fontsize=plot_opts.font_head_size)
    #     axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
    #     axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
    #     axs.legend(loc="upper left",fontsize=6)

    #     save_fig_path = (save_path
    #         /f"exp{EXP_TAG}_straincom_{STRAIN_COMP_STRS[cc]}_allmavm_{SIM_TAG}.png")
    #     fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")


    # print("Plotting mavm limits...")
    # for cc,aa in enumerate(ax_strs):
    #     pp = mavm_inds[cc]

    #     fig,axs=plt.subplots(1,1,
    #                 figsize=plot_opts.single_fig_size_landscape,
    #                 layout="constrained")
    #     fig.set_dpi(plot_opts.resolution)

    #     # SIM CDFS
    #     max_e = sim_cdf_eind['max'][pp,cc]
    #     axs.ecdf(sim_strain_common[max_e,:,pp,cc]
    #             ,ls="--",color=sim_c,linewidth=plot_opts.lw,
    #             label="sim.")

    #     min_e = sim_cdf_eind['min'][pp,cc]
    #     axs.ecdf(sim_strain_common[min_e,:,pp,cc]
    #             ,ls="--",color=sim_c,linewidth=plot_opts.lw)

    #     sim_cdf_high = stats.ecdf(sim_strain_common[max_e,:,pp,cc]).cdf
    #     sim_cdf_low = stats.ecdf(sim_strain_common[min_e,:,pp,cc]).cdf
    #     axs.fill_betweenx(sim_cdf_high.probabilities,
    #                     sim_cdf_low .quantiles,
    #                     sim_cdf_high.quantiles,
    #                     color=sim_c,
    #                     alpha=0.2)

    #     # MAVM
    #     mavm_c = "black"
    #     axs.plot(mavm_lims[aa]["min"]["F_"] - mavm_lims[aa]["min"]["d-"],
    #              mavm_lims[aa]["min"]["F_Y"], label="d-",
    #              ls="--",color=mavm_c,linewidth=plot_opts.lw*1.2)
    #     axs.plot(mavm_lims[aa]["max"]["F_"] + mavm_lims[aa]["max"]["d+"],
    #              mavm_lims[aa]["max"]["F_Y"], label="d+",
    #              ls="-",color=mavm_c,linewidth=plot_opts.lw*1.2)

    #     axs.fill_betweenx(mavm_lims[aa]["max"]["F_Y"],
    #                       mavm_lims[aa]["min"]["F_"] - mavm_lims[aa]["min"]["d-"],
    #                       mavm_lims[aa]["max"]["F_"] + mavm_lims[aa]["max"]["d+"],
    #                       color=mavm_c,
    #                       alpha=0.2)

    #     this_coord = coords_common[mavm_inds[cc],:]
    #     title_str = f"(x,y)=({this_coord[0]:.2f},{-1*this_coord[1]:.2f})"
    #     ax_str = f"strain {STRAIN_COMP_STRS[cc]} [-]"
    #     axs.set_title(title_str,fontsize=plot_opts.font_head_size)
    #     axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
    #     axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
    #     axs.legend(loc="upper left",fontsize=6)

    #     save_fig_path = (save_path
    #         / f"exp{EXP_TAG}_straincom_{STRAIN_COMP_STRS[cc]}_mavmlims_{SIM_TAG}.png")
    #     fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

    #===========================================================================
    # NEW FIGS START HERE
    #===========================================================================
    grid_num_pts = coords_common.shape[0]

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

        exp_disp_grid_avg = griddata(exp_coords_avg[:,0:2],
                                exp_strain_avg[:,ax_ind],
                                (x_grid,y_grid),
                                method="linear")

        # This will do minimal interpolation as the input points are the same as the sim
        sim_disp_grid_avg = griddata(sim_coords[:,0:2],
                                sim_strain_avg[:,ax_ind],
                                (x_grid,y_grid),
                                method="linear")

        disp_diff_avg = sim_disp_grid_avg - exp_disp_grid_avg

        color_max = np.nanmax((np.nanmax(sim_disp_grid_avg),np.nanmax(exp_disp_grid_avg)))
        color_min = np.nanmin((np.nanmin(sim_disp_grid_avg),np.nanmin(exp_disp_grid_avg)))

        cbar_font_size = 6.0

        plot_opts = pyvale.sensorsim.PlotOptsGeneral()
        fig_size = (plot_opts.a4_print_width,plot_opts.a4_print_width/(plot_opts.aspect_ratio*2.8))
        fig,ax = plt.subplots(1,4,figsize=fig_size,layout='constrained')
        fig.set_dpi(plot_opts.resolution)

        if scale_cbar:
            image = ax[0].imshow(exp_disp_grid_avg,
                                extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max),
                                vmin = color_min,
                                vmax = color_max)
        else:
            image = ax[0].imshow(exp_disp_grid_avg,
                                extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))

        ax[0].set_title(f"Exp. Avg. \n{field_str} [{FIELD_UNIT_STR}]",
                        fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
        cbar = plt.colorbar(image)


        if scale_cbar:
            image = ax[1].imshow(sim_disp_grid_avg,
                                extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max),
                                vmin = color_min,
                                vmax = color_max)
        else:
            image = ax[1].imshow(sim_disp_grid_avg,
                                extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))

        ax[1].set_title(f"Sim. Avg.\n{field_str} [{FIELD_UNIT_STR}]",
                        fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
        cbar = plt.colorbar(image)


        image = ax[2].imshow(disp_diff_avg,
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

    #---------------------------------------------------------------------------
    print(80*"-")
    print("COMPLETE.")
    #plt.show()










if __name__ == "__main__":
    main()