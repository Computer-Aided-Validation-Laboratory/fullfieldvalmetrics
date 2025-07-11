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

    comps = (0,1,2)
    (xx,yy,zz) = (0,1,2)

    ax_strs = ("x","y","z")

    plot_opts = pyvale.PlotOptsGeneral()
    exp_c: str = "tab:orange"
    sim_c: str = "tab:blue"
    mavm_c: str = "tab:green"

    SIM_TAG = "red"
    EXP_TAG = "All"
    STRAIN_COMP_STRS = ("xx","yy","xy")

    temp_path = Path.cwd() / f"temp_{SIM_TAG}"
    if not temp_path.is_dir():
        raise Exception("No temporary common processing files - run individual scripts first")

    save_path = Path.cwd() / "images_dicall_pulse25X"
    if not save_path.is_dir():
        save_path.mkdir(exist_ok=True,parents=True)

    #---------------------------------------------------------------------------
    # SIM: Load common grid data for the simulation
    print(80*"-")
    print("SIM: loading sim data on common grid")
    sim_strain_common_path = temp_path / f"sim_strain_common_{SIM_TAG}.npy"
    sim_strain_common = np.load(sim_strain_common_path)

    #---------------------------------------------------------------------------
    # EXP: Load common grid data for all 3 experiments
    print(80*"-")
    print("EXP: loading exp data on common grid")

    exp_data = []
    for ee in range(3):
        exp_common_path = temp_path / f"exp{ee}_strain_common.npy"
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

    coord_common_file = temp_path / "coord_common_for_strain.npy"
    coords_common = np.load(coord_common_file)
    print(f"{coords_common.shape=}")
    print()

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

    PLOT_COMMON_PT_CDFS = True

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
            ax_str = f"sim strain e_{STRAIN_COMP_STRS[cc]} [-]"
            axs.set_title(title_str,fontsize=plot_opts.font_head_size)
            axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
            axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
            #axs.legend(loc="upper left",fontsize=6)

            save_fig_path = (save_path/f"sim_strainncom_{STRAIN_COMP_STRS[cc]}_ptcdfsall_{SIM_TAG}.png")
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
            ax_str = f"strain e_{STRAIN_COMP_STRS[cc]} [-]"
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
    for cc,aa in enumerate(ax_strs):

        this_mavm = {}
        this_mavm_lim = {}

        pp = mavm_inds[cc]

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
        ax_str = f"strain e_{STRAIN_COMP_STRS[cc]} [-]"
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
        ax_str = f"strain {STRAIN_COMP_STRS[cc]} [-]"
        axs.set_title(title_str,fontsize=plot_opts.font_head_size)
        axs.set_xlabel(ax_str,fontsize=plot_opts.font_ax_size)
        axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
        axs.legend(loc="upper left",fontsize=6)

        save_fig_path = (save_path
            / f"exp{EXP_TAG}_straincom_{STRAIN_COMP_STRS[cc]}_mavmlims_{SIM_TAG}.png")
        fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

    #---------------------------------------------------------------------------
    print(80*"-")
    print("COMPLETE.")
    #plt.show()










if __name__ == "__main__":
    main()