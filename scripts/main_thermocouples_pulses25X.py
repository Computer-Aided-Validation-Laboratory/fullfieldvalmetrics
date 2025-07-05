from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import pyvale
import valmetrics as vm

def main() -> None:
    print(80*"=")
    print("PULSE 25X: Thermocouple MAVM analysis")
    print(80*"=")

    EXP_DIR = Path.cwd() / "STC_Exp_TCs"
    SIM_DIR = Path.cwd() / "STC_ProbSim_Full"

    sens_ax_labels = (r"Temp. [$^{\circ}C$]",)*10 + (r"Coil RMS Voltage [$V$]",)
    sens_tags = ("Temp",)*10 + ("Volts",)

    save_path = Path.cwd() / "images_pulse25X"
    if not save_path.is_dir():
        save_path.mkdir(exist_ok=True,parents=True)

    plot_opts = pyvale.PlotOptsGeneral()

    #---------------------------------------------------------------------------
    # Simulation: Load Data
    print(f"Loading simulation data from:\n    {SIM_DIR}")

    # Reduced: 5000 = 100 aleatory x 50 epistemic
    # Full: 400 aleatory x 250 epistemic
    # exp_data = exp_data.reshape(samps_n,epis_n,alea_n)
    #samps_n: int = 5000
    epis_n: int = 250#50
    alea_n: int = 400#100

    sim_keys = {"TC1":0,
                "TC2":1,
                "TC3":2,
                "TC4":3,
                "TC5":4,
                "TC6":5,
                "TC7":6,
                "TC8":7,
                "TC9":8,
                "TC10":9,
                "CV":15,}

    sim_path = SIM_DIR / "SamplingResults.csv"
    sim_data_arr = pd.read_csv(sim_path).to_numpy()

    sim_data = {}
    for kk in sim_keys:
        sim_data[kk] = sim_data_arr[:,sim_keys[kk]]

        sim_data[kk] = sim_data[kk].reshape(epis_n,alea_n)

        # print(kk)
        # print(f"{sim_data[kk].shape=}")
        # print()

    #---------------------------------------------------------------------------
    # Experiment: Load Data
    print(f"Loading experimental data from:\n    {EXP_DIR}")

    exp_file_pre = ["Pulse253","Pulse254","Pulse255"]
    exp_file_post = ["SteadyDICData","SteadyHIVEData","SteadyPICOData"]
    exp_file_slices = [slice(2,11),slice(2,3),slice(1,2)]
    # NOTE: actually in file = 3,5,6,8,9,10
    exp_data_keys = [["TC1","TC3","TC4","TC5","TC6","TC7","TC8","TC9","TC10"],
                     ["TC2"],
                     ["CV"]]

    exp_data = {}
    for pp in exp_file_pre:
        for ii,ss in enumerate(exp_file_post):
            file = EXP_DIR / f"{pp}_{ss}.csv"

            data = pd.read_csv(file)
            data = data.to_numpy()
            data = data[:,exp_file_slices[ii]]

            for jj,kk in enumerate(exp_data_keys[ii]):
                exp_data[kk] = data[:,jj]

    print("Loading experimental epistemic error data.")
    exp_epis_file = Path.cwd() / "STC_ProbSim_Meta" / "STC2_1550A_SteadySummary.csv"
    exp_epis_err = pd.read_csv(exp_epis_file)
    exp_epis_err = exp_epis_err.to_numpy()
    # Extract the last row and columns where the 10 TCs and coil voltage are
    exp_epis_err = np.array(exp_epis_err[-1,-11:],dtype=np.float64)

    # Move the coil voltage from the start to end to match everything else
    temp = np.copy(exp_epis_err)
    exp_epis_err[-1] = temp[0]
    exp_epis_err[0:-1] = temp[1:]
    del temp

    #---------------------------------------------------------------------------
    # Simulation: Find Min/Max CDF over all eps
    print()
    print(80*"-")
    print("Extracting max and min cdf over all simulations.")

    # Extract all the simulation cdfs by looping over the epistemic sampling
    sim_cdfs_all = {}
    sim_data_lims = {}
    sim_cdfs_lims ={}

    for kk in sim_data:
        cdfs = []

        max_data = sim_data[kk][0,:]
        min_data = sim_data[kk][0,:]

        min_cdf = stats.ecdf(sim_data[kk][0,:]).cdf
        max_cdf = stats.ecdf(sim_data[kk][0,:]).cdf

        sum_min_cdf = np.sum(min_cdf.quantiles)
        sum_max_cdf = np.sum(max_cdf.quantiles)

        accum_cdf = np.zeros_like(max_cdf.quantiles)

        for ee in range(epis_n):
            this_cdf = stats.ecdf(sim_data[kk][ee,:]).cdf
            this_cdf_sum = np.sum(this_cdf.quantiles)

            accum_cdf = accum_cdf + this_cdf.quantiles

            if this_cdf_sum > sum_max_cdf:
                sum_max_cdf = this_cdf_sum
                max_cdf = this_cdf
                max_data = sim_data[kk][ee,:]

            if this_cdf_sum < sum_min_cdf:
                sum_min_cdf = this_cdf_sum
                min_cdf = this_cdf
                min_data = sim_data[kk][ee,:]

            cdfs.append(this_cdf)

        sim_cdfs_all[kk] = cdfs

        mean_data = accum_cdf / epis_n
        this_data = {}
        this_data["max"] = max_data
        this_data["min"] = min_data
        this_data["nom"] = mean_data
        sim_data_lims[kk] = this_data

        mean_cdf = stats.ecdf(mean_data).cdf
        this_cdf = {}
        this_cdf["max"] = max_cdf
        this_cdf["min"] = min_cdf
        this_cdf["nom"] = mean_cdf
        sim_cdfs_lims[kk] = this_cdf


    PLOT_ALL_SIM_CDFS = False

    if PLOT_ALL_SIM_CDFS:
        for ii,kk in enumerate(sim_cdfs_all):
            fig, axs=plt.subplots(1,1,
                                figsize=plot_opts.single_fig_size_landscape,
                                layout="constrained")
            fig.set_dpi(plot_opts.resolution)

            for cc in sim_cdfs_all[kk]:
                axs.ecdf(cc.quantiles,color='xkcd:azure')

            axs.ecdf(sim_cdfs_lims[kk]["max"].quantiles,ls="--",color="black",)
            axs.ecdf(sim_cdfs_lims[kk]["min"].quantiles,ls="--",color="black")
            axs.ecdf(sim_cdfs_lims[kk]["nom"].quantiles,ls="-",color="black")

            axs.set_title(kk,fontsize=plot_opts.font_head_size)
            axs.set_xlabel(sens_ax_labels[ii],fontsize=plot_opts.font_ax_size)
            axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)

            save_fig_path = save_path / f"sim_allcdfs_{kk}.png"
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")

    #---------------------------------------------------------------------------
    # Experiment: Calculate min/max cdf based on epistemic error

    exp_data_lims = {}
    exp_cdfs_lims = {}
    for ii,kk in enumerate(exp_data):
        this_exp = {}
        this_exp["nom"] = exp_data[kk]
        this_exp["min"] = exp_data[kk] - exp_epis_err[ii]
        this_exp["max"] = exp_data[kk] + exp_epis_err[ii]
        exp_data_lims[kk] = this_exp

        this_cdf = {}
        if np.any(np.isnan(exp_data_lims[kk]["nom"])):
            this_cdf["nom"] = None
            this_cdf["min"] = None
            this_cdf["max"] = None

        else:
            this_cdf["nom"] = stats.ecdf(exp_data[kk]).cdf
            this_cdf["min"] = stats.ecdf(exp_data[kk] - exp_epis_err[ii]).cdf
            this_cdf["max"] = stats.ecdf(exp_data[kk] + exp_epis_err[ii]).cdf
        exp_cdfs_lims[kk] = this_cdf

    #---------------------------------------------------------------------------
    # Calculate MAVM
    PLOT_MAVM_INDIVIDUAL = False

    mavm = {}
    mavm_keys = ("min","max")

    print(80*"-")
    print("Calculating MAVM...")
    for ii,kk in enumerate(exp_data_lims): # Loop over sensors: TCs + CV
        this_mavm = {}
        for es in mavm_keys: # Loop over EXP stats: min,max
            for ss in mavm_keys: # Loop over SIM stats: min,max

                # Some TCs missing from exp data, skip them
                if not np.any(np.isnan(exp_data_lims[kk][es])):

                    print(80*"-")
                    print(f"sens-{kk}_sim-{ss}_exp-{es}")
                    print(f"{exp_data_lims[kk][es].shape=}")
                    print(f"{sim_data_lims[kk][ss].shape=}")
                    print()
                    print(f"{np.mean(exp_data_lims[kk][es])=}")
                    print(f"{np.mean(sim_data_lims[kk][ss])=}")
                    print()

                    stat_key = f"sim-{ss}_exp-{es}"

                    this_mavm[stat_key] = vm.mavm(sim_data_lims[kk][ss],
                                                  exp_data_lims[kk][es])

                    print(f"{this_mavm[stat_key]['d+']=}")
                    print(f"{this_mavm[stat_key]['d-']=}")
                    print(80*"-")

                    if PLOT_MAVM_INDIVIDUAL:
                        vm.mavm_figs(this_mavm[stat_key],
                                    title_str=kk,
                                    field_label=sens_ax_labels[ii],
                                    field_tag=sens_tags[ii],
                                    save_tag=stat_key,
                                    save_path=save_path)

    mavm[kk] = this_mavm

    #---------------------------------------------------------------------------
    # MAVM: Combined figures with all errors


    print(80*"-")
    print("Plotting combined MAVM figures...")
    for ii,kk in enumerate(exp_data_lims): # Loop over sensors: TCs + CV

        if np.any(np.isnan(exp_data_lims[kk]["nom"])):
            continue

        fig,axs=plt.subplots(1,1,
                         figsize=plot_opts.single_fig_size_landscape,
                         layout="constrained")
        fig.set_dpi(plot_opts.resolution)

        exp_c = "r"
        sim_c = "b"

        axs.ecdf(sim_cdfs_lims[kk]["nom"].quantiles,
                 ls="-",color=sim_c,label="sim. nom.")
        axs.ecdf(sim_cdfs_lims[kk]["max"].quantiles,
                 ls="--",color=sim_c,label="sim. lims.")
        axs.ecdf(sim_cdfs_lims[kk]["min"].quantiles,
                 ls="--",color=sim_c)

        axs.ecdf(exp_cdfs_lims[kk]["nom"].quantiles,
                 ls="-",color=exp_c,label="exp. nom.")
        axs.ecdf(exp_cdfs_lims[kk]["max"].quantiles,
                 ls="--",color=exp_c,label="exp. lims.")
        axs.ecdf(exp_cdfs_lims[kk]["min"].quantiles,
                 ls="--",color=exp_c)

        axs.fill_betweenx(sim_cdfs_lims[kk]["nom"].probabilities,
                         sim_cdfs_lims[kk]["min"].quantiles,
                         sim_cdfs_lims[kk]["max"].quantiles,
                         color=sim_c,
                         alpha=0.2)

        axs.fill_betweenx(exp_cdfs_lims[kk]["nom"].probabilities,
                         exp_cdfs_lims[kk]["min"].quantiles,
                         exp_cdfs_lims[kk]["max"].quantiles,
                         color=exp_c,
                         alpha=0.2)

        axs.legend(loc="upper left")
        axs.set_title(kk,fontsize=plot_opts.font_head_size)
        axs.set_xlabel(sens_ax_labels[ii],fontsize=plot_opts.font_ax_size)
        axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)

    plt.show()

    print(80*"-")
    print("MAVM calculation complete.")
    print(80*"-")


if __name__ == "__main__":
    main()