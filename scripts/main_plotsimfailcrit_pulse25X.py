from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import pyvale
import valmetrics as vm

def main() -> None:
    print(80*"=")
    print("PULSE 25X: Sim failure analysis")
    print(80*"=")

    SIM_DIR = Path.cwd() / "STC_ProbSim"

    sens_ax_labels = (r"Max Temp. [$^{\circ}C$]",r"Max VM Stress [$MPa$]",)
    sens_tags = ("Temp","Stress",)
    sens_num = len(sens_tags)

    save_path = Path.cwd() / "images_failcrit_pulse25X"
    if not save_path.is_dir():
        save_path.mkdir(exist_ok=True,parents=True)

    plot_opts = pyvale.sensorsim.PlotOptsGeneral()

    fig_ind: int = 0
    sim_c: str = "tab:blue"

    #---------------------------------------------------------------------------
    # Simulation: Load Data
    print(f"Loading simulation data from:\n    {SIM_DIR}\n")

    # Reduced: 5000 = 100 aleatory x 50 epistemic
    # Full: 400 aleatory x 250 epistemic
    # exp_data = exp_data.reshape(samps_n,epis_n,alea_n)
    #samps_n: int = 5000
    epis_n: int = 250#50
    alea_n: int = 400#100

    sim_keys = {"Max Temp.":11,
                "Max VM Stress":14,}

    sim_path = SIM_DIR / "SamplingResultsNonField.csv"
    sim_data_arr = pd.read_csv(sim_path).to_numpy()

    sim_data = {}
    for kk in sim_keys:
        sim_data[kk] = sim_data_arr[:,sim_keys[kk]]

        sim_data[kk] = sim_data[kk].reshape(epis_n,alea_n)

        print(kk)
        print(f"{sim_data[kk].shape=}")
        print()

    # Convert to MPa
    sim_data["Max VM Stress"] = sim_data["Max VM Stress"]*1e-6

    print()
    for kk in sim_data:
        print(f"{kk=}")
        print(f"{sim_data[kk][0,0]=}")
    print()

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


    PLOT_ALL_SIM_CDFS = True

    if PLOT_ALL_SIM_CDFS:
        for ii,kk in enumerate(sim_cdfs_all):
            fig, axs=plt.subplots(1,1,
                                figsize=plot_opts.single_fig_size_landscape,
                                layout="constrained")
            fig.set_dpi(plot_opts.resolution)

            for cc in sim_cdfs_all[kk]:
                axs.ecdf(cc.quantiles,color='tab:blue',linewidth=plot_opts.lw)

            axs.ecdf(sim_cdfs_lims[kk]["max"].quantiles
                     ,ls="--",color="black",linewidth=plot_opts.lw)
            axs.ecdf(sim_cdfs_lims[kk]["min"].quantiles
                     ,ls="--",color="black",linewidth=plot_opts.lw)
            axs.ecdf(sim_cdfs_lims[kk]["nom"].quantiles
                     ,ls="-",color="black",linewidth=plot_opts.lw)

            #axs.set_title(kk,fontsize=plot_opts.font_head_size)
            axs.set_xlabel(sens_ax_labels[ii],fontsize=plot_opts.font_ax_size)
            axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)

            save_fig_path = save_path / f"{fig_ind}_sim_allcdfs_{kk}.png"
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")


    print(80*"-")
    print("Complete.")
    print(80*"-")


if __name__ == "__main__":
    main()