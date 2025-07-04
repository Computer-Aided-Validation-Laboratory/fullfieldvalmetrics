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
    # Load Sim Data
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
    # Load Exp Data

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

    # for ee in exp_data:
    #     print(f"{ee}")
    #     print(f"{exp_data[ee].shape}")
    #     print()



    #---------------------------------------------------------------------------
    # Find Min/Max CDF over all eps


    # Extract all the simulation cdfs by looping over the epistemic sampling
    sim_cdfs = {}
    sim_max_cdf = {}
    sim_min_cdf = {}
    sim_max_data = {}
    sim_min_data = {}

    for kk in sim_data:
        cdfs = []

        max_data = sim_data[kk][0,:]
        min_data = sim_data[kk][0,:]

        min_cdf = stats.ecdf(sim_data[kk][0,:]).cdf
        max_cdf = stats.ecdf(sim_data[kk][0,:]).cdf

        sum_min_cdf = np.sum(min_cdf.quantiles)
        sum_max_cdf = np.sum(max_cdf.quantiles)

        for ee in range(epis_n):
            this_cdf = stats.ecdf(sim_data[kk][ee,:]).cdf
            this_cdf_sum = np.sum(this_cdf.quantiles)

            if this_cdf_sum > sum_max_cdf:
                sum_max_cdf = this_cdf_sum
                max_cdf = this_cdf
                max_data = sim_data[kk][ee,:]

            if this_cdf_sum < sum_min_cdf:
                sum_min_cdf = this_cdf_sum
                min_cdf = this_cdf
                min_data = sim_data[kk][ee,:]

            cdfs.append(this_cdf)

        sim_cdfs[kk] = cdfs
        sim_max_cdf[kk] = max_cdf
        sim_min_cdf[kk] = min_cdf
        sim_max_data[kk] = max_data
        sim_min_data[kk] = min_data


    plot_all_sim_cdfs = False
    if plot_all_sim_cdfs:
        for ii,kk in enumerate(sim_cdfs):
            fig, axs=plt.subplots(1,1,
                                figsize=plot_opts.single_fig_size_landscape,
                                layout="constrained")
            fig.set_dpi(plot_opts.resolution)

            for cc in sim_cdfs[kk]:
                axs.ecdf(cc.quantiles,color='xkcd:azure')

            axs.ecdf(sim_min_cdf[kk].quantiles,color="black")
            axs.ecdf(sim_max_cdf[kk].quantiles,color="black")

            axs.set_title(kk,fontsize=plot_opts.font_head_size)
            axs.set_xlabel(sens_ax_labels[ii],fontsize=plot_opts.font_ax_size)
            axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)

            save_fig_path = save_path / f"sim_allcdfs_{kk}.png"
            fig.savefig(save_fig_path,dpi=300,format="png",bbox_inches="tight")


    # for kk in exp_data:
    #     print(f"{kk=}")
    #     print(f"{exp_data[kk].shape=}")
    #     check_nan = np.any(np.isnan(exp_data[kk]))
    #     print(f"{check_nan=}")



    #---------------------------------------------------------------------------
    # Calculate MAVM
    mavm_epis_max = {}

    kk = "CV"


    print(80*"-")
    print(f"{exp_data[kk].shape=}")
    print(f"{sim_max_data[kk].shape=}")
    print()
    print(f"{np.mean(exp_data[kk])=}")
    print(f"{np.mean(sim_max_data[kk])=}")
    print()

    mavm_epis_max[kk] = vm.mavm(sim_max_data[kk],exp_data[kk],test="coilv")

    print(f"{mavm_epis_max[kk]['d+']=}")
    print(f"{mavm_epis_max[kk]['d-']=}")
    print(80*"-")

    vm.mavm_figs(mavm_epis_max[kk],
                    title_str=kk,
                    field_label=sens_ax_labels[ii],
                    field_tag=sens_tags[ii],
                    save_tag="maxepis",
                    save_path=save_path)






if __name__ == "__main__":
    main()