from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import valmetrics as vm

def main() -> None:
    print(80*"=")
    print("PULSE 38: Thermocouple MAVM analysis")
    print(80*"=")

    # - Analyse these TCs: 1,2,4,5,7,8
    tc_tags = ("TC1","TC2","TC4","TC5","TC7","TC8")
    num_tcs = len(tc_tags)

    #---------------------------------------------------------------------------
    # LOAD DATA
    sim_inds = np.array((1,2,4,5,7,8),dtype=np.uintp)-1

    sim_path = Path.cwd()/"Pulse38_ProbSim_Thermocouples"/"UncertaintyPropagationThermocouples.csv"
    sim_data = np.genfromtxt(sim_path,dtype=np.float64,delimiter=",",skip_header=1)

    #sim_tc_tags = ("TC1","TC2","TC3","TC4","TC5","TC6","TC7","TC8","TC9","TC10")
    sim_data = sim_data[:,11:]
    sim_data = sim_data[:,sim_inds]

    sim_data_mean = np.mean(sim_data,axis=0)
    sim_data_std = np.std(sim_data,axis=0)

    print()
    print("Simulation data:")
    print(f"{sim_inds.shape=}")
    print(f"{sim_data.shape=}")
    print(f"{sim_data_mean=}")
    print(f"{sim_data_std=}")
    print()

    exp_root_path = Path.cwd()/"Pulse38_Exp_Thermocouples"

    exp_path = exp_root_path/"38_SteadyDataDIC.csv"
    #exp_dic_tags = ("TC1","TC3","TC4","TC6","TC7","TC9","TC10")
    exp_dic_inds = np.array((1,3,5),dtype=np.uintp)-1
    exp_dic_data = np.genfromtxt(exp_path,dtype=np.float64,delimiter=",",skip_header=1)
    exp_dic_data = exp_dic_data[:,3:10]
    exp_dic_data = exp_dic_data[:,exp_dic_inds]


    exp_path = exp_root_path/"38_SteadyDataHIVE.csv"
    exp_hive_tags = ("TC2","TC5","TC8")
    exp_hive_data = np.genfromtxt(exp_path,dtype=np.float64,delimiter=",",skip_header=1)
    exp_hive_data = exp_hive_data[:,12:]

    # ("TC1","TC2","TC4","TC5","TC7","TC8")

    exp_data = [exp_dic_data[:,0],   # TC1
                exp_hive_data[:,0],  # TC2
                exp_dic_data[:,1],   # TC4
                exp_hive_data[:,1],  # TC5
                exp_dic_data[:,2],   # TC7
                exp_hive_data[:,2]]  # TC8

    exp_data_mean = []
    for ee in exp_data:
        exp_data_mean.append(np.mean(ee))

    print("Experimental data:")
    print(f"{len(exp_data)=}")
    print(f"{exp_dic_data.shape=}")
    print(f"{exp_hive_data.shape=}")
    print()
    print(f"{exp_data_mean=}")
    print()

    #---------------------------------------------------------------------------
    # ANALYSE MAVM
    mavm_res = []
    for ii,ee in enumerate(exp_data):
        mavm_res.append(vm.mavm(sim_data[:,ii],ee))

        field_label = r"Temp. [$^{\circ}C$]"
        field_tag = "Temp"
        title_str = tc_tags[ii]
        save_tag = ""

        vm.mavm_figs(mavm_res[ii],
              title_str,
              field_label,
              field_tag=field_tag,
              save_tag=save_tag)

    plt.show()

if __name__ == "__main__":
    main()