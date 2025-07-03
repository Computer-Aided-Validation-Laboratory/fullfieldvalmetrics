from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import valmetrics as vm

def main() -> None:
    print(80*"=")
    print("PULSE 25X: Thermocouple MAVM analysis")
    print(80*"=")

    EXP_DIR = Path.cwd() / "STC_Exp_TCs"
    SIM_DIR = Path.cwd() / "STC_ProbSim_Full"

    #---------------------------------------------------------------------------
    # Load Sim Data
    # Reduced: 5000 = 100 aleatory x 50 epistemic
    # Full: 400 aleatory
    # exp_data = exp_data.reshape(samps_n,epis_n,alea_n)
    samps_n: int = 5000
    epis_n: int = 50
    alea_n: int = 100


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

    for ee in exp_data:
        print(f"{ee}")
        print(f"{exp_data[ee].shape}")
        print()


if __name__ == "__main__":
    main()