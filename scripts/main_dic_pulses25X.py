import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import valmetrics as vm


def main() -> None:
    print(80*"=")
    print("MAVM Calc for DIC Data: Pulse 25X")
    print(80*"=")
    print()

    SIM_TAG = "25X"
    temp_path = Path.cwd() / f"temp_{SIM_TAG}"
    if not temp_path.is_dir():
        temp_path.mkdir()

    FE_DIR = Path.cwd()/ "STC_ProbSim_Reduced"

    DIC_DIRS = (
        Path.cwd() / "STC_Exp_Pulse253",
        Path.cwd() / "STC_Exp_Pulse254",
        Path.cwd() / "STC_Exp_Pulse255",
    )
    DIC_STEADY = {"253": (261,694),
                  "254": (252,694),
                  "255": (211,694)}

    if not FE_DIR.is_dir():
        raise FileNotFoundError(f"{FE_DIR}: directory does not exist.")

    for dd in DIC_DIRS:
        if not dd.is_dir():
            raise FileNotFoundError(f"{dd}: directory does not exist.")

    #---------------------------------------------------------------------------
    # Load simulation data
    print("LOAD SIM DATA")
    print(80*"-")

    sim_coord_path = FE_DIR / "Mesh.csv"
    sim_field_paths = {"disp_x": FE_DIR / "u (m)_All.npy",
                       "disp_y": FE_DIR / "v (m)_All.npy"}

    # Load simulation nodal coords
    print(f"Loading sim coords from:\n    {sim_coord_path}")
    start_time = time.perf_counter()
    sim_coords = pd.read_csv(sim_coord_path)
    sim_coords = sim_coords.to_numpy()
    end_time = time.perf_counter()
    print(f"Loading sim coords took: {end_time-start_time}s\n")

    # Load simulation field data
    print(f"Loading simulation data from:\n    {FE_DIR}")
    start_time = time.perf_counter()
    sim_fields = {}
    for ss in sim_field_paths:
        sim_fields[ss] = np.load(sim_field_paths[ss])


    print()
    print("SIM DATA SHAPES:")
    print("sim_coords.shape=(n_nodes,coord[x,y,z])")
    print(f"{sim_coords.shape=}")
    print("sim_field.shape=(n_nodes,field[DOE])")
    for ss in sim_fields:
        print(f"sim_field[{ss}].shape={sim_fields[ss].shape}")
    print()


    #---------------------------------------------------------------------------
    # Load experiment data



if __name__ == "__main__":
    main()