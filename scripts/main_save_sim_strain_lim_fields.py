import time
import gc # garbage collector
from pathlib import Path
import json
import numpy as np
import pandas as pd

def main() -> None:
    print(80*"=")
    print("Strain Limit Extraction for Sim DIC Data: Pulse 25X")
    print(80*"=")
    print()

    STRAIN_KEY = "strain_xy"
    
    #---------------------------------------------------------------------------
    # MISC CONSTANTS
    
    # Tensor indexing
    (xx,yy,zz) = (0,1,2)
    xy = 2

    #---------------------------------------------------------------------------
    # SIM: constants
    SIM_TAG = "fullv3"

    FE_DIR = Path.cwd()/ "STC_ProbSim_FieldsFull_25X_v3"
    conv_to_mm: float = 1000.0 # Simulation is in SI and exp is in mm

    # Reduced: 5000 = 100 aleatory x 50 epistemic
    # Full: 400 aleatory x 250 epistemic
    # exp_data = exp_data.reshape(samps_n,epis_n,alea_n)
    #samps_n: int = 5000
    SIM_EPIS_N: int = 250
    SIM_ALEA_N: int = 400

    #---------------------------------------------------------------------------
    # Load 
    json_file = f"sim_strain_cases_imagedef_{SIM_TAG}.json"        
    with open(json_file, 'r') as file:
         # Parse the JSON into a Python dictionary or list
         strain_pt_max = json.load(file)

    for kk,vv in strain_pt_max.items():
        print(f"{kk=}, {vv}")
    print()

    ext_inds = {}
    ext_inds["max"] = strain_pt_max["max_inds"] 
    ext_inds["med"] = strain_pt_max["med_inds"]
    ext_inds["min"] = strain_pt_max["min_inds"]
    
    #---------------------------------------------------------------------------
    # Load simulation data
    print("LOAD SIM DATA")
    print(80*"-")
    sim_data = {}

    sim_coord_path = FE_DIR / "Mesh.csv"
    sim_field_paths = {"strain_xx": FE_DIR / "solid.el22 (1)_All.npy",
                       "strain_yy": FE_DIR / "solid.el33 (1)_All.npy",
                       "strain_xy": FE_DIR / "solid.el23 (1)_All.npy",}

    # Load simulation nodal coords
    print(f"Loading sim coords from:\n    {sim_coord_path}")
    start_time = time.perf_counter()
    sim_coords_df = pd.read_csv(sim_coord_path)
    sim_coords = sim_coords_df.to_numpy()
    sim_coords = sim_coords*conv_to_mm
    end_time = time.perf_counter()
    print(f"Loading sim coords took: {end_time-start_time}s\n")
    del sim_coords_df

    sim_num_nodes = sim_coords.shape[0]

    # Load simulation field data
    print(f"Loading simulation data from:\n    {FE_DIR}")
    start_time = time.perf_counter()

    # Push the displacement data into a single matrix from the separate files
    # shape=(n_epis,n_alea,n_nodes)
    sim_strain = np.zeros((SIM_EPIS_N,SIM_ALEA_N,sim_num_nodes),dtype=np.float64)

    sim_strain = (np.load(sim_field_paths[STRAIN_KEY])
                  .T
                  .reshape(SIM_EPIS_N,SIM_ALEA_N,sim_num_nodes))

         
    # First column of sim coords is the node number, remove it
    sim_coords = sim_coords[:,1:]
    # Add a column of zeros so we have a z coord of 0 as only x and y are given
    # in the coords file
    sim_coords = np.hstack((sim_coords,np.zeros((sim_coords.shape[0],1))))

    print()
    print(f"{sim_coords.shape=}")
    print(f"{sim_strain.shape=}")
    print()

    #---------------------------------------------------------------------------
    # Extract cases and save each to disk 
    save_path = FE_DIR
    
    for kk,ii in ext_inds.items():
        save_file = f"sim_sample_{kk}_{STRAIN_KEY}.npy"
        save_strain = sim_strain[ii[0],ii[1],:]
        print(f"Saving, {kk=}, {ii=}, {save_strain.shape=}")
        np.save(save_path/save_file,save_strain,allow_pickle=False)

if __name__ == "__main__":
    main()

