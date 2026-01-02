import time
import gc # garbage collector
from pathlib import Path
import json
import numpy as np
import pandas as pd

def main() -> None:
    print(80*"=")
    print("Strain Limit Calc for Sim DIC Data: Pulse 25X")
    print(80*"=")
    print()

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
    # Load simulation data
    print("LOAD SIM DATA")
    print(80*"-")
    sim_data = {}

    sim_coord_path = FE_DIR / "Mesh.csv"
    # sim_field_paths = {"strain_xx": FE_DIR / "solid.el11 (1)_All.npy",
    #                    "strain_yy": FE_DIR / "solid.el22 (1)_All.npy",
    #                    "strain_xy": FE_DIR / "solid.el12 (1)_All.npy",}
    sim_field_paths = {"strain_xx": FE_DIR / "solid.el22 (1)_All.npy",
                       "strain_yy": FE_DIR / "solid.el33 (1)_All.npy",}
                       #"strain_xy": FE_DIR / "solid.el23 (1)_All.npy",}

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
    # shape=(n_epis,n_alea,n_nodes,n_comps[xx,yy,xy])
    sim_strain = np.zeros((SIM_EPIS_N,SIM_ALEA_N,sim_num_nodes,2),dtype=np.float64)

    for ss in sim_field_paths:
        # shape=(n_pts,n_doe)
        sim_temp = np.load(sim_field_paths[ss]).T.reshape(SIM_EPIS_N,SIM_ALEA_N,sim_num_nodes)

        if ss == "strain_xx":
            sim_strain[:,:,:,xx] = sim_temp        
        elif ss == "strain_yy":
            sim_strain[:,:,:,yy] = sim_temp
         
    del sim_temp
    gc.collect()

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
    # shape=(n_epis,n_alea,n_nodes,n_comps[xx,yy,xy])
    # 1. Which sample gives the highest average xx and yy strains?
    # 2. Find the max strain for all samples - max(ax=2)

    # Take the largest strain component as it will drive the error
    strain_abs = np.max(np.abs(sim_strain),axis=3)
    del sim_strain
    gc.collect()
    
    print(f"{strain_abs.shape=}")

    strain_max_abs = np.max(strain_abs)
    strain_max_flat_ind = np.argmax(strain_abs)
    strain_max_inds = np.unravel_index(strain_max_flat_ind,strain_abs.shape)

    print(80*"=")
    print(f"{strain_max_abs=}")
    print(f"{strain_max_flat_ind=}")
    print(f"{strain_max_inds=}")
    print(80*"=")

    # NOTE: operating on the average of the xx and yy strains here
    strain_pt_max_arr = np.max(strain_abs,axis=2)

    strain_pt_max = {}
    strain_pt_max["max"] = float(np.max(strain_pt_max_arr))
    strain_pt_max["min"] = float(np.min(strain_pt_max_arr))
    strain_pt_max["med"] = float(np.median(strain_pt_max_arr))

    strain_pt_max["max_flat"] = int(np.argmax(strain_pt_max_arr))
    strain_pt_max["min_flat"] = int(np.argmin(strain_pt_max_arr))
    strain_pt_max["med_flat"] = int(np.abs(
        strain_pt_max_arr-strain_pt_max["med"]
    ).argmin())

    strain_pt_max["max_inds"] = np.array(np.unravel_index(strain_pt_max["max_flat"],
                                              strain_pt_max_arr.shape)).tolist()
    strain_pt_max["min_inds"] = np.array(np.unravel_index(strain_pt_max["min_flat"],
                                              strain_pt_max_arr.shape)).tolist()
    strain_pt_max["med_inds"] = np.array(np.unravel_index(strain_pt_max["med_flat"],
                                              strain_pt_max_arr.shape)).tolist()

    print()
    print(f"{strain_pt_max_arr.shape=}")
    print()
    print(f"{strain_pt_max['max']=}")
    print(f"{strain_pt_max['med']=}")
    print(f"{strain_pt_max['min']=}")
    print()
    print(f"{strain_pt_max['max_flat']=}")
    print(f"{strain_pt_max['med_flat']=}")
    print(f"{strain_pt_max['min_flat']=}")
    print()
    print(f"{strain_pt_max['max_inds']=}")
    print(f"{strain_pt_max['med_inds']=}")
    print(f"{strain_pt_max['min_inds']=}")
    print()
 
    json_file = f"sim_strain_cases_imagedef_{SIM_TAG}.json"
    with open(json_file, "w") as json_file:
        json.dump(strain_pt_max, json_file, indent=4)
    

if __name__ == "__main__":
    main()

