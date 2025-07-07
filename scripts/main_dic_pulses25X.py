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

    conv_to_mm: float = 1.0 # Seems like sim is in mm

    SIM_TAG = "25X"
    temp_path = Path.cwd() / f"temp_{SIM_TAG}"
    if not temp_path.is_dir():
        temp_path.mkdir()

    FE_DIR = Path.cwd()/ "STC_ProbSim_Reduced"

    EXP_IND: int = 0
    DIC_DIRS = (
        Path.cwd() / "STC_Exp_Pulse253",
        Path.cwd() / "STC_Exp_Pulse254",
        Path.cwd() / "STC_Exp_Pulse255",
    )
    DIC_STEADY = [(261,694),
                  (252,694),
                  (211,694)]

    if not FE_DIR.is_dir():
        raise FileNotFoundError(f"{FE_DIR}: directory does not exist.")

    for dd in DIC_DIRS:
        if not dd.is_dir():
            raise FileNotFoundError(f"{dd}: directory does not exist.")


    save_path = Path.cwd() / "images_dic_pulse25X"
    if not save_path.is_dir():
        save_path.mkdir(exist_ok=True,parents=True)


    #---------------------------------------------------------------------------
    # Load simulation data
    print("LOAD SIM DATA")
    print(80*"-")
    sim_data = {}

    sim_coord_path = FE_DIR / "Mesh.csv"
    sim_field_paths = {"disp_x": FE_DIR / "u (m)_All.npy",
                       "disp_y": FE_DIR / "v (m)_All.npy",
                       "disp_z": FE_DIR / "w (m)_All.npy",}

    # Load simulation nodal coords
    print(f"Loading sim coords from:\n    {sim_coord_path}")
    start_time = time.perf_counter()
    sim_coords_df = pd.read_csv(sim_coord_path)
    sim_coords = sim_coords_df.to_numpy()
    sim_coords = sim_coords*conv_to_mm
    end_time = time.perf_counter()
    print(f"Loading sim coords took: {end_time-start_time}s\n")
    del sim_coords_df

    # Load simulation field data
    print(f"Loading simulation data from:\n    {FE_DIR}")
    start_time = time.perf_counter()
    sim_temp = {}
    for ss in sim_field_paths:
        sim_temp[ss] = np.load(sim_field_paths[ss])
        if "disp" in ss:
            sim_temp[ss] = sim_temp[ss]*conv_to_mm

    # Push the displacement data into a single matrix from the separate files
    sim_num_nodes = sim_temp["disp_x"].shape[0]
    sim_num_doe = sim_temp["disp_x"].shape[1]
    sim_disp = np.zeros((sim_num_nodes,sim_num_doe,3),dtype=np.float64)

    sim_disp[:,:,0] = sim_temp["disp_x"]
    sim_disp[:,:,1] = sim_temp["disp_y"]
    sim_disp[:,:,2] = sim_temp["disp_z"]
    del sim_temp

    #---------------------------------------------------------------------------
    # Load experiment data
    print("LOAD EXP DATA")
    print(80*"-")
    exp_coord_temp = temp_path / f"exp_coords_{EXP_IND}.npy"
    exp_disp_temp = temp_path / f"exp_disp_{EXP_IND}.npy"

    if not exp_coord_temp.is_file() and not exp_disp_temp.is_file():

        exp_field_slices = {"coords":slice(2,5),
                            "disp":slice(5,8),}

        exp_load_opts = vm.ExpDataLoadOpts(skip_header=1,
                                           threads_num=8)

        print(f"Loading exp data from:\n    {DIC_DIRS[EXP_IND]}")
        start_time = time.perf_counter()
        exp_data = vm.load_exp_data(DIC_DIRS[EXP_IND],
                                    exp_field_slices,
                                    slice(DIC_STEADY[EXP_IND][0],DIC_STEADY[EXP_IND][1]),
                                    exp_load_opts)
        end_time = time.perf_counter()
        print(f"Loading exp data took: {end_time-start_time}s\n")

        print()
        print("Saving data to numpy binary")
        np.save(exp_coord_temp,exp_data["coords"])
        np.save(exp_disp_temp,exp_data["coords"])

    else:
        exp_data = {}
        print("Loading numpy exp data from:"
              + f"\n    {exp_coord_temp}"
              + f"\n    {exp_disp_temp}")
        start_time = time.perf_counter()
        exp_data["coords"] = np.load(exp_coord_temp)
        exp_data["disp"] = np.load(exp_disp_temp)
        end_time = time.perf_counter()
        print(f"Loading numpy exp data took: {end_time-start_time}s\n")

    print(80*"-")
    print("SIM DATA: SHAPES")
    print("sim_coords.shape=(n_nodes,coord[x,y,z])")
    print(f"{sim_coords.shape=}")
    print("sim_field.shape=(n_nodes,n_doe,n_comp[x,y,z])")
    for ss in sim_data:
        if "coords" not in ss:
            print(f"sim_field[{ss}].shape={sim_data[ss].shape}")
    print()
    print("EXP DATA: SHAPES")
    for ee in exp_data:
        print(f"exp_data[{ee}].shape=(n_subsets,n_frames,n_comps[x,y,z])")
        print(f"{exp_data[ee].shape=}")
    print(80*"-")

    exp_coords = exp_data["coords"]
    exp_disp = exp_data["disp"]
    del exp_data

    #---------------------------------------------------------------------------
    # Transform simulation coords
    # NOTE: field arrays have shape=(n_doe_samps,n_pts,n_comps)

    # Expects shape=(n_pts,coord[x,y,z]), outputs 4x4 transform matrix
    sim_to_world_mat = vm.fit_coord_matrix(sim_coords)
    world_to_sim_mat = np.linalg.inv(sim_to_world_mat)
    print("Sim to world matrix:")
    print(sim_to_world_mat)
    print()
    print("World to sim matrix:")
    print(world_to_sim_mat)
    print()

    print("Adding w coord and rotating sim coords")
    sim_with_w = np.hstack([sim_coords,
                            np.ones([sim_coords.shape[0],1])])
    print(f"{sim_with_w.shape=}")

    sim_coords = np.matmul(world_to_sim_mat,sim_with_w.T).T
    print(f"{sim_coords.shape=}")

    print("Returning sim coords by removing w coord:")
    sim_coords = sim_coords[:,:-1]
    print(f"{sim_coords.shape=}")
    del sim_with_w
    print()

    sim_disp_t = np.zeros_like(sim_disp)
    for ss in range(0,sim_num_nodes):
        sim_disp_t[ss,:,:] = np.matmul(world_to_sim_mat[:-1,:-1],
                                       sim_disp[ss,:,:].T).T

        rigid_disp = np.atleast_2d(np.mean(sim_disp_t[ss,:,:],axis=0)).T
        rigid_disp = np.tile(rigid_disp,sim_num_doe).T
        sim_disp_t[ss,:,:] -= rigid_disp

    sim_disp = sim_disp_t
    del sim_disp_t
    print(f"{sim_disp.shape=}")
    print()

    #---------------------------------------------------------------------------
    # Transform Exp Coords: required for each frame
    # NOTE: exp field arrays have shape=(n_frames,n_pts,n_comps)

    print("Transforming experimental coords.")
    print(f"{exp_coords.shape=}")

    exp_coord_t = np.zeros_like(exp_coords)
    exp_disp_t = np.zeros_like(exp_disp)

    for ff in range(0,exp_coords.shape[0]):
        exp_to_world_mat = vm.fit_coord_matrix(exp_coords[ff,:,:])
        world_to_exp_mat = np.linalg.inv(exp_to_world_mat)

        exp_with_w = np.hstack([exp_coords[ff,:,:],np.ones([exp_coords.shape[1],1])])

        exp_coord_temp = np.matmul(world_to_exp_mat,exp_with_w.T).T
        exp_coord_t[ff,:,:] = exp_coord_temp[:,:-1]

        # Flip the y coord for the experiment?
        exp_coord_t[ff,:,1] = -exp_coord_t[ff,:,1]

        exp_disp_t[ff,:,:] = np.matmul(world_to_exp_mat[:-1,:-1],exp_disp[ff,:,:].T).T
        rigid_disp = np.atleast_2d(np.mean(exp_disp_t[ff,:,:],axis=0)).T
        rigid_disp = np.tile(rigid_disp,exp_disp.shape[1]).T
        exp_disp_t[ff,:,:] -= rigid_disp

    exp_coords = exp_coord_t
    exp_disp = exp_disp_t
    del exp_coord_t, exp_disp_t

    print("After transformation:")
    print(f"{exp_coords.shape=}")
    print(f"{exp_disp.shape=}")
    print()


    #---------------------------------------------------------------------------
    # Comparison of simulation and experimental coords
    PLOT_COORD_COMP = True

    if PLOT_COORD_COMP:
        down_samp = 5
        frame = 700

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        ax.scatter(exp_coords[frame,::down_samp,0],
                    exp_coords[frame,::down_samp,1],
                    exp_coords[frame,::down_samp,2])
        ax.scatter(sim_coords[:,0],
                    sim_coords[:,1],
                    sim_coords[:,2])
        #ax.set_zlim(-1.0,1.0)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(exp_coords[frame,::down_samp,0],exp_coords[frame,::down_samp,1])
        ax.scatter(sim_coords[:,0],sim_coords[:,1])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.show()





if __name__ == "__main__":
    main()