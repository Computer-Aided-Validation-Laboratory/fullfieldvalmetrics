'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''

import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import valmetrics as vm


def main() -> None:
    print(80*"=")
    print("MAVM Calc for DIC Data")
    print(80*"=")


    SIM_TAG = "v2"
    temp_path = Path.cwd() / f"temp_{SIM_TAG}"
    if not temp_path.is_dir():
        temp_path.mkdir()

    FE_DIR = Path.cwd()/ "Pulse38_ProbSim_v2_Disp"
    DIC_DIR = Path.cwd() / "Pulse38_Exp_DIC"

    if not FE_DIR.is_dir():
        raise FileNotFoundError(f"{FE_DIR}: directory does not exist.")
    if not DIC_DIR.is_dir():
        raise FileNotFoundError(f"{DIC_DIR}: directory does not exist.")

    #---------------------------------------------------------------------------
    force_load_csv = False
    sim_coord_path = temp_path / f"sim_coords_{SIM_TAG}.npy"
    sim_disp_path = temp_path / f"sim_disp_{SIM_TAG}.npy"

    if force_load_csv or (not sim_coord_path.is_file() and not sim_disp_path.is_file()):
        print(f"Loading csv simulation displacement data from:\n{FE_DIR}")
        start_time = time.perf_counter()
        (sim_coords,sim_disp) = vm.load_sim_data(FE_DIR,skip_header=9)
        end_time = time.perf_counter()
        print(f"Loading csv sim data took: {end_time-start_time}\n")

        np.save(sim_coord_path,sim_coords)
        np.save(sim_disp_path,sim_disp)
    else:
        print("Loading presaved binary npy sim data.")
        start_time = time.perf_counter()
        sim_coords = np.load(sim_coord_path)
        sim_disp = np.load(sim_disp_path)
        end_time = time.perf_counter()
        print(f"Loading binary sim data took: {end_time-start_time}s\n")

    print(f"{sim_coords.shape=}")
    print(f"{sim_disp.shape=}")
    print()

    sim_coords = 1000*sim_coords
    sim_disp = 1000*sim_disp

    #---------------------------------------------------------------------------
    exp_coord_path = temp_path / "exp_coords.npy"
    exp_disp_path = temp_path / "exp_disp.npy"
    exp_strain_path = temp_path / "exp_strain.npy"

    if not exp_coord_path.is_file() and not exp_disp_path.is_file():
        start_time = time.perf_counter()
        print(f"Loading csv experimental displacement data from:\n{DIC_DIR}")
        (exp_coords,exp_disp,exp_strain)= vm.load_exp_data(DIC_DIR,
                                            num_load=None,
                                            run_para=16)
        end_time = time.perf_counter()
        print(f"Loading exp data took: {end_time-start_time}s")


        print("Saving numpy arrays in binary format for faster reading...")
        np.save(exp_coord_path,exp_coords)
        np.save(exp_disp_path,exp_disp)
        np.save(exp_strain_path,exp_strain)
    else:
        print("Loading exp data from pre-saved npy binary format.")
        start_time = time.perf_counter()
        exp_coords = np.load(exp_coord_path)
        exp_disp = np.load(exp_disp_path)
        end_time = time.perf_counter()
        print(f"Loading exp data from npy took: {end_time-start_time} s")

    print()
    print(f"{exp_coords.shape=}")
    print(f"{exp_disp.shape=}")
    #print(f"{exp_strain.shape=}")
    print()

    #---------------------------------------------------------------------------
    # Transform Sim Coords: only required once
    print("Transforming simulation coords.")
    sim_coords = np.hstack((sim_coords,np.zeros((sim_coords.shape[0],1))))

    sim_to_world_mat = vm.fit_coord_matrix(sim_coords)
    world_to_sim_mat = np.linalg.inv(sim_to_world_mat)
    print("Sim to world matrix:")
    print(sim_to_world_mat)
    print()
    print("World to sim matrix:")
    print(world_to_sim_mat)
    print()

    sim_with_w = np.hstack([sim_coords,np.ones([sim_coords.shape[0],1])])
    print(f"{sim_with_w.shape=}")

    print("Returning sim coords by removing w coord:")
    sim_coords = np.matmul(world_to_sim_mat,sim_with_w.T).T
    print(f"{sim_coords.shape=}")
    sim_coords = sim_coords[:,:-1]
    print(f"{sim_coords.shape=}")
    print()

    sim_disp_t = np.zeros_like(sim_disp)
    for ss in range(0,sim_disp.shape[0]):
        sim_disp_t[ss,:,:] = np.matmul(world_to_sim_mat[:-1,:-1],sim_disp[ss,:,:].T).T

        rigid_disp = np.atleast_2d(np.mean(sim_disp_t[ss,:,:],axis=0)).T
        rigid_disp = np.tile(rigid_disp,sim_disp.shape[1]).T
        sim_disp_t[ss,:,:] -= rigid_disp

    sim_disp = sim_disp_t
    del sim_disp_t


    #---------------------------------------------------------------------------
    # Transform Exp Coords: required for each frame
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

    # down_samp = 5
    # frame = 700

    # fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")

    # ax.scatter(exp_coords[frame,::down_samp,0],
    #            exp_coords[frame,::down_samp,1],
    #            exp_coords[frame,::down_samp,2])
    # ax.scatter(sim_coords[:,0],
    #            sim_coords[:,1],
    #            sim_coords[:,2])
    # ax.set_zlim(-1.0,1.0)
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.scatter(exp_coords[frame,::down_samp,0],exp_coords[frame,::down_samp,1])
    # ax.scatter(sim_coords[:,0],sim_coords[:,1])
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # plt.show()

    #---------------------------------------------------------------------------
    # Plot displacement fields on transformed coords
    sim_disp = sim_disp[:,:,[1,2,0]]
    # Based on the figures:
    # exp_disp_0 = sim_disp_1 = X
    # exp_disp_1 = sim_disp_2 = Y
    # exp_disp_2 = sim_disp_0 = Z

    sim_x_min = np.min(sim_coords[:,0])
    sim_x_max = np.max(sim_coords[:,0])
    sim_y_min = np.min(sim_coords[:,1])
    sim_y_max = np.max(sim_coords[:,1])

    plot_disp_sim_exp = True
    if plot_disp_sim_exp:
        frame = 500
        div_n = 1000

        x_vec = np.linspace(sim_x_min,sim_x_max,div_n)
        y_vec = np.linspace(sim_y_min,sim_y_max,div_n)

        (x_grid,y_grid) = np.meshgrid(x_vec,y_vec)

        for aa in range(0,3):
            exp_disp_grid = griddata(exp_coords[frame,:,0:2],
                                    exp_disp[frame,:,aa],
                                    (x_grid,y_grid),
                                    method="linear")

            fig,ax = plt.subplots()
            image = ax.imshow(exp_disp_grid,extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))
            #ax.scatter(exp_coords[frame,:,0],exp_coords[frame,:,1])
            plt.title(f"Exp Data: disp_{aa}")
            plt.colorbar(image)
            plt.savefig(Path("images")/f"exp_map_{SIM_TAG}_disp{aa}.png")


        for aa in range(0,3):
            sim_disp_grid = griddata(sim_coords[:,0:2],
                                    sim_disp[0,:,aa],
                                    (x_grid,y_grid),
                                    method="linear")

            fig,ax = plt.subplots()
            image = ax.imshow(sim_disp_grid,extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))
            #ax.scatter(sim_coords[:,0],sim_coords[:,1])
            plt.title(f"Sim Data: disp_{aa}")
            plt.colorbar(image)
            plt.savefig(Path("images")/f"sim_map_{SIM_TAG}_disp{aa}.png")

    #---------------------------------------------------------------------------
    # Find the steady state portion of the experiment for averaging
    print("Analysing experiment displacement traces to extract steady state region.")
    exp_coords_avg = np.mean(exp_coords,axis=0)
    print(f"{exp_coords_avg.shape=}")

    find_point_0 = np.array([20,-15]) # mm
    find_point_1 = np.array([0,-15])  # mm

    trace_inds_0 = vm.find_nearest_points(exp_coords_avg,find_point_0,k=5)
    trace_inds_1 = vm.find_nearest_points(exp_coords_avg,find_point_1,k=5)

    print(f"{exp_coords_avg[trace_inds_0,:]=}")
    print(f"{exp_coords_avg[trace_inds_1,:]=}")

    # Plot traces from a few experimental points near the top to find steady state
    # NOTE: coords are flipped compared to plotted maps above!
    # EXPERIMENT STEADY STATE: 300-650

    plot_disp_traces = True
    if plot_disp_traces:
        ax_ind: int = 0
        fig,ax = plt.subplots()
        for ii in trace_inds_0:
            ax.scatter(np.arange(0,exp_disp.shape[0]),exp_disp[:,ii,ax_ind])
        plt.title(f"Exp: disp_{ax_ind} traces")
        ax.set_xlabel("frame [#]")
        ax.set_ylabel(f"disp_{ax_ind} [mm]")
        plt.savefig(Path("images")/f"exp_disp_traces_{SIM_TAG}_{ax_ind}.png")

        ax_ind: int = 1
        fig,ax = plt.subplots()
        for ii in trace_inds_1:
            ax.scatter(np.arange(0,exp_disp.shape[0]),exp_disp[:,ii,ax_ind])
        plt.title(f"Exp: disp_{ax_ind} traces")
        ax.set_xlabel("frame [#]")
        ax.set_ylabel(f"disp_{ax_ind} [mm]")
        plt.savefig(Path("images")/f"exp_disp_traces_{SIM_TAG}_{ax_ind}.png")

        ax_ind: int = 2
        fig,ax = plt.subplots()
        for ii in trace_inds_0:
            ax.scatter(np.arange(0,exp_disp.shape[0]),exp_disp[:,ii,ax_ind])
        plt.title(f"Exp: disp_{ax_ind} traces")
        ax.set_xlabel("frame [#]")
        ax.set_ylabel(f"disp_{ax_ind} [mm]")
        plt.savefig(Path("images")/f"exp_disp_traces_{SIM_TAG}_{ax_ind}.png")


    #---------------------------------------------------------------------------
    # Average fields from experiment and simulation to plot the difference
    print("\nAveraging experiment steady state and simulation for full-field comparison.")
    exp_avg_start: int = 300
    exp_avg_end: int = 650

    exp_coords = exp_coords[exp_avg_start:exp_avg_end,:,:]
    exp_disp = exp_disp[exp_avg_start:exp_avg_end,:,:]

    exp_coords_avg = np.mean(exp_coords[exp_avg_start:exp_avg_end,:,:],axis=0)
    exp_disp_avg = np.mean(exp_disp[exp_avg_start:exp_avg_end,:,:],axis=0)
    sim_disp_avg = np.mean(sim_disp,axis=0)

    print(f"{exp_disp_avg.shape=}")
    print(f"{sim_disp_avg.shape=}")

    elem_size = np.min(np.sqrt(np.sum((sim_coords[1:,:] - sim_coords[0,:])**2,axis=1)))

    tol = 1e-6
    scale = 1/tol
    round_arr = np.round(sim_coords[:,0] * scale) / scale
    num_elem_x = np.unique(round_arr)
    round_arr = np.round(sim_coords[:,1]* scale) / scale
    num_elem_y = np.unique(round_arr)

    print(f"{elem_size=}")
    print()
    print(f"{sim_x_min=}")
    print(f"{sim_x_max=}")
    print(f"{sim_y_min=}")
    print(f"{sim_y_max=}")
    print(f"{(sim_x_max-sim_x_min)=}")
    print(f"{(sim_y_max-sim_y_min)=}")
    print()
    print(f"{num_elem_x.shape=}")
    print(f"{num_elem_y.shape=}")
    print()

    ax_inds = (0,1,2)
    ax_strs = ("x","y","z")

    plot_on = True
    if plot_on:
        for ii,ss in zip(ax_inds,ax_strs):
            vm.plot_disp_comp_maps(sim_coords,
                            sim_disp_avg,
                            exp_coords_avg,
                            exp_disp_avg,
                            ii,
                            ss,
                            scale_cbar=True,
                            save_tag=SIM_TAG)
            vm.plot_disp_comp_maps(sim_coords,
                            sim_disp_avg,
                            exp_coords_avg,
                            exp_disp_avg,
                            ii,
                            ss,
                            scale_cbar=False,
                            save_tag=SIM_TAG)


    #---------------------------------------------------------------------------
    # Calculate the MAVM for a few key points:
    print("Starting MAVM calculation.")

    # Interpolate all displacements onto a common grid
    sim_x_min = np.min(sim_coords[:,0])
    sim_x_max = np.max(sim_coords[:,0])
    sim_y_min = np.min(sim_coords[:,1])
    sim_y_max = np.max(sim_coords[:,1])

    step = 0.5
    x_vec = np.arange(sim_x_min,sim_x_max,step)
    y_vec = np.arange(sim_y_min,sim_y_max,step)
    (x_grid,y_grid) = np.meshgrid(x_vec,y_vec)
    grid_shape = x_grid.shape
    grid_pts = x_grid.size

    force_interp_common = False
    sim_disp_common_path = temp_path / f"sim_disp_common_{SIM_TAG}.npy"
    exp_disp_common_path = temp_path / f"exp_disp_common_{SIM_TAG}.npy"

    if force_interp_common or(not sim_disp_common_path.is_file() and not exp_disp_common_path.is_file()):
        print("Interpolating simulation displacements to common grid.")
        start_time = time.perf_counter()
        sim_disp_common = vm.interp_sim_to_common_grid(sim_coords,
                                                       sim_disp,
                                                       x_grid,
                                                       y_grid,
                                                       run_para=16)
        end_time = time.perf_counter()
        print(f"Interpolating sim. displacements took: {end_time-start_time}s\n")


        print("Interpolating experiment displacements to common grid.")
        start_time = time.perf_counter()
        exp_disp_common = vm.interp_exp_to_common_grid(exp_coords,
                                                       exp_disp,
                                                       x_grid,
                                                       y_grid,
                                                       run_para=16)
        end_time = time.perf_counter()
        print(f"Interpolating exp. displacements took: {end_time-start_time}s\n")

        print("Saving interpolated common grid data in npy format for speed.")
        np.save(sim_disp_common_path,sim_disp_common)
        np.save(exp_disp_common_path,exp_disp_common)
    else:
        print("Loading pre-interpolated sim and exp disp data for speed.")
        sim_disp_common = np.load(sim_disp_common_path)
        exp_disp_common = np.load(exp_disp_common_path)


    coords_common = np.vstack((x_grid.flatten(),y_grid.flatten())).T

    print()
    print("Interpolated data shapes:")
    print(f"{sim_disp_common.shape=}")
    print(f"{exp_disp_common.shape=}")
    print(f"{coords_common.shape=}")
    print()

    find_point_x = np.array([24.0,-16.0]) # mm
    find_point_y = np.array([0.0,-16.0])  # mm

    mavm_inds = {}
    mavm_inds["x"] = vm.find_nearest_points(coords_common,find_point_x,k=3)
    mavm_inds["y"] = vm.find_nearest_points(coords_common,find_point_y,k=3)

    xx: int = 0
    yy: int = 1

    ax_str = "x"
    mavm_res = {}

    mavm_res[ax_str] = vm.mavm(sim_disp_common[:,mavm_inds[ax_str][0],xx],
                               exp_disp_common[:,mavm_inds[ax_str][0],xx])
    ax_str = "y"
    mavm_res[ax_str] = vm.mavm(sim_disp_common[:,mavm_inds[ax_str][0],yy],
                               exp_disp_common[:,mavm_inds[ax_str][0],yy])


    print(80*"-")
    print(f"{mavm_inds['x']=}")
    print(f"{mavm_inds['y']=}")
    print(f"{coords_common[mavm_inds['x'],:]=}")
    print(f"{coords_common[mavm_inds['y'],:]=}")
    print(80*"-")
    print()

    plot_mavm = True
    if plot_mavm:
        ax_str = "x"
        field_label = f"disp. {ax_str} [mm]"
        vm.mavm_figs(mavm_res[ax_str],
                f"(x,y)=({coords_common[mavm_inds[ax_str][0],0]:.2f},{-1*coords_common[mavm_inds[ax_str][0],1]:.2f})",
                field_label,
                save_tag=SIM_TAG)

        ax_str = "y"
        field_label = f"disp. {ax_str} [mm]"
        vm.mavm_figs(mavm_res[ax_str],
              f"(x,y)=({coords_common[mavm_inds[ax_str][0],0]:.2f},{-1*coords_common[mavm_inds[ax_str][0],1]:.2f})",
              field_label,
              save_tag=SIM_TAG)

    print(80*"-")
    print(f"{type(mavm_res['x']['d+'])=}")
    print(f"{type(mavm_res['x']['d-'])=}")
    print(f"{type(mavm_res['y']['d+'])=}")
    print(f"{type(mavm_res['y']['d-'])=}")
    print()
    print(f"{mavm_res['x']['d+']=}")
    print(f"{mavm_res['x']['d-']=}")
    print(f"{mavm_res['y']['d+']=}")
    print(f"{mavm_res['y']['d-']=}")
    print(80*"-")
    #---------------------------------------------------------------------------
    # Calculate the mavm d+,d- full-field
    mavm_d_plus_path = temp_path / f"mavm_d_plus_{SIM_TAG}.npy"
    mavm_d_minus_path = temp_path / f"mavm_d_minus_{SIM_TAG}.npy"

    if not mavm_d_plus_path.is_file() and not mavm_d_minus_path.is_file():
        print("Calculating MAVM d+ and d- over all points for all disp comps.")
        mavm_d_plus = np.zeros((grid_pts,3))
        mavm_d_minus = np.zeros((grid_pts,3))
        for pp in range(0,grid_pts):

            for aa in range(0,3):
                if np.count_nonzero(np.isnan(exp_disp_common[:,pp,aa])) > 0:
                    mavm_d_plus[pp,aa] = np.nan
                    mavm_d_minus[pp,aa] = np.nan
                else:
                    mavm_res = vm.mavm(sim_disp_common[:,pp,aa],exp_disp_common[:,pp,aa])
                    mavm_d_plus[pp,aa] = mavm_res["d+"]
                    mavm_d_minus[pp,aa] = mavm_res["d-"]

        print("Saving MAVM calculation for faster loading.")
        np.save(mavm_d_plus_path,mavm_d_plus)
        np.save(mavm_d_minus_path,mavm_d_minus)
    else:
        print("Loading previous MAVM d+ and d- from npy.")
        mavm_d_plus = np.load(mavm_d_plus_path)
        mavm_d_minus = np.load(mavm_d_minus_path)


    print(f"{mavm_d_plus.shape=}")
    print(f"{mavm_d_minus.shape=}")

    ax_strs = ("x","y","z")
    ax_inds = (0,1,2)
    extent = (sim_x_min,sim_x_max,sim_y_min,sim_y_max)
    for ii,ss in zip(ax_inds,ax_strs):
        vm.plot_mavm_map(mavm_d_plus,
                      mavm_d_minus,
                      ii,
                      ss,
                      grid_shape,
                      extent,
                      save_tag=SIM_TAG)

    #---------------------------------------------------------------------------
    # Final show to pop all produced figures
    print("COMPLETE.")
    plt.show()


if __name__ == "__main__":
    main()

