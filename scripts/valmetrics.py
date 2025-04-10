'''
================================================================================
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
from typing import Any
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from scipy import stats
from scipy.interpolate import griddata
import pyvale


def load_sim_data(data_path: Path, skip_header: int = 0) -> tuple[np.ndarray,np.ndarray]:
    csv_files = list(data_path.glob("*.csv"))
    csv_files = sorted(csv_files)

    data = np.genfromtxt(csv_files[0],
                         skip_header=skip_header,
                         dtype=np.float64,
                         delimiter=",")


    # Coords are the same for all FE files, store once:
    # fe_coords.shape = (num_nodes, coord[x,y]) = (num_nodes,2)
    sim_coords = data[:,0:2]

    # fe_data.shape = (num_files,num_nodes,disp[x,y,z])
    sim_disp = np.zeros((len(csv_files),sim_coords.shape[0],3))
    # The last three columns of the file are displacements
    sim_disp[0,:,:] = data[:,2:]

    # We have loaded the first data frame so we can remove it now
    csv_files.pop(0)

    for ii,ff in enumerate(csv_files):
        data = np.genfromtxt(ff,
                             skip_header=skip_header,
                             dtype=np.float64,
                             delimiter=",")
        sim_disp[ii+1,:,:] = data[:,2:]

    # fe_coords.shape = (num_nodes, coord[x,y]) = (num_nodes,2)
    # fe_data.shape = (num_files,num_nodes,disp[x,y,z])
    return (sim_coords,sim_disp)

def load_exp_data(data_path: Path, num_load: int | None = None, run_para: int | None = None) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    csv_files = list(data_path.glob("*.csv"))
    csv_files = sorted(csv_files)

    if num_load is not None:
        csv_files = csv_files[0:num_load]

    # Coords change with every DIC data file, need to account for this
    data = np.genfromtxt(csv_files[0],
                         dtype=np.float64,
                         delimiter=",",
                         skip_header=1)

    # print(f"{data.shape}")

    # exp_coords.shape=(num_files,num_points,coord[x,y,z])
    exp_coords = np.zeros((len(csv_files),data.shape[0],3))
    # exp_disp.shape=(num_files,num_points,disp[x,y,z])
    exp_disp = np.zeros((len(csv_files),data.shape[0],3))
    # exp_strain.shape=(num_files,num_points,strain[xx,yy,xy])
    exp_strain = np.zeros((len(csv_files),data.shape[0],3))

    exp_coords[0,:,:] = data[:,2:5]
    exp_disp[0,:,:] = data[:,5:8]
    exp_strain[0,:,:] = data[:,18:21]

    # print(f"{exp_coords[0,0,:]=}")
    # print(f"{exp_disp[0,0,:]=}")
    # print(f"{exp_strain[0,0,:]=}")

    # We have loaded the first data frame so we can remove it now
    csv_files.pop(0)

    if run_para is not None:

        with Pool(run_para) as pool:
            processes = []

            for ff in csv_files:
                processes.append(pool.apply_async(_load_one_exp, args=(ff,)))

            data_list = [pp.get() for pp in processes]

        exp_data = np.zeros((data.shape[0],9))
        exp_data[:,0:3] = data[:,2:5]
        exp_data[:,3:6] = data[:,5:8]
        exp_data[:,6:9] = data[:,18:21]
        data_list.insert(0,exp_data)

        # print(f"{len(data_list)=}")
        # print(f"{data_list[0].shape=}")
        # print(f"{data_list[1].shape=}")

        data_arr = np.stack(data_list)
        exp_coords = data_arr[:,:,0:3]
        exp_disp = data_arr[:,:,3:6]
        exp_strain = data_arr[:,:,6:9]

        # print()
        # print(f"{data_arr.shape=}")
        # print(f"{exp_coords.shape=}")
        # print(f"{exp_disp.shape=}")
        # print(f"{exp_strain.shape=}")#
        # print()

    else:
        for ii,ff in enumerate(csv_files):
            print(f"Loading experiment data file: {ii}.")
            data = np.genfromtxt(ff,
                            dtype=np.float64,
                            delimiter=",",
                            skip_header=1)
            exp_coords[ii,:,:] = data[:,2:5]
            exp_disp[ii,:,:] = data[:,5:8]
            exp_strain[ii,:,:] = data[:,18:21]


    # exp_coords.shape=(num_files,num_points,coord[x,y,z])
    # exp_disp.shape=(num_files,num_points,disp[x,y,z])
    # exp_strain.shape=(num_files,num_points,strain[xx,yy,xy])
    return (exp_coords,exp_disp,exp_strain)

def _load_one_exp(path: Path) -> np.ndarray:
        data = np.genfromtxt(path,
                        dtype=np.float64,
                        delimiter=",",
                        skip_header=1)
        exp_data = np.zeros((data.shape[0],9))
        exp_data[:,0:3] = data[:,2:5]
        exp_data[:,3:6] = data[:,5:8]
        exp_data[:,6:9] = data[:,18:21]
        return exp_data

def fit_coord_matrix(coords) -> np.ndarray:
    coord_mat = np.zeros((4,4))
    coord_mat[-1,-1] = 1

    origin = np.mean(coords,axis=0)
    coord_mat[:-1,-1] = origin

    # print(f"{origin=}")

    cov_mat = np.cov(coords.T)
    (_,_,v_mat) = np.linalg.svd(cov_mat)

    # print(f"{cov_mat.shape=}")
    # print(f"{v_mat.shape=}")

    # V matrix contains eigen vectors which are the axes
    axis_x = v_mat[0,:]
    axis_y = v_mat[1,:]
    axis_z = v_mat[2,:]

    # print(v_mat)
    # print()
    # print(axis_x)
    # print(axis_y)
    # print(axis_z)

    # Ensure that the coordinate system is right-handed.
    if np.dot(np.cross(axis_x, axis_y), axis_z) < 0:
        axis_y = -axis_y

    coord_mat[:-1,0] = axis_x
    coord_mat[:-1,1] = axis_y
    coord_mat[:-1,2] = axis_z

    return coord_mat

def find_nearest_points(coords: np.ndarray,
                        find_point: np.ndarray,
                        k: int = 5) -> np.ndarray:
    if coords.shape[1] >= 3:
        coords = coords[:,0:2]

    distances = np.sqrt(np.sum((coords - find_point)**2,axis=1))
    return np.argsort(distances)[:k]

def plot_disp_comp_maps(sim_coords: np.ndarray,
                       sim_disp_avg: np.ndarray,
                       exp_coords_avg: np.ndarray,
                       exp_disp_avg: np.ndarray,
                       ax_ind: int,
                       ax_str: str,
                       scale_cbar: bool = True,
                       save_tag: str = "") -> None:

    sim_x_min = np.min(sim_coords[:,0])
    sim_x_max = np.max(sim_coords[:,0])
    sim_y_min = np.min(sim_coords[:,1])
    sim_y_max = np.max(sim_coords[:,1])

    step = 0.5
    x_vec = np.arange(sim_x_min,sim_x_max,step)
    y_vec = np.arange(sim_y_min,sim_y_max,step)
    (x_grid,y_grid) = np.meshgrid(x_vec,y_vec)

    exp_disp_grid_avg = griddata(exp_coords_avg[:,0:2],
                             exp_disp_avg[:,ax_ind],
                             (x_grid,y_grid),
                             method="linear")

    # This will do minimal interpolation as the input points are the same as the sim
    sim_disp_grid_avg = griddata(sim_coords[:,0:2],
                             sim_disp_avg[:,ax_ind],
                             (x_grid,y_grid),
                             method="linear")

    disp_diff_avg = sim_disp_grid_avg - exp_disp_grid_avg

    color_max = np.nanmax((np.nanmax(sim_disp_grid_avg),np.nanmax(exp_disp_grid_avg)))
    color_min = np.nanmin((np.nanmin(sim_disp_grid_avg),np.nanmin(exp_disp_grid_avg)))

    # print(80*"-")
    # print(f"{sim_disp_avg.shape=}")
    # print(f"{sim_disp_avg[:,ax_ind].shape=}")
    # print(f"{sim_disp_grid_avg.shape=}")
    # print()
    # print(f"{np.max(sim_disp_grid_avg)=}")
    # print(f"{np.min(sim_disp_grid_avsaveg)=}")
    # print()
    # print(f"{color_max=}")
    # print(f"{color_min=}")
    # print(80*"-")

    cbar_font_size = 6.0

    plot_opts = pyvale.PlotOptsGeneral()
    fig_size = (plot_opts.a4_print_width,plot_opts.a4_print_width/(plot_opts.aspect_ratio*2.8))
    fig,ax = plt.subplots(1,3,figsize=fig_size,layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    if scale_cbar:
        image = ax[0].imshow(exp_disp_grid_avg,
                            extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max),
                            vmin = color_min,
                            vmax = color_max)
    else:
        image = ax[0].imshow(exp_disp_grid_avg,
                            extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))

    ax[0].set_title(f"Exp. Avg. \ndisp. {ax_str} [mm]",
                    fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
    ax[0].set_xlabel("x [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax[0].set_ylabel("y [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    cbar = plt.colorbar(image)
    cbar.ax.tick_params(labelsize=cbar_font_size)

    if scale_cbar:
        image = ax[1].imshow(sim_disp_grid_avg,
                            extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max),
                            vmin = color_min,
                            vmax = color_max)
    else:
        image = ax[1].imshow(sim_disp_grid_avg,
                            extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max))

    ax[1].set_title(f"Sim. Avg.\ndisp. {ax_str} [mm]",
                    fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
    ax[1].set_xlabel("x [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax[1].set_ylabel("y [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    cbar = plt.colorbar(image)
    cbar.ax.tick_params(labelsize=cbar_font_size)

    image = ax[2].imshow(disp_diff_avg,
                         extent=(sim_x_min,sim_x_max,sim_y_min,sim_y_max),
                         cmap="RdBu")
    ax[2].set_title(f"(Sim. - Exp.)\ndisp. {ax_str} [mm]",
                    fontsize=plot_opts.font_head_size, fontname=plot_opts.font_name)
    ax[2].set_xlabel("x [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    ax[2].set_ylabel("y [mm]",
                fontsize=plot_opts.font_ax_size, fontname=plot_opts.font_name)
    cbar = plt.colorbar(image)
    cbar.ax.tick_params(labelsize=cbar_font_size)

    if scale_cbar:
        if save_tag:
            save_path = Path("images")/f"disp_comp_{ax_str}_{save_tag}.png"
        else:
            save_path = Path("images")/f"disp_comp_{ax_str}.png"
    else:
        if save_tag:
            save_path = Path("images")/f"disp_comp_{ax_str}_cbarfree_{save_tag}.png"
        else:
            save_path = Path("images")/f"disp_comp_{ax_str}_cbarfree.png"

    fig.savefig(save_path,dpi=300,format="png",bbox_inches="tight")


def _interp_one_instance(coords: np.ndarray,
                         disp: np.ndarray,
                         x_grid: np.ndarray,
                         y_grid: np.ndarray) -> np.ndarray:
    disp_common = np.zeros((x_grid.size,3))

    for aa in range(0,3):
        sim_disp_grid = griddata(coords,
                                disp[:,aa],
                                (x_grid,y_grid),
                                method="linear")
        disp_common[:,aa] = sim_disp_grid.flatten()

    return disp_common


def interp_sim_to_common_grid(coords: np.ndarray,
                          disp: np.ndarray,
                          x_grid: np.ndarray,
                          y_grid: np.ndarray,
                          run_para: None | int = None) -> np.ndarray:


    if run_para is not None:
        with Pool(run_para) as pool:
            processes = []

            for ss in range(0,disp.shape[0]):
                processes.append(pool.apply_async(_interp_one_instance,
                                                  args=(coords[:,0:2],
                                                        disp[ss,:,:],
                                                        x_grid,
                                                        y_grid)))

            data_list = [pp.get() for pp in processes]

        disp_common = np.stack(data_list)

        print(80*"-")
        print(f"{len(data_list)=}")
        print(f"{data_list[0].shape=}")
        print(f"{disp_common.shape=}")
        print(80*"-")

        return disp_common


    # Non-parallel run
    for ss in range(0,disp.shape[0]):
        print(f"Interpolating: {ss}")
        for aa in range(0,3):
            disp_grid = griddata(coords[:,0:2],
                                        disp[ss,:,aa],
                                    (x_grid,y_grid),
                                    method="linear")
            disp_common[ss,:,aa] = disp_grid.flatten()

    return disp_common



def interp_exp_to_common_grid(coords: np.ndarray,
                              disp: np.ndarray,
                              x_grid: np.ndarray,
                              y_grid: np.ndarray,
                              run_para: None | int = None) -> np.ndarray:

    if run_para is not None:
        with Pool(run_para) as pool:
            processes = []

            for ss in range(0,disp.shape[0]):
                processes.append(pool.apply_async(_interp_one_instance,
                                                  args=(coords[ss,:,0:2],
                                                        disp[ss,:,:],
                                                        x_grid,
                                                        y_grid)))

            data_list = [pp.get() for pp in processes]

        disp_common = np.stack(data_list)

        print(80*"-")
        print(f"{len(data_list)=}")
        print(f"{data_list[0].shape=}")
        print(f"{disp_common.shape=}")
        print(80*"-")

        return disp_common

    # Non-parallel run
    disp_common = np.zeros((disp.shape[0],x_grid.size,3))

    for ss in range(0,disp.shape[0]):
        print(f"Interpolating: {ss}")
        for aa in range(0,3):
            disp_grid = griddata(coords[ss,:,0:2],
                                        disp[ss,:,aa],
                                    (x_grid,y_grid),
                                    method="linear")
            disp_common[ss,:,aa] = disp_grid.flatten()

    return disp_common



#-------------------------------------------------------------------------------
# MAVM Calculation
def mavm(model_data,
         exp_data
         ) -> dict[str,Any]:
    """
    Calculates the Modified Area Validation Metric.
    Adapted from Whiting et al., 2023, "Assessment of Model Validation, Calibration, and Prediction Approaches in the Presence of Uncertainty", Journal of Verification, Validation and Uncertainty Quantification, Vol. 8.
    Downloaded from http://asmedigitalcollection.asme.org/verification/article-pdf/8/1/011001/6974199/vvuq_008_01_011001.pdf on 24 May 2024.
    """

    # find empirical cdf
    model_cdf = stats.ecdf(model_data).cdf
    exp_cdf = stats.ecdf(exp_data).cdf

    F_ = model_cdf.quantiles
    Sn_ = exp_cdf.quantiles

    df = len(Sn_)-1
    t_alph = stats.t.ppf(0.95,df)

    Sn_conf = [Sn_ - t_alph*(np.nanstd(Sn_)/np.sqrt(len(Sn_))),
               Sn_ + t_alph*(np.nanstd(Sn_)/np.sqrt(len(Sn_)))]

    Sn_Y = exp_cdf.probabilities
    F_Y = model_cdf.probabilities

    P_F = 1/len(F_)
    P_Sn = 1/len(exp_cdf.quantiles)

    d_conf_plus = []
    d_conf_minus = []

    for k in [0,1]:

        ii = 0
        d_rem = 0

        d_plus = 0
        d_minus = 0

        Sn = Sn_conf[k]

        #If more experimental data points than model data points
        if len(Sn) > len(F_):

            for jj in range(0,len(F_)):
                if d_rem != 0:
                    d_ = (Sn[ii] - F_[jj]) * (P_Sn*(ii+1) - P_F*jj)
                    if d_ > 0:
                        d_plus += d_
                    else:
                        d_minus += d_
                    ii += 1
                while (jj+1)*P_F > (ii+1)*P_Sn:
                    d_ = (Sn[ii] - F_[jj])*P_F
                    if d_ > 0:
                        d_plus += d_
                    else:
                        d_minus += d_

                    ii += 1
                d_rem = (Sn[ii]-F_[jj])*(P_F*(jj+1) - P_Sn*ii)
                if d_rem > 0:
                    d_plus += d_rem
                else:
                    d_minus += d_rem

        #If more model data points than experimental data points (more typical)
        elif len(Sn) <= len(F_):

            for jj in range(0,len(Sn)):

                if d_rem != 0:
                    d_ = (Sn[jj]-F_[ii])*(P_F*(ii+1) - P_Sn*jj)
                    if d_ > 0:
                        d_plus += d_
                    else:
                        d_minus += d_
                    ii += 1

                while (ii+1)*P_F < (jj+1)*P_Sn:
                    d_ = (Sn[jj]-F_[ii])*P_F
                    if d_ > 0:
                        d_plus += d_
                    else:
                        d_minus += d_

                    ii += 1

                d_rem = (Sn[jj]-F_[ii])*(P_Sn*(jj+1) - P_F*ii)
                if d_rem > 0:
                    d_plus += d_rem
                else:
                    d_minus += d_rem

        d_conf_plus.append(np.abs(d_plus))
        d_conf_minus.append(np.abs(d_minus))

    d_plus = np.nanmax(d_conf_plus)
    d_minus = np.nanmax(d_conf_minus)


    output_dict = {"model_cdf":model_cdf,
                   "exp_cdf":exp_cdf,
                   "d+":d_plus,
                   "d-":d_minus,
                   "Sn_conf":Sn_conf,
                   "F_":F_,
                   "F_Y":F_Y,}

    return output_dict

def mavm_figs(mavm_res: dict[str,Any],
              title_str: str,
              field_label: str,
              save_tag: str = "") -> None:

    model_cdf = mavm_res["model_cdf"]
    exp_cdf = mavm_res["exp_cdf"]
    Sn_conf = mavm_res["Sn_conf"]
    d_plus = mavm_res["d+"]
    d_minus = mavm_res["d-"]
    F_ = mavm_res["F_"]
    F_Y = mavm_res["F_Y"]

    # fig,axs=plt.subplots(1,1)
    # model_cdf.plot(axs,label="model")
    # exp_cdf.plot(axs,label="experiment")
    # axs.legend()
    # axs.set_xlabel(field_label)
    # axs.set_ylabel("Probability")
    plot_opts = pyvale.PlotOptsGeneral()

    # plot empirical cdf with conf. int. cdfs
    fig,axs=plt.subplots(1,1,
                         figsize=plot_opts.single_fig_size_landscape,
                         layout="constrained")
    fig.set_dpi(plot_opts.resolution)

    axs.ecdf(model_cdf.quantiles,label="sim.")
    axs.ecdf(exp_cdf.quantiles,label="exp.")
    axs.ecdf(Sn_conf[0],ls="dashed",color="k",label=r"95% C.I.")
    axs.ecdf(Sn_conf[1],ls="dashed",color="k")
    axs.legend()

    axs.set_title(title_str,fontsize=plot_opts.font_head_size)
    axs.set_xlabel(field_label,fontsize=plot_opts.font_ax_size)
    axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)
    if save_tag:
        save_path = Path("images") / f"mavm ci {field_label} {title_str} {save_tag}.png"
    else:
        save_path = Path("images") / f"mavm ci {field_label} {title_str}.png"

    fig.savefig(save_path,dpi=300,format="png",bbox_inches="tight")

    fig,axs=plt.subplots(1,1,
                         figsize=plot_opts.single_fig_size_landscape,
                         layout="constrained")
    fig.set_dpi(plot_opts.resolution)

    axs.plot(F_,F_Y,"k-")
    axs.plot(F_+d_plus,F_Y,"k--")
    axs.plot(F_-d_minus,F_Y,"k--")
    axs.fill_betweenx(F_Y,F_-d_minus,F_+d_plus,color="k",alpha=0.2)

    axs.set_title(title_str,fontsize=plot_opts.font_head_size)
    axs.set_xlabel(field_label,fontsize=plot_opts.font_ax_size)
    axs.set_ylabel("Probability",fontsize=plot_opts.font_ax_size)

    if save_tag:
        save_path = Path("images") / f"mavm fill {field_label} {title_str} {save_tag}.png"
    else:
        save_path = Path("images") / f"mavm fill {field_label} {title_str}.png"

    fig.savefig(save_path,dpi=300,format="png",bbox_inches="tight")

def plot_mavm_map(mavm_d_plus: np.ndarray,
                  mavm_d_minus:np.ndarray,
                  ax_ind: int,
                  ax_str: str,
                  grid_shape: tuple[int,int],
                  extent: tuple[float,float,float,float],
                  save_tag: str = "") -> None:

    plot_opts = pyvale.PlotOptsGeneral()
    fig_size = (plot_opts.a4_print_width,plot_opts.a4_print_width/(plot_opts.aspect_ratio*2))

    fig,ax = plt.subplots(1,2,figsize=fig_size,layout='constrained')
    fig.set_dpi(plot_opts.resolution)

    mavm_dp_grid = np.reshape(mavm_d_plus[:,ax_ind],grid_shape)
    mavm_dm_grid = np.reshape(mavm_d_minus[:,ax_ind],grid_shape)

    image = ax[0].imshow(mavm_dp_grid,
                      extent=extent)
    cbar = plt.colorbar(image)
    ax[0].set_title(f"MAVM d+\ndisp. {ax_str} [mm]")
    ax[0].set_xlabel("x [mm]",fontsize=plot_opts.font_ax_size)
    ax[0].set_ylabel("y [mm]",fontsize=plot_opts.font_ax_size)

    image = ax[1].imshow(mavm_dm_grid,
                      extent=extent)
    cbar = plt.colorbar(image)
    ax[1].set_title(f"MAVM d-\ndisp. {ax_str} [mm]")
    ax[1].set_xlabel("x [mm]",fontsize=plot_opts.font_ax_size)
    ax[1].set_ylabel("y [mm]",fontsize=plot_opts.font_ax_size)

    if save_tag:
        save_path = Path("images")/f"mavm_map_disp{ax_str}_{save_tag}.png"
    else:
        save_path = Path("images")/f"mavm_map_disp{ax_str}.png"
    fig.savefig(save_path,dpi=300,format="png",bbox_inches="tight")


