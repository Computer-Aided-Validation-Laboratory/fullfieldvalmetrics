'''
pyvale: the python validation engine
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
'''
import time
from pprint import pprint
from typing import Any
from pathlib import Path
from sklearn import metrics
from scipy import integrate, stats, interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyDOE import *
import os, math
import numpy as np
import numpy.testing as npt
import pandas as pd
from multiprocessing.pool import Pool

np.random.seed(1000)

#USER_DIR = Path.home() / 'git/herding-moose/'
PLOT_RES = 1#

class SimDataset:

    def __init__(self, output_dir, file_name, num_tcs):

        print("Reading dataset...",end=" ")

        out_file_path = output_dir / file_name
        out_data = pd.read_csv(out_file_path)

        self.coil_voltage = np.array(out_data["Coil voltage"])
        self.coil_current = np.array(out_data["Coil Current"])
        self.sample_max_temp = np.array(out_data["Sample Maximum Temperature (degC)"])
        self.coil_max_temp = np.array(out_data["Coil Maximum Temperature (degC)"])
        tc_all = []
        for i in range(num_tcs):
            tc_all.append(out_data[f"TC{i+1} (degC)"])
        self.tc_all = np.array(tc_all)

        print("Done.")

        return None

class UQEvalPtsDataset:

    def __init__(self, output_dir, file_name, tc_nums, driver):

        print("Reading dataset...",end=" ")

        out_file_path = output_dir / file_name
        out_data = pd.read_csv(out_file_path)

        if driver == "current":
            self.coil_current = np.array(out_data["Coil current"])
        elif driver == "voltage":
            self.coil_voltage = np.array(out_data["Coil voltage"])
        tc_all = []
        for i in tc_nums:
            tc_all.append(out_data[f"Temperature (degC), TC{i}"])
        self.tc_all = np.array(tc_all)
        self.tc_inds = np.array(tc_nums)

        print("Done.")

        return None

class SteadyDataset:

    def __init__(self, output_dir, hive_file_name, dic_file_name, tc_nums):

        print("Reading dataset...",end=" ")

        hive_file_path = output_dir / hive_file_name
        hive_data = pd.read_csv(hive_file_path)

        dic_file_path = output_dir / dic_file_name
        dic_data = pd.read_csv(dic_file_path)

        self.coil_current = np.array(hive_data["Coil RMS Current"])
        self.coil_voltage = np.array(hive_data["Coil RMS Voltage"])
        tc_all = []
        tc_source = []
        for i in tc_nums:
            if f"TC{i}" in hive_data.keys().to_list():
                tc_all.append(np.array(hive_data[f"TC{i}"]))
                tc_source.append("H")
            elif f"TC{i}" in dic_data.keys().to_list():
                tc_all.append(np.array(dic_data[f"TC{i}"]))
                tc_source.append("D")
        self.tc_all = tc_all#np.array(tc_all)
        self.tc_inds = np.array(tc_nums)
        self.tc_source = np.array(tc_source)

        self.dic_time = np.array(dic_data["Timestamp"])
        self.hive_time = np.array(hive_data["Timestamp"])

        fig,axs = plt.subplots(3)#,sharex=True)
        fig.suptitle("Steady state experimental data")
        #axs[0].plot(self.hive_time,self.coil_current)
        #axs[0].set_ylabel("Coil current [A]")
        #axs[1].plot(self.hive_time,self.coil_voltage)
        #axs[1].set_ylabel("Coil voltage [V]")
        axs[0].hist(self.coil_current,histtype="step")
        axs[0].set_xlabel("Coil current [A]")
        axs[1].hist(self.coil_voltage, histtype="step")
        axs[1].set_xlabel("Coil voltage [V]")
        for ii in range(len(tc_nums)):
            if tc_source[ii]=="H":
                #axs[2].plot(self.hive_time,self.tc_all[ii],label=f"TC{tc_nums[ii]}")
                axs[2].hist(self.tc_all[ii],label=f"TC{tc_nums[ii]}",histtype="step")
            elif tc_source[ii]=="D":
                #axs[2].plot(self.dic_time,self.tc_all[ii],label=f"TC{tc_nums[ii]}")
                axs[2].hist(self.tc_all[ii],label=f"TC{tc_nums[ii]}",histtype="step")
        axs[2].legend(ncol=2)
        #axs[2].set_xlabel("Time [s]")
        #axs[2].set_ylabel("Temperature [$\degree$C]")
        axs[2].set_xlabel("Temperature [$\degree$C]")
        fig.tight_layout()
        plt.show()

        print("Done")

        return None



class ExpCombDataset:

    def __init__(self, output_dir, file_name, num_tcs):

        print("Reading dataset...",end=" ")

        out_file_path = output_dir / file_name
        out_data = pd.read_csv(out_file_path)

        print(out_data.keys().to_list())

        self.time = np.array(out_data["TimeStamp"])
        tc_all = []
        tc_inds = []
        for i in range(num_tcs):
            if i==0:
                tc_inds.append(i+1)
                tc_all.append(out_data[f" K-Type TC [degC]"])
            elif f" K-Type TC {i+1} [degC]" in out_data.keys().to_list() and (~np.any(out_data[f" K-Type TC {i+1} [degC]"]=="Bad Thermocouple") and ~np.all(out_data[f" K-Type TC {i+1} [degC]"]==0)):
                tc_inds.append(i+1)
                tc_all.append(out_data[f" K-Type TC {i+1} [degC]"])
        self.tc_all = np.array(tc_all)
        self.tc_inds = np.array(tc_inds)

        print("Done.")

        return None

class FrontFaceFEData_Single:

    def __init__(self, output_dir,file_name):

        #print("Reading dataset...",end=" ")

        out_file_path = output_dir / file_name
        out_data = pd.read_csv(out_file_path,header=None)

        self.xc = np.array(out_data[0])
        self.yc = np.array(out_data[1])
        self.dx = np.array(out_data[2])
        self.dy = np.array(out_data[3])
        self.dz = np.array(out_data[4])

        #print("Done.")

        return None

class FrontFaceFEDataset:

    def __init__(self,output_dir):

        allFiles = os.listdir(output_dir)
        selectFiles = np.array([filename for filename in allFiles if "FrontFaceDisplacements" in filename])

        print(f"Reading {len(selectFiles)} timesteps...",end=" ")

        xct = []
        yct = []
        dxt = []
        dyt = []
        dzt = []

        for num in range(len(selectFiles)):
            file_name = f"FrontFaceDisplacements_{num:02d}.csv"
            data = FrontFaceFEData_Single(output_dir,file_name)
            xct.append(data.xc)
            yct.append(data.yc)
            dxt.append(data.dx)
            dyt.append(data.dy)
            dzt.append(data.dz)

        self.xct = np.array(xct)
        self.yct = np.array(yct)
        self.dxt = np.array(dxt)
        self.dyt = np.array(dyt)
        self.dzt = np.array(dzt)

        print("Done.")

        return None

class DICDisplacement_Single:

    def __init__(self,output_dir,file_name):

        out_file_path = output_dir / file_name
        out_data = pd.read_csv(out_file_path,usecols=[0,1,2,3,4,5,6,7],
                                names=["xp","yp","xc","yc","zc","dx","dy","dz"],
                                skiprows=1)

        self.xpix = np.array(out_data["xp"])
        self.ypix = np.array(out_data["yp"])
        self.xc = np.array(out_data["xc"])*1e-3
        self.yc = np.array(out_data["yc"])*1e-3
        self.zc = np.array(out_data["zc"])*1e-3
        self.dx = np.array(out_data["dx"])*1e-3
        self.dy = np.array(out_data["dy"])*1e-3
        self.dz = np.array(out_data["dz"])*1e-3

        return None

class DICDisplacement:

    def __init__(self,output_dir):

        allFiles = os.listdir(output_dir)
        selectFiles = np.array([filename for filename in allFiles if ".tiff.csv" in filename])

        print(f"Reading {len(selectFiles)} datasets...",end=" ")

        xpix = []
        ypix = []
        xct = []
        yct = []
        dxt = []
        dyt = []
        dzt = []

        for num in range(len(selectFiles)):
            file_name = f"Image_{num:04d}_0.tiff.csv"
            data = DICDisplacement_Single(output_dir,file_name)
            xpix.append(data.xpix)
            ypix.append(data.ypix)
            xct.append(data.xc)
            yct.append(data.yc)
            dxt.append(data.dx)
            dyt.append(data.dy)
            dzt.append(data.dz)

        self.xpix = np.array(xpix)
        self.ypix = np.array(ypix)
        self.xct = np.array(xct)
        self.yct = np.array(yct)
        self.dxt = np.array(dxt)
        self.dyt = np.array(dyt)
        self.dzt = np.array(dzt)

        print("Done.")

        return None

def read_sim_data(output_dir,file_name,num_tcs,driver):

    dataset = SimDataset(output_dir,file_name,num_tcs)

    print("TC dataset size:",dataset.tc_all.shape)

    if driver == "voltage":
        plt.figure()
        for ii in range(dataset.tc_all.shape[0]):
            plt.plot(dataset.coil_voltage,dataset.tc_all[ii],".",label=f"TC{ii+1}")
        plt.plot(dataset.coil_voltage,dataset.sample_max_temp,"k.",label=f"Sample max temp")
        plt.legend()
        plt.xlabel("Coil voltage [V]")
        plt.ylabel("Thermocouple reading [$\degree$C]")
        plt.show()

        #rounded_voltage = np.rint(dataset.coil_voltage)
        voltage_values = np.unique(dataset.coil_voltage)
        #voltage_values = np.unique(rounded_voltage)

        tc_temps = []
        max_temps = []
        for vv in voltage_values:
            voltage_indices = np.argwhere(dataset.coil_voltage==vv).flatten()
            #voltage_indices = np.argwhere(rounded_voltage==vv).flatten()
            tc_temps.append(dataset.tc_all[:,voltage_indices])
            max_temps.append(dataset.sample_max_temp[voltage_indices])

        tc_temps = np.array(tc_temps) # tc_temps: [voltage index, thermocouple #, sample #]
        max_temps = np.array(max_temps) # max_temps: [voltage index, sample #]

    elif driver == "current":
        plt.figure()
        for ii in range(dataset.tc_all.shape[0]):
            plt.plot(dataset.coil_current,dataset.tc_all[ii],".",label=f"TC{ii+1}")
        plt.plot(dataset.coil_current,dataset.sample_max_temp,"k.",label=f"Sample max temp")
        plt.legend()
        plt.xlabel("Coil current [A]")
        plt.ylabel("Thermocouple reading [$\degree$C]")
        plt.show()

        current_values = np.unique(dataset.coil_current)

        tc_temps = []
        max_temps = []
        for vv in current_values:
            current_indices = np.argwhere(dataset.coil_current==vv).flatten()
            tc_temps.append(dataset.tc_all[:,current_indices])
            max_temps.append(dataset.sample_max_temp[current_indices])

        tc_temps = np.array(tc_temps) # tc_temps: [voltage index, thermocouple #, sample #]
        max_temps = np.array(max_temps) # max_temps: [voltage index, sample #]

    plt.figure()
    for tc_ in range(dataset.tc_all.shape[0]):
        plt.ecdf(tc_temps[:,tc_].flatten(),label=f"TC{tc_+1}")
    plt.ecdf(max_temps.flatten(),label="Max temp")
    plt.legend()
    plt.xlabel(f"Thermocouple reading [$\degree$C]")
    plt.ylabel("Cumulative distribution function")
    plt.show()

    if driver=="voltage":
        return voltage_values,tc_temps,max_temps
    elif driver=="current":
        return current_values,tc_temps,max_temps


def read_exp_data(output_dir,file_name,tc_nums,driver):

    dataset = UQEvalPtsDataset(output_dir,file_name,tc_nums,driver)

    print("TC dataset size:",dataset.tc_all.shape)

    fig,axs = plt.subplots(2)
    if driver=="current":
        fig.suptitle("Current-driven forward UQ")
        axs[0].hist(dataset.coil_current,bins=100,histtype="step")
        axs[0].set_xlabel("Coil current [A]")
    elif driver=="voltage":
        fig.suptitle("Voltage-driven forward UQ")
        axs[0].hist(dataset.coil_voltage,bins=100,histtype="step")
        axs[0].set_xlabel("Coil voltage [V]")
    for ii in range(dataset.tc_all.shape[0]):
        axs[1].hist(dataset.tc_all[ii],bins=100,histtype="step",label=f"TC{dataset.tc_inds[ii]}")
    axs[1].set_xlabel("Temperature [$\degree$C]")
    axs[1].legend()
    fig.tight_layout()
    fig.show()


    return dataset

def read_exp_comb_data(output_dir,file_name,num_tcs,driver):

    dataset = ExpCombDataset(output_dir,file_name,num_tcs)

    print("TC dataset size:",dataset.tc_all.shape)

    plt.figure()
    for ii in range(dataset.tc_all.shape[0]):
        plt.plot(dataset.time,dataset.tc_all[ii],".",label=f"TC{ii+1}")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Thermocouple reading [$\degree$C]")
    plt.show()

    return 0



def mavm(model_data,exp_data,plotRes=None):
    """
    Calculates the Modified Area Validation Metric.
    Adapted from Whiting et al., 2023, "Assessment of Model Validation, Calibration, and Prediction Approaches in the Presence of Uncertainty", Journal of Verification, Validation and Uncertainty Quantification, Vol. 8.
    Downloaded from http://asmedigitalcollection.asme.org/verification/article-pdf/8/1/011001/6974199/vvuq_008_01_011001.pdf on 24 May 2024.
    """

    if plotRes == None:
        plotRes = PLOT_RES

    # find empirical cdf
    model_cdf = stats.ecdf(model_data).cdf
    exp_cdf = stats.ecdf(exp_data).cdf

    if plotRes:
        # plot empirical cdf
        fig,axs=plt.subplots(1,1)
        model_cdf.plot(axs,label="model")
        exp_cdf.plot(axs,label="experiment")
        axs.legend()
        axs.set_xlabel(r"Temperature [$\degree$C]")
        axs.set_ylabel("Probability")
        plt.show()

    F_ = model_cdf.quantiles
    Sn_ = exp_cdf.quantiles


    df = len(Sn_)-1
    t_alph = stats.t.ppf(0.95,df)

    Sn_conf = [Sn_ - t_alph*(np.nanstd(Sn_)/np.sqrt(len(Sn_))),
               Sn_ + t_alph*(np.nanstd(Sn_)/np.sqrt(len(Sn_)))]


    Sn_Y = exp_cdf.probabilities
    F_Y = model_cdf.probabilities


    if plotRes:
        # plot empirical cdf with conf. int. cdfs
        fig,axs=plt.subplots(1,1)
        axs.ecdf(model_cdf.quantiles,label="model")
        axs.ecdf(exp_cdf.quantiles,label="experiment")
        axs.ecdf(Sn_conf[0],ls="dashed",color="k",label="95% C.I.")
        axs.ecdf(Sn_conf[1],ls="dashed",color="k")
        axs.legend()
        axs.set_xlabel(r"Temperature [$\degree$C]")
        axs.set_ylabel("Probability")
        plt.show()


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


    if plotRes:
        plt.figure()
        plt.plot(F_,F_Y,"k-")
        plt.plot(F_+d_plus,F_Y,"k--")
        plt.plot(F_-d_minus,F_Y,"k--")
        plt.fill_betweenx(F_Y,F_-d_minus,F_+d_plus,color="k",alpha=0.2)
        plt.xlabel(r"Temperature [$\degree$C]")
        plt.ylabel("Probability")
        plt.tight_layout()
        plt.show()

    output_dict = {"model_cdf":model_cdf,
                   "exp_cdf":exp_cdf,
                   "d+":d_plus,
                   "d-":d_minus}

    return output_dict

#-------------------------------------------------------------------------------
# LF Code
def load_sim_data(data_path: Path) -> tuple[np.ndarray,np.ndarray]:
    csv_files = list(data_path.glob("*.csv"))
    csv_files = sorted(csv_files)

    data = np.genfromtxt(csv_files[0],dtype=np.float64,delimiter=",")

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
        data = np.genfromtxt(ff,dtype=np.float64,delimiter=",")
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

#-------------------------------------------------------------------------------
def main() -> None:
    #---------------------------------------------------------------------------
    # LF NOTES:
    #---------------------------------------------------------------------------
    # DIC DATA:
    # - 790 time steps
    #
    # FE DATA:
    # - 100 different probabilistic sims
    # - Does not have z coord as it is a plane face
    # - First two columns of FE data are all the same store once

    #---------------------------------------------------------------------------
    print(80*"=")
    print("MAVM Calc for DIC Data")
    print(80*"=")

    OUTPUT_DIR = Path.cwd()
    FE_DIR = Path.cwd()/ "Pulse38_ProbSim_Disp_CamView"
    DIC_DIR = Path.cwd() / "Pulse38_Exp_DIC"

    if not FE_DIR.is_dir():
        raise FileNotFoundError(f"{FE_DIR}: directory does not exist.")
    if not DIC_DIR.is_dir():
        raise FileNotFoundError(f"{DIC_DIR}: directory does not exist.")

    #---------------------------------------------------------------------------
    sim_coord_path = Path.cwd() / "sim_coords.npy"
    sim_disp_path = Path.cwd() / "sim_disp.npy"

    if not sim_coord_path.is_file() and not sim_disp_path.is_file():
        print(f"Loading csv simulation displacement data from:\n{FE_DIR}")
        start_time = time.perf_counter()
        (sim_coords,sim_disp) = load_sim_data(FE_DIR)
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

    #---------------------------------------------------------------------------
    exp_coord_path = Path.cwd() / "exp_coords.npy"
    exp_disp_path = Path.cwd() / "exp_disp.npy"
    exp_strain_path = Path.cwd() / "exp_strain.npy"
    if not exp_coord_path.is_file() and not exp_disp_path.is_file():
        start_time = time.perf_counter()
        print(f"Loading csv experimental displacement data from:\n{DIC_DIR}")
        (exp_coords,exp_disp,exp_strain)= load_exp_data(DIC_DIR,
                                            num_load=None,
                                            run_para=16)
        end_time = time.perf_counter()
        print(f"Loading exp data took: {end_time-start_time}s")


        print(f"Saving numpy arrays in binary format for faster reading...")
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
    return

    print("Reading DIC data...\n")
    dicDisp = DICDisplacement(DIC_DIR)

    print(f"{dicDisp.xct.shape=}")


    av_feDispX = np.nanmean(feDisp.xct[0])
    av_feDispY = np.nanmean(feDisp.yct[0])
    av_dicDispX = np.nanmean(dicDisp.xct[0])
    av_dicDispY = np.nanmean(dicDisp.yct[0])

    # Regularise dic coords
    dicDisp.xct -= av_dicDispX - av_feDispX
    dicDisp.yct -= av_dicDispY - av_feDispY

    plt.figure()
    plt.scatter(dicDisp.xct[0],dicDisp.yct[0],c=dicDisp.xct[1],cmap="plasma")
    plt.axis("equal")
    plt.colorbar(orientation="horizontal",aspect=100,label="xc [m]")
    plt.show()

    plt.figure()
    plt.scatter(feDisp.xct[0],feDisp.yct[0],c=feDisp.xct[1],cmap="plasma")
    plt.axis("equal")
    plt.colorbar(orientation="horizontal",aspect=100,label="xc [m]")
    plt.show()


    # gridify DIC data
    xmin = np.nanmin(dicDisp.xpix[0])
    xmax = np.nanmax(dicDisp.xpix[0])
    ymin = np.nanmin(dicDisp.ypix[0])
    ymax = np.nanmax(dicDisp.ypix[0])
    xdiff = np.nanmin(np.abs([diff for diff in np.diff(dicDisp.xpix[0]) if diff != 0]))
    ydiff = np.nanmin(np.abs([diff for diff in np.diff(dicDisp.ypix[0]) if diff != 0]))


    dicMeshGrid = np.mgrid[xmin:xmax:xdiff, ymin:ymax:ydiff]

    dicMeshData = np.full(dicMeshGrid[0].shape,np.nan)
    dicXCoords = np.full(dicMeshGrid[0].shape,np.nan)
    dicYCoords = np.full(dicMeshGrid[0].shape,np.nan)

    for xx in range(dicMeshGrid.shape[1]):
        for yy in range(dicMeshGrid.shape[2]):
            xmatch = np.where(dicDisp.xpix[0]==dicMeshGrid[0,xx,yy])[0]
            ymatch = np.where(dicDisp.ypix[0]==dicMeshGrid[1,xx,yy])[0]
            index = np.array([ii for ii in xmatch if ii in ymatch])
            if index.size==0:
                pass
            else:
                index=index[0]
                dicMeshData[xx,yy] = dicDisp.dzt[1,index]
                dicXCoords[xx,yy] = dicDisp.xct[0,index]
                dicYCoords[xx,yy] = dicDisp.yct[0,index]

    plt.figure()
    plt.imshow(dicMeshData,cmap="plasma")
    plt.show()

    # gridify FE data
    xmin = np.nanmin(feDisp.xct[0])
    xmax = np.nanmax(feDisp.xct[0])
    ymin = np.nanmin(feDisp.yct[0])
    ymax = np.nanmax(feDisp.yct[0])
    xdiff = np.nanmin(np.abs([diff for diff in np.diff(feDisp.xct[0]) if diff != 0]))
    ydiff = np.nanmin(np.abs([diff for diff in np.diff(feDisp.yct[0]) if diff != 0]))

    feMeshGrid = np.mgrid[xmin:xmax:xdiff, ymin:ymax:ydiff]

    feMeshData = np.full(feMeshGrid[0].shape,np.nan)

    for xx in range(feMeshGrid.shape[1]):
        for yy in range(feMeshGrid.shape[2]):
            xmatch = np.isclose(feDisp.xct[0],feMeshGrid[0,xx,yy])
            ymatch = np.isclose(feDisp.yct[0],feMeshGrid[1,xx,yy])
            index = np.where((xmatch==True) & (ymatch==True))

            index = index[0][0]
            feMeshData[xx,yy] = feDisp.dzt[0,index]

    plt.figure()
    plt.imshow(feMeshData,cmap="plasma")
    plt.show()

    # Try the interp

    feInterpolator = interpolate.RegularGridInterpolator((feMeshGrid[0],feMeshGrid[1]),feMeshData)

    # Remaining: interpolating FE to DIC properly, implementing mavm (which should then be straightforward - see function below.

    return None

if __name__ == "__main__":
    main()

