import copy
import os
import pickle

import h5py
import numpy as np
import logging
import time
from cfd_tools.cartesian_systems.misc_tools import *
from dedalus.tools import post
import socket


logger = logging.getLogger(__name__)

def get_roots():
    # WHEN RUNNING IN MATHS SERVERS
    projects_path_local = os.path.expanduser('~') + '/Documents/projects/'
    data_path_fawcett = os.path.expanduser('~') + '/../../nfs/st01/hpc-fluids-rrk26/tal43/'

    if os.path.exists(projects_path_local):
        core_root = projects_path_local + 'vKolmogorov3D/'
        data_root = projects_path_local + 'vKolmogorov3D/storage/'
    elif os.path.exists(data_path_fawcett):
        core_root = os.path.expanduser('~') + '/projects/vKolmogorov3D/'
        data_root = data_path_fawcett + 'vKolmogorov3D/'
    else:
        core_root = os.path.expanduser('~') + '/projects/vKolmogorov3D/'

    return core_root, data_root

def get_ic_file(material_params, system_params, solver_params, restart=False, suffix='', subdir='', ic_dict_if_reinit=None, **kwargs):

    if 'suffix_name' in kwargs.keys(): suffix = f"recent-{kwargs['suffix_name']}"

    save_folder = get_fpath_sim(material_params, system_params, solver_params, subdir=subdir, suffix=suffix, **kwargs)

    if os.path.exists(save_folder) and not restart:
        post.merge_process_files(save_folder, cleanup=True)
        comm = MPI.COMM_WORLD
        comm.Barrier()
        time.sleep(2)   # to ensure merging is done by the time we get s file
        if len(os.listdir(save_folder)) != 0:
            fname = sorted(os.listdir(save_folder), key=lambda x: int(x.split('_s')[-1][:-3]))[-1]
            ic_file = os.path.join(save_folder, fname)
            noise_coeff = 0
        else:
            ic_file = None
            noise_coeff = 1e-2

    else:
        ic_file = None
        noise_coeff = 1e-2

    if ic_file is None and ic_dict_if_reinit is not None:
        if 'suffix' in ic_dict_if_reinit.keys(): 
            suffix = ic_dict_if_reinit['suffix']
            del ic_dict_if_reinit['suffix']
        if 'subdir' in ic_dict_if_reinit.keys(): 
            subdir = ic_dict_if_reinit['subdir']
            del ic_dict_if_reinit['subdir']
        ic_file, noise_coeff = get_ic_file(material_params, system_params, solver_params, restart=False, closest_made_to_params=False,
                    suffix=suffix, subdir=subdir, ic_dict_if_reinit=None, **ic_dict_if_reinit)
        if 'noise_coeff' in ic_dict_if_reinit.keys(): 
            noise_coeff = ic_dict_if_reinit['noise_coeff']


    return ic_file, noise_coeff


def get_fpath_sim(material_params, system_params, solver_params, suffix='', subdir='', **kwargs):

    params_copy = copy.deepcopy(material_params)
    params_copy.update(system_params)
    params_copy.update(solver_params)

    for param_name, param in kwargs.items():    # overwrite anything in params with kwargs
        params_copy[param_name] = param

    ndim = params_copy['ndim']

    _, data_root = get_roots()

    if ndim == 1:
        Ny = params_copy['Ny']
        name = f"sim_W_{params_copy['W']:.6g}_Re_{params_copy['Re']:.6g}_beta_{params_copy['beta']:.6g}_eps_{params_copy['eps']:.6g}_L_{params_copy['L']:.5g}_Lx_{params_copy['Lx']:.5g}_ndim_{ndim}_N_{Ny}_{suffix}/"
    elif ndim == 2:
        Nx, Ny = params_copy['Nx'], params_copy['Ny']
        name = f"sim_W_{params_copy['W']:.6g}_Re_{params_copy['Re']:.6g}_beta_{params_copy['beta']:.6g}_eps_{params_copy['eps']:.6g}_L_{params_copy['L']:.5g}_Lx_{params_copy['Lx']:.5g}_ndim_{ndim}_N_{Nx}-{Ny}_{suffix}/"
    elif ndim == 3:
        Nx, Ny, Nz = params_copy['Nx'], params_copy['Ny'], params_copy['Nz']
        name = f"sim_W_{params_copy['W']:.6g}_Re_{params_copy['Re']:.6g}_beta_{params_copy['beta']:.6g}_eps_{params_copy['eps']:.6g}_L_{params_copy['L']:.5g}_Lx_{params_copy['Lx']:.5g}_Lz_{params_copy['Lz']:.5g}_ndim_{ndim}_N_{Nx}-{Ny}-{Nz}_{suffix}/"
    else:
        raise Exception

    save_folder = os.path.join(data_root, 'simulations', subdir, name.replace('.', ','))

    return save_folder

####################################################################################################################################
# EVERYTHING UNDER HERE DOESN'T NEED TO CHANGE WHEN A NEW SYSTEM IS MADE
####################################################################################################################################
def get_h5_data(material_params, system_params, solver_params, suffix='', subdir='', s=-1):

    fpath = get_fpath_sim(material_params, system_params, solver_params, suffix=suffix, subdir=subdir)

    data_fields, data_metric = get_h5_data_from_fpath(fpath, s)

    return data_fields, data_metric

def get_metric_from_params(material_params, system_params, solver_params, suffix, subdir, metric='trace', deviation=True):

    fpath = get_fpath_sim(material_params, system_params, solver_params, suffix=suffix, subdir=subdir)
    t_all, metric_all = get_metric_from_fpath(fpath, metric=metric, deviation=deviation)

    return t_all, metric_all

def get_s_list(material_params, system_params, solver_params, suffix='', subdir=''):

    fpath = get_fpath_sim(material_params, system_params, solver_params, suffix=suffix, subdir=subdir)

    get_s_list_from_fpath(fpath)


def get_metric_from_fpath(fpath, metric='trace', deviation=True):

    s_list = get_s_list_from_fpath(fpath)
    t, metric = get_metric_from_fpath_and_s_list(fpath, s_list, metric=metric, deviation=deviation)

    return t, metric
