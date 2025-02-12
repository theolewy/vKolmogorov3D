import copy
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import logging
from cfd_tools.cartesian_systems.plotter import *

import scipy
from scipy.interpolate import CubicSpline

from tools.solvers.kolmogorov import BaseFlow

plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['font.size'] = 14

logger = logging.getLogger(__name__)

from .misc_tools import *


def plot_base_flow(BaseFlow, fname='base_flow', field_names=['u', 'v']):
    BaseFlow._set_scale(1)
    y_grid = BaseFlow.y

    num_fields = len(field_names)
    num_columns = (num_fields + 1) // 2 

    fig, axs = plt.subplots(2, num_columns, figsize=(13, 9))

    for i, field_name in enumerate(field_names):
        row = 0 if i < num_columns else 1
        column = i % num_columns

        field = getattr(BaseFlow, field_name)
        axs[row, column].plot(field['g'].real, y_grid)
        axs[row, column].set(ylabel='y')
        axs[row, column].set(xlabel=field_name)
        
    fig.suptitle('Base Flow' + str(BaseFlow.material_params), wrap=True)

    plt.tight_layout()

    core_root, _ = get_roots()

    images_dir = os.path.join(core_root, 'images', 'eigenplots')

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    plt.savefig(os.path.join(images_dir, f'{fname}.png'))

    plt.close()

def eigenplots(fname, EVP):
    core_root, _ = get_roots()
    images_dir = os.path.join(core_root, 'images', 'eigenplots')

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    fig = EVP.problem.plot_mode(EVP.index)
    fig.suptitle(str(EVP.material_params))
    fig.tight_layout()
    fig.savefig(fname=os.path.join(images_dir, 'eigenfunctions_' + fname + '.png'))
    plt.close()

    plt.scatter(EVP.problem.evalues.real, EVP.material_params['kx'] * EVP.problem.evalues.imag, color='red')
    plt.axhline(0, color='k', alpha=0.2)
    plt.xlabel(r"$c_r$")
    plt.ylabel(r"$k_x c_i$")
    plt.title(str(EVP.material_params), wrap=True)
    plt.tight_layout()

    plt.savefig(os.path.join(images_dir, 'eigenvalues_' + fname + '.png'))

    plt.close()

def plot_2D_efunction(solver, c, aspect=2, **kwargs):

    kx = solver.params['kx']
    rheology = solver.rheology
    emode = solver.EVP_solver.problem.eigenmode(solver.EVP_solver.index)
    psi = emode.fields[1]['g'] / kx
    t11 = emode.fields[2]['g']
    t22 = emode.fields[4]['g']
    trace = t11 + t22

    y = solver.EVP_solver.y

    if 'oneD' in kwargs.keys() and ((kwargs['oneD'] == 'auto' and kx < 1e-4) or kwargs['oneD'] == True):
        oneD_mode = True
        kx = 0
    else:
        oneD_mode = False

    if oneD_mode:
        x = np.linspace(0, 2 * np.pi, 100)
    else:
        x = np.linspace(0, 2 * np.pi / kx, 100)

    xx, yy = np.meshgrid(x, y, indexing='ij')

    psi, trace = np.expand_dims(psi, axis=0), np.expand_dims(trace, axis=0)

    x, y = np.expand_dims(x, axis=1), np.expand_dims(y, axis=0)

    psi = psi * np.exp(1j * kx * x)
    trace = trace * np.exp(1j * kx * x)

    trace = trace.real
    psi = psi.real

    x, y = x.flatten(), y.flatten()

    plt.figure()
    plt.axes().set_aspect(aspect)

    # plt.colorbar()

    plt.xticks([0, 2 * np.pi], [0, r"$2\pi$"])
    plt.ylim([-1, 1])
    plt.xlim([0, 2 * np.pi])
    plt.ylabel(r"$y$")


    if oneD_mode:
        plt.xlabel(r"$x$")
        plt.pcolormesh(xx, yy, trace, cmap='inferno', shading='gouraud',
                       vmin=trace.min(), vmax=trace.max())

        plt.contour(x, y, psi.T, cmap='Greys', alpha=0.8, levels=3)
        plt.text(x=x[-1] * 0.45, y=1.08, s=f"c={c:.4f}", fontsize=16, color='black')
        marker_loc = x[-1] * -0.14


    else:
        plt.xlabel(r"$kx$")
        plt.pcolormesh(xx * kx, yy, trace, cmap='inferno', shading='gouraud',
                       vmin=trace.min(), vmax=trace.max())

        plt.contour(x * kx, y, psi.T, cmap='Greys', alpha=0.8, levels=4)
        plt.text(x=x[-1] * kx * 0.55, y=1.08, s=f"c={c:.4f}", fontsize=16, color='black')
        marker_loc = x[-1] * kx * -0.14

    if 'marker' in kwargs.keys() and kwargs['marker'] is not None:
        color, mark = kwargs['marker'][0], kwargs['marker'][1]
        plt.scatter([marker_loc], [0.8], s=400, marker=mark, color=color, zorder=5, clip_on=False)

    plt.tight_layout()

    kx, W, eps, delta, a, Z = solver.params['kx'], solver.params['W'], solver.params['eps'], solver.params['delta'], solver.params['a'], solver.params['Z']

    if rheology == 'DJS':
        plt.savefig(fname=f'images/eigenfunctions_2D/eigenfunctions_2D_W_{W}_a_{a}_delta_{delta}_kx_{kx}_eps_{eps}.png', bbox_inches='tight')
    else:
        plt.savefig(fname=f'images/eigenfunctions_2D/eigenfunctions_2D_W_{W}_Z_{Z}_delta_{delta}_kx_{kx}_eps_{eps}.png', bbox_inches='tight')

    plt.close()


def plot_metric_from_params(material_params, system_params, solver_params, suffix, subdir='', metric='trace',
                            deviation=True):
    t, metric_list = get_metric_from_params(material_params, system_params, solver_params, suffix, subdir=subdir,
                                            metric=metric, deviation=deviation)

    ylabel = metric if not deviation else f'{metric}_deviation'

    core_root, data_root = get_roots()
    plot_metric(np.abs(metric_list), t, fname='', ylabel=ylabel, save=False, core_root=core_root)
    return t, metric_list


def plot_snap_from_params(material_params, system_params, solver_params, suffix, subdir='', fname='', field='trace',
                          save=False, title=True, cb=True, cmin=None, cmax=None, s=-1, deviation=False):

    data_fields, data_metric = get_h5_data(material_params, system_params, solver_params, suffix=suffix, subdir=subdir,
                                           s=s)

    field = data_fields[field]

    z, r, t = data_fields['z'], data_fields['r'], data_fields['t']

    if system_params['ndim'] == 1:       # i.e. 1D plot
        z = np.array([0, 2*np.pi])
        field = np.repeat(np.expand_dims(field, axis=1), repeats=2, axis=1)
    elif system_params['ndim'] == 3:

        field = field[:, 20, :, :]   # fix theta

    # u = data_fields['u'][-1, :, :]
    # v = data_fields['v'][-1, :, :]
    # psi = scipy.integrate.cumtrapz(y=u, x=y, axis=1, initial=0)

    plot_from_array(field[-1], z, r, det_C=None, fname=fname, subdirectory='', cb=cb, save=save, cmin=cmin, cmax=cmax)
    # plt.contour(x, y, psi.T, cmap='Greys', alpha=0.8)

    logger.info(f"Note, this run had resolution {z.shape[0]} x {r.shape[0]} and was from t={data_fields['t'][-1]}")

    if title:
        plt.title(f"Nz = {z.shape[0]}, Nr = {r.shape[0]}, t = {data_fields['t'][-1]:.4g}")


def check_localised(W, eps, beta, L, Re, Lx, Lz,  Nx, Ny, Nz, suffix='', subdir=''):
    
    material_params = {'W': W, 'beta': beta, 'Re': Re, 'L':L, 'eps': eps}
    system_params = {'system_type': 'channel', 'df':'m', 'Lx': Lx,  'Lz': Lz, 'ndim': 3, 'n':1}
    solver_params = {'Nx': Nx, 'Ny': Ny, 'Nz': Nz}

    fpath = get_fpath_sim(material_params, system_params, solver_params, suffix=suffix, subdir=subdir)
    post.merge_process_files(fpath, cleanup=True)

    logger.info("Now getting base flow...")
    base_solver = BaseFlow(solver_params=solver_params, system_params=system_params)
    base_flow = base_solver.ensure_converged_base(material_params=material_params, logger_on=True)
    logger.info("Obtained base flow...")

    data_fields, _ = get_h5_data(material_params, system_params, solver_params, suffix=suffix, subdir=subdir, s=-1)

    x, y, z = data_fields['x'], data_fields['y'], data_fields['z']

    fields = ['p', 'c11', 'c12', 'c22', 'u', 'v']

    for field_name in fields:
        base_field = base_flow[field_name]
        field_array = data_fields[field_name][-1,:,:,:] - base_field[None, :, None]
        field_int = np.max(np.abs(field_array), axis=(0,2))
        # field_int /= np.max(field_int)

        plt.plot(z, field_int)
        
    plt.legend(fields)

    plt.xlabel('z')
    plt.ylabel(r'$max_{x,y}(f)/max_{x,y,z}(f)$')

    core_root, _ = get_roots()
    fpath = os.path.join(core_root, 'images', f'localisation_W_{W}_eps_{eps}_L_{L}_Re_{Re}_beta_{beta}_Lx_{Lx}_Lz_{Lz}_Nx_{Nx}_Ny_{Ny}_Nz_{Nz}.jpg')
    plt.savefig(fpath, bbox_inches='tight')
    plt.close()
    