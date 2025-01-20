import copy
import os

import pickle
import posixpath
import signal
import time

import numpy as np
from dedalus import public as de
from dedalus.extras import flow_tools
from mpi4py import MPI
from dedalus.tools import post

from eigentools import Eigenproblem
import matplotlib.pyplot as plt

import logging
from cfd_tools.cartesian_systems.cartesian_system_base import CartesianBaseFlow, CartesianEVP, CartesianNumericSolver, CartesianTimeStepper
from scipy.sparse.linalg import ArpackNoConvergence

from tools.plotter import *
from tools.misc_tools import *


logger = logging.getLogger(__name__)


class BaseFlow(CartesianBaseFlow):

    def __init__(self, solver_params, system_params, **kwargs):
        super().__init__(solver_params, system_params, **kwargs)

    def _set_vars_coords_param_names(self):

        self.variables = ['u', 'v', 'w', 'p', 'c11', 'c12', 'c22', 'c33', 'c13', 'c23',
                           'uy', 'vy', 'wy', 'c11y', 'c12y', 'c22y', 'c33y', 'c13y', 'c23y']
        self.material_param_names = ['Re', 'W', 'eps', 'beta', 'L']
        self.params_cause_base_to_vary = ['W', 'eps', 'beta', 'L']
        self.inhomo_coord_name = 'y'
        self.homo_coord_names = ['x', 'z']
        
    def _build_domain(self, **kwargs):

        self.y_basis = de.Chebyshev(self.inhomo_coord_name, self.Ny,
                                    interval=(-np.pi*self.n, np.pi*self.n),
                                    dealias=1.5)
        
        comm = kwargs['comm'] if 'comm' in kwargs.keys() else None
        self.domain = de.Domain([self.y_basis], grid_dtype=float, comm=comm)

        setattr(self, self.inhomo_coord_name, self.domain.grid(0, scales=1))    # self.y
        setattr(self, f"{self.inhomo_coord_name}_dealias", self.domain.grid(0, scales=1.5)) # self.y_dealias

    def _set_system_specific_substitutions(self):
        self.problem.substitutions['F'] = "(1 + eps * beta * W) / (1 + eps * W) * cos(y)"
        
    def _continue_base_from_simple_params(self, material_params):
            # probably worth improve this at some point if need be
            n_steps = 20
            W_end = material_params['W']
            if self.current_base_material_params is not None:    # we currently have an accurate loaded base state
                W_initial = self.current_base_material_params['W']
                W_list = np.logspace(np.log10(W_initial), np.log10(W_end), n_steps + 1)[1:]
            else:   # we don't currently have an accurate loaded base state
                W_initial = 2
                W_list = np.logspace(np.log10(W_initial), np.log10(W_end), n_steps + 1)

            for W in W_list:
                material_params['W'] = W
                base_flow, info = self.converge_base_for_params(material_params, logger_on=False)
                if info['failed']: raise Exception("Continuation from easy parameters didn't work")
            
            return base_flow

    def _equations(self):
        

        self.problem.add_equation('Re * dt(u) + dx(p) - beta * lap(u, uy) = F - Re * adv(u) + (1 - beta) * (t11x + t12y + t13z)')
        self.problem.add_equation('Re * dt(v) + dy(p) - beta * lap(v, vy) =   - Re * adv(v) + (1 - beta) * (t12x + t22y + t23z)')
        self.problem.add_equation('Re * dt(w) + dz(p) - beta * lap(w, wy) =   - Re * adv(w) + (1 - beta) * (t13x + t23y + t33z)')

        self.problem.add_equation('dt(c11) - eps * lap(c11, c11y) = ' + \
                                  '- adv(c11)' + \
                                  '+ 2 * c11 * dx(u) + 2 * c12 * dy(u) + 2 * c13 * dz(u)' + \
                                  '- t11')

        self.problem.add_equation('dt(c12) - eps * lap(c12, c12y) = ' + \
                                  '- adv(c12)' + \
                                  '+ c11 * dx(v) + c12 * dy(v) + c13 * dz(v)' + \
                                  '+ c12 * dx(u) + c22 * dy(u) + c23 * dz(u)' + \
                                  '- t12')

        self.problem.add_equation('dt(c22) - eps * lap(c22, c22y) = ' + \
                                  '- adv(c22)' + \
                                  '+ 2 * c12 * dx(v) + 2 * c22 * dy(v) + 2 * c23 * dz(v)' + \
                                  '- t22')
        
        self.problem.add_equation('dt(c13) - eps * lap(c13, c13y) = ' + \
                                  '- adv(c13)' + \
                                  '+ c11 * dx(w) + c12 * dy(w) + c13 * dz(w)' + \
                                  '+ c13 * dx(u) + c23 * dy(u) + c33 * dz(u)' + \
                                  '- t13')
        
        self.problem.add_equation('dt(c23) - eps * lap(c23, c23y) = ' + \
                                  '- adv(c23)' + \
                                  '+ c12 * dx(w) + c22 * dy(w) + c23 * dz(w)' + \
                                  '+ c13 * dx(v) + c23 * dy(v) + c33 * dz(v)' + \
                                  '- t23')
        
        self.problem.add_equation('dt(c33) - eps * lap(c33, c33y) = ' + \
                                  '- adv(c33)' + \
                                  '+ 2 * c13 * dx(w) + 2 * c23 * dy(w) + 2 * c33 * dz(w)' + \
                                  ' - t33')

        self.problem.add_equation('dx(u) + vy + dz(w) = 0')

        # first derivatives

        self.problem.add_equation('dy(u) - uy = 0')
        self.problem.add_equation('dy(v) - vy = 0')
        self.problem.add_equation('dy(w) - wy = 0')
        self.problem.add_equation('dy(c11) - c11y = 0')
        self.problem.add_equation('dy(c12) - c12y = 0')
        self.problem.add_equation('dy(c22) - c22y = 0')
        self.problem.add_equation('dy(c33) - c33y = 0')
        self.problem.add_equation('dy(c13) - c13y = 0')
        self.problem.add_equation('dy(c23) - c23y = 0')

        self.problem.add_bc('left(c11) - right(c11) = 0')
        self.problem.add_bc('left(c12) - right(c12) = 0')
        self.problem.add_bc('left(c22) - right(c22) = 0')
        self.problem.add_bc('left(c33) - right(c33) = 0')
        self.problem.add_bc('left(c13) - right(c13) = 0')
        self.problem.add_bc('left(c23) - right(c23) = 0')
        self.problem.add_bc('left(c11y) - right(c11y) = 0')
        self.problem.add_bc('left(c12y) - right(c12y) = 0')
        self.problem.add_bc('left(c22y) - right(c22y) = 0')
        self.problem.add_bc('left(c33y) - right(c33y) = 0')
        self.problem.add_bc('left(c13y) - right(c13y) = 0')
        self.problem.add_bc('left(c23y) - right(c23y) = 0')

        self.problem.add_bc('left(v) = 0')
        self.problem.add_bc('left(p) = 0')

        # NB when solved analytically, we get U = cosy + Ay + B. Set bc of U to ensure A=B=0
        self.problem.add_bc('left(u) = -1')
        self.problem.add_bc('right(u) = -1')

        self.problem.add_bc('left(w)  = 0')
        self.problem.add_bc('right(w)  = 0')


    def _guess_base(self):

        self._set_scale(1)

        # Method 1: Guess something close
        W, eps = self.W, self.eps
        y = self.y

        self.u['g'] = np.cos(y)
        self.v['g'] = np.zeros_like(y)
        self.w['g'] = np.zeros_like(y)
        self.p['g'] = np.zeros_like(y)

        self.c11['g'] = W ** 2 / (1 + eps * W) / (1 + 4 * eps * W) * (4 * eps * W + 1 - np.cos(2 * y)) + 1
        self.c12['g'] = - W / (1 + eps * W) * np.sin(y)
        self.c22['g'] = 1
        self.c33['g'] = 1

        self.u.differentiate('y', out=self.uy)
        self.c11.differentiate('y', out=self.c11y)
        self.c12.differentiate('y', out=self.c12y)
        self.c22.differentiate('y', out=self.c22y)
        self.c33.differentiate('y', out=self.c33y)

    def plot_base_state(self, fname='base_flow', field_names=['u', 'v', 'c11', 'c12', 'c22', 'p']):
        plot_base_flow(self, fname=fname, field_names=field_names)

class EVP(CartesianEVP):
    
    def __init__(self, solver_params, system_params, **kwargs):
        super().__init__(solver_params, system_params, **kwargs)
    
    def _set_vars_coords_param_names(self):

        self.variables = ['u', 'v', 'p', 'c11', 'c12', 'c22', 'c33',
                           'uy', 'vy', 'c11y', 'c12y', 'c22y', 'c33y']
        self.material_param_names = ['Re', 'W', 'eps', 'beta', 'L', 'kx']
        self.inhomo_coord_name = 'y'
        self.homo_coord_names = ['x', 'z']
        

    def _build_domain(self, **kwargs):

        self.y_basis = de.Chebyshev(self.inhomo_coord_name, self.Ny,
                                    interval=(-np.pi*self.n, np.pi*self.n),
                                    dealias=1.5)

        comm = kwargs['comm'] if 'comm' in kwargs.keys() else None
        self.domain = de.Domain([self.y_basis], grid_dtype=complex, comm=comm)

        # e.g. self.y
        setattr(self, self.inhomo_coord_name, self.domain.grid(0, scales=1))    
        # e.g. self.y_dealias
        setattr(self, f"{self.inhomo_coord_name}_dealias", self.domain.grid(0, scales=1.5))

    def _set_system_specific_substitutions(self):
        pass

    def _equations(self):

        # momentum equation
        self.problem.add_equation('Re * (dt(u) + U*dx(u) + Uy * v) + dx(p) - beta * lap(u, uy) -  (1 - beta) * (t11x + t12y) = 0')
        self.problem.add_equation('Re * (dt(v) + U*dx(v)) + dy(p) - beta * lap(v, vy) -  (1 - beta) * (t12x + t22y) = 0')

        # incompressibility
        self.problem.add_equation('dx(u) + vy = 0')

        # conformation equation
        self.problem.add_equation('dt(c11) + t11 - eps * lap(c11, c11y) = ' + \
                                    '- U * dx(c11) - v * C11y' + \
                                    '+ 2 * C11 * dx(u) + 2 * c12 * Uy +  2 * C12 * dy(u)')

        self.problem.add_equation('dt(c12) + t12 - eps * lap(c12, c12y) = ' + \
                                    '- U * dx(c12) - v * C12y' + \
                                    '+ C11 * dx(v) + C22 * dy(u) + c22 * Uy')

        self.problem.add_equation('dt(c22) + t22 - eps * lap(c22, c22y) = ' + \
                                    '- U * dx(c22) - v * C22y' + \
                                    '+ 2 * C12 * dx(v) + 2 * C22 * dy(v)')

        self.problem.add_equation('dt(c33) + t33 - eps * lap(c33, c33y) - 0 = ' + \
                                    '- U * dx(c33) - v * C33y' + \
                                    '+ 0')

        # first derivatives
        self.problem.add_equation('uy - dy(u) = 0')
        self.problem.add_equation('vy - dy(v) = 0')
        self.problem.add_equation('c11y - dy(c11) = 0')
        self.problem.add_equation('c12y - dy(c12) = 0')
        self.problem.add_equation('c22y - dy(c22) = 0')
        self.problem.add_equation('c33y - dy(c33) = 0')

        # periodic boundary conditions
        self.problem.add_bc('left(u) - right(u) = 0')
        self.problem.add_bc('left(uy) - right(uy) = 0')
        self.problem.add_bc('left(v) - right(v) = 0')
        self.problem.add_bc('left(p) - right(p) = 0')
        self.problem.add_bc('left(c11) - right(c11) = 0')
        self.problem.add_bc('left(c11y) - right(c11y) = 0')
        self.problem.add_bc('left(c12) - right(c12) = 0')
        self.problem.add_bc('left(c12y) - right(c12y) = 0')
        self.problem.add_bc('left(c22) - right(c22) = 0')
        self.problem.add_bc('left(c22y) - right(c22y) = 0')
        self.problem.add_bc('left(c33) - right(c33) = 0')
        self.problem.add_bc('left(c33y) - right(c33y) = 0')

class NumericSolver(CartesianNumericSolver):

    def __init__(self, system_params, solver_params, 
                 save_plots=False, **kwargs):
        super().__init__(system_params, solver_params, 
                 save_plots=False, **kwargs)
        
    def _set_solvers(self, **kwargs):

        self.base_solver = BaseFlow(self.solver_params, self.system_params, **kwargs)
        self.EVP_solver = EVP(self.solver_params, self.system_params, **kwargs)

    def plot_key_images(self, fname):
        eigenplots(fname, self.EVP_solver)
        self.base_solver.plot_base_state(fname)

class TimeStepper3D(CartesianTimeStepper):

    def __init__(self, material_params, system_params, solver_params, logger_on=True, **kwargs):
        super().__init__(material_params, system_params, solver_params, logger_on=logger_on, **kwargs)
        self.core_root, self.data_root = get_roots()

    def _set_solver(self):
        self.numeric_solver = NumericSolver(system_params=self.system_params, solver_params=self.solver_params, comm=MPI.COMM_SELF)

    def _set_low_dim_timestepper(self, ndim_low):
        system_params_low_dim = copy.deepcopy(self.system_params)
        system_params_low_dim['ndim'] = ndim_low
        self.low_dim_timestepper = TimeStepper3D(material_params=self.material_params, system_params=system_params_low_dim, solver_params=self.solver_params)

    def _set_vars_coords_param_names(self):

        self.variables = ['u', 'v', 'w', 'p', 'c11', 'c12', 'c22', 'c33', 'c13', 'c23']
        self.noise_on_variables = ['u', 'v', 'c11', 'c12']  # use psi to denote 2D streamfunction
        self.material_param_names = ['Re', 'W', 'eps', 'beta', 'L']
        self.inhomo_coord_name = 'y'
        self.homo_coord_names = ['x', 'z']

    def build_domain(self, **kwargs):

        if self.ndim == 3:

            self.x_basis = de.Fourier('x', self.Nx, interval=(0, self.Lx), dealias=3 / 2)
            self.y_basis = de.Fourier('y', self.Ny, interval=(-np.pi*self.n, np.pi*self.n), dealias=3 / 2)
            self.z_basis = de.Fourier('z', self.Nz, interval=(-self.Lz/2, self.Lz/2), dealias=3 / 2)
            self.domain = de.Domain([self.x_basis, self.z_basis, self.y_basis], grid_dtype=np.float64)
            self.x = self.domain.grid(0)
            self.z = self.domain.grid(1)
            self.y = self.domain.grid(2)
            self.y_dealias = self.domain.grid(2, scales=3 / 2)
            self.area = 2 * np.pi * self.n * self.Lx * self.Lz

        elif self.ndim == 2:

            self.x_basis = de.Fourier('x', self.Nx, interval=(0, self.Lx), dealias=3 / 2)
            self.y_basis = de.Fourier('y', self.Ny, interval=(-np.pi*self.n, np.pi*self.n), dealias=3 / 2)
            self.domain = de.Domain([self.x_basis, self.y_basis], grid_dtype=np.float64)
            self.x = self.domain.grid(0)
            self.y = self.domain.grid(1)
            self.y_dealias = self.domain.grid(1, scales=3 / 2)
            self.area = 2 * np.pi * self.n * self.Lx

        else:

            raise Exception("ndim must be 1 or 2")

    def _set_system_specific_substitutions(self):

        self.problem.substitutions['F'] = "(1 + eps * beta * W) / (1 + eps * W) * cos(y)"
        self.problem.substitutions['uy'] = "dy(u)"
        self.problem.substitutions['vy'] = "dy(v)"
        self.problem.substitutions['wy'] = "dy(w)"
        self.problem.substitutions['c11y'] = "dy(c11)"
        self.problem.substitutions['c22y'] = "dy(c22)"
        self.problem.substitutions['c33y'] = "dy(c33)"
        self.problem.substitutions['c12y'] = "dy(c12)"
        self.problem.substitutions['c13y'] = "dy(c13)"
        self.problem.substitutions['c23y'] = "dy(c23)"

        self.problem.substitutions['dt_dash(A)'] = "dt(A) - c*dx(A)"

    def equations(self):

        self.problem.add_equation('Re * dt_dash(u) + dx(p) - beta * lap(u, uy) = F - Re * adv(u) + (1 - beta) * (t11x + t12y + t13z)')
        self.problem.add_equation('Re * dt_dash(v) + dy(p) - beta * lap(v, vy) =   - Re * adv(v) + (1 - beta) * (t12x + t22y + t23z)')
        self.problem.add_equation('Re * dt_dash(w) + dz(p) - beta * lap(w, wy) =   - Re * adv(w) + (1 - beta) * (t13x + t23y + t33z)')

        self.problem.add_equation('dt_dash(c11) - eps * lap(c11, c11y) = ' + \
                                  '- adv(c11)' + \
                                  '+ 2 * c11 * dx(u) + 2 * c12 * dy(u) + 2 * c13 * dz(u)' + \
                                  '- t11')

        self.problem.add_equation('dt_dash(c12) - eps * lap(c12, c12y) = ' + \
                                  '- adv(c12)' + \
                                  '+ c11 * dx(v) + c12 * dy(v) + c13 * dz(v)' + \
                                  '+ c12 * dx(u) + c22 * dy(u) + c23 * dz(u)' + \
                                  '- t12')

        self.problem.add_equation('dt_dash(c22) - eps * lap(c22, c22y) = ' + \
                                  '- adv(c22)' + \
                                  '+ 2 * c12 * dx(v) + 2 * c22 * dy(v) + 2 * c23 * dz(v)' + \
                                  '- t22')
        
        self.problem.add_equation('dt_dash(c13) - eps * lap(c13, c13y) = ' + \
                                  '- adv(c13)' + \
                                  '+ c11 * dx(w) + c12 * dy(w) + c13 * dz(w)' + \
                                  '+ c13 * dx(u) + c23 * dy(u) + c33 * dz(u)' + \
                                  '- t13')
        
        self.problem.add_equation('dt_dash(c23) - eps * lap(c23, c23y) = ' + \
                                  '- adv(c23)' + \
                                  '+ c12 * dx(w) + c22 * dy(w) + c23 * dz(w)' + \
                                  '+ c13 * dx(v) + c23 * dy(v) + c33 * dz(v)' + \
                                  '- t23')
        
        self.problem.add_equation('dt_dash(c33) - eps * lap(c33, c33y) = ' + \
                                  '- adv(c33)' + \
                                  '+ 2 * c13 * dx(w) + 2 * c23 * dy(w) + 2 * c33 * dz(w)' + \
                                  ' - t33')

        if self.ndim == 3:
            self.problem.add_equation('dx(u) + dy(v) + dz(w) = 0', condition=('nx!=0 or ny!=0 or nz!=0'))
            self.problem.add_equation('p = 0', condition=('nx==0 and ny==0 and nz==0'))
        elif self.ndim == 2:
            self.problem.add_equation('dx(u) + dy(v) + dz(w) = 0', condition=('nx!=0 or ny!=0'))
            self.problem.add_equation('p = 0', condition=('nx==0 and ny==0'))


    def plot_snaps(self, subdirectory='', suffix_end='', plot_dev=True):

        flow = self.get_flow(scale=1, combine_processes=True)
        
        base_flow = self.base_flow_full
        # base_flow=None

        if on_local_device():
            fname=f"snaps3D_iter_{self.solver.iteration}"
        else:
            fname=f"snaps3D_W_{self.W}_Re_{self.Re}_eps_{self.eps}_beta_{self.beta}_L_{self.L}_Lx_{self.Lx:.4g}_Nx_{self.Nx}_Ny_{self.Ny}_{suffix_end}".replace('.', ',')

        if self.ndim == 1:
            raise Exception("Made 1D plotting possible")
        elif self.ndim == 2:
            self.plot_snaps_2D(flow=flow, base_flow=base_flow, subdirectory=subdirectory, plot_dev=plot_dev, fname=fname)
        elif self.ndim == 3:
            self.plot_snaps_3D(flow=flow, base_flow=base_flow, subdirectory=subdirectory, plot_dev=plot_dev, fname=fname)

        self.set_scale(1.5)
    
    def plot_metrics(self, subdirectory='', suffix_end=''):

        fname = 'norm' if on_local_device() else f"norm_W_{self.W}_Re_{self.Re}_eps_{self.eps}_beta_{self.beta}_L_{self.L}_Lx_{self.Lx:.4g}_Nx_{self.Nx}_Ny_{self.Ny}_{suffix_end}".replace('.', ',')
        fpath = os.path.join(self.core_root, 'images', 'simulations', subdirectory)
        os.makedirs(fpath, exist_ok=True)
        fpath = os.path.join(fpath, fname)
        
        plot_metrics(np.abs(self.v_metric_list), np.abs(self.KE_metric_list), np.abs(self.trace_metric_list),
                        '|v|_dev', 'KE_dev', 'trace_dev', 
                        self.time_list, self.material_params, fpath)

    def _enforce_symmetry(self):

        odd_fields_y = ['v', 'c12', 'c23']
        even_fields_y = ['u', 'w', 'c11', 'c22', 'c33', 'p', 'c13']

        # y is Fourier, so freq are [0, 1, 2, ... , -2, -1]. Even means k(1) = k(-1) and odd means k(1) = -k(-1)
        for field_name in odd_fields_y:
            field = getattr(self, field_name)
            field['c'][:,:,1:] = (field['c'][:,:,1:] - field['c'][:,:,1:][:,:,::-1]) / 2
            field['c'][:,:,0] = 0

        for field_name in even_fields_y:
            field = getattr(self, field_name)
            field['c'][:,:,1:] = (field['c'][:,:,1:] + field['c'][:,:,1:][:,:,::-1]) / 2
    
        odd_fields_z = ['w', 'c13', 'c23']
        even_fields_z = ['u', 'v', 'c11', 'c12', 'c22', 'c33', 'p']

        # z is Fourier, so freq are [0, 1, 2, ... , -2, -1]. Even means k(1) = k(-1) and odd means k(1) = -k(-1)
        for field_name in odd_fields_z:
            field = getattr(self, field_name)
            field['c'][:,1:,:] = (field['c'][:,1:,:] - field['c'][:,1:,:][:,::-1,:1]) / 2
        
        for field_name in even_fields_z:
            field = getattr(self, field_name)
            field['c'][:,1:,:] = (field['c'][:,1:,:] + field['c'][:,1:,:][:,::-1,:]) / 2


    def simulate(self, T=np.infty, ifreq=200, converge_cadence=None, convergence_limit=1e-4,
                 end_converge=False, end_laminar=False, end_laminar_threshold=1e-6, 
                 plot=True, plot_dev=True, plot_subdirectory="",
                 save_over_long=False, save_full_data=False, suffix_end='', save_subdir='',  **kwargs):

        # save infrequent data for a long period of time
        long_time_folder = self.save_on_long(save_over_long, suffix_end, save_subdir, **kwargs)
        # save all data 
        full_save_folder = self.save_all_data(save_full_data, suffix_end, **kwargs)

        if not on_local_device(**kwargs):
            overwriting_handler = self.add_tasks(save_freq='recent', suffix=f"recent-{suffix_end}", subdir=save_subdir, save_all_fields=True, 
                            mode='overwrite')
            logger.info(f"Most recent h5 file task is saved at {'/'.join(self.save_folder.split('/')[7:])}")

        if 'enforce_symmetry' in kwargs.keys() and kwargs['enforce_symmetry']: logger.info("ENFORCING SYMMETRY")

        self.trace_metric_list = []
        self.KE_metric_list = []
        self.v_metric_list = []
        self.time_list = []

        logger.info('Starting loop')
        self.start_sim_time = self.solver.sim_time
        self.solver.stop_wall_time = 60 * 60 * 11   # 11 hours

        if 'track_TW' in kwargs.keys() and kwargs['track_TW']: 
            self.x_track = self.start_x_track = self._get_arrow_junction()
            if on_local_device(): plot=False

        stop = False

        while self.solver.ok and not stop:
            
            # so that tasks continually overwrite a single h5 file
            if not on_local_device(**kwargs): overwriting_handler.set_num, overwriting_handler.file_write_num = 1, 0  

            self.solver.step(dt=self.dt)

            if self.try_converge(converge_cadence):
                success = self.attempt_to_converge_state()
                self.plot_snaps(subdirectory=plot_subdirectory)
                if success:
                    stop = 'converged'
                    break

            if 'enforce_symmetry' in kwargs.keys() and kwargs['enforce_symmetry'] and self.solver.iteration % 10 == 0:
                self._enforce_symmetry()
            if 'track_TW' in kwargs.keys() and kwargs['track_TW'] and self.solver.iteration % 500 == 0:
                self._track_TW()

            if self.solver.iteration % ifreq == 0:
                KE, self.KE_base = self.flow.volume_average('KE'), self.flow.volume_average('KE_base')
                KE_metric = (KE - self.KE_base) / self.KE_base
                trace, self.trace_base = self.flow.volume_average('trace'), self.flow.volume_average('trace_base')
                trace_metric = (trace - self.trace_base) / self.trace_base
                v, self.v_base = self.flow.volume_average('|v|'), self.flow.volume_average('|V|')
                v_metric = (v - self.v_base) / self.v_base if not np.isclose(self.v_base, 0) else v

                logger.info('It: %i, t: %.3e, dt: %.2e trace_norm: %.4e, min(det(C)): %.4e, KE_norm: %.4e' \
                            % (self.solver.iteration, self.solver.sim_time, self.dt,
                               trace_metric, self.flow.min('det(C)'),
                               KE_metric))
                
                self.trace_metric_list.append(trace_metric)
                self.KE_metric_list.append(KE_metric)
                self.v_metric_list.append(v_metric)
                self.time_list.append(self.solver.sim_time)

                stop = self.stop_simulation_conditions(self.trace_metric_list, self.KE_metric_list, self.time_list, T, convergence_limit,
                                                    end_converge=end_converge, end_laminar=end_laminar, end_laminar_threshold=end_laminar_threshold)
                if plot == True:                
                    subdirectory = plot_subdirectory if on_local_device() else f"HPC_{plot_subdirectory}"
                    self.plot_snaps(subdirectory=subdirectory, suffix_end=suffix_end, plot_dev=plot_dev)
                    self.plot_metrics(subdirectory=subdirectory, suffix_end=suffix_end)


        if stop == 'error':
            return stop

        if save_over_long:
            post.merge_process_files(long_time_folder, cleanup=True)
        if save_full_data:
            post.merge_process_files(full_save_folder, cleanup=True)

        return stop
    
    def add_tasks(self, save_freq='slow', suffix='', subdir='', save_all_fields=True, mode='append'):

        # Save trajectory

        self.save_folder = get_fpath_sim(self.material_params, self.system_params, self.solver_params,
                                                       suffix=suffix, subdir=subdir)

        if not os.path.exists(self.save_folder): os.makedirs(self.save_folder, exist_ok=True)

        if save_freq == 'full':
            sim_dt = 50
        elif save_freq == 'long':
            sim_dt = 0.1
        elif save_freq == 'recent':
            sim_dt = 0.5
        else:
            sim_dt = save_freq
        
        if mode == 'overwrite':
            max_writes=1
        else:
            max_writes=100

        snapshots = self.solver.evaluator.add_file_handler(self.save_folder, \
                                                           sim_dt=sim_dt, max_writes=max_writes, mode=mode)
        
        handler = [handler for handler in self.solver.evaluator.handlers if hasattr(handler, 'base_path') and posixpath.abspath(handler.base_path).endswith(suffix)][0]

        if save_all_fields:
            snapshots.add_system(self.solver.state, layout='g')

        snapshots.add_task("integ(c11 + c22 + c33, 'y', 'x') / area", layout='g', name='<trace>')
        snapshots.add_task("integ( u ** 2 + v ** 2 + w ** 2, 'x', 'y') / area / 2", layout='g', name='<KE>')
        snapshots.add_task("integ( U ** 2 + V ** 2 + WW ** 2, 'x', 'y') / area / 2", layout='g', name='<KE_lam>')
        snapshots.add_task("integ(C11 + C22 + C33, 'y', 'x') / area", layout='g', name='<trace_lam>')
        
        return handler