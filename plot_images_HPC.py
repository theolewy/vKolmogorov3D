from tools.plotter import *
import matplotlib.pyplot as plt

check_localised(W=20, eps=1e-3, beta=0.9, L=np.inf, Re=0.5, Lx=3*np.pi, Lz=4*np.pi,  Nx=64, Ny=64, Nz=64, suffix='recent-localised', subdir='arrowhead_3D')
check_localised(W=20, eps=2e-4, beta=0.9, L=np.inf, Re=0.5, Lx=3*np.pi, Lz=4*np.pi,  Nx=128, Ny=128, Nz=128, suffix='recent-localised', subdir='arrowhead_3D')