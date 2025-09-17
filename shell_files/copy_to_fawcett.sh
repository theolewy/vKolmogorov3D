eval "cd_nfs"
eval "cd vKolmogorov3D/simulations/arrowhead_3D"

eval "rm -rf sim_W_20_Re_0,5_beta_0,9_eps_0,001_L_inf_Lx_9,4248_Lz_12,566_ndim_3_N_64-64-64_long-test-drift-pert-0,02-method-2/"
eval "rm -rf sim_W_20_Re_0,5_beta_0,9_eps_0,001_L_inf_Lx_9,4248_Lz_12,566_ndim_3_N_64-64-64_recent-test-drift-pert-0,02-method-2/"

eval "cp_from_swirles vKolmogorov3D/simulations/arrowhead_3D/ sim_W_20_Re_0,5_beta_0,9_eps_0,001_L_inf_Lx_9,4248_Lz_12,566_ndim_3_N_64-64-64_long-test-drift-pert-0,02-method-2"
eval "cp_from_swirles vKolmogorov3D/simulations/arrowhead_3D/ sim_W_20_Re_0,5_beta_0,9_eps_0,001_L_inf_Lx_9,4248_Lz_12,566_ndim_3_N_64-64-64_recent-test-drift-pert-0,02-method-2"