import time
import random
import os
import numpy as np

import facet_rest_sdac as fs_sd
from mps_reader_preprocessor import read_mps_preprocess
from polyhedron import Polyhedron

def facet_main(mps_fn='', results_dir='results', max_time=300, sd_method='dual_simplex', reset=False):

	print('Reading {}...'.format(mps_fn))
	c, B, d, A, b = read_mps_preprocess(mps_fn)
	print('Building polyhedron...')
	P = Polyhedron(B, d, A, b, c)

	print('Finding feasible solution...')
	x_feasible, vbasis, cbasis = P.find_feasible_solution(verbose=False)

	print('\nSolving with simplex method...')
	lp_result = P.solve_lp(verbose=False, record_objs=True)
	print('\nSolution using simplex method:')
	print(lp_result)

	print('\nSolving with original steepest descent...')
	old_sd_result = fs_sd.old_sdac(mps_fn,results_dir, max_time, reset,sd_method)

	print('\nSolving with circuit aug with updated active inds steepest descent...')
	circ_up_act_sd_result = fs_sd.circ_up_act_rest(mps_fn,results_dir, max_time, reset,sd_method)

	print('\nSolving with circuit aug with initial active inds steepest descent...')
	circ_init_act_sd_result = fs_sd.circ_init_act_rest(mps_fn,results_dir, max_time, reset,sd_method)

	print('\nFinish solving with simplex with updated active inds steepest descent...')
	simp_up_act_sd_result = fs_sd.simp_up_act_rest(mps_fn,results_dir, max_time, reset,sd_method)

	print('\nFinish solving with simplex with initial active inds steepest descent...')
	simp_init_act_sd_result = fs_sd.simp_init_act_rest(mps_fn,results_dir, max_time, reset,sd_method)

	#####Save results
	if results_dir:
		if not os.path.exists(results_dir): 
			os.mkdir(results_dir)
		if mps_fn:
			prefix = os.path.basename(mps_fn).split('.')[0]
			lp_fn = os.path.join(results_dir, prefix + '_lp.p')
			old_sd_fn = os.path.join(results_dir, prefix + '_old_sd.p')
			circ_up_act_sd_fn = os.path.join(results_dir, prefix + '_circ_up_act_sd.p')
			circ_init_act_sd_fn = os.path.join(results_dir, prefix + '_circ_init_act_sd.p')
			simp_up_act_sd_fn = os.path.join(results_dir, prefix + '_simp_up_act_sd.p')
			simp_init_act_sd_fn = os.path.join(results_dir, prefix + '_simp_init_act_sd.p')
			lp_result.save(lp_fn)
			old_sd_result.save(old_sd_fn)
			circ_up_act_sd_result.save(circ_up_act_sd_fn)
			circ_init_act_sd_result.save(circ_init_act_sd_fn)
			simp_up_act_sd_result.save(simp_up_act_sd_fn)
			simp_init_act_sd_result.save(simp_init_act_sd_fn)