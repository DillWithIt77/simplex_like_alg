import time
import random
import os
import numpy as np

import facet_rest_sdac as fs_sd
from mps_reader_preprocessor import read_mps_preprocess
from polyhedron import Polyhedron

def facet_main(mps_fn='', results_dir='results', max_time=300, sd_method='dual_simplex', reset=False, **kwargs):
	mat_input = kwargs.get('mat_input', False)
    A = kwargs.get('A', [])
    B = kwargs.get('B', [])
    b = kwargs.get('b', [])
    d = kwargs.get('d', [])
    c = kwargs.get('c', [])

    if mat_input == False:
		print('Reading {}...'.format(mps_fn))
		c, B, d, A, b = read_mps_preprocess(mps_fn)
	else:
		print('Using matrix input')

	print('Building polyhedron...')
	P = Polyhedron(B, d, A, b, c)

	print('Finding feasible solution...')
	x_feasible, vbasis, cbasis = P.find_feasible_solution(verbose=False)

	print('\nSolving with simplex method...')
	lp_result = P.solve_lp(verbose=False, record_objs=True)
	print('\nSolution using simplex method:')
	print(lp_result)

	print('\nSolving with original steepest descent...')
	if mat_input == False:
		old_sd_result = fs_sd.old_sdac(mps_fn,results_dir, max_time, reset,sd_method)
	else:
		old_sd_result = fs_sd.old_sdac(mps_fn,results_dir, max_time, reset,sd_method, A = A, B = B, b = b, d = d, c = c, mat_input = True)

	print('\nSolving with circuit aug with updated active inds steepest descent...')
	if mat_input == False:
		circ_up_act_sd_result = fs_sd.circ_up_act_rest(mps_fn,results_dir, max_time, reset,sd_method)
	else:
		circ_up_act_sd_result = fs_sd.circ_up_act_rest(mps_fn,results_dir, max_time, reset,sd_method, A = A, B = B, b = b, d = d, c = c, mat_input = True)

	print('\nSolving with circuit aug with initial active inds steepest descent...')
	if mat_input == False:
		circ_init_act_sd_result = fs_sd.circ_init_act_rest(mps_fn,results_dir, max_time, reset,sd_method)
	else:
		circ_init_act_sd_result = fs_sd.circ_init_act_rest(mps_fn,results_dir, max_time, reset,sd_method, A = A, B = B, b = b, d = d, c = c, mat_input = True)

	print('\nFinish solving with simplex with updated active inds steepest descent...')
	if mat_input == False:
		simp_up_act_sd_result = fs_sd.simp_up_act_rest(mps_fn,results_dir, max_time, reset,sd_method)
	else:
		simp_up_act_sd_result = fs_sd.simp_up_act_rest(mps_fn,results_dir, max_time, reset,sd_method, A = A, B = B, b = b, d = d, c = c, mat_input = True)

	print('\nFinish solving with simplex with initial active inds steepest descent...')
	if mat_input == False:
		simp_init_act_sd_result = fs_sd.simp_init_act_rest(mps_fn,results_dir, max_time, reset,sd_method, A = A, B = B, b = b, d = d, c = c, mat_input = True)
	else:
		simp_init_act_sd_result = fs_sd.simp_init_act_rest(mps_fn,results_dir, max_time, reset,sd_method, A = A, B = B, b = b, d = d, c = c, mat_input = True)

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