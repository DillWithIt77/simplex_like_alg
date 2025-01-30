import time
import random
import os
import numpy as np

import hybrid_sdac as hyb_sd
from mps_reader_preprocessor import read_mps_preprocess
from polyhedron import Polyhedron

def hybrid_main(mps_fn='', results_dir='results', max_time=300, sd_method='dual_simplex', reset=False, **kwargs):
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

	# print('\nSolving with original steepest descent...')
	# if mat_input == False:
		# old_sd_result = hyb_sd.old_sdac(mps_fn,results_dir, max_time, reset,sd_method)
	# else:
		# old_sd_result = hyb_sd.old_sdac(mps_fn,results_dir, max_time, reset,sd_method, A = A, B = B, b = b, d = d, c = c, mat_input = True)

	# print('\nSolving with alt steepest descent...')
	# if mat_input == False:
		# alt_sd_result = hyb_sd.alt_sdac(mps_fn,results_dir, max_time, reset,sd_method)
	# else:
		# alt_sd_result = hyb_sd.alt_sdac(mps_fn,results_dir, max_time, reset,sd_method, A = A, B = B, b = b, d = d, c = c, mat_input = True)

	print('\nSolving with hybrid steepest descent with init active inds...')
	if mat_input == False:
		reg_hybrid_init_act_sd_result = hyb_sd.hybrid_init_act(mps_fn,results_dir, max_time, reset,sd_method)
	else:
		reg_hybrid_init_act_sd_result = hyb_sd.hybrid_init_act(mps_fn,results_dir, max_time, reset,sd_method, A = A, B = B, b = b, d = d, c = c, mat_input = True)
	# print(hybrid_init_act_sd_result)

	print('\nSolving with hybrid steepest descent with updated active inds...')
	if mat_input == False:
		reg_hybrid_up_act_sd_result = hyb_sd.hybrid_up_act(mps_fn,results_dir, max_time, reset,sd_method)
	else:
		reg_hybrid_up_act_sd_result = hyb_sd.hybrid_up_act(mps_fn,results_dir, max_time, reset,sd_method, A = A, B = B, b = b, d = d, c = c, mat_input = True)
	# print(hybrid_up_act_sd_result)

	# print('\nSolving with alt hybrid steepest descent with init active inds...')
	# if mat_input == False:
		# alt_hybrid_init_act_sd_result = hyb_sd.alt_hybrid_init_act(mps_fn,results_dir, max_time, reset,sd_method)
	# else:
		# alt_hybrid_init_act_sd_result = hyb_sd.alt_hybrid_init_act(mps_fn,results_dir, max_time, reset,sd_method, A = A, B = B, b = b, d = d, c = c, mat_input = True)
	# # print(alt_hybrid_init_act_sd_result)

	# print('\nSolving with alt hybrid steepest descent with updated active inds...')
	# if mat_input == False:
		# alt_hybrid_up_act_sd_result = hyb_sd.alt_hybrid_up_act(mps_fn,results_dir, max_time, reset,sd_method)
	# else:
		# alt_hybrid_up_act_sd_result = hyb_sd.alt_hybrid_up_act(mps_fn,results_dir, max_time, reset,sd_method, A = A, B = B, b = b, d = d, c = c, mat_input = True)
	# # print(alt_hybrid_up_act_sd_result)

	# print('\nSolving with heuristic, no T...')
	# if mat_input == False:
		# heu_no_t_result = hyb_sd.heuristic_no_t(mps_fn,results_dir, max_time, reset, sd_method, s = 1)
	# else:
		# heu_no_t_result = hyb_sd.heuristic_no_t(mps_fn,results_dir, max_time, reset, sd_method, s = 1, A = A, B = B, b = b, d = d, c = c, mat_input = True)


	#####Save results
	if results_dir:
		if not os.path.exists(results_dir): os.mkdir(results_dir)
		if mps_fn:
			prefix = os.path.basename(mps_fn).split('.')[0]
			lp_fn = os.path.join(results_dir, prefix + '_lp.p')
			# old_sd_fn = os.path.join(results_dir, prefix + '_old_sd.p')
			# alt_sd_fn = os.path.join(results_dir, prefix + '_alt_sd.p')
			reg_hybrid_init_act_sd_fn = os.path.join(results_dir, prefix + '_reg_hybrid_up_act_sd.p')
			reg_hybrid_up_act_sd_fn = os.path.join(results_dir, prefix + '_reg_hybrid_init_act_sd.p')
			# alt_hybrid_init_act_sd_fn = os.path.join(results_dir, prefix + '_alt_hybrid_up_act_sd.p')
			# alt_hybrid_up_act_sd_fn = os.path.join(results_dir, prefix + '_alt_hybrid_init_act_sd.p')
			# heu_no_t_fn = os.path.join(results_dir, prefix + '_heu_no_t.p')
			lp_result.save(lp_fn)
			# old_sd_result.save(old_sd_fn)
			# alt_sd_result.save(alt_sd_fn)
			reg_hybrid_init_act_sd_result.save(reg_hybrid_init_act_sd_fn)
			reg_hybrid_up_act_sd_result.save(reg_hybrid_up_act_sd_fn)
			# alt_hybrid_init_act_sd_result.save(alt_hybrid_init_act_sd_fn)
			# alt_hybrid_up_act_sd_result.save(alt_hybrid_up_act_sd_fn)
			# heu_no_t_result.save(heu_no_t_fn)