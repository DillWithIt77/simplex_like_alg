import numpy as np
import gurobipy as gp
import math
import sympy
import contextlib
import time
import os
from utils import result, METHODS, INF, EPS
from mps_reader_preprocessor import read_mps_preprocess
from polyhedral_model import PolyhedralModel
from polyhedron import Polyhedron


def old_sdac(mps_fn, results_dir, max_time, reset, sd_method):
    ### Build Initial Polyhedron from file
    print(f'Reading {mps_fn}...')
    c, B, d, A, b = read_mps_preprocess(mps_fn)
    print('Building polyhedron...')
    P = Polyhedron(B, d, A, b, c)

    print('Finding feasible solution...')
    x_feasible, vbasis, cbasis = P.find_feasible_solution(verbose=False)

    x = x_feasible
    c=None
    verbose=False
    method='dual_simplex'
    reset=False
    max_time=300
    first_warm_start=None
    save_first_steps=0
    problem_name=''

    if c is not None:
      P.set_objective(c)

    t0 = time.time()
    x_current = x
    if save_first_steps:
      np.save('solutions/{}_0.npy'.format(problem_name), x_current)      
    active_inds = P.get_active_constraints(x_current)
    # print(len(active_inds))
    
    pm = P.build_polyhedral_model(active_inds=active_inds, method=method)

    if first_warm_start is not None:
      print('Using custom warm start')
      pm.set_solution(first_warm_start)
    t1 = time.time()
    build_time = t1 - t0
    print('Polyhedral model build time: {}'.format(build_time))
    
    sub_times = {'sd': [], 'step': [], 'solve': [], 'phase_times': []}    
    descent_circuits = []
    obj_values = []
    step_sizes = []
    iter_times = []
    simplex_iters = []
    iteration = 0
    obj_value = P.c.dot(x_current)
    obj_values.append(obj_value)
    t2 = time.time()
    iter_times.append(t2 - t1)


    # compute steepest-descent direction
    descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(verbose=verbose)
    simplex_iters.append(num_steps)
    sub_times['solve'].append(solve_time)
    sub_times['phase_times'].append(phase_times)
    
    t3 = time.time()
    sub_times['sd'].append(t3 - t2)
    # return sub_times['sd'][-1], sub_times['solve'][-1], sub_times['phase_times'][-1]

    return result(status=0, x=x_current, 
                  obj=P.c.dot(x_current), n_iters=len(step_sizes),
                  iter_times=iter_times, alg_type='steepest-descent',
                  circuits=descent_circuits, steps=step_sizes,
                  simplex_iters=simplex_iters, solve_times=sub_times['solve'],
                  sub_times=sub_times, obj_values=obj_values)

def new_sdac(mps_fn, results_dir, max_time, reset, sd_method):
    ### Build Initial Polyhedron from file
    print(f'Reading {mps_fn}...')
    c, B, d, A, b = read_mps_preprocess(mps_fn)
    print('Building polyhedron...')
    P = Polyhedron(B, d, A, b, c)

    print('Finding feasible solution...')
    x_feasible,vbasis, cbasis = P.find_feasible_solution(verbose=False)

    x = x_feasible
    c=None
    verbose=False
    method='dual_simplex'
    reset=False
    max_time=300
    first_warm_start=None
    save_first_steps=0
    problem_name=''

    if c is not None:
        P.set_objective(c)

    t0 = time.time()
    x_current = x
    if save_first_steps:
      np.save('solutions/{}_0.npy'.format(problem_name), x_current)      
    active_inds = P.get_active_constraints(x_current)
    
    pm = P.build_polyhedral_model(active_inds=active_inds, method=method)

    if first_warm_start is not None:
      print('Using custom warm start')
      pm.set_solution(first_warm_start)
    t1 = time.time()
    build_time = t1 - t0
    print('Polyhedral model build time: {}'.format(build_time))
    
    sub_times = {'sd': [], 'step': [], 'solve': [], 'phase_times': []}    
    descent_circuits = []
    obj_values = []
    step_sizes = []
    iter_times = []
    simplex_iters = []
    iteration = 0
    obj_value = P.c.dot(x_current)
    num_dec_places = Polyhedron.num_aft_dec(obj_value)
    obj_values.append(obj_value)
    t2 = time.time()
    iter_times.append(t2 - t1)

    ####get edge for initial circuit direction here#########
    x_feasible_2= P.second_vert(x_current, obj_value, num_dec_places, verbose=False, vbasis = vbasis, cbasis = cbasis)
    edge = np.array(x_feasible_2) - np.array(x_current)
    B_edge = np.dot(B,np.array(edge))
    norm = np.linalg.norm(np.array(B_edge),1)
    scaled_edge = edge/(norm)
    normed_B_edge = B_edge/(norm)
    init_y_pos = []
    init_y_neg = []
    for entry in normed_B_edge:
      if entry > 0:
        init_y_pos.append(entry)
        init_y_neg.append(0)
      else:
        init_y_pos.append(0)
        init_y_neg.append(-entry)
    ########################################################

    # compute steepest-descent direction
    t3 = time.time()
    descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(y_pos = init_y_pos, y_neg = init_y_neg, 
                                                                                                         edge = scaled_edge,verbose=verbose)
    simplex_iters.append(num_steps)
    sub_times['solve'].append(solve_time)
    sub_times['phase_times'].append(phase_times)
    
    t4 = time.time()
    sub_times['sd'].append(t4 - t3)

    # return sub_times['sd'][-1], sub_times['solve'][-1], sub_times['phase_times'][-1]
    # return (t3-t2)

    return result(status=0, x=x_current, 
                  obj=P.c.dot(x_current), n_iters=len(step_sizes),
                  iter_times=iter_times, alg_type='steepest-descent',
                  circuits=descent_circuits, steps=step_sizes,
                  simplex_iters=simplex_iters, solve_times=sub_times['solve'],
                  sub_times=sub_times, obj_values=obj_values)

def trivial_sdac(mps_fn, results_dir, max_time, reset, sd_method):
    ### Build Initial Polyhedron from file
    print(f'Reading {mps_fn}...')
    c, B, d, A, b = read_mps_preprocess(mps_fn)
    print('Building polyhedron...')
    P = Polyhedron(B, d, A, b, c)

    print('Finding feasible solution...')
    x_feasible,vbasis, cbasis = P.find_feasible_solution(verbose=False)

    x = x_feasible
    c=None
    verbose=False
    method='dual_simplex'
    reset=False
    max_time=300
    first_warm_start=None
    save_first_steps=0
    problem_name=''

    if c is not None:
        P.set_objective(c)

    t0 = time.time()
    x_current = x
    if save_first_steps:
      np.save('solutions/{}_0.npy'.format(problem_name), x_current)      
    active_inds = P.get_active_constraints(x_current)
    
    pm = P.build_polyhedral_model(active_inds=active_inds, method=method)

    if first_warm_start is not None:
      print('Using custom warm start')
      pm.set_solution(first_warm_start)
    t1 = time.time()
    build_time = t1 - t0
    print('Polyhedral model build time: {}'.format(build_time))
    
    sub_times = {'sd': [], 'step': [], 'solve': [], 'phase_times': []}    
    descent_circuits = []
    obj_values = []
    step_sizes = []
    iter_times = []
    simplex_iters = []
    iteration = 0
    obj_value = P.c.dot(x_current)
    num_dec_places = Polyhedron.num_aft_dec(obj_value)
    obj_values.append(obj_value)
    t2 = time.time()
    iter_times.append(t2 - t1)

    #############create trivial circuit here################
    init_y_pos = np.zeros(P.m_B)
    init_y_neg = np.zeros(P.m_B)

    for i in range(P.m_B):
      if i not in active_inds:
        init_y_pos[i] = 0.5
        init_y_neg[i] = 0.5
        break;
    ########################################################

    # compute steepest-descent direction
    t3 = time.time()
    descent_direction, y_pos, y_neg, steepness, num_steps, solve_time, phase_times = pm.compute_sd_direction(y_pos = init_y_pos, y_neg = init_y_neg, 
                                                                                                             trivial =True, verbose=verbose)
    simplex_iters.append(num_steps)
    sub_times['solve'].append(solve_time)
    sub_times['phase_times'].append(phase_times)
    
    t4 = time.time()
    sub_times['sd'].append(t4 - t3)

    # return sub_times['sd'][-1], sub_times['solve'][-1], sub_times['phase_times'][-1]
    return (t3-t2)

    # return result(status=0, x=x_current, 
    #               obj=P.c.dot(x_current), n_iters=len(step_sizes),
    #               iter_times=iter_times, alg_type='steepest-descent',
    #               circuits=descent_circuits, steps=step_sizes,
    #               simplex_iters=simplex_iters, solve_times=sub_times['solve'],
    #               sub_times=sub_times, obj_values=obj_values)
