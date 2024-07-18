##############################old polyhedral model code#############################################################
# import numpy
# import sympy
# import cdd


# #Given matrices A and B, computes set of circuits C(A,B) by enumerating the extreme rays
# #  of the corresponding polyhedral model 
# #Input: A_eq and B_ineq are (m_a x n) and (m x n) numpy arrays;
# #       sign is m-dimensional vector or list which gives desired orthant of Bg if any
# #Output: list of circuits in C(A,B) given by n-dimensional numpy arrays
# def enumerate_circuits(B_ineq, A_eq=None, sign=None):
#     A = A_eq
#     B = B_ineq
#     m_B,n = B.shape
#     m_a = 0
#     if A is not None:
#         m_a = A.shape[0]
    
#     y_vars = determine_y_vars(m_B, sign=sign)
#     n_y_vars = len(y_vars)
    
#     #####This is where I will need to edit to add in extra constraints mentioned in Steffen's Grant####
#     #####figure out how this relates to the sign_comp_steep circs in polyhedron file######
    
#     #build constraint matrix M for conic polyhedral model Mr >= 0,
#     #where first column of M is the r.h.s. vector 0.
#     M = build_augmented_matrix(B, y_vars, A=A)
#     M = numpy.concatenate((M, -1*M))
#     I_y = numpy.concatenate((numpy.zeros((n_y_vars, n)), numpy.eye(n_y_vars)), axis=1).astype(int)
#     M = numpy.concatenate((M, I_y))
#     M = numpy.concatenate((numpy.zeros((2*m_a + 2*m_B + n_y_vars, 1), dtype=int), M), axis=1)
    
#     #use cdd to enumerate extreme rays of the cone <- might be able to replace this with vertex enumeration things
#     mat = cdd.Matrix(M, number_type='fraction')
#     mat.rep_type = cdd.RepType.INEQUALITY
#     poly = cdd.Polyhedron(mat)
#     #print(poly)
#     rays = numpy.array(poly.get_generators())
#     #print(rays)

#     #obtain circuits from extreme rays
#     circuits = []
#     num_rays = rays.shape[0]
#     for i in range(num_rays):
#         g = rays[i,1:n+1]
#         if not numpy.array_equal(g, numpy.zeros(n)):
#             g = g*sympy.lcm([g[j].denominator for j in range(n) if g[j] != 0]) #normalize
#             circuits.append(g)  
#     return circuits


# #returns list of y-variables for polyhedral model associated with given sign list.
# #each y-variable is described by a tuple (i, 1 or -1) which indicates which row i
# #of B the y-variable corresponds to and whether the variable is the positive or negative
# #part of (Bx)_i
# def determine_y_vars(m,sign=None):
#     y_vars = []
#     if sign is None:
#        for i in range(m):
#            y_vars.append([i, 1])
#            y_vars.append([i, -1])
#     else:
#         for i in range(m):
#             s = sign[i]
#             if s is None:
#                 y_vars.append([i, 1])
#                 y_vars.append([i, -1])
#             elif s > 0:
#                 y_vars.append([i, 1])
#             elif s < 0:
#                 y_vars.append([i, -1])
#     return y_vars


# #returns augmented equality matrix for polyhedral model
# def build_augmented_matrix(B, y_vars, A=None):
#     m_B, n = B.shape
#     n_y_vars = len(y_vars)
    
#     M = numpy.concatenate((B, numpy.zeros((m_B, n_y_vars), dtype=int)), axis=1)
#     for j in range(n_y_vars):
#         i = y_vars[j][0]
#         M[i][n+j] = -1*y_vars[j][1]
    
#     if A is not None:
#         m_A = A.shape[0]
#         A_aug = numpy.concatenate((A, numpy.zeros((m_A, n_y_vars), dtype=int)), axis=1)
#         M = numpy.concatenate((A_aug, M), axis=0)
        
#     return M
############################################################################################################    

import numpy as np
import gurobipy as gp
import contextlib
import time

from utils import METHODS, INF, EPS

class PolyhedralModel():
    
    # Given matrices B and A (and optional inds argument / objective function),
    # builds a polyhedral model for computing steepest-descent circuits
    # as a gurobi linear program model
    def __init__(self, B, A=None, active_inds=[], c=None, primal=True, method='dual_simplex'):
        
        print('Building polyhedral model. Solve method: {}'.format(method))
        
        self.model = gp.Model()
        self.primal = primal
        
        # add variables and constraints to the model
        self.m_B, self.n = B.shape
        if self.primal:
            self.x = []
            self.y_pos = []
            self.y_neg = []
            for i in range(self.n):
                self.x.append(self.model.addVar(lb=-INF, ub=INF, name='x_{}'.format(i)))
            for i in range(self.m_B):
                self.y_pos.append(self.model.addVar(lb=0.0, ub=1.0, name='y_pos_{}'.format(i)))
                self.y_neg.append(self.model.addVar(lb=0.0, ub=1.0, name='y_neg_{}'.format(i)))
                self.model.addConstr(gp.LinExpr(list(B[i]) + [-1, 1], 
                                                self.x + [self.y_pos[i], self.y_neg[i]]) == 0, 
                                                name='B_{}'.format(i))
    
            self.model.addConstr(gp.LinExpr([1]*(2*self.m_B), self.y_pos + self.y_neg) == 1, 
                                 name='1_norm')
                
            if A is not None:
                self.m_A = A.shape[0]
                for i in range(self.m_A):
                    self.model.addConstr(gp.LinExpr(A[i], self.x) == 0, 
                                         name='A_{}'.format(i))
            else:
                self.m_A = 0
                
            if c is not None:
                self.set_objective(c)
                  
            self.set_active_inds(active_inds)
            self.set_method(method)
                
        else:
            raise RuntimeError('Not yet implemented')
            
        self.model.update()
        print('Polyhedral model built!')
                
                
    def set_objective(self, c):
        self.c = c
        self.model.setObjective(gp.LinExpr(c, self.x))
            
    def set_active_inds(self, active_inds):
        self.active_inds = active_inds
        for i in range(self.m_B):
            self.y_pos[i].lb = 0.0
            self.y_pos[i].ub = 1.0
        for i in self.active_inds:
            self.y_pos[i].ub = 0.0
                
    def set_method(self, method):
        self.method = method
        #with contextlib.redirect_stdout(None):
        self.model.Params.method = METHODS[method]
          
    # warm start the model with the provided solution   
    def set_solution(self, g):
        print(g.shape)
        print(self.n)
        for i in range(self.n):
            self.x[i].lb = g[i]
            self.x[i].ub = g[i]
        for i in range(self.m_B):
            self.y_pos[i].ub = 1.0
            
        # solve the modified model to obtain desired solution    
        self.model.Params.method = 0
        self.model.optimize()
        
        # reset the model contraints back to its original state
        for i in range(self.n):
            self.x[i].lb = -INF
            self.x[i].ub = INF
        ####commented out as to not change solution#####
        # self.model.setObjective(gp.LinExpr(np.zeros(self.n), self.x))                
        # self.model.optimize()
        # if self.model.status != gp.GRB.Status.OPTIMAL:
        #     raise RuntimeError('Failed to set solution for polyhedral model') 
            
        # self.set_objective(self.c)                
        # self.set_active_inds(self.active_inds)
        # self.set_method(self.method)
        #self.model.update()
        
                
    def compute_sd_direction(self, verbose=False, **kwargs):
        init_edge = kwargs.get('edge', np.zeros(self.n))

        flag = 1 if verbose else 0
        self.model.setParam(gp.GRB.Param.OutputFlag, flag)
        
        t0 = time.time()
        if np.all(init_edge != 0):
            self.model.set_solution(init_edge)
        self.model._phase1_time = None
        self.model._is_dualinf = True
        def dualinf_callback(model, where):
            if where == gp.GRB.Callback.SIMPLEX:
                if model._is_dualinf:
                    dualinf = model.cbGet(gp.GRB.Callback.SPX_DUALINF)
                    if dualinf < EPS:
                        model._phase1_time = time.time() - t0
                        model._is_dualinf = False
        
        self.model.optimize(dualinf_callback)
        if self.model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Failed to find steepst-descent direction.')
        
        phase2_time = time.time() - self.model._phase1_time
        phase_times = (self.model._phase1_time, phase2_time)
        g = self.model.getAttr('x', self.x)
        y_pos = self.model.getAttr('x', self.y_pos)
        y_neg = self.model.getAttr('x', self.y_neg)
        steepness = self.model.objVal
        num_steps = self.model.getAttr('IterCount')
        solve_time = self.model.getAttr('Runtime')
        
        return np.asarray(g), np.asarray(y_pos), np.asarray(y_neg), steepness, num_steps, solve_time, phase_times
    
    # find a feasible solution for the polyhedral model
    def find_feasible_solution(self, verbose=False):
        c_orig = np.copy(self.c)
        c = np.zeros(self.n)
        self.set_objective(c)           
        
        self.model.optimize()
        if self.model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Failed to find feasible solution.')        
        
        self.set_objective(c_orig)
        x_feasible = self.model.getAttr('x', self.x)  
        return x_feasible
                
    def reset(self):
        self.model.reset()    
    
        

