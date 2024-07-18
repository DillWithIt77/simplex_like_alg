###################################old polyhedron code############################################################
# import numpy
# import sympy
# from scipy import optimize

# import naive_algorithm
# import polyhedral_model as pm

# #constant epsilon
# EPS = 10**-8

# #class for representing a general polyhedron of the form:
# # P = {x in R^n : Ax = b, Bx <= d}
# class Polyhedron:
    
#     #initiallize with matrices and vectors given by numpy arrays
#     def __init__(self, B, d, A=None, b=None):
#         self.B = B
#         self.d = d
#         self.A = A
#         self.b = b
        
    
#     #use naive algorithm to enumerate set of circuits C(A,B)
#     def naive_circuit_enumeration(self):
#         return naive_algorithm.enumerate_circuits(B_ineq=self.B, A_eq=self.A)
    
    
#     #use polyhedral model to enumerate set of circuits C(A,B)
#     def polyhedral_model_circuit_enumeration(self, sign=None):
#         return pm.enumerate_circuits(B_ineq=self.B, A_eq=self.A, sign=sign)
    
    
#     #use polyhedral model to enumerate subset of circuits that are 
#     #sign-compatible with a given direction u with respect to B.
#     def get_sign_compatible_circuits(self, u):
#         sign = self.B.dot(u)
#         return pm.enumerate_circuits(B_ineq=self.B, A_eq=self.A, sign=sign)
    
    
#     #given an x in P, enumerates circuits of P which are strictly feasible at x
#     def get_strictly_feasible_circuits(self, x):
#         sign = self.get_feasibility_sign(x)
#         return pm.enumerate_circuits(B_ineq=self.B, A_eq=self.A, sign=sign)
    
    
#     #returns sign list associated with feasible directions at x in P
#     def get_feasibility_sign(self, x):
#         B_x = self.B.dot(x)
#         m = self.B.shape[0]
#         sign = []
#         for i in range(m):
#             if B_x[i] == self.d[i]:
#                 sign.append(-1)
#             else:
#                 sign.append(None)
#         return sign
    
    
#     #enumerates circuits the related standard form polyhedron 
#     #returns list of the actual circuits of P and list of its standard form circuits
#     def get_standard_form_circuits(self):
#         m_B, n = self.B.shape
#         M = numpy.concatenate((self.B, numpy.eye(m_B, dtype=int)), axis=1)
#         if self.A is not None:
#             m_A = self.A.shape[0]
#             M_A = numpy.concatenate((self.A, numpy.zeros((m_A, m_B), dtype=int)), axis=1)
#             M = numpy.concatenate((M_A, M), axis=0)
#         standard_circuits = naive_algorithm.enumerate_circuits( B_ineq=numpy.eye((n+m_B), dtype=int), A_eq = M)
           
#         #post-processing to determine the actual ciruits of P
#         circuits = []
#         for y in standard_circuits:
#             g = y[:n]
#             B_g = self.B.dot(g)
#             B_0 = numpy.zeros((1, n), dtype=int)
#             for i in range(m_B):
#                 if B_g[i]==0:
#                     B_0 = numpy.concatenate((B_0, self.B[i,:].reshape((1, n))))
#             if self.A is not None:
#                 B_0 = numpy.concatenate((self.A, B_0))
#             rank = numpy.linalg.matrix_rank(numpy.matrix(B_0))
#             if rank == n - 1:
#                 circuits.append(g)
        
#         return circuits, standard_circuits
    
    
#     #given a point in P and a linear objective c, compuate a steepest descent circuit at x
#     def get_steepest_descent_circuit(self, x, c):
#         sign = self.get_feasibility_sign(x)
#         return self.get_steepest_descent_sign_comp_circuit(sign=sign, c=c)
        
        
#     #determine if a vector g is a circuit direction of P
#     def is_circuit(self, g):
#         m,n = self.B.shape
#         B_g = self.B.dot(g)
#         B_0 = numpy.zeros((1,n), dtype=int)
#         for i in range(m):
#             if abs(B_g[i]) <= EPS:
#                 B_0 = numpy.concatenate((B_0, self.B[i,:].reshape((1,n))))
#         if self.A is not None:
#                 B_0 = numpy.concatenate((self.A, B_0))
#         rank = numpy.linalg.matrix_rank(numpy.matrix(B_0))
#         return (rank == n - 1)
    
    
#     #return normalized circuit given a circuit direction of P
#     def get_normalized_circuit(self, g):
#         m,n = self.B.shape
#         B_g = self.B.dot(g)
#         B_0 = numpy.zeros((1,n), dtype=int)
        
#         for i in range(m):
#             if abs(B_g[i]) <= EPS:
#                 B_0 = numpy.concatenate((B_0, self.B[i,:].reshape((1,n))))
#         if self.A is not None:
#                 B_0 = numpy.concatenate((self.A, B_0), axis=0)
                
#         D = sympy.Matrix(B_0)
#         ker_D = D.nullspace()
#         if len(ker_D) != 1:
#             raise ValueError('The direction ' + str(g.T)  +' is not a circuit of P')
#         circuit = numpy.array(ker_D[0]).reshape(n)
#         circuit= circuit*sympy.lcm([circuit[i].q for i in range(n) if circuit[i] != 0]) #normalize
        
#         #make sure circuit has correct sign
#         for i in range(n):
#             if abs(g[i]) >= EPS:
#                 if circuit[i]*g[i] < 0:
#                     circuit = -1*circuit
#                 break
#         return circuit     
    
    
#     #given a point x in P with feasible direction g, compute the maximum step size alpha
#     def get_max_step_size(self, x, g):
#         m,n = self.B.shape
#         B_g = self.B.dot(g)     
#         B_x = self.B.dot(x)
#         alpha = float('inf')
        
#         for i in range(m):
#             if B_g[i] > 0:
#                 a = (self.d[i] - B_x[i])/float(B_g[i])
#                 if a <= alpha:
#                     alpha = a 
#         return alpha
            
#     #given a u in ker(A) or two points in P, construct a sign-compatible sum of circuits.
#     #if a linear function f(x) = c^T x is provided, returns an f-optimal sign-compatible sum
#     #returns the list of circuits and corresponding positive weights lambdas for the sum
#     def get_sign_compatible_sum(self, u=None, x_1=None, x_2=None, c=None):
#         m,n = self.B.shape 
#         if c is None:
#             c = numpy.ones(n, dtype=int)
#         if u is not None:
#             w = u
#         else:
#             w = x_2 - x_1
            
#         if self.A is not None:
#             if any(self.A.dot(w) != 0):
#                 raise ValueError('The direction w or x_2 - x_1 must belong to ker(A).')
       
#         circuits = []
#         lambdas = []
        
#         while not self.is_circuit(w):
#             sign = self.B.dot(w)
#             g = self.get_steepest_descent_sign_comp_circuit(sign=sign, c=c)[0]
#             B_g = self.B.dot(g)
#             B_w = self.B.dot(w)       
#             lam = min([(B_w[i]/B_g[i]) for i in range(m) if B_w[i]*B_g[i] > 0])
            
#             circuits.append(g)
#             lambdas.append(lam)           
#             w = w - lam*g
            
#         g = self.get_normalized_circuit(w)
#         circuits.append(g)
#         for i in range(n):
#             if g[i] > 0:
#                 lambdas.append(w[i]/g[i])
#                 break
#         return circuits, lambdas
        
    
#     #####Might also need to make edits here for simplex-like stuff##### 
#     # things to add: r.h.s for (Bx_0)_i (maybe make serperate function to deteremine which i's are satisfied)  
     
#     #returns steepest descent circuit with respect to c sign-compatible with sign    
#     def get_steepest_descent_sign_comp_circuit(self, sign, c):
#         m_B, n = self.B.shape
#         y_vars = pm.determine_y_vars(m_B, sign=sign)
#         n_y_vars = len(y_vars)
        
#         #constraint matrix and r.h.s vector for linear program
#         M = pm.build_augmented_matrix(self.B, y_vars, A=self.A)
#         one_norm = numpy.concatenate((numpy.zeros(n, dtype=int),numpy.ones(n_y_vars, dtype=int))).reshape(1,n+n_y_vars)
#         M = numpy.concatenate((M, one_norm))
#         b_eq = numpy.zeros(M.shape[0], dtype=int)
#         b_eq[-1]=1
        
#         #upper and lower bounds for variables
#         bounds = [(None, None)]*n
#         for i in range(n_y_vars):
#             bounds.append((0, None))
        
#         #objective function
#         obj = numpy.concatenate((c, numpy.zeros(n_y_vars, dtype=int)))
        
#         #solve linear program
#         result = optimize.linprog(c=obj, A_eq=M, b_eq=b_eq, bounds=bounds, method='simplex')
#         #print(result)
#         if result.status == 2:
#             raise ValueError('Unable to find feasible circuit direction')
#         g = result.x[:n]
#         steepness = result.fun
        
#         if steepness == 0:
#             return numpy.zeros(n), 0
        
#         if self.is_circuit(g):
#             #normalize to coprime integer components
#             circuit = self.get_normalized_circuit(g)       
#             return circuit, steepness     
        
#         #if the linear program does not return a vertex solution, optimize over optimal face until a vertex solution is found
#         #there are issues with scipy's optimize.linprog simplex solver when A_eq is not full row rank,
#         #so this should only be used for toy examples
#         M = numpy.concatenate((M, obj.reshape((1,n + n_y_vars))))
#         b_eq = numpy.append(b_eq, steepness)

#         count = 1
#         while not self.is_circuit(g):
#             if count > 10:
#                 raise ValueError('Failed to find vertex solution to linear program') 
            
#             rand_obj = numpy.random.randint(low=0,high=100,size=n)
#             rand_obj = numpy.concatenate((rand_obj, numpy.zeros(n_y_vars,dtype=int)))
            
#             result = optimize.linprog(c=rand_obj, A_eq=M, b_eq=b_eq, bounds=bounds, method='interior-point')
#             #print(result)
#             g = result.x[:n]
            
#             if self.is_circuit(g):
#                 circuit = self.get_normalized_circuit(g) 
#                 return circuit, steepness 
#             count += 1
###################################################################################################################

import numpy as np
import sympy
import gurobipy as gp
import contextlib
import time

from polyhedral_model import PolyhedralModel
from utils import result, EPS, INF, METHODS


#class for representing a general polyhedron of the form:
# P = {x in R^n : Ax = b, Bx <= d}, with objective c
class Polyhedron:
    
    # initiallize with matrices and vectors given by numpy arrays
    def __init__(self, B, d, A=None, b=None, c=None):
        self.B = B
        self.d = d
        self.A = A
        self.b = b
        self.c = c
        
        self.m_B, self.n = self.B.shape
        self.m_A = self.A.shape[0] if self.A is not None else 0
        self.model = None
        print('Problem size: n = {},  m_B = {},  m_A = {}'.format(self.n, self.m_B, self.m_A))
        
    # construct polyhedral model for computing circuits
    def build_polyhedral_model(self, active_inds=[], primal=True, method='dual_simplex'):
        pm = PolyhedralModel(B=self.B, A=self.A, c=self.c, active_inds=active_inds, method=method)
        return pm
    
    # set current problem solution and get active constraints
    def get_active_constraints(self, x):
        B_x = self.B.dot(x)
        inds = []
        for i in range(self.m_B):
            if self.d[i] - B_x[i] <= EPS:
                inds.append(i)
        self.x_current = x
        self.B_x_current = B_x
        self.active_inds = [False] * self.m_B
        for i in inds:
            self.active_inds[i] = True
        return inds   
    
    #given a point x in P with feasible direction g, compute the maximum step size alpha
    def get_max_step_size(self, x, g, active_inds=None, y_pos=None):
        inds = range(self.m_B)
        if y_pos is not None:
            inds = [i for i in range(self.m_B) if y_pos[i] > 0.0]
        inds = [i for i in inds if i not in active_inds]

        alpha = float('inf')
        active_ind = None
        for i in inds:
            B_g_i = y_pos[i] if y_pos is not None else self.B[i].dot(g)
            if B_g_i <= EPS:
                continue
            B_x_i = self.B[i].dot(x)
            a = (self.d[i] - B_x_i) / float(B_g_i)
            if a <= alpha:
                alpha = a 
                active_ind = i
        return alpha, active_ind
    
    # use saved information about active facets to compute maximal step size along given direction
    def take_maximal_step(self, g, y_pos, y_neg):
        assert hasattr(self, 'x_current') and hasattr(self, 'active_inds')  
        B_g = y_pos - y_neg
        alpha = float('inf')
        stopping_inds = []
        for i in range(self.m_B):
            B_g_i = B_g[i]
            if abs(B_g_i) <= EPS: 
                continue
            elif B_g_i > 0:
                if self.active_inds[i]: 
                    continue
                a = (self.d[i] - self.B_x_current[i]) / float(B_g_i)
                if abs(alpha - a) < EPS:
                    self.active_inds[i] = True
                    stopping_inds.append(i)
                elif a < alpha:
                    alpha = a
                    for j in stopping_inds:
                        self.active_inds[j] = False
                    stopping_inds = [i]
                    self.active_inds[i] = True
            elif B_g_i < 0:
                self.active_inds[i] = False   
        
        # take step with size alpha
        if alpha < EPS:
            print('Degenerate step computed. Changing active facets...')
            for i in stopping_inds:
                self.active_inds[i] = True           
            #raise RuntimeError('Invalid step size: {}'.format(alpha))
        self.x_current += alpha * g
        self.B_x_current += alpha * B_g
        
        # return solution, step size, and list of active constraints
        return self.x_current, alpha, [i for i in range(self.m_B) if self.active_inds[i]]
    

    # build a gurobi LP for the polyhedron          
    def build_gurobi_model(self, c=None, verbose=False, method='primal_simplex'):
        if c is None:
            c = self.c
        assert c is not None, 'Provide an objective function'

        self.model = gp.Model()
        self.x = []
        for i in range(self.n):
            self.x.append(self.model.addVar(lb=-INF, ub=INF, name='x_{}'.format(i)))
        for i in range(self.m_A):
            self.model.addConstr(gp.LinExpr(self.A[i], self.x) == self.b[i], name='A_{}'.format(i))
        for i in range(self.m_B):
            self.model.addConstr(gp.LinExpr(self.B[i], self.x) <= self.d[i], name='B_{}'.format(i))
            
        self.set_objective(c)     
        self.set_verbose(verbose)
        self.set_method(method)

    
    # (re)set objective function
    def set_objective(self, c):
        self.c = c
        if self.model is not None:
            self.model.setObjective(gp.LinExpr(self.c, self.x))

          
    # change model verbose settings
    def set_verbose(self, verbose):
        flag = 1 if verbose else 0
        with contextlib.redirect_stdout(None):
            self.model.setParam(gp.GRB.Param.OutputFlag, flag)
            
    # set lp solve method        
    def set_method(self, method):
        self.method = method
        with contextlib.redirect_stdout(None):
            self.model.Params.method = METHODS[method]
                        
    # find a feasible solution in the polyhedron
    def find_feasible_solution(self, verbose=False):
        c_orig = np.copy(self.c)
        c = np.zeros(self.n)
        if self.model is None:
            self.build_gurobi_model(c, verbose)
        else:
            self.set_objective(c)           
        
        self.model.optimize()
        if self.model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Failed to find feasible solution.')        
        
        self.set_objective(c_orig)
        x_feasible = self.model.getAttr('x', self.x)  
        return x_feasible
               
    # sovle linear program using the given method
    def solve_lp(self, c=None, verbose=False, record_objs=True, method='primal_simplex'):
        
        if c is None:
            assert self.c is not None, 'Need objective function'
            c = self.c
         
        if self.model is None:
            self.build_gurobi_model(c=c)
        self.set_objective(c)
        self.set_method(method)
        t0 = time.time()
            
        obj_values = []
        iter_times = []
        iter_counts = []
        def obj_callback(model, where):
            if where == gp.GRB.Callback.SIMPLEX:
                obj = model.cbGet(gp.GRB.Callback.SPX_OBJVAL)
                obj_values.append(obj)
                iter_times.append(time.time() - t0)
                
                iter_count = model.cbGet(gp.GRB.Callback.SPX_ITRCNT)
                iter_counts.append(iter_count)
                
        if record_objs:
            self.model.optimize(obj_callback)
        else:
            self.model.optimize()
        if self.model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Model failed to solve')
            
        x_optimal = self.model.getAttr('x', self.x)        
        obj_optimal = self.model.objVal
        num_steps = self.model.getAttr('IterCount')
        #solve_time = self.model.getAttr('Runtime')
        solve_time = time.time() - t0
        output = result(0, x=x_optimal, obj=obj_optimal, n_iters=num_steps, solve_time=solve_time,
                        iter_times=iter_times, obj_values=obj_values, iter_counts=iter_counts)        
        return output
    
    # warm start the model with the provided solution
    # (not needed if model already used to find feasible solution)
    def set_solution(self, x):
        for i in range(self.n):
            self.x[i].lb = x[i]
            self.x[i].ub = x[i]
            
        self.model.optimize()
        if self.model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Failed to find feasible solution.')
        print('Feasible solution found with objective {}'.format(self.model.objVal))
        
        # reset the model to its original state
        for i in range(self.n):
            self.x[i].lb = -INF
            self.x[i].ub = INF
        ###commented out so as to now change solution####
        # c_orig = np.copy(self.c)
        # self.set_objective(np.zeros(self.n))                  
        # self.model.optimize()
        # if self.model.status != gp.GRB.Status.OPTIMAL:
        #     raise RuntimeError('Failed to set feasible solution.')                
        # self.set_objective(c_orig)
        
                
    # return normalized circuit given a circuit direction of P
    def get_normalized_circuit(self, g):
        B_g = self.B.dot(g)
        B_0 = np.zeros((1, self.n), dtype=int)
        
        for i in range(self.m_B):
            if abs(B_g[i]) <= EPS:
                B_0 = np.concatenate((B_0, self.B[i,:].reshape((1 ,self.n))))
        if self.A is not None:
            B_0 = np.concatenate((self.A, B_0), axis=0)
                
        D = sympy.Matrix(B_0)
        ker_D = D.nullspace()
        #if len(ker_D) != 1:
        #    raise ValueError('The direction {} is not a circuit of P'.format(g.T))
        circuit = np.array(ker_D[0]).reshape(self.n)
        
        #normalize
        circuit= circuit*sympy.lcm([circuit[i].q for i in range(self.n) if circuit[i] != 0])
        
        #make sure circuit has correct sign
        for i in range(self.n):
            if abs(g[i]) >= EPS:
                if circuit[i]*g[i] < 0:
                    circuit = -1*circuit
                break
        return circuit
    
    
    # add random facets containing the given points well keeping the other given points feasible
    def add_facets(self, include_point, feasible_points=[], n_facets=1):
        if not isinstance(feasible_points, list):
            feasible_points = [feasible_points]
        print('Adding {} facets...'.format(n_facets))
        for _ in range(n_facets):
            while True:
                row = np.random.randint(-100, 100, size=self.n)
                rhs = row.dot(include_point)
                keep = True
                for point in feasible_points:
                    if row.dot(point) > rhs:
                        keep = False
                if keep: 
                    break
                    
            self.B = np.concatenate((self.B, np.expand_dims(row, axis=0)))
            self.d = np.concatenate((self.d, np.array([rhs])))

    def second_vert(self, init_x, c=None, verbose=False, record_objs=True, method='primal_simplex'):
        ######try to find a way for this to not restart (if doesn't work then try XPRESS)
        if c is None:
            assert self.c is not None, 'Need objective function'
        c = self.c
         
        if self.model is None:
            self.build_gurobi_model(c=c)
        self.set_solution(init_x)
        self.set_objective(c)
        self.set_method(method)
        init_obj = self.model.getObjective().getValue()
        
        t0 = time.time()
            
        def obj_callback(model, where):
            if where == gp.GRB.Callback.SIMPLEX:
                obj = model.cbGet(gp.GRB.Callback.SPX_OBJVAL)
                # print(f'Objective Value: {obj}')
                if obj != init_obj:
                    model.terminate() 

        self.model.optimize(obj_callback)
        x_optimal = self.model.getAttr('x', self.x)
                
        return x_optimal
