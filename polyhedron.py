import numpy as np
from scipy.sparse import issparse
import sympy
import gurobipy as gp
import contextlib
import time
from fractions import Fraction

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
        # self.active_inds = [False] * self.m_B
        print('Problem size: n = {},  m_B = {},  m_A = {}'.format(self.n, self.m_B, self.m_A))
        
    # construct polyhedral model for computing circuits
    def build_polyhedral_model(self, active_inds=[], primal=True, method='dual_simplex', alt = False):
        pm = PolyhedralModel(B=self.B, A=self.A, c=self.c, active_inds=active_inds, method=method, alt = alt)
        return pm
    
    # set current problem solution and get active constraints
    def get_active_constraints(self, x, alt = False):
        B_x = self.B.dot(x)
        inds = []
        M = -float('inf')
        omega = float('inf')
        for i in range(self.m_B):
            diff = self.d[i] - B_x[i] 
            if self.d[i] - B_x[i] <= EPS:
                inds.append(i)
            if diff > M:
                M = diff
            if diff > 0 and diff < omega:
                omega = diff
        self.x_current = x
        self.B_x_current = B_x
        self.active_inds = [False] * self.m_B
        for i in inds:
            self.active_inds[i] = True
        if alt:
            return inds, M, omega
        else:
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
    def take_maximal_step(self, g, y_pos, y_neg,**kwargs):
        init_inds = kwargs.get('init_inds',[])
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
            # raise RuntimeError('Invalid step size: {}'.format(alpha))
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
        
        if issparse(self.A):
            for i in range(self.m_A):
                # Extract the nonzero elements in row `i`
                start_idx, end_idx = self.A.indptr[i], self.A.indptr[i + 1]
    
                # Get the nonzero values and corresponding column indices
                row_values = self.A.data[start_idx:end_idx]  # Nonzero values in the row
                row_columns = self.A.indices[start_idx:end_idx]  # Corresponding column indices

                # Use addTerms() to efficiently create the expression
                expr = gp.LinExpr()
                expr.addTerms(row_values, [self.x[j] for j in row_columns])

                # Add the constraint to the model
                self.model.addConstr(expr == self.b[i], name='A_{}'.format(i))
        else:
            for i in range(self.m_A):
                self.model.addConstr(gp.LinExpr(self.A[i], self.x) == self.b[i], name='A_{}'.format(i))
        if issparse(self.B):
            for i in range(self.m_B):
                # Extract the nonzero elements in row `i`
                start_idx, end_idx = self.B.indptr[i], self.B.indptr[i + 1]
    
                # Get the nonzero values and corresponding column indices
                row_values = self.B.data[start_idx:end_idx]  # Nonzero values in the row
                row_columns = self.B.indices[start_idx:end_idx]  # Corresponding column indices

                # Use addTerms() to efficiently create the expression
                expr = gp.LinExpr()
                expr.addTerms(row_values, [self.x[j] for j in row_columns])

                # Add the constraint to the model
                self.model.addConstr(expr <= self.d[i], name='B_{}'.format(i))
        else:
            for i in range(self.m_B):
                self.model.addConstr(gp.LinExpr(self.B[i], self.x) <= self.d[i], name='B_{}'.format(i))
            
        self.set_objective(c)     
        self.set_verbose(verbose)
        self.set_method(method)
        self.model.update()

    
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
        vbasis = self.model.getAttr(gp.GRB.Attr.VBasis)
        cbasis = self.model.getAttr(gp.GRB.Attr.CBasis)
        return x_feasible,vbasis,cbasis
               
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
            print(f'Model Status: {self.model.status}')
            raise RuntimeError('Model failed to solve')
            
        x_optimal = self.model.getAttr('x', self.x)        
        obj_optimal = self.model.objVal
        num_steps = self.model.getAttr('IterCount')
        solve_time = self.model.getAttr('Runtime')
        # solve_time = time.time() - t0
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
            print('set solution is not feasible')
            raise RuntimeError('Failed to find feasible solution.')
        print('Feasible solution found with objective {}'.format(self.model.objVal))
        
        # reset the model to its original state
        for i in range(self.n):
            self.x[i].lb = -INF
            self.x[i].ub = INF
        ###commented out so as to now change solution####
        c_orig = np.copy(self.c)
        self.set_objective(np.zeros(self.n))                  
        self.model.optimize()
        if self.model.status != gp.GRB.Status.OPTIMAL:
            print('something weird with the return to normal')
            raise RuntimeError('Failed to set feasible solution.')                
        self.set_objective(c_orig)
        
                
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
        circuit= circuit*sympy.lcm([Fraction.from_float(float(circuit[i])).limit_denominator().denominator for i in range(self.n) if circuit[i] != 0])
        
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

    def second_vert(self, init_x, init_obj, num_dec_places, c=None, verbose=False, method='primal_simplex', **kwargs):
        vbasis = kwargs.get('vbasis', None)
        cbasis = kwargs.get('cbasis', None)
        simp_switch = kwargs.get('simp_switch', False)

        if c is None:
            # print(self.c)
            c = self.c
        assert self.c is not None, 'Need objective function'
         
        if self.model is None:
            self.build_gurobi_model(c=c)

        self.set_objective(c)
        if vbasis is not None and cbasis is not None:
            self.model.setAttr("VBasis", self.model.getVars(), vbasis)
            self. model.setAttr("CBasis", self.model.getConstrs(), cbasis)
        else: 
            i = 0
            for var in self.model.getVars():
                var.Start = init_x[i]
                i+=1
        self.model.update()
        self.set_method(method)
        t0 = time.time()

        # for var in self.model.getVars():
        #     print(var.Start)

        # print(self.c.dot(init_x))
            
        def obj_callback(model, where):
            if where == gp.GRB.Callback.SIMPLEX:
                obj = model.cbGet(gp.GRB.Callback.SPX_OBJVAL)
                prim_inf = model.cbGet(gp.GRB.Callback.SPX_PRIMINF)
                dual_inf = model.cbGet(gp.GRB.Callback.SPX_DUALINF)
                itcnt = model.cbGet(gp.GRB.Callback.SPX_ITRCNT)
                if abs(obj-init_obj)>= 10**(-(num_dec_places-2)) and (itcnt > 0) and (prim_inf == 0.0) and (dual_inf == 0.0):
                    model.terminate()

        self.model.optimize(obj_callback)
        x_optimal = self.model.getAttr('x', self.x)
        # if x_optimal != init_x:
        if not np.array_equal(x_optimal,init_x):
            return x_optimal
        elif self.model.status == gp.GRB.Status.OPTIMAL:
            return x_optimal
        else:
            raise RuntimeError('Failed to find new vertex.')

    def num_aft_dec(number): 
        # Convert number to string
        number_str = str(number)
    
        # Split string at the decimal point
        if '.' in number_str:
            # Get the part after the decimal point
            decimal_part = number_str.split('.')[1]
        
            # Remove trailing zeros
            decimal_part = decimal_part.rstrip('0')
        
            # Return the length of the remaining decimal part
            return len(decimal_part)
        else:
            # No decimal point in the number
            return 0

    def solve_lp_ws(self, x, c=None, verbose=False, record_objs=True, method='primal_simplex'):
    
        if c is None:
            assert self.c is not None, 'Need objective function'
            c = self.c
         
        if self.model is None:
            self.build_gurobi_model(c=c)

        i = 0
        for var in self.model.getVars():
            var.Start = x[i]
            i+=1

        self.set_objective(c)
        self.set_method(method)
        self.model.update()
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
        solve_time = self.model.getAttr('Runtime')
        output = result(0, x=x_optimal, obj=obj_optimal, n_iters=num_steps, solve_time=solve_time,
                        iter_times=iter_times, obj_values=obj_values, iter_counts=iter_counts)        
        return output

    def bfs_soln_chck(self, x):
        feasible = True
        for i in range(self.n):
            self.x[i].lb = x[i]
            self.x[i].ub = x[i]
        self.model.update()
        self.model.optimize()
        # if self.model.status == gp.GRB.Status.OPTIMAL or self.model.status == gp.GRB.status.FEASIBLE:
        #     active_constraints = []
        #     for cont in self.model.getConstrs():
        #         slack = cont.getAttr("slack")
        #         if abs(slack) < 1e-6:  # Adjust tolerance as needed
        #             active_constraints.append(cont)

        #     active_matrix = []
        #     for cont in active_constraints:
        #         coeffs = [v.getAttr("X") for v in self.model.getVars()]  # Replace with proper coefficient retrieval
        #         active_matrix.append(coeffs)

        #     active_matrix = np.array(active_matrix)

        #     # Check rank
        #     rank = np.linalg.matrix_rank(active_matrix)
        #     if rank == self.n:
        #         feasible = True



            # vbasis = self.model.getAttr(gp.GRB.Attr.VBasis)
    
            # # Count basic variables
            # num_basic_vars = sum(1 for v in vbasis if v == 0)  # 0 means 'basic'
            # # num_constraints = self.model.numConstrs

            # if num_basic_vars == self.n:
            #     feasible = True

        if self.model.status != gp.GRB.Status.OPTIMAL:
            feasible  = False
        
        # reset the model to its original state
        for i in range(self.n):
            self.x[i].lb = -INF
            self.x[i].ub = INF
        c_orig = np.copy(self.c)
        self.model.update()
        self.set_objective(np.zeros(self.n))                  
        self.model.optimize()
        # if self.model.status != gp.GRB.Status.OPTIMAL:
        #     feasible = False               
        self.set_objective(c_orig)
        return feasible
        
        # feasible = False
        # for i in range(self.n):
        #     self.x[i].lb = x[i]
        #     self.x[i].ub = x[i]
        # self.model.update()
        # self.model.optimize()
        # if self.model.status == gp.GRB.Status.OPTIMAL:
        #     vbasis = self.model.getAttr(gp.GRB.Attr.VBasis)
        #     cbasis = self.model.getAttr(gp.GRB.Attr.CBasis)
    
        #     # Count basic variables
        #     num_basic_vars = sum(1 for v in vbasis if v == 0)  # 0 means 'basic'

    
        #     if num_basic_vars == self.m_A:  # Dimension check
        #         feasible = True

        # if self.model.status != gp.GRB.Status.OPTIMAL:
        #     feasible  = False
        
        # reset the model to its original state
        # for i in range(self.n):
        #     self.x[i].lb = -INF
        #     self.x[i].ub = INF
        # c_orig = np.copy(self.c)
        # self.model.update()
        # self.set_objective(np.zeros(self.n))                  
        # self.model.optimize()
        # # if self.model.status != gp.GRB.Status.OPTIMAL:
        # #     feasible = False               
        # self.set_objective(c_orig)
        # return feasible