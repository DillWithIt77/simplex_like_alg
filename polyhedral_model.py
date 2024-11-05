import numpy as np
import gurobipy as gp
import contextlib
import time

from utils import METHODS, INF, EPS

class PolyhedralModel():
    
    # Given matrices B and A (and optional inds argument / objective function),
    # builds a polyhedral model for computing steepest-descent circuits
    # as a gurobi linear program model
    def __init__(self, B, A=None, active_inds=[], c=None, primal=True, method='dual_simplex', alt = False):
        
        print('Building polyhedral model. Solve method: {}'.format(method))
        
        self.model = gp.Model()
        self.primal = primal
        
        # add variables and constraints to the model
        self.m_B, self.n = B.shape
        if self.primal:
            self.x = []
            self.y_pos = []
            self.y_neg = []
            self.x_pos = []
            self.x_neg = []
            for i in range(self.n):
                self.x.append(self.model.addVar(lb=-INF, ub=INF, name='x_{}'.format(i)))
            for i in range(self.m_B):
                self.y_pos.append(self.model.addVar(lb=0.0, ub=1.0, name='y_pos_{}'.format(i)))
                self.y_neg.append(self.model.addVar(lb=0.0, ub=1.0, name='y_neg_{}'.format(i)))
                self.model.addConstr(gp.LinExpr(list(B[i]) + [-1, 1], 
                                                self.x + [self.y_pos[i], self.y_neg[i]]) == 0, 
                                                name='B_{}'.format(i))
            if alt:
                self.model.addConstr(gp.LinExpr([1]*(self.m_B), self.y_pos) <= 1, 
                                 name='alt_1_norm')
            else:
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
            
    def set_active_inds(self, active_inds,**kwargs):
        # switch = kwargs.get('simp_switch', False)
        init_inds = kwargs.get('inds', None)
        self.active_inds = active_inds
        for i in range(self.m_B):
            self.y_pos[i].lb = 0.0
            self.y_pos[i].ub = 1.0
            self.y_neg[i].lb = 0.0
            self.y_neg[i].ub = 1.0
        for i in self.active_inds:
            self.y_pos[i].ub = 0.0
        if init_inds is not None:
            for i in init_inds:
                self.y_neg[i].ub = 0.0 

                
    def set_method(self, method):
        self.method = method
        #with contextlib.redirect_stdout(None):
        self.model.Params.method = METHODS[method]
          
    # warm start the model with the provided solution   
    def set_solution(self, g):
        for i in range(self.n):
            self.x[i].lb = g[i]
            self.x[i].ub = g[i]
        for i in range(self.m_B):
            self.y_pos[i].ub = 1.0
            self.y_neg[i].ub = 1.0
            
        # solve the modified model to obtain desired solution    
        self.model.Params.method = 0
        self.model.optimize()
        
        # reset the model contraints back to its original state
        for i in range(self.n):
            self.x[i].lb = -INF
            self.x[i].ub = INF
        ####commented out as to not change solution#####
        self.model.setObjective(gp.LinExpr(np.zeros(self.n), self.x))                
        self.model.optimize()
        if self.model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Failed to set solution for polyhedral model') 
            
        self.set_objective(self.c) 
        self.set_active_inds(self.active_inds)
        self.set_method(self.method)              
        # self.model.update()
        
                
    def compute_sd_direction(self, verbose=False, **kwargs):
        init_y_pos = kwargs.get('y_pos', None)
        init_y_neg = kwargs.get('y_neg', None)
        init_edge = kwargs.get('edge', None)
        trivial = kwargs.get('trivial', False)

        warm_start_chck = kwargs.get('check', False)

        flag = 1 if verbose else 0
        self.model.setParam(gp.GRB.Param.OutputFlag, flag)

        self.set_objective(self.c)
        if trivial:
            print('trivial warm start')
            init_soln = np.append(np.append(np.zeros(self.n),init_y_pos),init_y_neg)
            i = 0
            for var in self.model.getVars():
                var.Start = init_soln[i]
                i +=1
            self.model.update()

        elif init_edge is not None and init_y_pos is not None and init_y_neg is not None:
            print('standard form warm start')
            init_soln = np.append(np.append(init_edge,init_y_pos),init_y_neg)
            i = 0
            for var in self.model.getVars():
                var.Start = init_soln[i]
                i +=1
            self.model.update()

        elif init_y_pos is not None and init_y_neg is not None:
            print('other warm start')
        
            for i in range(self.m_B):
                self.y_pos[i].lb = init_y_pos[i]
                self.y_pos[i].ub = init_y_pos[i]
                self.y_neg[i].lb = init_y_neg[i]
                self.y_neg[i].ub = init_y_neg[i]
            
            self.model.update()
            self.model.optimize()

            if self.model.status != gp.GRB.Status.OPTIMAL:
                print('Could not warm start with given edge')
            if self.model.status == gp.GRB.Status.INFEASIBLE:
                print("Model is infeasible.")

                # Compute IIS to find violated constraints
                self.model.computeIIS()
                print("The following constraints are infeasible:")
    
                for constr in self.model.getConstrs():
                    if constr.IISConstr:
                        print(f"Constraint {constr.ConstrName} is infeasible.")
            else:
                print("Model is feasible or optimization was successful.")

            if self.model.status == gp.GRB.Status.OPTIMAL:
                vbasis = self.model.getAttr(gp.GRB.Attr.VBasis)
                cbasis = self.model.getAttr(gp.GRB.Attr.CBasis)

            scaled_edge = self.model.getAttr('x', self.x)
            print(scaled_edge)

            for var in self.model.getVars():
                print(var.Start)

            self.model.setAttr("VBasis", self.model.getVars(), vbasis)
            self. model.setAttr("CBasis", self.model.getConstrs(), cbasis)
            for i in range(self.n):
                self.x[i].lb = -INF
                self.x[i].ub = INF
            self.set_objective(self.c)
            self.set_active_inds(self.active_inds)
            self.set_method(self.method)              
            self.model.update()

        if warm_start_chck:
            for var in self.model.getVars():
                if var.Start != 0:
                    print(var.Start)
        
        t0 = time.time()
        self.model._phase1_time = 0 #None
        self.model._is_dualinf = True
        def dualinf_callback(model, where):
            if where == gp.GRB.Callback.SIMPLEX:
                if model._is_dualinf:
                    dualinf = model.cbGet(gp.GRB.Callback.SPX_DUALINF)
                    if dualinf < EPS:
                        model._phase1_time = time.time() - t0
                        model._is_dualinf = False
        
        self.model.optimize(dualinf_callback)
        # print(f' model status: {self.model.status}')
        if self.model.status != gp.GRB.Status.OPTIMAL:
            raise RuntimeError('Failed to find steepst-descent direction.')
        
        phase2_time = time.time() - self.model._phase1_time
        phase_times = (self.model._phase1_time, phase2_time)
        g = self.model.getAttr('x', self.x)
        # print(f'circuit g: {g}')
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
    
        

