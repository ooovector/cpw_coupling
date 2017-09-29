from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import root
from scipy.constants import epsilon_0, mu_0
import sympy

class transmission_line_system_element:
    def __init__(self):
        pass
    
class resistor:
    def num_terminals(self):
        return 2
    def num_degrees_of_freedom(self):
        return 0
    def boundary_condition(self, omega):
        return np.asarray([[1, -1, 0, self.R], [0,0,1,-1]])
    def derivative(self):
        return (np.asarray([[]]),np.asarray([[]]))
    def __init__(self, R=None):
        self.R = R
        pass
    
class capacitor:
    def num_terminals(self):
        return 2
    def num_degrees_of_freedom(self):
        return 0
    def boundary_condition(self, omega):
        return np.asarray([[1j*omega*self.C, -1j*omega*self.C, 0, 1], [0,0,1,-1]])
    def __init__(self, C=None):
        self.C = C
        pass
    
class inductor:
    def num_terminals(self):
        return 2
    def num_degrees_of_freedom(self):
        return 0
    def boundary_condition(self, omega):
        return np.asarray([[1, -1, 0, 1j*omega*self.L], [0,0,1,-1]])
    def __init__(self, L=None):
        self.L = L
        pass
    
class short:
    def num_terminals(self):
        return 1
    def num_degrees_of_freedom(self):
        return 0
    def boundary_condition(self, omega):
        return np.asarray([[1, 0]])
    def __init__(self):
        pass
    
class port:
    def num_terminals(self):
        return 1
    def num_degrees_of_freedom(self):
        return 1
    def boundary_condition(self, omega):
        return np.asarray([[1,0,self.Z0], [0,1,1]])
    def __init__(self, Z0=None):
        self.Z0 = Z0
    
class transmission_line:
    def num_terminals(self):
        return 2
    def num_degrees_of_freedom(self):
        return 2
    def boundary_condition(self, omega):
        v_inv = (1j*self.Rl+omega*self.Ll)**(1./2.)*(1j*self.Gl+omega*self.Cl)**(1./2.)
        Y0 = ((self.Gl-1j*omega*self.Cl)/(self.Rl-1j*omega*self.Ll))**(1./2.)
        
        if (type(omega).__module__ == np.__name__):
            return np.asarray([[1, 0, 0, 0, -1/Y0, -1/Y0],\
                           [0, 1, 0, 0, -np.exp(1j*v_inv*self.l)/Y0, -np.exp(-1j*v_inv*self.l)/Y0], \
                           [0, 0, 1, 0, -1, 1],\
                           [0, 0, 0, 1, np.exp(1j*v_inv*self.l), -np.exp(-1j*v_inv*self.l)]])
        elif (type(omega).__module__ == sympy.symbol.__name__):
            return sympy.Matrix([[1, 0, 0, 0, -1/Y0, -1/Y0],\
                           [0, 1, 0, 0, -sympy.exp(sympy.I*v_inv*self.l)/Y0, -sympy.exp(-sympy.I*v_inv*self.l)/Y0], \
                           [0, 0, 1, 0, -1, 1],\
                           [0, 0, 0, 1, sympy.exp(sympy.I*v_inv*self.l), -sympy.exp(-sympy.I*v_inv*self.l)]])
            
    def __init__(self, l=None, Ll=None, Cl=None, Rl=None, Gl=None):
        self.l = l
        self.Ll = Ll
        self.Cl = Cl
        self.Rl = Rl
        self.Gl = Gl
        pass

class transmission_line_coupler:
    def num_terminals(self):
        return self.n*2
    def num_degrees_of_freedom(self):
        return self.n*2
    def propagating_modes(self):
        M = np.hstack((np.vstack((self.Rl, self.Cl)), np.vstack((self.Ll, self.Gl))))
        if M.dtype == object:
            M = sympy.Matrix(M)
            mode_amplitudes, cl = M.diagonalize()
            cl = sympy.Matrix([cl[i,i] for i in range(M.shape[0])])
        else:
            cl, mode_amplitudes = np.linalg.eig(M)
        gammas = cl/sympy.I
        modes = []
        for mode_id, gamma in enumerate(gammas):
            modes.append((gamma, mode_amplitudes[:,mode_id]))
        return modes
    
    def boundary_condition(self, omega):
        boundary_condition_matrix = np.zeros((self.num_terminals()*2, self.num_terminals()*2+self.num_degrees_of_freedom()), dtype=type(omega))
        boundary_condition_matrix[:, :self.num_terminals()*2] = np.identity(self.num_terminals()*2)
        #boundary_condition_matrix[self.num_terminals():, :self.num_terminals()] = np.identity(self.num_terminals())
        
        if (type(omega).__module__ == np.__name__):
            exp = np.exp
        elif (type(omega).__module__ == sympy.symbol.__name__):
            exp = sympy.exp
            boundary_condition_matrix = sympy.Matrix(boundary_condition_matrix)
        for mode_pair_id, mode_pair in enumerate(self.propagating_modes()):
            
            boundary_condition_matrix[       0:self.n,  self.n*4+mode_pair_id] = -np.asarray(mode_pair[1][:self.n])
            boundary_condition_matrix[  self.n:self.n*2,self.n*4+mode_pair_id] = -np.asarray(mode_pair[1][:self.n])*exp(mode_pair[0]*self.l*omega)
            boundary_condition_matrix[self.n*2:self.n*3,self.n*4+mode_pair_id] = np.asarray(mode_pair[1][self.n:])
            boundary_condition_matrix[self.n*3:        ,self.n*4+mode_pair_id] = -np.asarray(mode_pair[1][self.n:])*exp(mode_pair[0]*self.l*omega)
        return boundary_condition_matrix
    
    def __init__(self, n=2, l=None, Ll=None, Cl=None, Rl=None, Gl=None):
        self.n = n
        self.l = l
        self.Ll = Ll
        self.Cl = Cl
        self.Rl = Rl
        self.Gl = Gl
        pass

class transmission_line_system:
    def __init__(self):
        self.nodes = []
        self.elements = []
        self.node_multiplicity = {}
        self.terminal_node_mapping = []
        self.dof_mapping = []
        
    def matrix_reduce(self, matrix):
        row_indeces = []
        column_indeces = []
        
        used_rows = []
        used_columns = []
        
        blocks = []
        limiter = 0
        while len(used_rows) < matrix.shape[0]:
            this_rows = []
            this_columns = []
            # initial row:
            for row in range(matrix.shape[0]):
                if row not in used_rows:
                    this_rows = [row]
                    limiter +=1
                    #print('rows ', this_rows)
                    break
                    
            restart = True # continue this loop until nothing is added
            while restart:
                restart = False
                for i in range(matrix.shape[0]): # loop over rows
                    if i in this_rows:
                        for j in range(matrix.shape[0]):
                            if matrix[i,j] != 0 and j not in this_columns:
                                this_columns.append(j)
                                restart = True
                                limiter +=1
                                #print('columns ', this_columns, ' i ', i, ' j ', j)
                                break
                            
                    for j in this_columns: # loop over columns
                        if matrix[i,j] != 0 and i not in this_rows:
                            this_rows.append(i)
                            restart = True
                            limiter +=1
                            #print('rows ', this_rows, ' i ', i, ' j ', j)
                            break
                if limiter> 1000000:
                    return None
                
            used_rows.extend(this_rows)
            blocks.append([tuple(this_rows), tuple(this_columns), matrix[this_rows, this_columns]])    
            
        return blocks
        
    def create_boundary_problem_matrix(self, omega):
        # count nodes
        self.dof_mapping = [n for n in self.nodes] # nodal voltages
        self.dof_mapping.extend([(e_id, p_id) for e_id, e in enumerate(self.elements) for p_id in range(e.num_terminals())]) 
                                                     # currents incident into each terminal
        self.dof_mapping.extend([(e_id, int_dof_id) for e_id, e in enumerate(self.elements) for int_dof_id in range(e.num_degrees_of_freedom())])
                                                     # number of element-internal degrees of freedom
            
        # full dof number
        num_dof = len(self.dof_mapping)
        
        # number of nodes
        node_no = len(self.nodes)
        # number of internal dofs
        internal_dof_no = np.sum(e.num_degrees_of_freedom() for e in self.elements)        
        # number of terminals
        terminal_no = np.sum(e.num_terminals() for e in self.elements)
        
        # dynamic equations reflect the element's IV characteristic
        dynamic_equation_no = terminal_no + internal_dof_no
        # kinetic equations are Kirchhof's law that the sum of nodal currents is zero
        kinetic_equation_no = node_no
        
        num_equations = dynamic_equation_no +kinetic_equation_no
        
        boundary_condition_matrix = np.zeros((num_equations, num_dof), dtype=type(omega))
            
        # filling dynamic equations
        equation_id = 0
        current_offset = 0
        internal_dof_offset = 0
        for e_id, e in enumerate(self.elements):
            equations = e.boundary_condition(omega)
            for element_equation_id in range(equations.shape[0]):
                equation = equations[element_equation_id, :]
                for terminal_id, terminal_node in enumerate(self.terminal_node_mapping[e_id]):
                    node_id = self.nodes.index(terminal_node)
                    boundary_condition_matrix[equation_id, node_id] = equation[terminal_id] #nodal voltages
                    boundary_condition_matrix[equation_id, node_no+current_offset+terminal_id] = equation[terminal_id+e.num_terminals()] #nodal current
                for internal_dof_id in range(e.num_degrees_of_freedom()):
                    boundary_condition_matrix[equation_id, node_no+terminal_no+internal_dof_offset+internal_dof_id] = equation[2*e.num_terminals() + internal_dof_id]
                equation_id += 1
            internal_dof_offset += e.num_degrees_of_freedom()
            current_offset += e.num_terminals()
            
        full_terminal_id = 0
        # filling kinetic equations
        for e_id, e in enumerate(self.elements):
            for terminal_id, node in enumerate(self.terminal_node_mapping[e_id]):
                boundary_condition_matrix[dynamic_equation_no+self.nodes.index(node), node_no+full_terminal_id] = 1
                full_terminal_id += 1
        
        return boundary_condition_matrix
    
    #def solve_eigenmodes_perturbative(self, unperturbed_system):
        
    def add_element(self, element, nodes):
        self.elements.append(element)
        for node in nodes:
            if node not in self.nodes:
                self.node_multiplicity[node] = 0
                self.nodes.append(node)
            self.node_multiplicity[node] += 1
        self.terminal_node_mapping.append(nodes)
        return
    
def matrix_list_determinant_derivative(Ms, variable):
    out = []
    def check_dependence(exp, symbol):
        if exp == symbol:
            return True
        for sub in exp.args:
            if check_dependence(sub, symbol):
                return True
        return False
        
    for M in Ms:
        for row_id in range(M.shape[0]):
            # check if there are nonzero columns
            dependence = False
            for column_id in range(M.shape[1]):
                if check_dependence(M[row_id, column_id], variable):
                    dependence = True
            if dependence:
                DM = sympy.Matrix(M)
                for column_id in range(M.shape[1]):
                    DM[row_id,column_id] = sympy.diff(DM[row_id,column_id], variable)
                out.append(DM)
        
    return out

def matrix_list_determinant(Ms):
    out =sympy.zeros(1)[0]
    for M in Ms:
        out += M.det()
        
    return out

def matrix_list_subs(Ms, subs):
    out = []
    for M in Ms:
        replacement = sympy.zeros(*M.shape)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                replacement[i,j] = M[i,j].xreplace(subs)
        out.append(replacement)
        
    return out