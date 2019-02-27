#!/usr/bin/python

from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import root
from scipy.constants import epsilon_0, mu_0

class coplanar_coupler:
    def __init__(self, s1=None,s2=None,w1=None,w2=None,w3=None):
        self.s1 = s1
        self.s2 = s2
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.port = 'notch'
    
    def z_branch_points_cpw(self, s1=None, s2=None, w1=None, w2=None, w3=None):
        if self.port == 'notch':
            return self.z_branch_points_notch(s1,s2,w1,w2,w3)
        elif self.port == 'butt':
            return self.z_branch_points_butt(s1,s2,w1,w2,w3)
    
    # 2 coplanar waveguides on the real axis
    def z_branch_points_notch(self, s1=None,s2=None,w1=None,w2=None,w3=None):
        if not s1: s1 = self.s1
        if not s2: s2 = self.s2
        if not w1: w1 = self.w1
        if not w2: w2 = self.w2
        if not w3: w3 = self.w3
                
        z_B = 0
        z_C = z_B+s2
        z_D = z_C+w2
        z_E = z_D+s2
        z_F = z_E+w3
        z_G = z_F+s1
        z_H = z_G+w1
        z_I = z_H+s1    
        return [np.complex(x) for x in [z_B, z_C, z_D, z_E, z_F, z_G, z_H, z_I]]

    def z_branch_points_butt(self, s1=None, s2=None, w1=None, w2=None, w3=None):
        if not s1: s1 = self.s1
        if not s2: s2 = self.s2
        if not w1: w1 = self.w1
        if not w2: w2 = self.w2
            
        z_B = -w1/2.-s1
        z_C = -w1/2.
        z_D = -w2/2.-s2
        z_E = -w2/2.
        z_F = w2/2.
        z_G = w2/2.+s2
        z_H = w1/2.
        z_I = w1/2.+s1
        
        return [np.complex(x) for x in [z_B, z_C, z_D, z_E, z_F, z_G, z_H, z_I]]
    
    def conformal_transform_difference_by_parts(self,z0,z1,branch_point1,branch_point2,branch_points,zero_points):
        # 1) solving integral part (positive)
        f1 = lambda z: np.log(z-(branch_points[branch_point1] + branch_points[branch_point2])/2 + \
                           np.sqrt((z-branch_points[branch_point1])*(z-branch_points[branch_point2]))) * \
                           np.prod([z-zero_point for zero_point in zero_points]) / \
                           np.prod([np.sqrt(z-branch_point) for branch_point_id, branch_point \
                           in enumerate(branch_points) if branch_point_id not in [branch_point1, branch_point2]])
        f2 = lambda z: np.log(z-(branch_points[branch_point1] + branch_points[branch_point2])/2 + \
                           np.sqrt((z-branch_points[branch_point1])*(z-branch_points[branch_point2]))) * \
                           np.sum([np.prod([(z-zero_point) for zero_point_id, zero_point in enumerate(zero_points) \
                           if zero_point_id != diff_zero_point_id])/ \
                           np.prod([np.sqrt(z-branch_point) for branch_point_id, branch_point \
                           in enumerate(branch_points) if branch_point_id not in [branch_point1, branch_point2]]) \
                           for diff_zero_point_id, diff_zero_point in enumerate(zero_points)]+ \
                           [np.prod([z-zero_point for zero_point in zero_points]) / \
                           np.prod([np.sqrt(z-branch_point) for branch_point_id, branch_point \
                           in enumerate(branch_points) if branch_point_id not in [branch_point1, branch_point2]]) / \
                           (z-diff_branch_point)*(-0.5) for diff_branch_point_id, diff_branch_point in enumerate(branch_points) \
                           if diff_branch_point_id not in [branch_point1, branch_point2] ])

        integral_part = f1(z1)-f1(z0)
        numerical_part = np.complex(*tuple(integrate.quad(lambda t: f2(t*(z1-z0)+z0).real, 0, 1)))+1j*np.complex(*tuple(integrate.quad(lambda t: f2(t*(z1-z0)+z0).imag, 0, 1)))

        return integral_part - (z1-z0)*numerical_part   

    def w_special_points(self, z_branch_points, z_zero_points):
        w_branch_points = []
        for branch_point_id, z_branch_point in enumerate(z_branch_points):
            if branch_point_id==0:
                w_branch_points.append(0)
            else:            
                w_branch_points.append(w_branch_points[branch_point_id-1] + \
                self.conformal_transform_difference_by_parts(\
                z_branch_points[branch_point_id-1],z_branch_points[branch_point_id],\
                branch_point_id-1,branch_point_id,z_branch_points,z_zero_points))

        w_zero_points = []        
        for zero_point_id, z_zero_point in enumerate(z_zero_points):
            nearest_branch_points = np.argsort(np.abs(np.asarray(z_branch_points)-z_zero_point))[:2]
            w_zero_points.append(w_branch_points[nearest_branch_points[0]] + \
                self.conformal_transform_difference_by_parts(\
                    z_branch_points[nearest_branch_points[0]],\
                    z_zero_point,\
                    nearest_branch_points[0],\
                    nearest_branch_points[1],\
                    z_branch_points,
                    z_zero_points))
        return w_branch_points, w_zero_points
    
    def find_zero_points(self, z_branch_points, z_zero_points_initial, z_zero_point_types, constraint_point_ids, constraint_types):
        #z_B,z_C,z_D,z_E,z_F,z_G,z_H,z_I = z_branch_points           
        def constraint(t_zero_points):
            z_zero_points = [t if point_type=='real' else 1j*t for t,point_type in zip(t_zero_points, z_zero_point_types)]
            constraint_values = [ self.conformal_transform_difference_by_parts(\
                                    z_branch_points[constraint_point_pair[0]], z_branch_points[constraint_point_pair[1]],\
                                    constraint_point_pair[0], constraint_point_pair[1],\
                                    z_branch_points,z_zero_points).real if constraint_types[constraint_id] == 'real' else \
                                  self.conformal_transform_difference_by_parts(\
                                    z_branch_points[constraint_point_pair[0]], z_branch_points[constraint_point_pair[1]],\
                                    constraint_point_pair[0], constraint_point_pair[1],\
                                    z_branch_points,z_zero_points).imag
                                      for constraint_id, constraint_point_pair in enumerate(constraint_point_ids)]
            return constraint_values
        
        self.zero_points = root(constraint, z_zero_points_initial).x
        return self.zero_points
    
    def find_zero_points_slow(self, z_branch_points, z_zero_points_initial, z_zero_point_types, constraint_point_ids, constraint_types):
        def constraint(t_zero_points):
            w_branch_points, w_zero_points = self.w_special_points(z_branch_points, \
            [t if point_type=='real' else 1j*t for t,point_type in zip(t_zero_points, z_zero_point_types)]) 
            z_B,z_C,z_D,z_E,z_F,z_G,z_H,z_I = z_branch_points           
            w_B,w_C,w_D,w_E,w_F,w_G,w_H,w_I = w_branch_points
            constraint_values = \
            [w_branch_points[constraint_point_id[0]].real-w_branch_points[constraint_point_id[1]].real \
             if constraint_types[constraint_id] == 'real' else \
             w_branch_points[constraint_point_id[0]].imag-w_branch_points[constraint_point_id[1]].imag \
             for constraint_id, constraint_point_id in enumerate(constraint_point_ids)]
            return constraint_values

        self.zero_points = root(constraint, z_zero_points_initial).x
        return self.zero_points

    def conformal_transform_derivative(self, z,z_branch_points,z_zero_points):
        result = 1
        for branch_point_id, branch_point in enumerate(z_branch_points):
            result /= np.sqrt(z-branch_point)
        for zero_point_id, zero_point in enumerate(z_zero_points):
            result *= (z-zero_point)
        return result
    
    def coupling_matrices_dimensionless(self, s1=None,s2=None,w1=None,w2=None,w3=None):
        if not s1: s1 = self.s2
        if not s2: s2 = self.s1
        if not w1: w1 = self.w1
        if not w2: w2 = self.w2
        if not w3: w3 = self.w3
        z_branch_points = self.z_branch_points_cpw(s1,s2,w1,w2,w3)
        z_zero_points_initial = [(z_branch_points[4]+z_branch_points[5]).real/2, \
                                 (z_branch_points[6]+z_branch_points[7]).real/2]
        z_zero_points = self.find_zero_points(z_branch_points, z_zero_points_initial, ['real', 'real'], \
                                                [(4,5),(6,7)], ['imag', 'imag'])
        w_branch_points, w_zero_points = self.w_special_points(z_branch_points, z_zero_points)
        w_B,w_C,w_D,w_E,w_F,w_G,w_H,w_I = w_branch_points
        c1_ = [(w_C.real-w_D.real)/(w_C.imag-w_B.imag), (w_G.real-w_H.real)/(w_C.imag-w_B.imag)]

        if not s1: s1 = self.s2
        if not s2: s2 = self.s1
        if not w1: w1 = self.w2
        if not w2: w2 = self.w1
        if not w3: w3 = self.w3
            
        z_branch_points = self.z_branch_points_cpw(s2,s1,w2,w1,w3)
        z_zero_points_initial = [(z_branch_points[4]+z_branch_points[5]).real/2, \
                                 (z_branch_points[6]+z_branch_points[7]).real/2]
        z_zero_points = self.find_zero_points(z_branch_points, z_zero_points_initial, ['real', 'real'], \
                                                [(4,5),(6,7)], ['imag', 'imag'])
        w_branch_points, w_zero_points = self.w_special_points(z_branch_points, z_zero_points)
        w_B,w_C,w_D,w_E,w_F,w_G,w_H,w_I = w_branch_points
        c2_ = [(w_G.real-w_H.real)/(w_C.imag-w_B.imag), (w_C.real-w_D.real)/(w_C.imag-w_B.imag)]

        C_dimensionless = np.asarray([c1_, c2_])
        L_dimensionless = np.linalg.inv(np.asarray(C_dimensionless))
        
        self.C_dimensionless = C_dimensionless
        self.L_dimensionless = L_dimensionless
        return C_dimensionless, L_dimensionless
    
    #def coupling_matrices_dimensionless_full(self, s1=None,s2=None,w1=None,w2=None,w3=None):
    def coupling_matrices_2lines(self, z_branch_points):
        conductors = [{'all_cond': ((1,2),(3,4),), 'grounded':((3,4),), 'conductor':(1,2), 'groundings':((4,5),)},\
                      {'all_cond': ((1,2),(3,4),), 'grounded':((1,2),), 'conductor':(3,4), 'groundings':((0,1),)}]
        
        C = []
        for conductor in conductors:
            z_zero_points_initial = [np.mean([z_branch_points[z] for z in g]).real\
                                     for g in conductor['groundings']]
            print (z_zero_points_initial, z_branch_points)
            z_zero_points = self.find_zero_points(z_branch_points, z_zero_points_initial, ['real',], \
                                                  conductor['groundings'], ['imag',])
            print (z_zero_points)
            w_branch_points, w_zero_points = self.w_special_points(z_branch_points, z_zero_points)
            C.append([(w_branch_points[c[0]]-w_branch_points[c[1]]).real / \
                      (w_branch_points[conductor['conductor'][0]]-w_branch_points[0]).imag \
                      for c in conductor['all_cond']])

        C_dimensionless = np.asarray(C)
        L_dimensionless = np.linalg.inv(np.asarray(C_dimensionless))
        
        self.C_dimensionless = C_dimensionless
        self.L_dimensionless = L_dimensionless
        return C_dimensionless, L_dimensionless
    
    def coupling_matrices_dimensionless_full(self, z_branch_points):
        conductors = [{'all_cond': ((1,2),(3,4),(5,6)), 'grounded':((3,4),(5,6)), 'conductor':(1,2), 'groundings':((4,5),(6,7))},\
                      {'all_cond': ((1,2),(3,4),(5,6)), 'grounded':((1,2),(5,6)), 'conductor':(3,4), 'groundings':((0,1),(6,7))},\
                      {'all_cond': ((1,2),(3,4),(5,6)), 'grounded':((1,2),(3,4)), 'conductor':(5,6), 'groundings':((0,1),(2,3))}]
        
        #z_branch_points = self.z_branch_points_cpw(s1,s2,w1,w2,w3)
        C = []
        for conductor in conductors:
            z_zero_points_initial = [np.mean([z_branch_points[z] for z in g]).real\
                                     for g in conductor['groundings']]
            z_zero_points = self.find_zero_points(z_branch_points, z_zero_points_initial, ['real', 'real'], \
                                                  conductor['groundings'], ['imag', 'imag'])
            w_branch_points, w_zero_points = self.w_special_points(z_branch_points, z_zero_points)
            C.append([(w_branch_points[c[0]]-w_branch_points[c[1]]).real / \
                      (w_branch_points[conductor['conductor'][0]]-w_branch_points[0]).imag \
                      for c in conductor['all_cond']])

        C_dimensionless = np.asarray(C)
        L_dimensionless = np.linalg.inv(np.asarray(C_dimensionless))
        
        self.C_dimensionless = C_dimensionless
        self.L_dimensionless = L_dimensionless
        return C_dimensionless, L_dimensionless
    
    def LC(self, a, b):
        import cmath
        import math
        from scipy.special import ellipk, ellipkinc
        if len(a) != len(b):
            raise ValueError('a should be same length as b.')
        if len(a) < 2:
            raise ValueError('system should have at least one non-ground conductor.')
            
        def w_stitch_sequence(z, a, b):
            if len(a):
                w1 = np.sqrt((z-a[0])*(z-b[0]))
                return w_stitch_sequence(w1, [np.sqrt((a_i-a[0])*(b_i-b[0])) for a_i, b_i in zip(a[1:], b[1:])])
            else:
                return z
        
        C = []
        
        for conductor_id in range(len(a)-1):
            # remapping a's and b's to wa, wa.
            c_a = [v for i, v in enumerate(a) if (i != conductor_id) and (i != conductor_id+1)]
            c_b = [v for i, v in enumerate(b) if (i != conductor_id) and (i != conductor_id+1)]
            w1a = [w_stitch_sequence(v, c_a, c_b) for v in a]
            w1b = [w_stitch_sequence(v, c_a, c_b) for v in b]
        
            # do a transform w2 =a/(w1-e)+b
            # e is chosen such that we get a symmetric CPW
            a0 = w1a[conductor_id]
            a1 = w1a[conductor_id+1]
            b0 = w1b[conductor_id]
            b1 = w1b[conductor_id+1]
            # equation for e:
            # (a1-z)(b1-z)*((a0+b0)/2-z)=(a0-z)(b0-z)*((a1+b1)/2-z)
            alpha = b0+a1-b1-a0
            if alpha > 0:
                beta = (a1*b0-a0*b1)/alpha
                gamma = (a0*b0*a1-a0*b0*b1+b0*a1*b1-a0*a1*b1)/alpha
                e1 = beta+cmath.sqrt(beta**2-gamma)
                e2 = beta-cmath.sqrt(beta**2-gamma)
            else:
                e1 = 0
                e2 = 0
            
            print (1/(e1-a0)-1/(e1-b0), 1/(e1-a1)-1/(e1-b1))
            print (1/(e2-a0)-1/(e2-b0), 1/(e2-a1)-1/(e2-b1))
            
            A = 1/(1/(e1-a0)-1/(e1-b1))
            B = (1/(e1-a0)+1/(e1-b1))/2
            
            w2a = [A/(e1-w1)+B for w1 in w1a]
            w2b = [A/(e1-w1)+B for w1 in w1b]
            
            k = abs(1/(A/(e1-a1)+B))
            
            Kk = ellipk(k**2)
            Kkp = ellipk(1-k**2)
            phi = Kkp/2
            
            print (w2a, w2b)
            
            for c2id in range(len(a)-1):
                print (w2b[c2id], w2a[c2id+1])
                C.append([(ellipkinc(math.asin(w2b[c2id]),k**2)-ellipkinc(math.asin(w2a[c2id+1]),k**2))/phi for c2id in range(len(a)-1)])
            
            print (w2a, w2b, k, Kk, Kkp, C)
        C = np.asarray(C)
        L = np.linalg.inv(C)
        return L,C
            
            
    def coupling_matrices(self,epsilon_eff=None,s1=None,s2=None,w1=None,w2=None,w3=None,mode='notch'):
        if not epsilon_eff: epsilon_eff = self.epsilon_eff
        if mode=='notch':
            z_branch_points = self.z_branch_points_notch(s1=None,s2=None,w1=None,w2=None,w3=None)
            C_dimensionless, L_dimensionless = self.coupling_matrices_dimensionless_full(z_branch_points)
            C_dimensionless = np.asarray(C_dimensionless)[[[0,2],[0,2]], [[0,0],[2,2]]]
            L_dimensionless = np.linalg.inv(C_dimensionless)
            #C_dimensionless, L_dimensionless = self.coupling_matrices_dimensionless(s1=None,s2=None,w1=None,w2=None,w3=None)
        else:
            z_branch_points = self.z_branch_points_butt(s1=None,s2=None,w1=None,w2=None,w3=None)
            C_dimensionless, L_dimensionless = self.coupling_matrices_dimensionless_full(z_branch_points)
            
        C_l = 2*C_dimensionless*epsilon_0*epsilon_eff
        L_l = L_dimensionless*mu_0/2
        coupling_matrix = L_dimensionless*mu_0/2/np.sqrt(epsilon_0*mu_0*epsilon_eff)
        
        return C_l, L_l, coupling_matrix
    