import sympy as sp
import numpy as np
import scipy.io as sio

from dynamics.biped_dynamics import BipedDynamics, Constraints
from trajectory.trajectory import PhaseTrajectory

class TransverseLinearization:
    def __init__(self, trj : PhaseTrajectory, constr : Constraints, d : BipedDynamics):
        
        self.dynamics = d
        self.constraints = constr
        self.trajectory = trj

        self.get_trans_funs()
        self.Ku = self.mat_file()


    def get_trans_funs(self):
        K = self.dynamics.params.K
        hip_mass = self.dynamics.params.hip_mass
        leg_mass = self.dynamics.params.leg_mass
        torso_mass = self.dynamics.params.torso_mass
        torso_com = self.dynamics.params.torso_com
        leg_length = self.dynamics.params.leg_length
        gravity_acceleration =  self.dynamics.params.gravity_acceleration

        q1, q2, q3 = sp.symbols('q1 q2 q3')
        dq1, dq2, dq3 = sp.symbols('dq1 dq2 dq3')
        ddq1, ddq2, ddq3 = sp.symbols('ddq1 ddq2 ddq3')
        v = sp.symbols('v')

        y1, y2, dy1, dy2, ddy1, ddy2 = sp.symbols('y1 y2 dy1 dy2 ddy1 ddy2')
        phi2, phi3, phi2_p, phi3_p, phi2_pp, phi3_pp = sp.symbols('phi_2 phi_3 phi_2\' phi_3\' phi_2\'\' phi_3\'\'')

        M = sp.Matrix([[(hip_mass + 5/4 * leg_mass + torso_mass)* leg_length**2, 1/2 * leg_mass * leg_length**2 * sp.cos(q1 - q2), torso_mass * leg_length * torso_com * sp.cos(q1 - q3)],
                       [-1/2*leg_mass* leg_length**2 * sp.cos(q1 - q2), 1/4 * leg_mass * leg_length**2, 0],
                       [torso_mass * leg_length * torso_com * sp.cos(q1 - q3), 0, torso_mass * torso_com**2]])
        
        C = sp.Matrix([[  0, - 1/2 * leg_mass * leg_length**2 *sp.sin(q1-q2) * dq2, torso_mass*leg_length*torso_com*sp.sin(q1 -q3)*dq3],
                       [  1/2 * leg_mass * leg_length**2 *sp.sin(q1-q2) * dq1, 0, 0],
                       [- torso_mass*leg_length*torso_com*sp.sin(q1 -q3)*dq1, 0, 0]])
        
        G = sp.Matrix([[(q1-q3)*K - (hip_mass + 3/2 *leg_mass + torso_mass) * leg_length * gravity_acceleration * sp.sin(q1)],
                       [(q2-q3)*K + 1/2 *leg_mass * gravity_acceleration*leg_length*sp.sin(q2)],
                       [(2*q3 - q2 - q1)*K - torso_mass*gravity_acceleration*torso_com*sp.sin(q3)]])
        
        B = sp.Matrix([[1],[-1],[0]])
        
        dq_vec = sp.Matrix([[dq1],
                            [dq2],
                            [dq3]])
        
        ddq_vec = sp.Matrix([[ddq1],
                             [ddq2],
                             [ddq3]])
        
        lin_comb_coeffs = sp.Matrix([[1, 1, -leg_length/torso_com * sp.cos(q1 -q3)]])
        assert (lin_comb_coeffs @ B)[0,0] == 0

        full_expression = lin_comb_coeffs @ (M @ ddq_vec + C @ dq_vec + G)
        full_expression = sp.simplify(sp.expand(full_expression))
        full_expression = full_expression.subs([(q2, y1 + phi2), (q3, y2 + phi3)])
        full_expression = full_expression.subs([(dq2, dy1 + phi2_p * dq1), (dq3, dy2 + phi3_p * dq1)])
        full_expression = full_expression.subs([(ddq2, v + phi2_p * ddq1 + phi2_pp * dq1**2), (ddq3, ddy2 + phi3_p * ddq1 + phi3_pp * dq1**2)])

        alpha = (lin_comb_coeffs.subs(q3, phi3) @ M.subs([(q2, phi2), (q3, phi3)]) @ sp.Matrix([[1],[phi2_p],[phi3_p]]))[0,0]
        _alpha_pttp = sp.lambdify([q1, phi2, phi3, phi2_p, phi3_p, phi2_pp, phi3_pp], alpha)
        self.alpha =  lambda theta: _alpha_pttp(theta, *self.constraints(theta)[0:6])
        
        beta = (lin_comb_coeffs.subs(q3, phi3) @ (M.subs([(q2, phi2), (q3, phi3)]) @ sp.Matrix([[0],[phi2_pp],[phi3_pp]]) + C.subs([(q2, phi2), (q3, phi3), (dq1, 1), (dq2, phi2_p), (dq3, phi3_p)]) @ sp.Matrix([[1],[phi2_p],[phi3_p]])))[0,0]
        _beta_pttp = sp.lambdify([q1, phi2, phi3, phi2_p, phi3_p, phi2_pp, phi3_pp], beta)
        self.beta = lambda theta: _beta_pttp(theta, *self.constraints(theta)[0:6])
        
        gamma = (lin_comb_coeffs.subs(q3, phi3) @ G.subs([(q2, phi2), (q3, phi3)]))[0,0]
        _gamma_pttp = sp.lambdify([q1, phi2, phi3, phi2_p, phi3_p, phi2_pp, phi3_pp], gamma)
        self.gamma = lambda theta: _gamma_pttp(theta, *self.constraints(theta)[0:6])

        v1 = sp.Matrix([[-phi2_p, 1, 0]])
        v2 = sp.Matrix([[-phi3_p, 0, 1]])

        Mi = M.inv()

        R1 = (v1 @ Mi @ (- G - C @ dq_vec))[0,0] - phi2_pp * dq1**2
        R1 = R1.subs([(q2, y1 + phi2), (q3, y2 + phi3)]).subs([(dq2, dy1 + phi2_p * dq1), (dq3, dy2 + phi3_p * dq1)])

        R2 = (v2 @ Mi @ (- G - C @ dq_vec))[0,0] - phi3_pp * dq1**2
        R2 = R2.subs([(q2, y1 + phi2), (q3, y2 + phi3)]).subs([(dq2, dy1 + phi2_p * dq1), (dq3, dy2 + phi3_p * dq1)])

        N1 = (v1 @ Mi @ B)[0,0]
        N1 = N1.subs([(q2, y1 + phi2), (q3, y2 + phi3)])

        N2 = (v2 @ Mi @ B)[0,0]
        N2 = N2.subs([(q2, y1 + phi2), (q3, y2 + phi3)])

        R = R2 - R1 * N2/N1

        def A(theta):

            dtheta, ddtheta = self.trajectory.theta_sp(theta)

            _A21 = ((dq1 * sp.diff(R, dq1) - ddq1 * sp.diff(R, q1)) / (2 * (dq1**2 + ddq1**2))).subs([(y1, 0), (y2, 0), (dy1, 0), (dy2, 0)])
            _A21_pttp = sp.lambdify([q1, dq1, ddq1, phi2, phi3, phi2_p, phi3_p, phi2_pp, phi3_pp], _A21)
            A21 = lambda theta, dtheta, ddtheta: _A21_pttp(theta, dtheta, ddtheta, *self.constraints(theta)[0:6])

            _A22 = sp.diff(R, y1).subs([(y1, 0), (y2, 0), (dy1, 0), (dy2, 0)])
            _A22_pttp = sp.lambdify([q1, dq1, ddq1, phi2, phi3, phi2_p, phi3_p, phi2_pp, phi3_pp], _A22)
            A22 = lambda theta, dtheta, ddtheta: _A22_pttp(theta, dtheta, ddtheta, *self.constraints(theta)[0:6])

            _A23 = sp.diff(R, y2).subs([(y1, 0), (y2, 0), (dy1, 0), (dy2, 0)])
            _A23_pttp = sp.lambdify([q1, dq1, ddq1, phi2, phi3, phi2_p, phi3_p, phi2_pp, phi3_pp], _A23)
            A23 = lambda theta, dtheta, ddtheta: _A23_pttp(theta, dtheta, ddtheta, *self.constraints(theta)[0:6])

            _A24 = sp.diff(R,dy1).subs([(y1, 0), (y2, 0), (dy1, 0), (dy2, 0)])
            _A24_pttp = sp.lambdify([q1, dq1, ddq1, phi2, phi3, phi2_p, phi3_p, phi2_pp, phi3_pp], _A24)
            A24 = lambda theta, dtheta, ddtheta: _A24_pttp(theta, dtheta, ddtheta, *self.constraints(theta)[0:6])

            _A25 = sp.diff(R,dy2).subs([(y1, 0), (y2, 0), (dy1, 0), (dy2, 0)])
            _A25_pttp = sp.lambdify([q1, dq1, ddq1, phi2, phi3, phi2_p, phi3_p, phi2_pp, phi3_pp], _A25)
            A25 = lambda theta, dtheta, ddtheta: _A25_pttp(theta, dtheta, ddtheta, *self.constraints(theta)[0:6])

            return [A21(theta, dtheta, ddtheta), A22(theta, dtheta, ddtheta), A23(theta, dtheta, ddtheta), A24(theta, dtheta, ddtheta), A25(theta, dtheta, ddtheta)]

        self.A = A

        _N = N2 / N1
        _N_pttp = sp.lambdify([q1, y1, y2, phi2, phi3, phi2_p, phi3_p], _N)
        self.N = lambda theta, y1, y2: _N_pttp(theta, y1, y2, *self.constraints(theta)[0:4])

        _N1_pttp = sp.lambdify([q1, y1, y2, phi2, phi3, phi2_p, phi3_p], N1)
        self.N1 = lambda theta, y1, y2: _N1_pttp(theta, y1, y2, *self.constraints(theta)[0:4])

        #TODO выполнить замену, чтобы оставить зависимость только от q1, dq1, y1, dy1, y2, dy2
        #TODO взять частные производные и получить функции для линеаризации

        # print(full_expression)
        # print(alpha * ddq1 + beta * dq1**2 + gamma)

        gi = alpha * ddq1 + beta * dq1**2 + gamma - full_expression[0,0]

        def ab(theta):
            dtheta, ddtheta = self.trajectory.theta_sp(theta)

            _gI = ((dq1 * sp.diff(gi, dq1) - ddq1 * sp.diff(gi, q1)) / (2 * (dq1**2 + ddq1**2))).subs([(y1, 0), (y2, 0), (dy1, 0), (dy2, 0), (v, 0)])
            _gI_pttp = sp.lambdify([q1, dq1, ddq1, phi2, phi3, phi2_p, phi3_p, phi2_pp, phi3_pp], _gI)
            gI = lambda theta, dtheta, ddtheta: _gI_pttp(theta, dtheta, ddtheta, *self.constraints(theta)[0:6])

            _gy1 = sp.diff(gi, y1).subs([(y1, 0), (y2, 0), (dy1, 0), (dy2, 0), (v, 0)])
            _gy1_pttp = sp.lambdify([q1, dq1, ddq1, phi2, phi3, phi2_p, phi3_p, phi2_pp, phi3_pp], _gy1)
            gy1 = lambda theta, dtheta, ddtheta: _gy1_pttp(theta, dtheta, ddtheta, *self.constraints(theta)[0:6])

            _gy2 = sp.diff(gi, y2).subs([(y1, 0), (y2, 0), (dy1, 0), (dy2, 0), (v, 0)])
            _gy2_pttp = sp.lambdify([q1, dq1, ddq1, phi2, phi3, phi2_p, phi3_p, phi2_pp, phi3_pp], _gy2)
            gy2 = lambda theta, dtheta, ddtheta: _gy2_pttp(theta, dtheta, ddtheta, *self.constraints(theta)[0:6])

            _gdy1 = sp.diff(gi, dy1).subs([(y1, 0), (y2, 0), (dy1, 0), (dy2, 0), (v, 0)])
            _gdy1_pttp = sp.lambdify([q1, dq1, ddq1, phi2, phi3, phi2_p, phi3_p, phi2_pp, phi3_pp], _gdy1)
            gdy1 = lambda theta, dtheta, ddtheta: _gdy1_pttp(theta, dtheta, ddtheta, *self.constraints(theta)[0:6])

            _gdy2 = sp.diff(gi, dy2).subs([(y1, 0), (y2, 0), (dy1, 0), (dy2, 0), (v, 0)])
            _gdy2_pttp = sp.lambdify([q1, dq1, ddq1, phi2, phi3, phi2_p, phi3_p, phi2_pp, phi3_pp], _gdy2)
            gdy2 = lambda theta, dtheta, ddtheta: _gdy2_pttp(theta, dtheta, ddtheta, *self.constraints(theta)[0:6])

            _gv = sp.diff(gi, v).subs([(y1, 0), (y2, 0), (dy1, 0), (dy2, 0), (v, 0)])
            _gv_pttp = sp.lambdify([q1, dq1, ddq1, phi2, phi3, phi2_p, phi3_p, phi2_pp, phi3_pp], _gv)
            gv = lambda theta, dtheta, ddtheta: _gv_pttp(theta, dtheta, ddtheta, *self.constraints(theta)[0:6])

            a = np.array([gI(theta, dtheta, ddtheta) - self.beta(theta), gy1(theta, dtheta, ddtheta), gy2(theta, dtheta, ddtheta), gdy1(theta, dtheta, ddtheta), gdy2(theta, dtheta, ddtheta)]) * 2*dtheta / self.alpha(theta)

            return a, gv(theta, dtheta, ddtheta) * 2*dtheta / self.alpha(theta)
        
        # print(ab(0))
        
        self.ab = ab

    def matrix(self, i):
        theta = self.trajectory.theta[i]

        A = np.zeros((5,5))
        B = np.zeros((5,1))

        a, b = self.ab(theta)
        A2 = self.A(theta)
        B2 = self.N(theta, 0, 0)
        A[0] = a
        A[4] = A2
        A[1,3] = A[2,4] = 1

        B[0,0] = b
        B[4,0] = B2
        B[3,0] = 1

        return A, B
    
    def mat_file(self):

        n = len(self.trajectory.t)
        A = np.zeros((n, 5, 5))
        B = np.zeros((n, 5, 1))

        for i in range(n):
            A[i], B[i] = self.matrix(i)

        A = np.transpose(A, (1,2,0))
        B = np.transpose(B, (1,2,0))

        sio.savemat("matrix.mat", {'t':self.trajectory.t, 'A':A, 'B':B})  

        P_real = sio.loadmat("fbcoeffs.mat")['P_real']
        K = np.zeros((len(self.trajectory.t), 1, 5))
        for i in range(len(self.trajectory.t)):
            K[i] = B[:, :, i].T @ P_real[:,:,i]
        
        print(K[0])
        return K

  

# def get_coeffs(trj : PhaseTrajectory, constr : Constraint, d : Dynamics):
#   A, B = get_transverse_linearization(trajectory, solution['theta_constraint'], solution['dynamics'])
#   A = np.transpose(A, (1,2,0))
#   B = np.transpose(B, (1,2,0))

#   sio.savemat(name_mat, {'t':trajectory.t, 'A':A, 'B':B})

#   P_real = sio.loadmat(name_coeffs)['P_real']
#   K = np.zeros((len(trajectory.t), 1, 3))

#   for i in range(len(trajectory.t)):
#       K[i] = B[:, :, i].T @ P_real[:,:,i]
  
#   return K

