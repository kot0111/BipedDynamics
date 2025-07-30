import numpy as np
import scipy as sp

from dynamics.parameters import BipedParameters

import matplotlib.pyplot as plt
import os.path


# def cross(a, b):
#     return a[0] * b[1] - a[1] * b[0]

class BipedDynamics:

    def __init__(self, params : BipedParameters):
        self.params = params

    def __eval_M(self, state):
        q1, q2, q3, dq1, dq2, dq3 = state

        leg_mass = self.params.leg_mass
        hip_mass = self.params.hip_mass
        torso_mass = self.params.torso_mass
        leg_length = self.params.leg_length
        torso_com = self.params.torso_com

        M = np.zeros((3,3))
        M[0, 0] = (hip_mass + (5/4) * leg_mass + torso_mass) * leg_length**2
        M[0, 1] = - 0.5 * leg_mass * leg_length**2 * np.cos(q1 - q2)
        M[0, 2] = torso_mass * leg_length * torso_com * np.cos(q1 - q3)

        M[1, 0] = - 0.5 * leg_mass * leg_length**2 * np.cos(q1 - q2)
        M[1, 1] = 0.25 * leg_mass * leg_length**2

        M[2, 0] = torso_mass * leg_length * torso_com * np.cos(q1 - q3)
        M[2, 2] = torso_mass * torso_com**2

        return M

    def __eval_Impact(self, state):
        
        q1, q2, q3, dq1, dq2, dq3 = state

        leg_mass = self.params.leg_mass
        hip_mass = self.params.hip_mass
        torso_mass = self.params.torso_mass
        leg_length = self.params.leg_length
        torso_com = self.params.torso_com

        Me11 = self.__eval_M(state)
        Me12 = np.zeros((3,2))

        Me12[0,0] =   (1.5 * leg_mass +  hip_mass + torso_mass) * leg_length * np.cos(q1) #De14
        Me12[0,1] = - (1.5 * leg_mass +  hip_mass + torso_mass) * leg_length * np.sin(q1) #De15
        Me12[1,0] = - 0.5 * leg_mass * leg_length * np.cos(q2) #De24
        Me12[1,1] =   0.5 * leg_mass * leg_length * np.sin(q2)#De25
        Me12[2,0] =   torso_mass * torso_com * np.cos(q3)#De34
        Me12[2,1] = - torso_mass * torso_com * np.sin(q3)#De35

        Me22 = np.zeros((2,2))

        Me22[0,0] = \
        Me22[1,1] = 2 * leg_mass + hip_mass + torso_mass #De44, De55

        Me = np.block([[Me11, Me12],[Me12.T, Me22]])

        E = np.zeros((2,5))
        E[0,0] =   leg_length * np.cos(q1)
        E[0,1] = - leg_length * np.cos(q2)
        E[1,0] = - leg_length * np.sin(q1)
        E[1,1] =   leg_length * np.sin(q2)
        E[0,3] = \
        E[1,4] = 1

        return Me, E
    
    def __eval_C(self, state):
        q1, q2, q3, dq1, dq2, dq3 = state

        leg_mass = self.params.leg_mass
        leg_length = self.params.leg_length
        torso_mass = self.params.torso_mass
        torso_com = self.params.torso_com

        C = np.zeros((3,3))
        C[0,1] = -0.5 * leg_mass * leg_length**2 * np.sin(q1 - q2) * dq2
        C[0,2] = torso_mass * leg_length * torso_com * np.sin(q1 - q3) * dq3
        C[1,0] = 0.5 * leg_mass * leg_length**2 * np.sin(q1 - q2) * dq1
        C[2,0] = -torso_mass * leg_length * torso_com * np.sin(q1 - q3) * dq1

        return C
    
    def __eval_G(self, state):
        q1, q2, q3, dq1, dq2, dq3 = state

        K = self.params.K
        hip_mass = self.params.hip_mass
        leg_mass = self.params.leg_mass
        torso_mass = self.params.torso_mass
        torso_com = self.params.torso_com
        leg_length = self.params.leg_length
        gravity_acceleration =  self.params.gravity_acceleration

        G1 = (q1 - q3) * K - (hip_mass + 1.5 * leg_mass + torso_mass) * leg_length * gravity_acceleration * np.sin(q1) #Какая-то фигня с углом под синусом
        G2 = (q2 - q3) * K + 0.5 * leg_mass * gravity_acceleration * leg_length * np.sin(q2)
        G3 = (2*q3 - q2 - q1) * K - torso_mass * gravity_acceleration * torso_com * np.sin(q3)
        G = np.array([[G1], [G2], [G3]])

        return G
    
    def __eval_B(self, state):
        return np.array([[1],[-1],[0]])
    
    def impact(self, state_m, pivot_m):

        q1_m, q2_m, q3_m, dq1_m, dq2_m, dq3_m = state_m

        leg_length = self.params.leg_length

        x_m = pivot_m
        y_m = 0
        x_p = x_m + leg_length * np.sin(q1_m) - leg_length * np.sin(q2_m)
        y_p = y_m + leg_length * np.cos(q1_m) - leg_length * np.cos(q2_m)
        pivot_p = x_p

        Me, E = self.__eval_Impact(state_m)
        
        A = np.block([[Me, -E.T],[E, np.zeros((2,2))]])
        dqe_m = np.array([state_m[3:6]])
        dqe_m = np.hstack((dqe_m, np.zeros((1,2)))).T
        B = np.vstack((Me @ dqe_m, np.zeros((2,1))))
        
        x = np.linalg.inv(A) @ B

        dq2_p, dq1_p, dq3_p = tuple(map(float, x.flatten()[0:3]))
        q2_p, q1_p, q3_p = state_m[0:3]

        state_p = np.array([q1_p, q2_p, q3_p, dq1_p, dq2_p, dq3_p])

        return state_p, pivot_p
    
    def dynamics(self, state):
        M = self.__eval_M(state)
        C = self.__eval_C(state)
        G = self.__eval_G(state)
        B = self.__eval_B(state)
        return M, C, G, B
    
    def rhs_fun(self, state, u):

        M, C, G, B = self.dynamics(state)
        Mi = np.linalg.inv(M)

        ddq = Mi @ ( - G - C @ np.array([state[3:6]]).T + B * u)

        rhs = np.concatenate((np.array([state[3:6]]), ddq.T), axis=1)[0]
        
        return(rhs)
    
    
class Constraints:
    def __init__(self, dynamics: BipedDynamics, opt=True, **kwargs):

        self.dynamics = dynamics

        if 'first_initial_state' in kwargs:
            first_initial_state = np.copy(kwargs['first_initial_state'])

            if 'method' in kwargs:
                self.method = kwargs['method']
            else:
                self.method = 'TNC'
            


            initial_state = first_initial_state

            if (opt == True):
                solved_state = self._optimize_trajectory(first_initial_state, method = self.method)

                print(solved_state.success)
                print(solved_state.x)
                initial_state = solved_state.x

            self.initial_state = initial_state
            
            t, y = self._get_trajectory(initial_state)

        elif 'phase_trajectory' in kwargs:

            phase_trajectory = kwargs['phase_trajectory']

            t = phase_trajectory.theta
            u = phase_trajectory.feed_forward
            q = phase_trajectory.q
            dq = phase_trajectory.dq
            ddq = phase_trajectory.ddq

            qp = dq / dq[:, [0]]
            qpp = (ddq - qp * ddq[:, [0]]) / (dq[:, [0]] ** 2)
    
            y = np.concatenate((q[:, 1:], qp[:,1:], qpp[:,1:], np.reshape(u, (-1,1))), 1).T

            self.initial_state = [self.dynamics.params.K, q[0,0], q[0,2], dq[0,0], qp[0,1], qp[0,2], u[0]]

        fs =18

        i = 0 

        while os.path.exists('./figures/constraints' + str(i) + '0.png'):
            i += 1

        fig, ax1 = plt.subplots(1,1, figsize=(6,4))
        ax1.plot(t, y[6])
        ax1.grid(True)
        ax1.set_ylabel(r'u [Nm]', fontsize=fs)
        ax1.set_xlabel(r'$q_1 = \theta$ [rad]', fontsize=fs)
        ax1.set_title(r'Номинальное управление', fontsize=fs)
        fig.tight_layout(pad = 1.0)
        fig.savefig('./figures/constraints' + str(i) + '1.png')

        fig, ax2 = plt.subplots(1,1, figsize=(6,4))
        ax2.plot(t, y[0])
        ax2.grid(True)
        ax2.set_ylabel(r'$q_2$ [rad]', fontsize=fs)
        ax2.set_xlabel(r'$q_1 = \theta$ [rad]', fontsize=fs)
        ax2.set_title(r'Переносная нога', fontsize=fs)
        fig.tight_layout(pad = 1.0)
        fig.savefig('./figures/constraints' + str(i) + '2.png')

        fig, ax3 = plt.subplots(1,1, figsize=(6,4))
        ax3.plot(t, y[1])
        ax3.grid(True)
        ax3.set_ylabel(r'$q_3$ [rad]', fontsize=fs)
        ax3.set_xlabel(r'$q_1 = \theta$ [rad]', fontsize=fs)
        ax3.set_title(r'Торс', fontsize=fs)
        fig.tight_layout(pad = 1.0)
        fig.savefig('./figures/constraints' + str(i) + '3.png')

        # ax2.plot(y[0], y[2])
        # ax2.grid(True)
        # ax2.set_ylabel(r"$ q_2'$ ")
        # ax2.set_xlabel(r"$q_2$ [rad]")
        # ax2.set_title(r'Swing leg')

        # ax3.plot(y[1], y[3])
        # ax3.grid(True)
        # ax3.set_ylabel(r"$q_3'$")
        # ax3.set_xlabel(r'$q_3$ [rad]')
        # ax3.set_title(r'Torso')

        # ax1.text(-0.1,-0.2, "a)", size=fs, ha="center", 
        #  transform=ax1.transAxes)
        # ax2.text(-0.1,-0.2, "b)", size=fs, ha="center", 
        #  transform=ax2.transAxes)
        # ax3.text(-0.1,-0.2, "c)", size=fs, ha="center", 
        #  transform=ax3.transAxes)

        # fig.tight_layout(pad = 1.0)
        # fig.savefig('./figures/constraints' + str(i) + '.png')
        
        plt.show()
        
        self.spline = sp.interpolate.make_interp_spline(t, y.T, k=5)
        self.theta = t   
        
    def _optimize_trajectory(self, chi_approximation, method = 'TNC'):

        def functional(init_state):

            K, theta, phi3, theta_dot, phi2_prime, phi3_prime, u = init_state

            state_plus = np.array([theta, -theta, phi3, theta_dot, phi2_prime * theta_dot, phi3_prime * theta_dot])

            th, y = self._get_trajectory(init_state)

            theta_dot_minus = np.sqrt(max(np.mean(self._getD2array(th[-1], y[:,-1])),0.1))

            state_minus = np.array([th[-1], y[0,-1], y[1,-1], theta_dot_minus, theta_dot_minus * y[2,-1],  theta_dot_minus * y[3,-1]])

            new_state_plus, _pivot = self.dynamics.impact(state_minus, (0,0))

            print("Initial state:", init_state)
            print(state_plus)
            print(new_state_plus)
            print(np.linalg.norm(state_plus - new_state_plus))
            print()

            return np.linalg.norm(state_plus - new_state_plus)
        
        bnds = ((10, 50), (-0.4, -0.2), (0, None), (0.1, None), (None, None), (0, None), (None, None))
        
        res = sp.optimize.minimize(functional, chi_approximation, bounds=bnds, tol=1e-5, method =method)
        
        print("Results:")
        test = functional(res.x)

        return res

        

    def _get_trajectory(self, initial_state, delta = 0.0000005):
        
        constraints_array = self._get_full_zero_constraints(initial_state)

        def rhs(theta, const_array, epsilon=1e-6):

            q2, q3, q2p, q3p, q2pp, q3pp, u = const_array

            D1 = self._getD1array(theta, const_array)

            chi = np.zeros((3))
            chi += (self._getD2array(theta + delta, const_array) - self._getD2array(theta - delta, const_array)) / (2 * delta)

            for i in range(4):

                tmp_const_array1 = list(const_array.copy())
                tmp_const_array2 = list(const_array.copy())

                tmp_const_array1[i] += delta
                tmp_const_array2[i] -= delta

                chi += (self._getD2array(theta, tmp_const_array1) - self._getD2array(theta, tmp_const_array2)) * const_array[i+2] / (2 * delta)

            tmp_mat = np.zeros((3,3))
            
            for i in range(3):

                tmp_const_array1 = list(const_array.copy())
                tmp_const_array2 = list(const_array.copy())

                tmp_const_array1[i + 4] += delta
                tmp_const_array2[i + 4] -= delta

                tmp_mat[:, i] =  (self._getD2array(theta, tmp_const_array1) - self._getD2array(theta, tmp_const_array2)) / (2 * delta)

            if np.abs(np.linalg.det(tmp_mat)) < epsilon:
                status = 0
                # print("Determinant is too small",  np.linalg.det(tmp_mat))
                return status, np.array([q2p, q3p, q2pp, q3pp, 0.1, 0, 0])

            status = 1
            q2ppp, q3ppp, up = np.linalg.solve(tmp_mat, 2 * D1 - chi)

            return status, np.array([q2p, q3p, q2pp, q3pp, q2ppp, q3ppp, up])

        def wrapped_rhs(t, y):
            return rhs(t, y)[1]

        #TODO Заменить решатель на Эйлера
        # result = sp.integrate.solve_ivp(wrapped_rhs, [initial_state[1], -initial_state[1]], constraints_array, t_eval=np.linspace(initial_state[1], -initial_state[1], 100), method='RK45')
        # return result.t, result.y

        t, y = self._solve_ivp(rhs, [initial_state[1], -initial_state[1]], np.array(constraints_array), step = 1e-4, epsilon=1e-6)
        return t, y
        
    
    def _solve_ivp(self, rhs, t_span, y0, step = 1e-4, epsilon=1e-6):

        y_arr = [y0.copy()]
        t_arr = [t_span[0]]

        t = t_span[0]
        y = y0

        while t < t_span[1]:

            status_tmp, derivs = rhs(t, y)

            while True:
                y += derivs * step
                t += step
                status, _ = rhs(t, y, epsilon=epsilon)
                if status ==  1:
                    break
            
            # print(y)
            # input()

            y_arr.append(y.copy())
            t_arr.append(t)

        return np.array(t_arr), np.array(y_arr).T

    def _getD1array(self, theta, constraints_array):
        alpha, beta, gamma, u = self._abgu_coeffs(theta, constraints_array)
        D1 = np.zeros((3))

        for i in range(3):
            tmp_mat = np.zeros((2,2))
            tmp_mat[:,0] = np.delete(alpha, (i), axis=0)
            tmp_mat[:,1] = np.delete(beta, (i), axis=0)
            
            D1[i] = (np.linalg.inv(tmp_mat) @ (np.delete(u, (i), axis=0) - np.delete(gamma, (i), axis=0)))[0]
           
        return D1
    
    def _getD2array(self, theta, constraints_array):
        alpha, beta, gamma, u = self._abgu_coeffs(theta, constraints_array)
        D2 = np.zeros((3))

        for i in range(3):
            tmp_mat = np.zeros((2,2))
            tmp_mat[:,0] = np.delete(alpha, (i), axis=0)
            tmp_mat[:,1] = np.delete(beta, (i), axis=0)
            
            D2[i] = (np.linalg.inv(tmp_mat) @ (np.delete(u, (i), axis=0) - np.delete(gamma, (i), axis=0)))[1] 

        return D2

    def _get_full_zero_constraints(self, initial_state):

        K, theta, phi3, theta_dot, phi2_prime, phi3_prime, u = initial_state
        self.dynamics.params.K = K

        Q = [theta, -theta, phi3]
        Q1 = [1, phi2_prime, phi3_prime]

        M,C,G,B = self.dynamics.dynamics(Q + Q1)

        tmp_vec = np.linalg.inv(M) @ (B * u - C @ np.array([Q1]).T * (theta_dot**2) - G ) / (theta_dot**2) 

        tmp_mat = np.zeros((3,3))
        tmp_mat[:,0] = Q1
        Q2 = (np.eye(3) - tmp_mat) @ tmp_vec
        
        return -theta, phi3, phi2_prime, phi3_prime, float(Q2[1,0]), float(Q2[2,0]), u 

    def _abgu_coeffs(self, theta, constraints_array):

        q2, q3, q2p, q3p, q2pp, q3pp, u = constraints_array

        Q = [theta, q2, q3]
        Q1 = [1, q2p, q3p]
        Q2 = [0, q2pp, q3pp]

        M,C,G,B = self.dynamics.dynamics(Q + Q1)

        A = np.zeros((3,3))
        A[0,2] = A[1,0] = A[1,1] = A[2,1] = 1
    
        alpha = A @ M @ Q1
        beta = A @ (M @ Q2 + C @ Q1)
        gamma = A @ G[:,0]

        return alpha, beta, gamma, (A @ B[:,0]) * u

    def __call__(self, phi, der=0):
        return self.spline(phi, der)

def reduced_dynamics(self, constr: Constraints):
        def rhs(t, st):
            th,dth = st
            B_perp = [1, 0, 0]
            alpha, beta, gamma, u = constr._abgu_coeffs(th, constr(th))
            alpha = B_perp @ alpha
            beta = B_perp @ beta
            gamma = B_perp @ gamma
            ddth =  (-gamma - beta * dth**2) / alpha
            return [dth, ddth]
        

        initial_state = constr.initial_state

        def hit_ground(t, y):
            return np.abs(y[0]) + constr(y[0])[0]
            # return y[0] + initial_state[1]
        hit_ground.terminal = True
        hit_ground.direction = 1
        
        t = np.arange(0, 1.00, 0.003)
        sol = sp.integrate.solve_ivp(rhs, [t[0], t[-1]], [initial_state[1], initial_state[3]], t_eval=t, max_step=1e-3, events=hit_ground)

        ddth = np.zeros((1, len(sol.t)))

        for i in range(len(sol.t)):
            ddth[0,i] = rhs(0, sol.y[:,i])[1]
    
        sol.y = np.concatenate((sol.y, ddth), axis=0)

        return sol.t, sol.y 

BipedDynamics.reduced_dynamics = reduced_dynamics

    

