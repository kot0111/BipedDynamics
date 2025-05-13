from transverse_linearization.linearization import TransverseLinearization
import scipy as sp
import numpy as np

class Feedback:
    def __init__(self, tl: TransverseLinearization):
        self.tl = tl
        self.constraints = tl.constraints

        self.trajectory = tl.trajectory
        theta = self.trajectory.theta
        t = self.trajectory.t
        dtheta = self.trajectory.dtheta

        self.t_star = sp.interpolate.make_interp_spline(theta, t, 5)
        self.t_star.extrapolate = 'extrapolate'

        self.K = sp.interpolate.make_interp_spline(t, np.reshape(tl.Ku, (-1, 5)), 5)
        self.K.extrapolate = 'periodic'

        self.th_sp = self.trajectory.theta_sp
        self.th_sp.extrapolate = 'extrapolate'

        self.u_nominal = []
        self.Ku = []

    def get_transverse(self, state):
        theta, q2, q3, dtheta, dq2, dq3 = state

        clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
        theta = clamp(theta, self.trajectory.theta[0], self.trajectory.theta[-1])

        I = self.tl.get_integral(theta, dtheta)
        # I = dtheta**2 - self.Psi(theta) * (self.trajectory.dtheta[0]**2  - self.Part(theta))
        # print(I)
        # I = 0
        # print(I)

        y1 = q2 - self.constraints(theta)[0]
        y2 = q3 - self.constraints(theta)[1]
        dy1 = dq2 - self.constraints(theta)[2] * dtheta
        dy2 = dq3 - self.constraints(theta)[3] * dtheta
        
        return [I, y1, y2, dy1, dy2]
    
    def __call__(self, t, y, state):
        theta, q2, q3, dtheta, dq2, dq3 = state

        clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
        theta = clamp(theta, self.trajectory.theta[0], self.trajectory.theta[-1])

        t_star = self.t_star(theta)

        xp = np.array([self.get_transverse(state)]).T
        v = - self.K(t_star) @ xp / self.tl.N1(theta, 0, 0)
        
        u = v + self.constraints(theta)[6]

        self.u_nominal.append(self.constraints(theta)[6])
        self.Ku.append(self.K(t_star))

        return u
    
class ISMFeedback:
    def __init__(self, tl: TransverseLinearization):
        self.tl = tl
        self.constraints = tl.constraints
        self.dynamics = tl.dynamics

        self.trajectory = tl.trajectory
        theta = self.trajectory.theta
        dtheta = self.trajectory.dtheta

        self.K = sp.interpolate.make_interp_spline(theta, np.reshape(tl.Ku, (-1, 5)), 5)
        self.K.extrapolate = 'periodic'

        self.th_sp = self.trajectory.theta_sp
        self.th_sp.extrapolate = 'extrapolate'

        self.state = None
        self.zero_state = None

    def get_transverse(self, state):
        theta, q2, q3, dtheta, dq2, dq3 = state

        clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
        theta = clamp(theta, self.trajectory.theta[0], self.trajectory.theta[-1])

        I = self.tl.get_integral(theta, dtheta)
        # I = dtheta**2 - self.Psi(theta) * (self.trajectory.dtheta[0]**2  - self.Part(theta))
        # print(I)
        # I = 0
        # print(I)

        y1 = q2 - self.constraints(theta)[0]
        y2 = q3 - self.constraints(theta)[1]
        dy1 = dq2 - self.constraints(theta)[2] * dtheta
        dy2 = dq3 - self.constraints(theta)[3] * dtheta
        
        return [I, y1, y2, dy1, dy2]
    
    def __call__(self, t, y, state, ism = True):

        theta, q2, q3, dtheta, dq2, dq3 = state

        clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
        theta = clamp(theta, self.trajectory.theta[0], self.trajectory.theta[-1])

        xp = np.array([self.get_transverse(state)]).T
        v = - self.K(theta) @ xp / self.tl.N1(theta, 0, 0)

        # G = np.array([[10,0,0,1,-3]])
        G = np.array([[0,0,0,0,-1,0]])

        if ism:
            if self.state is None:
                self.state = np.zeros((7,))
                self.state[0] = t
                self.state[1:7] = state
                # self.zero_state = self.state

            else:

                non_ism_u = self.__call__(t, y, self.state[1:7], ism=False)
                rhs = self.dynamics.rhs_fun(self.state[1:7], non_ism_u)
                self.state[1:7] += rhs * (t - self.state[0])
                self.state[0] = t

                expected_trans = np.array([self.get_transverse(self.state[1:7])]).T
                #xp = real_trans = self.get_transverse(state)
                
                # sigma = G @ (xp - expected_trans)
                sigma = G @ (state - self.state[1:7])

                # print(self.tl.N1(theta, 0, 0))
                # print(v)
                # input()
                # v += - 20 * np.sign(sigma[0]) / self.tl.N1(theta, 0, 0)
                v += - 5 * np.sign(sigma[0]) 
                # u += - 1 * np.sign(sigma[0]) / self.tl.N1(theta, 0, 0)

        u = v + self.constraints(theta)[6]

        # print(u)

        return u