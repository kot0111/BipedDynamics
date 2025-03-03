from transverse_linearization.linearization import TransverseLinearization
import scipy as sp
import numpy as np

class Feedback:
    def __init__(self, tl: TransverseLinearization):
        self.tl = tl
        self.constraints = tl.constraints

        self.trajectory = tl.trajectory
        theta = self.trajectory.theta
        dtheta = self.trajectory.dtheta

        self.K = sp.interpolate.make_interp_spline(theta, np.reshape(tl.Ku, (-1, 5)), 5)
        self.K.extrapolate = 'periodic'

        self.th_sp = self.trajectory.theta_sp
        self.th_sp.extrapolate = 'extrapolate'
        

    def get_transverse(self, state):
        theta, q2, q3, dtheta, dq2, dq3 = state

        #TODO Ну и так понятно 
        I = 0

        y1 = q2 - self.constraints(theta)[0]
        y2 = q3 - self.constraints(theta)[1]
        dy1 = dq2 - self.constraints(theta)[2] * dtheta
        dy2 = dq3 - self.constraints(theta)[3] * dtheta
        
        return [I, y1, y2, dy1, dy2]
    
    def __call__(self, t, y, state):
        theta, q2, q3, dtheta, dq2, dq3 = state

        xp = np.array([self.get_transverse(state)]).T
        v = - self.K(theta) @ xp / self.tl.N1(theta, xp[1], xp[2])

        #TODO Может стоит пересчитывать номинальное управление по нормальному, а не на идеальной траектории
        u = v + self.constraints(theta)[6]

        return u