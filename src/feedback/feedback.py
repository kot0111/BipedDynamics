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

        self._init_psi()
        self._init_part()

        
    def _init_psi(self):

        Psi_list = []

        def rhs(th):
            return 2 * self.tl.beta(th) / self.tl.alpha(th)
        
        for theta in self.trajectory.theta:
            Psi_list.append(np.exp( - sp.integrate.quad(rhs, self.trajectory.theta[0], theta)[0]))
    

        Psi = sp.interpolate.make_interp_spline(self.trajectory.theta, Psi_list, 5)
        Psi.extrapolate = 'extrapolate'
        self.Psi = Psi

    def _init_part(self):

        Part_list = []

        def rhs(th):
            return 2 * self.tl.gamma(th) / (self.tl.alpha(th) * self.Psi(th))
        
        for theta in self.trajectory.theta:
            Part_list.append(sp.integrate.quad(rhs, self.trajectory.theta[0], theta)[0])
        
        Part = sp.interpolate.make_interp_spline(self.trajectory.theta, Part_list, 5)
        # print(result.y[0])
        Part.extrapolate = 'extrapolate'
        self.Part = Part


    def get_transverse(self, state):
        theta, q2, q3, dtheta, dq2, dq3 = state

        I = dtheta**2 - self.Psi(theta) * (self.trajectory.dtheta[0]**2  - self.Part(theta))
        # print(I)
        # print(t)
        # # I2 = self.trajectory.dtheta[10]**2 - self.Psi(self.trajectory.theta[10]) * (self.trajectory.dtheta[0]**2  - self.Part(self.trajectory.theta[10]))
        # # print(I2)
        # input()
        # I = 0
        print(I)

        y1 = q2 - self.constraints(theta)[0]
        y2 = q3 - self.constraints(theta)[1]
        dy1 = dq2 - self.constraints(theta)[2] * dtheta
        dy2 = dq3 - self.constraints(theta)[3] * dtheta
        
        return [I, y1, y2, dy1, dy2]
    
    def __call__(self, t, y, state):
        theta, q2, q3, dtheta, dq2, dq3 = state

        xp = np.array([self.get_transverse(state)]).T
        v = - self.K(theta) @ xp / self.tl.N1(theta, 0, 0)
        
        u = v + self.constraints(theta)[6]

        return u