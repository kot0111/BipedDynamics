from dynamics.biped_dynamics import BipedDynamics, Constraints

import numpy as np
import scipy as sp


def get_trajectory(d : BipedDynamics, constraints : Constraints):
        t, path = d.reduced_dynamics(constraints)
        th = path[0]
        dth = path[1]
        ddth = path[2]
        q2 = np.array([constraints(ph)[0] for ph in th])
        q3 = np.array([constraints(ph)[1] for ph in th])
        dq2 = np.array([constraints(ph)[2] * dph for ph, dph, _ in zip(*path)])
        dq3 = np.array([constraints(ph)[3] * dph for ph, dph, _ in zip(*path)])
        ddq2 = np.array([constraints(ph)[2] * ddph + constraints(ph)[4] * dph**2 for ph, dph, ddph in zip(*path)])
        ddq3 = np.array([constraints(ph)[3] * ddph + constraints(ph)[5] * dph**2 for ph, dph, ddph in zip(*path)])
        
        return {
        't':t,
        'q1': th,
        'q2': q2,
        'q3': q3,
        'dq1': dth,
        'dq2': dq2,
        'dq3': dq3,
        'ddq1': ddth,
        'ddq2': ddq2,
        'ddq3': ddq3
        }

class PhaseTrajectory:
    def __init__(self, **kwargs):
        if 'phase' in kwargs:
            self.phase = np.copy(kwargs['phase'])
        elif 'q1' in kwargs:
            assert 'q2' in kwargs
            assert 'q3' in kwargs
            assert 'dq1' in kwargs
            assert 'dq2' in kwargs
            assert 'dq3' in kwargs
            assert 'ddq1' in kwargs
            assert 'ddq2' in kwargs
            assert 'ddq3' in kwargs
            self.phase = np.array([
                kwargs['q1'],
                kwargs['q2'],
                kwargs['q3'],
                kwargs['dq1'],
                kwargs['dq2'],
                kwargs['dq3'],
                kwargs['ddq1'],
                kwargs['ddq2'],
                kwargs['ddq3']
            ]).T
        else:
            assert False, 'Expect phase coordinates and time'
        if 'u' in kwargs:
            self.u = np.copy(kwargs['u'])
        else:
            self.u = None
        assert 't' in kwargs
        self.t = np.copy(kwargs['t'])

        self.theta_sp = sp.interpolate.make_interp_spline(self.phase[:,0], self.phase[:, [3,6]], k=5)

    @property
    def theta(self):
        return self.phase[:,0]

    @property
    def dtheta(self):
        return self.phase[:,3]

    @property
    def q(self):
        return self.phase[:,0:3]

    @property
    def dq(self):
        return self.phase[:,3:6]
    
    @property
    def ddq(self):
        return self.phase[:,6:9]