import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from dynamics.biped_dynamics import  BipedDynamics
from dynamics.parameters import BipedParameters
from trajectory.trajectory import PhaseTrajectory
from scipy.integrate import ode
from dataclasses import dataclass

def animate(trj : PhaseTrajectory, params : BipedParameters, redraw_flag = 0):
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-2.,2.), ylim=(-0.1, 1.6))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([],[],'o-', lw=2, c='b')
    line2, = ax.plot([],[],'o-', lw=2, c='k')
    baseLine, = ax.plot([],[],'-',lw=2, c='b')
    time_template = 'time = %.2fs'
    time_text = ax.text(0.045, 0.9, '', transform=ax.transAxes)

    leg_lenght = params.leg_length
    torso_lenght = params.torso_com

    q1, q2, q3 = trj.q.T

    x1 = trj.pivot * redraw_flag + leg_lenght * np.sin(q1)
    y1 = leg_lenght * np.cos(q1)
    x2 = x1 - leg_lenght * np.sin(q2)
    y2 = y1 - leg_lenght * np.cos(q2)
    x3 = x1 + torso_lenght * np.sin(q3)
    y3 = y1 + torso_lenght * np.cos(q3)

    interv = int((trj.t[10] - trj.t[9]) * 1000)
    if interv < 5:
        k = 10
    else:
        k = 1

    def animate(j):
        i = j * k
        thisx = [trj.pivot[i] * redraw_flag, x1[i], x2[i], x1[i], x3[i]]

        thisy = [0, y1[i], y2[i], y1[i], y3[i]]

        line.set_data(thisx, thisy)
        
        if (redraw_flag):
            ax.set_xlim(x1[i] - 1, x1[i] + 1)
            ax.figure.canvas.draw() 

        #line2.set_data([0, x1f3, x2f3, x3f3], [0, y1f3, y2f3, y3f3])
        # baseLine.set_data([Lfoot, 0], [0, 0])
        time_text.set_text(time_template % (trj.t[i]))

        return line, time_text
    
    

    anim = animation.FuncAnimation(
        fig, func=animate, frames=q1.shape[0] // k, interval=interv * k, blit=True, repeat=True)
    

    plt.show()

@dataclass
class SimulationResult:
    trajectory : PhaseTrajectory
    controller_internal_state : list = None

class BipedSimulator:
    def __init__(self, bippr : BipedParameters, fb : callable, matched_dist : callable = None):
        self.dynamics = BipedDynamics(bippr)
        self.fb = fb
        self.step = 1e-3
        self.t = None
        if matched_dist is not None:
            self.matched_dist = matched_dist

    def __update_disturbed_output(self):
        q1,q2,q3,_,_,_ = self.state
        # theta = discretize(theta, self.theta_step)
        # theta = float(self.encoder_delay(self.t, theta))
        # alpha = theta + phi
        # alpha += np.random.normal(scale=self.camera_noise)
        # alpha = float(self.camera_delay(self.t, alpha))
        # phi = alpha - theta
        self.disturbed_output = np.array([q1, q2, q3])

    def __init_disturbed_output(self, initial_time : float, initial_state : np.ndarray):
        self.t = initial_time
        self.state = np.reshape(initial_state, (-1,))
        q10,q20,q30,_,_,_ = self.state
        # alpha0 = theta0 + phi0
        # self.encoder_delay.set_initial_value(self.t, theta0)
        # self.camera_delay.set_initial_value(self.t, alpha0)
        # self.motor_delay.set_initial_value(self.t, 0.)
        self.disturbed_output = np.array([q10, q20, q30])

    def run(self, initial_state : np.ndarray, tstart : float, tend : float, get_last_step : bool  = False) -> SimulationResult:
       
        self.__init_disturbed_output(tstart, initial_state)
        self.u = float(self.fb(self.t, self.disturbed_output, self.state))
        self.pivot = 0

        if hasattr(self, 'matched_dist'):
            self.noise = float(self.matched_dist(self.t))
        else:
            self.noise = 0.0
        
        rhs = lambda _, x: self.dynamics.rhs_fun(x, self.u + self.noise)

        solt = [self.t]
        solx = [self.state]
        solu = [self.u]
        pivot = [self.pivot]
        if hasattr(self.fb, 'state'):
            solfb = [np.copy(self.fb.state)]
        else:
            solfb = None

        n_last_step_start = 0

        while self.t < tend - self.step:

            if get_last_step:
                n_last_step_start = len(solt) - 1

            integrator = ode(rhs)
            integrator.set_initial_value(self.state, self.t)
            integrator.set_integrator('dopri5', max_step=self.step)

            while True:
                
                ddq = rhs(0, self.state)

                if not integrator.successful():
                    print('[warn] integrator doesn\'t feel good at t = ', self.t)

                # step of integration
                integrator.integrate(self.t + self.step)
                self.t = integrator.t
                self.state = integrator.y

                # sensors noise
                self.__update_disturbed_output()

                # call controller
                u = self.fb(self.t, self.disturbed_output, self.state)
                if u is None:
                    break

                # u = np.clip(u, -self.torque_max, self.torque_max)
                # u_delayed = self.motor_delay(self.t, u)
                u_delayed = u
                
                # friction

                # dtheta = self.state[2]
                # if abs(dtheta) < 1e-3:
                #     if abs(u_delayed) < self.motor_dry_friction:
                #         u_delayed = 0
                #     else:
                #         u_delayed -= self.motor_dry_friction * np.sign(u_delayed)
                # else:
                #     u_delayed -= self.motor_dry_friction * np.sign(dtheta)

                self.u = float(u_delayed)
                self.noise = float(self.matched_dist(self.t)) if hasattr(self, 'matched_dist') else 0.0

                solx[-1] = np.concatenate((solx[-1], ddq[3:6]))

                # print(solx[-1])
                # print(ddq)
                # input()

                solt.append(self.t)
                solx.append(self.state.copy())
                solu.append(self.u)
                pivot.append(self.pivot)

                if hasattr(self.fb, 'state'):
                    solfb.append(np.copy(self.fb.state))

                if((self.state[0] > 0) and (self.state[0] + self.state[1] > 0)):
                    ddq = rhs(0, self.state)
                    solx[-1] = np.concatenate((solx[-1], ddq[3:6]))
                    break

            self.state, self.pivot = self.dynamics.impact(self.state, self.pivot)

            if self.t < tend - self.step:

                solt.append(self.t)
                solx.append(self.state.copy())
                solu.append(self.u)
                pivot.append(self.pivot)

                if hasattr(self.fb, 'state'):
                    solfb.append(np.copy(self.fb.state))
                    self.fb.state = None

        # ddq = rhs(0, self.state)
        # solx[-1] = np.concatenate((solx[-1], ddq[3:6]))

        # print(solx[-10:])

        gap = 1

        if get_last_step:
            gap = len(solt[n_last_step_start:]) // 150
            
        # print(gap)
        last_not_in_array =  bool(len(solt[n_last_step_start:]) % gap)

        t = solt[-1]
        x = solx[-1].copy()
        u = solu[-1]
        piv = pivot[-1]

        solt = solt[n_last_step_start::gap]
        solx = solx[n_last_step_start::gap]
        solu = solu[n_last_step_start::gap]
        pivot = pivot[n_last_step_start::gap]

        if last_not_in_array:
            solt.append(t)
            solx.append(x.copy())
            solu.append(u)
            pivot.append(piv)

        if hasattr(self.fb, 'state'):
            fb = solfb[-1].copy()
            solfb = solfb[n_last_step_start::gap]
            if last_not_in_array:
                solfb.append(fb.copy())
        
        result = SimulationResult(
            trajectory = PhaseTrajectory(
                t = np.asanyarray(solt),
                phase = np.asanyarray(solx),
                u = np.asanyarray(solu),
                pivot = np.asanyarray(pivot)
            ),
            controller_internal_state = solfb
        )
        return result