import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from dynamics.biped_dynamics import PhaseTrajectory, BipedDynamics
from dynamics.parameters import BipedParameters


def animate(trj : PhaseTrajectory):
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(autoscale_on=False, xlim=(-2.,2.), ylim=(-0.1, 2.0))
    ax.set_aspect('equal')
    ax.grid()

    line, = ax.plot([],[],'o-', lw=2, c='b')
    line2, = ax.plot([],[],'o-', lw=2, c='k')
    baseLine, = ax.plot([],[],'-',lw=2, c='b')
    time_template = 'time = %.1fs'
    time_text = ax.text(0.045, 0.9, '', transform=ax.transAxes)

    q1, q2, q3 = trj.q.T

    x1 = np.sin(q1)
    y1 = np.cos(q1)
    x2 = x1 - np.sin(q2)
    y2 = y1 - np.cos(q2)
    x3 = x1 + np.sin(q3)
    y3 = y1 + np.cos(q3)

    def animate(i):
        thisx = [0, x1[i], x2[i], x1[i], x3[i]]

        thisy = [0, y1[i], y2[i], y1[i], y3[i]]

        line.set_data(thisx, thisy)
        #line2.set_data([0, x1f3, x2f3, x3f3], [0, y1f3, y2f3, y3f3])
        # baseLine.set_data([Lfoot, 0], [0, 0])
        time_text.set_text(time_template % (trj.t[i]))

        return line, time_text
    
    anim = animation.FuncAnimation(
        fig, func=animate, frames=q1.shape[0], interval=5, blit=True, repeat=True)
    

    plt.show()

class BipedSimulator:
    def __init__(self, bippr : BipedDynamics, fb : callable):
        dyn = BipedDynamics(bippr)
        