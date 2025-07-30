from dynamics.parameters import load_biped_parameters
from dynamics.biped_dynamics import BipedDynamics, Constraints
from trajectory.trajectory import PhaseTrajectory, get_trajectory
from sim.biped_simulation import animate, BipedSimulator
from transverse_linearization.linearization import TransverseLinearization
from feedback.feedback import Feedback, ISMFeedback
import matplotlib.pyplot as plt

import numpy as np


if __name__ == "__main__":

    System = {}
    
    System['parameters'] = load_biped_parameters("biped.json")
    System['dynamics'] = BipedDynamics(System['parameters'])

    # System['constraints'] = Constraints(System['dynamics'], opt=True, first_initial_state = [20.3429, -0.3375, 0.5816, 1.3470, -0.0883, 0.1620, 2.7132])
    # System['constraints'] = Constraints(System['dynamics'], opt=True, first_initial_state = [20.34290121, -0.33750024,  0.58160114,  1.3469997,  -0.08829946,  0.16200075, 2.71320124])
    # System['constraints'] = Constraints(System['dynamics'], opt=True, first_initial_state = [19.85407185, -0.3386739, 0.57926869, 1.34735879, -0.08935826, 0.16256198, 2.71171412], method='L-BFGS-B')
    # System['constraints'] = Constraints(System['dynamics'], opt=True, first_initial_state = [20.08404631, -0.33812162, 0.58036547, 1.34719, -0.08886039,  0.16229759, 2.71241317], method='L-BFGS-B')
   
    System['constraints'] = Constraints(System['dynamics'], opt=False, first_initial_state = [19.85407185, -0.3386739, 0.57926869, 1.34735879, -0.08935826, 0.16256198, 2.71171412])

    print(System['constraints'](-0.1))

    trajectory = get_trajectory(System['dynamics'], System['constraints'])

    System['trajectory'] = PhaseTrajectory(**trajectory)
    System['transverse_linearization'] = TransverseLinearization(System['trajectory'], System['constraints'], System['dynamics'])

    K, theta, phi3, theta_dot, phi2_prime, phi3_prime, u = System['constraints'].initial_state
    state_plus = np.array([theta, -theta, phi3, theta_dot, phi2_prime * theta_dot, phi3_prime * theta_dot])

    feedback = Feedback(System['transverse_linearization'])
    sim = BipedSimulator(System['parameters'], feedback)
    result = sim.run(state_plus, 0, 30.0, get_last_step=True)
    animate(result.trajectory, System['parameters'],redraw_flag=1)

    t = feedback.trajectory.t
    K = feedback.K(t)
    fs = 18

    fig3, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1, figsize=(16,12))
    ax0.grid(True)
    ax0.set_ylabel(r'$K_I(t)$', fontsize=fs)
    ax0.axes.xaxis.set_ticklabels([])

    ax1.grid(True)
    ax1.set_ylabel(r'$K{y_1}(t)$', fontsize=fs)
    ax1.axes.xaxis.set_ticklabels([])

    ax2.grid(True)
    ax2.set_ylabel(r'$K_{y_2}(t)$', fontsize=fs)
    ax2.axes.xaxis.set_ticklabels([])

    ax3.grid(True)
    ax3.set_ylabel(r'$K_{\dot{y}_1}(t)$', fontsize=fs)
    ax3.axes.xaxis.set_ticklabels([])

    ax4.grid(True)
    ax4.set_ylabel(r'$K_{\dot{y}_2}(t)$', fontsize=fs)
    ax4.set_xlabel(r'time [sec]', fontsize=fs)

    ax0.plot(t, K[:, 0])
    ax1.plot(t, K[:, 1])
    ax2.plot(t, K[:, 2])
    ax3.plot(t, K[:, 3])
    ax4.plot(t, K[:, 4])

    plt.show()

    # fig, (ax) = plt.subplots()
    # ax.plot(result.trajectory.t, result.trajectory.feed_forward)
    # ax.set_ylabel(r'$\tau$ [Nm]')
    # ax.set_xlabel(r'$q_1 = \theta$ [rad]')
    # ax.set_title(r'Torque')
    # plt.show()

    # ONE MORE TIME

    System['trajectory'] = result.trajectory
    System['constraints'] = Constraints(System['dynamics'], opt=False, phase_trajectory = result.trajectory)
    System['transverse_linearization'] = TransverseLinearization(System['trajectory'], System['constraints'], System['dynamics'])

    K, theta, phi3, theta_dot, phi2_prime, phi3_prime, u = System['constraints'].initial_state
    state_plus = np.array([0.5 * theta, - 0.5 * theta, 0, theta_dot, phi2_prime * 0.5 * theta_dot, 0])
    # state_plus = np.array([theta, - theta, phi3, theta_dot, phi2_prime * theta_dot, phi3_prime * theta_dot])
    # # System['parameters_mod'] = load_biped_parameters("biped_mod.json")

    noise = lambda t: (5 * np.sin(2 * np.pi * 4 * t) + np.sin(2 * np.pi * 40 *t) + 3) * 10
    
    # noise = None

    feedback = Feedback(System['transverse_linearization'])
    sim = BipedSimulator(System['parameters'], feedback, matched_dist=noise)
    result1 = sim.run(state_plus, 0, 10.0, get_last_step=False)
    # animate(result1.trajectory, System['parameters'],redraw_flag=1)

    feedback = ISMFeedback(System['transverse_linearization'])
    sim = BipedSimulator(System['parameters'], feedback, matched_dist=noise, step=1e-4)
    result2 = sim.run(state_plus, 0, 10.0, get_last_step=False)
    # animate(result2.trajectory, System['parameters'],redraw_flag=1)

    fs = 24
    fig1, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1, figsize=(16,10))
    fig1.suptitle(r'Трансверсальные координаты', fontsize=fs+1)

    
    t = result1.trajectory.t
    trj = result1.trajectory.trajectory

    trcor = np.zeros((len(t), 5))

    for i in range(len(t)):
        trcor[i, :] = feedback.get_transverse(trj[i, :])

    ax0.plot(t, trcor[:, 0])
    ax1.plot(t, trcor[:, 1])
    ax2.plot(t, trcor[:, 2])
    ax3.plot(t, trcor[:, 3])
    ax4.plot(t, trcor[:, 4])
    
    trcor = np.zeros((len(result2.trajectory.t), 5))

    trj = result2.trajectory.trajectory
    for i in range(len(result2.trajectory.t)):
        trcor[i, :] = feedback.get_transverse(trj[i, :])

    # fig2, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, 1)

    # ax0.plot(result2.trajectory.t, trcor[:, 0])
    # ax1.plot(result2.trajectory.t, trcor[:, 1])
    # ax2.plot(result2.trajectory.t, trcor[:, 2])
    # ax3.plot(result2.trajectory.t, trcor[:, 3])
    # ax4.plot(result2.trajectory.t, trcor[:, 4])

  
    ax0.grid(True)
    ax0.set_ylabel(r'$I(t)$', fontsize=fs)
    ax0.axes.xaxis.set_ticklabels([])
    

    ax1.grid(True)
    ax1.set_ylabel(r'$y_1(t)$', fontsize=fs)
    ax1.axes.xaxis.set_ticklabels([])

    ax2.grid(True)
    ax2.set_ylabel(r'$y_2(t)$', fontsize=fs)
    ax2.axes.xaxis.set_ticklabels([])

    ax3.grid(True)
    ax3.set_ylabel(r'$\dot{y}_1(t)$', fontsize=fs)
    ax3.axes.xaxis.set_ticklabels([])

    ax4.grid(True)
    ax4.set_ylabel(r'$\dot{y}_2(t)$', fontsize=fs)
    ax4.set_xlabel(r'time [sec]', fontsize=fs)

    # ax0.set_xlim(8,10)
    # ax1.set_xlim(8,10)
    # ax2.set_xlim(8,10)
    # ax3.set_xlim(8,10)
    # ax4.set_xlim(8,10)
    

    fig1.tight_layout(pad = 1.0)
    fig1.align_ylabels()
    
    # plt.figlegend(['LR', 'LR + ISM'], fontsize=fs, framealpha=0.0, ncols = 2)
    plt.savefig('./figures/transverse.png', dpi=300, bbox_inches='tight')
    plt.show()

    fig1, (ax1) = plt.subplots()
    ax1.plot(result1.trajectory.t, result1.trajectory.feed_forward)
    ax1.plot(result1.trajectory.t, noise(result1.trajectory.t))
    ax1.set_ylabel(r'$\tau$ [Nm]')
    ax1.set_xlabel(r'$q_1 = \theta$ [rad]')
    ax1.set_title(r'Torque1')

    # fig2, (ax2) = plt.subplots()
    # ax2.plot(result2.trajectory.t, result2.trajectory.feed_forward)
    # ax2.set_ylabel(r'$\tau$ [Nm]')
    # ax2.set_xlabel(r'$q_1 = \theta$ [rad]')
    # ax2.set_title(r'Torque2')
    plt.show()

    
