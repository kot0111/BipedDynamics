from dynamics.parameters import load_biped_parameters
from dynamics.biped_dynamics import BipedDynamics, Constraints
from trajectory.trajectory import PhaseTrajectory, get_trajectory
from sim.biped_simulation import animate, BipedSimulator
from transverse_linearization.linearization import TransverseLinearization
from feedback.feedback import Feedback

import numpy as np


if __name__ == "__main__":

    System = {}
    
    System['parameters'] = load_biped_parameters("biped.json")
    System['dynamics'] = BipedDynamics(System['parameters'])

    # System['constraints'] = Constraints(System['dynamics'], (20.3429, -0.3375, 0.5816, 1.3470, -0.0883, 0.1620, 2.7132))
    System['constraints'] = Constraints(System['dynamics'], opt=False, first_initial_state = [20.34290121, -0.33750024,  0.58160114,  1.3469997,  -0.08829946,  0.16200075, 2.71320124])

    trajectory = get_trajectory(System['dynamics'], System['constraints'])

    System['trajectory'] = PhaseTrajectory(**trajectory)

    # animate(System['trajectory'])

    System['transverse_linearization'] = TransverseLinearization(System['trajectory'], System['constraints'], System['dynamics'])

    K, theta, phi3, theta_dot, phi2_prime, phi3_prime, u = System['constraints'].initial_state
    state_plus = np.array([theta, -theta, phi3, theta_dot, phi2_prime * theta_dot, phi3_prime * theta_dot])

    feedback = Feedback(System['transverse_linearization'])
    sim = BipedSimulator(System['parameters'], feedback)
    result = sim.run(state_plus, 0, 20.0, get_last_step=True)
    animate(result.trajectory, redraw_flag=1)

    # ONE MORE TIME

    System['trajectory'] = result.trajectory
    System['constraints'] = Constraints(System['dynamics'], opt=False, phase_trajectory = result.trajectory)
    System['transverse_linearization'] = TransverseLinearization(System['trajectory'], System['constraints'], System['dynamics'])

    K, theta, phi3, theta_dot, phi2_prime, phi3_prime, u = System['constraints'].initial_state
    state_plus = np.array([theta, -theta, phi3, theta_dot, phi2_prime * theta_dot, phi3_prime * theta_dot])

    feedback = Feedback(System['transverse_linearization'])
    sim = BipedSimulator(System['parameters'], feedback)
    result = sim.run(state_plus, 0, 20.0, get_last_step=False)
    animate(result.trajectory, redraw_flag=1)

