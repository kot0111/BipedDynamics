from dynamics.parameters import load_biped_parameters
from dynamics.biped_dynamics import BipedDynamics, Constraints
from trajectory.trajectory import PhaseTrajectory, get_trajectory
from sim.biped_simulation import animate
from transverse_linearization.linearization import TransverseLinearization


if __name__ == "__main__":

    System = {}
    
    System['parameters'] = load_biped_parameters("biped.json")
    System['dynamics'] = BipedDynamics(System['parameters'])

    # System['constraints'] = Constraints(System['dynamics'], (20.3429, -0.3375, 0.5816, 1.3470, -0.0883, 0.1620, 2.7132))
    System['constraints'] = Constraints(System['dynamics'], [20.34290121, -0.33750024,  0.58160114,  1.3469997,  -0.08829946,  0.16200075, 2.71320124], opt=False)

    trajectory = get_trajectory(System['dynamics'], System['constraints'])

    System['trajectory'] = PhaseTrajectory(**trajectory)

    animate(System['trajectory'])

    System['transverse_linearization'] = TransverseLinearization(System['trajectory'], System['constraints'], System['dynamics'])


