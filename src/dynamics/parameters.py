import json
from dataclasses import dataclass
from os.path import exists
from importlib.resources import files, open_text
# import scipy as sp

@dataclass
class BipedParameters:
    leg_mass: float
    hip_mass: float
    torso_mass: float
    leg_length: float
    leg_com: float
    torso_com: float
    gravity_acceleration: float
    
    def __post_init__(self):
        self.K = 20.34290121
    

def load_biped_parameters(configname):
    if exists(configname):
        with open(configname, 'r') as f:
            return BipedParameters(**json.load(f))

    if files('config').joinpath(configname).exists():
        with open_text('config', configname) as f:
            return BipedParameters(**json.load(f))

    assert False, f'Can\'t find {configname}'