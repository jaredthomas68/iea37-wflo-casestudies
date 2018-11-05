"""
iea_model.py
Created by Jared J. Thomas, Nov. 2018
Brigham Young University
"""

from openmdao.api import IndepVarComp, Group
from iea_bp_model import iea_bp_model
import numpy as np

def add_iea_bp_params_IndepVarComps(openmdao_object):

    # params for Bastankhah model with yaw
    openmdao_object.add('bp0', IndepVarComp('model_params:wec', val=1.0, pass_by_object=True,
                                             desc='adjust wec relaxation factor'), promotes=['*'])

class iea_bp_wrapper(Group):

    def __init__(self, nTurbs, direction_id=0, wake_model_options=None):
        super(iea_bp_wrapper, self).__init__()

        self.add('f_1', iea_bp_model(nTurbines=nTurbs, direction_id=direction_id, options=wake_model_options),
                 promotes=['*'])
