import numpy as np
from scipy.integrate import quad
from scipy.io import loadmat
import pylab as plt
import time

from openmdao.api import Component, Problem, Group

from _iea_fortran import iea_bp_model_fortran, iea_bp_model_fortran_dv

class iea_bp_model(Component):

    def __init__(self, nTurbines, direction_id=0, options=None):
        super(iea_bp_model, self).__init__()

        self.deriv_options['type'] = 'user'

        self.nTurbines = nTurbines
        self.direction_id = direction_id

        # used
        self.add_param('turbineXw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineYw', val=np.zeros(nTurbines), units='m')
        self.add_param('rotorDiameter', val=np.zeros(nTurbines)+126.4, units='m')
        self.add_param('wind_speed', val=8.0, units='m/s')

        # other
        self.add_param('axialInduction', val=np.zeros(nTurbines))
        self.add_param('generatorEfficiency', val=np.zeros(nTurbines))
        self.add_param('hubHeight', val=np.zeros(nTurbines))
        self.add_param('yaw%i' % direction_id, val=np.zeros(nTurbines))
        self.add_param('Ct' % direction_id, val=np.zeros(nTurbines))

        self.add_param('model_params:wec', val=1.0)

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')

    def solve_nonlinear(self, params, unknowns, resids):

        nTurbines = self.nTurbines
        direction_id = self.direction_id

        # rename inputs and outputs
        turbineXw = params['turbineXw']
        turbineYw = params['turbineYw']
        rotorDiameter = params['rotorDiameter']
        wind_speed = params['wind_speed']
        wec = params['model_params:wec']

        velocitiesTurbines = iea_bp_model_fortran(turbineXw, turbineYw, rotorDiameter, wind_speed, wec)

        unknowns['wtVelocity%i' % direction_id] = velocitiesTurbines


    def linearize(self, params, unknowns, resids):
        nTurbines = self.nTurbines
        direction_id = self.direction_id

        # rename inputs and outputs
        turbineXw = params['turbineXw']
        turbineYw = params['turbineYw']
        rotorDiameter = params['rotorDiameter']
        wind_speed = params['wind_speed']
        wec = params['model_params:wec']

        # define jacobian size
        nTurbines = len(turbineXw)
        nDirs = nTurbines

        # define input array to direct differentiation
        wtVelocityb = np.eye(nDirs, nTurbines)

        turbineXwd = np.eye(nDirs, nTurbines)
        turbineYwd = np.zeros([nDirs, nTurbines])
        rotorDiameterd = np.zeros([nDirs, nTurbines])

        _, wtVelocityb = iea_bp_model_fortran_dv(turbineXw, turbineXwd, turbineYw, turbineYwd,
                             rotorDiameter, rotorDiameterd, wind_speed, wec)

        wtVelocityb_dxwd = wtVelocityb

        turbineXwd = np.zeros([nDirs, nTurbines])
        turbineYwd = np.eye(nDirs, nTurbines)
        rotorDiameterd = np.zeros([nDirs, nTurbines])

        _, wtVelocityb = iea_bp_model_fortran_dv(turbineXw, turbineXwd, turbineYw, turbineYwd,
                             rotorDiameter, rotorDiameterd, wind_speed, wec)

        wtVelocityb_dywd = wtVelocityb

        turbineXwd = np.zeros([nDirs, nTurbines])
        turbineYwd = np.zeros([nDirs, nTurbines])
        rotorDiameterd = np.eye(nDirs, nTurbines)

        _, wtVelocityb = iea_bp_model_fortran_dv(turbineXw, turbineXwd, turbineYw, turbineYwd,
                                              rotorDiameter, rotorDiameterd, wind_speed, wec)

        wtVelocityb_drd = wtVelocityb

        # initialize Jacobian dict
        J = {}

        J['wtVelocity%i' % direction_id, 'turbineXw'] = np.transpose(wtVelocityb_dxwd)
        J['wtVelocity%i' % direction_id, 'turbineYw'] = np.transpose(wtVelocityb_dywd)
        J['wtVelocity%i' % direction_id, 'rotorDiameter'] = np.transpose(wtVelocityb_drd)

        return J


if __name__ == "__main__":

    nTurbines = 2
    nDirections = 1

    rotor_diameter = 126.4
    rotorArea = np.pi*rotor_diameter*rotor_diameter/4.0
    axialInduction = 1.0/3.0
    CP = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    # CP =0.768 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    CT = 4.0*axialInduction*(1.0-axialInduction)
    generator_efficiency = 0.944

    # Define turbine characteristics
    axialInduction = np.array([axialInduction, axialInduction])
    rotorDiameter = np.array([rotor_diameter, rotor_diameter])
    # rotorDiameter = np.array([rotorDiameter, 0.0001*rotorDiameter])
    yaw = np.array([0., 0.])

    # Define site measurements
    wind_direction = 30.
    wind_speed = 8.    # m/s
    air_density = 1.1716

    Ct = np.array([CT, CT])
    Cp = np.array([CP, CP])

    turbineX = np.array([0.0, 7.*rotor_diameter])
    turbineY = np.array([0.0, 0.0])

    prob = Problem()
    prob.root = Group()
    prob.root.add('model', GaussianWake(nTurbines), promotes=['*'])

    prob.setup()

    prob['turbineXw'] = turbineX
    prob['turbineYw'] = turbineY

    GaussianWakeVelocity = list()

    yawrange = np.linspace(-40., 40., 400)

    for yaw1 in yawrange:

        prob['yaw0'] = np.array([yaw1, 0.0])
        prob['Ct'] = Ct*np.cos(prob['yaw0']*np.pi/180.)**2

        prob.run()

        velocitiesTurbines = prob['wtVelocity0']

        GaussianWakeVelocity.append(list(velocitiesTurbines))

    GaussianWakeVelocity = np.array(GaussianWakeVelocity)

    fig, axes = plt.subplots(ncols=2, nrows=2, sharey=False)
    axes[0, 0].plot(yawrange, GaussianWakeVelocity[:, 0]/wind_speed, 'b')
    axes[0, 0].plot(yawrange, GaussianWakeVelocity[:, 1]/wind_speed, 'b')

    axes[0, 0].set_xlabel('yaw angle (deg.)')
    axes[0, 0].set_ylabel('Velcoity ($V_{eff}/V_o$)')

    posrange = np.linspace(-3.*rotor_diameter, 3.*rotor_diameter, 100)

    prob['yaw0'] = np.array([0.0, 0.0])

    GaussianWakeVelocity = list()

    for pos2 in posrange:

        prob['turbineYw'] = np.array([0.0, pos2])

        prob.run()

        velocitiesTurbines = prob['wtVelocity0']

        GaussianWakeVelocity.append(list(velocitiesTurbines))

    GaussianWakeVelocity = np.array(GaussianWakeVelocity)

    wind_speed = 1.0
    axes[0, 1].plot(posrange/rotor_diameter, GaussianWakeVelocity[:, 0]/wind_speed, 'b')
    axes[0, 1].plot(posrange/rotor_diameter, GaussianWakeVelocity[:, 1]/wind_speed, 'b')
    axes[0, 1].set_xlabel('y/D')
    axes[0, 1].set_ylabel('Velocity ($V_{eff}/V_o$)')

    posrange = np.linspace(-3.*rotorDiameter[0], 3.*rotorDiameter[0], num=1000)

    posrange = np.linspace(-1.*rotorDiameter[0], 30.*rotorDiameter[0], num=2000)
    yaw = np.array([0.0, 0.0])
    wind_direction = 0.0

    GaussianWakeVelocity = list()
    for pos2 in posrange:

        prob['turbineXw'] = np.array([0.0, pos2])
        prob['turbineYw'] = np.array([0.0, 0.0])

        prob.run()

        velocitiesTurbines = prob['wtVelocity0']

        GaussianWakeVelocity.append(list(velocitiesTurbines))

    GaussianWakeVelocity = np.array(GaussianWakeVelocity)

    axes[1, 1].plot(posrange/rotor_diameter, GaussianWakeVelocity[:, 1], 'y', label='GH')
    axes[1, 1].plot(np.array([7, 7]), np.array([2, 8]), '--k', label='tuning point')

    plt.xlabel('x/D')
    plt.ylabel('Velocity (m/s)')
    plt.legend(loc=4)

    plt.show()