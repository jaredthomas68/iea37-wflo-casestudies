from __future__ import print_function   # For Python 3 compatibility
import numpy as np
import sys
import yaml                             # For reading .yaml files
from math import radians as DegToRad    # For converting degrees to radians
from openmdao.api import Problem, Group, Component, pyOptSparseDriver, IndepVarComp, ExecComp
from plantenergy.GeneralWindFarmComponents import BoundaryComp, SpacingComp, config

from iea37aepcalc_wec import *

class AEPcomp(Component):

    def __init__(self, nTurbines, nDirections):

        super(AEPcomp, self).__init__()

        self.nTurbines = nTurbines

        self.fd_options['force_fd'] = True

        # Turbine properties
        self.add_param('turbineX', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineY', val=np.zeros(nTurbines), units='m')
        self.add_param('rotorDiameter', val=0.0, units='m')
        self.add_param('turb_ci', val=0.0)
        self.add_param('turb_co', val=0.0)
        self.add_param('rated_ws', val=0.0)
        self.add_param('rated_pwr', val=0.0)

        # Wind flow properties
        self.add_param('windSpeed', val=0.0, units='m/s')
        self.add_param('windFrequency', val=np.zeros(nDirections))
        self.add_param('windDirection', val=np.zeros(nDirections), units='deg')

        # WEC
        self.add_param('rel_fac', val=1.0)

        self.add_output('AEP', val=0.0)
        self.add_output('dirPowers', val=np.zeros(nDirections))

    def solve_nonlinear(self, params, unknowns, resids):
        # Turbine properties
        turbineX = params['turbineX']
        turbineY = params['turbineY']
        rotorDiameter = params['rotorDiameter']
        turb_ci = params['turb_ci']
        turb_co = params['turb_co']
        rated_ws = params['rated_ws']
        rated_pwr = params['rated_pwr']

        # Wind flow properties
        windSpeed = params['windSpeed']
        windFrequency = params['windFrequency']
        windDirection = params['windDirection']

        # WEC
        rel_fac = params['rel_fac']

        # convert to required data type
        turb_coords.x = turbineX
        turb_coords.y = turbineY

        # get AEP
        AEP = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                        turb_diam, turb_ci, turb_co, rated_ws, rated_pwr, rel_fac)

        config.obj_func_calls += 1
        unknowns['AEP'] = np.sum(AEP)
        unknowns['dirPowers'] = AEP

    def linearize(self, params, unknowns, resids):

        J = {}

        return J

if __name__ == "__main__":
    """Used for demonstration.

    An example command line syntax to run this file is:

        python iea37-aepcalc.py iea37-ex16.yaml

    For Python .yaml capability, in the terminal type "pip install pyyaml".
    """
    global func_calls
    func_calls = 0

    input_val = sys.argv[1]
    nTurbines = int(input_val)
    loc_file = 'iea37-ex%i.yaml' % nTurbines

    # Read necessary values from .yaml files
    # Get turbine locations and auxiliary <.yaml> filenames
    turb_coords, fname_turb, fname_wr = getTurbLocYAML(loc_file)
    # Get the array wind sampling bins, frequency at each bin, and wind speed
    wind_dir, wind_freq, wind_speed = getWindRoseYAML(fname_wr)
    # Pull the needed turbine attributes from file
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML(
        fname_turb)

    rel_factors = np.array([3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0])
    # nTurbines = turb_coords.x.size
    min_spacing = 2.0
    boundary_center = np.array([0.0, 0.0])

    if nTurbines == 16:
        boundary_radius = 1300.
    elif nTurbines == 36:
        boundary_radius = 2000.
    elif nTurbines == 64:
        boundary_radius = 3000.

    prob = Problem(root=Group())
    prob.root.add('AEPcomp', AEPcomp(nTurbines=nTurbines, nDirections=wind_dir.size), promotes=['*'])

    # add indep var comps for des vars
    prob.root.add('dv0', IndepVarComp('turbineX', val=np.zeros(nTurbines), units='m'), promotes=['*'])
    prob.root.add('dv1', IndepVarComp('turbineY', val=np.zeros(nTurbines), units='m'), promotes=['*'])

    # # set up turbine spacing constraint
    prob.root.add('SpacingConstraint', SpacingComp(nTurbines=nTurbines), promotes=['*'])
    prob.root.add('sv0', IndepVarComp('rotorDiameter', val=0.0, units='m'), promotes=['*'])
    prob.root.add('spacing_con', ExecComp('sc = wtSeparationSquared-(minSpacing*rotorDiameter)**2',
                                     minSpacing=np.array([min_spacing]), rotorDiameter=0.0,
                                     sc=np.zeros(int(((nTurbines - 1.) * nTurbines / 2.))),
                                     wtSeparationSquared=np.zeros(int(((nTurbines - 1.) * nTurbines / 2.)))), promotes=['*'])


    # set up boundary constraint
    prob.root.add('BoundaryConstraint', BoundaryComp(nTurbines=nTurbines, nVertices=1), promotes=['*'])
    prob.root.add('bv0', IndepVarComp('boundary_radius', val=0.0, units='m',
                                 pass_by_obj=True, desc='radius of wind farm boundary'), promotes=['*'])
    prob.root.add('bv1', IndepVarComp('boundary_center', val=np.array([0., 0.]), units='m', pass_by_obj=True,
                                 desc='x and y positions of circular wind farm boundary center'), promotes=['*'])

    # set up objective
    prob.root.add('obj_comp', ExecComp('obj = -1.*AEP', AEP=0.0), promotes=['*'])

    # set up optimizer
    prob.root.deriv_options['type'] = 'fd'
    prob.root.deriv_options['form'] = 'forward'

    # set optimizer options (pyoptsparse)
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'  # NSGA2, CONMIN, SNOPT, SLSQP, COBYLA
    # prob.driver.options['gradient method'] = 'snopt_fd'
    prob.driver.opt_settings['Major optimality tolerance'] = 1E-4


    # add design variables
    prob.driver.add_desvar('turbineX')
    prob.driver.add_desvar('turbineY')

    # add constraints
    prob.driver.add_constraint('sc', lower=np.zeros(int((nTurbines - 1.) * nTurbines / 2.)), scaler=1.0E-2, active_tol=2. * turb_diam**2)
    prob.driver.add_constraint('boundaryDistances', lower=(np.zeros(1 * nTurbines)), scaler=1E-2, active_tol=2. * turb_diam )

    prob.driver.add_objective('obj', scaler=1E-3)

    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True
    prob.root.ln_solver.options['mode'] = 'rev'

    prob.setup(check=True)

    # turbine properties
    prob['turbineX'] = turb_coords.x
    prob['turbineY'] = turb_coords.y
    prob['rotorDiameter'] = turb_diam
    prob['turb_ci'] = turb_ci
    prob['turb_co'] = turb_co
    prob['rated_ws'] = rated_ws
    prob['rated_pwr'] = rated_pwr

    # farm properties
    prob['boundary_radius'] = boundary_radius
    prob['boundary_center'] = boundary_center

    # flow properties
    prob['windSpeed'] = wind_speed
    prob['windFrequency'] = wind_freq
    prob['windDirection'] = wind_dir

    for rel_fac in rel_factors:
        prob.driver.opt_settings['Print file'] = 'SNOPT_print_case1_turbs%i_rel%.2f.out' % (nTurbines, rel_fac)
        prob.driver.opt_settings['Summary file'] = 'SNOPT_summary_case1_turbs%i_rel%.2f.out' % (nTurbines, rel_fac)
        prob['rel_fac'] = rel_fac
        print('starting rel fac:', rel_fac)
        prob.run()
        # prob.run_once()
        print('AEP, ef:', prob['AEP'], rel_fac)
    AEP = prob['AEP']

    outfilename = 'case1_loc_turbs%i.txt' % nTurbines
    print(AEP)
    print('ndirs:', wind_dir.size)
    print(config.obj_func_calls)
    np.savetxt(outfilename, np.c_[prob['turbineX'], prob['turbineY']], header='X, Y, direction, dir power, aep, fcalls')
    print('dirc, pow:', np.c_[prob['windDirection'], prob['dirPowers']])
    print('AEP:', prob['AEP'])
    print('Func calls:', config.obj_func_calls)
    # Print AEP for each binned direction, with 5 digits behind the decimal.
    print(np.array2string(AEP, precision=5, floatmode='fixed',
                          separator=', ', max_line_width=62))
    # Print AEP summed for all directions
    print(np.around(np.sum(AEP), decimals=5))
