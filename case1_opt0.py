import numpy as np
import matplotlib.pylab as plt
import sys
from openmdao.api import Problem, pyOptSparseDriver
from iea37aepcalc_wec import getTurbAtrbtYAML, getTurbLocYAML, getWindRoseYAML
from ieacase1.OptimizationGroups import OptAEP
from ieacase1.iea_bp_model_wrapper import iea_bp_wrapper, add_iea_bp_params_IndepVarComps
from ieacase1 import config

from time import time
if __name__ == "__main__":
    """Used for demonstration.

    An example command line syntax to run this file is:

        python iea37-aepcalc.py iea37-ex16.yaml

    For Python .yaml capability, in the terminal type "pip install pyyaml".
    """
    input_val = sys.argv[1]
    input_val = 64
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

    nDirections = wind_dir.size

    rel_factors = np.array([3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25, 1.0])
    # rel_factors = np.array([3.0, 2.0, 1.0])
    # nTurbines = turb_coords.x.size
    min_spacing = 2.0
    boundary_center = np.array([0.0, 0.0])

    if nTurbines == 16:
        boundary_radius = 1300.
    elif nTurbines == 36:
        boundary_radius = 2000.
    elif nTurbines == 64:
        boundary_radius = 3000.

    prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, wake_model=iea_bp_wrapper,
                               use_rotor_components=False, nVertices=1, differentiable=True,
                               params_IdepVar_func=add_iea_bp_params_IndepVarComps, params_IndepVar_args={},
                               rec_func_calls = True))


    # set optimizer options (pyoptsparse)
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'  # NSGA2, CONMIN, SNOPT, SLSQP, COBYLA
    # prob.driver.options['gradient method'] = 'snopt_fd'
    prob.driver.opt_settings['Major optimality tolerance'] = 1E-4
    prob.driver.opt_settings['Verify level'] = 0


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
    prob['rotorDiameter'] = np.ones(nTurbines)*turb_diam
    prob['cut_in_speed'] = np.ones(nTurbines)*turb_ci
    prob['cut_out_speed'] = np.ones(nTurbines)*turb_co
    prob['rated_wind_speed'] = np.ones(nTurbines)*rated_ws
    prob['rated_power'] = np.ones(nTurbines)*rated_pwr

    # farm properties
    prob['boundary_radius'] = boundary_radius
    prob['boundary_center'] = boundary_center

    # flow properties
    prob['windSpeeds'] = np.ones(nDirections)*wind_speed
    prob['windFrequencies'] = wind_freq
    prob['windDirections'] = wind_dir
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(turb_coords.x, turb_coords.y, c='b')
    tic = time()
    for rel_fac in rel_factors:
        prob.driver.opt_settings['Print file'] = 'SNOPT_print_case1_turbs%i_rel%.2f.out' % (nTurbines, rel_fac)
        prob.driver.opt_settings['Summary file'] = 'SNOPT_summary_case1_turbs%i_rel%.2f.out' % (nTurbines, rel_fac)
        prob['model_params:wec'] = rel_fac
        print('starting rel fac:', rel_fac)
        prob.run()
        # prob.run_once()
        print('AEP, ef:', prob['AEP'], rel_fac)
        # prob.check_partial_derivatives()

    toc = time()
    AEP = prob['AEP']
    # plt.figure()
    # ax[1].scatter(prob['turbineX'], prob['turbineY'], c='r')
    # boundary0 = plt.Circle([0., 0.], boundary_radius, facecolor='none', edgecolor='k')
    # boundary1 = plt.Circle([0., 0.], boundary_radius, facecolor='none', edgecolor='k')
    # ax[0].add_artist(boundary0)
    # ax[1].add_artist(boundary1)
    # plt.show()
    outfilename = 'case1_loc_turbs%i.txt' % nTurbines
    print(AEP)
    print('ndirs:', wind_dir.size)
    print(config.obj_func_calls)
    np.savetxt(outfilename, np.c_[prob['turbineX'], prob['turbineY']], header='X, Y, direction, dir power, aep, fcalls')
    print('direc, pow:', np.c_[prob['windDirections'], prob['dirPowers']])
    print('AEP:', prob['AEP']*1E-6)
    print('AEP imp:', (prob['AEP']*1E-6 - 366941.57116)/366941.57116)
    print('Func calls:', np.sum(config.obj_func_calls_array + config.sens_func_calls_array))
    print('approx total time:', toc-tic)
    # Print AEP for each binned direction, with 5 digits behind the decimal.
    # print(np.array2string(AEP, precision=5, floatmode='fixed',
    #                       separator=', ', max_line_width=62))
    # # Print AEP summed for all directions
    # print(np.around(np.sum(AEP), decimals=5))