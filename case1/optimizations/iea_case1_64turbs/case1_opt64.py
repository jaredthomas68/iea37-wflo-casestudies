from __future__ import print_function

import numpy as np
import matplotlib.pylab as plt
import sys, yaml
from openmdao.api import Problem, pyOptSparseDriver
from iea37aepcalc_wec import getTurbAtrbtYAML, getTurbLocYAML, getWindRoseYAML
from ieacase1.OptimizationGroups import OptAEP
from ieacase1.iea_bp_model_wrapper import iea_bp_wrapper, add_iea_bp_params_IndepVarComps
from ieacase1 import config


from time import time
if __name__ == "__main__":

    ######################### for MPI functionality #########################
    from openmdao.core.mpi_wrap import MPI

    if MPI:  # pragma: no cover
        # if you called this script with 'mpirun', then use the petsc data passing
        from openmdao.core.petsc_impl import PetscImpl as impl

        print("In MPI, impl = ", impl)

    else:
        # if you didn't use 'mpirun', then use the numpy data passing
        from openmdao.api import BasicImpl as impl


    def mpi_print(prob, *args):
        """ helper function to only print on rank 0 """
        if prob.root.comm.rank == 0:
            print(*args)


    prob = Problem(impl=impl)

    #########################################################################

    """Used for demonstration.

    An example command line syntax to run this file is:

        python iea37-aepcalc.py iea37-ex16.yaml

    For Python .yaml capability, in the terminal type "pip install pyyaml".
    """
    runID = int(sys.argv[1])
    nTurbines = 64
    layout_directory = '../../input_files/layouts/'
    loc_file = 'nTurbs%i_spacing5_layout_%i.yaml' % (nTurbines, runID)

    output_directory = 'output_files_snopt_wec/'
    opt_algorithm = 'snoptwec'
    wind_rose_file = 'iea37cases'
    size = 16
    model = 'IEABPA'
    ti_calculation_method = 0
    ti_opt_method = 0

    # Read necessary values from .yaml files
    # Get turbine locations and auxiliary <.yaml> filenames
    turb_coords, fname_turb, fname_wr = getTurbLocYAML(layout_directory+loc_file)
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

    prob = Problem(impl=impl, root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, wake_model=iea_bp_wrapper,
                               use_rotor_components=False, nVertices=1, differentiable=True,
                               params_IdepVar_func=add_iea_bp_params_IndepVarComps, params_IndepVar_args={},
                               rec_func_calls = True))


    # set optimizer options (pyoptsparse)
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'  # NSGA2, CONMIN, SNOPT, SLSQP, COBYLA
    # prob.driver.options['gradient method'] = 'snopt_fd'
    prob.driver.opt_settings['Major optimality tolerance'] = 1E-0
    prob.driver.opt_settings['Verify level'] = 0


    # add design variables
    prob.driver.add_desvar('turbineX', lower=np.zeros(nTurbines)-boundary_radius, upper=np.zeros(nTurbines)+boundary_radius)
    prob.driver.add_desvar('turbineY', lower=np.zeros(nTurbines)-boundary_radius, upper=np.zeros(nTurbines)+boundary_radius)

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

    prob.run_once()
    AEP_init_calc = AEP_init_opt = prob['AEP']

    tic = time()
    for rel_fac in rel_factors:
        prob.driver.opt_settings['Print file'] = output_directory + 'SNOPT_print_case1_turbs%i_runID%i_rel%.2f.out' % (
        nTurbines, runID, rel_fac)
        prob.driver.opt_settings[
            'Summary file'] = output_directory + 'SNOPT_summary_case1_turbs%i_runID%i_rel%.2f.out' % (
        nTurbines, runID, rel_fac)

        prob['model_params:wec'] = rel_fac
        print('starting rel fac:', rel_fac)
        tictic = time()
        prob.run()
        toctoc = time()
        run_time = toctoc - tictic
        # prob.run_once()
        print('AEP, ef:', prob['AEP'], rel_fac)
        # prob.check_partial_derivatives()
        print( "improvement: ", (prob['AEP']-AEP_init_calc)/AEP_init_calc)

        AEP_run_calc = AEP_run_opt = prob['AEP']

        if prob.root.comm.rank == 0:
            with open(layout_directory + "iea37-ex%i.yaml" % (nTurbines), 'r') as f:
                loaded_yaml = yaml.safe_load(f)

            loaded_yaml['definitions']['position']['items']['xc'] = np.matrix.tolist(
                np.matrix(prob['turbineX']))
            loaded_yaml['definitions']['position']['items']['yc'] = np.matrix.tolist(
                np.matrix(prob['turbineY']))

            loaded_yaml['definitions']['plant_energy']['properties']['annual_energy_production'][
                'binned'] = np.matrix.tolist(np.matrix(prob['dirPowers']*24*365*wind_freq*1E-6))
            loaded_yaml['definitions']['plant_energy']['properties']['annual_energy_production']['default'] = \
                float(prob['AEP'] * 1E-6)

            loaded_yaml['definitions']['plant_energy']['properties']['wake_model_selection']['items'][
                0] = 'byuflowlab/BastankhahAndPorteAgel, byuflowlab/wakeexchange/plantenergy'
            print(rel_fac, ti_opt_method)
            with open(
                    output_directory + '%s_multistart_locations_%iturbs_%sWindRose_%idirs_%s_run%i_EF%.3f_TItype%i.yaml' % (
                            opt_algorithm, nTurbines, wind_rose_file, size, model, runID,
                            rel_fac, ti_opt_method),
                    "w") as f:
                yaml.dump(loaded_yaml, f)
            # if save_time:
            #     np.savetxt(output_directory + '%s_multistart_time_%iturbs_%sWindRose_%idirs_%s_run%i_EF%.3f.txt' % (
            #         opt_algorithm, nTurbs, wind_rose_file, size, MODELS[model], run_number, expansion_factor),
            #                np.c_[run_time],
            #                header="run time")
            output_file = output_directory + '%s_multistart_rundata_%iturbs_%sWindRose_%idirs_%s_run%i.txt' \
                          % (opt_algorithm, nTurbines, wind_rose_file, size, model, runID)
            f = open(output_file, "a")

            if rel_fac == rel_factors[0]:
                header = "run number, exp fac, ti calc, ti opt, aep init calc (kW), aep init opt (kW), " \
                         "aep run calc (kW), aep run opt (kW), run time (s), obj func calls, sens func calls"
            else:
                header = ''

            np.savetxt(f, np.c_[runID, rel_fac, ti_calculation_method, ti_opt_method,
                                AEP_init_calc, AEP_init_opt, AEP_run_calc, AEP_run_opt, run_time,
                                config.obj_func_calls_array[0], config.sens_func_calls_array[0]],
                       header=header)
            f.close()

    toc = time()
    AEP = prob['AEP']
    # plt.figure()
    # ax[1].scatter(prob['turbineX'], prob['turbineY'], c='r')
    # boundary0 = plt.Circle([0., 0.], boundary_radius, facecolor='none', edgecolor='k')
    # boundary1 = plt.Circle([0., 0.], boundary_radius, facecolor='none', edgecolor='k')
    # ax[0].add_artist(boundary0)
    # ax[1].add_artist(boundary1)
    # plt.show()
    # outfilename = 'case1_loc_turbs%i.txt' % nTurbines
    print(AEP)
    print('ndirs:', wind_dir.size)
    print(config.obj_func_calls)
    # np.savetxt(outfilename, np.c_[prob['turbineX'], prob['turbineY']], header='X, Y, direction, dir power, aep, fcalls')
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

