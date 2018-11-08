
from matplotlib import pylab as plt
import numpy as np
import yaml

def plot_round_farm(turbineX, turbineY, rotor_diameter, boundary_center, boundary_radius, min_spacing=2.,
                    save_start=False, show_start=True, save_file=None):
    boundary_center_x = boundary_center[0]
    boundary_center_y = boundary_center[1]

    boundary_circle = plt.Circle((boundary_center_x / rotor_diameter, boundary_center_y / rotor_diameter),
                                 boundary_radius / rotor_diameter, facecolor='none', edgecolor='k', linestyle='--')
    farm_boundary_circle = plt.Circle((boundary_center_x / rotor_diameter, boundary_center_y / rotor_diameter),
                                 boundary_radius / rotor_diameter + 0.5, facecolor='none', edgecolor='k', linestyle='-')

    fig, ax = plt.subplots()
    for x, y in zip(turbineX / rotor_diameter, turbineY / rotor_diameter):
        circle_start = plt.Circle((x, y), 0.5, facecolor='none', edgecolor='r', linestyle='-', label='Start')
        ax.add_artist(circle_start)
    # ax.plot(turbineX / rotor_diameter, turbineY / rotor_diameter, 'sk', label='Original', mfc=None)
    # ax.plot(prob['turbineX'] / rotor_diameter, prob['turbineY'] / rotor_diameter, '^g', label='Optimized', mfc=None)
    ax.add_patch(boundary_circle)
    ax.add_patch(farm_boundary_circle)
    plt.axis('equal')
    ax.legend([circle_start], ['turbines'])
    ax.set_xlabel('Turbine X Position ($X/D_r$)')
    ax.set_ylabel('Turbine Y Position ($Y/D_r$)')
    ax.set_xlim([(boundary_center_x - boundary_radius) / rotor_diameter - 1.,
                 (boundary_center_x + boundary_radius) / rotor_diameter + 1.])
    ax.set_ylim([(boundary_center_y - boundary_radius) / rotor_diameter - 1.,
                 (boundary_center_y + boundary_radius) / rotor_diameter + 1.])

    if save_start:
        if save_file is None:
            plt.savefig('round_farm_%iTurbines_%0.2fDSpacing.pdf' % (turbineX.size, min_spacing))
        else:
            plt.savefig(save_file)
    if show_start:
        plt.show()

if __name__ == "__main__":


    # load starting locations
    layout_number = 0
    # input_directory = "../../input_files/layouts/"
    input_directory = "./output_files_snopt_wec/"

    data = np.loadtxt(input_directory+'snoptwec_multistart_rundata_64turbs_iea37casesWindRose_16dirs_IEABPA_all.txt')
    ef = data[:, 1]
    id = data[:, 0]

    run_time = data[:, 8]
    run_time_sum = np.zeros(200)
    for i in np.arange(0, 200):
        run_time_sum[i] = np.sum(run_time[id==i])
    data = data[ef == 1, :]


    shift = 1
    id = data[:, 0]
    orig_aep = data[id==0, 3 + shift]
    end_aep = data[:, 5 + shift]
    fcalls = data[:, 8 + shift]
    scalls = data[:, 9 + shift]


    best_run_idx = np.argmax(end_aep)

    print "average aep was:", np.average(end_aep)
    print "std aep was:", np.std(end_aep)
    print "\n"
    print "average improvement was:", np.average((end_aep-orig_aep)/orig_aep)
    print "std improvement was:", np.std((end_aep-orig_aep)/orig_aep)
    print "\n"
    print "best run was:", id[best_run_idx]
    print "AEP was:", end_aep[best_run_idx]
    print "improvement was:", (end_aep[best_run_idx]-orig_aep)/orig_aep
    print "run time:", run_time_sum[best_run_idx]
    print "fcalls:", np.sum([fcalls[best_run_idx], scalls[best_run_idx]])

    print "run 0 imp:", (end_aep[id==0]-orig_aep)/orig_aep

    EF = 1.0
    TI = 0
    run_id = id[np.argmax(end_aep)]
    layout_file = "snoptwec_multistart_locations_64turbs_iea37casesWindRose_16dirs_IEABPA_run%i_EF%.3f_TItype%i.yaml" % (layout_number, EF, TI)
    rotor_diameter = 130.

    with open(input_directory + layout_file, 'r') as f:
        layout_data = yaml.safe_load(f)

    turbineX = np.asarray(layout_data['definitions']['position']['items']['xc'])[0, :]
    turbineY = np.asarray(layout_data['definitions']['position']['items']['yc'])[0, :]

    nTurbines = turbineX.size
    # ef 5to1 1.4381417805509602
    # ef 3 to 1 1.4376982363067519
    # ef 5to1 by 0.25 imp = 1.4378647258529969
    # create boundary specifications
    if nTurbines == 16:
        boundary_radius = 1300.
    elif nTurbines == 36:
        boundary_radius = 2000.
    elif nTurbines == 64:
        boundary_radius = 3000.
    center = np.array([0.0, 0.0])
    start_min_spacing = 5.
    nVertices = 1
    boundary_center_x = center[0]
    boundary_center_y = center[1]
    # xmax = np.max(turbineX)
    # ymax = np.max(turbineY)
    # xmin = np.min(turbineX)
    # ymin = np.min(turbineY)
    boundary_radius_plot = boundary_radius + 0.5 * rotor_diameter

    plot_round_farm(turbineX, turbineY, rotor_diameter, [boundary_center_x, boundary_center_y], boundary_radius)