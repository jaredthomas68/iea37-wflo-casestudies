
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

    EF = 1.0
    TI = 5
    run_id = 0
    layout_file = "snopt_multistart_locations_9turbs_iea37casesWindRose_16dirs_BPA_run%i_EF%.3f_TItype%i.yaml" % (layout_number, EF, TI)
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
    boundary_radius = 900.
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