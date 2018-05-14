import numpy as np
import coordinate_methods
import cell_list_method
import cdist_method


def measure_chi_4(filename, num_spatial_dimensions, frame_cutoff, particle_diameter, use_cell_list=1):

    particle_positions = coordinate_methods.read_xyz_file(filename, num_spatial_dimensions)
    if use_cell_list:
        distance, distance_squared = cell_list_method.get_all_overlaps(particle_positions, frame_cutoff, 0.3 * particle_diameter)
    else:
        distance, distance_squared = cdist_method.get_all_overlaps(particle_positions, frame_cutoff, 0.3 * particle_diameter)

    normalised_distance = normalise_by_observations(distance)
    normalised_distancesquared = normalise_by_observations(distance_squared)

    chi_squared = normalised_distancesquared - normalised_distance ** 2

    density = get_particle_density(particle_positions, particle_diameter)
    average_particles = get_average_particles(particle_positions)

    temperature = 1
    chi_squared = chi_squared * density / average_particles * temperature

    return chi_squared


def normalise_by_observations(result_list):
    """
    Normalise the overlap by dividing the overlap by the number of observations
    :param result_list: A 2 by N list where N is the number of particles. The first column is the number of overlaps,
    the second is the number of observations
    :return: a list of normalised overlap values
    """
    # Some time separations may have zero observations
    with np.errstate(divide='ignore', invalid='ignore'):
        normalised_array = result_list[:, 0] / result_list[:, 1]
    return normalised_array


def get_average_particles(particle_positions):
    """ Given a set of frames of particle positions get the average number of particles in each frame
    :param particle_positions: A list of f numpy arrays of size N by d where f is the number of frames,
     N is the number of particles and d is the number of spatial dimensions
    :return: The average number fo particles per frame for the series
    """
    total_particles = 0

    for frame in particle_positions:
        total_particles += len(frame)

    average_particles = total_particles / len(particle_positions)

    return average_particles


def get_particle_density(particle_positions, particle_diameter):
    """ Get particle density by frame and take an average. Calculating by frame prevents issues due to drift.
    Get the box size from the first frame and divide by the average number of particles to get the particle density
    :param particle_positions: A list of f numpy arrays of size N by d where f is the number of frames,
     N is the number of particles and d is the number of spatial dimensions
    :param particle_diameter: The diameter of a particle in the units of the coordinates
    :return: The average density of the box througout the simulation in units of particle diamters
    """
    total_density = 0

    for frame in particle_positions:
        frame_particles = len(frame)
        lengths = (np.amax(frame, axis=0) - np.amin(frame, axis=0)) / particle_diameter

        volume = 1
        for side_length in lengths:
            volume *= side_length

        total_density += volume / frame_particles

    average_density = total_density / len(particle_positions)

    return average_density
