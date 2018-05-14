import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
import coordinate_methods
import cell_list


def simple_chi_four(particle_positions, frame_cutoff, displacement_threshold):
    """
    The main chi4 loop. Go through pairs of frames, get the overlap and sum it up
    :param particle_positions: A list of f numpy arrays of size N by d where f is the number of frames,
     N is the number of particles and d is the number of spatial dimensions
    :param frame_cutoff: The maximum number of frames to process for each time difference
    :param displacement_threshold: The distance threshold for determining overlap
    :return:
    """
    num_frames = len(particle_positions)
    sq_displacement_threshold = displacement_threshold ** 2
    distance = np.zeros((num_frames, 3))
    distancesquared = np.zeros((num_frames, 3))

    if frame_cutoff > num_frames:
        frame_cutoff = num_frames
    # the number of operations involves the (num_frames - 1)'th triangle number
    pbar = tqdm(total=int(((frame_cutoff - 1) ** 2 + (frame_cutoff - 1)) / 2 + (num_frames - frame_cutoff) * frame_cutoff))
    for ref_frame, ref_frame_number in enumerate(particle_positions):
        num_iterations = 0
        for cur_frame, cur_frame_number in enumerate(particle_positions, start=ref_frame_number + 1):
            overlap_count = 0
            if cur_frame_number >= num_frames - frame_cutoff:
                overlap_count += count_overlap(cur_frame, ref_frame, sq_displacement_threshold)
                num_iterations += 1
                distance[cur_frame_number - ref_frame_number, 0:2] += [overlap_count, 1]
                distancesquared[cur_frame_number - ref_frame_number, 0:2] += [(overlap_count * overlap_count), 1]
        pbar.update(num_iterations)

    return distance, distancesquared


def cell_list_chi_4(particle_positions, frame_cutoff, displacement_threshold):
    """
    The main chi4 loop. Go through pairs of frames, get the overlap and sum it up
    :param particle_positions: A list of f numpy arrays of size N by d where f is the number of frames,
     N is the number of particles and d is the number of spatial dimensions
    :param frame_cutoff: The maximum number of frames to process for each time difference
    :param displacement_threshold: The distance threshold for determining overlap
    :return:
    """
    num_frames = len(particle_positions)
    sq_displacement_threshold = displacement_threshold ** 2
    distance = np.zeros((num_frames, 3))
    distancesquared = np.zeros((num_frames, 3))

    num_cells, cell_size, box_size = cell_list.get_cell_size(particle_positions, displacement_threshold)
    cell_heads, links = cell_list.setup_cell_list(particle_positions, cell_list, num_cells)

    if frame_cutoff > num_frames:
        frame_cutoff = num_frames

    pbar = tqdm(total=int(((frame_cutoff - 1) ** 2 + (frame_cutoff - 1)) / 2 + (num_frames - frame_cutoff) * frame_cutoff))

    for cur_frame_index, cur_frame in enumerate(particle_positions):
        num_iterations = 0
        for ref_frame_index, ref_frame in enumerate(particle_positions, start=cur_frame_index + 1):
            overlap_count = 0
            for cell_vector_index in cell_list.loop_over_inner_cells(num_cells):
                cell_scalar_index = cell_list.get_scalar_cell_index(cell_vector_index, num_cells)
                for neighbour_vector_index in cell_list.loop_over_neighbour_cells(cell_vector_index, num_cells):
                    neighbour_scalar_index = cell_list.get_scalar_cell_index(neighbour_vector_index, num_cells)
                    particle_i = cell_heads[cur_frame_index][cell_scalar_index]
                    while particle_i != -1:
                        particle_j = cell_heads[ref_frame_index][neighbour_scalar_index]
                        while particle_j != -1:
                            if particle_i < particle_j:
                                overlap_count += check_overlap(particle_i, particle_j, cur_frame, ref_frame, sq_displacement_threshold)
                            particle_j = links[ref_frame_index][particle_j]
                        particle_i = links[cur_frame_index][particle_i]
            distance[cur_frame_index - ref_frame_index, 0:2] += [overlap_count, 1]
            distancesquared[cur_frame_index - ref_frame_index, 0:2] += [(overlap_count * overlap_count), 1]
            num_iterations += 1
        pbar.update(num_iterations)

    return distance, distancesquared


def check_overlap(particle_i, particle_j, cur_frame, ref_frame, squared_correlation_length):
    diff = cur_frame[particle_i, :] - ref_frame[particle_j, :]
    squared_distance = diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2
    if squared_correlation_length > squared_distance > 0:
        return 1
    else:
        return 0


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


def count_overlap(cur_frame, ref_frame, sq_displacement_threshold):
    """
    Given two timesteps work out the overlap using cdist
    :param cur_frame: an N by d list of particle positions where N is the number of particles and d is the number of spatial dimensions
    :param ref_frame: an N by d list of particle positions where N is the number of particles and d is the number of spatial dimensions
    :param sq_displacement_threshold: The square if the displacement threshold used for determining overlap
    :return: The number of particle overlaps
    """
    distance_matrix = cdist(cur_frame, ref_frame, metric='sqeuclidean')
    overlap_matrix = distance_matrix < sq_displacement_threshold
    num_overlaps = np.count_nonzero(overlap_matrix)
    return num_overlaps


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


def main(filename, num_spatial_dimensions, frame_cutoff, particle_diameter):

    particle_positions = coordinate_methods.read_xyz_file(filename, num_spatial_dimensions)
    distance, distance_squared = simple_chi_four(particle_positions, frame_cutoff, 0.3 * particle_diameter)

    normalised_distance = normalise_by_observations(distance)
    normalised_distancesquared = normalise_by_observations(distance_squared)

    chi_squared = normalised_distancesquared - normalised_distance ** 2

    density = get_particle_density(particle_positions, particle_diameter)
    average_particles = get_average_particles(particle_positions)

    temperature = 1
    chi_squared = chi_squared * density / average_particles * temperature

    np.savetxt(filename + "_chi4.txt", chi_squared)
    np.savetxt(filename + "_w.txt", normalised_distance)