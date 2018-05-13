import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist, squareform
import coordinate_methods


def calculate_chi_four(particle_positions, frame_cutoff, particle_diameter):

    num_frames = len(particle_positions)
    displacement_threshold = 0.3 * particle_diameter
    sq_displacement_threshold = displacement_threshold ** 2
    distance = np.zeros((num_frames, 3))
    distancesquared = np.zeros((num_frames, 3))

    if frame_cutoff > num_frames:
        frame_cutoff = num_frames
    # the number of operations involves the (num_frames - 1)'th triangle number
    pbar = tqdm(total=int(((frame_cutoff - 1) ** 2 + (frame_cutoff - 1)) / 2 + (num_frames - frame_cutoff) * frame_cutoff))
    for ref_frame, ref_frame_number in enumerate(particle_positions):
        num_iterations = 0
        for cur_frame_number in range(ref_frame_number + 1, num_frames):
            cur_frame = particle_positions[cur_frame_number]
            overlap_count = 0
            if cur_frame_number >= num_frames - frame_cutoff:
                overlap_count += count_overlap(cur_frame, ref_frame, sq_displacement_threshold)
                num_iterations += 1
                distance[cur_frame_number - ref_frame_number, 0:2] += [overlap_count, 1]
                distancesquared[cur_frame_number - ref_frame_number, 0:2] += [(overlap_count * overlap_count), 1]
        pbar.update(num_iterations)

    normalised_distance = normalise_by_observations(distance)
    normalised_distancesquared = normalise_by_observations(distancesquared)

    chi_squared = normalised_distancesquared - normalised_distance ** 2

    density, average_particles = get_particle_density(particle_positions, particle_diameter)

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


def get_particle_density(particle_positions, particle_diameter):
    """ Get particle density by frame and take an average. Calculating by frame prevents issues due to drift.
    Get the box size from the first frame and divide by the average number of particles to get the particle density
    :param particle_positions: A list of f numpy arrays of size N by d where f is the number of frames,
     N is the number of particles and d is the number of spatial dimensions
    :param particle_diameter: The diameter of a particle in the units of the coordinates
    :return: The average density of the box througout the simulation
    """
    total_density = 0
    total_particles = 0

    for frame in particle_positions:
        frame_particles = len(frame)
        lengths = (np.amax(frame) - np.amin(frame)) / particle_diameter

        volume = 1
        for side_length in lengths:
            volume *= side_length

        total_density += volume / frame_particles
        total_particles += frame_particles

    average_density = total_density / len(particle_positions)
    average_particles = total_particles / len(particle_positions)

    return average_density, average_particles


def main(filename, num_spatial_dimensions, frame_cutoff, particle_diameter):

    particle_positions = coordinate_methods.read_xyz_file(filename, num_spatial_dimensions)
    chisquaredresults, distance = calculate_chi_four(particle_positions, frame_cutoff, particle_diameter)

    np.savetxt(filename + "_chi4.txt", chisquaredresults)
    np.savetxt(filename + "_w.txt", distance)