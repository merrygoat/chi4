import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist


def calculate_chi_four(sorted_particle_positions, frame_cutoff, particle_diameter):

    num_frames = len(sorted_particle_positions)
    displacement_threshold = 0.3 * particle_diameter
    sq_displacement_threshold = displacement_threshold ** 2
    num_particles = len(sorted_particle_positions[0])
    n_bins = int(num_particles / 20)
    distance = np.zeros((num_frames, 3))
    distancesquared = np.zeros((num_frames, 3))

    if frame_cutoff > num_frames or frame_cutoff == 0:
        frame_cutoff = num_frames
    # the number of operations involves the (num_frames - 1)'th triangle number
    tri_frames = num_frames - 1
    pbar = tqdm(total=((tri_frames ** 2 + tri_frames) / 2) - ((((tri_frames - frame_cutoff) ** 2) + (tri_frames - frame_cutoff)) / 2))
    for ref_frame_number in range(num_frames):
        num_iterations = 0
        for cur_frame_number in range(ref_frame_number + 1, num_frames):
            overlap_count = 0
            if cur_frame_number >= num_frames - frame_cutoff:
                dist_bins = np.linspace(sorted_particle_positions[ref_frame_number].min(), sorted_particle_positions[ref_frame_number].max(), n_bins)
                for bin_num in range(n_bins - 1):
                    overlap_count += count_overlap(bin_num, cur_frame_number, dist_bins, ref_frame_number, sorted_particle_positions, sq_displacement_threshold)
                num_iterations += 1

                distance[cur_frame_number - ref_frame_number, 0:2] += [overlap_count, 1]
                distancesquared[cur_frame_number - ref_frame_number, 0:2] += [(overlap_count * overlap_count), 1]

        pbar.update(num_iterations)

    normalised_results = normalization(distance, distancesquared, sorted_particle_positions, particle_diameter)
    return normalised_results


def count_overlap(bin_num, cur_frame_number, dist_bins, ref_frame_number, sorted_particle_positions, sq_displacement_threshold):
    ref_lims = dist_bins[bin_num:bin_num + 2]
    cur_lims = [ref_lims[0] - sq_displacement_threshold, ref_lims[1] + sq_displacement_threshold]
    ref_indices = np.searchsorted(sorted_particle_positions[ref_frame_number][:, 0], ref_lims)
    cur_indices = np.searchsorted(sorted_particle_positions[cur_frame_number][:, 0], cur_lims)
    current_frame_particles = sorted_particle_positions[cur_frame_number][cur_indices[0]:cur_indices[1]]
    reference_frame_particles = sorted_particle_positions[ref_frame_number][ref_indices[0]:ref_indices[1]]
    distance_matrix = cdist(current_frame_particles, reference_frame_particles, metric='sqeuclidean')
    overlap_matrix = distance_matrix < sq_displacement_threshold
    num_overlaps = np.count_nonzero(overlap_matrix)
    return num_overlaps


def normalization(distance, distancesquared, data, particle_diameter):

    # normalise by dividing by number of observations
    with np.errstate(divide='ignore', invalid='ignore'):  # suppress divide by zero warning since value is 0 at T=0
        distance[:, 2] = distance[:, 0] / distance[:, 1]
        distancesquared[:, 2] = distancesquared[:, 0] / distancesquared[:, 1]
    # distance squared minus squared distance
    chisquared = distancesquared[:, 2] - distance[:, 2] ** 2

    # Work out total number of particles
    total_particles = 0
    for frame in data:
        total_particles += len(frame)
    average_particles = total_particles/len(data)
    num_spatial_dimensions = data[0][0].shape[0]

    # Work out box volume
    side_lengths = (data[0].max(axis=0) - data[0].min(axis=0)) / particle_diameter
    volume = 1
    for side in side_lengths:
        volume *= side
    density = average_particles / volume
    temperature = 1
    # Normalise chi squared
    chisquared = chisquared / density * average_particles * temperature
    return chisquared, distance


def read_xyz_file(filename, dimensions):

    print("Reading data from XYZ file.")

    particle_positions = []
    frame_number = 0
    line_number = 0
    with open(filename, 'r') as input_file:
        for line in input_file:
            if line_number == 0:
                # Check for blank line at end of file
                if line != "":
                    frame_particles = int(line)
                    particle_positions.append(np.zeros((frame_particles, dimensions)))
            elif line_number == 1:
                comment = line
            else:
                for x in range(dimensions):
                    particle_positions[frame_number][line_number-2][x] = line.split()[1:][x]
            line_number += 1
            # If we have reached the last particle in the frame, reset counter for next frame
            if line_number == (frame_particles + 2):
                line_number = 0
                frame_number += 1

    print("XYZ read complete.")

    return particle_positions


def sort_lists(particle_lists):
    # Sort the particles in each frame by the x coordinate
    for frame in range(len(particle_lists)):
        particle_lists[frame] = particle_lists[frame][particle_lists[frame][:, 0].argsort()]
    return particle_lists


def main(filename, num_spatial_dimensions, frame_cutoff, particle_diameter):

    particle_positions = read_xyz_file(filename, num_spatial_dimensions)
    sorted_particle_positions = sort_lists(particle_positions)
    chisquaredresults, distance = calculate_chi_four(sorted_particle_positions, frame_cutoff, particle_diameter)

    np.savetxt(filename + "_chi4.txt", chisquaredresults)
    np.savetxt(filename + "_w.txt", distance)


if __name__ == '__main__':
    filename = "F:/sample DH/Pastore/2dtracking/complete trajectories/xyztrajectory.xyz"
    num_spatial_dimensions = 2
    frame_cutoff = 0
    particle_diameter = 18

    main(filename, num_spatial_dimensions, frame_cutoff, particle_diameter)

