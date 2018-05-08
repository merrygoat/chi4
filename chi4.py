import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist


def calculate_chi_four(sorted_particle_positions, frame_cutoff, particle_diameter):

    num_frames = len(sorted_particle_positions)
    displacement_threshold = 0.3 * particle_diameter
    sq_displacement_threshold = displacement_threshold ** 2
    num_particles = len(sorted_particle_positions[0])
    n_bins = int(num_particles / 50)
    distance = np.zeros((num_frames, 4))
    distancesquared = np.zeros((num_frames, 3))

    if frame_cutoff > num_frames:
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

                numobservations = sorted_particle_positions[cur_frame_number].shape[0] * sorted_particle_positions[ref_frame_number].shape[0]
                result = overlap_count / np.sqrt(numobservations)
                distance[cur_frame_number - ref_frame_number, 0:2] += [result, 1]
                distancesquared[cur_frame_number - ref_frame_number, 0:2] += [(result * result), 1]

        pbar.update(num_iterations)

    return normalization(distance, distancesquared, sorted_particle_positions, particle_diameter)


def count_overlap(bin_num, cur_frame_number, dist_bins, ref_frame_number, sorted_particle_positions, sq_displacement_threshold):
    ref_lims = dist_bins[bin_num:bin_num + 2]
    cur_lims = [ref_lims[0] - sq_displacement_threshold, ref_lims[1] + sq_displacement_threshold]
    ref_indices = np.searchsorted(sorted_particle_positions[ref_frame_number][:, 0], ref_lims)
    cur_indices = np.searchsorted(sorted_particle_positions[cur_frame_number][:, 0], cur_lims)
    distance_matrix = cdist(sorted_particle_positions[cur_frame_number][cur_indices[0]:cur_indices[1]],
                            sorted_particle_positions[ref_frame_number][ref_indices[0]:ref_indices[1]],
                            metric='sqeuclidean')
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

    # Work out box volume
    x_len = (data[0][:, 0].max() - data[0][:, 0].min()) / particle_diameter
    y_len = (data[0][:, 1].max() - data[0][:, 1].min()) / particle_diameter
    z_len = (data[0][:, 2].max() - data[0][:, 2].min()) / particle_diameter
    volume = x_len * y_len * z_len
    # Normalise chi squared
    chisquared = chisquared * average_particles * average_particles / volume
    return chisquared


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
                particle_positions[frame_number][line_number-2] = line.split()[1:]
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
    chisquaredresults = calculate_chi_four(sorted_particle_positions, frame_cutoff, particle_diameter)

    np.savetxt(filename + "_chi4.txt", chisquaredresults)


main("F:/sample DH/Paddy/10_01_06-vf0.56/i/paddy_track/full_trajectory_no_drift.xyz", 3, 10, 10.75)

