import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist


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
    for ref_frame_number in range(num_frames):
        num_iterations = 0
        for cur_frame_number in range(ref_frame_number + 1, num_frames):
            overlap_count = 0
            if cur_frame_number >= num_frames - frame_cutoff:
                overlap_count += count_overlap(cur_frame_number, ref_frame_number, particle_positions, sq_displacement_threshold)
                num_iterations += 1
                distance[cur_frame_number - ref_frame_number, 0:2] += [overlap_count, 1]
                distancesquared[cur_frame_number - ref_frame_number, 0:2] += [(overlap_count * overlap_count), 1]
        pbar.update(num_iterations)

    normalised_results = normalization(distance, distancesquared, particle_positions, particle_diameter)
    return normalised_results


def count_overlap(cur_frame_number, ref_frame_number, sorted_particle_positions, sq_displacement_threshold):
    distance_matrix = cdist(sorted_particle_positions[cur_frame_number], sorted_particle_positions[ref_frame_number], metric='sqeuclidean')
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
    density = volume / average_particles
    temperature = 1
    # Normalise chi squared
    chisquared = chisquared * density / average_particles * temperature
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
                particle_positions[frame_number][line_number-2] = line.split()[1:]
            line_number += 1
            # If we have reached the last particle in the frame, reset counter for next frame
            if line_number == (frame_particles + 2):
                line_number = 0
                frame_number += 1

    print("XYZ read complete.")

    return particle_positions


def main(filename, num_spatial_dimensions, frame_cutoff, particle_diameter):

    particle_positions = read_xyz_file(filename, num_spatial_dimensions)
    chisquaredresults, distance = calculate_chi_four(particle_positions, frame_cutoff, particle_diameter)

    np.savetxt(filename + "_chi4.txt", chisquaredresults)
    np.savetxt(filename + "_w.txt", distance)


main("run155.xyz", 3, 2, 1)
