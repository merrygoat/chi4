import numpy as np
from tqdm import trange


def cdistmethod(data, numframes, cutoff, numberdensity, threshold):
    distance = np.zeros((numframes, 4))
    distancesquared = np.zeros((numframes, 3))
    sqthreshold = threshold ** 2

    if cutoff > numframes:
        cutoff = numframes

    for ref_frame_number in trange(cutoff):
        ref_frame = data[ref_frame_number]
        for cur_frame_number in range(ref_frame_number+1, len(data)):
            cur_frame = data[cur_frame_number]
            overlap_count = 0
            for particle in ref_frame:
                # Find the particles in cur_frame within threshold of the particles in ref_frame
                search_indices = np.searchsorted(cur_frame[:, 0], [particle[0] - threshold, particle[0] + threshold])
                # Get the squared distance between the close particles in cur_frame and the particle
                neighbour_distances = np.sum((cur_frame[search_indices[0]:search_indices[1], :] - particle) ** 2, axis=1)
                overlap_count += np.count_nonzero(neighbour_distances < sqthreshold)

            numobservations = cur_frame.shape[0] * ref_frame.shape[0]
            result = overlap_count / np.sqrt(numobservations)
            distance[cur_frame_number - ref_frame_number, 0:2] += [result, 1]
            distancesquared[cur_frame_number - ref_frame_number, 0:2] += [(result * result), 1]

    # normalise by dividing by number of observations
    with np.errstate(divide='ignore', invalid='ignore'):  # suppress divide by zero warning since value is 0 at T=0
        distance[:, 2] = distance[:, 0] / distance[:, 1]
        distancesquared[:, 2] = distancesquared[:, 0] / distancesquared[:, 1]
    # square distance
    distance[:, 3] = distance[:, 2] ** 2
    chisquared = distancesquared[:, 2] - distance[:, 3]

    # Work out total number of particles
    total_particles = 0
    for frame in data:
        total_particles += len(frame)
    average_particles = total_particles/len(data)
    # normalise chi squared
    chisquared = chisquared * numberdensity * average_particles
    return chisquared


def read_xyz_file(filename, dimensions):
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

    return particle_positions


def sort_lists(particle_lists):
    for frame in range(len(particle_lists)):
        particle_lists[frame] = particle_lists[frame][particle_lists[frame][:, 0].argsort()]
    return particle_lists


def main(filename, dimensions, threshold, cutoff, numberdensity):

    data = sort_lists(read_xyz_file(filename, dimensions))
    num_frames = len(data)

    chisquaredresults = cdistmethod(data, num_frames, cutoff, numberdensity, threshold)

    np.savetxt(filename + "_chi4.txt", chisquaredresults)

main("sample.xyz", 3, 0.3, 250, 1.1)