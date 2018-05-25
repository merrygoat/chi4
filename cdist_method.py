from scipy.spatial.distance import cdist
import numpy as np
from tqdm import tqdm


def count_overlaps_between_frames(cur_frame, ref_frame, sq_displacement_threshold):
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


def get_all_overlaps(particle_positions, frame_cutoff, displacement_threshold):
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

    if frame_cutoff > num_frames or frame_cutoff == 0:
        frame_cutoff = num_frames
    # the number of operations involves the (num_frames - 1)'th triangle number
    pbar = tqdm(total=int(((frame_cutoff - 1) ** 2 + (frame_cutoff - 1)) / 2 + (num_frames - frame_cutoff) * frame_cutoff))
    for ref_frame_number, ref_frame in enumerate(particle_positions):
        num_iterations = 0
        for cur_frame_number, cur_frame in enumerate(particle_positions):
            if cur_frame_number > ref_frame_number:
                overlap_count = 0
                if cur_frame_number >= num_frames - frame_cutoff:
                    overlap_count += count_overlaps_between_frames(cur_frame, ref_frame, sq_displacement_threshold)
                    num_iterations += 1
                    distance[cur_frame_number - ref_frame_number, 0:2] += [overlap_count, 1]
                    distancesquared[cur_frame_number - ref_frame_number, 0:2] += [(overlap_count * overlap_count), 1]
        pbar.update(num_iterations)

    return distance, distancesquared
