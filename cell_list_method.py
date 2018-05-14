import math
import numpy as np
from tqdm import tqdm


def get_cell_size(particle_coordinates, correlation_length, pbcs=0):
    # Given a set of particle coordinates and a correlation length, generate cells

    spatial_dimensions = 3
    max_coord = [0] * spatial_dimensions
    min_coord = [100] * spatial_dimensions
    box_size = []
    num_cells = []
    cell_size = []
    if correlation_length < 1:
        correlation_length = 1

    # Loop through all the cells to find the highest and lowest coodinates
    for frame in particle_coordinates:
        for dimension in range(spatial_dimensions):
            if max(frame[:, dimension]) > max_coord[dimension]:
                max_coord[dimension] = max(frame[:, dimension])
            if min(frame[:, dimension]) < min_coord[dimension]:
                min_coord[dimension] = min(frame[:, dimension])

    # Using the highest and lowest coordinates, generate cell numbers and sizes
    for dimension in range(spatial_dimensions):
        box_size.append(max_coord[dimension] - min_coord[dimension])
        num_cells.append(math.floor(box_size[dimension] / correlation_length))
        cell_size.append(box_size[dimension] / num_cells[dimension])

    if pbcs == 0:
        for dimension_index, dimension in enumerate(num_cells):
            num_cells[dimension_index] += 1

    return num_cells, cell_size, box_size


def get_scalar_cell_index(cell_indices, num_cells):
    x_index = cell_indices[0]
    y_index = cell_indices[1]
    z_index = cell_indices[2]
    n_cells_x = num_cells[0]
    n_cells_y = num_cells[1]
    n_cells_z = num_cells[2]

    return ((x_index + n_cells_x) % n_cells_x) * n_cells_y * n_cells_z + \
           ((y_index + n_cells_y) % n_cells_y) * n_cells_z + \
           ((z_index + n_cells_z) % n_cells_z)


def get_vector_cell_index(particle_position, cell_lengths):
    return [int(x/y) for x, y in zip(particle_position, cell_lengths)]


def setup_cell_list(particle_coordinates, cell_size, num_cells):
    """
    :param particle_coordinates: A list of f numpy arrays of size N by d where f is the number of frames,
     N is the number of particles and d is the number of spatial dimensions
    :param cell_size: a list of cell sizes, one value for each dimension
    :param num_cells: a list of integers, the total number of cells in each dimension
    :return:
    """
    heads = []
    particle_links = []
    total_cells = 1

    for dimension in num_cells:
        total_cells = total_cells * dimension

    for frame_index, frame in enumerate(particle_coordinates):
        heads.append([-1] * total_cells)
        particle_links.append([-1] * len(frame))
        for particle_index, particle in enumerate(frame):
            vector_index = get_vector_cell_index(particle, cell_size)
            scalar_index = get_scalar_cell_index(vector_index, num_cells)
            if heads[frame_index][scalar_index] == -1:
                heads[frame_index][scalar_index] = particle_index
            else:
                particle_links[frame_index][particle_index] = heads[frame_index][scalar_index]
                heads[frame_index][scalar_index] = particle_index
    return heads, particle_links


def loop_over_inner_cells(num_cells):
    """
    A generator to return the vector index of all cells sequentially
    :param num_cells: a list of integers, the total number of cells in each dimension
    """
    # A generator to return the index of the inner cell
    for x_cell in range(num_cells[0]):
        for y_cell in range(num_cells[1]):
            for z_cell in range(num_cells[2]):
                yield [x_cell, y_cell, z_cell]


def loop_over_neighbour_cells(vector_cell_index, num_cells):
    """
    A generator to return the vector index of neighbouring cells
    :param vector_cell_index: a list of integer cell indices, one for each dimension
    :param num_cells: a list of integers, the total number of cells in each dimension
    """
    for x_neighbour in range(vector_cell_index[0] - 1, vector_cell_index[0] + 2):
        for y_neighbour in range(vector_cell_index[1] - 1, vector_cell_index[1] + 2):
            for z_neighbour in range(vector_cell_index[2] - 1, vector_cell_index[2] + 2):
                neighbour_vector_index = [x_neighbour, y_neighbour, z_neighbour]
                # Correct neighbour index for boundaries
                for index, value in enumerate(neighbour_vector_index):
                    if value > num_cells[index] - 1:
                        neighbour_vector_index[index] -= num_cells[index]
                    if value < 0:
                        neighbour_vector_index[index] += num_cells[index]
                yield neighbour_vector_index


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

    num_cells, cell_size, box_size = get_cell_size(particle_positions, displacement_threshold)
    cell_heads, links = setup_cell_list(particle_positions, cell_size, num_cells)

    if frame_cutoff > num_frames:
        frame_cutoff = num_frames

    pbar = tqdm(total=int(((frame_cutoff - 1) ** 2 + (frame_cutoff - 1)) / 2 + (num_frames - frame_cutoff) * frame_cutoff))

    for cur_frame_index, cur_frame in enumerate(particle_positions):
        num_iterations = 0
        for ref_frame_index, ref_frame in enumerate(particle_positions):
            if ref_frame_index > cur_frame_index:
                overlap_count = 0
                for cell_vector_index in loop_over_inner_cells(num_cells):
                    cell_scalar_index = get_scalar_cell_index(cell_vector_index, num_cells)
                    for neighbour_vector_index in loop_over_neighbour_cells(cell_vector_index, num_cells):
                        neighbour_scalar_index = get_scalar_cell_index(neighbour_vector_index, num_cells)
                        particle_i = cell_heads[cur_frame_index][cell_scalar_index]
                        while particle_i != -1:
                            particle_j = cell_heads[ref_frame_index][neighbour_scalar_index]
                            while particle_j != -1:
                                if particle_i < particle_j:
                                    overlap_count += check_overlap_between_particles(particle_i, particle_j, cur_frame, ref_frame, sq_displacement_threshold)
                                particle_j = links[ref_frame_index][particle_j]
                            particle_i = links[cur_frame_index][particle_i]
                distance[cur_frame_index - ref_frame_index, 0:2] += [overlap_count, 1]
                distancesquared[cur_frame_index - ref_frame_index, 0:2] += [(overlap_count * overlap_count), 1]
                num_iterations += 1
        pbar.update(num_iterations)

    return distance, distancesquared


def check_overlap_between_particles(particle_i, particle_j, cur_frame, ref_frame, squared_correlation_length):
    diff = cur_frame[particle_i, :] - ref_frame[particle_j, :]
    squared_distance = diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2
    if squared_distance < squared_correlation_length:
        return 1
    else:
        return 0
