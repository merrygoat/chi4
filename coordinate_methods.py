import numpy as np


def read_xyz_file(filename, num_spatial_dimensions):
    """
    A simple XYZ file reader. Assumes file is well formed. Not failsafe for ill formed files.
    :param filename: The name of the xyz file to read
    :param num_spatial_dimensions: The number of spatial dimensions in the xyz file
    :return: A list of length A of B by dimensions numpy arrays where A is the number of frames
     and B is the number of particles in each frame.
    """
    print("Reading data from XYZ file.")

    particle_positions = []
    frame_number = 0
    line_number = 0
    frame_particles = 0
    with open(filename, 'r') as input_file:
        for line in input_file:
            if line_number == 0:
                # Check for blank line at end of file
                if line != "":
                    frame_particles = int(line)
                    particle_positions.append(np.zeros((frame_particles, num_spatial_dimensions)))
            elif line_number == 1:
                pass
            else:
                for dimension in range(num_spatial_dimensions):
                    particle_positions[frame_number][line_number-2][dimension] = line.split()[1:][dimension]
            line_number += 1
            # If we have reached the last particle in the frame, reset counter for next frame
            if line_number == (frame_particles + 2):
                line_number = 0
                frame_number += 1

    print("XYZ read complete.")

    return particle_positions
