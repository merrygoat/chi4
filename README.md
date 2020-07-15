# chi4
Code to calculate chi 4 from particle trajectory data

See example_usage ipython notebook for example of usage.

input parameters are:
filename - The name of the input file. xyz and cg files accepted. See readme for more details.
num_dimensions - The number of spatial dimensions of the input data.
threshold - The chi 4 dispalcement threshold.
number_density - Used along with the number of particles (determined automatically) for normalisation of the chi4.

## Input files
Input files accepted are XYZ or croker/grier. xyz has the format:

number_of_particles _in_frame1
comment line
paticle_1_species particle_1_xposition particle_1_yposition particle_1_zposition
paticle_2_species particle_2_xposition particle_2_yposition particle_2_zposition
...

Frames follow directly on from one another in the same format. For chi4 measurements there do not have to be the same number of particles in each frame.

Croker/Grier files have the format

particle1_xposition particle1_yposition particle1_zposition framenumber particlenumber
particle2_xposition particle2_yposition particle2_zposition framenumber particlenumber
...

2D or 3D data are accepted, the number of dimensions must be specifed as a parameter. 2D data follows the same format ass abov except from the omission of the z position.

## Publication

This code is free to use under the MIT licence. If used in work leading to a publication, please cite the relevent version using the DOI:  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3946702.svg)](https://doi.org/10.5281/zenodo.3946702)
