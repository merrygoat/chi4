import coordinate_methods
import chi4
import numpy as np
import math
import cdist_method
import cell_list_method


class TestCoordinateMethods:
    @staticmethod
    def test_xyz_read():
        # Read the sample xyz file. Check coordinates are correct.
        coordinates = coordinate_methods.read_xyz_file("sample_configurations/sample_configuration.xyz", 3)

        assert len(coordinates) == 10
        assert len(coordinates[0]) == 3375


class TestChi4:
    @staticmethod
    def test_normalise_by_observations():
        sample_data = np.array([[0, 0], [88, 159], [60, 86], [30, 59], [26, 57]])
        anticipated_result = np.array([np.nan, 88/159, 60/86, 30/59, 26/57])
        actual_result = chi4.normalise_by_observations(sample_data)
        assert np.allclose(actual_result, anticipated_result, equal_nan=True) is True

    @staticmethod
    def test_get_average_particles():
        # cubic unit cell of size [2, 2, 2]
        sample_coordinates = [np.array([[0, 0, 0], [0, 0, 2], [0, 2, 0], [0, 2, 2], [2, 0, 0], [2, 0, 2], [2, 2, 0], [2, 2, 2]]),
                              np.array([[0, 2, 0], [0, 2, 2], [2, 0, 0], [2, 0, 2], [2, 2, 0], [2, 2, 2]])]

        average_particles = chi4.get_average_particles(sample_coordinates)
        assert math.isclose(7, average_particles) is True

    @staticmethod
    def test_get_density():
        # cubic unit cell of size [2, 2, 2]
        sample_coordinates = [np.array([[0, 0, 0], [0, 0, 2], [0, 2, 0], [0, 2, 2], [2, 0, 0], [2, 0, 2], [2, 2, 0], [2, 2, 2]])]

        density = chi4.get_particle_density(sample_coordinates, 1)
        assert math.isclose(1, density) is True

    @staticmethod
    def test_simple_chi_4():
        chi4_result = chi4.measure_chi_4("sample_configurations/sample_configuration.xyz", 3, 100, 1, 0)
        print(chi4_result)

    @staticmethod
    def test_cell_list_chi_4():
        chi4_result = chi4.measure_chi_4("sample_configurations/sample_configuration.xyz", 3, 100, 1, 1)
        print(chi4_result)


class TestCdistMethod:
    @staticmethod
    def test_count_overlap():
        frame_1 = [[1, 3, 1], [1, 4, 0], [0, 3, 4], [4, 2, 5], [0, 5, 3], [3, 1, 1], [0, 4, 4], [1, 1, 3]]
        frame_2 = [[1, 1, 4], [1, 3, 4], [0, 2, 3], [0, 1, 2], [1, 1, 3], [3, 2, 3], [2, 1, 0], [2, 0, 3]]
        assert cdist_method.count_overlaps_between_frames(frame_1, frame_2, 10) == 27
        assert cdist_method.count_overlaps_between_frames(frame_1, frame_2, 5) == 9

    @staticmethod
    def test_get_all_overlaps():
        pass


class TestCellListMethod:
    # cubic unit cell of size [10, 7, 5]
    sample_coordinates = [np.array([[0, 0, 0], [0, 0, 5], [0, 7, 0], [0, 7, 5], [10, 0, 0], [10, 0, 5], [10, 7, 0], [10, 7, 5]])]

    def test_cell_size(self):
        # Given some coordinates, check the correct number of cells are generated.
        num_cells, cell_size, box_size = cell_list_method.get_cell_size(self.sample_coordinates, correlation_length=2, pbcs=0)
        assert num_cells[0] == 6
        assert num_cells[1] == 4
        assert num_cells[2] == 3
        assert math.isclose(cell_size[0], 2)
        assert math.isclose(cell_size[1], 7/3)
        assert math.isclose(cell_size[2], 2.5)
        assert box_size == [10, 7, 5]

    @staticmethod
    def test_get_all_overlaps():
        pass

    @staticmethod
    def test_get_scalar_cell_index():
        # Given cell indices and cell dimensions get a scalar cell index

        assert cell_list_method.get_scalar_cell_index([0, 0, 0], [10, 10, 10]) == 0
        assert cell_list_method.get_scalar_cell_index([0, 0, 1], [10, 10, 10]) == 1
        assert cell_list_method.get_scalar_cell_index([0, 1, 0], [10, 10, 10]) == 10
        assert cell_list_method.get_scalar_cell_index([1, 0, 0], [10, 10, 10]) == 100
        assert cell_list_method.get_scalar_cell_index([0, 0, 9], [10, 10, 10]) == 9
        assert cell_list_method.get_scalar_cell_index([0, 9, 0], [10, 10, 10]) == 90
        assert cell_list_method.get_scalar_cell_index([9, 0, 0], [10, 10, 10]) == 900
        assert cell_list_method.get_scalar_cell_index([1, 1, 1], [5, 5, 5]) == 31

    @staticmethod
    def test_get_vector_cell_index():
        # Given a particle position and the size of the cells, find the cell indices of the particle
        assert cell_list_method.get_vector_cell_index([0, 1.5, 2.5], [1, 1, 1]) == [0, 1, 2]

    @staticmethod
    def test_setup_cell_list():
        # Given some particles and some cells make the linked list and cell heads
        heads, links = cell_list_method.setup_cell_list([np.array([[0.5, 1.5, 0.5], [1.5, 1.5, 1.5], [0.5, 1.25, 0.5], [0.8, 0.3, 1.6]])], [1, 1, 1], [2, 2, 2])

        assert np.array_equal(heads, [[-1, 3, 2, -1, -1, -1, -1, 1]])
        assert np.array_equal(links, [[-1, -1, 0, -1]])

    @staticmethod
    def test_loop_over_inner_cells():
        # Given a number of cells, loop through them sequentially
        index_list = []
        for index in cell_list_method.loop_over_inner_cells([2, 2, 2]):
            index_list.append(index)
        assert index_list == [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

    @staticmethod
    def test_loop_over_neighbour_cells():
        # Given a cell to start from and the total number of cells list the neighbours of the cell
        index_list = []
        for index in cell_list_method.loop_over_neighbour_cells([1, 1, 1], [3, 3, 3]):
            index_list.append(index)
        assert index_list == [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2],
                              [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2],
                              [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]
        index_list = []
        for index in cell_list_method.loop_over_neighbour_cells([2, 2, 2], [3, 3, 3]):
            index_list.append(index)
        assert index_list == [[1, 1, 1], [1, 1, 2], [1, 1, 0], [1, 2, 1], [1, 2, 2], [1, 2, 0], [1, 0, 1], [1, 0, 2], [1, 0, 0],
                              [2, 1, 1], [2, 1, 2], [2, 1, 0], [2, 2, 1], [2, 2, 2], [2, 2, 0], [2, 0, 1], [2, 0, 2], [2, 0, 0],
                              [0, 1, 1], [0, 1, 2], [0, 1, 0], [0, 2, 1], [0, 2, 2], [0, 2, 0], [0, 0, 1], [0, 0, 2], [0, 0, 0]]

    @staticmethod
    def test_check_overlap_between_particles():
        frame_1 = np.array([[1, 1, 1], [1, 2, 3]])
        frame_2 = np.array([[1, 1, 1], [10, 10, 10]])
        assert cell_list_method.check_overlap_between_particles(particle_i=0, particle_j=0, cur_frame=frame_1,
                                                                ref_frame=frame_2, squared_correlation_length=25) == 1
        assert cell_list_method.check_overlap_between_particles(particle_i=0, particle_j=1, cur_frame=frame_1,
                                                                ref_frame=frame_2, squared_correlation_length=25) == 0
