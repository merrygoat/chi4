import coordinate_methods
import chi4
import numpy as np


class TestCoordinateMethods:
    @staticmethod
    def test_xyz_read():
        # Read the sample xyz file. Check coordinates are correct.
        coordinates = coordinate_methods.read_xyz_file("sample_configurations/sample_configuration.xyz", 3)

        assert len(coordinates) == 10
        assert len(coordinates[0]) == 3375


class TestChi4Methods:
    @staticmethod
    def test_count_overlap():
        frame_1 = [[1, 3, 1], [1, 4, 0], [0, 3, 4], [4, 2, 5], [0, 5, 3], [3, 1, 1], [0, 4, 4], [1, 1, 3]]
        frame_2 = [[1, 1, 4], [1, 3, 4], [0, 2, 3], [0, 1, 2], [1, 1, 3], [3, 2, 3], [2, 1, 0], [2, 0, 3]]
        assert chi4.count_overlap(frame_1, frame_2, 10) == 27
        assert chi4.count_overlap(frame_1, frame_2, 5) == 9

    @staticmethod
    def test_normalise_by_observations():
        sample_data = np.array([[0, 0], [88, 159], [60, 86], [30, 59], [26, 57]])
        anticipated_result = np.array([np.nan, 88/159, 60/86, 30/59, 26/57])
        actual_result = chi4.normalise_by_observations(sample_data)
        assert np.allclose(actual_result, anticipated_result, equal_nan=True) is True

