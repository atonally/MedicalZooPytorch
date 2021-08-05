import unittest
from time import struct_time
from unittest.mock import patch, MagicMock

import medzoopytorch.utils.general


class GeneralUtilsTestCase(unittest.TestCase):

    @patch("medzoopytorch.utils.general.time.gmtime", autospec=True)
    def test_something(self,
                       gmtime_mock: MagicMock):

        expected_gmtime_result = struct_time((2021, 8, 4, 21, 6, 17, 2, 216, 0))
        gmtime_mock.return_value = expected_gmtime_result
        expected_result = "04_08___21_06"

        result = medzoopytorch.utils.general.datestr()

        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
