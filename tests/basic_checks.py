import unittest
import os
import importlib.util

class TestAutoMLAutogluon(unittest.TestCase):

    def test_library_autogluon_installed(self):
        """ Test if autogluon library is installed """
        autogluon_installed = importlib.util.find_spec("autogluon") is not None
        self.assertTrue(autogluon_installed, "autogluon library is not installed")

    def test_library_lightgbm_installed(self):
        """ Test if lightgbm library is installed """
        lightgbm_installed = importlib.util.find_spec("lightgbm") is not None
        self.assertTrue(lightgbm_installed, "lightgbm library is not installed")

    def test_library_matplotlib_installed(self):
        """ Test if matplotlib library is installed """
        matplotlib_installed = importlib.util.find_spec("matplotlib") is not None
        self.assertTrue(matplotlib_installed, "matplotlib library is not installed")
      
    def test_library_numpy_installed(self):
        """ Test if numpy library is installed """
        numpy_installed = importlib.util.find_spec("numpy") is not None
        self.assertTrue(numpy_installed, "numpy library is not installed")
      
    def test_library_python_dotenv_installed(self):
        """ Test if python-dotenv library is installed """
        python_dotenv_installed = importlib.util.find_spec("dotenv") is not None
        self.assertTrue(python_dotenv_installed, "python-dotenv library is not installed")
        
    def test_library_pytz_installed(self):
        """ Test if pytz library is installed """
        pytz_installed = importlib.util.find_spec("pytz") is not None
        self.assertTrue(pytz_installed, "pytz library is not installed")

    def test_library_scikit_installed(self):
        """ Test if scikit library is installed """
        scikit_installed = importlib.util.find_spec("sklearn") is not None
        self.assertTrue(scikit_installed, "scikit library is not installed")


    def test_library_scipy_installed(self):
        """ Test if scipy library is installed """
        scipy_installed = importlib.util.find_spec("scipy") is not None
        self.assertTrue(scipy_installed, "scipy library is not installed")

    def test_library_seaborn_installed(self):
        """ Test if seaborn library is installed """
        seaborn_installed = importlib.util.find_spec("seaborn") is not None
        self.assertTrue(seaborn_installed, "seaborn library is not installed")

    def test_library_statsmodels_installed(self):
        """ Test if statsmodels library is installed """
        statsmodels_installed = importlib.util.find_spec("statsmodels") is not None
        self.assertTrue(statsmodels_installed, "statsmodels library is not installed")
        

if __name__ == '__main__':
    unittest.main()
