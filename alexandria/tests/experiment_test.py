import unittest

from alexandria.experiment import Experiment

def fail(who):
    who.assertTrue( False )

class TestExperiment(unittest.TestCase):
    def test_init(self):
        exp = Experiment(name='experiment 1')
        

if __name__ == '__main__':
    unittest.main()
