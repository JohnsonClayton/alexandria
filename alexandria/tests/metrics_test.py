import unittest

from alexandria.metrics import Metric

def fail(who):
    who.assertTrue( False )

class TestMetrics(unittest.TestCase):
    def test_init(self):
        metric = Metric()
        metric
        fail(self)

if __name__ == '__main__':
    unittest.main()