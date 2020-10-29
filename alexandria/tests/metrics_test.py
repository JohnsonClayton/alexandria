import unittest

from alexandria.metrics import Metrics

def fail(who):
    who.assertTrue( False )

class TestMetrics(unittest.TestCase):
    def test_init(self):
        # What would be initialized for a metric value?
        m = Metrics()
        
    def test_addPair(self):
        m = Metrics()

        pair1 = {'key1': 'value1'}
        m.addPair(pair1)
        pair2 = {'key2': 512}
        m.addPair(pair2)
        pair3 = {'key3': 4.32}
        m.addPair(pair3)
        ultimate_d = {
            'key1': 'value1',
            'key2': 512,
            'key3': 4.32
        }

        self.assertEqual( m.getPair('key1'), pair1 )
        self.assertEqual( m.getPair('key2'), pair2 )
        self.assertEqual( m.getPair('key3'), pair3 )
        self.assertEqual( m.getPairs(), ultimate_d )



if __name__ == '__main__':
    unittest.main()