import unittest

from alexandria.metrics import Metrics

def fail(who):
    who.assertTrue( False )

class TestMetrics(unittest.TestCase):
    def test_init(self):
        # What would be initialized for a metric value?
        m = Metrics()
        self.assertEqual( m.metrics, {} )
        
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

        # Key for the pair cannot be anything but string
        m = Metrics()
        try:
            p1 = {3.45: 512}
            m.addPair(p1)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'key value must be string type, not {}'.format( str( type( list(p1.keys())[0] ) ) ) )

        m = Metrics()
        try:
            p1 = {512: 4.23}
            m.addPair(p1)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'key value must be string type, not {}'.format( str( type(list(p1.keys())[0]) ) ) )

        # Key does not exist in object failure
        m = Metrics()
        m.addPair(pair1)
        m.addPair(pair2)
        m.addPair(pair3)
        try:
            key4 = 'key4'
            pair4 = m.getPair(key4)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'key value \'{}\' not found'.format( str( type(key4) ) ) )


        # Value for the pair must be string, integer, or real/float types
        m = Metrics()
        try:
            p1 = {'key': [512]}
            m.addPair(p1)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'value must be string, integer, float, or dict type, not {}'.format( str( type(list(p1.values())[0]) ) ) )

        # Fail if provided dictionary with more than one value
        m = Metrics()
        try:
            p1 = {
                'key1': 'value1',
                'key2': 'value2'
            }
            m.addPair(p1)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'provided pair contains more than one pairing, please use \'addPairs\' method instead.')

        # Fail if provided empty dictionary
        m = Metrics()
        try:
            p1 = {}
            m.addPair(p1)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'provided pair is an empty dictionary!')

    def test_addPairs(self):
        # Check for basic functionality
        m = Metrics()

        pairs = {
            'key1': 'value1',
            'key2': 512,
            'key3': 4.32
        }

        m.addPairs(pairs)
        self.assertEqual( m.getPairs(), pairs )

        # If we don't provide a dictionary, then error
        m = Metrics()

        try:
            pairs = [{
                'key1': 'value1',
                'key2': 512,
                'key3': 4.32
            }]

            m.addPairs(pairs)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'provided pairs must be dictionary type, not {}'.format( str(type(pairs)) ) )



if __name__ == '__main__':
    unittest.main()