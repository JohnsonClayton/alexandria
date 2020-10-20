import unittest

from alexandria.models import SklearnModel

def fail(who):
    who.assertTrue( False )

class TestSklearnModel(unittest.TestCase):
    def test_init(self):
        model = SklearnModel()
        model
        fail(self)

if __name__ == '__main__':
    unittest.main()