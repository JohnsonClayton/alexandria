import unittest

from alexandria.models import ModelsManager

def fail(who):
    who.assertTrue( False )

class TestModelsManager(unittest.TestCase):
    def test_init(self):
        # Test instantiations
        mm = ModelsManager()

if __name__ == '__main__':
    unittest.main()