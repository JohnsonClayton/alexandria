import unittest

from alexandria.experiment import Experiment
from alexandria.dataset import DatasetManager
from alexandria.models import Models

def fail(who):
    who.assertTrue( False )

class TestExperiment(unittest.TestCase):
    def test_init(self):
        # Check that the initializations occur correctly
        name = 'experiment 1'
        exp = Experiment(name=name)

        self.assertEqual( exp.name, name )
        self.assertEqual( type(exp.dm), DatasetManager )
        self.assertEqual( type(exp.models), Models )

        # Fail if Experiment name is not a string
        try:
            name = None
            exp = Experiment(name=name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment \'name\' argument must be string, not {}'.format(  str( type( name ) ) ))

        try:
            name = ['name 1', 'name 2']
            exp = Experiment(name=name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment \'name\' argument must be string, not {}'.format(  str( type( name ) ) ))

        try:
            name = 512
            exp = Experiment(name=name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment \'name\' argument must be string, not {}'.format(  str( type( name ) ) ))

        # TO-DO: Add more checks here

    def test_setName(self):
        # Check that it will set the Experiment name
        old_name = 'exp 1'
        exp = Experiment(name=old_name)
        new_name = 'cooler exp 1'
        exp.setName(new_name)
        self.assertEqual( exp.getName(), new_name )

        # Fail if incorrect types are provided
        old_name = 'exp 1'
        exp = Experiment(name=old_name)
        try:
            new_name = None
            exp.setName(new_name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment \'name\' argument must be string, not {}'.format(  str( type( new_name ) ) ))
    
        self.assertEqual( exp.getName(), old_name )

        exp = Experiment(name=old_name)
        try:
            new_name = ['hot cool name!']
            exp.setName(new_name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment \'name\' argument must be string, not {}'.format(  str( type( new_name ) ) ))
    
        self.assertEqual( exp.getName(), old_name )

        exp = Experiment(name=old_name)
        try:
            new_name = 512
            exp.setName(new_name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment \'name\' argument must be string, not {}'.format(  str( type( new_name ) ) ))
    
        self.assertEqual( exp.getName(), old_name )

        exp = Experiment(name=old_name)
        try:
            new_name = {'name':'cooler name'}
            exp.setName(new_name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment \'name\' argument must be string, not {}'.format(  str( type( new_name ) ) ))
    
        self.assertEqual( exp.getName(), old_name )


if __name__ == '__main__':
    unittest.main()
