import unittest

from alexandria.experiment import Experiments, Experiment

def fail(who):
    who.assertTrue( False )

class TestExperiments(unittest.TestCase):
    def test_init(self):
        # Check the constructor will create x number of experiments
        num = 5
        exps = Experiments(num=num)
        experiment_count = exps.numExperiments()
        self.assertEqual( experiment_count, num )

        exps_list = exps.getExperimentsList()
        self.assertEqual( exps_list[0].getName(), 'default_experiment 1' )
        self.assertEqual( exps_list[1].getName(), 'default_experiment 2' )
        self.assertEqual( exps_list[2].getName(), 'default_experiment 3' )
        self.assertEqual( exps_list[3].getName(), 'default_experiment 4' )
        self.assertEqual( exps_list[4].getName(), 'default_experiment 5' )


        # Num cannot be any other data type other than int
        try:
            num = dict()
            exps = Experiments(num=num)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'num must be int type, not {}'.format( str( type( num ) ) ))

        # Check for error where number of experiments is zero or negative
        try:
            num = -1
            exps = Experiments(num=num)
            experiment_count = exps.numExperiments()

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Cannot create negative experiments: {}'.format(num) )

        # Check that provided experiments are added to the Experiments object through constructor correctly
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')

        exps = Experiments(experiments=[exp1, exp2, exp3])

        experiment_count = exps.numExperiments()
        experiment_list = exps.getExperimentsList()
        self.assertEqual( experiment_count, 3 )
        self.assertEqual( experiment_list[0].name, 'experiment 1' )
        self.assertEqual( experiment_list[1].name, 'experiment 2' )
        self.assertEqual( experiment_list[2].name, 'experiment 3' )

        # Check to make sure that they are added in the order that is expected
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')

        exps = Experiments(experiments=[exp3, exp1, exp2])

        experiment_count = exps.numExperiments()
        experiment_list = exps.getExperimentsList()
        self.assertEqual( experiment_count, 3 )
        self.assertEqual( experiment_list[0].name, 'experiment 3' )
        self.assertEqual( experiment_list[1].name, 'experiment 1' )
        self.assertEqual( experiment_list[2].name, 'experiment 2' )

        # Throw an error if object is not an Experiment object
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')
        exp4 = 50

        try:
            exps = Experiments(experiments=[exp1, exp2, exp3, exp4])

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment provided must be Experiment type, not {}'.format( str(type(exp4)) ) )
        
        exp1 = Experiment(name='experiment 1')
        exp2 = {'experiment 2': Experiment(name='experiment 2')}
        exp3 = Experiment(name='experiment 3')
        exp4 = Experiment(name='experiment 4')

        try:
            exps = Experiments(experiments=[exp1, exp2, exp3, exp4])

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment provided must be Experiment type, not {}'.format( str(type(exp2)) ) )

        # Can pass empty list of experiments
        exps = Experiments(experiments=[])
        experiment_count = exps.numExperiments()

        self.assertEqual( experiment_count, 0 )

        #Can pass single Experiment object
        exp1 = Experiment(name='experiment 1')
        exps = Experiments(experiments=exp1)
        experiment_list = exps.getExperimentsList()

        self.assertEqual( len(experiment_list), 1 )
        self.assertEqual( experiment_list[0], exp1 )

        # Cannot pass non-list or non-Experiment objects as experiments
        try:
            experiments = {'test': 15, 'experiments': 20}
            exps = Experiments(experiments=experiments)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'experiments argument must be a list of Experiment objects or a single Experiment object, not {}'.format( str(type(experiments)) ) )


    def test_addExperiment(self):
        # Check that it adds an experiment object
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')
        exp4 = Experiment(name='experiment 4')

        experiments = Experiments(experiments=[exp1, exp2, exp3, exp4])
        
        actual = experiments.numExperiments()
        expected = 4
        self.assertEqual( actual, expected )

        exp5 = Experiment(name='experiment 5')
        experiments.addExperiment(exp5)

        actual = experiments.numExperiments()
        expected = 5
        self.assertEqual( actual, expected )

        experiments_list = experiments.getExperimentsList()
        last_exp = experiments_list[-1]

        self.assertEqual( last_exp, exp5 )

        exp6 = Experiment(name='experiment 6')
        experiments.addExperiment(exp6)

        actual = experiments.numExperiments()
        expected = 6
        self.assertEqual( actual, expected )

        experiments_list = experiments.getExperimentsList()
        last_exp = experiments_list[-1]

        self.assertEqual( last_exp, exp6 )

        # Make sure we cannot add non-Experiment objects to the object
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')
        exp4 = Experiment(name='experiment 4')

        experiments = Experiments(experiments=[exp1, exp2, exp3, exp4])

        exp5 = [Experiment(name='experiment 5')]
        try:
            experiments.addExperiment(exp5)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment must be Experiment type, not {}'.format( str( type(exp5) ) ) )

        expected = 4
        actual = experiments.numExperiments()
        self.assertEqual( actual, expected )

        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')
        exp4 = Experiment(name='experiment 4')

        experiments = Experiments(experiments=[exp1, exp2, exp3, exp4])

        exp5 = 150000000
        try:
            experiments.addExperiment(exp5)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment must be Experiment type, not {}'.format( str( type(exp5) ) ) )

        expected = 4
        actual = experiments.numExperiments()
        self.assertEqual( actual, expected )

        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')
        exp4 = Experiment(name='experiment 4')

        experiments = Experiments(experiments=[exp1, exp2, exp3, exp4])

        exp5 = None
        try:
            experiments.addExperiment(exp5)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment must be Experiment type, not {}'.format( str( type(exp5) ) ) )

        expected = 4
        actual = experiments.numExperiments()
        self.assertEqual( actual, expected )


    def test_addExperiments(self):
        # Check that it adds lists of experiment objects
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')
        exp4 = Experiment(name='experiment 4')

        experiments = Experiments(experiments=[exp1, exp2, exp3, exp4])
        
        actual = experiments.numExperiments()
        expected = 4
        self.assertEqual( actual, expected )

        exp5 = Experiment(name='experiment 5')
        exp6 = Experiment(name='experiment 6')
        exp7 = Experiment(name='experiment 7')
        experiments.addExperiments( [exp5, exp6, exp7] )

        actual = experiments.numExperiments()
        expected = 7
        self.assertEqual( actual, expected )

        # Check that it will add all of the experiments from another Experiments object
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')
        exp4 = Experiment(name='experiment 4')

        experiments1 = Experiments(experiments=[exp1, exp2, exp3, exp4])
        
        actual = experiments1.numExperiments()
        expected = 4
        self.assertEqual( actual, expected )

        exp5 = Experiment(name='experiment 5')
        exp6 = Experiment(name='experiment 6')
        exp7 = Experiment(name='experiment 7')
        experiments2 = Experiments(experiments=[exp5, exp6, exp7])

        experiments1.addExperiments( experiments2 )

        actual = experiments1.numExperiments()
        expected = 7
        self.assertEqual( actual, expected )

        experiments_list = experiments1.getExperimentsList()
        last_three_exps = experiments_list[-3:]

        self.assertEqual( last_three_exps[0], exp5 )
        self.assertEqual( last_three_exps[1], exp6 )
        self.assertEqual( last_three_exps[2], exp7 )

        # Fails if provided Experiments object has no Experiment objects
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')
        exp4 = Experiment(name='experiment 4')

        experiments1 = Experiments(experiments=[exp1, exp2, exp3, exp4])
        
        actual = experiments1.numExperiments()
        expected = 4
        self.assertEqual( actual, expected )

        experiments2 = Experiments(experiments=[])

        try:
            experiments1.addExperiments( experiments2 )

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Provided Experiments object doesn\'t have Experiment objects to add!' )

        # Cannot add empty list 
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')
        exp4 = Experiment(name='experiment 4')

        experiments = Experiments(experiments=[exp1, exp2, exp3, exp4])

        try:
            new_experiments = []
            experiments.addExperiments(new_experiments)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Provided Experiment list is empty!' )

        actual = experiments.numExperiments()
        expected = 4
        self.assertEqual( actual, expected )

        # Cannot add other data types
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')
        exp4 = Experiment(name='experiment 4')

        experiments = Experiments(experiments=[exp1, exp2, exp3, exp4])

        try:
            new_experiments = [5000, Experiment(name='experiment 5')]
            experiments.addExperiments(new_experiments)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Provided list contains non-Experiment object: {}'.format( str( type( new_experiments[0] ) ) ) )

        actual = experiments.numExperiments()
        expected = 4
        self.assertEqual( actual, expected )

        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')
        exp4 = Experiment(name='experiment 4')

        experiments = Experiments(experiments=[exp1, exp2, exp3, exp4])

        try:
            new_experiments = [Experiment(name='experiment 5'), [Experiment(name='experiment 6')], 1200, dict()]
            experiments.addExperiments(new_experiments)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Provided list contains non-Experiment object: {}'.format( str( type( new_experiments[1] ) ) ) )

        actual = experiments.numExperiments()
        expected = 4
        self.assertEqual( actual, expected )

        # Fails if nothing is added
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')
        exp4 = Experiment(name='experiment 4')

        experiments = Experiments(experiments=[exp1, exp2, exp3, exp4])

        try:
            new_experiments = None
            experiments.addExperiments(new_experiments)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'exps argument must be a list of Experiment objects or a single Experiment object, not {}'.format( str(type(new_experiments)) ) )

        actual = experiments.numExperiments()
        expected = 4
        self.assertEqual( actual, expected )

    def test_removeExperiment(self):
        # Test that it will remove a valid test
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')

        exps = Experiments(experiments=[exp1, exp2, exp3])

        expected = 3
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        exps.removeExperiment('experiment 2')

        expected = 2
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        exp_list = exps.getExperimentsList()
        self.assertEqual( exp_list[0], exp1 )
        self.assertEqual( exp_list[1], exp3 )

        # Fail if the name is not a string
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')

        exps = Experiments(experiments=[exp1, exp2, exp3])

        expected = 3
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        try:
            exp_name = 54
            exps.removeExperiment(exp_name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment name must be string!' )

        expected = 3
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')

        exps = Experiments(experiments=[exp1, exp2, exp3])

        expected = 3
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        try:
            exp_name = None
            exps.removeExperiment(exp_name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment name must be string!' )

        expected = 3
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        # Fail if the name cannot be found
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')

        exps = Experiments(experiments=[exp1, exp2, exp3])

        expected = 3
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        try:
            exp_name = 'nonexistent experiment'
            exps.removeExperiment(exp_name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment does not exist: \'{}\''.format(exp_name) )

        expected = 3
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

    def test_removeExperiments(self):
        exp1 = Experiment(name='exp 1')
        exp2 = Experiment(name='exp 2')
        exp3 = Experiment(name='exp 3')
        exp4 = Experiment(name='exp 4')

        exps = Experiments(experiments=[exp1, exp2, exp3, exp4])

        expected = 4
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        exps.removeExperiments(['exp 1', 'exp 2'])
        
        expected = 2
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        exp_list = exps.getExperimentsList()
        self.assertEqual( exp_list[0], exp3 )
        self.assertEqual( exp_list[1], exp4 )

        # Allow for sending only one string not in a list
        exp1 = Experiment(name='exp 1')
        exp2 = Experiment(name='exp 2')
        exp3 = Experiment(name='exp 3')
        exp4 = Experiment(name='exp 4')

        exps = Experiments(experiments=[exp1, exp2, exp3, exp4])

        exps.removeExperiments('exp 2')

        expected = 3
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        exp_list = exps.getExperimentsList()
        self.assertEqual( exp_list[0], exp1 )
        self.assertEqual( exp_list[1], exp3 )
        self.assertEqual( exp_list[2], exp4 )

        # Fail if the name cannot be found
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='experiment 2')
        exp3 = Experiment(name='experiment 3')

        exps = Experiments(experiments=[exp1, exp2, exp3])

        expected = 3
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        try:
            exp_name = 'nonexistent experiment'
            exps.removeExperiments(exp_name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment does not exist: \'{}\''.format(exp_name) )

        expected = 3
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        # Fail if an experiment cannot be found in the Experiments object
        exp1 = Experiment(name='exp 1')
        exp2 = Experiment(name='exp 2')
        exp3 = Experiment(name='exp 3')
        exp4 = Experiment(name='exp 4')

        exps = Experiments(experiments=[exp1, exp2, exp3, exp4])

        expected = 4
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        try:
            exps_to_remove = ['exp 1', 'exp 20']
            exps.removeExperiments(exps_to_remove)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment does not exist: \'{}\''.format(exps_to_remove[1]) )

        expected = 4
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        # Fail if a list of strings is not provided (or a single string)
        exp1 = Experiment(name='exp 1')
        exp2 = Experiment(name='exp 2')
        exp3 = Experiment(name='exp 3')
        exp4 = Experiment(name='exp 4')

        exps = Experiments(experiments=[exp1, exp2, exp3, exp4])

        expected = 4
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        try:
            remove_me = ['test str', 20, dict()]
            exps.removeExperiments(remove_me)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Provided list must only contain strings!' )

        expected = 4
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

        try:
            remove_me = {'name': 'exp 1'}
            exps.removeExperiments(remove_me)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Argument must be list or string type, not {}'.format( str( type(remove_me) ) ) )

        expected = 4
        actual = exps.numExperiments()
        self.assertEqual( actual, expected )

    def test_getExperiment(self):
        # Check normal functionality
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='exp 2')
        exp3 = Experiment(name='experiment three')

        exps = Experiments(experiments=[exp1, exp2, exp3])
        self.assertEqual( exps.getExperiment('experiment 1'), exp1 )
        self.assertEqual( exps.getExperiment('exp 2'), exp2 )
        self.assertEqual( exps.getExperiment('experiment three'), exp3 )

        # Cannot get if the name is not a string 
        try:
            name = None
            exps.getExperiment(name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment name must be string type, not {}'.format( str( type(name) ) ) )

        try:
            name = ['name number 1', 'name number 2']
            exps.getExperiment(name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment name must be string type, not {}'.format( str( type(name) ) ) )

        # Cannot request name that is not present in the Experiments object
        try:
            name = 'name number 1'
            exps.getExperiment(name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment does not exist: {}'.format( name ))

    def test_hasExperiment(self):
        exp1 = Experiment(name='experiment 1')
        exp2 = Experiment(name='exp 2')
        exp3 = Experiment(name='experiment three')

        exps = Experiments(experiments=[exp1, exp2, exp3])

        self.assertTrue( exps.hasExperiment('experiment 1') )
        self.assertTrue( exps.hasExperiment('exp 2') )
        self.assertTrue( exps.hasExperiment('experiment three') )
        self.assertFalse( exps.hasExperiment('test experiment') )
        self.assertFalse( exps.hasExperiment('dummy test') )

        # Cannot pass any other data type than string
        try:
            name = 54
            exps.hasExperiment(name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment name must be string type, not {}'.format( str( type(name) ) ) )

        try:
            name = {'experiment_name': True}
            exps.hasExperiment(name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment name must be string type, not {}'.format( str( type(name) ) ) )

        try:
            name = None
            exps.hasExperiment(name)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'Experiment name must be string type, not {}'.format( str( type(name) ) ) )


if __name__ == '__main__':
    unittest.main()
