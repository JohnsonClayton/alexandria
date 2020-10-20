from .experiment import Experiment

class Experiments:
    def __init__(self, num=0, experiments=[]):
        self.exp_list = []

        if type(num) == int:
            if num >= 0:
                for i in range(num):
                    self.exp_list.append( Experiment(name='default_experiment {}'.format(i+1)) )
            else:
                raise ValueError('Cannot create negative experiments: {}'.format( str(num) ))
        else:
            raise ValueError('num must be int type, not {}'.format( str( type( num ) ) ))

        # Check the types of all the values in the experiments argument
        if type(experiments) == list:
            for exp in experiments:
                if type(exp) != Experiment:
                    raise ValueError('Experiment provided must be Experiment type, not {}'.format( str(type(exp)) ))
            for exp in experiments:
                self.exp_list.append(exp)
        elif type(experiments) == Experiment:
            self.exp_list.append(experiments)
        else:
            raise ValueError('experiments argument must be a list of Experiment objects or a single Experiment object, not {}'.format( str(type(experiments)) ) )

    def numExperiments(self):
        return len(self.exp_list)

    def getExperimentsList(self):
        return self.exp_list.copy()

    def addExperiment(self, exp):
        if type(exp) == Experiment:
            self.exp_list.append(exp)
        else:
            raise ValueError('Experiment must be Experiment type, not {}'.format( str( type(exp) )))

    def addExperiments(self, exps):
        if type(exps) == Experiment:
            self.addExperiment( exps )
        elif type(exps) == list:
            if len(exps) > 0:
                for exp in exps:
                    if type(exp) != Experiment:
                        raise ValueError('Provided list contains non-Experiment object: {}'.format( str( type( exp ) ) ))
                for exp in exps:
                    self.exp_list.append(exp)
            else:
                raise ValueError('Provided Experiment list is empty!')
        elif type(exps) == Experiments:
            exps = exps.getExperimentsList()
            if len(exps) > 0:
                self.addExperiments( exps )
            else:
                raise ValueError('Provided Experiments object doesn\'t have Experiment objects to add!')
        else:
            raise ValueError('exps argument must be a list of Experiment objects or a single Experiment object, not {}'.format( str(type(exps)) ) )

    def hasExperiment(self, exp_name):
        if type(exp_name) == str:
            for exp in self.exp_list:
                if exp.getName() == exp_name:
                    return True
            return False
        else:
            raise ValueError('Experiment name must be string type, not {}'.format( str( type(exp_name) ) ))

    def getExperiment(self, exp_name):
        if type(exp_name) == str:
            if self.hasExperiment(exp_name):
                for exp in self.exp_list:
                    if exp.getName() == exp_name:
                        return exp
            else:
                raise ValueError('Experiment does not exist: {}'.format( exp_name ))
        else:
            raise ValueError('Experiment name must be string type, not {}'.format( str( type(exp_name) ) ))


    def removeExperiment(self, exp_name):
        if type(exp_name) == str:
            l0 = self.numExperiments()
            for exp in self.exp_list:
                if exp.getName() == exp_name:
                    self.exp_list.remove(exp)
            l1 = self.numExperiments()
            if l1 == l0:
                raise ValueError('Experiment does not exist: \'{}\''.format(exp_name))
        else:
            raise ValueError('Experiment name must be string!')

    def removeExperiments(self, exp_names):
        if type(exp_names) == list:
            # This looks really inefficient, however the errors thrown here will help out a ton!
            #   Plus, if you're removing so many experiments where THIS is the issue, you have larger problems
            for exp_name in exp_names:
                if type(exp_name) != str:
                    raise ValueError('Provided list must only contain strings!')
            for exp_name in exp_names:
                if not self.hasExperiment(exp_name):
                    raise ValueError('Experiment does not exist: \'{}\''.format( exp_name ))
            for exp_name in exp_names:
                self.removeExperiment(exp_name)
        elif type(exp_names) == str:
            self.removeExperiment(exp_names)
        else:
            raise ValueError('Argument must be list or string type, not {}'.format( str( type(exp_names) ) ) )