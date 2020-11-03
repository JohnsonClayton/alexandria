import unittest

from alexandria.models.sklearn import NaiveBayes

from sklearn.naive_bayes import BernoulliNB, CategoricalNB, ComplementNB, GaussianNB, MultinomialNB

def fail(who):
    who.assertTrue( False )

class TestSklearnNaiveBayes(unittest.TestCase):
    def test_init(self):
        # Automatic build should be a GaussianNB model
        nb = NaiveBayes()
        nb.exp_type = 'classification'
        self.assertEqual( nb.getName(), 'naive bayes.Gaussian' )
        self.assertIsInstance( nb.getBuiltModel(), GaussianNB )

        # Should be able to change flavor from constructor
        nb = NaiveBayes(flavor='bernoulli')
        nb.exp_type = 'classification'
        self.assertEqual( nb.getName(), 'naive bayes.Bernoulli' )
        self.assertIsInstance( nb.getBuiltModel(), BernoulliNB )

        nb = NaiveBayes(flavor='complement')
        nb.exp_type = 'classification'
        self.assertEqual( nb.getName(), 'naive bayes.Complement' )
        self.assertIsInstance( nb.getBuiltModel(), ComplementNB )

        nb = NaiveBayes(flavor='Multinomial')
        nb.exp_type = 'classification'
        self.assertEqual( nb.getName(), 'naive bayes.Multinomial' )
        self.assertIsInstance( nb.getBuiltModel(), MultinomialNB )

        nb = NaiveBayes(flavor='cat')
        nb.exp_type = 'classification'
        self.assertEqual( nb.getName(), 'naive bayes.Categorical' )
        self.assertIsInstance( nb.getBuiltModel(), CategoricalNB )

        # Fail out if the flavor is not a valid type
        try:
            flavor = 5
            nb = NaiveBayes(flavor=flavor)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'flavor argument must be string type, not {}'.format(type(flavor)) )

        # Fail out if the flavor is unknown
        try:
            flavor = 'chocolate fudge'
            nb = NaiveBayes(flavor=flavor)

            fail(self)
        except ValueError as ve:
            self.assertEqual( str(ve), 'unknown flavor of Naive Bayes: {}'.format(flavor)) 

if __name__ == '__main__':
    unittest.main()