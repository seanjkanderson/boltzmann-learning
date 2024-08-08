import unittest
import numpy as np

from utils import generate_probabilities_matrix


class TestUtils(unittest.TestCase):

    def test_generate_probabilities_matrix(self):
        indep_probs = np.array([[0., 0., 1.], [0., 0.25, 0.]])

        outcomes, probs = generate_probabilities_matrix(indep_probs)

        possible_combos = ['000', '100', '010', '001', '101', '011', '110', '111']

        self.assertEqual(set(possible_combos), set(outcomes))

        self.assertTrue(np.all(probs.sum(axis=1) == 1.))
        self.assertEqual(probs[0, outcomes.index('001')], 1.)
        self.assertEqual(probs[1, outcomes.index('000')], .75)
        self.assertEqual(probs[1, outcomes.index('010')], .25)

