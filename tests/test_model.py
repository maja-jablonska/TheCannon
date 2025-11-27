import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TheCannon import CannonModel, Dataset

class TestCannonModel(unittest.TestCase):
    def setUp(self):
        self.order = 2
        self.model = CannonModel(self.order, useErrors=False)
        
        # Create dummy dataset for training
        self.wl = np.linspace(4000, 5000, 10)
        self.n_pixels = len(self.wl)
        self.n_objects = 20
        self.n_labels = 2
        
        self.tr_ID = np.array([f"star{i}" for i in range(self.n_objects)])
        self.tr_flux = np.random.rand(self.n_objects, self.n_pixels) + 1.0
        self.tr_ivar = np.ones((self.n_objects, self.n_pixels)) * 100
        self.tr_label = np.random.rand(self.n_objects, self.n_labels)
        
        self.test_ID = np.array([f"test{i}" for i in range(self.n_objects)])
        self.test_flux = np.random.rand(self.n_objects, self.n_pixels) + 1.0
        self.test_ivar = np.ones((self.n_objects, self.n_pixels)) * 100
        
        self.ds = Dataset(
            self.wl, self.tr_ID, self.tr_flux, self.tr_ivar, self.tr_label,
            self.test_ID, self.test_flux, self.test_ivar
        )
        self.ds.set_label_names(['Teff', 'logg'])

    def test_initialization(self):
        self.assertEqual(self.model.order, self.order)
        self.assertFalse(self.model.useErrors)
        self.assertIsNone(self.model.coeffs)

    def test_train(self):
        # This is a functional test as well, running the training
        self.model.train(self.ds)
        self.assertIsNotNone(self.model.coeffs)
        self.assertIsNotNone(self.model.scatters)
        self.assertIsNotNone(self.model.chisqs)
        self.assertIsNotNone(self.model.pivots)
        
        # Check shapes
        # Coeffs shape: [n_pixels, n_terms]
        # Number of terms for quadratic model with 2 labels: 
        # 1 (const) + 2 (linear) + 2 (quad) + 1 (cross) = 6
        n_terms = 6 
        self.assertEqual(self.model.coeffs.shape, (self.n_pixels, n_terms))

    def test_infer_labels(self):
        self.model.train(self.ds)
        # Infer labels on the test set (which is just random noise here, but should run)
        errs = self.model.infer_labels(self.ds)
        
        self.assertIsNotNone(self.ds.test_label_vals)
        self.assertEqual(self.ds.test_label_vals.shape, (self.n_objects, self.n_labels))

if __name__ == '__main__':
    unittest.main()
