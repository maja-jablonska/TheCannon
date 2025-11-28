import unittest
import numpy as np
import sys
import os
import tempfile

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from TheCannon import Dataset
try:
    from astropy.io import fits
except ImportError:
    fits = None

class TestDataset(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        self.wl = np.linspace(4000, 5000, 100)
        self.n_pixels = len(self.wl)
        self.n_objects = 10
        self.n_labels = 3
        
        self.tr_ID = np.array([f"star{i}" for i in range(self.n_objects)])
        self.tr_flux = np.ones((self.n_objects, self.n_pixels))
        self.tr_ivar = np.ones((self.n_objects, self.n_pixels))
        self.tr_label = np.zeros((self.n_objects, self.n_labels))
        
        self.test_ID = np.array([f"test{i}" for i in range(self.n_objects)])
        self.test_flux = np.ones((self.n_objects, self.n_pixels))
        self.test_ivar = np.ones((self.n_objects, self.n_pixels))
        
        self.ds = Dataset(
            self.wl, self.tr_ID, self.tr_flux, self.tr_ivar, self.tr_label,
            self.test_ID, self.test_flux, self.test_ivar
        )

    def test_initialization(self):
        self.assertEqual(len(self.ds.wl), self.n_pixels)
        self.assertEqual(self.ds.tr_flux.shape, (self.n_objects, self.n_pixels))
        self.assertEqual(self.ds.test_flux.shape, (self.n_objects, self.n_pixels))

    def test_SNR_calculation(self):
        # SNR should be roughly sqrt(flux * ivar) * flux? 
        # The code says: median(flux * ivar**0.5)
        # With flux=1, ivar=1, SNR should be 1
        self.assertTrue(np.allclose(self.ds.tr_SNR, 1.0))
        self.assertTrue(np.allclose(self.ds.test_SNR, 1.0))

    def test_label_names(self):
        names = ['Teff', 'logg', 'FeH']
        self.ds.set_label_names(names)
        self.assertEqual(self.ds.get_plotting_labels(), names)

    def test_continuum_normalize_mock(self):
        # Mocking the continuum normalization process simply by checking if methods exist
        # and run without error on dummy data.
        # Since continuum normalization is complex and involves multiprocessing, 
        # we might just test the single process version or a simple case.
        
        # For now, let's just test that we can set the continuum mask
        mask = np.ones(self.n_pixels, dtype=bool)
        self.ds.set_continuum(mask)
        self.assertTrue(np.array_equal(self.ds.contmask, mask))

    def test_chunked_snr(self):
        ds = Dataset(
            self.wl, self.tr_ID, self.tr_flux, self.tr_ivar, self.tr_label,
            self.test_ID, self.test_flux, self.test_ivar, snr_chunk_size=3
        )
        self.assertTrue(np.allclose(ds.tr_SNR, 1.0))
        self.assertTrue(np.allclose(ds.test_SNR, 1.0))

    def test_smooth_spectra_batches(self):
        batches = list(Dataset.smooth_spectra(
            self.wl, self.tr_flux, self.tr_ivar, batch_size=4))
        self.assertEqual(len(batches), int(np.ceil(self.n_objects / 4.0)))
        wl_block, flux_block, ivar_block = batches[0]
        self.assertEqual(flux_block.shape[1], self.n_pixels // 2)
        self.assertEqual(ivar_block.shape[1], self.n_pixels // 2)
        self.assertTrue(np.allclose(flux_block, 1.0))

    @unittest.skipUnless(fits is not None, "astropy is required for FITS-backed tests")
    def test_fits_backed_flux(self):
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as tmp:
            fits.PrimaryHDU(self.tr_flux).writeto(tmp.name, overwrite=True)
            tmp_path = tmp.name

        try:
            ds = Dataset(
                self.wl, self.tr_ID, tmp_path, self.tr_ivar, self.tr_label,
                self.test_ID, tmp_path, self.test_ivar
            )
            self.assertTrue(np.allclose(ds.tr_SNR, 1.0))
            self.assertEqual(ds.tr_flux.shape, self.tr_flux.shape)
        finally:
            os.remove(tmp_path)

if __name__ == '__main__':
    unittest.main()
