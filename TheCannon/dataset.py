from __future__ import (absolute_import, division, print_function)
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rc
rc('text', usetex=True)
rc('font', family='serif')
from .helpers.corner import corner
from .helpers import Table
from .find_continuum_pixels import *
try:
    from .normalization import \
        (_cont_norm_gaussian_smooth,
         _cont_norm_running_quantile,
         _cont_norm_running_quantile_regions,
         _cont_norm_running_quantile_mp,
         _cont_norm_running_quantile_regions_mp,
         _find_cont_fitfunc,
         _find_cont_fitfunc_regions,
         _cont_norm,
         _cont_norm_regions)
    _NORMALIZATION_AVAILABLE = True
except ImportError:
    _cont_norm_gaussian_smooth = None
    _cont_norm_running_quantile = None
    _cont_norm_running_quantile_regions = None
    _cont_norm_running_quantile_mp = None
    _cont_norm_running_quantile_regions_mp = None
    _find_cont_fitfunc = None
    _find_cont_fitfunc_regions = None
    _cont_norm = None
    _cont_norm_regions = None
    _NORMALIZATION_AVAILABLE = False
from .find_continuum_pixels import _find_contpix,_find_contpix_regions
from multiprocessing import cpu_count
try:
    from astropy.io import fits
    from astropy.table import Table
except ImportError:  # pragma: no cover - optional dependency
    fits = None
    Table = None


class _ArraySource(object):
    """Lightweight wrapper around array-like inputs with optional cleanup."""

    def __init__(self, data, closer=None):
        self.data = data
        self._closer = closer

    def close(self):
        if self._closer is not None:
            self._closer()

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape

n_cpus = cpu_count()

PY3 = sys.version_info[0] > 2

if PY3:
    basestring = (str, bytes)
else:
    basestring = (str, unicode)


def _require_normalization():
    if not _NORMALIZATION_AVAILABLE:
        raise ImportError(
            "Normalization utilities require optional dependencies (jax/astropy)."
        )


class Dataset(object):
    """ A class to represent Cannon input: a dataset of spectra and labels """

    def __init__(self, wl, tr_ID, tr_flux, tr_ivar, tr_label, test_ID,
                 test_flux, test_ivar, snr_chunk_size=None):
        """ Initiate a Dataset object

        Parameters
        ----------
        wl: grid of wavelength values, onto which all spectra are mapped
        tr_ID: array of IDs of training objects
        tr_flux: array of flux values for training objects, [nobj, npix]. May also
            be a tuple of (path, dataset/extension) pointing to an HDF5 or FITS
            dataset for on-demand reads, or a FITS filepath for memmapped
            reading.
        tr_ivar: array [nobj, npix] of inverse variance values for training
            objects
        tr_label: array [nobj, nlabel]
        test_ID: array [nobj[ of IDs of test objects
        test_flux: array [nobj, npix] of flux values for test objects. Same
            on-disk options as ``tr_flux`` are supported.
        test_ivar: array [nobj, npix] of inverse variance values for test
            objects
        snr_chunk_size: optional int
            Number of spectra to process at once when computing SNRs.
        """
        print("Loading dataset")
        print("This may take a while...")
        self.wl = wl
        self.tr_ID = tr_ID
        self.test_ID = test_ID
        self.tr_label = tr_label
        self._managed_sources = []

        self.tr_flux = self._load_array_source(tr_flux, 'tr_flux').data
        self.tr_ivar = self._load_array_source(tr_ivar, 'tr_ivar').data
        self.test_flux = self._load_array_source(test_flux, 'test_flux').data
        self.test_ivar = self._load_array_source(test_ivar, 'test_ivar').data
        self._label_names = None
        self.ranges = None

        # calculate SNR using vectorized masked operations
        self.tr_SNR = self._vectorized_snr(self.tr_flux, self.tr_ivar,
                                           chunk_size=snr_chunk_size)
        self.test_SNR = self._vectorized_snr(self.test_flux, self.test_ivar,
                                             chunk_size=snr_chunk_size)

    def close(self):
        """Close any managed on-disk array sources."""

        for source in self._managed_sources:
            source.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


    def _SNR(self, flux, ivar):
        """ Calculate the SNR of a spectrum, ignoring bad pixels

        Parameters
        ----------
        flux: numpy ndarray
            pixel intensities
        ivar: numpy ndarray
            inverse variances corresponding to flux

        Returns
        -------
        SNR: float
        """
        take = ivar > 0
        SNR = float(np.median(flux[take]*(ivar[take]**0.5)))
        return SNR

    def _load_array_source(self, source, name):
        """Return an array-like source, optionally backed by disk storage."""

        if isinstance(source, _ArraySource):
            self._managed_sources.append(source)
            return source

        if isinstance(source, tuple) and len(source) == 2:
            path, key = source
            if isinstance(path, basestring) and path.endswith(('.h5', '.hdf5')):
                try:
                    import h5py
                except ImportError:
                    raise ImportError(
                        "h5py is required to read HDF5 flux/ivar sources")
                handle = h5py.File(path, 'r')
                data = handle[key]
                wrapped = _ArraySource(data, closer=handle.close)
                self._managed_sources.append(wrapped)
                return wrapped
            if isinstance(path, basestring) and path.endswith(('.fits', '.fit', '.fz')):
                if fits is None:
                    raise ImportError(
                        "astropy is required to read FITS-backed sources")
                hdul = fits.open(path, memmap=True)
                data = hdul[key].data
                wrapped = _ArraySource(data, closer=hdul.close)
                self._managed_sources.append(wrapped)
                return wrapped

        if isinstance(source, basestring) and source.endswith((
                '.fits', '.fit', '.fz')):
            if fits is None:
                raise ImportError(
                    "astropy is required to read FITS-backed sources")
            hdul = fits.open(source, memmap=True)
            data = hdul[0].data
            wrapped = _ArraySource(data, closer=hdul.close)
            self._managed_sources.append(wrapped)
            return wrapped

        wrapped = _ArraySource(source)
        self._managed_sources.append(wrapped)
        return wrapped

    def _vectorized_snr(self, fluxes, ivars, chunk_size=None):
        """Vectorized SNR calculation using masked medians per spectrum."""

        n_objects = len(fluxes)
        if chunk_size is None:
            chunk_size = n_objects

        snr_values = []
        for start in range(0, n_objects, chunk_size):
            stop = min(start + chunk_size, n_objects)
            flux_chunk = fluxes[start:stop]
            ivar_chunk = ivars[start:stop]
            with np.errstate(invalid='ignore'):
                scaled = flux_chunk * np.sqrt(ivar_chunk)
            masked = np.where(ivar_chunk > 0, scaled, np.nan)
            snr_chunk = np.nanmedian(masked, axis=1)
            snr_values.append(snr_chunk)
        return np.concatenate(snr_values)


    def set_label_names(self, names):
        """ Set the label names for plotting

        Parameters
        ----------
        names: ndarray or list
            The names of the labels used for plotting, ex. in LaTeX syntax
        """
        self._label_names = names


    def get_plotting_labels(self):
        """ Return the label names used make plots 

        Returns
        -------
        label_names: ndarray
            The label names
        """
        if self._label_names is None:
            print("No label names yet!")
            return None
        else:
            return self._label_names


    @staticmethod
    def bin_flux(flux, ivar):
        """ bin two neighboring flux values """
        if np.sum(ivar)==0:
            return np.sum(flux)/2.
        return np.average(flux, weights=ivar)


    @staticmethod
    def smooth_spectrum(wl, flux, ivar):
        """ Bins down one spectrum

        Parameters
        ----------
        wl: numpy ndarray
            wavelengths
        flux: numpy ndarray
            flux values
        ivar: numpy ndarray
            inverse variances associated with fluxes

        Returns
        -------
        wl: numpy ndarray
            updated binned pixel wavelengths
        flux: numpy ndarray
            updated binned flux values
        ivar: numpy ndarray
            updated binned inverse variances
        """
        # if odd, discard the last point
        if len(wl)%2 == 1:
            wl = np.delete(wl, -1)
            flux = np.delete(flux, -1)
            ivar = np.delete(ivar, -1)
        wl = wl.reshape(-1,2)
        ivar = ivar.reshape(-1,2)
        flux = flux.reshape(-1,2)
        wl_binned = np.mean(wl, axis=1)
        ivar_binned = np.sqrt(np.sum(ivar**2, axis=1))
        flux_binned = np.array([Dataset.bin_flux(f,w) for f,w in zip(flux, ivar)])
        return wl_binned, flux_binned, ivar_binned

    @staticmethod
    def _iter_batches(values, batch_size):
        total = len(values)
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            yield values[start:stop]

    @classmethod
    def smooth_spectra(cls, wl, fluxes, ivars, batch_size=None):
        """Bin down spectra in memory-aware batches."""

        def _process_batch(batch_fluxes, batch_ivars):
            wl_blocks, flux_blocks, ivar_blocks = [], [], []
            for flux, ivar in zip(batch_fluxes, batch_ivars):
                wl_b, flux_b, ivar_b = cls.smooth_spectrum(wl, flux, ivar)
                wl_blocks.append(wl_b)
                flux_blocks.append(flux_b)
                ivar_blocks.append(ivar_b)
            return (np.array(wl_blocks), np.array(flux_blocks),
                    np.array(ivar_blocks))

        if batch_size is None:
            return _process_batch(fluxes, ivars)

        for f_batch, i_batch in zip(
                cls._iter_batches(fluxes, batch_size),
                cls._iter_batches(ivars, batch_size)):
            yield _process_batch(f_batch, i_batch)


    def smooth_dataset(self, batch_size=None):
        """ Bins down all of the spectra and updates the dataset """
        if batch_size is None:
            wl_block, tr_flux, tr_ivar = self.smooth_spectra(
                self.wl, self.tr_flux, self.tr_ivar)
            self.wl = wl_block[0]
            self.tr_flux = tr_flux
            self.tr_ivar = tr_ivar
            wl_block, test_flux, test_ivar = self.smooth_spectra(
                self.wl, self.test_flux, self.test_ivar)
            self.test_flux = test_flux
            self.test_ivar = test_ivar
        else:
            tr_flux_batches, tr_ivar_batches = [], []
            for wl_block, flux_batch, ivar_batch in self.smooth_spectra(
                    self.wl, self.tr_flux, self.tr_ivar, batch_size=batch_size):
                if not hasattr(self, '_smoothed_wl'):
                    self.wl = wl_block[0]
                    self._smoothed_wl = True
                tr_flux_batches.append(flux_batch)
                tr_ivar_batches.append(ivar_batch)
            self.tr_flux = np.vstack(tr_flux_batches)
            self.tr_ivar = np.vstack(tr_ivar_batches)

            test_flux_batches, test_ivar_batches = [], []
            for wl_block, flux_batch, ivar_batch in self.smooth_spectra(
                    self.wl, self.test_flux, self.test_ivar,
                    batch_size=batch_size):
                self.wl = wl_block[0]
                test_flux_batches.append(flux_batch)
                test_ivar_batches.append(ivar_batch)
            self.test_flux = np.vstack(test_flux_batches)
            self.test_ivar = np.vstack(test_ivar_batches)
 

    def diagnostics_SNR(self): 
        """ Plots SNR distributions of ref and test object spectra """
        print("Diagnostic for SNRs of reference and survey objects")
        fig = plt.figure()
        data = self.test_SNR
        plt.hist(data, bins=int(np.sqrt(len(data))), alpha=0.5, facecolor='r', 
                label="Survey Objects")
        data = self.tr_SNR
        plt.hist(data, bins=int(np.sqrt(len(data))), alpha=0.5, color='b',
                label="Ref Objects")
        plt.legend(loc='upper right')
        #plt.xscale('log')
        plt.title("SNR Comparison Between Reference and Survey Objects")
        #plt.xlabel("log(Formal SNR)")
        plt.xlabel("Formal SNR")
        plt.ylabel("Number of Objects")
        return fig

    
    def diagnostics_ref_labels(self):
        """ Plots all training labels against each other """
        return self._label_triangle_plot(self.tr_label)


    def _label_triangle_plot(self, label_vals):
        """Make a triangle plot for the selected labels

        Parameters
        ----------
        label_vals: numpy ndarray 
            values of the labels
        """
        labels = [r"$%s$" % l for l in self.get_plotting_labels()]
        print("Plotting every label against every other")
        fig = corner(label_vals, labels=labels, show_titles=True,
                     title_args={"fontsize":12})
        return fig


    def make_contmask(self, fluxes, ivars, frac):
        """ Identify continuum pixels using training spectra

        Does this for each region of the spectrum if dataset.ranges is not None
        
        Parameters
        ----------
        fluxes: ndarray
            Flux data values 

        ivars: ndarray
            Inverse variances corresponding to flux data values

        frac: float
            The fraction of pixels that should be identified as continuum

        Returns
        -------
        contmask: ndarray
            Mask with True indicating that the pixel is continuum
        """
        print("Finding continuum pixels...")
        if self.ranges is None:
            print("assuming continuous spectra")
            contmask = _find_contpix(self.wl, fluxes, ivars, frac)
        else:
            print("taking spectra in %s regions" %len(self.ranges))
            contmask = _find_contpix_regions(
                    self.wl, fluxes, ivars, frac, self.ranges)
        print("%s pixels returned as continuum" %sum(contmask))
        return contmask


    def set_continuum(self, contmask):
        """ Set the contmask attribute 

        Parameters
        ----------
        contmask: ndarray
            Mask with True indicating that the pixel is continuum
        """
        self.contmask = contmask


    def fit_continuum(self, deg, ffunc, n_proc=1):
        """ Fit a continuum to the continuum pixels

        Parameters
        ----------
        deg: int
            Degree of the fitting function
        ffunc: str
            Type of fitting function, 'sinusoid' or 'chebyshev'

        Returns
        -------
        tr_cont: ndarray
            Flux values corresponding to the fitted continuum of training objects
        test_cont: ndarray
            Flux values corresponding to the fitted continuum of test objects
        """
        _require_normalization()
        print("Fitting Continuum...")
        if self.ranges == None:
            tr_cont = _find_cont_fitfunc(
                self.tr_flux, self.tr_ivar, self.contmask, deg, ffunc,
                n_proc=n_proc)
            test_cont = _find_cont_fitfunc(
                self.test_flux, self.test_ivar, self.contmask, deg, ffunc,
                n_proc=n_proc)
        else:
            print("Fitting Continuum in %s Regions..." %len(self.ranges))
            tr_cont = _find_cont_fitfunc_regions(
                self.tr_flux, self.tr_ivar, self.contmask, deg, self.ranges,
                ffunc, n_proc=n_proc)
            test_cont = _find_cont_fitfunc_regions(
                self.test_flux, self.test_ivar, self.contmask, deg, self.ranges,
                ffunc, n_proc=n_proc)
        return tr_cont, test_cont


    def continuum_normalize_training_q(self, q, delta_lambda,
                                       n_proc=1, verbose=True):
        """ Continuum normalize the training set using a running quantile

        Parameters
        ----------
        q: float
            The quantile cut
        delta_lambda: float
            The width of the pixel range used to calculate the median


        Modified by:
            [08 Jun 2016] Bo Zhang (NAOC):    add multiprocessing option
        Note:
            Multiprocessing calls for more memory on computer.
        """

        _require_normalization()

        # don't use too many process
        assert n_proc >= 1 and n_proc <= 100*n_cpus
        if n_proc > n_cpus:
            print('Warning: n_proc is larger than your numbers of physical CPUs!')

        print("Continuum normalizing the tr set using running quantile...")

        if n_proc == 1:
            if self.ranges is None:
                return _cont_norm_running_quantile(
                    self.wl, self.tr_flux, self.tr_ivar,
                    q=q, delta_lambda=delta_lambda,
                    verbose=verbose)
            else:
                return _cont_norm_running_quantile_regions(
                    self.wl, self.tr_flux, self.tr_ivar,
                    q=q, delta_lambda=delta_lambda,
                    ranges=self.ranges, verbose=verbose)
        else:
            # use new version (multi process)
            print('Using %d processes ... ' % n_proc)
            if self.ranges is None:
                return _cont_norm_running_quantile_mp(
                    self.wl, self.tr_flux, self.tr_ivar,
                    q=q, delta_lambda=delta_lambda,
                    n_proc=n_proc, verbose=verbose)
            else:
                return _cont_norm_running_quantile_regions_mp(
                    self.wl, self.tr_flux, self.tr_ivar,
                    q=q, delta_lambda=delta_lambda, ranges=self.ranges,
                    n_proc=n_proc, verbose=verbose)


    def continuum_normalize(self, cont):
        """
        Continuum normalize spectra, in chunks if spectrum has regions

        Parameters
        ----------
        cont: ndarray
           Flux values corresponding to the continuum 

        Returns
        -------
        norm_tr_flux: ndarray
            Normalized flux values for the training objects
        norm_tr_ivar: ndarray
            Rescaled inverse variance values for the training objects
        norm_test_flux: ndarray
            Normalized flux values for the test objects
        norm_test_ivar: numpy ndarray
            Rescaled inverse variance values for the test objects
        """
        _require_normalization()
        tr_cont, test_cont = cont
        if self.ranges is None:
            print("assuming continuous spectra")
            norm_tr_flux, norm_tr_ivar = _cont_norm(
                    self.tr_flux, self.tr_ivar, tr_cont)
            norm_test_flux, norm_test_ivar = _cont_norm(
                    self.test_flux, self.test_ivar, test_cont)
        else:
            print("taking spectra in %s regions" %(len(self.ranges)))
            norm_tr_flux, norm_tr_ivar = _cont_norm_regions(
                    self.tr_flux, self.tr_ivar, tr_cont, self.ranges)
            norm_test_flux, norm_test_ivar = _cont_norm_regions(
                    self.test_flux, self.test_ivar, test_cont, self.ranges)
        return norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar


    def continuum_normalize_gaussian_smoothing(self, L):
        """ Continuum normalize using a Gaussian-weighted smoothed spectrum

        Parameters
        ----------
        dataset: Dataset
            the dataset to continuum normalize
        L: float
            the width of the Gaussian used for weighting
        """
        _require_normalization()
        norm_tr_flux, norm_tr_ivar, norm_test_flux, norm_test_ivar = \
                _cont_norm_gaussian_smooth(self, L)
        self.tr_flux = norm_tr_flux
        self.tr_ivar = norm_tr_ivar
        self.test_flux = norm_test_flux
        self.test_ivar = norm_test_ivar


    def diagnostics_test_step_flagstars(self):
        """ 
        Write files listing stars whose inferred labels lie outside 2 standard deviations from the reference label space 
        """
        label_names = self.get_plotting_labels()
        nlabels = len(label_names)
        reference_labels = self.tr_label
        test_labels = self.test_label_vals
        test_IDs = np.array(self.test_ID)
        mean = np.mean(reference_labels, 0)
        stdev = np.std(reference_labels, 0)
        lower = mean - 2 * stdev
        upper = mean + 2 * stdev
        for i in range(nlabels):
            label_name = label_names[i]
            test_vals = test_labels[:,i]
            warning = np.logical_or(test_vals < lower[i], test_vals > upper[i])
            filename = "flagged_stars_%s.txt" % i
            with open(filename, 'w') as output:
                for star in test_IDs[warning]:
                    output.write('{0:s}\n'.format(star))
            print("Reference label %s" % label_name)
            print("flagged %s stars beyond 2-sig of ref labels" % sum(warning))
            print("Saved list %s" % filename)


    def diagnostics_survey_labels(self):
        """ Plot all survey labels against each other """  
        fig = self._label_triangle_plot(self.test_label_vals)
   
   
    def diagnostics_1to1(self, figname="1to1_label"):
        """ Plots survey labels vs. training labels, color-coded by survey SNR """
        snr = self.test_SNR
        label_names = self.get_plotting_labels()
        nlabels = len(label_names)
        reference_labels = self.tr_label
        test_labels = self.test_label_vals

        for i in range(nlabels):
            name = label_names[i]
            orig = reference_labels[:,i]
            cannon = test_labels[:,i]
            # calculate bias and scatter
            scatter = np.round(np.std(orig-cannon),5)
            bias  = np.round(np.mean(orig-cannon),5)

            low = np.minimum(min(orig), min(cannon))
            high = np.maximum(max(orig), max(cannon))

            fig = plt.figure(figsize=(10,6))
            gs = gridspec.GridSpec(1,2,width_ratios=[2,1], wspace=0.3)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax1.plot([low, high], [low, high], 'k-', linewidth=2.0, label="x=y")
            ax1.set_xlim(low, high)
            ax1.set_ylim(low, high)
            ax1.legend(fontsize=14, loc='lower right')
            pl = ax1.scatter(orig, cannon, marker='x', c=snr,
                    vmin=50, vmax=200, alpha=0.7)
            cb = plt.colorbar(pl, ax=ax1, orientation='horizontal')
            cb.set_label('SNR from Test Set', fontsize=12)
            textstr = 'Scatter: %s \nBias: %s' %(scatter, bias)
            ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes,
                    fontsize=14, verticalalignment='top')
            ax1.tick_params(axis='x', labelsize=14)
            ax1.tick_params(axis='y', labelsize=14)
            ax1.set_xlabel("Reference Value", fontsize=14)
            ax1.set_ylabel("Cannon Test Value", fontsize=14)
            ax1.set_title("1-1 Plot of Label " + r"$%s$" % name)
            diff = cannon-orig
            npoints = len(diff)
            mu = np.mean(diff)
            sig = np.std(diff)
            #ax2.hist(diff, orientation='horizontal')
            ax2.hist(diff, range=[-3*sig,3*sig], color='k', bins=int(np.sqrt(npoints)),
                    orientation='horizontal', alpha=0.3, histtype='stepfilled')
            ax2.tick_params(axis='x', labelsize=14)
            ax2.tick_params(axis='y', labelsize=14)
            ax2.set_xlabel("Count", fontsize=14)
            ax2.set_ylabel("Difference", fontsize=14)
            ax2.axhline(y=0, c='k', lw=3, label='Difference=0')
            ax2.set_title("Training Versus Test Labels for $%s$" %name,
                    fontsize=14)
            ax2.legend(fontsize=14)
            figname_full = "%s_%s.png" %(figname, i)
            plt.savefig(figname_full)
            print("Diagnostic for label output vs. input")
            print("Saved fig %s" % figname_full)
            plt.close()


    def set_test_label_vals(self, vals):
        """ Set test label values  

        Parameters
        ----------
        vals: ndarray
            Test label values
        """
        self.test_label_vals = vals

    
    def diagnostics_best_fit_spectra(self, model):
        """ Plot results of best-fit spectra for ten random test objects """
        overlay_spectra(model, self)


    def write_to_fits(self, filepath, **kwargs):
        if fits is None:
            raise ImportError("astropy is required to write FITS outputs")
        hl = convert_to_hdulist(self)
        print('Writing to fits [%s] ...' % filepath)
        hl.writeto(filepath, **kwargs)


def convert_to_hdulist(ds):
    """ transform TheCannon dataset into fits HDU list """

    if fits is None:
        raise ImportError("astropy is required to write FITS outputs")

    # initialize data list
    data_list = [ds.wl,
                 ds.ranges,
                 ds.contmask*1,
                 ds.tr_ID,
                 ds.tr_SNR,
                 ds.tr_flux,
                 ds.tr_ivar,
                 ds.tr_label,
                 ds.test_ID,
                 ds.test_SNR,
                 ds.test_flux,
                 ds.test_ivar]
    name_list = ['wl',
                 'ranges',
                 'contmask',
                 'tr_ID',
                 'tr_SNR',
                 'tr_flux',
                 'tr_ivar',
                 'tr_label',
                 'test_ID',
                 'test_SNR',
                 'test_flux',
                 'test_ivar']
    hdu_format_list_rw = ['image',
                          'image',
                          'image',
                          'table',
                          'image',
                          'image',
                          'image',
                          'image',
                          'table',
                          'image',
                          'image',
                          'image']

    # construct Primary header
    header = fits.Header()
    header['author'] = 'Bo Zhang (@NAOC)'
    header['version'] = 'v1.0'
    header['last_modified'] = 'Sat 11 Jun 2016'
    header['data'] = 'TheCannon dataset object'

    # initialize HDU list
    hl = [fits.hdu.PrimaryHDU(header=header)]

    # construct HDU list
    assert len(data_list) == len(name_list)
    n_hdus = len(data_list)
    for i in range(n_hdus):
        if hdu_format_list_rw[i] == 'table':
            data = Table(data=data_list[i].reshape((-1, 1)),
                         names=[name_list[i]])
            hl.append(fits.BinTableHDU(data.as_array()))
        else:
            hl.append(fits.ImageHDU(data=np.array(data_list[i]),
                                    name=name_list[i]))

    return fits.HDUList(hl)
