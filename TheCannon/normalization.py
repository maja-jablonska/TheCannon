import jax
import jax.numpy as jnp
from functools import partial
import numpy as np # For some non-JAX operations if needed, or mostly jax.numpy

# Enable 64-bit precision if needed, though usually 32 is default in JAX
# jax.config.update("jax_enable_x64", True)

SMALL = 1.0/200

def gaussian_weight_matrix(wl, L):
    """ Matrix of Gaussian weights (JAX version) """
    # wl is 1D array of wavelengths
    # Returns matrix (N, N)
    diff = wl[:, None] - wl[None, :]
    return jnp.exp(-0.5 * (diff**2) / (L**2))

@partial(jax.jit, static_argnames=['deg'])
def _sinusoid_design_matrix(x, L, deg):
    """ Construct design matrix for sinusoid fitting """
    # p has 2*deg params.
    # func = sum_{n=0}^{deg-1} p[2n]*sin(k_n x) + p[2n+1]*cos(k_n x)
    # k_n = n * pi / L
    # We want matrix A such that A @ p = y
    # A has shape (len(x), 2*deg)
    
    N = deg # The original code uses deg as the number of modes?
    # Original:
    # N = int(len(p)/2) -> if p has 2*deg, then N=deg.
    # for n in range(0, N): ...
    
    n_vals = jnp.arange(N) # 0 to N-1
    k = n_vals * jnp.pi / L
    
    # x shape: (M,)
    # k shape: (N,)
    # k*x shape: (N, M) -> transpose to (M, N)
    kx = jnp.outer(x, k) # (M, N)
    
    sin_terms = jnp.sin(kx)
    cos_terms = jnp.cos(kx)
    
    # Interleave sin and cos terms?
    # p is [s0, c0, s1, c1, ...]
    # A should be [sin_0, cos_0, sin_1, cos_1, ...]
    
    # Stack along last axis
    # shape (M, N, 2)
    terms = jnp.stack([sin_terms, cos_terms], axis=-1)
    # Reshape to (M, 2*N)
    A = terms.reshape(len(x), 2*N)
    return A

@partial(jax.jit, static_argnames=['deg'])
def _chebyshev_design_matrix(x, deg, xmin, xmax):
    """ Construct design matrix for Chebyshev fitting """
    # Map x to [-1, 1] using provided min/max
    # Avoid divide by zero
    scale = 2.0 / (xmax - xmin + 1e-10)
    x_scaled = (x - xmin) * scale - 1.0
    
    # Implement chebvander manually as jax.numpy.polynomial is not available
    # T_0(x) = 1
    # T_1(x) = x
    # T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
    
    # x_scaled shape: (M,)
    # Output shape: (M, deg+1)
    
    M = x_scaled.shape[0]
    T = [jnp.ones(M), x_scaled]
    
    for i in range(2, deg + 1):
        next_T = 2 * x_scaled * T[-1] - T[-2]
        T.append(next_T)
        
    # If deg=0, we only need T[0]
    if deg == 0:
        return T[0][:, None]
    elif deg == 1:
        return jnp.stack(T[:2], axis=-1)
        
    return jnp.stack(T, axis=-1)

@jax.jit
def _eval_chebyshev_poly(x, p):
    """ Evaluate Chebyshev polynomial manually """
    # p coefficients c0, c1, ... c_deg
    # val = c0*T0 + c1*T1 + ...
    deg = len(p) - 1
    
    # Generate T basis
    # This duplicates work if we already have design matrix, but eval is separate
    # x shape (M,)
    M = x.shape[0]
    T = [jnp.ones(M), x]
    
    val = p[0] * T[0]
    if deg >= 1:
        val += p[1] * T[1]
        
    for i in range(2, deg + 1):
        next_T = 2 * x * T[-1] - T[-2]
        T.append(next_T)
        val += p[i] * next_T
        
    return val

@jax.jit
def _eval_chebyshev(x, p, deg, xmin, xmax):
    # Re-map x
    scale = 2.0 / (xmax - xmin + 1e-10)
    x_scaled = (x - xmin) * scale - 1.0
    return _eval_chebyshev_poly(x_scaled, p)

@jax.jit
def _solve_linear(A, y, ivar):
    """ Solve weighted linear least squares: (A.T W A) p = A.T W y """
    # W is diagonal with entries ivar
    # Equivalent to solving (sqrt(W) A) p = sqrt(W) y
    w_sqrt = jnp.sqrt(ivar)
    Aw = A * w_sqrt[:, None]
    yw = y * w_sqrt
    
    # Use lstsq
    p, residuals, rank, s = jnp.linalg.lstsq(Aw, yw, rcond=None)
    return p

@partial(jax.jit, static_argnames=['deg'])
def _eval_sinusoid(x, p, L, deg):
    A = _sinusoid_design_matrix(x, L, deg)
    return A @ p



@partial(jax.jit, static_argnames=['deg', 'ffunc'])
def _fit_continuum_single(flux, ivar, contmask, pix, deg, ffunc):
    """ Fit continuum for a single star """
    # JAX cannot handle dynamic shapes from boolean indexing inside JIT
    # We must use masking instead.
    
    # Mask weights: set ivar to 0 where contmask is False
    # This effectively removes them from the least squares fit
    yivar = jnp.where(contmask, ivar, 0.0)
    
    # Also handle zero ivar (original logic: set to SMALL**2)
    # But only for valid pixels? Original: yivar[yivar == 0] = SMALL**2
    # If we set ivar=0 for non-continuum, we don't want to set them to SMALL**2
    # So:
    # 1. Where contmask is True AND ivar is 0, set to SMALL**2
    # 2. Where contmask is False, set to 0
    
    yivar = jnp.where((yivar == 0) & contmask, SMALL**2, yivar)
    
    # Use all pixels for x and y, but weights will be 0 for non-continuum
    x = pix
    y = flux
    
    if ffunc == "sinusoid":
        # L calculation needs to be based on continuum pixels only?
        # Original: L = max(x[contmask]) - min(x[contmask])
        # We can compute min/max using mask
        # max: max(x * mask) ? No, x can be negative.
        # max(where(mask, x, -inf))
        
        x_masked_max = jnp.max(jnp.where(contmask, x, -jnp.inf))
        x_masked_min = jnp.min(jnp.where(contmask, x, jnp.inf))
        L = x_masked_max - x_masked_min
        
        A = _sinusoid_design_matrix(x, L, deg)
        p = _solve_linear(A, y, yivar)
        
        cont = _eval_sinusoid(pix, p, L, deg)
        
    elif ffunc == "chebyshev":
        x_masked_max = jnp.max(jnp.where(contmask, x, -jnp.inf))
        x_masked_min = jnp.min(jnp.where(contmask, x, jnp.inf))
        
        # Chebyshev design matrix on all pixels
        # But we need to map domain based on min/max of continuum pixels
        xmin = x_masked_min
        xmax = x_masked_max
        
        A = _chebyshev_design_matrix(x, deg, xmin, xmax)
        p = _solve_linear(A, y, yivar)
        
        cont = _eval_chebyshev(pix, p, deg, xmin, xmax)
        
    else:
        cont = jnp.zeros_like(pix, dtype=flux.dtype)
        
    return cont

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

def _find_cont_fitfunc(fluxes, ivars, contmask, deg, ffunc, n_proc=1, batch_size=500):
    """ JAX version of _find_cont_fitfunc """
    # fluxes: (nstars, npixels)
    nstars = fluxes.shape[0]
    npixels = fluxes.shape[1]
    pix = jnp.arange(npixels)
    
    # vmap over stars
    # partial application for fixed args
    fit_func = partial(_fit_continuum_single, contmask=contmask, pix=pix, deg=deg, ffunc=ffunc)
    
    # Batch processing for progress reporting and memory management
    cont_list = []
    
    # Determine batches
    indices = range(0, nstars, batch_size)
    if tqdm is not None:
        indices = tqdm(indices, desc="Fitting continuum", total=(nstars + batch_size - 1) // batch_size)
        
    for i in indices:
        batch_flux = fluxes[i:i+batch_size]
        batch_ivar = ivars[i:i+batch_size]
        
        # Run vmap on batch
        batch_cont = jax.vmap(fit_func)(batch_flux, batch_ivar)
        cont_list.append(batch_cont)
    
    # Concatenate results
    if len(cont_list) > 0:
        cont = jnp.concatenate(cont_list, axis=0)
    else:
        cont = jnp.array([])
        
    return np.array(cont) # Return numpy array for compatibility

# --- Weighted Median and Running Quantile ---

@jax.jit
def _weighted_median_single(values, weights, quantile):
    """ JAX weighted median """
    # Sort
    sindx = jnp.argsort(values)
    sorted_values = values[sindx]
    sorted_weights = weights[sindx]
    
    cvalues = jnp.cumsum(sorted_weights)
    total = cvalues[-1]
    
    # Normalize
    # Avoid divide by zero
    norm_cvalues = cvalues / jnp.where(total == 0, 1.0, total)
    
    # Find index
    idx = jnp.searchsorted(norm_cvalues, quantile, side='right')
    idx = jnp.clip(idx, 0, len(values) - 1)
    
    # If total weight is 0, return first value (matching original behavior roughly)
    return jnp.where(total == 0, values[0], sorted_values[idx])

@partial(jax.jit, static_argnames=['delta_lambda'])
def _running_quantile_single_star(wl, flux, ivar, q, delta_lambda):
    """ Running quantile for a single star """
    # Iterate over wavelengths
    # This is the tricky part to vectorise efficiently over pixels because window size varies.
    # However, we can map over the output pixels.
    
    def body_fun(i, _):
        lam = wl[i]
        # Find indices within delta_lambda
        # We can compute the mask: abs(wl - lam) < delta_lambda
        mask = jnp.abs(wl - lam) < delta_lambda
        
        # Extract values and weights
        # Since shapes must be static for JIT, we have to use the full array and mask weights
        # Masking weights to 0 effectively removes them from weighted median
        
        w_eff = ivar * mask
        val = _weighted_median_single(flux, w_eff, q)
        return val

    # Use scan or map
    # map over indices 0 to len(wl)-1
    indices = jnp.arange(len(wl))
    cont = jax.lax.map(lambda i: body_fun(i, None), indices)
    return cont

def _find_cont_running_quantile(wl, fluxes, ivars, q, delta_lambda, verbose=False, batch_size=100):
    """ JAX version of _find_cont_running_quantile """
    # vmap over stars
    # We want to map over fluxes and ivars, but keep wl, q, delta_lambda fixed.
    
    # Define a lambda to make it clear
    run_func = lambda f, i: _running_quantile_single_star(jnp.array(wl), f, i, q, delta_lambda)
    
    nstars = fluxes.shape[0]
    cont_list = []
    
    indices = range(0, nstars, batch_size)
    if verbose and tqdm is not None:
        indices = tqdm(indices, desc="Running quantile", total=(nstars + batch_size - 1) // batch_size)
        
    for i in indices:
        batch_flux = fluxes[i:i+batch_size]
        batch_ivar = ivars[i:i+batch_size]
        
        batch_cont = jax.vmap(run_func)(batch_flux, batch_ivar)
        cont_list.append(batch_cont)
        
    if len(cont_list) > 0:
        cont = jnp.concatenate(cont_list, axis=0)
    else:
        cont = jnp.array([])
        
    return np.array(cont)

# --- Wrappers for compatibility ---

def _cont_norm_running_quantile(wl, fluxes, ivars, q, delta_lambda, verbose=True):
    cont = _find_cont_running_quantile(wl, fluxes, ivars, q, delta_lambda, verbose=verbose)
    norm_fluxes = np.ones(fluxes.shape)
    bad = cont == 0
    norm_fluxes[~bad] = fluxes[~bad] / cont[~bad]
    norm_ivars = cont**2 * ivars
    return norm_fluxes, norm_ivars

def _cont_norm_running_quantile_mp(wl, fluxes, ivars, q, delta_lambda, n_proc=None, verbose=False):
    # JAX handles parallelism, ignore n_proc
    return _cont_norm_running_quantile(wl, fluxes, ivars, q, delta_lambda, verbose=verbose)

def _cont_norm_running_quantile_regions(wl, fluxes, ivars, q, delta_lambda, ranges, verbose=True):
    # Similar to original but calling JAX version
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        output = _cont_norm_running_quantile(
                wl[start:stop], fluxes[:,start:stop],
                ivars[:,start:stop], q, delta_lambda, verbose=verbose)
        norm_fluxes[:,start:stop] = output[0]
        norm_ivars[:,start:stop] = output[1]
    return norm_fluxes, norm_ivars

def _cont_norm_running_quantile_regions_mp(wl, fluxes, ivars, q, delta_lambda, ranges, n_proc=None, verbose=False):
    return _cont_norm_running_quantile_regions(wl, fluxes, ivars, q, delta_lambda, ranges, verbose=verbose)

def _find_cont_fitfunc_regions(fluxes, ivars, contmask, deg, ranges, ffunc, n_proc=1):
    cont = np.zeros(fluxes.shape)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        output = _find_cont_fitfunc(fluxes[:,start:stop],
                                    ivars[:,start:stop],
                                    contmask[start:stop],
                                    deg=deg, ffunc=ffunc,
                                    n_proc=n_proc)
        cont[:, start:stop] = output
    return cont

def _cont_norm(fluxes, ivars, cont):
    # Same as original
    norm_fluxes = np.ones(fluxes.shape)
    bad = cont == 0.
    norm_fluxes[~bad] = fluxes[~bad] / cont[~bad]
    norm_ivars = cont**2 * ivars
    return norm_fluxes, norm_ivars

def _cont_norm_regions(fluxes, ivars, cont, ranges):
    # Same as original
    nstars = fluxes.shape[0]
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        output = _cont_norm(fluxes[:,start:stop],
                           ivars[:,start:stop],
                           cont[:,start:stop])
        norm_fluxes[:,start:stop] = output[0]
        norm_ivars[:,start:stop] = output[1]
    for jj in range(nstars):
        bad = (norm_ivars[jj,:] == 0.)
        norm_fluxes[jj,:][bad] = 1.
    return norm_fluxes, norm_ivars

def _cont_norm_gaussian_smooth(dataset, L):
    # Implement if needed, but user focused on fitfunc and running quantile
    # For now, we can reuse the logic but call JAX functions if we implement _find_cont_gaussian_smooth
    pass
