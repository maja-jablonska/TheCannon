import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import scipy.optimize as op # For L-BFGS-B wrapper if needed
from TheCannon.train_model import _get_lvec

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Enable 64-bit precision if needed for stability
jax.config.update("jax_enable_x64", True)

def _chisq_single(labels, coeffs, flux, ivar, scatters, pivots, scales):
    """
    Chi-squared for a single star.
    labels: (nlabels,) - physical labels
    coeffs: (npixels, n_terms)
    flux: (npixels,)
    ivar: (npixels,)
    scatters: (npixels,)
    pivots: (nlabels,)
    scales: (nlabels,)
    """
    # Construct lvec for this star
    # _get_lvec expects (nstars, nlabels), so we reshape
    # But _get_lvec in train_model is vectorized.
    # We can write a single-star version or just use the vectorized one with shape (1, nlabels)
    
    # Let's implement a lightweight single-star lvec construction here to avoid overhead/shape issues inside JIT
    # Or rely on JIT to optimize it.
    
    # Normalize labels
    norm_labels = (labels - pivots) / scales
    
    # Linear terms
    linear_terms = norm_labels
    
    # Quadratic terms
    # outer product
    outer = jnp.outer(linear_terms, linear_terms)
    # Upper triangle
    triu_idx = jnp.triu_indices(len(labels))
    quadratic_terms = outer[triu_idx]
    
    # lvec: [1, linear, quadratic]
    lvec = jnp.hstack([jnp.array([1.0]), linear_terms, quadratic_terms])
    
    # Model prediction
    model_flux = jnp.dot(coeffs, lvec)
    
    # Residuals
    resid = flux - model_flux
    
    # Total variance = sigma^2 + scatter^2
    # sigma^2 = 1/ivar
    # Weight = 1 / (1/ivar + scatter^2) = ivar / (1 + ivar * scatter^2)
    
    # Handle bad pixels (ivar=0)
    # If ivar=0, weight should be 0.
    # The formula ivar / (1 + ivar * scatter^2) handles ivar=0 correctly (0 / 1 = 0)
    
    weight = ivar / (1.0 + ivar * scatters**2)
    
    chisq = jnp.sum(resid**2 * weight)
    
    return chisq

@partial(jax.jit, static_argnames=['nlabels'])
def _infer_single_star(flux, ivar, coeffs, scatters, pivots, scales, starting_guess, nlabels):
    """
    Infer labels for a single star using L-BFGS-B (via jax.scipy.optimize.minimize)
    """
    
    # Objective function
    obj_fun = lambda l: _chisq_single(l, coeffs, flux, ivar, scatters, pivots, scales)
    
    # Optimization
    # jax.scipy.optimize.minimize supports 'BFGS' but not 'L-BFGS-B' fully with bounds in all versions?
    # jax.scipy.optimize.minimize supports BFGS.
    # We assume unconstrained optimization for now (The Cannon usually is).
    
    res = jax.scipy.optimize.minimize(obj_fun, starting_guess, method='BFGS', options={'maxiter': 500})
    
    best_labels = res.x
    best_chisq = res.fun
    success = res.success
    status = res.status
    
    # Estimate covariance matrix (inverse Hessian)
    # Hessian of 0.5 * chisq is the curvature matrix.
    # Covariance ~ 2 * Hessian^-1 ?
    # If objective is chisq, then Likelihood L ~ exp(-0.5 * chisq).
    # logL ~ -0.5 * chisq.
    # Hessian(logL) = -0.5 * Hessian(chisq).
    # Covariance = -Hessian(logL)^-1 = 2 * Hessian(chisq)^-1.
    
    H = jax.hessian(obj_fun)(best_labels)
    
    # Invert Hessian
    # Add small regularization if singular?
    cov = 2.0 * jnp.linalg.inv(H)
    
    errs = jnp.sqrt(jnp.diag(cov))
    
    return best_labels, errs, best_chisq, success, status

def _infer_labels(model, dataset, starting_guess=None, batch_size=500):
    """
    JAX implementation of label inference.
    """
    print("Inferring Labels (JAX)")
    
    coeffs = model.coeffs
    scatters = model.scatters
    pivots = model.pivots
    scales = model.scales
    
    fluxes = dataset.test_flux
    ivars = dataset.test_ivar
    
    nstars = fluxes.shape[0]
    nlabels = len(pivots)
    
    if starting_guess is None:
        starting_guess = np.ones(nlabels)
        
    # Prepare batches
    labels_all = np.empty((nstars, nlabels), dtype=np.asarray(starting_guess).dtype)
    errs_all = np.empty((nstars, nlabels), dtype=np.asarray(starting_guess).dtype)
    chisqs_all = np.empty((nstars,), dtype=np.asarray(fluxes).dtype)
    
    indices = range(0, nstars, batch_size)
    if tqdm is not None:
        indices = tqdm(indices, desc="Inferring labels", total=(nstars + batch_size - 1) // batch_size)
    
    # JIT-compiled inference function for a batch
    # We map over fluxes and ivars
    
    infer_func = partial(_infer_single_star, coeffs=coeffs, scatters=scatters, pivots=pivots, scales=scales, starting_guess=jnp.array(starting_guess), nlabels=nlabels)
    
    batch_infer = jax.vmap(infer_func)
    
    for i in indices:
        batch_end = min(i + batch_size, nstars)

        batch_flux = fluxes[i:batch_end]
        batch_ivar = ivars[i:batch_end]
        
        # Handle bad pixels in input (ivar=0)
        # Original code sets flux to 1.0 where ivar=0, but weight handles it.
        # However, if flux is NaN, it propagates.
        # Let's ensure flux is finite where ivar > 0.
        # If ivar=0, flux value doesn't matter for chisq, but might for gradient if not careful.
        # Safe to set to 0 or 1.
        batch_flux = jnp.where(batch_ivar == 0, 1.0, batch_flux)
        
        b_labels, b_errs, b_chisqs, b_success, b_status = batch_infer(batch_flux, batch_ivar)

        labels_all[i:batch_end] = np.asarray(b_labels)
        errs_all[i:batch_end] = np.asarray(b_errs)
        chisqs_all[i:batch_end] = np.asarray(b_chisqs)

    # Update dataset
    dataset.set_test_label_vals(np.array(labels_all))

    return np.array(errs_all), np.array(chisqs_all)
