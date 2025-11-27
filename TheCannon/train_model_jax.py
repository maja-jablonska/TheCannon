import jax
import jax.numpy as jnp
from functools import partial
import numpy as np
import scipy.optimize as op # For L-BFGS-B if jax.scipy.optimize is insufficient or for compatibility
import jax.scipy.optimize

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Enable 64-bit precision if needed
# jax.config.update("jax_enable_x64", True)

def _get_lvec(label_vals, pivots, scales, derivs=False):
    """
    Constructs a label vector for an arbitrary number of labels
    Assumes that our model is quadratic in the labels
    """
    # JAX version
    # label_vals: (nstars, nlabels)
    nstars, nlabels = label_vals.shape
    
    # Normalize labels
    # pivots: (nlabels,)
    # scales: (nlabels,)
    linear_offsets = (label_vals - pivots[None, :]) / scales[None, :]
    
    # Quadratic offsets
    # We need to compute outer product for each star and take upper triangle
    # linear_offsets shape: (nstars, nlabels)
    
    # vmap outer product
    outer_products = jax.vmap(lambda x: jnp.outer(x, x))(linear_offsets) # (nstars, nlabels, nlabels)
    
    # Extract upper triangle indices
    triu_indices = jnp.triu_indices(nlabels)
    
    # vmap extraction? Or just indexing
    # outer_products[:, triu_indices[0], triu_indices[1]] works in numpy, should work in JAX
    quadratic_offsets = outer_products[:, triu_indices[0], triu_indices[1]]
    
    ones = jnp.ones((nstars, 1))
    
    lvec = jnp.hstack([ones, linear_offsets, quadratic_offsets])
    
    if derivs:
        # If derivatives are requested, we can compute them.
        # But for JAX optimization, we usually don't need them explicitly if we use AD.
        # However, to maintain API compatibility or for the manual derivative check in original code:
        pass 
        # For now, let's skip derivs implementation unless strictly needed by the caller.
        # The original code uses them in training_step_objective_function, but we will rewrite that.
        return lvec, None
        
    return lvec

def get_pivots_and_scales(label_vals):
    qs = np.percentile(label_vals, (2.5, 50, 97.5), axis=0)
    pivots = qs[1]
    scales = (qs[2] - qs[0])/4.
    return pivots, scales

# --- Standard Training (No Label Errors) ---

@jax.jit
def _solve_coeff(flux, ivar, lvec, scatter):
    """ Solve for coefficients for a single pixel and single scatter value """
    # flux: (nstars,)
    # ivar: (nstars,)
    # lvec: (nstars, n_terms)
    # scatter: scalar
    
    Cinv = ivar / (1.0 + ivar * scatter**2)
    
    # Solve (L^T Cinv L) c = L^T Cinv f
    # Weighted least squares: (sqrt(Cinv) L) c = sqrt(Cinv) f
    
    w_sqrt = jnp.sqrt(Cinv)
    Aw = lvec * w_sqrt[:, None]
    yw = flux * w_sqrt
    
    # Use lstsq
    coeff, residuals, rank, s = jnp.linalg.lstsq(Aw, yw, rcond=None)
    
    # Calculate chi-squared and logdet
    chi = w_sqrt * (flux - lvec @ coeff)
    chisq = jnp.sum(chi**2)
    logdet_Cinv = jnp.sum(jnp.log(Cinv))
    
    # Objective to minimize: chisq - logdet_Cinv
    # Wait, original code maximizes likelihood?
    # lnL = 0.5 * logdet_Cinv - 0.5 * chisq
    # We want to maximize lnL, so minimize -lnL ~ chisq - logdet_Cinv
    
    obj = chisq - logdet_Cinv
    
    return coeff, obj, chisq

@jax.jit
def _train_pixel(flux, ivar, lvec, scatter_grid):
    """ Train a single pixel by checking scatter grid """
    # flux: (nstars,)
    # ivar: (nstars,)
    # lvec: (nstars, n_terms)
    # scatter_grid: (n_scatters,)
    
    # vmap over scatter grid
    solve_func = lambda s: _solve_coeff(flux, ivar, lvec, s)
    coeffs, objs, chisqs = jax.vmap(solve_func)(scatter_grid)
    
    # Find best scatter
    best_idx = jnp.argmin(objs)
    best_scatter = scatter_grid[best_idx]
    best_coeff = coeffs[best_idx]
    best_chisq = chisqs[best_idx]
    
    return best_coeff, best_scatter, best_chisq

def _train_model(ds, batch_size=500):
    label_vals = ds.tr_label
    fluxes = ds.tr_flux
    ivars = ds.tr_ivar
    
    # Handle low ivars
    ivars = jnp.where(ivars < 0.01, 0.01, ivars)
    
    pivots, scales = get_pivots_and_scales(label_vals)
    lvec = _get_lvec(label_vals, pivots, scales)
    
    # Transpose to (npixels, nstars)
    fluxes = fluxes.T
    ivars = ivars.T
    
    npixels = fluxes.shape[0]
    nstars = fluxes.shape[1]
    
    # Scatter grid
    ln_scatter_vals = jnp.arange(jnp.log(0.0001), 0., 0.5)
    scatter_grid = jnp.exp(ln_scatter_vals)
    
    # Batch processing over pixels
    coeffs_list = []
    scatters_list = []
    chisqs_list = []
    
    indices = range(0, npixels, batch_size)
    if tqdm is not None:
        indices = tqdm(indices, desc="Training model", total=(npixels + batch_size - 1) // batch_size)
        
    train_func = partial(_train_pixel, lvec=lvec, scatter_grid=scatter_grid)
    
    for i in indices:
        batch_flux = fluxes[i:i+batch_size]
        batch_ivar = ivars[i:i+batch_size]
        
        b_coeffs, b_scatters, b_chisqs = jax.vmap(train_func)(batch_flux, batch_ivar)
        
        coeffs_list.append(b_coeffs)
        scatters_list.append(b_scatters)
        chisqs_list.append(b_chisqs)
        
    coeffs = jnp.concatenate(coeffs_list, axis=0)
    scatters = jnp.concatenate(scatters_list, axis=0)
    chisqs = jnp.concatenate(chisqs_list, axis=0)
    
    return np.array(coeffs), np.array(scatters), np.array(chisqs), pivots, scales

# --- Training with Label Errors ---

def _objective_function(params, fluxes, ivars, lvec0, ldelta_vec, Nstars, Nlabels, Npix):
    """
    Objective function for training with label errors.
    Minimize -2 * lnL
    """
    # Unpack parameters
    # params structure: 
    # [coeffs (Npix*Nlabels), scatters (Npix), labels (Nstars*Nlabels)]
    
    idx1 = Npix * Nlabels
    idx2 = idx1 + Npix
    
    coeffs_flat = params[:idx1]
    scatters = params[idx1:idx2]
    labels_flat = params[idx2:]
    
    coeffs = coeffs_flat.reshape((Npix, Nlabels))
    lvec = labels_flat.reshape((Nstars, Nlabels)) # These are the latent vector terms
    
    # Likelihood 1: Pixels
    # Model = coeffs @ lvec.T
    model = coeffs @ lvec.T # (Npix, Nstars)
    
    resids = fluxes - model
    
    # Inverse variance with scatter
    # ivars: (Npix, Nstars)
    # scatters: (Npix)
    inv_var = ivars / (1.0 + ivars * scatters[:, None]**2)
    
    lnL_pixels = -0.5 * jnp.sum(resids**2 * inv_var - jnp.log(inv_var / (2.0 * jnp.pi)))
    
    # Likelihood 2: Labels (Vector terms)
    # Prior on vector terms: Gaussian around input vector (lvec0) with variance ldelta_vec^2
    
    ldelta2 = ldelta_vec**2
    lnL_labels = -0.5 * jnp.sum((lvec - lvec0)**2 / ldelta2 + jnp.log(2.0 * jnp.pi * ldelta2))
    
    lnL_total = lnL_pixels + lnL_labels
    
    return -2.0 * lnL_total

@partial(jax.jit, static_argnames=['Nstars', 'Nlabels', 'Npix'])
def _objective_and_grad(params, fluxes, ivars, lvec0, ldelta_vec, Nstars, Nlabels, Npix):
    val, grads = jax.value_and_grad(_objective_function)(params, fluxes, ivars, lvec0, ldelta_vec, Nstars, Nlabels, Npix)
    return val, grads

def _train_model_new(ds):
    label_vals = ds.tr_label
    fluxes = ds.tr_flux
    ivars = ds.tr_ivar
    ldelta = ds.tr_delta
    
    # Handle low ivars
    ivars = jnp.where(ivars < 0.01, 0.01, ivars)
    
    pivots, scales = get_pivots_and_scales(label_vals)
    lvec, _ = _get_lvec(label_vals, pivots, scales, derivs=True) # Get original lvec
    
    scaled_ldelta = ldelta / scales[None, :]
    
    # Transpose fluxes/ivars to (Npix, Nstars)
    fluxes = fluxes.T
    ivars = ivars.T
    
    Npix = fluxes.shape[0]
    Nstars = fluxes.shape[1]
    Nlabels = lvec.shape[1] # Vector size
    
    # Construct ldelta_vec (uncertainties for all terms)
    linear_offsets = scaled_ldelta
    # Quadratic uncertainties: approx 2 * l * dl? 
    # Original code:
    # quadratic_offsets = np.array([np.outer(m, m)[np.triu_indices(label_vals.shape[1])]for m in (linear_offsets)]) * 10
    # This looks like it squares the delta? (dl)^2? And multiplies by 10?
    # Original code logic seems ad-hoc or specific approximation.
    # Let's replicate it.
    
    # linear_offsets: (Nstars, N_phys_labels)
    # We need outer product of linear_offsets with itself?
    outer = jax.vmap(lambda x: jnp.outer(x, x))(linear_offsets)
    triu_idx = jnp.triu_indices(linear_offsets.shape[1])
    quadratic_offsets = outer[:, triu_idx[0], triu_idx[1]] * 10.0
    
    ones = jnp.ones((Nstars, 1)) * 0.001
    ldelta_vec = jnp.hstack([ones, linear_offsets, quadratic_offsets])
    
    # Initial guess
    # Use standard training to get coeffs and scatters
    print("Running initial standard training...")
    coeffs_init, scatters_init, _, _, _ = _train_model(ds)
    
    # Flatten parameters
    # [coeffs, scatters, labels]
    x0_coeffs = coeffs_init.flatten()
    x0_scatters = scatters_init
    x0_labels = lvec.flatten()
    
    x0 = jnp.concatenate([x0_coeffs, x0_scatters, x0_labels])
    
    print("Optimizing full model with label errors...")
    
    # Use scipy.optimize.minimize with JAX gradients
    # We need a wrapper because scipy expects numpy arrays
    
    fun = lambda p: _objective_and_grad(p, fluxes, ivars, lvec, ldelta_vec, Nstars, Nlabels, Npix)
    
    def func_wrapper(p):
        v, g = fun(p)
        return float(v), np.array(g)
    
    res = op.minimize(func_wrapper, np.array(x0), method='L-BFGS-B', jac=True,
                      options={'gtol': 1e-12, 'ftol': 1e-12, 'maxiter': 1000, 'disp': True})
    
    print(f"Optimization success: {res.success}")
    
    # Unpack results
    x = jnp.array(res.x)
    idx1 = Npix * Nlabels
    idx2 = idx1 + Npix
    
    coeffs = x[:idx1].reshape((Npix, Nlabels))
    scatters = x[idx1:idx2]
    new_labels_vec = x[idx2:].reshape((Nstars, Nlabels))
    
    # Calculate chi-squareds
    # Re-evaluate objective components or just chisq
    model = coeffs @ new_labels_vec.T
    resids = fluxes - model
    inv_var = ivars / (1.0 + ivars * scatters[:, None]**2)
    chisqs = jnp.sum(resids**2 * inv_var, axis=0) # Sum over pixels? Or return per pixel?
    # Original returns 'chisqs' which seems to be per pixel in _train_model, but here?
    # In _train_model_new original:
    # chisqs = np.array(chisqs) -> from train_all_wavelength -> returns 'chis' -> 0?
    # Original `train_all_wavelength` returns `chis = 0`.
    # So `chisqs` is just 0 in original code?
    # "chis = 0" on line 97 of original.
    # So we can return zeros or actual chisqs. Let's return actual chisqs per star?
    # Original `_train_model` returns `all_chisqs` which is per pixel.
    # Let's return per pixel chisqs.
    chisqs_pix = jnp.sum(resids**2 * inv_var, axis=1)
    
    return np.array(coeffs), np.array(scatters), np.array(new_labels_vec), np.array(chisqs_pix), pivots, scales
