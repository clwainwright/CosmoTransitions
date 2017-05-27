"""
A module for finding instantons between vacua in multiple field dimensions.

The basic strategy is an iterative process:

    1. Make an ansatz for the path along which the field will travel.
    2. Split up the equations of motion into components that are parallel and
       perpendicular to the direction of travel along the path.
    3. The direction of motion parallel to the path reduces to a
       one-dimensional equation of motion, which can be solved using the
       overshoot / undershoot techniques in :mod:`tunneling1D`. Solve it.
    4. Treating the motion of the field as a classical particle moving in an
       inverted potential, calculate the normal forces that would need to act
       on the particle to keep it on the path. If this forces are (close enough
       to) zero, the ansatz was correctly. Otherwise iteratively deform the path
       in the direction of the normal forces, stopping when the forces go to
       zero.
    5. Loop back to step 3 until no further deformation is necessary.

The classes :class:`Deformation_Spline` and :class:`Deformation_Points` will
perform step 3, while :func:`fullTunneling` will run the entire loop.

For more explicit details, see the original paper
`Comput. Phys. Commun. 183 (2012)`_ [`arXiv:1109.4189`_].

.. _`Comput. Phys. Commun. 183 (2012)`:
    http://dx.doi.org/10.1016/j.cpc.2012.04.004

.. _`arXiv:1109.4189`: http://arxiv.org/abs/1109.4189
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import optimize, interpolate, integrate
from collections import namedtuple

from . import tunneling1D
from . import helper_functions

import sys
if sys.version_info >= (3,0):
    xrange = range

__version__ = "2.0a2"


class DeformationError(Exception):
    """Raised when path deformation fails."""
    pass


class Deformation_Spline:
    """
    Deform a path in the presence of a potential such that the normal forces
    along the path are zero.

    This class fits a spline to the points, and does the actual deformation
    on the spline rather than on the points themselves. This make the path
    somewhat smoother than it would otherwise be (which is generally desirable),
    but it does make it difficult to resolve sharp turns in the path.

    Parameters
    ----------
    phi : array_like
        The list of points that constitutes the original path. Should have
        shape ``(n_points, n_dimensions)``.
    dphidr : array_like
        The 'speed' along the path at the initial points. This does not change
        as the path deforms. Should have shape ``(n_points,)``. Gets saved into
        the attribute `v2` as ``v2 = dphidr[:,np.newaxis]**2``.
    dV : callable
        The potential gradient as a function of phi. The output shape should be
        the same as the input shape, which will be ``(..., n_dimensions)``.
    nb : int, optional
        Number of basis splines to use.
    kb : int, optional
        Order of basis splines.
    v2min : float, optional
        The smallest the square of dphidr is allowed to be, relative
        to the characteristic force exterted by F_ext. Note that the
        self-correcting nature of the deformation goes away when dphidr=0.
    fix_start, fix_end : bool, optional
        If True, the force on the first/last point along the path is set to
        zero, so the point will not change in the deformation step.
    save_all_steps : bool, optional
        If True, each step gets saved into ``self.phi_list`` and
        ``self.F_list``.

    Attributes
    ----------
    phi : array_like
        Set during initialization, and then rewritten at each step.
    num_steps : int
        Total number of steps taken.
    """
    """
    Additional (private) attributes
    -------------------------------
    _L : float
        Total length of the path, set during initialization.
    _t : array_like
        Array from (0,1] marking the locations of each point.
    _X, _dX, _d2X : array_like
        Spline basis functions and their derivatives evaluated at `_t`. Set
        during initialization.
    _beta : array_like
        The spline coefficients for each dimension. Recalculated each step.
    _F_prev, _phi_prev : array_like
        The normal force and the path points at the last step.
    """
    def __init__(self, phi, dphidr, dV, nb=10, kb=3, v2min=0.0,
                 fix_start=False, fix_end=False, save_all_steps=False):
        # First step: convert phi to a set of path lengths.
        phi = np.asanyarray(phi)
        dphi = phi[1:]-phi[:-1]
        dL = np.sqrt(np.sum(dphi*dphi,axis=-1))
        y = np.cumsum(dL)
        self._L = y[-1]
        self._t = np.append(0,y)/self._L
        self._t[0] = 1e-100  # Without this, the first data point isn't in
                            # any bin (this matters for dX).

        # Create the starting spline:
        # make the knots and then the spline matrices at each point t
        t0 = np.append(np.append([0.]*(kb-1), np.linspace(0,1,nb+3-kb)),
                       [1.]*(kb-1))
        self._X,self._dX,self._d2X = helper_functions.Nbspld2(t0, self._t, kb)
        self._t = self._t[:,np.newaxis]  # Shape (n, 1)
        # subtract off the linear component.
        phi0, phi1 = phi[:1], phi[-1:]  # These are shape (1,N)
        phi_lin = phi0 + (phi1-phi0)*self._t
        self._beta, residues, rank, s = np.linalg.lstsq(self._X, phi-phi_lin)

        # save the points for future use.
        self.phi = np.asanyarray(phi)  # shape (n,N)
        self.v2 = np.asanyarray(dphidr)[:,np.newaxis]**2  # shape (n,1)
        self.dV = dV
        self.F_list = []
        self.phi_list = []
        self._phi_prev = self._F_prev = None
        self.save_all_steps = save_all_steps
        self.fix_start, self.fix_end = fix_start, fix_end
        self.num_steps = 0

        # ensure that v2 isn't too small:
        v2 = dphidr**2
        v2min *= np.max(np.sum(dV(self.phi)**2, -1)**.5*self._L/nb)
        v2[v2 < v2min] = v2min
        self.v2 = v2[:,np.newaxis]

    _forces_rval = namedtuple("forces_rval", "F_norm dV")
    def forces(self):
        """
        Calculate the normal force and potential gradient on the path.

        Returns
        -------
        F_norm, dV : array_like
        """
        X, dX, d2X = self._X, self._dX, self._d2X
        beta = self._beta
        """First find phi, dphi, and d2phi. Note that dphi needs to get a
        linear component added in, while d2phi does not."""
        phi = self.phi
        dphi = np.sum(beta[np.newaxis,:,:]*dX[:,:,np.newaxis], axis=1) \
            + (self.phi[-1]-self.phi[1])[np.newaxis,:]
        d2phi = np.sum(beta[np.newaxis,:,:]*d2X[:,:,np.newaxis], axis=1)
        """Compute dphi/ds, where s is the path length instead of the path
        parameter t. This is just the direction along the path."""
        dphi_sq = np.sum(dphi*dphi, axis=-1)[:,np.newaxis]
        dphids = dphi/np.sqrt(dphi_sq)
        """Then find the acceleration along the path, i.e. d2phi/ds2:"""
        d2phids2 = (d2phi - dphi * np.sum(dphi*d2phi, axis=-1)[:,np.newaxis] /
                    dphi_sq)/dphi_sq
        """Now we have the path at the points t, as well its derivatives with
        respect to it's path length. We still need to get the normal force
        acting on the path."""
        dV = self.dV(phi)
        dV_perp = dV - np.sum(dV*dphids, axis=-1)[:,np.newaxis]*dphids
        F_norm = d2phids2 * self.v2 - dV_perp
        if (self.fix_start):
            F_norm[0] = 0.0
        if (self.fix_end):
            F_norm[-1] = 0.0
        return self._forces_rval(F_norm, dV)

    _step_rval = namedtuple("step_rval", "stepsize step_reversed fRatio")
    def step(self, lastStep, maxstep=.1, minstep=1e-4, reverseCheck=.15,
             stepIncrease=1.5, stepDecrease=5., checkAfterFit=True,
             verbose=False):
        """
        Deform the path one step.

        Each point is pushed in the direction of the normal force - the force
        that the path exerts on a classical particle moving with speed `dphidr`
        in a potential with gradient `dV` such that the particle stays on the
        path. A stepsize of 1 corresponds to moving the path an amount
        ``L*N/(dV_max)``, where `L` is the length of the (original) path,
        `N` is the normal force, and `dV_max` is the maximum force exerted by
        the potential along the path.

        Parameters
        ----------
        lastStep : float
            Size of the last step.
        maxstep, minstep : float, optional
        reverseCheck : float, optional
            Percentage of points for which the force can reverse direcitons
            (relative to the last step) before the stepsize is decreased.
            If ``reverseCheck >= 1``, the stepsize is kept at `lastStep`.
        stepIncrease, stepDecrease : float, optional
            The amount to increase or decrease stepsize over the last step.
            Both should be bigger than 1.
        checkAfterFit : bool, optional
            If True, the convergence test is performed after the points are fit
            to a spline. If False, it's done beforehand.
        verbose : bool, optional
            If True, output is printed at each step.

        Returns
        -------
        stepsize : float
            The stepsize used for this step.
        step_reversed : bool
            True if this step was reversed, otherwise False
        fRatio : float
            The ratio of the maximum normal force to the maximum potential
            gradient. When the path is a perfect fit, this should go to zero. If
            ``checkAfterFit == True``, the normal force in this ratio is defined
            by the change in phi this step *after* being fit to a spline. Note
            that if the spline does a poor job of fitting the points after the
            deformation in this step (which might be the case if there are not
            enough basis functions), and if ``checkAfterFit == False``, this
            ratio can be non-zero or large even if there is no change in `phi`.

        Notes
        -----
        In prior versions of this function (CosmoTransitions v1.0.2 and
        earlier), the start and end points of the
        path were effectively held fixed during the main deformation. This was
        because the line ``phi_lin = phi[:1] + ...`` was calculated *before* the
        line ``phi = phi+F*stepsize``. Since the spline basis functions are
        zero at the start and end points (the spline is added on top of the
        straight line between the end points), when the points were later taken
        from the spline the end points wouldn't move. This was by design, since
        for thin-walled bubbles the endpoints should stay fixed at the two
        vacua. However, this caused problems for thick-walled bubbles where the
        end points should move.

        To get around this, prior versions added an extra block of code to move
        the end points before the main deformation. However, this was
        unnecessarily complicated and led to error-prone code. In this version,
        the end points are always allowed to move if the force `F` is non-zero.
        In the thin-walled case, the force should be almost exactly zero at
        the end points anyways (there is zero potential gradient and `dphidr` is
        zero), so they should stay fixed on their own.
        """
        # Find out the direction of the deformation.
        F,dV = self.forces()
        F_max = np.max(np.sqrt(np.sum(F*F,-1)))
        dV_max = np.max(np.sqrt(np.sum(dV*dV,-1)))
        fRatio1 = F_max / dV_max
        # Rescale the normal force so that it's relative to L:
        F *= self._L / dV_max

        # Now, see how big the stepsize should be
        stepsize = lastStep
        phi = self.phi
        assert(maxstep > minstep)
        step_reversed = False
        if reverseCheck < 1 and self._F_prev is not None:
            FdotFlast = np.sum(F*self._F_prev, axis=1)
            if np.sum(FdotFlast < 0) > len(FdotFlast)*reverseCheck:
                # we want to reverse the last step
                if stepsize > minstep:
                    step_reversed = True
                    phi = self._phi_prev
                    F = self._F_prev
                    if verbose: print("step reversed")
                    stepsize = lastStep/stepDecrease
            else:
                """ No (large number of) indices reversed, just do a regular
                step. Increase the stepsize a bit over the last one."""
                stepsize = lastStep * stepIncrease
        if stepsize > maxstep: stepsize = maxstep
        if stepsize < minstep: stepsize = minstep

        # Save the state before the step
        self._phi_prev = phi
        self._F_prev = F
        if self.save_all_steps:
            self.phi_list.append(phi)
            self.F_list.append(F)

        """Now make the step. It's important to not use += so that this doesn't
        change the value stored in self.phi_list."""
        phi = phi+F*stepsize

        # fit to the spline
        phi_lin = phi[:1] + (phi[-1:]-phi[:1])*self._t
        phi -= phi_lin
        self._beta, residues, rank, s = np.linalg.lstsq(self._X, phi)
        phi = np.sum(self._beta[np.newaxis,:,:]*self._X[:,:,np.newaxis], axis=1)
        phi += phi_lin
        self.phi = phi

        Ffit = (phi-self._phi_prev)/stepsize
        fRatio2 = np.max(np.sqrt(np.sum(Ffit*Ffit,-1)))/self._L

        if verbose:
            print("step: %i; stepsize: %0.2e; fRatio1 %0.2e; fRatio2: %0.2e"
                  % (self.num_steps, stepsize, fRatio1, fRatio2))

        fRatio = fRatio2 if checkAfterFit else fRatio1
        return self._step_rval(stepsize, step_reversed, fRatio)

    def deformPath(self, startstep=2e-3,
                   fRatioConv=.02, converge_0=5., fRatioIncrease=5.,
                   maxiter=500, verbose=True, callback=None, step_params={}):
        """
        Deform the path many individual steps, stopping either when the
        convergence criterium is reached, when the maximum number of iterations
        is reached, or when the path appears to be running away from
        convergence.

        Parameters
        ----------
        startstep : float, optional
            Starting stepsize used in :func:`step`.
        fRatioConv : float, optional
            The routine will stop when the maximum normal force on the path
            divided by the maximum potential gradient is less than this.
        converge_0 : float, optional
            On the first step, use a different convergence criterion. Check if
            ``fRatio < convergence_0 * fRatioConv``.
        fRatioIncrease :float, optional
            The maximum fractional amount that fRatio can increase before
            raising an error.
        maxiter : int, optional
            Maximum number of steps to take (ignoring reversed steps).
        verbose : bool, optional
            If True, print the ending condition.
        callback : callable, optional
            Called after each step. Should accept an instance of this class as a
            parameter, and return False if deformation should stop.
        step_params : dict, optional
            Parameters to pass to :func:`step`.

        Returns
        -------
        deformation_converged : bool
            True if the routine stopped because it converged (as determined by
            `fRatioConv`), False otherwise.
        """
        minfRatio = np.inf
        minfRatio_index = 0
        minfRatio_beta = None
        minfRatio_phi = None
        stepsize = startstep
        deformation_converged = False
        while True:
            self.num_steps += 1
            stepsize, step_reversed, fRatio = self.step(stepsize, **step_params)
            if callback is not None and not callback(self):
                break
            minfRatio = min(minfRatio, fRatio)
            if fRatio < fRatioConv or (self.num_steps == 1
                                       and fRatio < converge_0*fRatioConv):
                if verbose:
                    print("Path deformation converged. " +
                          "%i steps. fRatio = %0.5e" % (self.num_steps,fRatio))
                deformation_converged = True
                break
            if minfRatio == fRatio:
                minfRatio_beta = self._beta
                minfRatio_index = self.num_steps
                minfRatio_phi = self.phi
            if fRatio > fRatioIncrease*minfRatio and not step_reversed:
                self._beta = minfRatio_beta
                self.phi = minfRatio_phi
                self.phi_list = self.phi_list[:minfRatio_index]
                self.F_list = self.F_list[:minfRatio_index]
                err_msg = ("Deformation doesn't appear to be converging."
                           "Stopping at the point of best convergence.")
                if verbose: print(err_msg)
                raise DeformationError(err_msg)
            if self.num_steps >= maxiter:
                if verbose:
                    print("Maximum number of deformation iterations reached.")
                break
        return deformation_converged


class Deformation_Points:
    """
    Deform a path in the presence of a potential such that the normal forces
    along the path are zero.

    Unlike :class:`Deformation_Spline`, this class changes the points
    themselves rather than fitting a spline to the points. It is a more
    straightforward implementation, and when run with comparable inputs (i.e.,
    the number of basis splines is about the same as the number of points), this
    method tends to be somewhat faster. The individual stepsizes here change
    with the total number of points, whereas in the spline implementation they
    mostly depend on the number of basis functions. However, as long as the path
    is fairly smooth, the total number of splines in that class can probably be
    smaller than the total number of points in this class, so this class will
    tend to be somewhat slower.

    The two implementations should converge upon the same answer when the
    number of points and basis functions get large.

    Parameters
    ----------
    phi : array_like
        The list of points that constitutes the original path. Should have
        shape ``(n_points, n_dimensions)``.
    dphidr : array_like
        The 'speed' along the path at the initial points. This does not change
        as the path deforms. Should have shape ``(n_points,)``. Gets saved into
        the attribute ``self.v2`` as ``v2 = dphidr[:,np.newaxis]**2``.
    dV : callable
        The potential gradient as a function of phi. The output shape should be
        the same as the input shape, which will be ``(..., n_dimensions)``.
    fix_start, fix_end : bool, optional
        If True, the force on the first/last point along the path is set to
        zero, so the point will not change in the deformation step.
    save_all_steps : bool, optional
        If True, each step gets saved into ``self.phi_list`` and
        ``self.F_list``.

    Attributes
    ----------
    phi : array_like
        Set during initialization, and then rewritten at each step.
    num_steps : int
        Total number of steps taken.
    """
    def __init__(self, phi, dphidr, dV,
                 fix_start=False, fix_end=False, save_all_steps=False):
        self.phi = np.asanyarray(phi)  # shape (n,N)
        self.v2 = np.asanyarray(dphidr)[:,np.newaxis]**2  # shape (n,1)
        self.dV = dV
        self.F_list = []
        self.phi_list = []
        self.save_all_steps = save_all_steps
        self.fix_start, self.fix_end = fix_start, fix_end
        self.num_steps = 0

    _forces_rval = namedtuple("forces_rval", "F_norm dV")
    def forces(self, phi=None):
        """
        Calculate the normal force and potential gradient on the path.

        Returns
        -------
        F_norm, dV : array_like
        """
        if phi is None: phi = self.phi
        # Let `t` be some variable that parametrizes the points such that
        # t_i = i. Calculate the derivs of phi w/ respect to t.
        dphi = helper_functions.deriv14_const_dx(phi.T).T
        d2phi = helper_functions.deriv23_const_dx(phi.T).T
        # Let `x` be some variable that parametrizes the path such that
        # |dphi/dx| = 1. Calculate the derivs.
        dphi_abssq = np.sum(dphi*dphi, axis=-1)[:,np.newaxis]
        dphi /= np.sqrt(dphi_abssq)  # This is now dphi/dx
        d2phi /= dphi_abssq  # = d2phi/dx2 + (dphi/dx)(d2phi/dt2)/(dphi/dt)^2
        d2phi -= np.sum(d2phi*dphi, axis=-1)[:,np.newaxis] * dphi  # = d2phi/dx2
        # Calculate the total force.
        dV = self.dV(phi)
        dV_perp = dV - np.sum(dV*dphi, axis=-1)[:,np.newaxis] * dphi
        F_norm = d2phi*self.v2 - dV_perp
        if (self.fix_start):
            F_norm[0] = 0.0
        if (self.fix_end):
            F_norm[-1] = 0.0
        return self._forces_rval(F_norm, dV)

    _step_rval = namedtuple("step_rval", "stepsize fRatio")
    def step(self, stepsize, minstep, diff_check=0.1, step_decrease=2.):
        """
        Take two half-steps in the direction of the normal force.

        Parameters
        ----------
        stepsize : float
            Determines change in `phi`: ``phi += F_norm*stepsize``.
        minstep : float
            The smallest the stepsize is allowed to be.
        diff_check : float, optional
            The stepsize is chosen such that difference between the forces at
            beginning of the step and halfway through the step is small
            compared to the force itself: ``max(F2-F1) < diff_check * max(F1)``,
            where ``max`` here really means the maximum absolute value of the
            force in each direction.
        step_decrease : float, optional
            Amount by which to decrease the stepsize if the step is too big.

        Returns
        -------
        stepsize : float
            The stepsize used for this step.
        fRatio : float
            The ratio of the maximum normal force to the maximum potential
            gradient. When the path is a perfect fit, this should go to zero.
        """
        F1,dV = self.forces()
        F_max = np.max(np.sqrt(np.sum(F1*F1,-1)))
        dV_max = np.max(np.sqrt(np.sum(dV*dV,-1)))
        fRatio = F_max / dV_max

        if self.save_all_steps:
            self.phi_list.append(self.phi)
            self.F_list.append(F1)

        while True:
            # Take one full step
          #  phi1 = self.phi + F*stepsize
            # Take two half steps
            phi2 = self.phi + F1*(stepsize*0.5)
            F2 = self.forces(phi2)[0]
            if stepsize <= minstep:
                stepsize = minstep
                break
          #  phi2 += F2*(stepsize*0.5)
            DF_max = np.max(np.abs(F2-F1), axis=0)
            F_max = np.max(np.abs(F1), axis=0)
            if (DF_max < diff_check*F_max).all():
                break
            stepsize /= step_decrease
        self.phi = phi2 + F2*(stepsize*0.5)

        return self._step_rval(stepsize, fRatio)

    def deformPath(self, startstep=.1, minstep=1e-6, step_increase=1.5,
                   fRatioConv=.02, converge_0=5., fRatioIncrease=20.,
                   maxiter=500, verbose=1, callback=None, step_params={}):
        """
        Deform the path many individual steps, stopping either when the
        convergence criterium is reached, when the maximum number of iterations
        is reached, or when the path appears to be running away from
        convergence.

        Parameters
        ----------
        startstep, maxstep : float, optional
            Starting and maximum stepsizes used in :func:`step`, rescaled by
            ``|phi[0]-phi[1]| / (max(dV)*num_points)``.
        fRatioConv : float, optional
            The routine will stop when the maximum normal force on the path
            divided by the maximum potential gradient is less than this.
        converge_0 : float, optional
            On the first step, use a different convergence criterion. Check if
            ``fRatio < convergence_0 * fRatioConv``.
        fRatioIncrease :float, optional
            The maximum fractional amount that fRatio can increase before
            raising an error.
        maxiter : int, optional
            Maximum number of steps to take (ignoring reversed steps).
        verbose : int, optional
            If ``verbose >= 1``, print the ending condition.
            If ``verbose >= 2``, print `fRatio` and `stepsize` at each step.
        callback : callable, optional
            Called after each step. Should accept an instance of this class as a
            parameter, and return False if deformation should stop.
        step_params : dict, optional
            Parameters to pass to :func:`step`.

        Returns
        -------
        deformation_converged : bool
            True if the routine stopped because it converged (as determined by
            `fRatioConv`), False otherwise.
        """
        minfRatio = np.inf
        Delta_phi = np.sum(np.sqrt((self.phi[0]-self.phi[-1])**2))
        dV_max = np.max(np.sum(self.dV(self.phi)**2, axis=-1))**0.5
        step_scale = Delta_phi / (len(self.phi) * dV_max)
        stepsize = startstep * step_scale
        minstep *= step_scale
        deformation_converged = False
        while True:
            self.num_steps += 1
            stepsize, fRatio = self.step(stepsize, minstep, **step_params)
            if verbose >= 2:
                print("step: %i; stepsize: %0.2e; fRatio: %0.2e"
                      % (self.num_steps, stepsize, fRatio))
            stepsize *= step_increase
            if callback is not None and not callback(self):
                break
            minfRatio = min(minfRatio, fRatio)
            if fRatio < fRatioConv or (self.num_steps == 1
                                       and fRatio < converge_0*fRatioConv):
                if verbose >= 1:
                    print("Path deformation converged." +
                          "%i steps. fRatio = %0.5e" % (self.num_steps,fRatio))
                deformation_converged = True
                break
            if minfRatio == fRatio:
                minfRatio_index = self.num_steps
                minfRatio_phi = self.phi
            if fRatio > fRatioIncrease*minfRatio:
                self.phi = minfRatio_phi
                self.phi_list = self.phi_list[:minfRatio_index]
                self.F_list = self.F_list[:minfRatio_index]
                err_msg = ("Deformation doesn't appear to be converging."
                           "Stopping at the point of best convergence.")
                if verbose >= 1: print(err_msg)
                raise DeformationError(err_msg)
            if self.num_steps >= maxiter:
                if verbose >= 1: print("Maximum number of iterations reached.")
                break
        return deformation_converged


_extrapolatePhi_rtype = namedtuple("extrapolatePhi_rval", "phi s L")
def _extrapolatePhi(phi0, V=None, tails=0.2):
    """
    Returns a list of points along the path, going linearly
    beyond the path to include the nearest minima.

    Parameters
    ----------
    phi0 : array_like
        The (multi-dimensional) path to extend.
    V : callable or None
        The potential to minimize, or None if the path should be extended a
        fixed amount beyond its ends.
    tails : float
        The amount relative to the path length to extrapolate beyond the end of
        the path (if V is None) or beyond the minima (if V is not None).

    Returns
    -------
    phi : array_like
        The extended list of points. The spacing between points in the extended
        regions should be approximately the same as the spacing between the
        input points.
    s : array_like
        The distance along the path (starting at ``phi0[0]``).
    L : float
        Total length of the path excluding tails.
    """
    phi1 = phi = phi0
    dphi = np.append(0,np.sum((phi1[1:]-phi1[:-1])**2,1)**.5)
    s1 = np.cumsum(dphi)
    L = s1[-1]
    npoints = phi1.shape[0]

    phi_hat0 = (phi[1]-phi[0])/np.sum((phi[1]-phi[0])**2)**.5
    if V is None:
        s0min = 0.0
    else:
        V0 = lambda x: V(phi[0] + phi_hat0*x*L)
        s0min = optimize.fmin(V0, 0.0, disp=0, xtol=1e-5)[0]*L
    if s0min > 0: s0min = 0.0
    s0 = np.linspace(s0min - L*tails, 0.0, npoints*tails)[:-1]
    phi0 = phi[0] + phi_hat0*s0[:,np.newaxis]

    phi_hat2 = (phi[-1]-phi[-2])/np.sum((phi[-1]-phi[-2])**2)**.5
    if V is None:
        s2min = 0.0
    else:
        V2 = lambda x: V(phi[-1] + phi_hat2*(x-1)*L)
        s2min = optimize.fmin(V2, 1, disp=0, xtol=1e-5)[0]*L
    if s2min < L: s2min = L
    s2 = np.linspace(L, s2min + L*tails, npoints*tails)[1:]
    phi2 = phi[-1] + phi_hat2*(s2[:,np.newaxis]-L)

    phi = np.append(phi0, np.append(phi1, phi2, 0), 0)
    s = np.append(s0, np.append(s1, s2))

    return _extrapolatePhi_rtype(phi, s, L)


def _pathDeriv(phi):
    """Calculates to 4th order if len(phi) >= 5, otherwise 1st/2nd order."""
    if len(phi) >= 5:
        dphi = helper_functions.deriv14_const_dx(phi.T).T
    elif len(phi) > 2:
        dphi = np.empty_like(phi)
        dphi[1:-1] = 0.5*(phi[2:] - phi[:-2])
        dphi[0] = -1.5*phi[0] + 2*phi[1] - 0.5*phi[2]
        dphi[-1] = +1.5*phi[-1] - 2*phi[-2] + 0.5*phi[-3]
    else:
        dphi = np.empty_like(phi)
        dphi[:] = phi[1]-phi[0]
    return dphi


class SplinePath:
    """
    Fit a spline to a path in field space, and find the potential on that path.

    The spline-fitting happens in several steps:

      1. The derivatives of the input points are found, and used to
         determine the path length and direction at each point.
      2. If `extend_to_minima` is True, additional points are added at each end
         of the path such that ends lie on local minima.
      3. The points are fit to a spline, with the knots given by the path
         distances from the first point.
      4. If `reeval_distances` is True, the distances to each point are
         re-evaluated using the spline. A new spline is fit with more accurate
         knots.

    The potential as a function of distance can be defined in one of two ways.
    If `V_spline_samples` is None, the potential as a function of distance `x`
    along the path is given by `V[pts(x)]`, where `pts(x)` is the spline
    function that defines the path. If `V_spline_samples` is not None, the
    potential is first evaluated `V_spline_samples` times along the path, and
    another spline is fit to the output. In other words, when `V_spline_samples`
    is None, the input potential `V` is evaluated for every value `x` passed to
    to the class method :meth:`V`, whereas if `V_spline_samples` is not None,
    the input potential is only evaluated during initialization.

    Parameters
    ----------
    pts : array_like
        The points that describe the path, with shape ``(num_points, N_dim)``.
    V : callable
        The potential function. Input arrays will be shape ``(npts, N_dim)`` and
        output should have shape ``(npts,)``. Can be None.
    dV : callable, optional.
        The gradient of the potential. Input arrays will be shape
        ``(npts, N_dim)`` and output should have shape ``(npts, N_dim)``. Only
        used if ``V_spline_samples=None``.
    V_spline_samples : int or None, optional
        Number of samples to take along the path to create the spline
        interpolation functions. If None, the potential is evaluated directly
        from `V` given in the input. If not None, `V_spline_samples` should be
        large enough to resolve the smallest features in the potential. For
        example, the potential may have a very narrow potential barrier over
        which multiple samples should be taken.
    extend_to_minima : bool, optional
        If True, the input path is extended at each end until it hits local
        minima.
    reeval_distances : bool, optional
        If True, get more accurate distances to each knot by integrating along
        the spline.

    Attributes
    ----------
    L : float
        The total length of the path.
    """
    def __init__(self, pts, V, dV=None, V_spline_samples=100,
                 extend_to_minima=False, reeval_distances=True):
        assert len(pts) > 1
        # 1. Find derivs
        dpts = _pathDeriv(pts)
        # 2. Extend the path
        if extend_to_minima:
            def V_lin(x, p0, dp0, V): return V(p0+x*dp0)
            # extend at the front of the path
            xmin = optimize.fmin(V_lin, 0.0, args=(pts[0], dpts[0], V),
                                 xtol=1e-6, disp=0)[0]
            if xmin > 0.0: xmin = 0.0
            nx = np.ceil(abs(xmin)-.5) + 1
            x = np.linspace(xmin, 0, nx)[:, np.newaxis]
            pt_ext = pts[0] + x*dpts[0]
            pts = np.append(pt_ext, pts[1:], axis=0)
            # extend at the end of the path
            xmin = optimize.fmin(V_lin, 0.0, args=(pts[-1], dpts[-1], V),
                                 xtol=1e-6, disp=0)[0]
            if xmin < 0.0: xmin = 0.0
            nx = np.ceil(abs(xmin)-.5) + 1
            x = np.linspace(xmin, 0, nx)[::-1, np.newaxis]
            pt_ext = pts[-1] + x*dpts[-1]
            pts = np.append(pts[:-1], pt_ext, axis=0)
            # Recalculate the derivative
            dpts = _pathDeriv(pts)
        # 3. Find knot positions and fit the spline.
        pdist = integrate.cumtrapz(np.sqrt(np.sum(dpts*dpts, axis=1)),
                                   initial=0.0)
        self.L = pdist[-1]
        k = min(len(pts)-1, 3)  # degree of the spline
        self._path_tck = interpolate.splprep(pts.T, u=pdist, s=0, k=k)[0]
        # 4. Re-evaluate the distance to each point.
        if reeval_distances:
            def dpdx(_, x):
                dp = np.array(interpolate.splev(x, self._path_tck, der=1))
                return np.sqrt(np.sum(dp*dp))
            pdist = integrate.odeint(dpdx, 0., pdist,
                                     rtol=0, atol=pdist[-1]*1e-8)[:,0]
            self.L = pdist[-1]
            self._path_tck = interpolate.splprep(pts.T, u=pdist, s=0, k=k)[0]
        # Now make the potential spline.
        self._V = V
        self._dV = dV
        self._V_tck = None
        if V_spline_samples is not None:
            x = np.linspace(0,self.L,V_spline_samples)
            # extend 20% beyond this so that we more accurately model the
            # path end points
            x_ext = np.arange(x[1], self.L*.2, x[1])
            x = np.append(-x_ext[::-1], x)
            x = np.append(x, self.L+x_ext)
            y = self.V(x)
            self._V_tck = interpolate.splrep(x,y,s=0)

    def V(self, x):
        """The potential as a function of the distance `x` along the path."""
        if self._V_tck is not None:
            return interpolate.splev(x, self._V_tck, der=0)
        else:
            pts = interpolate.splev(x, self._path_tck)
            return self._V(np.array(pts).T)

    def dV(self, x):
        """`dV/dx` as a function of the distance `x` along the path."""
        if self._V_tck is not None:
            return interpolate.splev(x, self._V_tck, der=1)
        else:
            pts = interpolate.splev(x, self._path_tck)
            dpdx = interpolate.splev(x, self._path_tck, der=1)
            dV = self._dV(np.array(pts).T)
            return np.sum(dV.T*dpdx, axis=0)

    def d2V(self, x):
        """`d^2V/dx^2` as a function of the distance `x` along the path."""
        if self._V_tck is not None:
            return interpolate.splev(x, self._V_tck, der=2)
        else:
            raise RuntimeError("No spline specified. Cannot calculate d2V.")

    def pts(self, x):
        """
        Returns the path points as a function of the distance `x` along the
        path. Return value is an array with shape ``(len(x), N_dim)``.
        """
        pts = interpolate.splev(x, self._path_tck)
        return np.array(pts).T


def fullTunneling(path_pts, V, dV, maxiter=20, fixEndCutoff=.03,
                  save_all_steps=False, verbose=False,
                  callback=None, callback_data=None,
                  V_spline_samples=100,
                  tunneling_class=tunneling1D.SingleFieldInstanton,
                  tunneling_init_params={},
                  tunneling_findProfile_params={},
                  deformation_class=Deformation_Spline,
                  deformation_init_params={},
                  deformation_deform_params={}):
    """
    Calculate the instanton solution in multiple field dimension.

    This function works by looping four steps:

      1. Fit a spline to the path given by phi.
      2. Calculate the one-dimensional tunneling along this path.
      3. Deform the path to satisfy the transverse equations of motion.
      4. Check for convergence, and then go back to step 1.

    Parameters
    ----------
    path_pts : array_like
        An array of points that constitute the initial guess for the tunneling
        path, with shape ``(num_points, N_dim)``. The first point should be at
        (or near) the lower minimum (the minimum to which the field is
        tunneling), and the last point should be at the metastable minimum.
    V, dV : callable
        The potential function and its gradient. Both should accept input of
        shape ``(num_points, N_dim)`` and shape ``(N_dim,)``.
    maxiter : int, optional
        Maximum number of allowed deformation / tunneling iterations.
    save_all_steps : bool, optional
        If True, additionally output every single deformation sub-step.
    verbose : bool, optional
        If True, print a message at the start of each step.
    callback : callable
        User supplied function that is evaluted just prior to deforming the
        path. Should return True if the path should be deformed, and False if
        the deformation should be aborted. Should accept 4 arguments: a
        :class:`SplinePath` instance which describes the tunneling path, a
        tunneling object (instance of ``tunneling_class``), the profile found
        by the tunneling object, and extra callback data.
    callback_data : any type
        Extra data to pass to the callback function if there is one. If this
        function is called from :func:`transitionFinder.findAllTransitions`,
        the callback data is the tunneling temperature.

    Other Parameters
    ----------------
    V_spline_samples : int, optional
        Passed to :class:`SplinePath`. If None, no second derivative will be
        passed to the tunneling class, and it will instead be evaluted using
        finite differences.
    tunneling_class : class, optional
        Either :class:`tunneling1D.SingleFieldInstanton` or a subclass.
    tunneling_init_params : dict, optional
        Extra parameters to pass to the tunneling class.
    tunneling_findProfile_params : dict, optional
        Extra parameters to pass to ``tunneling_class.findProfile()``.
    deformation_class : class, optional
        Either :class:`Deformation_Spline` or :class:`Deformation_Points`, or
        some other object that exposes the same interface.
    deformation_init_params : dict, optional
        Extra parameters to pass to the deformation class.
    deformation_deform_params : dict, optional
        Extra parameters to pass to ``deformation_class.deformPath()``.

    Returns
    -------
    profile1D : namedtuple
        The return value from ``tunneling_class.findProfile()``.
    Phi : array_like or list
        The points that constitute the final deformed path. They are in
        one-to-one correspondence with the points in `profile1D`.
    action : float
        The Euclidean action of the instanton.
    fRatio : float
        A ratio of the largest transverse force on the final path relative to
        the largest potential gradient. This would be zero if the solution were
        perfect.
    saved_steps : list
        A list of lists, with each sub-list containing the saved steps for each
        deformation. Only written to if `save_all_steps` is True.

    Examples
    --------
    The following code shows typical usage for :func:`fullTunneling`. Most of
    the code is in setting up the potentials and plotting; it only takes one
    line to actually calculate each instanton.

    .. plot:: ../examples/fullTunneling.py
       :include-source:

    The callback function can be useful when this function is run from
    :func:`transitionFinder.findAllTransitions`. In that case, one doesn't need
    to accurately calculate the tunneling path when one knows that the action is
    already below some threshold. For example, the following callback function
    will abort the deformation when ``action / T <= 120``::

        def callback(path, tobj, profile, T):
            action = tobj.findAction(profile)
            return action / T > 120
    """
    assert maxiter > 0
    pts = np.asanyarray(path_pts)
    saved_steps = []
    deformation_init_params['save_all_steps'] = save_all_steps
    for num_iter in xrange(1, maxiter+1):
        if verbose: print("Starting tunneling step %i" % num_iter)
        # 1. Fit the spline to the path.
        path = SplinePath(pts, V, dV, V_spline_samples=V_spline_samples,
                          extend_to_minima=True)
        # 2. Do 1d tunneling along the path.
        if V_spline_samples is not None:
            tobj = tunneling_class(0.0, path.L, path.V, path.dV, path.d2V,
                                   **tunneling_init_params)
        else:
            tobj = tunneling_class(0.0, path.L, path.V, path.dV, None,
                                   **tunneling_init_params)
        profile1D = tobj.findProfile(**tunneling_findProfile_params)
        phi, dphi = profile1D.Phi, profile1D.dPhi
        phi, dphi = tobj.evenlySpacedPhi(phi, dphi, npoints=len(phi),
                                         fixAbs=False)
        dphi[0] = dphi[-1] = 0.0  # enforce this
        # 3. Deform the path.
        pts = path.pts(phi)  # multi-dimensional points
        deform_obj = deformation_class(pts, dphi, dV, **deformation_init_params)
        if callback and not callback(path, tobj, profile1D, callback_data):
            break
        try:
            converged = deform_obj.deformPath(**deformation_deform_params)
        except DeformationError as err:
            print(err.args[0])
            converged = False
        pts = deform_obj.phi
        if save_all_steps: saved_steps.append(deform_obj.phi_list)
        # 4. Check convergence. If the deformation converged after one step,
        # then assume that `path` is a good solution.
        if (converged and deform_obj.num_steps < 2):
            break
    else:
        if verbose: print("Reached maxiter in fullTunneling. No convergence.")
    # Calculate the ratio of max perpendicular force to max gradient.
    # Make sure that we go back a step and use the forces on the path, not the
    # most recently deformed path.
    deform_obj = deformation_class(pts, dphi, dV, **deformation_init_params)
    F, dV = deform_obj.forces()
    F_max = np.max(np.sqrt(np.sum(F*F,-1)))
    dV_max = np.max(np.sqrt(np.sum(dV*dV,-1)))
    fRatio = F_max / dV_max
    # Assemble the output
    rtuple = namedtuple("fullTunneling_rval",
                        "profile1D Phi action fRatio saved_steps")
    Phi = path.pts(profile1D.Phi)
    action = tobj.findAction(profile1D)
    return rtuple(profile1D, Phi, action, fRatio, saved_steps)
