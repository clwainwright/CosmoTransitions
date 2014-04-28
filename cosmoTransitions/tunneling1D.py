"""
This module (along with a few functions in :mod:`helper_functions`) contains
everything that is needed to calculate instantons in one field dimension. 
The primary class is :class:`SingleFieldInstanton`, which can calculate the
instanton solution in any number of spatial dimensions using the overshoot / 
undershoot method. Additional classes inherit common functionality from this
one, and can be used to calculate the bubble wall profile with constant
friction (:class:`WallWithConstFriction`) instead of radius-dependent friction,
or to calculate the instanton in the presence of gravity (*not yet
implemented*).

.. todo::
    Create and document a *CDL_Instanton* class for tunneling with gravity.
"""

__version__ = "2.0a2"

import numpy as np
from scipy import optimize, integrate, special, interpolate
from collections import namedtuple

import helper_functions
from helper_functions import rkqs, IntegrationError, clampVal
from helper_functions import cubicInterpFunction


class PotentialError(Exception):
    """
    Used when the potential does not have the expected characteristics.
    
    The error messages should be tuples, with the second item being one of
    ``("no barrier", "stable, not metastable")``.
    """
    pass

class SingleFieldInstanton:
    """
    This class will calculate properties of an instanton with a single scalar
    Field without gravity using the overshoot/undershoot method. 
    
    Most users will probably be primarily interested in the functions 
    :func:`findProfile` and :func:`findAction`.

    Note
    ----
    When the bubble is thin-walled (due to nearly degenerate minima), an 
    approximate solution is found to the equations of motion and integration 
    starts close to the wall itself (instead of always starting at the center
    of the bubble). This way the overshoot/undershoot method runs just as fast
    for extremely thin-walled bubbles as it does for thick-walled bubbles.
    
    Parameters
    ----------
    phi_absMin : float
        The field value at the stable vacuum to which the instanton
        tunnels. Nowhere in the code is it *required* that there actually be a 
        minimum at `phi_absMin`, but the :func:`findProfile` function will only 
        use initial conditions between `phi_absMin` and `phi_metaMin`, and the
        code is optimized for thin-walled bubbles when the center of the 
        instanton is close to `phi_absMin`.
    phi_metaMin : float
        The field value in the metastable vacuum.
    V : callable
        The potential function. It should take as its single parameter the field
        value `phi`.
    dV, d2V : callable, optional
        The potential's first and second derivatives. If not None, these
        override the methods :func:`dV` and :func:`d2V`.
    phi_eps : float, optional
        A small value used to calculate derivatives (if not overriden by
        the user) and in the function :func:`dV_from_absMin`. The input should
        be unitless; it is later rescaled by ``abs(phi_absMin - phi_metaMin)``.
    alpha : int or float, optional
        The coefficient for the friction term in the ODE. This is also
        the number of spacetime dimensions minus 1.
    phi_bar : float, optional
        The field value at the edge of the barrier. If `None`, it is found by 
        :func:`findBarrierLocation`.
    rscale : float, optional
        The approximate radial scale of the instanton. If `None` it is found by 
        :func:`findRScale`.
      
    Raises
    ------
    PotentialError
        when the barrier is non-existent or when the presumably stable minimum 
        has a higher energy that the metastable minimum.
    
    Examples
    --------
    Thick and thin-walled bubbles:
        
    .. plot::
        :include-source:
      
        from tunneling1D import SingleFieldInstanton
        import matplotlib.pyplot as plt
        
        # Thin-walled
        def V1(phi): return 0.25*phi**4 - 0.49*phi**3 + 0.235 * phi**2
        def dV1(phi): return phi*(phi-.47)*(phi-1)
        profile = SingleFieldInstanton(1.0, 0.0, V1, dV1).findProfile()
        plt.plot(profile.R, profile.Phi)
        
        # Thick-walled
        def V2(phi): return 0.25*phi**4 - 0.4*phi**3 + 0.1 * phi**2
        def dV2(phi): return phi*(phi-.2)*(phi-1)
        profile = SingleFieldInstanton(1.0, 0.0, V2, dV2).findProfile()
        plt.plot(profile.R, profile.Phi)

        plt.xlabel(r"Radius $r$")
        plt.ylabel(r"Field $\phi$")
        plt.show()
    """
    def __init__(self, phi_absMin, phi_metaMin, V, 
                 dV=None, d2V=None, phi_eps=1e-3, alpha=2,
                 phi_bar=None, rscale=None):
        self.phi_absMin, self.phi_metaMin = phi_absMin, phi_metaMin
        self.V = V
        if V(phi_metaMin) <= V(phi_absMin):
            raise PotentialError("V(phi_metaMin) <= V(phi_absMin); "
                                 "tunneling cannot occur.", "stable, not metastable")
        if dV is not None:
            self.dV = dV
        if d2V is not None:
            self.d2V = d2V
        if phi_bar is None:
            self.phi_bar = self.findBarrierLocation()
        else:
            self.phi_bar = phi_bar
        if rscale is None:
            self.rscale = self.findRScale()
        else:
            self.rscale = rscale
        self.alpha = alpha
        self.phi_eps = phi_eps * abs(phi_absMin - phi_metaMin)
    
    def dV(self, phi):
        R"""
        Calculates `dV/dphi` using finite differences.
        
        The finite difference is given by `self.phi_eps`, and the derivative 
        is calculated to fourth order.
        """
        eps = self.phi_eps
        V = self.V
        return (V(phi-2*eps) - 8*V(phi-eps) + 8*V(phi+eps) - V(phi+2*eps)
                ) / (12.*eps)
        
    def dV_from_absMin(self, delta_phi):
        R"""
        Calculates `dV/dphi` at ``phi = phi_absMin + delta_phi``.
        
        It is sometimes helpful to find `dV/dphi` extremely close to the
        minimum. In this case, floating-point error can be significant. To get
        increased accuracy, this function expands about the minimum in
        a Taylor series and uses that for nearby values. That is, 
        :math:`V'(\phi) \approx V''(\phi_{\rm absMin})(\phi-\phi_{\rm absMin})`.
        For values that are farther away, it instead uses :func:`dV`.
        It blends the two methods so that there are no numerical 
        discontinuities.
        
        This uses `self.phi_eps` to determine whether the field is considered 
        nearby or not.
        """
        phi = self.phi_absMin + delta_phi
        dV = self.dV(phi)
        # If phi is very close to phi_absMin, it should be safer to assume
        # that dV is zero exactly at phi_absMin and instead calculate dV from 
        # d2V.
        if self.phi_eps > 0:
            dV_ = self.d2V(phi) * delta_phi
            # blend the two together so that there are no discontinuites
            blend_factor = np.exp(-(delta_phi/self.phi_eps)**2)
            dV = dV_*blend_factor + dV*(1-blend_factor)
        return dV

    def d2V(self, phi):
        R"""
        Calculates `d^2V/dphi^2` using finite differences.
        
        The finite difference is given by `self.phi_eps`, and the derivative 
        is calculated to fourth order.
        """
        eps = self.phi_eps
        V = self.V
        return (-V(phi-2*eps) + 16*V(phi-eps) - 30*V(phi) 
                + 16*V(phi+eps) - V(phi+2*eps)) / (12.*eps*eps)
        
    def findBarrierLocation(self):
        R"""
        Find edge of the potential barrier.
        
        Returns
        -------
        phi_barrier : float
            The value such that `V(phi_barrier) = V(phi_metaMin)`
        """
        phi_tol = abs(self.phi_metaMin - self.phi_absMin) * 1e-12
        V_phimeta = self.V(self.phi_metaMin)
        phi1 = self.phi_metaMin
        phi2 = self.phi_absMin
        phi0 = 0.5 * (phi1+phi2)
        
        # Do a very simple binary search to narrow down on the right answer.
        while abs(phi1-phi2) > phi_tol:
            V0 = self.V(phi0)
            if V0 > V_phimeta:
                phi1 = phi0
            else:
                phi2 = phi0
            phi0 = 0.5 * (phi1+phi2)
        return phi0
        
    def findRScale(self):
        R"""
        Find the characteristic length scale for tunneling over the potential
        barrier.
        
        The characteristic length scale should formally be given by the period
        of oscillations about the top of the potential barrier. However, it is
        perfectly acceptable for the potential barrier to have a flat top, in
        which case a naive calculation of the length scale would be infinite.
        Instead, this function finds the top of the barrier along with a cubic
        function that has a maximum at the barrier top and a minimum at the
        metastable minimum. The returned length scale is then the period of
        oscillations about this cubic maximum. 

        Raises
        ------
        PotentialError
            when the barrier is non-existent.
        """
        """
        NOT USED:
        We could also do a sanity check in case the barrier goes to zero.
        A second way of finding the scale is to see how long it would take
        the field to roll from one minimum to the other if the potential were
        purely linear and there were no friction. 

        Parameters
        ----------
        second_check : float
            If bigger than zero, do the sanity check. Return value is then the
            larger of the first scale and the second scale times
            `second_check`.
        """
        def negV(phi): 
            return -self.V(clampVal(phi, self.phi_bar, self.phi_metaMin))
        phi_guess = 0.5 * (self.phi_bar + self.phi_metaMin)
        phi_tol = abs(self.phi_bar - self.phi_metaMin) * 1e-6
        phi_bar_top = optimize.fmin(negV, phi_guess, xtol=phi_tol, disp=0)[0]
        if not (self.phi_bar < phi_bar_top < self.phi_metaMin or 
                self.phi_bar > phi_bar_top > self.phi_metaMin):
            raise PotentialError("Minimization is placing the top of the "
            "potential barrier outside of the interval defined by "
            "phi_bar and phi_metaMin. Assume that the barrier does not exist.",
            "no barrier")

        Vtop = self.V(phi_bar_top) - self.V(self.phi_metaMin)
        xtop = phi_bar_top - self.phi_metaMin
        # Cubic function given by (ignoring linear and constant terms):
        # f(x) = C [(-1/3)x^3 + (1/2)x^2 xtop]
        # C = 6 Vtop / xtop^3
        # f''(xtop) = - C xtop
        # d2V = -6*Vtop / xtop**2
        # rscale = 1 / sqrt(d2V)
        if Vtop <= 0:
            raise PotentialError("Barrier height is not positive, "
                                 "does not exist.", "no barrier")
        rscale1 = abs(xtop) / np.sqrt(abs(6*Vtop))
        return rscale1
        # The following would calculate it a separate way, but this goes
        # to infinity when delta_V goes to zero, so it's a bad way of doing it
        delta_phi = abs(self.phi_absMin - self.phi_metaMin)
        delta_V = abs(self.V(self.phi_absMin) - self.V(self.phi_metaMin))
        rscale2 = np.sqrt(2*delta_phi**2 / (delta_V+1e-100))
        return max(rscale1, rscale2)
        
    _exactSolution_rval = namedtuple("exactSolution_rval", "phi dphi")
    def exactSolution(self, r, phi0, dV, d2V):
        R"""
        Find `phi(r)` given `phi(r=0)`, assuming a quadratic potential.
        
        Parameters
        ----------
        r : float
            The radius at which the solution should be calculated.
        phi0 : float
            The field at `r=0`.
        dV, d2V : float
            The potential's first and second derivatives evaluated at `phi0`.
        
        Returns
        -------
        phi, dphi : float
            The field and its derivative evaluated at `r`.
        
        Notes
        -----
        
        If the potential at the point :math:`\phi_0` is a simple quadratic, the
        solution to the instanton equation of motion can be determined exactly.
        The non-singular solution to 

        .. math::
          \frac{d^2\phi}{dr^2} + \frac{\alpha}{r}\frac{d\phi}{dr} =
          V'(\phi_0) + V''(\phi_0) (\phi-\phi_0)
        
        is
        
        .. math::
          \phi(r)-\phi_0 = \frac{V'}{V''}\left[
          \Gamma(\nu+1)\left(\frac{\beta r}{2}\right)^{-\nu} I_\nu(\beta r) - 1
          \right]
          
        where :math:`\nu = \frac{\alpha-1}{2}`, :math:`I_\nu` is the modified
        Bessel function, and :math:`\beta^2 = V''(\phi_0) > 0`. If instead 
        :math:`-\beta^2 = V''(\phi_0) < 0`, the solution is the same but with 
        :math:`I_\nu \rightarrow J_\nu`.

        """
        beta = np.sqrt(abs(d2V))
        beta_r = beta*r
        nu = 0.5 * (self.alpha - 1)
        gamma = special.gamma # Gamma function
        iv, jv = special.iv, special.jv # (modified) Bessel function
        if beta_r < 1e-2:
            # Use a small-r approximation for the Bessel function.
            s = +1 if d2V > 0 else -1
            phi = 0.0
            dphi = 0.0
            for k in xrange(1,4):
                _ = (0.5*beta_r)**(2*k-2) * s**k / (gamma(k+1)*gamma(k+1+nu))
                phi += _
                dphi += _ * (2*k)
            phi *= 0.25 * gamma(nu+1) * r**2 * dV * s
            dphi *= 0.25 * gamma(nu+1) * r * dV * s 
            phi += phi0
        elif d2V > 0:
            import warnings
            # If beta_r is very large, this will throw off overflow and divide
            # by zero errors in iv(). It will return np.inf though, which is
            # what we want. Just ignore the warnings.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phi = (gamma(nu+1)*(0.5*beta_r)**-nu *iv(nu, beta_r)-1) * dV/d2V
                dphi = -nu*((0.5*beta_r)**-nu / r) * iv(nu, beta_r)
                dphi += (0.5*beta_r)**-nu * 0.5*beta \
                        * (iv(nu-1, beta_r)+iv(nu+1, beta_r))
                dphi *= gamma(nu+1) * dV/d2V 
                phi += phi0
        else:
            phi = (gamma(nu+1)*(0.5*beta_r)**-nu * jv(nu, beta_r) - 1) * dV/d2V
            dphi = -nu*((0.5*beta_r)**-nu / r) * jv(nu, beta_r)
            dphi += (0.5*beta_r)**-nu * 0.5*beta \
                    * (jv(nu-1, beta_r)-jv(nu+1, beta_r))
            dphi *= gamma(nu+1) * dV/d2V 
            phi += phi0
        return self._exactSolution_rval(phi, dphi)
        
    _initialConditions_rval = namedtuple(
        "initialConditions_rval", "r0 phi dphi")
    def initialConditions(self, delta_phi0, rmin, delta_phi_cutoff):
        R"""
        Finds the initial conditions for integration.
        
        The instanton equations of motion are singular at `r=0`, so we
        need to start the integration at some larger radius. This
        function finds the value `r0` such that `phi(r0) = phi_cutoff`.
        If there is no such value, it returns the intial conditions at `rmin`.
                
        Parameters
        ----------
        delta_phi0 : float
            `delta_phi0 = phi(r=0) - phi_absMin`
        rmin : float
            The smallest acceptable radius at which to start integration.
        delta_phi_cutoff : float
            The desired value for `phi(r0)`. 
            `delta_phi_cutoff = phi(r0) - phi_absMin`.
          
        Returns
        -------
        r0, phi, dphi : float
            The initial radius and the field and its derivative at that radius.
        
        Notes
        -----
        The field values are calculated using :func:`exactSolution`.
        """        
        phi0 = self.phi_absMin + delta_phi0
        dV = self.dV_from_absMin(delta_phi0)
        d2V = self.d2V(phi0)
        phi_r0, dphi_r0 = self.exactSolution(rmin, phi0, dV, d2V)
        if abs(phi_r0 - self.phi_absMin) > abs(delta_phi_cutoff):
            # The initial conditions at rmin work. Stop here.
            return self._initialConditions_rval(rmin, phi_r0, dphi_r0)
        if np.sign(dphi_r0) != np.sign(delta_phi0):
            # The field is evolving in the wrong direction.
            # Increasing r0 won't increase |delta_phi_r0|/
            return rmin, phi_r0, dphi_r0 
        # Find the smallest r0 such that delta_phi_r0 > delta_phi_cutoff
        r = rmin
        while np.isfinite(r):
            rlast = r
            r *= 10
            phi, dphi = self.exactSolution(r, phi0, dV, d2V)
            if abs(phi - self.phi_absMin) > abs(delta_phi_cutoff):
                break
        # Now find where phi - self.phi_absMin = delta_phi_cutoff exactly
        def deltaPhiDiff(r_):
            p = self.exactSolution(r_, phi0, dV, d2V)[0]
            return abs(p - self.phi_absMin) - abs(delta_phi_cutoff)
        r0 = optimize.brentq(deltaPhiDiff, rlast, r, disp=False)
        phi_r0, dphi_r0 = self.exactSolution(r0, phi0, dV, d2V)
        return self._initialConditions_rval(r0, phi_r0, dphi_r0)
    
    def equationOfMotion(self, y, r):
        """
        Used to integrate the bubble wall. 
        """
        return np.array([y[1], self.dV(y[0])-self.alpha*y[1]/r])

    _integrateProfile_rval = namedtuple(
        "integrateProfile_rval", "r y convergence_type")
    def integrateProfile(self, r0, y0, dr0, 
                         epsfrac, epsabs, drmin, rmax, *eqn_args):
        R"""
        Integrate the bubble wall equation:
        
        .. math::
          \frac{d^2\phi}{dr^2} + \frac{\alpha}{r}\frac{d\phi}{dr} =
          \frac{dV}{d\phi}.
          
        The integration will stop when it either overshoots or undershoots
        the false vacuum minimum, or when it converges upon the false vacuum
        minimum.

        Parameters
        ----------
        r0 : float 
            The starting radius for the integration.
        y0 : array_like
            The starting values [phi(r0), dphi(r0)].
        dr0 : float
            The starting integration stepsize.
        epsfrac, epsabs : float
            The error tolerances used for integration. This is fed into 
            :func:`helper_functions.rkqs` and is used to test for convergence.
        drmin : float 
            The minimum allowed value of `dr` before raising an error.
        rmax : float
            The maximum allowed value of `r-r0` before raising an error.
        eqn_args : tuple
            Extra arguments to pass to :func:`equationOfMotion`. Useful for
            subclasses.
          
        Returns
        -------
        r : float
            The final radius.
        y : array_like
            The final field values [phi, dphi]
        convergence_type : str
            Either 'overshoot', 'undershoot', or 'converged'.
        
        Raises
        ------
        helper_functions.IntegrationError
        """
        dr = dr0
        # dY is the ODE that we use
        def dY(y,r,args=eqn_args): 
            return self.equationOfMotion(y,r,*args)
        dydr0 = dY(y0, r0)
        ysign = np.sign(y0[0]-self.phi_metaMin) 
            # positive means we're heading down, negative means heading up.
        rmax += r0
        
        i = 1
        convergence_type = None
        while True:
            dy, dr, drnext = rkqs(y0, dydr0, r0, dY, dr, epsfrac, epsabs)
            r1 = r0 + dr
            y1 = y0 + dy
            dydr1 = dY(y1,r1)
        
            # Check for completion
            if (r1 > rmax):
                raise IntegrationError("r > rmax")
            elif (dr < drmin):
                raise IntegrationError("dr < drmin")
            elif( (abs(y1 - np.array([self.phi_metaMin,0])) < 3*epsabs).all() ):
                r,y = r1,y1
                convergence_type = "converged"
                break
                
            elif( y1[1]*ysign > 0 or (y1[0]-self.phi_metaMin)*ysign < 0 ):
                f = cubicInterpFunction(y0, dr*dydr0, y1, dr*dydr1)
                if(y1[1]*ysign > 0):
                    # Extrapolate to where dphi(r) = 0
                    x = optimize.brentq(lambda x: f(x)[1], 0, 1 )
                    convergence_type = "undershoot"
                else:
                    # Extrapolate to where phi(r) = phi_metaMin
                    x = optimize.brentq(lambda x: f(x)[0]-self.phi_metaMin, 0,1)
                    convergence_type = "overshoot"
                r = r0 + dr*x
                y = f(x)
                break
            # Advance the integration variables
            r0,y0,dydr0 = r1,y1,dydr1
            dr = drnext
        # Check convergence for a second time. 
        # The extrapolation in overshoot/undershoot might have gotten us within
        # the acceptable error.
        if (abs(y - np.array([self.phi_metaMin,0])) < 3*epsabs).all():
            convergence_type = "converged"
        return self._integrateProfile_rval(r, y, convergence_type)
        
    profile_rval = namedtuple("Profile1D", "R Phi dPhi Rerr")
    def integrateAndSaveProfile(self, R, y0, dr, 
                                epsfrac, epsabs,drmin, *eqn_args):
        """
        Integrate the bubble profile, saving the output in an array.
        
        Parameters
        ----------
        R: array_like 
            The array of points at which we want to save the profile.
        y0 : float 
            The starting values [phi(r0), dphi(r0)].
        dr : float
            Starting stepsize.
        epsfrac, epsabs : float 
            The error tolerances used for integration. This
            is fed into :func:`helper_functions.rkqs`.
        drmin : float
            The smallest allowed stepsize.
        eqn_args : tuple
            Extra arguments to pass to :func:`equationOfMotion`. Useful for
            subclasses.
 
        Returns
        -------
        R, Phi, dPhi : array_like 
            Radii and field values which make up the bubble profile.
        Rerr : float or None
            The first value of `r` at which ``dr < drmin``, or `None` if
            ``dr >= drmin`` always.

        Notes
        -----
        Subclasses can use this function without overriding it even if the
        subclass uses more fields/values in its equation of motion (i.e., 
        ``len(y0) > 2``). This is accomplished by setting the class variable
        `profile_rval` to a different named tuple type with more than four
        inputs. The first three should always be *R, Phi, dPhi*, and the last
        one should be *Rerr*, but additional values can be stuck in between.
        """
        N = len(R)
        R, r0 = np.array(R), R[0]
        Yout = np.zeros((N,len(y0)))
        Yout[0] = y0
        # dY is the ODE that we use
        def dY(y,r,args=eqn_args): 
            return self.equationOfMotion(y,r,*args)
        dydr0 = dY(y0, r0)
        Rerr = None
                
        i = 1
        while i < N:
            dy, dr, drnext = rkqs(y0, dydr0, r0, dY, dr, epsfrac, epsabs)
            if (dr >= drmin):    
                r1 = r0 + dr
                y1 = y0 + dy
            else:
                y1 = y0 + dy*drmin/dr
                dr = drnext = drmin
                r1 = r0 + dr
                if Rerr is not None: Rerr = r1
            dydr1 = dY(y1,r1)
            # Fill the arrays, if necessary
            if (r0 < R[i] <= r1):
                f = cubicInterpFunction(y0, dr*dydr0, y1, dr*dydr1)
                while (i < N and r0 < R[i] <= r1):
                    x = (R[i]-r0)/dr
                    Yout[i] = f(x)
                    i += 1
            
            # Advance the integration variables
            r0,y0,dydr0 = r1,y1,dydr1
            dr = drnext
        
        rval = (R,)+tuple(Yout.T)+eqn_args+(Rerr,)
        return self.profile_rval(*rval)
        
    def findProfile(self, xguess=None, xtol=1e-4, phitol=1e-4, 
                    thinCutoff=.01, npoints=500, rmin=1e-4, rmax=1e4,
                    max_interior_pts=None):
        R"""
        Calculate the bubble profile by iteratively over/undershooting.
        
        This will call :func:`integrateProfile` many times, trying to find
        the correct initial condition `phi(r=0)` such that the field ends up
        in the metastable vacuum at infinity. Once the correct initial
        condition is found, it calls :func:`integrateAndSaveProfile` to find
        the profile along the length of the wall.
        
        Parameters
        ----------
        xguess : float, optional
            The initial guess for `x`. If `None`, `xguess` is set such
            that ``phi_guess = self.phi_bar``.
        xtol : float, optional
            Target accuracy in `x`.
        phitol : float, optional
            Fractional error tolerance in integration.
        thinCutoff : float, optional
            Equal to `delta_phi_cutoff / (phi_metaMin - phi_absMin)`, where
            `delta_phi_cutoff` is used  in :func:`initialConditions`.
        npoints : int
            Number of points to return in the profile.
        rmin : float 
            Relative to ``self.rscale``. Sets the smallest starting
            radius, the starting stepsize, and the smallest allowed stepsize 
            (``0.01*rmin``).
        rmax : float
            Relative ``self.rscale``. Sets the maximum allowed integration
            distance.
        max_interior_pts : int
            Maximum number of points to place between ``r=0`` and the start of
            integration. If None, ``max_interior_pts=npoints/2``. If zero, no
            points are added to the bubble interior.

        Returns
        -------
        R, Phi, dPhi : array_like 
            Radii and field values which make up the bubble profile. Note that
            `R[0]` can be much bigger than zero for thin-walled bubbles.
        Rerr : float or None
            The first value of `r` at which ``dr < drmin``, or `None` if
            ``dr >= drmin`` always.
        
        Notes
        -----
        For very thin-walled bubbles, the initially value of `phi` can be
        extremely close to the stable minimum and small variations in `phi`
        can cause large variations in the integration. Rather than varying 
        `phi(r=0)` directly, it is easier to vary a parameter `x` defined by
        
        .. math::
           \phi(r=0) = \phi_{\rm absMin} 
           + e^{-x}(\phi_{\rm metaMin}-\phi_{\rm absMin})
           
        This way, `phi = phi_metaMin` when `x` is zero and 
        `phi = phi_absMin` when `x` is  infinity.
        """
        # Set x parameters
        xmin = xtol*10
        xmax = np.inf
        if xguess is not None:
            x = xguess
        else:
            x = -np.log(abs((self.phi_bar-self.phi_absMin) / 
                            (self.phi_metaMin-self.phi_absMin)))
        xincrease = 5.0 
            # The relative amount to increase x by if there is no upper bound.
        # --
        # Set r parameters
        rmin *= self.rscale
        dr0 = rmin
        drmin = 0.01*rmin
        rmax *= self.rscale
        # --
        # Set the phi parameters
        delta_phi = self.phi_metaMin - self.phi_absMin
        epsabs = abs(np.array([delta_phi, delta_phi/self.rscale])*phitol)
        epsfrac = np.array([1,1]) * phitol
        delta_phi_cutoff = thinCutoff * delta_phi
            # The sign for delta_phi_cutoff doesn't matter
        # --
        integration_args = (dr0, epsfrac, epsabs, drmin, rmax)
        rf = None
        while True:
            delta_phi0 = np.exp(-x)*delta_phi
            # r0, phi0, dphi0 = self.initialConditions(x, rmin, thinCutoff)
            r0_, phi0, dphi0 = self.initialConditions(
                                    delta_phi0, rmin, delta_phi_cutoff)
            if not np.isfinite(r0_) or not np.isfinite(x):
                # Use the last finite values instead
                # (assuming there are such values)
                assert rf is not None, "Failed to retrieve initial "\
                    "conditions on the first try."
                break
            r0 = r0_
            y0 = np.array([phi0, dphi0])
            rf, yf, ctype = self.integrateProfile(r0, y0, *integration_args)
            # Check for overshoot, undershoot
            if ctype == "converged":
                break
            elif ctype == "undershoot": # x is too low
                xmin = x
                x = x*xincrease if xmax == np.inf else .5*(xmin+xmax)
            elif ctype == "overshoot": # x is too high
                xmax = x
                x = .5*(xmin+xmax)
            # Check if we've reached xtol
            if (xmax-xmin) < xtol:
                break
                    
        # Integrate a second time, this time getting the points along the way
        R = np.linspace(r0, rf, npoints)
        profile = self.integrateAndSaveProfile(R, y0, dr0, 
                                               epsfrac, epsabs, drmin)
        # Make points interior to the bubble.
        if max_interior_pts is None:
            max_interior_pts = len(R) // 2
        if max_interior_pts > 0:
            dx0 = R[1]-R[0]
            if R[0] / dx0 <= max_interior_pts:
                n = np.ceil(R[0]/dx0)
                R_int = np.linspace(0, R[0], n+1)[:-1]
            else:
                n = max_interior_pts
                # R[0] = dx0 * (n + a*n*(n+1)/2)
                a = (R[0]/dx0 - n) * 2/(n*(n+1))
                N = np.arange(1,n+1)[::-1]
                R_int = R[0] - dx0*(N + 0.5*a*N*(N+1))
                R_int[0] = 0.0 # enforce this exactly
            Phi_int = np.empty_like(R_int)
            dPhi_int = np.empty_like(R_int)
            Phi_int[0] = self.phi_absMin + delta_phi0
            dPhi_int[0] = 0.0
            dV = self.dV_from_absMin(delta_phi0)
            d2V = self.d2V(Phi_int[0])
            for i in xrange(1,len(R_int)):
                Phi_int[i], dPhi_int[i] = self.exactSolution(
                                               R_int[i], Phi_int[0], dV, d2V)
            R = np.append(R_int, profile.R)
            Phi = np.append(Phi_int, profile.Phi)
            dPhi = np.append(dPhi_int, profile.dPhi)
            profile = self.profile_rval(R,Phi,dPhi, profile.Rerr)
        return profile
    
    def findAction(self, profile):
        R"""
        Calculate the Euclidean action for the instanton:
        
        .. math::
          S = \int [(d\phi/dr)^2 + V(\phi)] r^\alpha dr d\Omega_\alpha 
                  
        Arguments
        ---------
        profile 
            Output from :func:`findProfile()`.
        
        Returns
        -------
        float
            The Euclidean action.
        """
        r, phi, dphi = profile.R, profile.Phi, profile.dPhi
        # Find the area of an n-sphere (alpha=n):
        d = self.alpha+1 # Number of dimensions in the integration
        area = r**self.alpha * 2*np.pi**(d*.5)/special.gamma(d*.5) 
        # And integrate the profile
        integrand = 0.5 * dphi**2 + self.V(phi) - self.V(self.phi_metaMin)
        integrand *= area
        S = integrate.simps(integrand, r)
        # Find the bulk term in the bubble interior
        volume = r[0]**d * np.pi**(d*.5)/special.gamma(d*.5 + 1)
        S += volume * (self.V(phi[0]) - self.V(self.phi_metaMin))
        return S
        
    def evenlySpacedPhi(self, phi, dphi, npoints=100, k=3, fixAbs=True):
        """
        This method takes `phi` and `dphi` as input, which will probably
        come from the output of :func:`findProfile`, and returns a different
        set of arrays `phi2` and `dphi2` such that `phi2` is linearly spaced
        (instead of `r`).
        
        Parameters
        ----------
        phi, dphi : array_like
        npoints : int
            The number of points to output.
        k : int
            The degree of spline fitting. ``k=1`` means linear interpolation.
        fixAbs : bool
            If true, make phi go all the way to `phi_absMin`.
        """
        if fixAbs == True:
            phi = np.append(self.phi_absMin, np.append(phi, self.phi_metaMin))
            dphi = np.append(0.0, np.append(dphi, 0.0))
        else:
            phi = np.append(phi, self.phi_metaMin)
            dphi = np.append(dphi, 0.0)
        # Make sure that phi is increasing everywhere 
        # (this is uglier than it ought to be)
        i = helper_functions.monotonicIndices(phi)
        # Now do the interpolation
        tck = interpolate.splrep(phi[i], dphi[i], k=k)
        if fixAbs:
            p = np.linspace(self.phi_absMin, self.phi_metaMin, npoints)
        else:
            p = np.linspace(phi[i][0], self.phi_metaMin, npoints)
        dp = interpolate.splev(p, tck)
        return p, dp

class WallWithConstFriction(SingleFieldInstanton):
    """
    This class solves a modified version of the instanton equations of motion
    with a *constant* friction term.

    This may be useful if one wants to estimate the shape of a bubble wall 
    moving through a plasma. It will, however, be a rough estimate since a real
    friction force would most likely be field-dependent.
    """
    def findRScale(self):
        R"""
        Find the characteristic length scale for tunneling over the potential
        barrier.
        
        Since for this class the tunneling solution always goes between the two
        minima, we want to take the overall shape between the two (not just
        at the top of the barrier) to set the radial scale. This finds the scale
        by fitting a simple quadratic to the potential. 

        Raises
        ------
        PotentialError
            when the barrier is non-existent.
        """
        pA = self.phi_absMin
        pB = 0.5 * (self.phi_bar + self.phi_metaMin)
        pC = self.phi_metaMin
        yA = self.V(pA)
        yB = self.V(pB)
        yC = self.V(pC)
        # Let lmda be the quadratic coefficient that will fit these 3 points
        lmda = 2*((yA-yB)/(pA-pB) - (yB-yC)/(pB-pC)) / (pC-pA)
        if lmda <= 0.0:
            raise PotentialError("Cannot fit the potential to a negative "
                                 "quadratic.", "no barrier")
        omega = np.sqrt(lmda) # frequency of oscillations
        return np.pi / omega

    def initialConditions(self, F, phi0_rel=1e-3):
        R"""
        Get the initial conditions for integration.

        Parameters
        ----------
        F : float
            Magnitude of the friction term.
        phi0_rel : float
            The initial value for the field, relative to the two minima
            with 0.0 being at `phi_absMin` and 1.0 being at `phi_metaMin`
            (should be close to 0.0).
           
        Returns
        -------
        r0, phi, dphi : float
            The initial radius and the field and its derivative at that radius.
        
        Notes
        -----
        Approximate the equation of motion near the minimum as 

        .. math::
        
            \phi'' + F \phi' = (\phi-\phi_{absMin}) \frac{d^2V}{d\phi^2}
        
        which has solution 

        .. math::

            \phi(r) = (\phi_0-\phi_{absMin}) e^{kr} + \phi_{absMin}
        
        where :math:`k = (\sqrt{F^2 + 4 V''} - F) / 2`.
        """
        k = 0.5 * ( np.sqrt(F*F+4*self.d2V(self.phi_absMin)) - F )
        r0 = 0.0
        phi0 = self.phi_absMin + phi0_rel * (self.phi_metaMin-self.phi_absMin)
        dphi0 = k * (phi0 - self.phi_absMin)
        return self._initialConditions_rval(r0, phi0, dphi0)

    def equationOfMotion(self, y, r, F):
        """
        Used to integrate the bubble wall. 
        """
        return np.array([y[1], self.dV(y[0])-F*y[1]])

    profile_rval = namedtuple("Profile1D", "R Phi dPhi F Rerr")

    def findProfile(self, Fguess=None, Ftol=1e-4, phitol=1e-4, 
                    npoints=500, rmin=1e-4, rmax=1e4, phi0_rel=1e-3):
        R"""
        Calculate the bubble profile by iteratively over/undershooting.
        
        Parameters
        ----------
        Fguess : float, optional
            The initial guess for `F`. If `None`, `Fguess` is calculated from
            `self.rscale`.
        Ftol : float, optional
            Target accuracy in `F`, relative to `Fguess`.
        phitol : float, optional
            Fractional error tolerance in integration.
        npoints : int
            Number of points to return in the profile.
        rmin : float 
            Relative to ``self.rscale``. Sets the smallest starting
            radius, the starting stepsize, and the smallest allowed stepsize 
            (``0.01*rmin``).
        rmax : float
            Relative ``self.rscale``. Sets the maximum allowed integration
            distance.
        phi0_rel : float
            Passed to :func:`initialConditions`.

        Returns
        -------
        R, Phi, dPhi : array_like 
            Radii and field values which make up the bubble profile. Note that
            `R[0]` can be much bigger than zero for thin-walled bubbles.
        Rerr : float or None
            The first value of `r` at which ``dr < drmin``, or `None` if
            ``dr >= drmin`` always.
        """
        # Set r parameters
        rmin *= self.rscale
        dr0 = rmin
        drmin = 0.01*rmin
        rmax *= self.rscale
        # --
        # Set the phi parameters
        delta_phi = self.phi_metaMin - self.phi_absMin
        epsabs = abs(np.array([delta_phi, delta_phi/self.rscale])*phitol)
        epsfrac = np.array([1,1]) * phitol
        # --
        # Set F parameters
        Fmin = 0.0
        Fmax = np.inf
        if Fguess is not None:
            F = Fguess
        else:
            # Find F from conservation of energy
            # (total work done to slow down the field)
            Delta_V = self.V(self.phi_metaMin) - self.V(self.phi_absMin)
            F = Delta_V * self.rscale / delta_phi**2
        Ftol *= F
        Fincrease = 5.0 
            # The relative amount to increase F by if there is no upper bound.
        # --
        integration_args = [dr0, epsfrac, epsabs, drmin, rmax, F]
        rf = None
        while True:
            r0, phi0, dphi0 = self.initialConditions(F, phi0_rel)
            y0 = np.array([phi0, dphi0])
            integration_args[-1] = F
            rf, yf, ctype = self.integrateProfile(r0, y0, *integration_args)
            # Check for overshoot, undershoot
            if ctype == "converged":
                break
            elif ctype == "undershoot": # F is too high
                Fmax = F
                F = F/Fincrease if Fmin == 0.0 else .5*(Fmin+Fmax)
            elif ctype == "overshoot": # F is too low
                Fmin = F
                F = F*Fincrease if Fmax == np.inf else .5*(Fmin+Fmax)
            # Check if we've reached xtol
            if (Fmax-Fmin) < Ftol:
                break
                    
        # Integrate a second time, this time getting the points along the way
        R = np.linspace(r0, rf, npoints)
        profile = self.integrateAndSaveProfile(R, y0, dr0, 
                                               epsfrac, epsabs, drmin, F)
        return profile

    def findAction(self, profile):
        """
        Always returns `np.inf`.
        """
        return np.inf





