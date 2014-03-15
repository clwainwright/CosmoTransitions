"""
The primary task of the generic_potential module is to define the 
:class:`generic_potential` class, from which realistic scalar field models can
straightforwardly be constructed. The most important part of any such model is,
appropiately, the potential function and its gradient. This module is not
necessary to define a potential, but it does make the process somewhat simpler
by automatically calculating one-loop effects from a model-specific mass
spectrum, constructing numerical derivative functions, providing a 
simplified interface to the :mod:`transitionFinder` module, and providing
several methods for plotting the potential and its phases.
"""

__version__ = "2.0a1"


import numpy as np
from scipy import optimize, interpolate
import transitionFinder, pathDeformation
from finiteT import Jb_spline as Jb
from finiteT import Jf_spline as Jf
import helper_functions

class generic_potential():
    """
    An abstract class from which one can easily create finite-temperature
    effective potentials.
    
    This class acts as the skeleton around which different scalar field models
    can be formed. At a bare minimum, subclasses must implement :func:`init`,
    :func:`V0`, and :func:`boson_massSq`. Subclasses will also likely implement
    :func:`fermion_massSq` and :func:`approxZeroTMin`. Once the tree-level 
    potential and particle spectrum are defined, the one-loop zero-temperature
    potential (using MS-bar renormalization) and finite-temperature potential
    can be used without any further modification.
    
    The `__init__` function performs initialization specific for this abstract
    class. Subclasses should either override this initialization *but make sure
    to call the parent implementation*, or, more simply, override the
    :func:`init` method. In the base implementation, the former calls the latter
    and the latter does nothing. At a bare minimum, subclasses must set the
    `Ndim` parameter to the number of dynamic field dimensions in the model. 
    
    One of the main jobs of this class is to provide an easy interface for
    calculating the phase structure and phase transitions. These are given by
    the methods :func:`getPhases`, :func:`calcTcTrans`, and 
    :func:`findAllTransitions`.

    The following attributes can (and should!) be set during initialiation:
    
    Attributes
    ----------
    Ndim : int
        The number of dynamic field dimensions in the model. This *must* be
        overridden by subclasses during initialization.
    x_eps : float
        The epsilon to use in brute-force evalutations of the gradient and
        for the second derivatives. May be overridden by subclasses; 
        defaults to 0.001.
    T_eps : float
        The epsilon to use in brute-force evalutations of the temperature 
        derivative. May be overridden by subclasses; defaults to 0.001.
    deriv_order : int
        Sets the order to which finite difference derivatives are calculated.
        Must be 2 or 4. May be overridden by subclasses; defaults to 4.
    renormScaleSq : float
        The square of the renormalization scale to use in the MS-bar one-loop 
        zero-temp potential. May be overridden by subclasses; 
        defaults to 1000.0**2.
    Tmax : float
        The maximum temperature to which minima should be followed. No
        transitions are calculated above this temperature. This is also used
        as the overall temperature scale in :func:`getPhases`.
        May be overridden by subclasses; defaults to 1000.0.
    num_boson_dof : int or None
        Total number of bosonic degrees of freedom, including radiation.
        This is used to add a field-independent but temperature-dependent 
        contribution to the effective potential. It does not affect the relative
        pressure or energy density between phases, so it does not affect the
        critical or nucleation temperatures. If None, the total number of
        degrees of freedom will be taken directly from :meth:`boson_massSq`.
    num_fermion_dof : int or None
        Total number of fermionic degrees of freedom, including radiation.
        If None, the total number of degrees of freedom will be taken 
        directly from :meth:`fermion_massSq`.
    """
    def __init__(self, *args, **dargs):
        self.Ndim = 0
        self.x_eps = .001
        self.T_eps = .001
        self.deriv_order = 4
        self.renormScaleSq = 1000.**2
        self.Tmax = 1e3

        self.num_boson_dof = self.num_fermion_dof = None
        
        self.phases = self.transitions = None # These get set by getPhases
        self.TcTrans = None # Set by calcTcTrans()
        self.TnTrans = None # Set by calcFullTrans()
                
        self.init(*args, **dargs)
        
        if self.Ndim <= 0:
            raise ValueError("The number of dimensions in the potential must "
                             "be at least 1.")
        
    def init(self, *args, **dargs):
        """
        Subclasses should override this method (not __init__) to do all model
        initialization. At a bare minimum, subclasses need to specify the number
        of dimensions in the potential with ``self.Ndim``.
        """
        pass
        
    # EFFECTIVE POTENTIAL CALCULATIONS -----------------------
        
    def V0(self, X):
        """
        The tree-level potential. Should be overridden by subclasses.
        
        The input X can either be a single point (with length `Ndim`), or an
        arbitrarily shaped array of points (with a last axis again having shape
        `Ndim`). Subclass implementations should be able to handle either case.
        If the input is a single point, the output should be scalar. If the
        input is an array of points, the output should be an array with the same
        shape (except for the last axis with shape `Ndim`).
        """
        return X[...,0]*0
        
    def boson_massSq(self, X, T):
        """
        Calculate the boson particle spectrum. Should be overridden by 
        subclasses.
        
        Parameters
        ----------
        X : array_like
            Field value(s). 
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature at which to calculate the boson masses. Can be used
            for including thermal mass corrrections. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).
            
        Returns
        -------
        massSq : array_like
            A list of the boson particle masses at each input point `X`. The
            shape should be such that 
            ``massSq.shape == (X[...,0]*T).shape + (Nbosons,)``.
            That is, the particle index is the *last* index in the output array
            if the input array(s) are multidimensional.
        degrees_of_freedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            length `Ndim`.
        c : float or array_like
            A constant used in the one-loop zero-temperature effective
            potential. If an array, it should have length `Ndim`. Generally
            `c = 1/2` for gauge boson transverse modes, and `c = 3/2` for all
            other bosons.
        """
        # The following is an example placeholder which has the correct output
        # shape. Since dof is zero, it does not contribute to the potential.
        Nboson = 2
        phi1 = X[...,0]
        #phi2 = X[...,1] # Comment out so that the placeholder doesn't 
                         # raise an exception for Ndim < 2.
        m1 = .5 * phi1**2 + .2 * T**2 # First boson mass
        m2 = .6 * phi1**2 # Second boson mass, no thermal mass correction
        massSq = np.empty(m1.shape + (Nboson,)) # Important to make sure that
            # the shape comes from m1 and not m2, since the addition of the
            # thermal mass correction could change the output shape (if, for
            # example, T is an array and X is a single point).
        massSq[...,0] = m1
        massSq[...,1] = m2
        dof = np.array([0.,0.])
        c = np.array([0.5, 1.5]) 
        return massSq, dof, c
        
    def fermion_massSq(self, X):
        """
        Calculate the fermion particle spectrum. Should be overridden by 
        subclasses.
        
        Parameters
        ----------
        X : array_like
            Field value(s). 
            Either a single point (with length `Ndim`), or an array of points.

        Returns
        -------
        massSq : array_like
            A list of the fermion particle masses at each input point `X`. The
            shape should be such that  ``massSq.shape == (X[...,0]).shape``.
            That is, the particle index is the *last* index in the output array
            if the input array(s) are multidimensional.
        degrees_of_freedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            length `Ndim`.

        Notes
        -----
        Unlike :func:`boson_massSq`, no constant `c` is needed since it is
        assumed to be `c = 3/2` for all fermions. Also, no thermal mass
        corrections are needed.
        """
        # The following is an example placeholder which has the correct output
        # shape. Since dof is zero, it does not contribute to the potential.
        Nfermions = 2
        phi1 = X[...,0]
        #phi2 = X[...,1] # Comment out so that the placeholder doesn't 
                         # raise an exception for Ndim < 2.
        m1 = .5 * phi1**2 # First fermion mass
        m2 = .6 * phi1**2 # Second fermion mass
        massSq = np.empty(m1.shape + (Nfermions,))
        massSq[...,0] = m1
        massSq[...,1] = m2
        dof = np.array([0.,0.])
        return massSq, dof
        
    def V1(self, bosons, fermions):
        """
        The one-loop corrections to the zero-temperature potential
        using MS-bar renormalization.
        
        This is generally not called directly, but is instead used by 
        :func:`Vtot`.
        """
        # This does not need to be overridden.
        m2, n, c = bosons
        y = np.sum(n*m2*m2 * (np.log(np.abs(m2/self.renormScaleSq) + 1e-100)
                              - c), axis=-1)
        m2, n = fermions
        c = 1.5
        y -= np.sum(n*m2*m2 * (np.log(np.abs(m2/self.renormScaleSq) + 1e-100) 
                               - c), axis=-1)
        return y/(64*np.pi*np.pi)
        
    def V1T(self, bosons, fermions, T, include_radiation=True):
        """
        The one-loop finite-temperature potential.

        This is generally not called directly, but is instead used by 
        :func:`Vtot`. 

        Note
        ----
        The `Jf` and `Jb` functions used here are
        aliases for :func:`finiteT.Jf_spline` and :func:`finiteT.Jb_spline`, 
        each of which accept mass over temperature *squared* as inputs 
        (this allows for negative mass-squared values, which I take to be the
        real part defining integrals.

        .. todo::
            Implement new versions of Jf and Jb that return zero when m=0, only
            adding in the field-independent piece later if 
            ``include_radiation == True``. This should reduce floating point
            errors when taking derivatives at very high temperature, where
            the field-independent contribution is much larger than the 
            field-dependent contribution.
        """
        # This does not need to be overridden.
        T2 = (T*T)[..., np.newaxis] + 1e-100
             # the 1e-100 is to avoid divide by zero errors
        T4 = T*T*T*T
        m2, nb, c = bosons
        y = np.sum( nb*Jb(m2/T2), axis=-1)
        m2, nf = fermions
        y += np.sum( nf*Jf(m2/T2), axis=-1) 
        if include_radiation:
            if self.num_boson_dof is not None:
                nb = self.num_boson_dof - np.sum(nb)
                y -= nb * np.pi**4 / 45.
            if self.num_fermion_dof is not None:
                nf = self.num_fermion_dof - np.sum(nf)
                y -= nf * 7*np.pi**4 / 360.
        return y*T4/(2*np.pi*np.pi)

    def V1T_from_X(self, X, T, include_radiation=True):
        """
        Calculates the mass matrix and resulting one-loop finite-T potential.

        Useful when calculate temperature derivatives, when the zero-temperature
        contributions don't matter.
        """
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        y = self.V1T(bosons, fermions, T, include_radiation)
        return y
        
    def Vtot(self, X, T, include_radiation=True):
        """
        The total finite temperature effective potential.
        
        Parameters
        ----------
        X : array_like
            Field value(s). 
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature. The shapes of `X` and `T`
            should be such that ``X.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``X[...,0]*T`` is a valid operation).
        include_radiation : bool, optional
            Currently not used. 
            Later, if False, this will drop all field-independent radiation
            terms from the effective potential. Useful for calculating
            differences or derivatives.
        """
        T = np.asanyarray(T, dtype=float)
        X = np.asanyarray(X, dtype=float)
        bosons = self.boson_massSq(X,T)
        fermions = self.fermion_massSq(X)
        y = self.V0(X)
        y += self.V1(bosons, fermions)
        y += self.V1T(bosons, fermions, T, include_radiation)
        return y
        
    def DVtot(self, X, T):
        """
        The finite temperature effective potential, but offset
        such that V(0, T) = 0.
        """
        X0 = np.zeros(self.Ndim)
        return self.Vtot(X,T,False) - self.Vtot(X0,T,False)
        
    def gradV(self, X, T):
        """
        Find the gradient of the full effective potential.
        
        This uses :func:`helper_functions.gradientFunction` to calculate the
        gradient using finite differences, with differences
        given by `self.x_eps`. Note that `self.x_eps` is only used directly
        the first time this function is called, so subsequently changing it
        will not have an effect.
        """
        try:
            f = self._gradV
        except:
            # Create the gradient function
            self._gradV = helper_functions.gradientFunction(
                self.Vtot, self.x_eps, self.Ndim, self.deriv_order)
            f = self._gradV
        # Need to add extra axes to T since extra axes get added to X in
        # the helper function.
        T = np.asanyarray(T)[...,np.newaxis,np.newaxis]
        return f(X,T,False)
        
    def dgradV_dT(self, X, T):
        """
        Find the derivative of the gradient with respect to temperature.
        
        This is useful when trying to follow the minima of the potential as they
        move with temperature.
        """
        T_eps = self.T_eps
        try:
            gradVT = self._gradVT
        except:
            # Create the gradient function
            self._gradVT = helper_functions.gradientFunction(
                self.V1T_from_X, self.x_eps, self.Ndim, self.deriv_order)
            gradVT = self._gradVT
        # Need to add extra axes to T since extra axes get added to X in
        # the helper function.
        T = np.asanyarray(T)[...,np.newaxis,np.newaxis]
        assert (self.deriv_order == 2 or self.deriv_order == 4)
        if self.deriv_order == 2:
            y = gradVT(X,T+T_eps,False) - gradVT(X,T-T_eps,False)
            y *= 1./(2*T_eps)
        else:
            y = gradVT(X,T-2*T_eps,False)
            y -= 8*gradVT(X,T-T_eps,False)
            y += 8*gradVT(X,T+T_eps,False)
            y -= gradVT(X,T+2*T_eps,False)
            y *= 1./(12*T_eps)
        return y
        
    def massSqMatrix(self, X):
        """
        Calculate the tree-level mass matrix of the scalar field.
        
        This uses :func:`helper_functions.hessianFunction` to calculate the
        matrix using finite differences, with differences
        given by `self.x_eps`. Note that `self.x_eps` is only used directly
        the first time this function is called, so subsequently changing it
        will not have an effect.        
        
        The resulting matrix will have rank `Ndim`. This function may be useful
        for subclasses in finding the boson particle spectrum.
        """
        try:
            f = self._massSqMatrix
        except:
            # Create the gradient function
            self._massSqMatrix = helper_functions.hessianFunction(
                self.V0, self.x_eps, self.Ndim, self.deriv_order)
            f = self._massSqMatrix
        return f(X)
        
    def d2V(self, X, T):
        """
        Calculates the Hessian (second derivative) matrix for the 
        finite-temperature effective potential.
        
        This uses :func:`helper_functions.hessianFunction` to calculate the
        matrix using finite differences, with differences
        given by `self.x_eps`. Note that `self.x_eps` is only used directly
        the first time this function is called, so subsequently changing it
        will not have an effect.        
        """
        try:
            f = self._d2V
        except:
            # Create the gradient function
            self._d2V = helper_functions.hessianFunction(
                self.Vtot, self.x_eps, self.Ndim, self.deriv_order)
            f = self._d2V
        # Need to add extra axes to T since extra axes get added to X in
        # the helper function.
        T = np.asanyarray(T)[...,np.newaxis]
        return f(X,T, False)

    def energyDensity(self,X,T,include_radiation=True):
        T_eps = self.T_eps
        if self.deriv_order == 2:
            dVdT = self.Vtot(X,T+T_eps, include_radiation) 
            dVdT -= self.Vtot(X,T-T_eps, include_radiation)
            dVdT *= 1./(2*T_eps)
        else:
            dVdT = self.Vtot(X,T-2*T_eps, include_radiation)
            dVdT -= 8*self.Vtot(X,T-T_eps, include_radiation)
            dVdT += 8*self.Vtot(X,T+T_eps, include_radiation)
            dVdT -= self.Vtot(X,T+2*T_eps, include_radiation)
            dVdT *= 1./(12*T_eps)
        V = self.Vtot(X,T, include_radiation)
        return V - T*dVdT
        
    # MINIMIZATION AND TRANSITION ANALYSIS --------------------------------	
        
    def approxZeroTMin(self):
        """
        Returns approximate values of the zero-temperature minima. 
        
        This should be overridden by subclasses, although it is not strictly
        necessary if there is only one minimum at tree level. The precise values
        of the minima will later be found using :func:`scipy.optimize.fmin`.
        
        Returns
        -------
        minima : list
            A list of points of the approximate minima.
        """
        # This should be overridden.
        return [np.ones(self.Ndim)*self.renormScaleSq**.5]
        
    def findMinimum(self, X=None, T=0.0):
        """
        Convenience function for finding the nearest minimum to `X` at 
        temperature `T`.
        """
        if X == None:
            X = self.approxZeroTMin()[0]
        return optimize.fmin(self.Vtot, X, args=(T,), disp=0)
        
    def findT0(self):
        """
        Find the temperature at which the high-T minimum disappears. That is,
        find lowest temperature at which Hessian matrix evaluated at the origin
        has non-negative eigenvalues.
        
        Notes
        -----
        In prior versions of CosmoTransitions, `T0` was used to set the scale
        in :func:`getPhases`. This became problematic when `T0` was zero, so in
        this version `self.Tmax` is used as the scale. This function is now not
        called directly by anything in the core CosmoTransitions package, but
        is left as a convenience for subclasses.
        """
        X = self.findMinimum(np.zeros(self.Ndim), self.Tmax)
        f = lambda T: min(np.linalg.eigvalsh(self.d2V(X,T)))
        if f(0.0) > 0:
            # barrier at T = 0
            T0 = 0.0
        else:
            T0 = optimize.brentq(f, 0.0, self.Tmax)
        return T0
        
    def forbidPhaseCrit(self, X):
        """
        Returns True if a phase at point `X` should be discarded, 
        False otherwise.
        
        The default implementation returns False. Can be overridden by
        subclasses to ignore phases. This is useful if, for example, there is a
        Z2 symmetry in the potential and you don't want to double-count all of
        the phases.
        
        Notes
        -----
        In previous versions of CosmoTransitions, `forbidPhaseCrit` was set to
        None in `__init__`, and then if a subclass needed to forbid some region
        it could set ``self.forbidPhaseCrit = lambda x: ...``. Having this 
        instead be a proper method makes for cleaner code.
        
        The name "forbidPhaseCrit" is supposed to be short for "critera for
        forbidding a phase". Apologies for the name mangling; I'm not sure why
        I originally decided to leave off the "eria" in "criteria", but I should
        leave it as is for easier backwards compatability.
        """
        return False
        
    def getPhases(self,tracingArgs={}):
        """
        Find different phases as functions of temperature

        Parameters
        ----------
        tracingArgs : dict
            Parameters to pass to :func:`transitionFinder.traceMultiMin`.

        Returns
        -------
        dict
            Each item in the returned dictionary is an instance of
            :class:`transitionFinder.Phase`, and each phase is
            identified by a unique key. This value is also stored in
            `self.phases`.
        """
        tstop = self.Tmax
        points = []
        for x0 in self.approxZeroTMin():
            points.append([x0,0.0])
        tracingArgs_ = dict(forbidCrit=self.forbidPhaseCrit)
        tracingArgs_.update(tracingArgs)
        phases = transitionFinder.traceMultiMin(
            self.Vtot, self.dgradV_dT, self.d2V, points,
            tLow=0.0, tHigh=tstop, deltaX_target=100*self.x_eps,
            **tracingArgs_)
        self.phases = phases
        transitionFinder.removeRedundantPhases(
            self.Vtot, phases, self.x_eps*1e-2, self.x_eps*10)
        return self.phases

    def calcTcTrans(self, startHigh=False):
        """
        Runs :func:`transitionFinder.findCriticalTemperatures`, storing the
        result in `self.TcTrans`.

        In addition to the values output by 
        :func:`transitionFinder.findCriticalTemperatures`, this function adds
        the following entries to each transition dictionary:

        - *Delta_rho* : Energy difference between the two phases. Positive
          values mean the high-T phase has more energy.

        Returns
        -------
        self.TcTrans
        """
        if self.phases is None:
            self.getPhases()
        self.TcTrans = transitionFinder.findCriticalTemperatures(
            self.phases, self.Vtot, startHigh)
        for trans in self.TcTrans:
            T = trans['Tcrit']
            xlow = trans['low_vev']
            xhigh = trans['high_vev']
            trans['Delta_rho'] = self.energyDensity(xhigh,T,False) \
                - self.energyDensity(xlow,T,False)
        return self.TcTrans  

    def findAllTransitions(self, tunnelFromPhase_args={}):
        """
        Find all phase transitions up to `self.Tmax`, storing the transitions
        in `self.TnTrans`.

        In addition to the values output by 
        :func:`transitionFinder.tunnelFromPhase`, this function adds
        the following entries to each transition dictionary:

        - *Delta_rho* : Energy difference between the two phases. Positive
          values mean the high-T phase has more energy.
        - *Delta_p* : Pressure difference between the two phases. Should always
          be positive.
        - *crit_trans* : The transition at the critical temperature, or None
          if no critical temperature can be found.
        - *dS_dT* : Derivative of the Euclidean action with respect to
          temperature. NOT IMPLEMENTED YET.

        Parameters
        ----------
        tunnelFromPhase_args : dict
            Parameters to pass to :func:`transitionFinder.tunnelFromPhase`. 

        Returns
        -------
        self.TnTrans
        """
        if self.phases is None:
            self.getPhases()
        self.TnTrans = transitionFinder.findAllTransitions(
            self.phases, self.Vtot, self.gradV, tunnelFromPhase_args)
        # Add in the critical temperature
        if self.TcTrans is None:
            self.calcTcTrans()
        transitionFinder.addCritTempsForFullTransitions(
            self.phases, self.TcTrans, self.TnTrans)
        # Add in Delta_rho, Delta_p
        for trans in self.TnTrans:
            T = trans['Tnuc']
            xlow = trans['low_vev']
            xhigh = trans['high_vev']
            trans['Delta_rho'] = self.energyDensity(xhigh,T,False) \
                - self.energyDensity(xlow,T,False)
            trans['Delta_p'] = self.Vtot(xhigh,T,False) \
                - self.Vtot(xlow,T,False)
        return self.TnTrans     
        
    # PLOTTING ---------------------------------
        
    def plot2d(self, box, T=0, treelevel=False, offset=0, 
               xaxis=0, yaxis=1, n=50, clevs=200, cfrac=.8, **contourParams):
        """
        Makes a countour plot of the potential.       

        Parameters
        ----------
        box : tuple
            The bounding box for the plot, (xlow, xhigh, ylow, yhigh).
        T : float, optional
            The temperature
        offset : array_like
            A constant to add to all coordinates. Especially 
            helpful if Ndim > 2.
        xaxis, yaxis : int, optional
            The integers of the axes that we want to plot.
        n : int
            Number of points evaluated in each direction.
        clevs : int
            Number of contour levels to draw.
        cfrac : float
            The lowest contour is always at ``min(V)``, while the highest is
            at ``min(V) + cfrac*(max(V)-min(V))``. If ``cfrac < 1``, only part
            of the plot will be covered. Useful when the minima are more 
            important to resolve than the maximum.
        contourParams :
            Any extra parameters to be passed to :func:`plt.contour`.

        Note
        ----
        .. todo:: 
            Add an example plot. 
            Make documentation for the other plotting functions.             
        """
        import matplotlib.pyplot as plt
        xmin,xmax,ymin,ymax = box
        X = np.linspace(xmin, xmax, n).reshape(n,1)*np.ones((1,n))
        Y = np.linspace(ymin, ymax, n).reshape(1,n)*np.ones((n,1))
        XY = np.zeros((n,n,self.Ndim))
        XY[...,xaxis], XY[...,yaxis] = X,Y
        XY += offset
        Z = self.V0(XY) if treelevel else self.Vtot(XY,T)
        minZ, maxZ = min(Z.ravel()), max(Z.ravel())
        N = np.linspace(minZ, minZ+(maxZ-minZ)*cfrac, clevs)
        plt.contour(X,Y,Z, N, **contourParams)
        plt.axis(box)
        plt.show()
        
    def plot1d(self, x1, x2, T=0, treelevel=False, subtract=True, n=500, 
               **plotParams):
        import matplotlib.pyplot as plt
        if self.Ndim == 1:
            x = np.linspace(x1,x2,n)
            X = x[:,np.newaxis]
        else:
            dX = np.array(x2)-np.array(x1)
            X = dX*np.linspace(0,1,n)[:,np.newaxis] + x1
            x = np.linspace(0,1,n)*np.sum(dX**2)**.5
        if treelevel:
            y = self.V0(X) - self.V0(X*0) if subtract else self.V0(X)
        else:
            y = self.DVtot(X,T) if subtract else self.Vtot(X, T)
        plt.plot(x,y, **plotParams)
        plt.xlabel(R"$\phi$")
        plt.ylabel(R"$V(\phi)$")
        
    def plotPhasesV(self, useDV=True, **plotArgs):
        import matplotlib.pyplot as plt
        if self.phases == None:
            self.getPhases()
        for key, p in self.phases.iteritems():
            V = self.DVtot(p.X,p.T) if useDV else self.Vtot(p.X,p.T)
            plt.plot(p.T,V,**plotArgs)
        plt.xlabel(R"$T$")
        if useDV:
            plt.ylabel(R"$V[\phi_{min}(T), T] - V(0, T)$")
        else:
            plt.ylabel(R"$V[\phi_{min}(T), T]$")

    def plotPhasesPhi(self, **plotArgs):
        import matplotlib.pyplot as plt
        if self.phases == None:
            self.getPhases()
        for key, p in self.phases.iteritems():
            phi_mag = np.sum( p.X**2, -1 )**.5
            plt.plot(p.T, phi_mag, **plotArgs)
        plt.xlabel(R"$T$")
        plt.ylabel(R"$\phi(T)$")

# END GENERIC_POTENTIAL CLASS ------------------


# FUNCTIONS ON LISTS OF MODEL INSTANCES ---------------

def funcOnModels(f, models):
    """
    If you have a big array of models, this function allows you
    to extract big arrays of model outputs. For example, suppose
    that you have a 2x5x20 nested list of models and you want to
    find the last critical temperature of each model. Then use
    
    >>> Tcrit = funcOnModels(lambda A: A.TcTrans[-1]['Tcrit'], models)
    
    Tcrit will be a numpy array with shape (2,5,20).
    """
    M = []
    for a in models:
        if isinstance(a,list) or isinstance(a,tuple):
            M.append(funcOnModels(f, a))
        else:
            try:
                M.append(f(a))
            except:
                M.append(np.nan)
    return np.array(M)
    
def _linkTransitions(models, critTrans = True):
    """
    This function will take a list of models that represent the
    variation of some continuous model parameter, and output several
    lists of phase transitions such that all of the transitions
    in a single list roughly correspond to each other.

    NOT UPDATED FOR NEW COSMOTRANSITIONS.
    """
    allTrans = []
    for model in models:
        allTrans.append(model.TcTrans if critTrans else model.TnTrans)
    # allTrans is now a list of lists of transitions. 
    # We want to rearrange each sublist so that it matches the previous sublist.
    for j in xrange(len(allTrans)-1):
        trans1, trans2 = allTrans[j], allTrans[j+1]
        if trans1 == None: trans1 = []
        if trans2 == None: trans2 = []
        # First, clear the transiction dictionaries of link information
        for t in trans1+trans2:
            if t != None:
                t['link'] = None
                t['diff'] = np.inf
        for i1 in xrange(len(trans1)):
            t1 = trans1[i1] #t1 and t2 are individual transition dictionaries
            if t1 == None: continue
            for i2 in xrange(len(trans2)):
                t2 = trans2[i2] 
                if t2 == None: continue
                # See if t1 and t2 are each other's closest match
                diff = np.sum((t1['low vev']-t2['low vev'])**2)**.5 \
                    + np.sum((t1['high vev']-t2['high vev'])**2)**.5
                if diff < t1['diff'] and diff < t2['diff']:
                    t1['diff'] = t2['diff'] = diff
                    t1['link'], t2['link'] = i2, i1
        for i2 in xrange(len(trans2)):
            t2 = trans2[i2]
            if (t2 != None and t2['link'] != None 
                and trans1[t2['link']]['link'] != i2):
                    t2['link'] = None # doesn't link back.
        # Now each transition in tran2 is linked to its closest match in tran1, 
        # or None if it has no match
        newTrans = [None]*len(trans1)
        for t2 in trans2:
            if t2 == None:
                continue
            elif t2['link'] == None:
                # This transition doesn't match up with anything.
                newTrans.append(t2) 
            else:
                newTrans[t2['link']] = t2
        allTrans[j+1] = newTrans
    # Almost done. Just need to clean up the transitions and make sure that 
    # the allTrans list is rectangular.
    for trans in allTrans:
        for t in trans:
            if t != None:
                del t['link']
                del t['diff']
    n = len(allTrans[-1])
    for trans in allTrans:
        while len(trans) < n:
            trans.append(None)
    # Finally, transpose allTrans:
    allTrans2 = []
    for i in xrange(len(allTrans[0])):
        allTrans2.append([])
        for j in xrange(len(allTrans)):
            allTrans2[-1].append(allTrans[j][i])
    return allTrans2

                
        

        
