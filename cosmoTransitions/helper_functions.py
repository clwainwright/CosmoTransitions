"""
A collection of (mostly) stand alone helper functions.
"""

__version__ = "2.0a1"

import numpy as np
from collections import namedtuple

"""
Miscellaneous functions
~~~~~~~~~~~~~~~~~~~~~~~
"""

def setDefaultArgs(func, **kwargs):
    """
    Changes the default args in `func` to match `kwargs`.
    
    This can be useful when dealing with deeply nested functions for which the
    default parameters cannot be set directly in the top-level function.
    
    Raises
    ------ 
    ValueError 
        if `func` does not have default arguments that match `kwargs`.
    
    Example
    -------
      >>> def foo(bar="Hello world!"):
      ...     print bar
      >>> setDefaultArgs(foo, bar="The world has changed!")
      >>> foo()
      The world has changed!
    """
    import inspect
    args, varargs, varkwargs, defaults = inspect.getargspec(func)
    have_defaults = args[-len(defaults):]
    defaults = list(defaults) # so that we can write to it
    for kwd in kwargs:
        try:
            i = have_defaults.index(kwd)
        except ValueError:
            raise ValueError("Function '%s()' does not have default argument "
                             "named '%s'." % (func.__name__, kwd))
        defaults[i] = kwargs[kwd]
    func.__defaults__ = tuple(defaults)

def monotonicIndices(x):
    """
    Returns the indices of `x` such that `x[i]` is purely increasing.
    """
    x = np.array(x)
    if x[0] > x[-1]:
        x = x[::-1]
        is_reversed = True
    else:
        is_reversed = False
    I = [0]
    for i in xrange(1, len(x)-1):
        if x[i] > x[I[-1]] and x[i] < x[-1]:
            I.append(i)
    I.append(len(x)-1)
    if is_reversed:
        return len(x)-1-np.array(I)
    else:
        return np.array(I)

def clampVal(x, a, b):
    """
    Clamp the value `x` to be between `a` and `b`.

    Parameters
    ----------
    x, a, b : array_like
        Must have the same shape or be broadcastable.

    Returns
    -------
    array_like
    """
    s = np.sign(b-a) # +1 for b > a, -1 for b < a
    sa = 1+s*np.sign(x-a)
    x = (x*sa + a*(2-sa)) // 2
    sb = 1+s*np.sign(b-x)
    x = (x*sb + b*(2-sb)) // 2
    return x    
    
"""
Numerical integration
~~~~~~~~~~~~~~~~~~~~~
"""


class IntegrationError(Exception):
    """
    Used to indicate an integration error, primarily in :func:`rkqs`.
    """
    pass
    
_rkqs_rval = namedtuple("rkqs_rval", "Delta_y Delta_t dtnxt")
def rkqs(y,dydt,t,f, dt_try, epsfrac, epsabs, args=()):
    """
    Take a single 5th order Runge-Kutta step with error monitoring.
    
    This function is adapted from Numerical Recipes in C.
    
    The step size dynamically changes such that the error in `y` is smaller 
    than the larger of `epsfrac` and `epsabs`. That way, if one wants to 
    disregard the fractional error, set `epsfrac` to zero but keep `epsabs` 
    non-zero.
    
    Parameters
    ----------
    y, dydt : array_like
        The initial value and its derivative at the start of the step.
        They should satisfy ``dydt = f(y,t)``. `dydt` is included here for
        efficiency (in case the calling function already calculated it).
    t : float
        The integration variable.
    f : callable
        The derivative function.
    dt_try : float
        An initial guess for the step size.
    epsfrac, epsabs : array_like
        The maximual fractional and absolute errors. Should be either length 1
        or the same size as `y`.
    args : tuple
        Optional arguments for `f`.
        
    Returns
    -------
    Delta_y : array_like
        Change in `y` during this step.
    Delta_t : float
        Change in `t` during this step.
    dtnext : float
        Best guess for next step size.
        
    Raises
    ------
    IntegrationError
        If the step size gets smaller than the floating point error.
        
    References
    ----------
    Based on algorithms described in [1]_.
    
    .. [1] W. H. Press, et. al. "Numerical Recipes in C: The Art of Scientific
       Computing. Second Edition." Cambridge, 1992.
    """
    dt = dt_try
    while True:
        dy,yerr = _rkck(y,dydt,t,f,dt,args)
        errmax = np.nan_to_num( np.max( np.min([abs(yerr/epsabs), 
            abs(yerr)/((abs(y)+1e-300)*epsfrac)],axis=0) ) )
        if(errmax < 1.0):
            break # Step succeeded
        dttemp = 0.9*dt*errmax**-.25
        dt = max(dttemp,dt*.1) if dt > 0 else min(dttemp,dt*.1)
        if(t+dt==t):
            raise IntegrationError("Stepsize rounds down to zero.")
    if errmax > 1.89e-4:
        dtnext = 0.9 * dt * errmax**-.2
    else:
        dtnext = 5*dt
    return _rkqs_rval(dy, dt, dtnext)

def rkqs2(y,dydt,t,f, dt_try, inv_epsabs, args=()):
    """
    Same as :func:`rkqs`, but ``inv_epsabs = 1/epsabs`` and ``epsfrac`` is
    not used.
    """
    dt = dt_try
    while True:
        dy,yerr = _rkck(y,dydt,t,f,dt,args)
        errmax = np.max( yerr * inv_epsabs )
        if(errmax < 1.0):
            break # Step succeeded
        dttemp = 0.9*dt*errmax**-.25
        dt = max(dttemp,dt*.1) if dt > 0 else min(dttemp,dt*.1)
        if(t+dt==t):
            raise IntegrationError("Stepsize rounds down to zero.")
    if errmax > 1.89e-4:
        dtnext = 0.9 * dt * errmax**-.2
    else:
        dtnext = 5*dt
    return _rkqs_rval(dy, dt, dtnext)

def _rkck(y,dydt,t,f,dt,args=()):
    """
    Take one 5th-order Cash-Karp Runge-Kutta step.
    
    Returns
    -------
    array_like
        The change in `y` during this step.
    array_like
        An error estimate for `y`.
    """
    a2=0.2;a3=0.3;a4=0.6;a5=1.0;a6=0.875;b21=0.2
    b31=3.0/40.0;b32=9.0/40.0;b41=0.3;b42 = -0.9;b43=1.2;
    b51 = -11.0/54.0; b52=2.5;b53 = -70.0/27.0;b54=35.0/27.0;
    b61=1631.0/55296.0;b62=175.0/512.0;b63=575.0/13824.0;
    b64=44275.0/110592.0;b65=253.0/4096.0;c1=37.0/378.0;
    c3=250.0/621.0;c4=125.0/594.0;c6=512.0/1771.0;
    dc5 = -277.00/14336.0;
    dc1=c1-2825.0/27648.0;dc3=c3-18575.0/48384.0;
    dc4=c4-13525.0/55296.0;dc6=c6-0.25
    ytemp = y+b21*dt*dydt
    ak2 = f(ytemp, t+a2*dt, *args)
    ytemp = y+dt*(b31*dydt+b32*ak2)
    ak3 = f(ytemp, t+a3*dt, *args)
    ytemp = y+dt*(b41*dydt+b42*ak2+b43*ak3)
    ak4 = f(ytemp, t+a4*dt, *args)
    ytemp = y + dt*(b51*dydt+b52*ak2+b53*ak3+b54*ak4)
    ak5 = f(ytemp, t+a5*dt, *args)
    ytemp = y + dt*(b61*dydt+b62*ak2+b63*ak3+b64*ak4+b65*ak5)
    ak6 = f(ytemp, t+a6*dt, *args)
    dyout = dt*(c1*dydt+c3*ak3+c4*ak4+c6*ak6)
    yerr = dt*(dc1*dydt+dc3*ak3+dc4*ak4+dc5*ak5+dc6*ak6)
    return dyout, yerr
    
"""
Numerical derivatives
~~~~~~~~~~~~~~~~~~~~~

The *derivij()* functions accept arrays as input and return arrays as output.
In contrast, :class:`gradientFunction` and :class:hessianFunction` accept
functions as input and return callable class instances (essentially functions)
as output. The returned functions can then be used to find derivatives.
"""

def deriv14(y,x):
    R"""
    Calculates :math:`dy/dx` to fourth-order in :math:`\Delta x` using 
    finite differences. The derivative is taken along the last dimension of `y`.
    
    Both `y` and `x` should be numpy arrays. The derivatives are centered
    in the interior of the array, but not at the edges. The spacing in `x` 
    does not need to be uniform.
    """
    n = len(x)
    j = np.arange(5)
    j[j>4/2] -= 5
    i = np.arange(n) - j[:,np.newaxis]
    i[i<0] += 5
    i[i>=n] -= 5
    
    d1 = x[i[1]]-x[i[0]]
    d2 = x[i[2]]-x[i[0]]
    d3 = x[i[3]]-x[i[0]]
    d4 = x[i[4]]-x[i[0]]
    
    w4 = (d1*d2*d3) / ( 
        -d4 * (-d1*d2*d3 + d4 * (d1*d2+d2*d3+d3*d1 + d4 * (+d4-d1-d2-d3))))
    w3 = (d1*d2*d4) / ( 
        -d3 * (-d1*d2*d4 + d3 * (d1*d2+d2*d4+d4*d1 + d3 * (-d4-d1-d2+d3))))
    w2 = (d1*d4*d3) / ( 
        -d2 * (-d1*d4*d3 + d2 * (d1*d4+d4*d3+d3*d1 + d2 * (-d4-d1+d2-d3))))
    w1 = (d4*d2*d3) / ( 
        -d1 * (-d4*d2*d3 + d1 * (d4*d2+d2*d3+d3*d4 + d1 * (-d4+d1-d2-d3))))
    w0 = -(w1+w2+w3+w4)
    
    dy = w0*y[...,i[0]] + w1*y[...,i[1]] \
         + w2*y[...,i[2]] + w3*y[...,i[3]] + w4*y[...,i[4]]
    
    return dy
    
def deriv14_const_dx(y, dx=1.0):
    R"""
    Calculates :math:`dy/dx` to fourth-order in :math:`\Delta x` using 
    finite differences. The derivative is taken along the last dimension of `y`.
    
    The output of this function should be identical to :func:`deriv14` when the
    spacing in `x` is constant, but this will be faster.
    
    Parameters
    ----------
    y : array_like
    dx : float, optional
    """
    y = y.T # now the derivative is along the first dimension
    dy = np.empty_like(y)
    
    dy[2:-2] = y[:-4] - 8*y[1:-3] + 8*y[3:-1] - y[4:]
    dy[+0] = -25*y[+0] + 48*y[+1] - 36*y[+2] + 16*y[+3] - 3*y[+4]
    dy[+1] = -3*y[+0] - 10*y[+1] + 18*y[+2] - 6*y[+3] + y[+4]
    dy[-2] = +3*y[-1] + 10*y[-2] - 18*y[-3] + 6*y[-4] - y[-5]
    dy[-1] = +25*y[-1] - 48*y[-2] + 36*y[-3] - 16*y[-4] + 3*y[-5]
    
    return dy.T / (12.0 * dx)
    
def deriv1n(y,x,n):
    """
    Calculates :math:`dy/dx` to nth-order in :math:`\Delta x` using 
    finite differences. The derivative is taken along the last dimension of `y`.
    
    Both `y` and `x` should be numpy arrays. The derivatives are centered in the
    interior of the array, but not at the edges. The spacing in `x` does not
    need to be uniform.
    """
    nx = len(x)
    j = np.arange(n+1)
    j[j>n/2] -= n+1
    i = np.arange(nx) - j[:,np.newaxis]
    i[i<0] += n+1
    i[i>=nx] -= n+1
    
    d = np.empty((n,n,nx), dtype=x.dtype)*1.0
    d[0] = x[i[1:]] - x[i[0]]
    for j in xrange(1,n):
        d[j] = np.roll(d[j-1], -1, axis=0)
    d[:,0] *= -1
    w = np.zeros((n+1,nx), dtype=y.dtype)*1.
    
    # For example, when calculating w[1], we need only use
    # w[1]: d1 = d[0,0], d2 = d[0,1], d3 = d[0,2], ..., dn = d[0,n-1]
    # and for the other weights we just increment the first index:
    # w[2]: d2 = d[1,0], d3 = d[1,1], d4 = d[1,2], ..., dn = d[1,n-2], 
    #       d1 = d[1,n-1]
    # So we should be able to calculate all of them at once like this.
    s = ((2**np.arange(n-1)) & np.arange(2**(n-1))[:,np.newaxis])
    s[s>0] = (np.arange(1,n) * np.ones(2**(n-1))[:,np.newaxis])[s>0]
    w[1:] = (np.sum(np.product(d[:,s],axis=2), axis=1)*d[:,0] 
             / np.product(d[:,1:], axis=1))
    w[1:] = -w[1:]**-1
    w[0] = -np.sum(w[1:],axis=0)
    
    dy = np.sum(w*y[...,i], axis=-2)
    
    return dy
        
def deriv23(y,x):
    """
    Calculates :math:`d^2y/dx^2` to third-order in :math:`\Delta x` using 
    finite differences. The derivative is taken along the last dimension of `y`.
    
    Both `y` and `x` should be numpy arrays. The derivatives are centered in the
    interior of the array, but not at the edges. The spacing in `x` does not
    need to be uniform. The accuracy increases to fourth-order if the spacing
    is uniform.
    """
    n = len(x)
    j = np.arange(5)
    j[j>4/2] -= 5
    i = np.arange(n) - j[:,np.newaxis]
    i[i<0] += 5
    i[i>=n] -= 5
    
    d1 = x[i[1]]-x[i[0]]
    d2 = x[i[2]]-x[i[0]]
    d3 = x[i[3]]-x[i[0]]
    d4 = x[i[4]]-x[i[0]]
        
    w4 = 2*(d1*d2+d2*d3+d3*d1) / (
        d4 * (-d1*d2*d3 + d4 * (d1*d2+d2*d3+d3*d1 + d4 * (+d4-d1-d2-d3) ) ) )
    w3 = 2*(d1*d2+d2*d4+d4*d1) / (
        d3 * (-d1*d2*d4 + d3 * (d1*d2+d2*d4+d4*d1 + d3 * (-d4-d1-d2+d3) ) ) )
    w2 = 2*(d1*d4+d4*d3+d3*d1) / (
        d2 * (-d1*d4*d3 + d2 * (d1*d4+d4*d3+d3*d1 + d2 * (-d4-d1+d2-d3) ) ) )
    w1 = 2*(d4*d2+d2*d3+d3*d4) / (
        d1 * (-d4*d2*d3 + d1 * (d4*d2+d2*d3+d3*d4 + d1 * (-d4+d1-d2-d3) ) ) )
    w0 = -(w1+w2+w3+w4)
    
    d2y = w0*y[...,i[0]] + w1*y[...,i[1]] \
         + w2*y[...,i[2]] + w3*y[...,i[3]] + w4*y[...,i[4]]
    return d2y
    
def deriv23_const_dx(y, dx=1.0):
    """
    Calculates :math:`d^2y/dx^2` to third-order in :math:`\Delta x` using 
    finite differences. The derivative is taken along the last dimension of `y`.
    
    The output of this function should be identical to :func:`deriv23` when the
    spacing in `x` is constant, but this will be faster.
    
    Parameters
    ----------
    y : array_like
    dx : float, optional
    """
    y = y.T # now the derivative is along the first dimension
    dy = np.empty_like(y)
    
    dy[2:-2] = -y[:-4] + 16*y[1:-3] - 30*y[2:-2] + 16*y[3:-1] - y[4:]
    dy[+0] = 35*y[+0] - 104*y[+1] + 114*y[+2] - 56*y[+3] + 11*y[+4]
    dy[+1] = 11*y[+0] - 20*y[+1] + 6*y[+2] + 4*y[+3] - y[+4]
    dy[-2] = 11*y[-1] - 20*y[-2] + 6*y[-3] + 4*y[-4] - y[-5]
    dy[-1] = 35*y[-1] - 104*y[-2] + 114*y[-3] - 56*y[-4] + 11*y[-5]
    
    return dy.T / (12.0 * dx)

class gradientFunction:
    """
    Make a function which returns the gradient of some scalar function.
    
    Parameters
    ----------
    f : callable
        The first argument `x` should either be a single point with length
        `Ndim` or an array (or matrix, etc.) of points with shape 
        ``(..., Ndim)``, where ``...`` is some arbitrary shape. The return
        shape should be the same as the input shape, but with the last axis
        stripped off (i.e., it should be a scalar function). Additional
        required or optional arguments are allowed.
    eps : float or array_like
        The small change in `x` used to calculate the finite differences.
        Can either be a scalar or have length `Ndim`.
    Ndim : int
        Number of dimensions for each point.
    order : 2 or 4
        Calculate the derivatives to either 2nd or 4th order in `eps`.
        
    Example
    -------
    >>> def f(X):
    ...     x,y = np.asarray(X).T
    ...     return (x*x + x*y +3.*y*y*y).T
    >>> df = gradientFunction(f, eps=.01, Ndim=2, order=4)
    >>> x = np.array([[0,0],[0,1],[1,0],[1,1]])
    >>> print df(x)
        array([[ 0.,  0.], [ 1.,  9.], [ 2.,  1.], [ 3., 10.]])
    """
    def __init__(self, f, eps, Ndim, order=4):
        assert order == 2 or order == 4
        eps = np.asanyarray(eps)
        dx = np.empty((order, Ndim, Ndim))
        dx[:] = np.diag(np.ones(Ndim)*eps)
        dxT = dx.T
        coef = np.empty((order, Ndim))
        coef[:] = 1.0/eps
        coefT = coef.T
        if order == 2:
            dxT *= [-1, 1]
            coefT *= [-.5, .5]
        if order == 4:
            dxT *= [-2, -1, 1, 2]
            coefT *= [1, -8, 8, -1]
            coefT /= 12.0
        self.f = f
        self.dx = dx
        self.coef = coef
    
    def __call__(self, x, *args, **kwargs):
        """
        Calculate the gradient. Output shape is the same as the input shape.
        """
        x = np.asanyarray(x)[...,np.newaxis,np.newaxis,:]
        return np.sum(self.f(x+self.dx, *args, **kwargs)*self.coef, axis=-2)
    
class hessianFunction:
    """
    Make a function which returns the Hessian (second derivative) matrix of
    some scalar function.
    
    Parameters
    ----------
    f : callable
        The first argument `x` should either be a single point with length
        `Ndim` or an array (or matrix, etc.) of points with shape 
        ``(..., Ndim)``, where ``...`` is some arbitrary shape. The return
        shape should be the same as the input shape, but with the last axis
        stripped off (i.e., it should be a scalar function). Additional
        required or optional arguments are allowed.
    eps : float or array_like
        The small change in `x` used to calculate the finite differences.
        Can either be a scalar or have length `Ndim`.
    Ndim : int
        Number of dimensions for each point.
    order : 2 or 4
        Calculate the derivatives to either 2nd or 4th order in `eps`.
    """
    def __init__(self, f, eps, Ndim, order=4):
        assert order == 2 or order == 4
        eps = np.ones(Ndim) * eps
        dx = []
        coef = []
        for i in xrange(Ndim):
            dx.append([])
            coef.append([])
            for j in xrange(i):
                dx_ = np.zeros((order, order, Ndim))
                if (order == 2):
                    dx_[:,:,i] = np.array([-1,1]) * eps[i]
                    dx_ = np.rollaxis(dx_, 1)
                    dx_[:,:,j] = np.array([-1,1]) * eps[j]
                    coef_ = np.array([-.5, .5])
                    coef_ = coef_[:,np.newaxis] * coef_[np.newaxis,:]
                    coef_ /= eps[i]*eps[j]
                if (order == 4):
                    dx_[:,:,i] = np.array([-2,-1,1,2]) * eps[i]
                    dx_ = np.rollaxis(dx_, 1)
                    dx_[:,:,j] = np.array([-2,-1,1,2]) * eps[j]
                    coef_ = np.array([1, -8, 8, -1.])
                    coef_ = coef_[:,np.newaxis] * coef_[np.newaxis,:]
                    coef_ /= 144.*eps[i]*eps[j]
                dx[-1].append(dx_.reshape(order*order, Ndim))
                coef[-1].append(coef_.reshape(order*order))
            dx_ = np.zeros((order+1, Ndim))
            if order == 2:
                dx_[:,i] = np.array([-1,0,1]) * eps[i]
                coef_ = np.array([1,-2,1.]) / (eps[i]*eps[i])
            if order == 4:
                dx_[:,i] = np.array([-2,-1,0,1,2]) * eps[i]
                coef_ = np.array([-1,16,-30,16,-1]) / (eps[i]*eps[i]*12)
            dx[-1].append(dx_)
            coef[-1].append(coef_)
        self.f = f
        self.coef = coef
        self.dx = dx
        self.Ndim = Ndim
    
    def __call__(self, x, *args, **kwargs):
        """
        Calculate the gradient. Output shape is ``input.shape + (Ndim,)``.
        """
        Ndim = self.Ndim
        f = self.f
        coef = self.coef
        dx= self.dx
        x = np.asanyarray(x)
        y = np.empty(x.shape + (Ndim,))
        x = x[...,np.newaxis,:]
        for i in xrange(Ndim):
            for j in xrange(i):
                y_ = np.sum(f(x+dx[i][j], *args, **kwargs) 
                            * coef[i][j], axis=-1)
                y[...,i,j] = y[...,j,i] = y_
            y[...,i,i] = np.sum(f(x+dx[i][i], *args, **kwargs) 
                                * coef[i][i], axis=-1)        
        return y

"""
Two-point interpolation
~~~~~~~~~~~~~~~~~~~~~~~
"""
        
def makeInterpFuncs(y0, dy0, d2y0, y1, dy1, d2y1):
    """
    Create interpolating functions between two points with a quintic polynomial.
    
    If we're given the first and second derivatives of a function
    at x=0 and x=1, we can make a 5th-order interpolation between
    the two.
    """
    a0,a1,a2, z,dz,d2z = y0, dy0, 0.5*d2y0, y1, dy1, d2y1
    b1,b2,b3 = z-a0-a1-a2, dz-a1-2*a2, d2z-2*a2
    a5 = .5*b3 - 3*b2 + 6*b1
    a4 = b2 - 3*b1 - 2*a5
    a3 = b1 - a4 - a5
    coefs = np.array([a0,a1,a2,a3,a4,a5])
    pows = np.arange(6)
    df = lambda x, c=coefs[1:], p=pows[1:]: np.sum(p*c*x**(p-1))
    f = lambda x, c=coefs, p=pows: np.sum(c*x**p)
    return f, df

class cubicInterpFunction:
    """
    Create an interpolating function between two points with a cubic polynomial.

    Like :func:`makeInterpFuncs`, but only uses the first derivatives.
    """
    def __init__(self, y0, dy0, y1, dy1):
        # Easiest to treat this as a bezier curve
        y3 = y1
        y1 = y0 + dy0/3.0
        y2 = y3 - dy1/3.0
        self.Y = y0, y1, y2, y3

    def __call__(self, t):
        mt = 1-t
        y0, y1, y2, y3 = self.Y
        return y0*mt**3 + 3*y1*mt*mt*t + 3*y2*mt*t*t + y3*t**3

"""
Spline interpolation
~~~~~~~~~~~~~~~~~~~~
"""

def Nbspl(t, x, k = 3):
    """
    Calculate the B-spline basis functions for the knots t evaluated at the 
    points x.
    
    Parameters
    ----------
    t : array_like
        An array of knots which define the basis functions.
    x : array_like
        The different values at which to calculate the functions.
    k : int, optional
        The order of the spline. Must satisfy ``k <= len(t)-2``.
        
    Returns
    -------
    array_like
        Has shape ``(len(x), len(t)-k-1)``.
        
    Notes
    -----
    This is fairly speedy, although it does spend a fair amount of time
    calculating things that will end up being zero. On a 2.5Ghz machine, it
    takes a few milliseconds to calculate when ``len(x) == 500; len(t) == 20; 
    k == 3``.
    
    For more info on basis splines, see e.g. 
    http://en.wikipedia.org/wiki/B-spline. 
    
    Example
    -------
    .. plot::
        :include-source:
      
        from helper_functions import Nbspl
        t = [-1,-1,-1,-1, -.5, 0, .5, 1, 1, 1, 1]
        x = np.linspace(-1,1,500)
        y = Nbspl(t,x, k=3)
        plt.plot(x, y)
        plt.xlabel(r"$x$")
        plt.ylabel(r"$y_i(x)$")
        plt.show()    
    """
    kmax = k
    if kmax > len(t)-2:
        raise Exception, "Input error in Nbspl: require that k < len(t)-2"
    t = np.array(t)#[np.newaxis, :]
    t2 = t.copy()
    x = np.array(x)[:, np.newaxis]
    N = 1.0*((x > t[:-1]) & (x <= t[1:]))
    for k in xrange(1, kmax+1):
        dt = t[k:] - t[:-k]
        _dt = dt.copy()
        _dt[dt != 0] = 1./dt[dt != 0]
        N = N[:,:-1]*(x-t[:-k-1])*_dt[:-1] - N[:,1:]*(x-t[k+1:])*_dt[1:] 
    return N
    
def Nbspld1(t, x, k = 3):
    """Same as :func:`Nbspl`, but returns the first derivative too."""
    kmax = k
    if kmax > len(t)-2:
        raise Exception, "Input error in Nbspl: require that k < len(t)-2"
    t = np.array(t)#[np.newaxis, :]
    x = np.array(x)[:, np.newaxis]
    N = 1.0*((x > t[:-1]) & (x <= t[1:]))
    dN = np.zeros_like(N)
    for k in xrange(1, kmax+1):
        dt = t[k:] - t[:-k]
        _dt = dt.copy()
        _dt[dt != 0] = 1./dt[dt != 0]
        dN = dN[:,:-1]*(x-t[:-k-1])*_dt[:-1] - dN[:,1:]*(x-t[k+1:])*_dt[1:] 
        dN += N[:,:-1]*_dt[:-1] - N[:,1:]*_dt[1:] 
        N = N[:,:-1]*(x-t[:-k-1])*_dt[:-1] - N[:,1:]*(x-t[k+1:])*_dt[1:] 
    return N, dN
    
def Nbspld2(t, x, k = 3):
    """Same as :func:`Nbspl`, but returns first and second derivatives too."""
    kmax = k
    if kmax > len(t)-2:
        raise Exception, "Input error in Nbspl: require that k < len(t)-2"
    t = np.array(t)#[np.newaxis, :]
    x = np.array(x)[:, np.newaxis]
    N = 1.0*((x > t[:-1]) & (x <= t[1:]))
    dN = np.zeros_like(N)
    d2N = np.zeros_like(N)
    for k in xrange(1, kmax+1):
        dt = t[k:] - t[:-k]
        _dt = dt.copy()
        _dt[dt != 0] = 1./dt[dt != 0]
        d2N = d2N[:,:-1]*(x-t[:-k-1])*_dt[:-1] - d2N[:,1:]*(x-t[k+1:])*_dt[1:] \
            + 2*dN[:,:-1]*_dt[:-1] - 2*dN[:,1:]*_dt[1:]
        dN = dN[:,:-1]*(x-t[:-k-1])*_dt[:-1] - dN[:,1:]*(x-t[k+1:])*_dt[1:] \
            + N[:,:-1]*_dt[:-1] - N[:,1:]*_dt[1:]
        N = N[:,:-1]*(x-t[:-k-1])*_dt[:-1] - N[:,1:]*(x-t[k+1:])*_dt[1:] 
    return N, dN, d2N

