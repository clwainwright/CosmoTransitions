"""
This module provides the functions for the one-loop finite 
temperature corrections to a potential in QFT.  The two basic
functions are:

    Jb(x) = int[0->inf] dy +y^2 log( 1 - exp(-sqrt(x^2 + y^2)) )
    
    Jf(x) = int[0->inf] dy -y^2 log( 1 + exp(-sqrt(x^2 + y^2)) )

Call them by:

    Jb(x, approx='high', deriv=0, n = 8)

Here, approx can either be 'exact', 'spline', 'high', or 'low'.
Exact calculates the integral numerically, while high and low
calculate the high and low x expansions of J to order n.
Specify the derivative with the 'deriv' parameter.
"""
__version__ = "2.0a1"


import numpy
import scipy
from scipy import integrate, interpolate
from scipy import special
from scipy import optimize

pi = numpy.pi
euler_gamma = 0.577215661901532
log, exp, sqrt, fac = numpy.log, numpy.exp, numpy.sqrt, scipy.misc.factorial
array = numpy.array

import os
spline_data_path = os.path.dirname(__file__)

# The following are the exact integrals
def _Jf_exact(x):
    f = lambda y: -y*y*log(1+exp(-sqrt(y*y+x*x)))
    if(x.imag == 0):
        x = abs(x)
        return integrate.quad(f, 0, numpy.inf)[0]
    else:
        f1 = lambda y: -y*y*log(2*abs(numpy.cos(sqrt(abs(x*x)-y*y)/2)))
        return integrate.quad(f1,0,abs(x))[0] \
            + integrate.quad(f,abs(x),numpy.inf)[0]
        
def _Jf_exact2(theta): # Function of theta so that you can get negative values
    f = lambda y: -y*y*log(1+exp(-sqrt(y*y+theta))).real
    if theta >= 0:
        return integrate.quad(f, 0, numpy.inf)[0]
    else:
        f1 = lambda y: -y*y*log(2*abs(numpy.cos(sqrt(-theta-y*y)/2)))
        return integrate.quad(f, abs(theta)**.5, numpy.inf)[0] \
            + integrate.quad(f1, 0, abs(theta)**.5)[0]

def _Jb_exact(x):
    f = lambda y: y*y*log(1-exp(-sqrt(y*y+x*x)))
    if(x.imag == 0):
        x = abs(x)
        return integrate.quad(f, 0, numpy.inf)[0]
    else:
        f1 = lambda y: y*y*log(2*abs(numpy.sin(sqrt(abs(x*x)-y*y)/2)))
        return integrate.quad(f1,0,abs(x))[0] \
            + integrate.quad(f,abs(x),numpy.inf)[0]
        
def _Jb_exact2(theta): # Function of theta so that you can get negative values
    f = lambda y: y*y*log(1-exp(-sqrt(y*y+theta))).real
    if theta >= 0:
        return integrate.quad(f, 0, numpy.inf)[0]
    else:
        f1 = lambda y: y*y*log(2*abs(numpy.sin(sqrt(-theta-y*y)/2)))
        return integrate.quad(f, abs(theta)**.5, numpy.inf)[0] \
           + integrate.quad(f1, 0, abs(theta)**.5)[0]


def _dJf_exact(x):
    f = lambda y: y*y*(exp(sqrt(y*y+x*x))+1)**-1*x/sqrt(y*y+x*x)
    return integrate.quad(f, 0, numpy.inf)[0]

def _dJb_exact(x):
    f = lambda y: y*y*(exp(sqrt(y*y+x*x))-1)**-1*x/sqrt(y*y+x*x)
    return integrate.quad(f, 0, numpy.inf)[0]
    
# This function allows a 1D array to be passed to something that 
# normally can't handle it
def arrayFunc(f, x, typ = float):
    i = 0
    try:
        n = len(x)
    except:
        return f(x) # x isn't an array
    s = numpy.empty(n, typ)
    while(i < n):
        try:
            s[i] = f(x[i])
        except:
            s[i] = numpy.NaN
        i += 1
    return s

def Jf_exact(x):
    """Jf calculated directly from the integral."""
    return arrayFunc(_Jf_exact, x, complex)
def Jf_exact2(theta):
    """Jf calculated directly form the integral; input is theta = x^2."""
    return arrayFunc(_Jf_exact2, theta)
def Jb_exact(x):
    """Jb calculated directly from the integral."""
    return arrayFunc(_Jb_exact, x)
def Jb_exact2(theta):
    """Jb calculated directly form the integral; input is theta = x^2."""
    return arrayFunc(_Jb_exact2, theta)
def dJf_exact(x):
    """dJf/dx calculated directly from the integral."""
    return arrayFunc(_dJf_exact, x)
def dJb_exact(x):
    """dJb/dx calculated directly from the integral."""
    return arrayFunc(_dJb_exact, x)

# Spline fitting, Jf
_xfmin = -6.82200203 #-11.2403168
_xfmax = 1.35e3
try:
    f = open(spline_data_path+"/finiteT_f.dat", 'r')
    X = numpy.fromstring(f.read(),'float64')
    _xf,_yf = X.reshape(2,X.size/2)
    f.close()
except:
    # x = |xmin|*sinh(y), where y in linear 
    # (so that we're not overpopulating the uniteresting region)
    _xf = numpy.linspace(numpy.arcsinh(-1.3*20), 
                         numpy.arcsinh(-20*_xfmax/_xfmin), 1000)
    _xf = abs(_xfmin)*numpy.sinh(_xf)/20
    _yf = Jf_exact2(_xf)
    f = open(spline_data_path+"/finiteT_f.dat", 'w')
    f.write(array([_xf,_yf]).tostring())
    f.close()

_tckf = interpolate.splrep(_xf, _yf)
def Jf_spline(X,n=0):
    """Jf interpolated from a saved spline. Input is (m/T)^2."""
    X = numpy.array(X, copy=False)  
    x = X.ravel()
    y = interpolate.splev(x,_tckf, der=n).ravel()
    y[x < _xfmin] = interpolate.splev(_xfmin,_tckf, der=n)
    y[x > _xfmax] = 0
    return y.reshape(X.shape)
    
# Spline fitting, Jb
_xbmin = -3.72402637 
# We're setting the lower acceptable bound as the point where it's a minimum
# This guarantees that it's a monatonically increasing function, and the first 
# deriv is continuous.
_xbmax = 1.41e3
try:
    f = open(spline_data_path+"/finiteT_b.dat", 'r')
    X = numpy.fromstring(f.read(),'float64')
    _xb,_yb = X.reshape(2,X.size/2)
    f.close()
except:
    # x = |xmin|*sinh(y), where y in linear 
    # (so that we're not overpopulating the uniteresting region)
    _xb = numpy.linspace(numpy.arcsinh(-1.3*20), 
                         numpy.arcsinh(-20*_xbmax/_xbmin), 1000)
    _xb = abs(_xbmin)*numpy.sinh(_xb)/20
    _yb = Jb_exact2(_xb)
    f = open(spline_data_path+"/finiteT_b.dat", 'w')
    f.write(array([_xb,_yb]).tostring())
    f.close()
    
_tckb = interpolate.splrep(_xb, _yb)
def Jb_spline(X,n=0):
    """Jb interpolated from a saved spline. Input is (m/T)^2."""
    X = numpy.array(X, copy=False)  
    x = X.ravel()
    y = interpolate.splev(x,_tckb, der=n).ravel()
    y[x < _xbmin] = interpolate.splev(_xbmin,_tckb, der=n)
    y[x > _xbmax] = 0
    return y.reshape(X.shape)

    
# Now for the low x expansion (require that n <= 50)
a,b,c,d = -pi**4/45, pi*pi/12, -pi/6, -1/32.
logab = 1.5 - 2*euler_gamma + 2*log(4*pi)
l = numpy.arange(50)+1
g = -2*pi**3.5 * (-1)**l*(1+special.zetac(2*l+1))\
    *special.gamma(l+.5)/(fac(l+2)*(2*pi)**(2*l+4))
lowCoef_b = (a,b,c,d,logab,l,g)
del (a,b,c,d,logab,l,g) # clean up name space

a,b,d = -7*pi**4/360, pi*pi/24, 1/32.
logaf = 1.5 - 2*euler_gamma + 2*log(pi)
l = numpy.arange(50)+1
g = .25*pi**3.5 * (-1)**l*(1+special.zetac(2*l+1))\
    *special.gamma(l+.5)*(1-.5**(2*l+1))/(fac(l+2)*pi**(2*l+4))
lowCoef_f = (a,b,d,logaf,l,g)
del (a,b,d,logaf,l,g) # clean up name space

def Jb_low(x,n=20):
    """Jb calculated using the low-x (high-T) expansion."""
    (a,b,c,d,logab,l,g) = lowCoef_b
    y = a + x*x*(b + x*(c + d*x*(numpy.nan_to_num(log(x*x)) - logab)))
    i = 1
    while i <= n:
        y+= g[i-1]*x**(2*i+4)
        i+=1
    return y
    
def Jf_low(x,n=20):
    """Jf calculated using the low-x (high-T) expansion."""
    (a,b,d,logaf,l,g) = lowCoef_f
    y = a + x*x*(b + d*x*x*(numpy.nan_to_num(log(x*x)) - logaf))
    i = 1
    while i <= n:
        y+= g[i-1]*x**(2*i+4)
        i+=1
    return y


# The next few functions are all for the high approximation

def x2K2(k,x):
    #x = abs(x)
    y = -x*x*special.kn(2, k*x)/(k*k)
    if(isinstance(x, numpy.ndarray)):
        y[x==0] = numpy.ones(len(y[x==0]))*-2.0/k**4
    elif(x == 0):
        return -2.0/k**4
    return y

def dx2K2(k,x):
    y = abs(x)
    return numpy.nan_to_num( x*y*special.kn(1,k*y)/k )

def d2x2K2(k,x):
    x = abs(x)
    y = numpy.nan_to_num( x*(special.kn(1,k*x)/k -x*special.kn(0,k*x)) )
    if(isinstance(x, numpy.ndarray)):
        y[x==0] = numpy.ones(len(y[x==0]))*1.0/k**2
    elif(x == 0):
        return 1.0/k**2
    return y
    
def d3x2K2(k,x):
    y = abs(x)
    return numpy.nan_to_num( x*(y*k*special.kn(1,k*y) -3*special.kn(0,k*y)) )
            
def Jb_high(x, deriv = 0, n = 8):
    """Jb calculated using the high-x (low-T) expansion."""
    K = (x2K2, dx2K2, d2x2K2, d3x2K2)[deriv]
    y, k = 0.0, 1
    while k <= n:
        y += K(k,x)
        k += 1
    return y

def Jf_high(x, deriv = 0, n = 8):
    """Jf calculated using the high-x (low-T) expansion."""
    K = (x2K2, dx2K2, d2x2K2, d3x2K2)[deriv]
    y, k, i = 0.0, 1, 1
    while k <= n:
        y += i*K(k,x)
        i *= -1
        k += 1
    return y

# And here are the final functions:
# Note that if approx = 'spline', the function called is 
# J(theta) (x^2 -> theta so you can get negative mass squared)
def Jb(x, approx='high', deriv=0, n = 8):
    """
    A shorthand for calling one of the Jb functions above.

    Parameters
    ----------
    approx : str, optional
        One of 'exact', 'high', 'low', or 'spline'.
    deriv : int, optional
        The order of the derivative (0 for no derivative). 
        Must be <= (1, 3, 0, 3) for approx = (exact, high, low, spline).
    n : int, optional
        Number of terms to use in the low and high-T approximations.
    """
    if(approx == 'exact'):
        if(deriv == 0):
            return Jb_exact(x)
        elif(deriv == 1):
            return dJb_exact(x)
        else:
            raise ValueError("For approx=='exact', deriv must be 0 or 1.")
    elif(approx == 'spline'):
        return Jb_spline(x, deriv)
    elif(approx == 'low'):
        if(n > 100):
            raise ValueError("Must have n <= 100")
        if(deriv == 0):
            return Jb_low(x,n)
        else:
            raise ValueError("For approx=='low', deriv must be 0.")
    elif(approx == 'high'):
        if(deriv > 3):  
            raise ValueError("For approx=='high', deriv must be 3 or less.")
        else:
            return Jb_high(x, deriv, n)
    raise ValueError("Unexpected value for 'approx'.")

def Jf(x, approx='high', deriv=0, n = 8):
    """
    A shorthand for calling one of the Jf functions above.

    Parameters
    ----------
    approx : str, optional
        One of 'exact', 'high', 'low', or 'spline'.
    deriv : int, optional
        The order of the derivative (0 for no derivative). 
        Must be <= (1, 3, 0, 3) for approx = (exact, high, low, spline).
    n : int, optional
        Number of terms to use in the low and high-T approximations.
    """    
    if(approx == 'exact'):
        if(deriv == 0):
            return Jf_exact(x)
        elif(deriv == 1):
            return dJf_exact(x)
        else:
            raise ValueError("For approx=='exact', deriv must be 0 or 1.")
    elif(approx == 'spline'):
        return Jf_spline(x, deriv)
    elif(approx == 'low'):
        if(n > 100):
            raise ValueError("Must have n <= 100")
        if(deriv == 0):
            return Jf_low(x,n)
        else:
            raise ValueError("For approx=='low', deriv must be 0.")
    elif(approx == 'high'):
        if(deriv > 3):  
            raise ValueError("For approx=='high', deriv must be 3 or less.")
        else:
            return Jf_high(x, deriv, n)
    raise ValueError("Unexpected value for 'approx'.")
            

        
        
    

