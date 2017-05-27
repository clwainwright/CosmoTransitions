from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

import sys
if sys.version_info >= (3,0):
    xrange = range

__version__ = "2.0a2"


class MultiFieldPlotter:
    """
    This class tries to make it easier to view functions of more than two
    variables.

    For each set of two variables (or 'fields', since this is part of the
    CosmoTransitions package), this class will display a separate subplot in a
    managed figure. Each subplot is a different slice through the
    multi-dimensional space. By clicking on the subplots, the user can
    dynamically change the offsets of the slices in the other subplots.

    Parameters
    ----------
    bounds : array_like
        A list of ``(xmin, xmax)`` tuples for each dimension.
    f : callable
        The function to plot. The first argument must accept arrays of shape
        ``(..., Ndim)``, where `Ndim` is the number of dimensions.
    f_args : tuple, optional
        Extra agruments to pass to `f`.
    nx : int, optional
        Number of data points to plot in each dimension.
    contour_levs : int or array_like, optional
        If an array, a list of the contour levels to plot. If a list, the total
        number of contour levels across the bounding box (the contour levels are
        then calculated using :func:`calcContourLevels`).
    plot_1d : bool, optional
        If True, plot one-dimensional plots along with the contours. (not yet
        implemented)
    plot_flipped : bool, optional
        If True, plot the flipped contour for each field (so that the subplots
        form a square grid rather than a triangle).

    Attributes
    ----------
    figure : matplotlib.figure.Figure
    offset : array_like
        Each slice interesects the point given by `offset`. Initially set to
        the average of `bounds` and interactively modifiable by clicking on
        the plots.
    draws_offset : bool
        Set to True if the plots should draw the offset point (as intersecting
        lines).

    Example
    -------
    The following example will make three contour plots whose offsets can be
    changed interactively:

    >>> from multi_field_plotting import MultiFieldPlotter
    >>> def V(X): # Some potential that looks vaguely interesting
    ...     x,y,z = X[...,0], X[...,1], X[...,2]
    ...     return x*x - x**3 + x*y + y**2 - y*z**2 + z**4
    >>> mfp = MultiFieldPlotter([[-1,1.],[-1,1],[-1,1]], V)
    """
    def __init__(self, bounds, f, f_args=(), nx=40, contour_levs=50,
                 plot_1d=False, plot_flipped=False):
        self.bounds = np.array(bounds)
        self.f = f
        self.f_args = f_args
        self.nx = nx
        self.contour_levs = (contour_levs)
        self.contour_levs = np.array(contour_levs)
        if len(self.contour_levs.shape) == 0:
            self.calcContourLevels(self.contour_levs)
        self.plot_1d = plot_1d
        self.plot_flipped = plot_flipped
        self.figure = plt.figure()
        # Make the offset the center of the data bounds
        self.offset = np.average(bounds, axis=1)
        self.draws_offset = True if len(self.bounds) > 2 else False
        self.figure.canvas.mpl_connect('button_press_event', self._mouseDown)
        self.drawSubplot()

    def calcContourLevels(self, num_levs, nx=11):
        """
        Find the contour levels which span the bounds. Store in
        ``self.contour_levs``.

        Parameters
        ----------
        num_levs : int
            Desired number of contour levels.
        nx : int, optional
            The number of data points along each dimension that are used to
            find the minimum and maximum levels.
        """

        Ndim = len(self.bounds)
        X = np.empty([nx]*Ndim + [Ndim])
        for i in xrange(Ndim):
            xmin, xmax = self.bounds[i]
            Y = X.swapaxes(i, -2)
            Y[...,i] = np.linspace(xmin,xmax,nx)
        Z = self.f(X, *self.f_args)
        fmin = np.min(Z.ravel())
        fmax = np.max(Z.ravel())
        df = fmax-fmin
        self.contour_levs = np.linspace(fmin-df*.1, fmax+df*.1, num_levs*1.2)

    def drawSubplot(self, subplot='all'):
        """
        Performs the actual drawing.

        Parameters
        ----------
        subplot : (int, int) or 'all'
            The subplot to redraw. If a tuple, it should be field indicies of
            the x and y axes.
        """
        Ndim = len(self.bounds)
        if subplot == 'all':
            for i in xrange(Ndim):
                for j in xrange(Ndim):
                    self.drawSubplot((i,j))
            return
        if not self.plot_1d and subplot[0] == subplot[1]:
            return
        if not self.plot_flipped and subplot[0] > subplot[1]:
            return
        if self.plot_1d or self.plot_flipped:
            nrows_cols = Ndim
            plot_num = 1+subplot[0] + nrows_cols*subplot[1]
        else:
            nrows_cols = Ndim - 1
            plot_num = 1+subplot[0] + nrows_cols*(subplot[1]-1)
        ax = self.figure.add_subplot(nrows_cols,nrows_cols,plot_num)
        ax.clear()
        ax.xfield, ax.yfield = subplot
        if ax.yfield == Ndim-1:
            ax.set_xlabel("$x_%i$" % ax.xfield)
        if ax.xfield == 0:
            if ax.yfield == 0:
                ax.set_ylabel("$f(x_0)$")
            else:
                ax.set_ylabel("$x_%i$" % ax.yfield)
        # Generate the data and make the plot
        if ax.xfield == ax.yfield:
            pass  # 1d_plot
        else:
            X = np.empty((self.nx, self.nx, Ndim))
            X[:] = self.offset
            X[:,:,ax.xfield] = np.linspace(
                self.bounds[ax.xfield,0],
                self.bounds[ax.xfield,1], self.nx
            )[:,np.newaxis] * np.ones((self.nx, self.nx))
            X[:,:,ax.yfield] = np.linspace(
                self.bounds[ax.yfield,0],
                self.bounds[ax.yfield,1], self.nx
            )[np.newaxis,:] * np.ones((self.nx, self.nx))
            Z = self.f(X, *self.f_args)
            ax.contour(
                X[:,:,ax.xfield], X[:,:,ax.yfield], Z,
                self.contour_levs, cmap=plt.cm.Spectral)
          #  ax.pcolormesh(X[:,:,ax.xfield], X[:,:,ax.yfield], Z,
          #      cmap=plt.cm.Spectral)
            if self.draws_offset:
                xbounds = self.bounds[ax.xfield]
                ybounds = self.bounds[ax.yfield]
                x0 = self.offset[ax.xfield]
                y0 = self.offset[ax.yfield]
                ax.plot(xbounds, [y0,y0], 'k', lw=1.)
                ax.plot([x0,x0], ybounds, 'k', lw=1.)
        self.figure.show()

    def _mouseDown(self, event):
        ax = event.inaxes
        if not ax:
            return
        self.offset[ax.xfield] = event.xdata
        self.offset[ax.yfield] = event.ydata
        self.drawSubplot()
