import numpy as np
import matplotlib.pyplot as plt

from cosmoTransitions import pathDeformation as pd


class Potential:
    """
    A sample potential. The depth of the absolute minimum is controlled with
    the parameters `fx` and `fy`.

    This potential has no physical significance whatsoever.
    """
    def __init__(self, c=5., fx=10., fy=10.):
        self.params = c,fx,fy

    def V(self, X):
        """
        This is a two-dimensional potential, so the input should be some
        array with a *last* axis of length 2. That is, the final index in the
        array should always be the one that specifies the field (in this case
        *x* or *y*). This is the convention that CosmoTransitions uses
        throughout.
        """
        x,y = X[...,0], X[...,1]
        c, fx, fy = self.params
        r1 = x*x+c*y*y
        r2 = c*(x-1)**2 + (y-1)**2
        r3 = fx*(0.25*x**4 - x**3/3.)
        r3 += fy*(0.25*y**4 - y**3/3.)
        return r1*r2 + r3

    def dV(self, X):
        """
        The output of the gradient should have the same shape as the input.
        The last index specifies the direction of the gradient.
        """
        x,y = X[...,0], X[...,1]
        c, fx, fy = self.params
        r1 = x*x+c*y*y
        r2 = c*(x-1)**2 + (y-1)**2
        dr1dx = 2*x
        dr1dy = 2*c*y
        dr2dx = 2*c*(x-1)
        dr2dy = 2*(y-1)
        dVdx = r1*dr2dx + dr1dx*r2 + fx*x*x*(x-1)
        dVdy = r1*dr2dy + dr1dy*r2 + fy*y*y*(y-1)
        rval = np.empty_like(X)
        rval[...,0] = dVdx
        rval[...,1] = dVdy
        return rval

    def plotContour(self):
        nx = 100
        X = np.linspace(-.2,1.2,nx)[:,None] * np.ones((1,nx))
        Y = np.linspace(-.2,1.2,nx)[None,:] * np.ones((nx,1))
        XY = np.rollaxis(np.array([X,Y]), 0, 3)
        Z = self.V(XY)
        plt.contour(X,Y,Z, np.linspace(np.min(Z), np.max(Z)*.3, 200),
                    linewidths=0.5)


def makePlots():
    # Thin-walled instanton
    plt.figure()
    ax = plt.subplot(221)
    m = Potential(c=5, fx=0., fy=2.)
    m.plotContour()
    Y = pd.fullTunneling([[1,1.],[0,0]], m.V, m.dV)
    ax.plot(Y.Phi[:,0], Y.Phi[:,1], 'k', lw=1.5)
    ax.set_xlabel(r"$\phi_x$")
    ax.set_ylabel(r"$\phi_y$")
    ax.set_title("Thin-walled")
    ax = plt.subplot(223)
    ax.plot(Y.profile1D.R, Y.profile1D.Phi, 'r')
    ax.set_xlabel("$r$")
    ax.set_ylabel(r"$|\phi(r) - \phi_{\rm absMin}|$")

    # Thick-walled instanton
    ax = plt.subplot(222)
    m = Potential(c=5, fx=0., fy=80.)
    m.plotContour()
    Y = pd.fullTunneling([[1,1.],[0,0]], m.V, m.dV)
    ax.plot(Y.Phi[:,0], Y.Phi[:,1], 'k', lw=1.5)
    ax.set_xlabel(r"$\phi_x$")
    ax.set_title("Thick-walled")
    ax = plt.subplot(224)
    ax.plot(Y.profile1D.R, Y.profile1D.Phi, 'r')
    ax.set_xlabel("$r$")

    plt.show()


if __name__ == "__main__":
    makePlots()
