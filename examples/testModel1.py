
import numpy as np
from cosmoTransitions import generic_potential

v2 = 246.**2


class model1(generic_potential.generic_potential):
    """
    A sample model which makes use of the *generic_potential* class.

    This model doesn't have any physical significance. Instead, it is chosen
    to highlight some of the features of the *generic_potential* class.
    It consists of two scalar fields labeled *phi1* and *phi2*, plus a mixing
    term and an extra boson whose mass depends on both fields.
    It has low-temperature, mid-temperature, and high-temperature phases, all
    of which are found from the *getPhases()* function.
    """
    def init(self,m1=120.,m2=50.,mu=25.,Y1=.1,Y2=.15,n=30):
        """
          m1 - tree-level mass of first singlet when mu = 0.
          m2 - tree-level mass of second singlet when mu = 0.
          mu - mass coefficient for the mixing term.
          Y1 - Coupling of the extra boson to the two scalars individually
          Y2 - Coupling to the two scalars together: m^2 = Y2*s1*s2
          n - degrees of freedom of the boson that is coupling.
        """
        # The init method is called by the generic_potential class, after it
        # already does some of its own initialization in the default __init__()
        # method. This is necessary for all subclasses to implement.

        # This first line is absolutely essential in all subclasses.
        # It specifies the number of field-dimensions in the theory.
        self.Ndim = 2

        # self.renormScaleSq is the renormalization scale used in the
        # Coleman-Weinberg potential.
        self.renormScaleSq = v2

        # This next block sets all of the parameters that go into the potential
        # and the masses. This will obviously need to be changed for different
        # models.
        self.l1 = .5*m1**2/v2
        self.l2 = .5*m2**2/v2
        self.mu2 = mu**2
        self.Y1, self.Y2 = Y1, Y2
        self.n = n

    def forbidPhaseCrit(self, X):
        """
        forbidPhaseCrit is useful to set if there is, for example, a Z2 symmetry
        in the theory and you don't want to double-count all of the phases. In
        this case, we're throwing away all phases whose zeroth (since python
        starts arrays at 0) field component of the vev goes below -5. Note that
        we don't want to set this to just going below zero, since we are
        interested in phases with vevs exactly at 0, and floating point numbers
        will never be accurate enough to ensure that these aren't slightly
        negative.
        """
        return (np.array([X])[...,0] < -5.0).any()

    def V0(self, X):
        """
        This method defines the tree-level potential. It should generally be
        subclassed. (You could also subclass Vtot() directly, and put in all of
        quantum corrections yourself).
        """
        # X is the input field array. It is helpful to ensure that it is a
        # numpy array before splitting it into its components.
        X = np.asanyarray(X)
        # x and y are the two fields that make up the input. The array should
        # always be defined such that the very last axis contains the different
        # fields, hence the ellipses.
        # (For example, X can be an array of N two dimensional points and have
        # shape (N,2), but it should NOT be a series of two arrays of length N
        # and have shape (2,N).)
        phi1,phi2 = X[...,0], X[...,1]
        r = .25*self.l1*(phi1*phi1-v2)**2 + .25*self.l2*(phi2*phi2-v2)**2
        r -= self.mu2*phi1*phi2
        return r

    def boson_massSq(self, X, T):
        X = np.array(X)
        phi1,phi2 = X[...,0], X[...,1]

        # We need to define the field-dependnet boson masses. This is obviously
        # model-dependent.
        # Note that these can also include temperature-dependent corrections.
        a = self.l1*(3*phi1*phi1 - v2)
        b = self.l2*(3*phi2*phi2 - v2)
        A = .5*(a+b)
        B = np.sqrt(.25*(a-b)**2 + self.mu2**2)
        mb = self.Y1*(phi1*phi1+phi2*phi2) + self.Y2*phi1*phi2
        M = np.array([A+B, A-B, mb])

        # At this point, we have an array of boson masses, but each entry might
        # be an array itself. This happens if the input X is an array of points.
        # The generic_potential class requires that the output of this function
        # have the different masses lie along the last axis, just like the
        # different fields lie along the last axis of X, so we need to reorder
        # the axes. The next line does this, and should probably be included in
        # all subclasses.
        M = np.rollaxis(M, 0, len(M.shape))

        # The number of degrees of freedom for the masses. This should be a
        # one-dimensional array with the same number of entries as there are
        # masses.
        dof = np.array([1, 1, self.n])

        # c is a constant for each particle used in the Coleman-Weinberg
        # potential using MS-bar renormalization. It equals 1.5 for all scalars
        # and the longitudinal polarizations of the gauge bosons, and 0.5 for
        # transverse gauge bosons.
        c = np.array([1.5, 1.5, 1.5])

        return M, dof, c

    def approxZeroTMin(self):
        # There are generically two minima at zero temperature in this model,
        # and we want to include both of them.
        v = v2**.5
        return [np.array([v,v]), np.array([v,-v])]


def makePlots(m=None):
    import matplotlib.pyplot as plt
    if m is None:
        m = model1()
        m.findAllTransitions()
    # --
    plt.figure()
    m.plotPhasesPhi()
    plt.axis([0,300,-50,550])
    plt.title("Minima as a function of temperature")
    plt.show()
    # --
    plt.figure(figsize=(8,3))
    ax = plt.subplot(131)
    T = 0
    m.plot2d((-450,450,-450,450), T=T, cfrac=.4,clevs=65,n=100,lw=.5)
    ax.set_aspect('equal')
    ax.set_title("$T = %0.2f$" % T)
    ax.set_xlabel(R"$\phi_1$")
    ax.set_ylabel(R"$\phi_2$")
    ax = plt.subplot(132)
    T = m.TnTrans[1]['Tnuc']
    instanton = m.TnTrans[1]['instanton']
    phi = instanton.Phi
    m.plot2d((-450,450,-450,450), T=T, cfrac=.4,clevs=65,n=100,lw=.5)
    ax.plot(phi[:,0], phi[:,1], 'k')
    ax.set_aspect('equal')
    ax.set_title("$T = %0.2f$" % T)
    ax.set_yticklabels([])
    ax.set_xlabel(R"$\phi_1$")
    ax = plt.subplot(133)
    T = m.TnTrans[0]['Tnuc']
    m.plot2d((-450,450,-450,450), T=T, cfrac=.4,clevs=65,n=100,lw=.5)
    ax.set_aspect('equal')
    ax.set_title("$T = %0.2f$" % T)
    ax.set_yticklabels([])
    ax.set_xlabel(R"$\phi_1$")
    # --
    plt.figure()
    plt.plot(instanton.profile1D.R, instanton.profile1D.Phi)
    plt.xlabel("radius")
    plt.ylabel(R"$\phi-\phi_{min}$ (along the path)")
    plt.title("Tunneling profile")
