"""
Path deformation from CT
========================

Tests for path deformation applied to a 2d potential.

The potential is eq. 13 in https://arxiv.org/pdf/1109.4189.pdf parameterised
by the variable delta. The true vacuum was found by Mathematica in the below
test cases.

>>> action_2d(1., [1.06561, 2.24282])
2.0901...

>>> action_2d(0.5, [1.05114, 1.77899])
8.7948...

>>> action_2d(0.4, [1.04637, 1.66349])
12.972...

>>> action_2d(0.3, [1.0403, 1.53521])
21.024...

>>> action_2d(0.2, [1.03213, 1.38923])
39.648...

>>> action_2d(0.1, [1.0202, 1.21683])
119.83...

>>> action_2d(0.01, [1.00267, 1.02459])
7853.5...

>>> action_2d(0.001, [1.00028, 1.0025])
639494.2...
"""

from cosmoTransitions.pathDeformation import fullTunneling as path_deform
import numpy as np


def action_2d(delta, true_vacuum):
    """
    :param alpha: Parametrises shape of 1d potential
    :param E: Parameteries height of 1d potential

    :returns: Action from path deformation
    """
    def V(fields):
        """
        :returns: 2d potential
        """
        f, g = fields[..., 0], fields[..., 1]
        return (f**2 + g**2) * (1.8 * (f - 1.)**2 + 0.2 * (g - 1.)**2 - delta)

    def dV(fields):
        """
        :returns: Derivative of 2d potential
        """
        f, g = fields[..., 0], fields[..., 1]
        df = -10.8 * f**2 + 7.2 * f**3 - 3.6 * g**2 - 2. * f * (-2. + delta + 0.4 * g - 2. * g**2)
        dg = -7.2 * f * g + f**2 * (-0.4 + 4. * g) + g * (4. - 2. * delta - 1.2 * g + 0.8 * g**2)
        rval = np.empty_like(fields)
        rval[..., 0] = df
        rval[..., 1] = dg
        return rval

    n_points = 100
    guess_phi = np.array([np.linspace(t, 0., n_points) for t in true_vacuum]).T

    kwargs = {'deformation_deform_params': {'verbose': False}}

    profile_1d, phi, action = path_deform(guess_phi, V, dV, **kwargs)[:3]

    return action


if __name__ == '__main__':
    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS ^ doctest.REPORT_NDIFF)
