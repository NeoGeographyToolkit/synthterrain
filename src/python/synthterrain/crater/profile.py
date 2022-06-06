"""This module contains functions described by
Martin, Parkes, and Dunstan (2014),
https:doi.org/10.1109/TAES.2014.120282
."""

# Copyright 2022, synthterrain developers.
#
# Reuse is permitted under the terms of the license.
# The AUTHORS file and the LICENSE file are at the
# top level of this library.

# Consider attempting to implement the Mahanti et al. (2014,
# https://doi.org/10.1016/j.icarus.2014.06.023) Chebyshev polynomial approach.


import math

import numpy as np
from numpy.polynomial import Polynomial


class Crater():
    """A base class for establishing characteristics for a crater, in order
    to query its elevation at particular radial distances."""

    def __init__(
        self,
        diameter,
    ):
        self.diameter = diameter

    def r(self):
        return self.diameter / 2

    def profile(self, r):
        """Implementing classes must override this function.
           This function returns a numpy array of the same shape as
           *r*.
           This function returns the elevation of the crater profile
           at the radial distance *r* where *r* is a fraction of the
           crater radius.  Such that *r* = 0 is at the center, *r* = 1
           is at the crater diameter, and values of *r* greater than 1
           are distances outside the crater rim.

           Values returned in the numpy array are elevation values
           in the distance units of whatever units the self.diameter
           parameter is in.  Values of zero are considered pre-existing
           surface elevations.
        """
        raise NotImplementedError(
            f"The class {self.__name__} has not implemented elevation() as "
            "required."
        )


class FT_Crater(Crater):
    """A crater whose profile is defined by functions described in
       Fassett and Thomson (2014, https://doi.org/10.1002/2014JE004698),
       equation 4.
    """

    def profile(self, r, radius_fix=True):
        """Returns a numpy array of elevation values based in the input numpy
           array of radius fraction values, such that a radius fraction value
           of 1 is at the rim, less than that interior to the crater, etc.

           A ValueError will be thrown if any values in r are < 0.

           The Fassett and Thomson (2014) paper defined equations which
           placed the rim point at a radius fraction of 0.98, but that
           results in a crater with a smaller diameter than specifed.
           If radius_fix is True (the default) the returned profile will
           extend the interior slope and place the rim at radius fraction
           1.0, but this may cause a discontinuity at the rim.  If you
           would like a profile with the original behavior, set radius_fix
           to False.
        """

        if not isinstance(r, np.ndarray):
            r = np.ndarray(r)

        out_arr = np.zeros_like(r)

        if np.any(r < 0):
            raise ValueError(
                "The radius fraction value can't be less than zero."
            )

        # In F&T (2014) the boundary between inner and outer was at 0.98
        # which put the rim not at r=1, Caleb's subsequent code revised
        # this position to 1.0.
        if radius_fix:
            rim = 1.0
        else:
            rim = 0.98

        flat_idx = np.logical_and(0 <= r, r <= 0.2)
        inner_idx = np.logical_and(0.2 < r, r <= rim)
        outer_idx = np.logical_and(rim < r, r <= 1.5)

        inner_poly = Polynomial(
            [-0.228809953, 0.227533882, 0.083116795, -0.039499407]
        )
        outer_poly = Polynomial(
            [0.188253307, -0.187050452, 0.01844746, 0.01505647]
        )

        out_arr[flat_idx] = self.diameter * -0.181
        out_arr[inner_idx] = self.diameter * inner_poly(r[inner_idx])
        out_arr[outer_idx] = self.diameter * outer_poly(r[outer_idx])

        return out_arr


class FTmod_Crater(Crater):
    """
    This crater profile is based on Fassett and Thomson
    (2014, https://doi.org/10.1002/2014JE004698), equation 4, but modified
    in 2022 by Caleb Fassett (pers. comm).

    An optional *depth* parameter can be given to the constructor which
    specifies the initial depth of the crater.  If no *depth* is given,
    the "Stopar step" value for fresh craters of the given diameter is used.

    The modifications to the Fassett and Thomson (2014) shape are at the rim
    and at the floor.  Rather than being a sharp transition from crater
    interior to crater exterior, there is now a "flat" rim from 0.98 relative
    crater diameter to 1.02.

    The specification of the "depth" parameter will set the level of the flat
    floor in the middle of the crater.  In practice, this means that for
    smaller craters, the flat floor will have a larger relative radius than
    larger craters, because the d/D ratios for smaller craters are smaller,
    thus shallower, thus larger (relative) flat floors.
    """

    def __init__(
        self,
        diameter,
        depth=None
    ):
        super().__init__(diameter)
        if depth is None:
            self.depth = stopar_fresh_dd(self.diameter) * self.diameter
        else:
            self.depth = depth

    def profile(self, r):
        """Returns a numpy array of elevation values based in the input numpy
           array of radius fraction values, such that a radius fraction value
           of 1 is at the rim, less than that interior to the crater, etc.

           A ValueError will be thrown if any values in r are < 0.
        """

        if not isinstance(r, np.ndarray):
            r = np.ndarray(r)

        out_arr = np.zeros_like(r)

        if np.any(r < 0):
            raise ValueError(
                "The radius fraction value can't be less than zero."
            )

        inner_idx = np.logical_and(0 <= r, r <= 0.98)
        rim_idx = np.logical_and(0.98 < r, r <= 1.02)
        outer_idx = np.logical_and(1.02 < r, r <= 1.5)

        inner_poly = Polynomial(
            [-0.228809953, 0.227533882, 0.083116795, -0.039499407]
        )
        outer_poly = Polynomial(
            [0.188253307, -0.187050452, 0.01844746, 0.01505647]
        )

        rim_hoverd = 0.036822095

        out_arr[inner_idx] = inner_poly(r[inner_idx])
        out_arr[rim_idx] = rim_hoverd
        out_arr[outer_idx] = outer_poly(r[outer_idx])

        floor = rim_hoverd - (self.depth / self.diameter)
        out_arr[out_arr < floor] = floor

        return out_arr * self.diameter


class MPD_Crater(Crater):
    """A crater whose profile is defined by functions described in
       Martin, Parkes, and Dunstan (2014,
       https:doi.org/10.1109/TAES.2014.120282).  The published equations
       for beta and h3 result in non-realistic profiles.  For this class,
       the definition of beta has been adjusted so that it is a positive value
       (which we think was intended).  We have also replaced the published
       function for h3, with a cubic that actually matches up with h2 and h4,
       although the matching with h4 is imperfect, so there is likely a better
       representation for h3.
    """

    def __init__(
        self,
        diameter,
        depth,
        rim_height=None,
        emin=0,  # height of the ejecta at x = D/2
        pre_rim_elevation=0,
        plane_elevation=0
    ):
        self.h0 = self.height_naught(diameter)
        self.hr0 = self.height_r_naught(diameter)
        self.h = depth
        if rim_height is None:
            self.hr = self.hr0
        else:
            self.hr = rim_height

        self.emin = emin

        self.tr = pre_rim_elevation
        self.pr = plane_elevation

        # print(self.hr)
        # print(self.h)
        # print(self.hr0)
        # print(self.tr)
        # print(self.pr)

        super().__init__(diameter)

    def profile(self, r: float):
        return self.profile_x(r - 1)

    def profile_x(self, x: float):
        err_msg = (
            "The value of x must be greater than -1, as defined in "
            "Martin, Parkes, and Dunstan (2012), eqn 3."
        )
        if not -1 <= x:
            raise ValueError(err_msg)

        alpha = self.alpha(self.hr, self.h, self.hr0, self.tr, self.pr)
        beta = self.beta(self.hr, self.h, self.hr0, self.tr, self.pr)

        if -1 <= x <= alpha:
            return self.h1(x, self.hr, self.h, self.hr0)

        elif alpha <= x <= 0:
            return self.h2(
                x, self.hr, self.h, self.hr0, alpha, self.tr, self.pr
            )
        elif 0 <= x <= beta:
            # return self.h3(
            return self.h3_alt(
                self.diameter, self.emin,
                x, self.hr, self.h, self.hr0, alpha, beta,
                self.tr, self.pr
            )
        elif beta <= x:
            return self.h4(
                x, self.diameter, self.fc(x, self.emin, self.tr, self.pr)
            )

        else:
            # Really should not be able to get here.
            raise ValueError(err_msg)

    @staticmethod
    def height_naught(diameter: float):
        """H_0 as defined by Melosh, 1989. Eqn 1 in Martin, Parkes, and Dunstan."""
        return 0.196 * math.pow(diameter, 1.01)

    @staticmethod
    def height_r_naught(diameter: float):
        """H_r0 as defined by Melosh, 1989. Eqn 2 in Martin, Parkes, and
        Dunstan."""
        return 0.036 * math.pow(diameter, 1.01)

    @staticmethod
    def h1(x: float, hr: float, h: float, hr_naught: float):
        """Eqn 4 in Martin, Parkes, and Dunstan."""
        h_ = (hr_naught - hr + h)
        return (h_ * math.pow(x, 2)) + (2 * h_ * x) + hr_naught

    @staticmethod
    def h2(
        x: float, hr: float, h: float, hr_naught: float, alpha: float,
        tr=0, pr=0
    ):
        """Eqn 5 in Martin, Parkes, and Dunstan."""
        h_ = (hr_naught - hr + h)
        return ((h_ * (alpha + 1)) / alpha) * math.pow(x, 2) + hr + tr - pr

    @staticmethod
    def alpha(hr: float, h: float, hr_naught: float, tr=0, pr=0):
        """Eqn 6 in Martin, Parkes, and Dunstan."""
        # print(f"{hr}, {h}, {hr_naught}, {tr}, {pr}")
        return (hr + tr - pr - hr_naught) / (hr_naught - hr + h)

    @staticmethod
    def h3(
        x: float, hr: float, h: float, hr_naught: float,
        alpha: float, beta: float, tr=0, pr=0
    ):
        """Eqn 7 in Martin, Parkes, and Dunstan."""
        h_ = (hr_naught - hr + h)
        t1 = -1 * ((2 * h_) / (3 * math.pow(beta, 2))) * math.pow(x, 3)
        t2 = (h_ + ((2 * h_) / beta)) * math.pow(x, 2)
        return t1 + t2 + hr + tr - pr

    @staticmethod
    def h3_alt(
        diameter, emin,
        x: float, hr: float, h: float, hr_naught: float,
        alpha: float, beta: float, tr=0, pr=0
    ):
        """Improved cubic form."""
        # ax^3 + bx ^ 2 + cx + d = elevation
        # At x = 0, the cubic should be H_r, so d = H_r
        # The the positive critical point, should be at x=0, which
        # implies that c = 0.
        # The inflection point should be where this function meets up
        # with h4, so that means that the inflection point is at x = beta
        h4_at_beta = MPD_Crater.h4(
            beta, diameter, MPD_Crater.fc(beta, emin, tr, pr)
        )
        a = (hr - h4_at_beta) / (2 * math.pow(beta, 3))
        b = -3 * a * beta
        cubic = Polynomial([hr, 0, b, a])
        return cubic(x)

    @staticmethod
    def beta(hr: float, h: float, hr_naught: float, tr=0, pr=0):
        """Eqn 8 in Martin, Parkes, and Dunstan."""
        h_ = (hr_naught - hr + h)
        # This changes the order of hr_naught and hr from the
        # paper, as this ensures that this term will be positive.
        return (3 * (hr_naught - hr + tr - pr)) / (2 * h_)

    @staticmethod
    def h4(x: float, diameter: float, fc: float):
        """Eqn 9 in Martin, Parkes, and Dunstan."""
        return 0.14 * pow(diameter / 2, 0.74) * pow(x + 1, -3) + fc

    @staticmethod
    def fc(x: float, emin: float, tr=0, pr=0):
        return ((emin + tr - pr) * x) + (2 * (pr - tr)) - emin


def stopar_fresh_dd(diameter):
    """
    Returns a depth/Diameter ratio based on the set of graduated d/D
    categories in Stopar et al. (2017), defined down to 40 m.  This
    function also adds two extrapolated categories.
    """
    # The last two elements are extrapolated
    d_lower_bounds = (400, 200, 100, 40, 10, 0)
    dds = (0.21, 0.17, 0.15, 0.13, 0.11, 0.10)
    for d, dd in zip(d_lower_bounds, dds):
        if diameter >= d:
            return dd
    else:
        raise ValueError(f"Diameter was less than zero: {diameter}.")
