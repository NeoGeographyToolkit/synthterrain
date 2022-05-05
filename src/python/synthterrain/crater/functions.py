#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module contains abstract and concrete classes for representing
crater size-frequency distributions as probability distributions.
"""

# Copyright 2022, synthterrain developers.
#
# Reuse is permitted under the terms of the license.
# The AUTHORS file and the LICENSE file are at the
# top level of this library.

from abc import ABC, abstractmethod
import copy
import logging
import math
from numbers import Number

import numpy as np
from numpy.polynomial import Polynomial
from scipy.stats import rv_continuous

logger = logging.getLogger(__name__)

# If new equilibrium functions are added, add them to the equilibrium_functions
# list below the class definitions to expose them to users.


class Crater_rv_continuous(ABC, rv_continuous):
    """Base class for crater continuous distributions.  Provides
       some convenience functions common to all crater distributions.

       Crater distribution terminology can be reviewed in Robbins, et al. (2018,
       https://doi.org/10.1111/maps.12990), which in its
       terminology section states "In crater work, the CSFD can be thought
       of as a scaled version of "1−CDF."

       CSFD is the crater size-frequency distribution, a widely used
       representation in the scientific literature.  CDF is the statistical
       cumulative distribution function.

       Both the CSFD and the CDF are functions of d (the diameter of craters),
       and are related thusly, according to Robbins et al.:

           CSFD(d) ~ 1 - CDF(d)

       For any particular count of craters, the smallest crater value
       measured (d_min) gives the total number of craters in the set per
       unit area, CSFD(d_min).  Which implies this relation

           CDF(d) = 1 - (CSFD(d) / CSFD(d_min))

       If you scale by CSFC(d_min) which is the total number of craters,
       then you get a statistical CDF.

       When creating concrete classes that descend from Crater_rv_continuous,
       the csfd() function must be implemented.  There is a default
       implementation of ._cdf(), but it is advised that it be implemented
       directly for speed and efficiency.  It is also heavily advised
       (echoing the advice from scipy.stats.rv_continuous itself) that _ppf()
       also be directly implemented.

       It is assumed that the units of diameters (d) are in meters, and
       that the resulting CSFD(d) is in units of number per square meter.
       Implementing functions should support this.
    """

    def __init__(self, a, **kwargs):
        if a <= 0:
            raise ValueError(
                "The lower bound of the support of the distribution, a, must "
                "be > 0."
            )
        else:
            kwargs["a"] = a
            super().__init__(**kwargs)

    @abstractmethod
    def csfd(self, d):
        """
        Implementing classes must define this function which is the
        crater size-frequency distribution (CSFD, e.g.  defined in
        Robbins, et al. 2018, https://doi.org/10.1111/maps.12990).

        The argument *d* should could be a single value, a collection
        of values, or a numpy array representing diameters in meters.

        Returns a numpy array of the result of applying the CSFD to
        the diameters provided in *d*.
        """
        pass

    # Experiments with ISFD formulated in this manner produce results that
    # indicate there is something that I'm not formulating correctly, so
    # I'm commenting out these functions for now.
    #
    # def isfd(self, d):
    #     """
    #     The incremental size frequency distribution (ISFD, e. g.
    #     defined in Robbins, et al. 2018, https://doi.org/10.1111/maps.12990)
    #     is typically the actual, discrete counts, which are then used to
    #     construct a CSFD.  For the purposes of this class, since the CSFD
    #     is the primary definition, and is a continuous function, there is
    #     a continuous function that describes the ISFD, which is the negative
    #     derivative of the CSFD function.  And this is how classes that descend
    #     from this one should implement this function.
    #     """
    #     raise NotImplementedError(
    #         f"The isfd() function is not implemented for {_class__.__name__}.")

    def _cdf(self, d):
        """Returns an array-like which is the result of applying the
           cumulative density function to *d*, the input array-like of
           diameters.

           If the crater size-frequency distribution (CSFD) is C(d) (typically
           also expressed as N_cumulative(d) ), then

               cdf(d) = 1 - (C(d) / C(d_min))

           In the context of this class, d_min is a, the lower bound of the
           support of the distribution when this class is instantiated.

           As with the parent class, rv_continuous, implementers of derived
           classes are strongly encouraged to override this with more
           efficient implementations, also possibly implementing _ppf().
        """
        return np.ones_like(d) - (self.csfd(d) / self.csfd(self.a))

    def rvs(self, *args, **kwargs):
        """Overrides the parent rvs() function by adding an *area*
           parameter, all other arguments are identical.

           If an *area* parameter is provided, it is interpreted as the
           area in square meters which has accumulated craters.

           Specifying it will cause the *size* parameter (if given) to
           be overridden such that

                size = CSDF(d_min) * area

           and then the parent rvs() function will be called.

           Since the CSDF interpreted at the minimum crater size is
           the total number of craters per square meter, multiplying
           it by the desired area results in the desired number of craters.
        """
        if "area" in kwargs:
            kwargs["size"] = int(self.csfd(self.a) * kwargs["area"])
            del kwargs["area"]

        return super().rvs(*args, **kwargs)


class Test_Distribution(Crater_rv_continuous):
    """This is testing a simple function.
    """

    def csfd(self, d):
        """Returns the crater cumulative size frequency distribution function
           such that
                CSFD(d) = N_cum(d) = 29174 / d^(1.92)
        """
        return 29174 * np.float_power(d, -1.92)

    def _cdf(self, d):
        """Override of parent function to eliminate unnecessary division
           of 29174 by itself.
        """
        return np.ones_like(d) - (
            np.float_power(d, -1.92) / np.float_power(self.a, -1.92)
        )


class VIPER_Env_Spec(Crater_rv_continuous):
    """
    This distribution is from the VIPER Environmental Specification,
    VIPER-MSE-SPEC-001 (2021-09-16).  Sadly, no citation is provided
    for it in that document.

    The equation there is written such that diameters are in meters,
    but that the CSFD(d) is in number per square kilometer.  This class
    returns CSFD(d) in number per square meter.
    """

    def csfd(self, d):
        """
            CSFD( d <= 80 ) = (29174 / d^(1.92)) / (1000^2)
            CSFD( d > 80 ) = (156228 / d^(2.389)) / (1000^2)

        """
        if isinstance(d, Number):
            # Convert to numpy array, if needed.
            d = np.array([d, ])
        c = np.empty_like(d)
        c[d <= 80] = 29174 * np.float_power(d[d <= 80], -1.92)
        c[d > 80] = 156228 * np.float_power(d[d > 80], -2.389)
        return c / (1000 * 1000)

    # See comment on commented out parent isfd() function.
    # def isfd(self, d):
    #     """
    #     Returns the incremental size frequency distribution for diameters, *d*.
    #     """
    #     if isinstance(d, Number):
    #         # Convert to numpy array, if needed.
    #         d = np.array([d, ])
    #     c = np.empty_like(d)
    #     c[d <= 80] = 29174 * 1.92 * np.float_power(d[d <= 80], -2.92)
    #     c[d > 80] = 156228 * 2.389 * np.float_power(d[d > 80], -3.389)
    #     return c / (1000 * 1000)

    def _cdf(self, d):
        """Override parent function to eliminate unnecessary division
           by constants.
        """
        c = np.empty_like(d)
        c[d <= 80] = np.float_power(d[d <= 80], -1.92) / np.float_power(self.a, -1.92)
        c[d > 80] = np.float_power(d[d > 80], -2.389) / np.float_power(self.a, -2.389)
        return np.ones_like(d) - c

    def _ppf(self, q):
        """Override parent function to make things faster for .rvs()."""
        q80 = float(self._cdf(np.array([80, ])))
        ones = np.ones_like(q)
        p = np.empty_like(q)
        p[q <= q80] = np.float_power(
            (ones[q <= q80] / (ones[q <= q80] - q[q <= q80])),
            (1 / 1.92)
        )
        p[q > q80] = np.float_power(
            (ones[q > q80] / (ones[q > q80] - q[q > q80])),
            (1 / 2.389)
        )

        return self.a * p


class Trask(Crater_rv_continuous):
    """
    This describes an equilibrium function based on Trask's
    contribution, "Size and Spatial Distribution of Craters Estimated
    From the Ranger Photographs," in E. M. Shoemaker et al. (1966)
    Ranger VIII and IX:  Part II.  Experimenters’ Analyses and
    Interpretations.  JPL Technical Report 32-800, 382 pages.
    Available at
    https://www.lpi.usra.edu/lunar/documents/RangerVIII_and_IX.pdf
    Also described as "Standard lunar equilibrium (Trask, 1966)" in
    the craterstats package.

    In the 1966 work, it is written

         N = 10^(10.9) * d^(-2)

    Where N is cumulative number of craters per 10^6 km^2 and diameters
    greater than d (in meters).

    It is uncertain what range of crater diameters this simple relation
    holds for.
    """

    def csfd(self, d):
        """
        Returns the crater cumulative size frequency distribution function
        such that
                CSFD(d) = N_cum(d) = 10^(-1.1) * d^(-2)

        The exponent has been adjusted from 10.9 to -1.1 so this function
        returns counts per square meter.
        """
        return math.pow(10, -1.1) * np.float_power(d, -2)

    # See comment on commented out parent isfd() function.
    # def isfd(self, d):
    #     """
    #     Returns the value of the incremental size frequency distribution
    #     function such that the ISFD is the negative derivative of the CSFD:

    #         ISFD(d) = 10^(-1.1) * 2 * d^(-3)

    #     """
    #     return math.pow(10, -1.1) * 2 * np.float_power(d, -3)

    # def isfd_experimental(self, d):
    #     """
    #     Experimental
    #     """
    #     # Rather than attempt to do fancy math (which I may have well screwed
    #     # up) to derive a continuous function for isfd, let's just
    #     # "unintegrate" the csfd maybe ?
    #     i = self.csfd(d)
    #     i[:-1] = i[:-1] - self.csfd(d[1:])
    #     return i


class Coef_Distribution(Crater_rv_continuous):
    """This class instantiates a continuous crater distribution based
       on a polynomial.  This notation for a crater distribution is
       used by Neukum et al. (2001, https://doi.org/10.1023/A:1011989004263)
       and in the craterstats package.

       The coefficients generally assume that the diameter values are in
       kilometers, and the math here is based on that, but only matters for
       the specification of the coefficients.  The diameter values passed
       to csfd() are expected to be in meters, and the returned value
       is number per square meter.
    """

    def __init__(self, *args, coef=None, poly=None, **kwargs):
        if coef is None and poly is None:
            raise ValueError(
                "A Coef_Distribution object must be initiated with a "
                "*coef* array-like of coefficients from which a polynomial "
                "will be constructed, or a *poly* object which must be a "
                "numpy Polynomial object."
            )
        super().__init__(*args, **kwargs)

        if coef is not None:
            poly = Polynomial(coef)

        self.poly = poly

    def csfd(self, d):
        """Returns the crater cumulative size frequency distribution function
           such that
                CSFD(d) = N_cum(d) = 10^x / (1000 * 1000)

           where x is the summation of j from zero to n (typically ~ 11) of
                a_j * ( lg(d/1000) )^j

           where lg() is the base 10 logarithm, and the values a_j are the
           coefficients provided via the constructor.

           Since published coefficients are typically provided for diameter
           values in kilometers and areas in square kilometers, the equation
           for CSFD(d) is adjusted so that diameters can be provided in units
           of meters, and CSFD(d) is returned in counts per square meter.
        """
        # The 1000s are to take diameters in meters, convert to kilometers
        # for application by the polynomial, and then division by a square
        # kilometer to get the number per square meter.
        # lg(d / 1000) = lg( d * (1/1000)) = lg(d) + lg(1/1000) = lg(d) - 3
        # return np.power(10, self.poly(np.log10(d / 1000))) / (1000 * 1000)
        return np.float_power(10, self.poly(np.log10(d / 1000))) / (1000 * 1000)

    # See comment on commented out parent isfd() function.
    # def isfd(self, d):
    #     """
    #     Returns the incremental size frequency distribution or count for
    #     diameter *d*.

    #     This requires properly taking the derivative of the csfd() function
    #     which is:

    #         CSFD(d) = 10^x / (1000 * 1000)

    #     where "x" is really a polynomial function of d:

    #         x(d) = a_0 + a_1 * ( lg(d/1000) ) + a_2 * ( lg(d/1000) )^2 + ...

    #     So this means that CSFD(d) can be formulated as CSFD(x(d)), and the
    #     derivative of CSFD with respect to d is denoted using Lagrange's
    #     "prime" notation:

    #         ISFD(d) = CSFD'(d) = CSFD'(x(d)) * x'(d)

    #         CSFD'(x) = [x * 10^(x-1)] / (1000 * 1000)

    #         x'(d) = [a_1 / (d * ln(10))] + [2 * a_2 / (d * ln(10))] +
    #                     [3 * a_3 / (d * ln(10))^2] + ...

    #     For compactness in x'(d), assume the d values on the right-hand side
    #     have been pre-divided by 1000.  The natural log is denoted by "ln()".

    #     Assuming I've done the differentiation correctly, x'(d) is a
    #     polynomial with coefficients a_1, 2 * a_2, 3 * a_3,
    #     etc. that we can construct from the original polynomial, and the
    #     values of the variables are 1 / (d/1000) * ln(10).
    #     """
    #     # x = self.poly(np.log10(d) - 3)
    #     # csfd_prime = x * np.power(10, x - 1) / (1000 * 1000)

    #     # new_coefs = list()
    #     # for i, a in enumerate(self.poly.coef[1:], start=1):
    #     #     new_coefs.append(i * a)

    #     # x_prime = Polynomial(new_coefs)

    #     # return csfd_prime * x_prime(
    #     #     np.reciprocal(
    #     #         (d / 1000) * np.log(10)
    #     #     )
    #     # )

    #     # Okay, Neukum indicates that differentiating the CSFD by d is
    #     #   ISFD(d) = -1 * CSFD'(d) = -1 * [CSFD(d) / d] * x(d)
    #     #   where x(d) = a_1 + a_2 * ( lg(d) ) + a_3 * ( lg(d) )^2

    #     # new_coefs = list()
    #     # for i, a in enumerate(self.poly.coef[1:], start=1):
    #     #     new_coefs.append(i * a)
    #     # x_prime = Polynomial(new_coefs)
    #     x_prime = Polynomial(self.poly.coef[1:])

    #     # return -1 * (self.csfd(d) / (d / 1000)) * x_prime(np.log10(d / 1000))
    #     return np.absolute(
    #         (self.csfd(d) / (d / 1000)) * x_prime(np.log10(d / 1000))
    #     )

    def _cdf(self, d):
        """Override parent function to speed up."""
        return np.ones_like(d) - np.float_power(
            10,
            self.poly(np.log10(d / 1000)) - self.poly(np.log10(self.a / 1000))
        )


class NPF(Coef_Distribution):
    """
    This describes the Neukum et al. (2001,
    https://doi.org/10.1023/A:1011989004263) production function (NPF)
    defined in their equation 2 and with coefficients from the '"New" N(D)'
    column in their table 1.

    The craterstats package notes indicate that the published a0 value
    (-3.0876) is a typo, and uses -3.0768, which we use here.

    In this case, CSFD(N) is the cumulative number of craters per square
    area per Gyr.  So outputs from csfd() and isfd() must be multiplied
    by 10**9 to get values per year.

    Note that equation 2 is valid for diameters from 0.01 km to 300 km,
    set the *a* and *b* parameters appropriately (>=10, <= 300,000).
    """

    def __init__(self, a, b, **kwargs):

        if a < 10:
            raise ValueError(
                "The lower bound of the support of the distribution, a, must "
                "be >= 10."
            )
        if b > 300000:
            raise ValueError(
                "The upper bound of the support of the distribution, b, must "
                "be <= 300,000."
            )

        kwargs["a"] = a
        kwargs["b"] = b
        super().__init__(
            coef=[
                -3.0768, -3.557528, 0.781027, 1.021521, -0.156012, -0.444058,
                0.019977, 0.086850, -0.005874, -0.006809, 8.25e-04, 5.54e-05
            ],
            **kwargs
        )


class Grun(Coef_Distribution):
    """
    Grun et al. (1985, https://doi.org/10.1016/0019-1035(85)90121-6)
    describe a small particle impact flux, which can be converted into
    a production function for small craters, that matches well with the
    value of the Neukum et al. (2001) production function at d=10 m.
    """

    def __init__(self, a, b, **kwargs):
        if b > 10:
            raise ValueError(
                "The upper bound of the support of the distribution, b, must "
                "be <= 10."
            )
        kwargs["a"] = a
        kwargs["b"] = b

        # These two arrays are from Caleb Fassett, pers. comm., calculated
        # in the following manner.

        # Grun et al. equation A2 describes fluxes in terms of mass.
        # To convert mass, m, of a particle to diameter of crater, we first
        # assume a density, rho, of 2.5 g/cm^-3, and calculate a particle radius
        # by assuming spherical particles:
        #
        #   m / rho = (4 / 3) * pi * r^3
        #
        #   r = [ (3 * m) / (4 * pi * rho) ] ^(1/3)
        #
        # Housen & Holsapple (2011) scaling is then applied to these radii
        # to generate crater diameters.
        #
        # With masses from 10^-18 to 10^6 grams, and calcuating a diameter
        # at every mass decade, we get these 24 crater diameters in meters:
        diameters = np.array([
            6.91E-07, 1.49E-06, 3.21E-06, 6.91E-06, 1.49E-05,
            3.21E-05, 6.91E-05, 0.000148787, 0.000320552, 0.000690607,
            0.001487869, 0.003205516, 0.006906074, 0.014878687, 0.03205516,
            0.069060751, 0.148786877, 0.32055161, 0.690607508, 1.487868771,
            3.205516094, 6.906075072, 14.87868771, 32.05516094
        ]) / 1000
        # The Coef_Distribution polynomial needs diameters
        # in kilometers, so divide by 1000.

        # Now we can use Grun et al. equation A2 to calculate the flux in
        # number m^-2 s^-1.  Those values can be converted to units of
        # number km^-2 yr^-1.

        # The Grun mass range maxes out at 100 g, which only gets us to about
        # three meters in diameter, but we can extrapolate the Neukum
        # production function to match up with the fluxes calculated in this
        # manner.

        # These are in /km^2 /yr, so to convert to /km^2 /Gyr
        fluxes = 1e9 * np.array([
            3.26E+14, 2.14E+14, 3.13E+13, 3.44E+12, 3.75E+11,
            4.11E+10, 4.66E+09, 6.58E+08, 1.85E+08, 8.72E+07,
            3.58E+07, 9.46E+06, 1.48E+06, 1.46E+05, 1.03E+04,
            5.97E+02, 3.07E+01, 1.49E+00, 7.02E-02, 3.26E-03,
            1.50e-4, 1.46e-5, 1.15e-6, 1.07e-7
        ])
        # The final two elements are from Neukum, and the third to the last
        # is extrapolated from Neukum, but this allows for a smooth
        # transition.

        p = Polynomial.fit(np.log10(diameters), np.log10(fluxes), 11)

        super().__init__(
            poly=p,
            **kwargs
        )


class GNPF(NPF):
    """
    This describes a combination function such that it functions as a Neukum
    Production Function (NPF) for the size ranges where NPF is appropriate,
    and as a Grun function where that is appropriate.
    """

    def __init__(self, a, b, **kwargs):
        if b <= 10:
            raise ValueError(
                f"The upper bound, b, is {b}, you should use Grun, not GNPF."
            )

        if a >= 10:
            raise ValueError(
                f"The lower bound, a, is {a}, you should use NPF, not GNPF."
            )

        # Will now construct *this* as an NPF with a Grun hidden inside.
        npf_kwargs = copy.deepcopy(kwargs)
        npf_kwargs["a"] = 10
        npf_kwargs["b"] = b
        super().__init__(**npf_kwargs)  # Calls NPF __init__()

        grun_kwargs = copy.deepcopy(kwargs)
        grun_kwargs["a"] = a
        grun_kwargs["b"] = 10

        self.grun = Grun(**grun_kwargs)

    def csfd(self, d):
        if isinstance(d, Number):
            # Convert to numpy array, if needed.
            diam = np.array([float(d), ])
        else:
            diam = d
        c = np.empty_like(diam)
        c[diam >= 10] = super().csfd(diam[diam >= 10])
        c[diam < 10] = self.grun.csfd(diam[diam < 10])
        if isinstance(d, Number):
            return c[0]
        else:
            return c

    # def isfd(self, d):
    #     if isinstance(d, Number):
    #         # Convert to numpy array, if needed.
    #         d = np.array([d, ])
    #     i = np.empty_like(d)
    #     i[d >= 10] = super().isfd(d[d >= 10])
    #     i[d < 10] = self.grun.isfd(d[d < 10])
    #     return i

    def _cdf(self, d):
        if isinstance(d, Number):
            # Convert to numpy array, if needed.
            d = np.array([d, ])
        c = np.empty_like(d)
        c[d >= 10] = super()._cdf(d[d >= 10])
        c[d < 10] = self.grun._cdf(d[d < 10])
        return c


# If new equilibrium functions are added, add them to this list to expose them
# to users.  This must be defined here *after* the classes are defined.
equilibrium_functions = (Trask, VIPER_Env_Spec)
