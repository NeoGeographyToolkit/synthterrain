#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This module contains abstract and concrete classes for representing
crater size-frequency distributions as probability distributions.
"""

# Copyright 2022, United States Government as represented by the
# Administrator of the National Aeronautics and Space Administration.
# All rights reserved.

from abc import ABC, abstractmethod
import copy
import logging
import math
from numbers import Number

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import interp1d
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
        return np.ones_like(d, dtype=np.dtype(float)) - (self.csfd(d) / self.csfd(self.a))

    def count(self, area, diameter=None) -> int:
        """Returns the number of craters based on the *area* provided
        in square meters.  If *diameter* is None (the default), the
        calculation will be based on the cumulative number of craters
        at the minimum support, a, of this distribution.  Otherwise, the
        returned size will be the value of this distribution's CSFD
        at *diameter* multiplied by the *area*.
        """
        if diameter is None:
            d = self.a
        else:
            d = diameter
        return int(self.csfd(d) * area)

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
            kwargs["size"] = self.count(kwargs["area"])
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
        return np.ones_like(d, dtype=np.dtype(float)) - (
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
            diam = np.array([d, ])
        else:
            diam = d
        c = np.empty_like(diam, dtype=np.dtype(float))
        c[diam <= 80] = 29174 * np.float_power(diam[diam <= 80], -1.92)
        c[diam > 80] = 156228 * np.float_power(diam[diam > 80], -2.389)
        out = c / (1000 * 1000)
        if isinstance(d, Number):
            return out.item()
        else:
            return out

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
        c = np.empty_like(d, dtype=np.dtype(float))
        c[d <= 80] = np.float_power(d[d <= 80], -1.92) / np.float_power(self.a, -1.92)
        c[d > 80] = np.float_power(d[d > 80], -2.389) / np.float_power(self.a, -2.389)
        return np.ones_like(d, dtype=np.dtype(float)) - c

    def _ppf(self, q):
        """Override parent function to make things faster for .rvs()."""
        q80 = float(self._cdf(np.array([80, ])))
        ones = np.ones_like(q, dtype=np.dtype(float))
        p = np.empty_like(q, dtype=np.dtype(float))
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
        return np.ones_like(d, dtype=np.dtype(float)) - np.float_power(
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


class Interp_Distribution(Crater_rv_continuous):
    """This class instantiates a continuous crater distribution based
       on interpolation of a set of data points.

       The input arrays assume that the diameter values are in meters
       and the cumulative size frequency distribution values are in
       counts per square meter.
    """

    def __init__(self, *args, diameters=None, csfds=None, func=None, **kwargs):
        if diameters is None and func is None:
            raise ValueError(
                "An Interp_Distribution object must be initiated with "
                "*diameters* and *csfds* array-likes of data from which an "
                "interpolated function will be constructed, or a *func* object "
                "which must be callable with diameter values and will return "
                "csfd values."
            )
        super().__init__(*args, **kwargs)

        if diameters is not None:
            func = interp1d(np.log10(diameters), np.log10(csfds))

        self.func = func

    def csfd(self, d):
        """Returns the crater cumulative size frequency distribution function
           value for *d*.
        """
        return np.float_power(10, self.func(np.log10(d)))

    def _cdf(self, d):
        """Override parent function to speed up."""
        return np.ones_like(d, dtype=np.dtype(float)) - np.float_power(
            10,
            self.func(np.log10(d)) - self.func(np.log10(self.a))
        )


class Grun(Interp_Distribution):
    """
    Grun et al. (1985, https://doi.org/10.1016/0019-1035(85)90121-6)
    describe a small particle impact flux, which can be converted into
    a production function for small craters, that matches well with the
    value of the Neukum et al. (2001) production function at d=10 m.
    """

    def __init__(self, **kwargs):
        # This method for using Grun et al. (1985) to "simulate" a crater
        # distribution is from Caleb Fassett, pers. comm.
        diameters, fluxes = self.parameters()

        if "b" in kwargs:
            if kwargs["b"] > max(diameters):
                raise ValueError(
                    "The upper bound of the support of the distribution, b, "
                    f" must be <= {max(diameters)}."
                )
        else:
            kwargs["b"] = max(diameters)

        if "a" not in kwargs:
            kwargs["a"] = min(diameters)

        kwargs["diameters"] = diameters
        kwargs["csfds"] = fluxes

        super().__init__(**kwargs)

        # These comments temporarily preserve the Coef_Distribution model
        # # The Coef_Distribution polynomial needs diameters
        # # in kilometers, so divide by 1000.  And fluxes need
        # # to be in /km^2 /Gyr, so multiply by a million.
        # p = Polynomial.fit(
        #     np.log10(diameters / 1000), np.log10(fluxes * 1e6), 11
        # )

        # super().__init__(
        #     poly=p,
        #     **kwargs
        # )

    @staticmethod
    def parameters():
        """
        This function returns a set of diameters and fluxes which are used
        by the Grun() class.  This function (and its comments) exists to
        demonstrate the math for how the values were arrived at.
        """

        # Grun et al. equation A2 describes fluxes in terms of mass.

        # We'll generate "craters" based on masses from 10^-18 to the upper
        # valid limit of 10^2 grams for these equations.
        # masses = np.logspace(-18, 2, 21)  # in grams
        masses = np.logspace(-18, 2, 201)  # in grams
        # masses = np.logspace(-18, 0, 19)  # in grams
        # print(masses)

        # Constants indicated below equation A2 from Grun et al. (1985)
        # First element here is arbitrarily zero so that indexes match
        # with printed A2 equation for easier comparison.
        c = (0, 4e+29, 1.5e+44, 1.1e-2, 2.2e+3, 15.)
        gamma = (0, 1.85, 3.7, -0.52, 0.306, -4.38)

        def a_elem(mass, i):
            return c[i] * np.float_power(mass, gamma[i])

        def a2(mass):
            # Returns flux in m^-2 s^-1
            return (
                np.float_power(
                    a_elem(mass, 1) + a_elem(mass, 2) + c[3], gamma[3]
                ) +
                np.float_power(a_elem(mass, 4) + c[5], gamma[5])
            )

        # fluxes = a2(masses) * 86400.0 * 365.25  # convert to m^-2 yr^-1
        # fluxes = a2(masses) * 86400.0 * 365.25 * 1e6  # convert to km^-2 yr^-1
        fluxes = a2(masses) * 86400.0 * 365.25 * 1e9  # convert to m^-2 Gyr^-1
        # fluxes = a2(masses) * 1e6 * 86400.0 * 365.25 * 1e9  # to /km^2 /Gyr

        # To convert mass, m, of a particle to diameter of crater, we first
        # assume a density, rho, of 2.5 g/cm^-3, and calculate a particle radius
        # by assuming spherical particles:
        #
        #   m / rho = (4 / 3) * pi * r^3
        #
        #   r = [ (3 * m) / (4 * pi * rho) ] ^(1/3)
        #
        rho = 2.5e+6  # g/m^-3
        radii = np.float_power(
            (3 * masses) / (4 * math.pi * rho),
            1 / 3
        )  # should be radii in meters.

        # Now these "impactor" radii need to be converted to crater size via
        # Housen & Holsapple (2011) scaling.
        diameters = Grun.hoho_diameter(radii, masses / 1000, rho / 1000)

        # # The above largest diameter only gets you 2.7636 m diameter craters.  And
        # # Neukum doesn't start until 10 m, so we're going to pick out some
        # # diameters from Neukum to add to these so that the polynomial in Grun()
        # # spans the space.
        # npf = NPF(10, 100)
        # n_diams = np.array([10, 15, 20, 30, 50, 100])
        # n_fluxes = npf.csfd(n_diams)
        #
        # diameters = np.append(diameters, n_diams)
        # fluxes = np.append(fluxes, n_fluxes)

        # diameters in meters, and fluxes in m^-2 Gyr^-1
        return diameters, fluxes

    @staticmethod
    def hoho_diameter(
        radii,  # numpy array in meters
        masses,  # numpy array in kg
        rho,  # impactor density in kg m^-3
        gravity=1.62,  # m s^-2
        strength=1.0e4,  # Pa
        targdensity=1500.0,  # kg/m3 (rho)
        velocity=20000.0,  # m/s
        alpha=45.0,  # impact angle degrees
        nu=(1.0 / 3.0),  # ~1/3 to 0.4
        mu=0.43,  # ~0.4 to 0.55
        K1=0.132,
        K2=0.26,
        Kr=(1.1 * 1.3)  # Kr and KrRim
    ):
        # This function is adapted from Caleb's research code, but is based
        # on Holsapple (1993,
        # https://www.annualreviews.org/doi/epdf/10.1146/annurev.ea.21.050193.002001
        # ).  He says: # Varying mu makes a big difference in scaling, 0.41 from
        # Williams et al. would predict lower fluxes / longer equilibrium times
        # and a discontinuity with Neukum

        effvelocity = velocity * math.sin(math.radians(alpha))
        densityratio = (targdensity / rho)

        # impmass=((4.0*math.pi)/3.0)*impdensity*(impradius**3.0)  #impactormass
        pi2 = (gravity * radii) / math.pow(effvelocity, 2.0)

        pi3 = strength / (targdensity * math.pow(effvelocity, 2.0))

        expone = (6.0 * nu - 2.0 - mu) / (3.0 * mu)
        exptwo = (6.0 * nu - 2.0) / (3.0 * mu)
        expthree = (2.0 + mu) / 2.0
        expfour = (-3.0 * mu) / (2.0 + mu)
        piV = K1 * np.float_power(
            (pi2 * np.float_power(densityratio, expone)) +
            np.float_power(
                K2 * pi3 * np.float_power(densityratio, exptwo),
                expthree
            ),
            expfour
        )
        V = (masses * piV) / targdensity  # m3 for crater
        rim_radius = Kr * np.float_power(V, (1 / 3))

        return 2 * rim_radius


class GNPF_old(NPF):
    """
    This describes a combination function such that it functions as a Neukum
    Production Function (NPF) for the size ranges where NPF is appropriate,
    and as a Grun function where that is appropriate.
    """

    def __init__(self, a, b, interp="extendGrun", **kwargs):
        if b <= 2.5:
            raise ValueError(
                f"The upper bound, b, is {b}, you should use Grun, not GNPF."
            )

        if a >= 10:
            raise ValueError(
                f"The lower bound, a, is {a}, you should use NPF, not GNPF."
            )

        interp_types = ("extendGrun", "linear", "interp")
        if interp in interp_types:
            self.interp = interp
        else:
            raise ValueError(
                f"The interpolation method, {interp} "
                f"is not one of {interp_types}."
            )

        # Will now construct *this* as an NPF with a Grun hidden inside.
        npf_kwargs = copy.deepcopy(kwargs)
        npf_kwargs["a"] = 10
        npf_kwargs["b"] = b
        super().__init__(**npf_kwargs)  # Calls NPF __init__()

        grun_kwargs = copy.deepcopy(kwargs)
        grun_kwargs["a"] = a
        if self.interp == "extendGrun":
            grun_kwargs["b"] = 10
            grun_d, grun_f = Grun.parameters()
            # The above largest diameter only gets you 2.5 m diameter craters.
            # And Neukum doesn't start until 10 m, so we're going to pick out
            # some # diameters from Neukum to add to these so that the
            # polynomial spans the space.
            npf = NPF(10, 100)
            n_diams = np.array([10, 15, 20, 30, 50, 100])
            n_fluxes = npf.csfd(n_diams)

            diameters = np.append(grun_d, n_diams)
            fluxes = np.append(grun_f, n_fluxes)

            p = Polynomial.fit(
                np.log10(diameters / 1000), np.log10(fluxes * 1e6), 11
            )

            self.grun = Coef_Distribution(poly=p, **grun_kwargs)
        elif self.interp == "interp":
            grun_kwargs["b"] = 10
            grun_d, grun_f = Grun.parameters()
            npf = NPF(10, 100)
            n_diam = 10
            n_flux = npf.csfd(n_diam)
            diameters = np.append(grun_d, n_diam)
            fluxes = np.append(grun_f, n_flux)
            grun_kwargs["diameters"] = diameters
            grun_kwargs["csfds"] = fluxes
            self.grun = Interp_Distribution(**grun_kwargs)
        else:
            grun_kwargs["b"] = 2.5
            self.grun = Grun(**grun_kwargs)

        self.grunstop = 2.5

    def csfd(self, d):
        if isinstance(d, Number):
            # Convert to numpy array, if needed.
            diam = np.array([float(d), ])
        else:
            diam = d
        c = np.empty_like(diam, dtype=np.dtype(float))

        c[diam >= 10] = super().csfd(diam[diam >= 10])

        if self.interp == "extendGrun" or self.interp == "interp":
            c[diam < 10] = self.grun.csfd(diam[diam < 10])
        elif self.interp == "linear":
            d_interp = np.log10((self.grunstop, 10))
            c_interp = np.log10((
                self.grun.csfd(self.grunstop), super().csfd(10)
            ))
            # cs = CubicSpline(d_interp, c_interp)
            f = interp1d(d_interp, c_interp)

            overlap = np.logical_and(diam > self.grunstop, diam < 10)
            # c[diam < 2.5] = self.grun.csfd(diam[diam < 2.5])
            # c[overlap] = np.power(10, np.interp(
            #     np.log10(diam[overlap]),
            #     [np.log10(self.grunstop), np.log10(10)],
            #     [
            #         np.log10(self.grun.csfd(self.grunstop)),
            #         np.log10(super().csfd(10))
            #     ]
            # ))
            c[overlap] = np.float_power(10, f(np.log10(diam[overlap])))
            c[diam <= self.grunstop] = self.grun.csfd(diam[diam <= self.grunstop])
        else:
            raise ValueError(
                f"The interpolation method, {self.interp}, is not recognized."
            )

        if isinstance(d, Number):
            return c.item()
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
            diam = np.array([d, ])
        else:
            diam = d
        c = np.empty_like(diam, dtype=np.dtype(float))

        c[diam >= 10] = super()._cdf(diam[diam >= 10])
        if self.interp == "extendGrun":
            c[diam < 10] = self.grun._cdf(diam[diam < 10])
        elif self.interp == "linear":
            d_interp = np.log10((self.grunstop, 10))
            c_interp = np.log10((
                self.grun._cdf(self.grunstop), super()._cdf(10)
            ))
            # cs = CubicSpline(d_interp, c_interp)
            f = interp1d(d_interp, c_interp)

            overlap = np.logical_and(diam > self.grunstop, diam < 10)
            # c[overlap] = np.power(10, np.interp(
            #     np.log10(d[overlap]),
            #     [np.log10(self.grunstop), np.log10(10)],
            #     [
            #         np.log10(self.grun._cdf(self.grunstop)),
            #         np.log10(super()._cdf(10))
            #     ]
            # ))
            c[overlap] = np.float_power(10, f(np.log10(diam[overlap])))
            c[diam <= self.grunstop] = self.grun._cdf(diam[diam <= self.grunstop])

        if isinstance(d, Number):
            return c.item()
        else:
            return c


class GNPF(NPF):
    """
    This describes a combination function such that it functions as a Neukum
    Production Function (NPF) for the size ranges where NPF is appropriate,
    and as a Grun function where that is appropriate.
    """

    def __init__(self, a, b, **kwargs):
        if b <= 2.76:
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

        # Need to get Grun data points and extend to the first NPF point:
        grun_diams, grun_fluxes = Grun.parameters()
        npf = NPF(10, b, **kwargs)
        n_diam = 10
        n_flux = npf.csfd(n_diam)
        diameters = np.append(grun_diams, n_diam)
        fluxes = np.append(grun_fluxes, n_flux)
        grun_kwargs["diameters"] = diameters
        grun_kwargs["csfds"] = fluxes
        self.grun = Interp_Distribution(**grun_kwargs)

        return

    def csfd(self, d):
        if isinstance(d, Number):
            # Convert to numpy array, if needed.
            diam = np.array([float(d), ])
        else:
            diam = d
        c = np.empty_like(diam, dtype=np.dtype(float))

        c[diam >= 10] = super().csfd(diam[diam >= 10])
        c[diam < 10] = self.grun.csfd(diam[diam < 10])

        if isinstance(d, Number):
            return c.item()
        else:
            return c

    def _cdf(self, d):
        if isinstance(d, Number):
            # Convert to numpy array, if needed.
            diam = np.array([d, ])
        else:
            diam = d
        c = np.empty_like(diam, dtype=np.dtype(float))

        c[diam >= 10] = super()._cdf(diam[diam >= 10])
        c[diam < 10] = self.grun._cdf(diam[diam < 10])

        if isinstance(d, Number):
            return c.item()
        else:
            return c


class GNPF_fit(Coef_Distribution):
    """
    This describes a combination function such that it functions as a Neukum
    Production Function (NPF) for the size ranges where NPF is appropriate,
    and as a Grun function where that is appropriate.  Rather than being
    piecewise correct (which can cause unrealistic behavior where the two
    functions meet at 10 m diameter, this fits a new, single 11-degree
    polynomial across the span of both functions.  This resulting function
    is similar but not equal to Grun or Neukum in the ranges where they are
    appropriate, but does join together smoothly at 10 m diameters.
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

        # Need to get Grun data points:
        grun_diams, grun_fluxes = Grun.parameters()

        # Now need to get Neukum and sample points:
        npf = NPF(10, b, **kwargs)
        npf_diams = np.geomspace(10, 300000, 1000)
        npf_fluxes = np.float_power(10, npf.poly(np.log10(npf_diams / 1000)))

        # The Coef_Distribution polynomial needs diameters
        # in kilometers, so divide by 1000.  And fluxes need
        # to be in /km^2 /Gyr, so multiply by a million.
        diameters = np.append(grun_diams, npf_diams) / 1000
        fluxes = np.append(grun_fluxes * 1e6, npf_fluxes)

        # import sys
        # np.set_printoptions(threshold=sys.maxsize)
        # print(np.log10(diameters))
        # print(grun_fluxes)
        # print(npf_fluxes)
        # print(fluxes)
        # print(np.log10(fluxes))
        # np.set_printoptions(threshold=False)

        p = Polynomial.fit(
            np.log10(diameters), np.log10(fluxes), 11
        )

        kwargs["a"] = a
        kwargs["b"] = b
        super().__init__(
            poly=p,
            **kwargs
        )


# If new equilibrium functions are added, add them to this list to expose them
# to users.  This must be defined here *after* the classes are defined.
equilibrium_functions = (Trask, VIPER_Env_Spec)
