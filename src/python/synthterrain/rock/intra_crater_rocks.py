#!/usr/bin/env python3

import math
import numpy as np
from synthterrain import Rocks
from synthterrain import Utilities

class IntraCraterRocks(Rocks):
    # Intra-Crater rock distribution generator
    # The intra-crater rock distribution specifications 
    # are set via the tunable parameters. The generate()
    # def then produces an XML output file that
    # contains the intra-crater rock distribution.

    #------------------------------------------
    # Tunable parameters
    
    # Minimum diameter of the range of intra-crater 
    # rock diameters that  will be generated (meters)
    MIN_DIAMETER_M = 0.1

    # Step size of diameters of the range of 
    # intra-crater rock diameters that will be 
    # generated (meters)
    DELTA_DIAMETER_M = 0.001

    # Maximum diameter of the range of intra-crater 
    # rock diameters that  will be generated (meters)
    MAX_DIAMETER_M = 2

    EJECTA_EXTENT = 1

    # Determines how quickly the rock density
    # decreases beyond the rim of a crater
    EJECTA_SHARPNESS = 1

    ROCK_AGE_DECAY = 3

    # Rock density profile per the VIPER environmental
    # spec. The currently implemented density profiles 
    # are 'haworth', 'intercrater', and 'intercrater2'.
    ROCK_DENSITY_PROFILE = 'intercrater2'

    # Output XML filename
    OUTPUT_FILE = []

    #------------------------------------------
    # Plotting Flags

    PLOT_DENSITY_DISTRIBUTION = True

    PLOT_DIAMETER_DISTRIBUTION = True

    PLOT_LOCATION_PROBABILITY_MAP = True

    PLOT_LOCATIONS = True

        
    #------------------------------------------
    # Constructor
    # 
    # @param terrain: the terrain specification
    #            class
    #
    def __init__(self, terrain):
        super(Rocks, self).__init__(terrain)
        _craters = None

    
    #------------------------------------------
    # Generates an intra-crater rock distribution 
    # XML file. self def should be called 
    # after all tunable parameters have been set.
    #
    # @param self: 
    # @param craters: the crater distribution
    #            generator class containing the 
    #            the crater distribution 
    #            specifications
    #
    def generate(self, craters):

        Rocks.generate(
            self,
            self.MIN_DIAMETER_M,
            self.DELTA_DIAMETER_M,
            self.MAX_DIAMETER_M)

        self.craters = craters
        
        print('\n\n***** Intra-Crater Rocks *****')
        print('\nRock Density Profile: ' + str(self.ROCK_DENSITY_PROFILE))
        print('\nMin    rock diameter: ' + str(self.MIN_DIAMETER_M) + ' m')
        print('\nDelta  rock diameter: ' + str(self.DELTA_DIAMETER_M) + ' m')
        print('\nMax    rock diameter: ' + str(self.MAX_DIAMETER_M) + ' m')
        
        self.sampleRockLocations = None
        self.sampleRockDiameters = None
        self.placeRocks = None

        if self.OUTPUT_FILE:
            self.writeXml(self.OUTPUT_FILE);

    
    # PROTECTED
        
    #------------------------------------------
    # Creates a probability distribution of 
    # locations
    # 
    # @param self: 
    #
    def _sampleRockLocations(self):
        
        # TODO
        self.location_probability_map = np.zeros(self.terrain.dem_size([1, 0]), 'single')

        s = self.craters.ejecta_crater_indices.size()
        for i in range(0, s[0]*s[1]):

            # self is the euclidean distance from the center of the crater
            m_pos = self.craters.positions_xy[self.craters.ejecta_crater_indices[i], :] # 1x2 row vector
            d = math.sqrt(
                (self.terrain.xs - m_pos(1))^2 +
                (self.terrain.ys - m_pos(2))^2) # sizeof dem

            # Convert diameter to radius for easier computation of distance (meters)
            crater_radius_m = self.craters.diameters_m[self.craters.ejecta_crater_indices[i]] / 2

            # Generate an exponentially decaying ejecta field around a crater
            outer_probability_map = self.outerProbability[d, crater_radius_m] # sizeof dem

            # Further than 1 crater radius from the rim, the ejecta field goes to 0
            outer_probability_map = np.where(
              d > (self.EJECTA_EXTENT +1) * crater_radius_m,
              0,
              outer_probability_map) # sizeof dem

            # The inside of the crater has very low uniform ejecta
            inner_probability_map = self.innerProbability(d, crater_radius_m) # sizeof dem, logical array
            inner_strength = 0.05 * outer_probability_map.max()
            outer_probability_map = np.where(
                inner_probability_map == 1,
                inner_strength,
                outer_probability_map)
            # Do we need a min(densities inner) here to check against falloff > 1
            # which increases the center rate?

            # The rock density is inverse of the crater age,
            # which simulates buried rocks from old craters
            age_diff = 1 - self.craters.ages(self.craters.ejecta_crater_indices(i));

            # Add densities to total map
            # Rocks at interior of crater replace, ejecta field adds
            self.location_probability_map = (self.location_probability_map + 
                age_diff^self.ROCK_AGE_DECAY * outer_probability_map)

    
    #------------------------------------------
    # Creates a probability distribution of 
    # rock diameters then samples that distribution
    # to select diameters for all rocks
    # 
    # @param self: 
    #
    def _sampleRockDiameters(self):
        
        intercrater_area_sq_m = self.terrain.area_sq_m * self.ROCK_AREA_SCALAR

        eps = 2.2204e-16
        direct_rock_area_sq_m = np.sum(self.location_probability_map > eps)

        # Calculate the numer of rocks generated by craters given the ejecta blanket
        # area. Use some bounds to limit the amount of ejecta generated to within
        # expectation
        if direct_rock_area_sq_m < 0.02 * intercrater_area_sq_m:  # at least 2# of the area is high blocky
            intracrater_area_sq_m = 0.02 * intercrater_area_sq_m
        elif direct_rock_area_sq_m > 0.2 * intercrater_area_sq_m:  # at most 20# of the area is high blocky
            intracrater_area_sq_m = 0.2 * intercrater_area_sq_m
        else:
            intracrater_area_sq_m = direct_rock_area_sq_m

        rev_cum_dist = Rocks.calculateDensity(
            self.diameter_range_m,
            self.ROCK_DENSITY_PROFILE)

        num_rocks = np.round(rev_cum_dist(1) * intracrater_area_sq_m)

        print('\nNumber of Rocks: ' + str(num_rocks))

        if self.PLOT_DENSITY_DISTRIBUTION:
            self.plotDensityDistribution(21, rev_cum_dist, self.ROCK_DENSITY_PROFILE)

        # Generate probability distribution
        prob_dist = Utilities.revCDF_2_PDF(rev_cum_dist)

        # Sample the rocks sizes randomly
        self.diameters_m = np.random.choice(
            self.diameter_range_m,
            num_rocks,
            False,
            prob_dist)
        self.diameters_m = np.asarray(self.diameters_m)

        # TODO CHECK EDGES
        # Compare sample to ideal distribution
        prior_sample_hist = np.hist(
            self.diameters_m,
            self.diameter_range_m)

        # If the random terrain has too many or too few 
        # rocks of a certain size, we replace those rocks 
        # with the expected number of rocks that size.
        ideal_sample_hist = num_rocks * prob_dist

        rock_dist_error = abs(ideal_sample_hist - prior_sample_hist)
        for i in range(0,len(rock_dist_error)):
            if rock_dist_error[i] > np.round(self.SIGMA_FRACTION * ideal_sample_hist[i]):
                ideal_count = np.round(ideal_sample_hist(i))
                current_size = self.diameter_range_m(i)

                # get rid of any ideal size craters
                self.diameters_m = self.diameters_m[self.diameters_m != current_size]

                # add correct number of ideal size craters
                self.diameters_m = [[self.diameters_m],
                                    [current_size * np.ones(ideal_count,1)]]

        # TODO CHECK EDGES
        final_sample_hist = np.hist(
            self.diameters_m,
            self.diameter_range_m)

        if self.PLOT_DIAMETER_DISTRIBUTION:
            self.plotDiameterDistributions(
                22,
                ideal_sample_hist,
                prior_sample_hist,
                final_sample_hist)

    #------------------------------------------
    # Places rocks according to the location
    # probability distribution
    # 
    # @param self:
    #
    def _placeRocks(self):
        if self.PLOT_LOCATION_PROBABILITY_MAP:
            self.plotLocationProbabilityMap(23)

        Rocks.placeRocks(self);

        if self.PLOT_LOCATIONS:
            self.plotLocations(24);


    # PRIVATE
    
    #------------------------------------------
    # Outer crater rock probability map def
    # Ejecta field is exponential decay
    # 
    # @param self: 
    # @param d: distance array (sizeof terrain DEM)
    # @param crater_radius_m:
    # @return result:
    #
    def _outerProbability(self, d, crater_radius_m):
        return np.exp(- self.EJECTA_SHARPNESS * d / (crater_radius_m * 0.7))

    #------------------------------------------
    # Inner crater rock probability map def
    # Ejecta field is uniform
    # 
    # @param self: 
    # @param d: distance array (sizeof terrain DEM)
    # @param crater_radius_m:
    # @return result:
    #
    def _innerProbability(self, d, crater_radius_m):
        return d < crater_radius_m
