#!/usr/bin/env python3

import math
import numpy as np
from synthterrain.rock import rocks
from synthterrain.rock import utilities

class IntraCraterRocks(rocks.Rocks):
    # Intra-Crater rock distribution generator
    # The intra-crater rock distribution specifications 
    # are set via the tunable parameters. The generate()
    # def then produces an XML output file that
    # contains the intra-crater rock distribution.

    #------------------------------------------
    # Tunable parameters
    
# TODO: Load these from input parameters!

    EJECTA_EXTENT = 1

    # Determines how quickly the rock density
    # decreases beyond the rim of a crater
    EJECTA_SHARPNESS = 1

    ROCK_AGE_DECAY = 3
        
    #------------------------------------------
    # Constructor
    # 
    # @param terrain: the terrain specification
    #            class
    #
    def __init__(self, terrain):
        super().__init__(terrain)
        self._craters = None
        self._class_name = "Intra-Crater"

    
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
        self._craters = craters
        rocks.Rocks.generate(self, 10)

    
    # PROTECTED
        
    #------------------------------------------
    # Creates a probability distribution of 
    # locations
    # 
    # @param self: 
    #
    def _sampleRockLocations(self):
        
        s = (self._terrain.dem_size[1], self._terrain.dem_size[0])
        self._location_probability_map = np.zeros(s, 'single')

        num_craters = len(self._craters['x'])
        zero_sum_craters = 0
        for i in range(0, num_craters):

            # self is the euclidean distance from the center of the crater
            m_pos = np.array([self._craters['x'][i], self._craters['y'][i]])
            d = np.sqrt(
                np.power(self._terrain.xs + self._terrain.origin[0] - m_pos[0], 2) +
                np.power(self._terrain.ys + self._terrain.origin[1] - m_pos[1], 2)) # sizeof dem

            # Convert diameter to radius for easier computation of distance (meters)
            crater_radius_m = self._craters['diameter'][i] / 2

            # Generate an exponentially decaying ejecta field around a crater
            outer_probability_map = self._outerProbability(d, crater_radius_m) # sizeof dem
            EPSILON = 0.001
            if np.sum(outer_probability_map) < EPSILON:
                zero_sum_craters += 1
                continue

            # Further than 1 crater radius from the rim, the ejecta field goes to 0
            outer_probability_map = np.where(
              d > (self.EJECTA_EXTENT +1) * crater_radius_m,
              0,
              outer_probability_map) # sizeof dem

            # The inside of the crater has very low uniform ejecta
            inner_probability_map = self._innerProbability(d, crater_radius_m) # sizeof dem, logical array
            inner_strength = 0.05 * outer_probability_map.max()
            outer_probability_map = np.where(
                inner_probability_map == 1,
                inner_strength,
                outer_probability_map)
            # Do we need a min(densities inner) here to check against falloff > 1
            # which increases the center rate?

            # TODO: Resolve this difference between old and new craters class
            # The rock density is inverse of the crater age,
            # which simulates buried rocks from old craters
            #age_diff = 1 - self._craters.ages(self.craters.ejecta_crater_indices(i));
            age_diff = 1 - self._craters['age'][i]
            age_diff = 1 - 0.1 # TODO!!!

            # Add densities to total map
            # Rocks at interior of crater replace, ejecta field adds
            self._location_probability_map = (self._location_probability_map + 
                np.power(age_diff, self.ROCK_AGE_DECAY) * outer_probability_map)
        print('zero sum crater percentage = ' + str(zero_sum_craters / num_craters))
    

    #------------------------------------------
    # Compute the number of rocks and the cumulative distribution
    # 
    # @param self:
    #
    def _compute_num_rocks(self):

        intercrater_area_sq_m = self._terrain.area_sq_m * self.ROCK_AREA_SCALAR

        eps = 2.2204e-16
        threshold_values = np.where(self._location_probability_map > eps, self._location_probability_map, 0)
        direct_rock_area_sq_m = np.sum(threshold_values)

        # Calculate the number of rocks generated by craters given the ejecta blanket
        # area. Use some bounds to limit the amount of ejecta generated to within
        # expectation
        if direct_rock_area_sq_m < 0.02 * intercrater_area_sq_m:  # at least 2# of the area is high blocky
            intracrater_area_sq_m = 0.02 * intercrater_area_sq_m
        elif direct_rock_area_sq_m > 0.2 * intercrater_area_sq_m:  # at most 20# of the area is high blocky
            intracrater_area_sq_m = 0.2 * intercrater_area_sq_m
        else:
            intracrater_area_sq_m = direct_rock_area_sq_m

        min_rock_size = self._diameter_range_m[0]
        max_rock_size = self._diameter_range_m[-1]
        rock_calculator = rocks.RockSizeDistribution(self.ROCK_DENSITY_PROFILE,
                                                     a=min_rock_size, b=max_rock_size)
        rocks_per_m2 = rock_calculator.calculateDensity(min_rock_size)

        num_rocks = int(np.round(rocks_per_m2 * intracrater_area_sq_m))

        return num_rocks, rock_calculator

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
