#!/usr/bin/env python3

import numpy as np
from synthterrain.rock import rocks
from synthterrain.rock import utilities

class InterCraterRocks(rocks.Rocks):
    # Inter-Crater rock distribution generator
    # The inter-crater rock distribution specifications 
    # are set via the tunable parameters. The generate()
    # def then produces an XML output file that
    # contains the inter-crater rock distribution.

    #------------------------------------------
    # Constructor
    # 
    # @param terrain: the terrain specification
    #            class
    #
    def __init__(self, terrain):
        super().__init__(terrain)
        self._class_name = "Inter-Crater"


    #------------------------------------------
    # Generates an inter-crater rock distribution 
    # XML file. self def should be called 
    # after all tunable parameters have been set.
    #
    # @param self: 
    #
    def generate(self):
        rocks.Rocks.generate(self, 0)

    
    #------------------------------------------
    # Creates a probability distribution of 
    # locations
    # 
    # @param self: 
    #
    def _sampleRockLocations(self):
        
        if self._terrain.dem_size[0] < 10 or self._terrain.dem_size[1] < 10:
            self._location_probability_map = np.ones(self._terrain.dem_size)
        else:
            self._location_probability_map = self._terrain.random_generator.random(self._terrain.dem_size)

            # Perturb the density map locally with some 
            # perlin-like noise so rocks clump together more
            self._location_probability_map = utilities.addGradientNoise(
                self._location_probability_map, [0, 1])

            # Don't place rocks anywhere the probability is less than 0.5
            self._location_probability_map = np.where(
                self._location_probability_map < 0.5,
                self._location_probability_map, 0)
    
    #------------------------------------------
    # Compute the number of rocks and the cumulative distribution
    # 
    # @param self:
    # @param rock_calculator: RockSizeDistribution instance
    #
    def _compute_num_rocks(self, rock_calculator):

        # TODO: SHOULD WE SUBTRACT THE EJECTA CRATER AREA?
        intercrater_area_sq_m = self._terrain.area_sq_m * self.ROCK_AREA_SCALAR

        rocks_per_m2 = rock_calculator.calculateDensity(self._diameter_range_m[0])
        num_rocks = round(rocks_per_m2 * intercrater_area_sq_m)

        return num_rocks