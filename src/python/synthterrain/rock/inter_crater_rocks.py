#!/usr/bin/env python3

import numpy as np
from synthterrain.rock import rocks
from synthterrain.rock import utilities

class InterCraterRockGenerator(rocks.RockGenerator):
    """Inter-Crater rock distribution generator
       The inter-crater rock distribution specifications are set via the
       tunable parameters. The generate() def then produces an XML
       output file that contains the inter-crater rock distribution.
    """

    def __init__(self, raster, params=rocks.RockParams(), rand_seed=None):
        super().__init__(raster)
        self._class_name = "Inter-Crater"


    def generate(self):
        """Performs internal computations"""
        rocks.RockGenerator.generate(self)


    def _generate_location_probability_map(self):
        """Creates a probability distribution of locations"""
        
        if self._raster.dem_size_pixels[0] < 10 or self._raster.dem_size_pixels[1] < 10:
            location_probability_map = np.ones(self._raster.dem_size_pixels)
        else:
            location_probability_map = self._random_generator.random(self._raster.dem_size_pixels)

            # Perturb the density map locally with some 
            # perlin-like noise so rocks clump together more
            location_probability_map = utilities.addGradientNoise(
                location_probability_map, [0, 1])

            # Don't place rocks anywhere the probability is less than 0.5
            location_probability_map = np.where(location_probability_map < 0.5,
                                                location_probability_map, 0)
        return location_probability_map
    

    def _compute_num_rocks(self, rock_calculator):
        """Compute the number of rocks and the cumulative distribution
           @param rock_calculator: RockSizeDistribution instance
        """

        # TODO: SHOULD WE SUBTRACT THE EJECTA CRATER AREA?
        intercrater_area_sq_m = self._raster.area_sq_m * self.params.rock_area_scaler

        rocks_per_m2 = rock_calculator.calculateDensity(self._diameter_range_m[0])
        num_rocks = round(rocks_per_m2 * intercrater_area_sq_m)

        return num_rocks