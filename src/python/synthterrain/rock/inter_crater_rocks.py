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
    # Tunable parameters

    # TODO: Load these from input parameters!

    # Minimum diameter of the range of inter-crater 
    # rock diameters that  will be generated (meters) 
    MIN_DIAMETER_M = 0.1

    # Step size of diameters of the range of 
    # inter-crater rock diameters that will be 
    # generated (meters)
    DELTA_DIAMETER_M = 0.001

    # Maximum diameter of the range of inter-crater 
    # rock diameters that  will be generated (meters)
    MAX_DIAMETER_M = 2

    # Rock density profile per the VIPER environmental
    # spec. The currently implemented density profiles 
    # are 'haworth', 'intercrater', and 'intercrater2'.
    ROCK_DENSITY_PROFILE = 'intercrater2'

    # Output XML filename
    OUTPUT_FILE = []

    #------------------------------------------
    # Plotting Flags

    PLOT_DENSITY_DISTRIBUTION = True # TODO FIX PLOT

    PLOT_DIAMETER_DISTRIBUTION = True # TODO FIX PLOT

    PLOT_LOCATION_PROBABILITY_MAP = True

    PLOT_LOCATIONS = False

    

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

        rocks.Rocks.generate(self, self.MIN_DIAMETER_M, self.DELTA_DIAMETER_M, self.MAX_DIAMETER_M)

        print('\n\n***** Inter-Crater Rocks *****')
        print('\nRock Density Profile: ' + str(self.ROCK_DENSITY_PROFILE))
        print('\nMin    rock diameter: ' + str(self.MIN_DIAMETER_M) + ' m')
        print('\nDelta  rock diameter: ' + str(self.DELTA_DIAMETER_M) + ' m')
        print('\nMax    rock diameter: ' + str(self.MAX_DIAMETER_M) + ' m')
        
        print('_sampleRockLocations')
        self._sampleRockLocations()
        print('_sampleRockDiameters')
        self._sampleRockDiameters()
        print('_placeRocks')
        self._placeRocks()

        if self.OUTPUT_FILE:
            print('writeXml')
            self.writeXml(self.OUTPUT_FILE);

    
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
    # Creates a probability distribution of 
    # rock diameters then samples that distribution
    # to select diameters for all rocks
    # 
    # @param self: 
    #
    def _sampleRockDiameters(self):
        
        # TODO: SHOULD WE SUBTRACT THE EJECTA CRATER AREA?
        intercrater_area_sq_m = self._terrain.area_sq_m * self.ROCK_AREA_SCALAR

        rev_cum_dist = rocks.calculateDensity(self._diameter_range_m, self.ROCK_DENSITY_PROFILE)

        print('self._terrain.area_sq_m = ' + str(self._terrain.area_sq_m))
        print('rev_cum_dist[0] = ' + str(rev_cum_dist[0]))
        print('intercrater_area_sq_m = ' + str(intercrater_area_sq_m))
        num_rocks = round(rev_cum_dist[0] * intercrater_area_sq_m)

        print('\nNumber of Rocks: ' + str(num_rocks))

        if self.PLOT_DENSITY_DISTRIBUTION:
            self.plotDensityDistribution(11, rev_cum_dist, self.ROCK_DENSITY_PROFILE);

        # Generate probability distribution
        prob_dist = utilities.revCDF_2_PDF(rev_cum_dist);

        # TODO: num_rocks too high or something else?

        # Sample the rocks sizes randomly
        self._diameters_m = self._terrain.random_generator.choice(self._diameter_range_m,
                                            num_rocks,
                                            True, # Choose with replacement
                                            prob_dist)
        
        # TODO: Move bin right edges to bin centers?
        # Compare sample to ideal distribution
        prior_sample_hist = np.histogram(self._diameters_m, self._diameter_range_m)

        # If the random terrain has too many or too few 
        # rocks of a certain size, we replace those rocks 
        # with the expected number of rocks that size.
        ideal_sample_hist = num_rocks * prob_dist[:-1]

        rock_dist_error = abs(ideal_sample_hist - prior_sample_hist[0])
        for i in range(0, len(rock_dist_error)):
            if rock_dist_error[i] > np.round(self.SIGMA_FRACTION * ideal_sample_hist[i]):
                ideal_count = int(ideal_sample_hist[i])
                current_size = self._diameter_range_m[i]

                # get rid of any ideal size craters
                self._diameters_m = self._diameters_m[self._diameters_m != current_size]

                # add correct number of ideal size craters
                new_data = current_size * np.ones((ideal_count,))
                self._diameters_m = np.concatenate((self._diameters_m, new_data))

        # TODO: Move bin right edges to bin centers?
        final_sample_hist = np.histogram(self._diameters_m, self._diameter_range_m)

        if self.PLOT_DIAMETER_DISTRIBUTION:
            self.plotDiameterDistributions(
                12,
                ideal_sample_hist,
                prior_sample_hist[0],
                final_sample_hist[0])

    #------------------------------------------
    # Places rocks according to the location
    # probability distribution
    # 
    # @param self:
    #
    def _placeRocks(self):
        if self.PLOT_LOCATION_PROBABILITY_MAP:
            self.plotLocationProbabilityMap(13)

        rocks.Rocks.placeRocks(self)

        if self.PLOT_LOCATIONS:
            self.plotLocations(14)
