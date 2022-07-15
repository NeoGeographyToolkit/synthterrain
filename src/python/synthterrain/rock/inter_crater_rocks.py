#!/usr/bin/env python3

import numpy as np
from synthterrain import Rocks
from synthterrain import Utilities

class InterCraterRocks(Rocks):
    # Inter-Crater rock distribution generator
    # The inter-crater rock distribution specifications 
    # are set via the tunable parameters. The generate()
    # def then produces an XML output file that
    # contains the inter-crater rock distribution.
   
    #------------------------------------------
    # Tunable parameters

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

    
    #------------------------------------------
    # Generates an inter-crater rock distribution 
    # XML file. self def should be called 
    # after all tunable parameters have been set.
    #
    # @param self: 
    #
    def generate(self):

        Rocks.generate(self, self.MIN_DIAMETER_M, self.DELTA_DIAMETER_M, self.MAX_DIAMETER_M)

        print('\n\n***** Inter-Crater Rocks *****')
        print('\nRock Density Profile: ' + str(self.ROCK_DENSITY_PROFILE))
        print('\nMin    rock diameter: ' + str(self.MIN_DIAMETER_M) + ' m')
        print('\nDelta  rock diameter: ' + str(self.DELTA_DIAMETER_M) + ' m')
        print('\nMax    rock diameter: ' + str(self.MAX_DIAMETER_M) + ' m')
        
        self.sampleRockLocations;
        self.sampleRockDiameters;
        self.placeRocks;

        if self.OUTPUT_FILE:
            self.writeXml(self.OUTPUT_FILE);

    
    #------------------------------------------
    # Creates a probability distribution of 
    # locations
    # 
    # @param self: 
    #
    def _sampleRockLocations(self):
        
        if any(self.terrain.dem_size < 10):
            self.location_probability_map = np.ones(self.terrain.dem_size)
        else:
            self.location_probability_map = np.rand(self.terrain.dem_size)

            # Perturb the density map locally with some 
            # perlin-like noise so rocks clump together more
            self.location_probability_map = Utilities.addGradientNoise(
                self.location_probability_map, [0, 1])

            # Don't place rocks anywhere the probability is less than 0.5
            self.location_probability_map = np.where(
                self.location_probability_map < 0.5,
                self.location_probability_map, 0)
    
    #------------------------------------------
    # Creates a probability distribution of 
    # rock diameters then samples that distribution
    # to select diameters for all rocks
    # 
    # @param self: 
    #
    def _sampleRockDiameters(self):
        
        # TODO: SHOULD WE SUBTRACT THE EJECTA CRATER AREA?
        intercrater_area_sq_m = self.terrain.area_sq_m * self.ROCK_AREA_SCALAR;

        rev_cum_dist = Rocks.calculateDensity(self.diameter_range_m, self.ROCK_DENSITY_PROFILE)

        num_rocks = round(rev_cum_dist(1) * intercrater_area_sq_m);

        print('\nNumber of Rocks: ' + str(num_rocks))

        if self.PLOT_DENSITY_DISTRIBUTION:
            self.plotDensityDistribution(11, rev_cum_dist, self.ROCK_DENSITY_PROFILE);

        # Generate probability distribution
        prob_dist = Utilities.revCDF_2_PDF(rev_cum_dist);

        # Sample the rocks sizes randomly
        self.diameters_m = np.random.choice(self.diameter_range_m,
                                            num_rocks,
                                            False,
                                            prob_dist)
        self.diameters_m = np.asarray()

        # TODO: Move bin right edges to bin centers?
        # Compare sample to ideal distribution
        prior_sample_hist = np.hist(self.diameters_m, self.diameter_range_m)

        # If the random terrain has too many or too few 
        # rocks of a certain size, we replace those rocks 
        # with the expected number of rocks that size.
        ideal_sample_hist = num_rocks * prob_dist

        rock_dist_error = abs(ideal_sample_hist - prior_sample_hist)
        for i in range(0,len(rock_dist_error)):
            if rock_dist_error[i] > np.round(self.SIGMA_FRACTION * ideal_sample_hist[i]):
                ideal_count = np.round(ideal_sample_hist[i])
                current_size = self.diameter_range_m[i]

                # get rid of any ideal size craters
                self.diameters_m = self.diameters_m[self.diameters_m != current_size]

                # add correct number of ideal size craters
                self.diameters_m = [[self.diameters_m], [current_size * np.ones(ideal_count,1)]]

        # TODO: Move bin right edges to bin centers?
        final_sample_hist = np.hist(self.diameters_m, self.diameter_range_m)

        if self.PLOT_DIAMETER_DISTRIBUTION:
            self.plotDiameterDistributions(
                12,
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
            self.plotLocationProbabilityMap(13)

        Rocks.placeRocks(self)

        if self.PLOT_LOCATIONS:
            self.plotLocations(14)
