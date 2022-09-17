#!/usr/bin/env python3

import numpy as np
from scipy.stats import rv_continuous
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from synthterrain.rock import utilities


class Raster:
    """Simple class containing a raster description"""

    #------------------------------------------
    # Constructor
    #
    def __init__(self, origin, nrows, ncols, resolution_meters):
        self.origin = origin
        self.resolution_m = resolution_meters
        self.dem_size_pixels = (int(nrows), int(ncols))
        self.dem_size_m = (self.dem_size_pixels[0] * resolution_meters,
                           self.dem_size_pixels[1] * resolution_meters)
        self.area_sq_m = self.dem_size_m[0] * self.dem_size_m[1]
        [self.xs, self.ys] = np.meshgrid(
            np.arange(0, self.dem_size_m[0], resolution_meters),
            np.arange(0, self.dem_size_m[1], resolution_meters)
        )


class Rocks:
    # Base class for rock distribution generators
   
    #------------------------------------------
    # Tunable parameters

    RAND_SEED = None

    # Limit on how far randomly drawn values 
    # can differ from their measured means
    SIGMA_FRACTION = 0.1

    # Terrain area scale factor for altering
    # rock distributions without altering the
    # terrain size itself
    ROCK_AREA_SCALAR = 1.0

    # Rock density profile per the VIPER environmental
    # spec. The currently implemented density profiles 
    # are 'haworth', 'intercrater', and 'intercrater2'.
    ROCK_DENSITY_PROFILE = 'intercrater2'

    # TODO: Load these from input parameters!

    # Minimum diameter of the range of
    # rock diameters that  will be generated (meters) 
    MIN_DIAMETER_M = 0.1

    # Step size of diameters of the range of 
    # rock diameters that will be 
    # generated (meters)
    DELTA_DIAMETER_M = 0.02

    # Maximum diameter of the range of 
    # rock diameters that  will be generated (meters)
    MAX_DIAMETER_M = 2

    # Output XML filename
    OUTPUT_FILE = []


    # PROTECTED

    #------------------------------------------
    # Constructor
    #
    # @param terrain: the terrain specification
    #            class
    #
    def __init__(self, raster):
        self._raster = raster
        self._diameter_range_m = None
        self.diameters_m = None
        self._location_probability_map = None
        self.positions_xy = None
        self._class_name = "BASE" # Should be set by derived class
        self._rock_calculator = None

        if self.RAND_SEED:
            self._random_generator = np.random.default_rng(seed = self.RAND_SEED)
            #print('\nGenerating terrain with seed #d', self.RAND_SEED)
        else:
            self._random_generator = np.random.default_rng()

    #------------------------------------------
    # Subclasses should call self def from 
    # their own generate defs
    #
    # @param self:
    # @param plot_start_index:
    #
    def generate(self):
        
        # Generate range [self.MIN_DIAMETER_M, self.MAX_DIAMETER_M] inclusive
        self._diameter_range_m = np.arange(self.MIN_DIAMETER_M,
            self.MAX_DIAMETER_M+self.DELTA_DIAMETER_M,
            self.DELTA_DIAMETER_M)

        self.diameters_m = []
        self.positions_xy = []
        self._location_probability_map = []

        print('\n\n***** ' + self._class_name + ' Rocks *****')
        print('\nRock Density Profile: ' + str(self.ROCK_DENSITY_PROFILE))
        print('\nMin    rock diameter: ' + str(self.MIN_DIAMETER_M) + ' m')
        print('\nDelta  rock diameter: ' + str(self.DELTA_DIAMETER_M) + ' m')
        print('\nMax    rock diameter: ' + str(self.MAX_DIAMETER_M) + ' m')
        
        self._sampleRockLocations()
        self._sampleRockDiameters()
        self._placeRocks()

        if self.OUTPUT_FILE:
            self.writeXml(self.OUTPUT_FILE);

    
    #------------------------------------------
    # @param self: 
    # @param figureNumber:
    # @param rev_cum_dist:
    # @param profile:
    #
    def plotDensityDistribution(self, figureNumber):
        fig = plt.figure(figureNumber)
        ax = fig.add_subplot(111)
        ax.clear()
        densities = self._rock_calculator.calculateDensity(self._diameter_range_m)
        ax.loglog(self._diameter_range_m, densities, 'r+')
        ax.set_xlabel('Rock Diameter (m)')
        ax.set_ylabel('Cumulative Rock Number Density (#/m^2)')
        ax.set_title(self._class_name + ' Rock Density Distribution\nFit: ' + self.ROCK_DENSITY_PROFILE.upper())


    #------------------------------------------
    # @param self: 
    # @param figureNumber:
    # @param ideal_sample_hist:
    # @param prior_sample_hist:
    # @param final_sample_hist:
    #
    def plotDiameterDistributions(self, figureNumber):

        num_rocks = len(self.diameters_m)
        prob_dist = self._rock_calculator.pdf(self._diameter_range_m)
        prob_dist = prob_dist / np.sum(prob_dist) # This is scaled by self.DELTA_DIAMETER_M
        ideal_sample_hist = num_rocks * prob_dist[:-1]
        final_sample_hist = np.histogram(self.diameters_m, self._diameter_range_m)

        fig = plt.figure(figureNumber)
        ax = fig.add_subplot(111)
        ax.clear()
        ax.loglog(self._diameter_range_m[:-1], ideal_sample_hist, 'r+')
        ax.loglog(self._diameter_range_m[:-1], final_sample_hist[0], 'bo')
        ax.set_xlabel('Rock Diameter (m)')
        ax.set_ylabel('Rock Count')
        ax.set_title(self._class_name + ' Rock Diameter Distribution')

        ax.legend(['Ideal', 'Prior Sampled', 'Final Sampled'])

    
    #------------------------------------------
    # @param self:
    # @param figureNumber:
    #
    def plotLocationProbabilityMap(self, figureNumber):
        fig = plt.figure(figureNumber)
        ax = fig.add_subplot(111)
        ax.clear()
        im = ax.imshow(self._location_probability_map, vmin=0.0, vmax=1.0) # TODO mesh()
        ax.set_xlabel('Terrain X (m)')
        ax.set_ylabel('Terrain Y (m)')
        #ax.set_zlabel('Terrain Z (m)') # TODO
        ax.set_title(self._class_name + ' Rock Location Probability Map')

        ax.set_xlim([0, self._raster.dem_size_m[0]])
        ax.set_ylim([0, self._raster.dem_size_m[1]])
        # ax.view([0,90]) TODO

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')


    #------------------------------------------
    # @param self:
    # @param figureNumber:
    #
    def plotLocations(self, figureNumber):
        num_rocks = len(self.diameters_m)

        xy = utilities.downSample(self.positions_xy, 20000)[0]
        color = 'b'

        fig = plt.figure(figureNumber)
        ax = fig.add_subplot(111)
        ax.clear()
        ax.plot(xy[0,:], xy[1,:], 'o', markersize=1, color=color, markerfacecolor=color)
        ax.set_xlabel('Terrain X (m)')
        ax.set_ylabel('Terrain Y (m)')
        ax.set_title(self._class_name + ' Rock Locations\nRock Count = ' + str(num_rocks))

        ax.set_xlim([0, self._raster.dem_size_m[0]])
        ax.set_ylim([0, self._raster.dem_size_m[1]])


    #------------------------------------------
    # @param self:
    # @param z:
    # @param color:
    # @return h:
    #
    def plot3(self, z, color):
        [xy,idx] = utilities.downSample(self.positions_xy, 20000)
        h = plt.plot3(xy[:,0], xy[:,1], z[idx], 'o', 'MarkerSize', 1, 'Color', color, 'MarkerFaceColor',color)
        return h

    
    #------------------------------------------
    # Write the output XML rock distribution file
    # 
    # @param self: 
    # @param filename: the output filename
    #
    def writeXml(self, filename):
        fid = open(filename, 'wb')
        if not fid:
            raise Exception('\nUnable to open file ' + filename + 'n')
        
        fid.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fid.write('<RockList name="UserRocks">\n')
        s = f'    <RockData diameter="{np.as_array(self.diameters_m)}" x="{self.positions_xy[:,0] + self._raster.origin[0]}" y="{self.positions_xy[:,1] + self_raster.origin[1]}"/>\n'
        fid.write(s),
        fid.write('</RockList>\n')

    # Must be defined by child classes
    # @return num_rocks
    def _compute_num_rocks(self, rock_calculator):
        pass


    #------------------------------------------
    # Creates a probability distribution of 
    # rock diameters then samples that distribution
    # to select diameters for all rocks
    # 
    # @param self: 
    #
    def _sampleRockDiameters(self):

        self._rock_calculator = RockSizeDistribution(self.ROCK_DENSITY_PROFILE,
                                               a=self.MIN_DIAMETER_M, b=self.MAX_DIAMETER_M)

        num_rocks = self._compute_num_rocks(self._rock_calculator)

        # Generate probability distribution
        # Sample the rocks sizes randomly

        SEED = 13492463493612533854268 # TODO: Do we want a constant seed?
        rng_gen = np.random.default_rng(SEED)
        self.diameters_m = self._rock_calculator.rvs(size=num_rocks, random_state=rng_gen, scale=1)

    #------------------------------------------
    # Places rocks according to the location
    # probability distribution
    # 
    # @param self:
    #
    def _placeRocks(self):

        num_rocks = len(self.diameters_m)

        # Sample rough probability map first 
        # the probability map is voxelized, so we'll get 
        # whole number positions from sampling it
        prob_map_sum = np.sum(self._location_probability_map)

        EPSILON = 0.0001
        if prob_map_sum < EPSILON:
            raise Exception('The sum of the location probability map is zero!')
        flat_prob_map = self._location_probability_map.flatten() / prob_map_sum
        rock_positions_idx = self._random_generator.choice(
            range(0,len(flat_prob_map)),
            num_rocks,
            True, # Choose with replacement
            flat_prob_map # Weights
        )

        # Convert from 1D to 2D indices
        [rock_pos_y, rock_pos_x] = np.unravel_index(
            rock_positions_idx, self._raster.dem_size_pixels)
        # Convert to x,y offset from the origin
        rock_pos_x = rock_pos_x * self._raster.resolution_m
        rock_pos_y = rock_pos_y * self._raster.resolution_m

        # Sample uniformly in the grid within each 
        # fractional voxel, decide where the rock goes 
        # (we dont want rocks placed only on whole 
        # number coordinates)
        delta_pos = self._random_generator.random([2, num_rocks])

        rock_pos_x = np.mod(rock_pos_x + delta_pos[0,:], self._raster.dem_size_m[0])
        rock_pos_y = np.mod(rock_pos_y + delta_pos[1,:], self._raster.dem_size_m[1])

        self.positions_xy = np.stack((rock_pos_x, rock_pos_y))


# End class Rocks


#------------------------------------------
# Rock density def
# from VIPER-MSE-SPEC-001 (2/13/2020)
# 
# calculateDensity is equivalent CSFD (crater distribution) but
# for rocks.  See crater/functions.py for a more detailed description of CSFD
class RockSizeDistribution(rv_continuous):

    # @param profile: 'intercrater',
    #                 'intercrater2', or
    #                 'haworth'
    def __init__(self, profile, **kwargs):
        self._profile = profile.lower()
        super().__init__(**kwargs)

    # @param diameter_m: rock diameter(s) in meters
    # @return num_rocks_per_square_m:
    #         the number of rocks with diameters
    #         greater than or equal to the input
    #         argument per square METER
    def calculateDensity(self, diameter_m):

        if self._profile == 'intercrater':
            # Low blockiness cases
            A =  0.00010
            B = -1.75457
            # Worst-case
        elif self._profile == 'intercrater2':
            A =  0.00030
            B = -2.482
            # High blockiness case
        elif self._profile == 'haworth':
            A =  0.0020
            B = -2.6607
        else:
            raise Exception('Invalid rock density profile specified')

        num_rocks_per_square_m = A * np.power(diameter_m, B)
        return num_rocks_per_square_m

    def _cdf(self, diameter_m):
        return np.ones_like(diameter_m) - (self.calculateDensity(diameter_m) / self.calculateDensity(self.a))