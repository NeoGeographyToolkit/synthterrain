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


class RockParams:
    """Class containing all Rock class configurable parameters"""
    def __init__(self):

        # Limit on how far randomly drawn values
        # can differ from their measured means
        self.sigma_fraction = 0.1

        # Terrain area scale factor for altering
        # rock distributions without altering the
        # terrain size itself
        self.rock_area_scaler = 1.0

        # Rock density profile per the VIPER environmental
        # spec. The currently implemented density profiles 
        # are 'haworth', 'intercrater', and 'intercrater2'.
        self.rock_density_profile = 'intercrater2'

        # Minimum diameter of the range of
        # rock diameters that  will be generated (meters)
        self.min_diameter_m = 0.1

        # Step size of diameters of the range of
        # rock diameters that will be
        # generated (meters)
        self.delta_diameter_m = 0.02

        # Maximum diameter of the range of
        # rock diameters that  will be generated (meters)
        self.max_diameter_m = 2

        # --- For Intra-Craters ---

        self.ejecta_extent = 1

        # Determines how quickly the rock density
        # decreases beyond the rim of a crater
        self.ejecta_sharpness = 1

        self.rock_age_decay = 3


class RockGenerator:
    """Class for rock distribution generators"""

    def __init__(self, raster, craters, params=RockParams(), rand_seed=None):
        """Constructor"""
        self.params = params
        self.positions_xy = []
        self.diameters_m = []

        self._raster = raster
        self._craters = craters
        self._diameter_range_m = None
        self._location_probability_map = []
        self._rock_calculator = None

        if rand_seed:
            self._random_generator = np.random.default_rng(seed = rand_seed)
        else:
            self._random_generator = np.random.default_rng()


    def generate(self):
        """Subclasses should call self def from their own generate defs"""
        
        # Generate range [self.params.min_diameter_m, self.params.max_diameter_m] inclusive
        self._diameter_range_m = np.arange(self.params.min_diameter_m,
            self.params.max_diameter_m+self.params.delta_diameter_m,
            self.params.delta_diameter_m)

        print('\n\n***** Rock Generation *****')
        print('\nRock Density Profile: ' + str(self.params.rock_density_profile))
        print('\nMin    rock diameter: ' + str(self.params.min_diameter_m) + ' m')
        print('\nDelta  rock diameter: ' + str(self.params.delta_diameter_m) + ' m')
        print('\nMax    rock diameter: ' + str(self.params.max_diameter_m) + ' m')
        
        # Generate and merge the inter/intra probabilities
        inter_crater_probability = self._generate_inter_crater_location_probability_map()
        intra_crater_probability = self._generate_intra_crater_location_probability_map()

        self._location_probability_map = inter_crater_probability + intra_crater_probability
        self._location_probability_map = utilities.rescale(self._location_probability_map, [0, 1.0])

        self._rock_calculator = RockSizeDistribution(self.params.rock_density_profile,
                                               a=self.params.min_diameter_m, b=self.params.max_diameter_m)
        # TODO: More rocks?
        num_rocks = self._compute_num_rocks(self._rock_calculator)
        self.diameters_m = self._rock_calculator.rvs(size=num_rocks, random_state=self._random_generator, scale=1)

        self.positions_xy = self._select_rock_positions()


    def _compute_num_rocks(self, rock_calculator):
        """Compute the number of rocks and the cumulative distribution
           @param rock_calculator: RockSizeDistribution instance
        """

        intercrater_area_sq_m = self._raster.area_sq_m * self.params.rock_area_scaler

        rocks_per_m2 = rock_calculator.calculateDensity(self._diameter_range_m[0])
        num_rocks = int(round(rocks_per_m2 * intercrater_area_sq_m))

        return num_rocks

    def _select_rock_positions(self):
        """Places rocks according to the location probability distribution"""

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

        positions_xy = np.stack((rock_pos_x, rock_pos_y))
        return positions_xy

    def plotDensityDistribution(self, figureNumber):

        fig = plt.figure(figureNumber)
        ax = fig.add_subplot(111)
        ax.clear()
        densities = self._rock_calculator.calculateDensity(self._diameter_range_m)
        ax.loglog(self._diameter_range_m, densities, 'r+')
        ax.set_xlabel('Rock Diameter (m)')
        ax.set_ylabel('Cumulative Rock Number Density (#/m^2)')
        ax.set_title('Rock Density Distribution\nFit: ' + self.params.rock_density_profile.upper())


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
        ax.set_title('Rock Diameter Distribution')

        ax.legend(['Ideal', 'Prior Sampled', 'Final Sampled'])

    
    def plotLocationProbabilityMap(self, figureNumber):

        fig = plt.figure(figureNumber)
        ax = fig.add_subplot(111)
        ax.clear()
        im = ax.imshow(self._location_probability_map, vmin=0.0, vmax=1.0) # TODO mesh()
        ax.set_xlabel('Terrain X (m)')
        ax.set_ylabel('Terrain Y (m)')
        #ax.set_zlabel('Terrain Z (m)') # TODO
        ax.set_title('Rock Location Probability Map')

        ax.set_xlim([0, self._raster.dem_size_m[0]])
        ax.set_ylim([0, self._raster.dem_size_m[1]])
        # ax.view([0,90]) TODO

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')


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
        ax.set_title('Rock Locations\nRock Count = ' + str(num_rocks))

        ax.set_xlim([0, self._raster.dem_size_m[0]])
        ax.set_ylim([0, self._raster.dem_size_m[1]])
    

    def writeXml(self, filename):
        """Write the output XML rock distribution file
           @param filename: the output filename
        """
        fid = open(filename, 'w')
        if not fid:
            raise Exception('\nUnable to open file ' + filename + 'n')
        
        fid.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fid.write('<RockList name="UserRocks">\n')
        s = f'    <RockData diameter="{np.asarray(self.diameters_m)}" x="{self.positions_xy[:,0] + self._raster.origin[0]}" y="{self.positions_xy[:,1] + self._raster.origin[1]}"/>\n'
        fid.write(s),
        fid.write('</RockList>\n')


    def _generate_inter_crater_location_probability_map(self):
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



    def _generate_intra_crater_location_probability_map(self):
        """Creates a probability distribution of locations"""
        
        s = (self._raster.dem_size_pixels[1], self._raster.dem_size_pixels[0])
        location_probability_map = np.zeros(s, 'single')

        num_craters = len(self._craters['x'])
        zero_sum_craters = 0
        for i in range(0, num_craters):

            # self is the euclidean distance from the center of the crater
            m_pos = np.array([self._craters['x'][i], self._craters['y'][i]])
            d = np.sqrt(
                np.power(self._raster.xs + self._raster.origin[0] - m_pos[0], 2) +
                np.power(self._raster.ys + self._raster.origin[1] - m_pos[1], 2)) # sizeof dem

            # Convert diameter to radius for easier computation of distance (meters)
            crater_radius_m = self._craters['diameter'][i] / 2

            # Generate an exponentially decaying ejecta field around a crater
            outer_probability_map = self._intra_crater_outer_probability(d, crater_radius_m) # sizeof dem
            EPSILON = 0.001
            if np.sum(outer_probability_map) < EPSILON:
                zero_sum_craters += 1
                continue

            # Further than 1 crater radius from the rim, the ejecta field goes to 0
            outer_probability_map = np.where(
              d > (self.params.ejecta_extent +1) * crater_radius_m,
              0,
              outer_probability_map) # sizeof dem

            # The inside of the crater has very low uniform ejecta
            inner_probability_map = self._intra_crater_inner_probability(d, crater_radius_m) # sizeof dem, logical array
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
            location_probability_map = (location_probability_map +
                np.power(age_diff, self.params.rock_age_decay) * outer_probability_map)
        print('zero sum crater percentage = ' + str(zero_sum_craters / num_craters))
        return location_probability_map


    def _intra_crater_outer_probability(self, d, crater_radius_m):
        """Outer crater rock probability map def
           Ejecta field is exponential decay

           @param d: distance array (sizeof terrain DEM)
           @param crater_radius_m:
        """
        return np.exp(- self.params.ejecta_sharpness * d / (crater_radius_m * 0.7))

    def _intra_crater_inner_probability(self, d, crater_radius_m):
        """Inner crater rock probability map def
           Ejecta field is uniform

           @param d: distance array (sizeof terrain DEM)
           @param crater_radius_m:
        """
        return d < crater_radius_m




# End class Rocks

#------------------------------------------

class RockSizeDistribution(rv_continuous):
    """Rock density def from VIPER-MSE-SPEC-001 (2/13/2020)
       calculateDensity is equivalent CSFD (crater distribution) but
       for rocks.  See crater/functions.py for a more detailed description of CSFD
    """

    def __init__(self, profile, **kwargs):
        """@param profile: 'intercrater', 'intercrater2', or 'haworth'"""
        self._profile = profile.lower()
        super().__init__(**kwargs)


    def calculateDensity(self, diameter_m):
        """@param diameter_m: rock diameter(s) in meters
           @return num_rocks_per_square_m: the number of rocks with diameters
                   greater than or equal to the input argument per square METER
        """

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