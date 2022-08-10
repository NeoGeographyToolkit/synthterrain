#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from synthterrain.rock import utilities

class Rocks:
    # Base class for rock distribution generators
   
    #------------------------------------------
    # Tunable parameters

    # Limit on how far randomly drawn values 
    # can differ from their measured means
    SIGMA_FRACTION = 0.1

    # Terrain area scale factor for altering
    # rock distributions without altering the
    # terrain size itself
    ROCK_AREA_SCALAR = 1.0


    # PROTECTED

    #------------------------------------------
    # Constructor
    #
    # @param terrain: the terrain specification
    #            class
    #
    def __init__(self, terrain):
        self._terrain = terrain
        self._diameter_range_m = None
        self._diameters_m = None
        self._location_probability_map = None
        self.positions_xy = None
        self._class_name = "BASE" # Should be set by derived class
    
    #------------------------------------------
    # Subclasses should call self def from 
    # their own generate defs
    #
    # @param self: 
    # @param min_diameter_m:
    # @param step_diameter_m:
    # @param max_diameter_m:
    #
    def generate(self, min_rock_diameter_m, step_rock_diameter_m, max_rock_diameter_m):
        
        # Generate range [min_rock_diameter_m, max_rock_diameter_m] inclusive
        self._diameter_range_m = np.arange(min_rock_diameter_m,
            max_rock_diameter_m+step_rock_diameter_m, step_rock_diameter_m)

        self._diameters_m = []
        self.positions_xy = []
        self._location_probability_map = []

    
    #------------------------------------------
    # @param self: 
    # @param figureNumber:
    # @param rev_cum_dist:
    # @param profile:
    #
    def plotDensityDistribution(self, figureNumber, rev_cum_dist, profile):
        if self.PLOT_DENSITY_DISTRIBUTION:
            fig = plt.figure(figureNumber)
            ax = fig.add_subplot(111)
            ax.clear()
            ax.loglog(self._diameter_range_m, rev_cum_dist, 'r+')
            ax.set_xlabel('Rock Diameter (m)')
            ax.set_ylabel('Cumulative Rock Number Density (#/m^2)')
            ax.set_title(self._class_name + ' Rock Density Distribution\nFit: ' + profile.upper())


    
    #------------------------------------------
    # @param self: 
    # @param figureNumber:
    # @param ideal_sample_hist:
    # @param prior_sample_hist:
    # @param final_sample_hist:
    #
    def plotDiameterDistributions(self, figureNumber, ideal_sample_hist, prior_sample_hist, final_sample_hist):
        fig = plt.figure(figureNumber)
        ax = fig.add_subplot(111)
        ax.clear()
        ax.loglog(self._diameter_range_m[:-1], ideal_sample_hist, 'r+')
        ax.loglog(self._diameter_range_m[:-1], prior_sample_hist, 'g.')
        ax.loglog(self._diameter_range_m[:-1], final_sample_hist, 'bo')
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

        ax.set_xlim([0,self._terrain.dem_size[0]])
        ax.set_ylim([0,self._terrain.dem_size[1]])
        # ax.view([0,90]) TODO

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')


    #------------------------------------------
    # @param self:
    # @param figureNumber:
    #
    def plotLocations(self, figureNumber):
        num_rocks = len(self._diameters_m)
        print(self.positions_xy)
        xy = utilities.downSample(self.positions_xy, 20000)[0]
        color = 'b'

        fig = plt.figure(figureNumber)
        ax = fig.add_subplot(111)
        ax.clear()
        ax.plot(xy[0,:], xy[1,:], 'o', markersize=1, color=color, markerfacecolor=color)
        ax.set_xlabel('Terrain X (m)')
        ax.set_ylabel('Terrain Y (m)')
        ax.set_title(self._class_name + ' Rock Locations\nRock Count = ' + str(num_rocks))

        ax.set_xlim([0,self._terrain.dem_size[0]])
        ax.set_ylim([0,self._terrain.dem_size[1]])


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
        s = f'    <RockData diameter="{np.as_array(self._diameters_m)}" x="{self.positions_xy[:,0] + self._terrain.origin[0]}" y="{self.positions_xy[:,1] + self_terrain.origin[1]}"/>\n'
        fid.write(s),
        fid.write('</RockList>\n')

   
    #------------------------------------------
    # Places rocks according to the location
    # probability distribution
    # 
    # @param self:
    #
    def placeRocks(self):
        num_rocks = len(self._diameters_m)

        # Sample rough probability map first 
        # the probability map is voxelized, so we'll get 
        # whole number positions from sampling it
        prob_map_sum = np.sum(self._location_probability_map)
        print(self._location_probability_map)
        print(self._location_probability_map.shape)
        print(prob_map_sum)
        EPSILON = 0.0001
        if prob_map_sum < EPSILON:
            raise Exception('The sum of the location probability map is zero!')
        flat_prob_map = self._location_probability_map.flatten() / prob_map_sum
        rock_positions_idx = self._terrain.random_generator.choice(
            range(0,len(flat_prob_map)),
            num_rocks,
            True, # Choose with replacement
            flat_prob_map # Weights
        )

        [rock_pos_y, rock_pos_x] = np.unravel_index(
            rock_positions_idx, self._terrain.dem_size)

        # Sample uniformly in the grid within each 
        # fractional voxel, decide where the rock goes 
        # (we dont want rocks placed only on whole 
        # number coordinates)
        delta_pos = self._terrain.random_generator.random([2, num_rocks])

        rock_pos_x = np.mod(rock_pos_x + delta_pos[0,:], self._terrain.dem_size[0])
        rock_pos_y = np.mod(rock_pos_y + delta_pos[1,:], self._terrain.dem_size[1])

        self.positions_xy = np.stack((rock_pos_x, rock_pos_y))

# End class Rocks


#------------------------------------------
# Rock density def
# from VIPER-MSE-SPEC-001 (2/13/2020)
# 
# @param diameter_m: rock diameter(s) in meters
# @param profile: 'intercrater', 
#                 'intercrater2', or 
#                 'haworth'
# @return num_rocks_per_square_m:
#         the number of rocks with diameters
#         greater than or equal to the input
#         argument per square METER
#
def calculateDensity(diameter_m, profile):
    l_profile = profile.lower()
    if l_profile == 'intercrater':
        # Low blockiness cases
        A =  0.00010
        B = -1.75457
        # Worst-case
    elif l_profile == 'intercrater2':
        A =  0.00030
        B = -2.482
        # High blockiness case
    elif l_profile == 'haworth':
        A =  0.0020
        B = -2.6607
    else:
        raise Exception('Invalid rock density profile specified')

    num_rocks_per_square_m = A * np.power(diameter_m, B)
    return num_rocks_per_square_m

