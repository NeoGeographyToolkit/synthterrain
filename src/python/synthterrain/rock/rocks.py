#!/usr/bin/env python3

import matplotlib as plt
import numpy as np
from synthterrain import Utilities

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
        self._terrain = terrain;
        self._diameter_range_m = None
        self._diameters_m = None
        self._location_probability_map = None
        self.positions_xy = None
    
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
        
        self._diameter_range_m = range(min_rock_diameter_m, max_rock_diameter_m, step_rock_diameter_m)

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
            fig, ax = plt.figure(figureNumber)
            ax.clear()
            ax.loglog(self._diameter_range_m, rev_cum_dist, 'r+')
            ax.xlabel('Rock Diameter (m)')
            ax.ylabel('Cumulative Rock Number Density (#/m^2)')
            print(str(type(self))) # TODO FIX
            #if isinstance(self, InterCraterRocks):
            #    ax.title('Inter-Crater Rock Density Distribution\nFit: ' + profile.upper())
            #else:
            #    ax.title('Intra-Crater Rock Density Distribution\nFit: ' + profile.upper())

            # grid on; TODO
            plt.show()


    
    #------------------------------------------
    # @param self: 
    # @param figureNumber:
    # @param ideal_sample_hist:
    # @param prior_sample_hist:
    # @param final_sample_hist:
    #
    def plotDiameterDistributions(self, figureNumber, ideal_sample_hist, prior_sample_hist, final_sample_hist):
        fig, ax = plt.figure(figureNumber)
        ax.clear()
        ax.loglog(self._diameter_range_m, ideal_sample_hist, 'r+')
        #hold on;
        ax.loglog(self._diameter_range_m, prior_sample_hist, 'g.')
        ax.loglog(self._diameter_range_m, final_sample_hist, 'bo')
        #hold off;
        #grid on; # TODO
        ax.xlabel('Rock Diameter (m)')
        ax.ylabel('Rock Count')
        print(str(type(self))) # TODO FIX
        #if isa(self,'InterCraterRocks'):
        #    ax.title('Inter-Crater Rock Diameter Distribution')
        #else:
        #    ax.title('Intra-Crater Rock Diameter Distribution')

        ax.legend(['Ideal', 'Prior Sampled', 'Final Sampled'])
        plt.show()

    
    #------------------------------------------
    # @param self:
    # @param figureNumber:
    #
    def plotLocationProbabilityMap(self, figureNumber):
        fig, ax = plt.figure(figureNumber)
        ax.clear()
        ax.mesh(self._location_probability_map)
        ax.xlabel('Terrain X (m)')
        ax.ylabel('Terrain Y (m)')
        ax.zlabel('Terrain Z (m)')
        print(str(type(self))) # TODO FIX
        #if isa(self,'InterCraterRocks'):
        #    title('Inter-Crater Rock Location Probability Map')
        #else:
        #    title('Intra-Crater Rock Location Probability Map')

        ax.xlim([0,self.terrain.dem_size[0]])
        ax.ylim([0,self.terrain.dem_size[1]])
        ax.view([0,90])
        #colorbar TODO


    #------------------------------------------
    # @param self:
    # @param figureNumber:
    #
    def plotLocations(self, figureNumber):
        s = self._diameters_m.size()
        num_rocks = s[0]*s[1]
        xy = Utilities.downSample(self.positions_xy, 20000)
        color = 'b'

        fig, ax = plt.figure(figureNumber)
        ax.clear()
        ax.plot(xy[:,0], xy[:,1], 'o', 'MarkerSize', 1, 'Color', color, 'MarkerFaceColor',color)
        ax.xlabel('Terrain X (m)')
        ax.ylabel('Terrain Y (m)')
        print(str(type(self))) # TODO FIX
        #if isa(self,'InterCraterRocks'):
        #    title(sprintf('Inter-Crater Rock Locations\nRock Count = #d', num_rocks))
        #else:
        #    title(sprintf('Intra-Crater Rock Locations\nRock Count = #d', num_rocks))

        #grid on # TODO
        ax.xlim([0,self.terrain.dem_size[0]])
        ax.ylim([0,self.terrain.dem_size[1]])


    #------------------------------------------
    # @param self:
    # @param z:
    # @param color:
    # @return h:
    #
    def plot3(self, z, color):
        [xy,idx] = Utilities.downSample(self.positions_xy, 20000)
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
        s = f'    <RockData diameter="{np.as_array(self._diameters_m)}" x="{self.positions_xy[:,0] + self.terrain.origin[0]}" y="{self.positions_xy[:,1] + self.terrain.origin[1]}"/>\n'
        fid.write(s),
        fid.write('</RockList>\n')

   
    #------------------------------------------
    # Places rocks according to the location
    # probability distribution
    # 
    # @param self:
    #
    def _placeRocks(self):
        s = self._diameters_m.size()
        num_rocks = s[0] * s[1]

        # Sample rough probability map first 
        # the probability map is voxelized, so we'll get 
        # whole number positions from sampling it
        s = self._location_probability_map.size()
        rock_positions_idx = np.random.choice(
            range(0,s[0]*s[1]),
            num_rocks,
            False,
            np.asarray(self._location_probability_map) # Weights
        )

        [rock_pos_y, rock_pos_x] = np.ravel_multi_index(
            self.terrain.dem_size([1, 0]),
            rock_positions_idx)

        # Sample uniformly in the grid within each 
        # fractional voxel, decide where the rock goes 
        # (we dont want rocks placed only on whole 
        # number coordinates)
        delta_pos = np.rand(2, num_rocks)

        rock_pos_x = np.mod(rock_pos_x + delta_pos[0,:], self.terrain.dem_size[0])
        rock_pos_y = np.mod(rock_pos_y + delta_pos[1,:], self.terrain.dem_size[1])

        self.positions_xy[:,0] = rock_pos_x
        self.positions_xy[:,1] = rock_pos_y

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

    num_rocks_per_square_m = A * diameter_m ^ B
    return num_rocks_per_square_m

