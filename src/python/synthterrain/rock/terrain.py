#!/usr/bin/env python3

import math
import numpy as np
import matplotlib as plt
from synthterrain.rock import utilities

class Terrain:
    """Terrain specification class"""

    # PUBLIC

    # TODO: Configure from parameters

    #------------------------------------------
    # Tunable parameters

    # Optional seed for random number generator
    RAND_SEED = None

    #------------------------------------------
    # Plotting Flags

    PLOT_TERRAIN = False

    # PUBLIC
        
    #------------------------------------------
    # Constructor
    #
    def __init__(self):
        self.origin = None
        self.dem_size = [0, 0]
        self.area_sq_m = None
        self.xs = None
        self.ys = None


    #------------------------------------------
    # Sets the origin of the terrain (which corresponds to the lower left point on the DEM)
    # 
    # @param self:
    # @param origin_x: x origin of the dem
    # @param origin_y: y origin of the dem
    #
    def setOrigin(self, origin_x, origin_y):
        self.origin = [origin_x, origin_y]

    #------------------------------------------
    # Set the X dimension length of the terrain
    # 
    # @param self:
    # @param xSize: the length of the terrain along 
    #            the x-dimension in meters
    #
    def setXsize(self, xSize):
        self.dem_size[0] = math.floor(xSize)

    #------------------------------------------
    # Set the Y dimension length of the terrain
    # 
    # @param self:
    # @param ySize: the length of the terrain along 
    #            the y-dimension in meters
    #
    def setYsize(self, ySize):
        self.dem_size[1] = math.floor(ySize)

    #------------------------------------------
    # Set the optional random seed value
    #
    # @param self:
    # @param randSeed:
    #
    def setRandSeed(self, randSeed):
        self.RAND_SEED = randSeed

    #------------------------------------------
    # Generates crater and rock distributions.
    # self def should be called after all
    # tunable parameters have been set.
    #
    def generate(self):
        self.dem_size = np.floor(self.dem_size).astype(int)
        self.area_sq_m = self.dem_size[0] * self.dem_size[1]
        
        print('\n\n***** Terrain *****')
        if self.RAND_SEED:
            self.random_generator = np.random.default_rng(seed = self.RAND_SEED)
            print('\nGenerating terrain with seed #d', self.RAND_SEED)
        else:
            self.random_generator = np.random.default_rng()
        print('\nTerrain is %dm x %dm' % (self.dem_size[0], self.dem_size[1]))

        [self.xs,self.ys] = np.meshgrid(
            range(0,self.dem_size[0]),
            range(0,self.dem_size[1]))

    #------------------------------------------
    #
    # @param self:
    # @param craters:
    # @param interCraterRocks
    # @param intraCraterRocks
    #
    def plot(
            self,
            craters,
            interCraterRocks,
            intraCraterRocks):
        
        dem = Utilities.addCraterHeights(
            np.ones(self.dem_size([1, 0])+1),
            [0, self.dem_size[0], 0, self.dem_size[1]],
            craters.positions_xy,
            craters.diameters_m,
            5)

#             figure(31)
#             clf
#             imshow(dem, [])
#             set(gca,'YDir','normal')
#             colormap jet

        z1 = Terrain.getHeights(dem, interCraterRocks.positions_xy, 0.01)
        if craters.MIN_EJECTA_AREA_PERCENT > 0:
            z2 = Terrain.getHeights(dem, intraCraterRocks.positions_xy, 0.01)
        
        fig, ax = plt.figure(32)
        ax.clear()
        plt.imshow(dem) # TODO mesh(dem)
        #hold on
#             h1 = interCraterRocks.plot('k')
#             h2 = intraCraterRocks.plot('m')
        h1 = interCraterRocks.plot3(z1, 'k')
        if craters.MIN_EJECTA_AREA_PERCENT > 0:
            h2 = intraCraterRocks.plot3(z2, 'm')

        #hold off
        ax.xlabel('Terrain X (m)')
        ax.ylabel('Terrain Y (m)')
        ax.zlabel('Terrain Z (m)')
        ax.title('Terrain Craters and Rocks')
        #grid on TODO
        ax.xlim([0,self.dem_size(1)])
        ax.ylim([0,self.dem_size(2)])
        ax.view([0,90])
        ax.colorbar()
        if craters.MIN_EJECTA_AREA_PERCENT > 0:
            ax.legend([h1,h2],'Inter-Crater Rocks','Intra-Crater Rocks','Location','NorthEast')
        else:
            ax.legend(h1,'Inter-Crater Rocks','Location','NorthEast')
    
    # STATIC
    
    #------------------------------------------
    # @param dem: [x_dem_length, Y_dem_length] array
    # @param positions_xy: [x,y] position array
    # @param deltaZ: z-offset to add to calculated z-values
    #
    def getHeights(dem, positions_xy, deltaZ):
        dsize = dem.size()
        x = np.round(positions_xy[:,0])
        y = np.round(positions_xy[:,1])
        x = np.where(x<1, 1, x)
        x = np.where(x>dsize[0], dsize[0], x)
        y = np.where(y<1, 1, y)
        y = np.where(y>dsize[1], dsize[1], y)
        idx = np.ravel_multi_index(dsize, y, x)
        z = dem[idx]+deltaZ
        return z