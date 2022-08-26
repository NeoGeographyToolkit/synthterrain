#!/usr/bin/env python3

import scipy
from scipy.stats import norm
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from synthterrain.rock import utilities

# TODO: This class is deprecated!
class Craters:
    # Crater distribution generator
    # The crater distribution specifications are
    # set via the tunable parameters. The generate()
    # def then produces an XML output file that
    # contains the crater distribution.

    # PUBLIC

    #------------------------------------------
    # Tunable parameters

    # High-blockiness threshold
    # Craters larger than self diameter will
    # produce rocks given by a high blockiness
    # distribution
    BLOCK_GEN_CSIZE = 1000

    # Multiplier from crater radius to ejecta radius.
    # In VIPER-MSE-SPEC-001 (2/13/2020), the
    # ejecta radius is twice the crater radius
    EJECTA_RADIUS_SCALAR = 2.0

    # This is the minimum percentage of the terrain
    # area that must be covered by ejecta craters.
    # This value must be between 0 and 1.
    MIN_EJECTA_AREA_PERCENT = 0.0

    # Minimum diameter of the range of crater
    # diameters that  will be generated (meters)
    MIN_DIAMETER_M = 1

    # Step size of diameters of the range of crater
    # diameters that will be generated (meters)
    DELTA_DIAMETER_M = 1

    # Maximum diameter of the range of crater
    # diameters that will be generated (meters)
    MAX_DIAMETER_M = 500

    # Optional file for including handmarked craters
    INPUT_CRATER_FILE = None

    INPUT_CRATER_INTERPOLATE = "disabled"

    # Output XML filename
    OUTPUT_FILE = None

    #------------------------------------------
    # Plotting Flags

    PLOT_EXISTING_LOCATIONS = True

    PLOT_DIAMETER_DISTRIBUTION = True

    # PUBLIC
    
    #------------------------------------------
    # Constructor
    #
    # @param terrain: the terrain specification
    #            class
    #
    def __init__(self, terrain):
        self.terrain = terrain

        self.ejecta_crater_indices = None
        self.diameters_m = None
        self.positions_xy = None
        self.ages = None

        self._terrain = None
        self._diameter_range_m = None
        self._is_new = None

        # Existing craters read in from file
        self._exist_diameters_m = None
        self._exist_positions_xy = None
        self._exist_ages = None

    #------------------------------------------
    # Creates a filter for objects that are within the DEM
    #
    # @param self:
    # @param positions: list of positions to be filtered
    #
    def createPositionsFilter(self, positions):
        f = (np.less(positions[:,0], 0) | np.greater(positions[:,0], self.terrain.dem_size[0]) | 
                np.less(positions[:,1], 0) | np.greater(positions[:,1], self.terrain.dem_size[1]))
        return np.logical_not(f)
    
    #------------------------------------------
    # Generates a crater distribution XML file.
    # This def should be called after all 
    # tunable parameters have been set.
    #
    # @param self: 
    #
    def generate(self):
        
        print('\n\n***** Craters *****')
        print('\nMin   crater diameter: ' + str(self.MIN_DIAMETER_M))
        print('\nDelta crater diameter: ' + str(self.DELTA_DIAMETER_M))
        print('\nMax   crater diameter: ' + str(self.MAX_DIAMETER_M))
        
        read_existing_craters = False
        if self.INPUT_CRATER_FILE:
            read_existing_craters = True
        
        self.diameters_m = []
        self.positions_xy = []
        self.ages = []
        self.is_new = []

        self.diameter_range_m = np.arange(self.MIN_DIAMETER_M, self.MAX_DIAMETER_M+self.DELTA_DIAMETER_M, self.DELTA_DIAMETER_M)

        # TODO: Something is wrong with the numbers here!
        craters_per_sq_km = Craters.calculateDensity(self.diameter_range_m)
        rev_cum_dist = craters_per_sq_km * 1e-6 # Convert sq_km to sq_m
        rev_cum_dist_per_area = self.terrain.area_sq_m * rev_cum_dist
        num_craters = np.round(rev_cum_dist_per_area[0])
        num_craters = 100 # TODO!!!!
        num_craters_to_sample = num_craters
        print('\nEstimated number of craters in the terrain: %d\n' % num_craters)

        existing_hist_per_area = np.zeros((1, len(self.diameter_range_m)))
        if read_existing_craters: # TODO: Move into a function
            self.readExistingCraterFile(self.INPUT_CRATER_FILE)
            # Subtract the origin offset from the input craters, 
            # so the bottom left corner is at (0,0)
            print('self.terrain.origin = ' + str(self.terrain.origin))
            self.exist_positions_xy = self.exist_positions_xy - self.terrain.origin
            print('Total number of user-specified craters: %d\n' % len(self.exist_diameters_m))
            # Filter out craters that fall outside of the terrain bounds
            idx_bool = self.createPositionsFilter(self.exist_positions_xy) # TODO: Is this working?
            idx = np.nonzero(idx_bool)[0]
            if not idx: # We didn't actually get any new data
                read_existing_craters = False
            else:
                self.exist_positions_xy = self.exist_positions_xy[idx, :]
                self.exist_diameters_m = self.exist_diameters_m[idx]
                self.exist_ages = self.exist_ages[idx]
                num_craters_to_sample = num_craters - len(self.exist_diameters_m)
                print('Filtered Number of user-specified craters: %d\n' % len(self.exist_diameters_m))

                # TODO: Move bin right edges to bin centers?
                existing_hist_per_area = np.histogram(self.exist_diameters_m, self.diameter_range_m)[0]
                existing_hist_per_area = np.append(existing_hist_per_area, 0)

                # Plot existing crater locations
                if self.PLOT_EXISTING_LOCATIONS:
                    self.plotExistingLocations

                # Perform guided interpolation of data
                # Use the provided rev_cum_dist to guide the interpolation
                non_zero_indices = np.any(existing_hist_per_area != 0).nonzero()
                non_zero_indices = np.append(non_zero_indices, len(existing_hist_per_area))   # append the last index
                last_cur = non_zero_indices[0] # The detection algorithm can not detected crater sizes below 4.
                                                # By intializing last_cur to non_zero_indices[0] rather than zero 
                                                # we interpolate values starting with 4 and onward
                for i in range(0,len(non_zero_indices)-1):
                    cur = non_zero_indices[i + 0]
                    nxt = non_zero_indices[i + 1]
                    value = existing_hist_per_area[cur]
                    ## distribute value of existing_hist_per_area(cur) according to
                    ## one of the following methods:

                    ## Method A) equally
                    if self.INPUT_CRATER_INTERPOLATE == "uniform":
                        existing_hist_per_area[cur:nxt] = value / (nxt - cur)

                    ## Method B) guided interpolation infleunced by the rev_cum_dist
                    elif self.INPUT_CRATER_INTERPOLATE == "guided":
                        weights = rev_cum_dist[cur:nxt] / np.sum(rev_cum_dist[cur:nxt])
                        existing_hist_per_area[cur:nxt] = weights * value

                    ## Method C) same as above but treat `cur` as the center
                    elif self.INPUT_CRATER_INTERPOLATE == "guided_centered":
                        m1 = (last_cur + cur) / 2
                        m2 = (cur + nxt) / 2
                        last_cur = cur
                        weights = rev_cum_dist[m1:m2-1] / sum(rev_cum_dist[m1:m2-1])  # weights according to Env Spec
                        existing_hist_per_area[m1:m2-1] = weights * value

        env_spec_hist_in_area = utilities.revCDF_2_histogram(rev_cum_dist_per_area)

        combined_hist_per_area = env_spec_hist_in_area - existing_hist_per_area
        # clamp histogram values to a zero as min
        combined_hist_per_area = np.where(combined_hist_per_area > 0, combined_hist_per_area, 0)

        print('\nNumber of craters to sample..: ' + str(num_craters_to_sample))
        prob_dist = combined_hist_per_area / np.sum(combined_hist_per_area) # convert the histogram to a probablity dist

        # Given the number of craters, we now sample 
        # the sizes of these craters from the distribution
        # in the VIPER environment specification
        self.diameters_m = np.random.choice(
            self.diameter_range_m,
            num_craters_to_sample,
            True, # Choose with replacement
            prob_dist.squeeze())

        #if len(self.diameters_m,1) > len(self.diameters_m,2): # TODO!
        #    # Degenerate case where we have 1 item and 
        #    # return a column instead of row vector
        #    self.diameters_m = self.diameters_m'

        # Generate the probability distribution of sampled crater sizes
        sample_distribution = np.histogram(
            self.diameters_m,
            self.diameter_range_m)

        indices = np.any(self.diameter_range_m >= self.BLOCK_GEN_CSIZE).nonzero()
        if not indices:
            raise Exception('No diameters >= self.BLOCK_GEN_CSIZE!')
        idx = indices[0]
        num_large_craters = sum(sample_distribution[0][idx])
        print('\nNumber of craters >= %dm in diameter: %d\n'
                % (self.BLOCK_GEN_CSIZE, num_large_craters))

        # Plot crater diameter sample distribution
        if self.PLOT_DIAMETER_DISTRIBUTION:
            ideal_distribution = num_craters_to_sample * prob_dist.squeeze()[:-1]
            self.plotDiameterDistribution(
                ideal_distribution,
                sample_distribution[0])

        # For crater diameters, randomize delta size between 
        # steps to get a continuous size distribution.
        a = np.random.rand(1, num_craters_to_sample)
        self.diameters_m = self.diameters_m + self.DELTA_DIAMETER_M * a
        self.diameters_m = self.diameters_m.squeeze()
        
        # Crater positions are uniform (sizes are not)
        # Crater positions are drawn from a uniform distribution
        # over the terrain. Crater sizes are drawn from a power-law
        # distribution in the VIPER environmental spec.
        self.positions_xy = np.random.rand(num_craters_to_sample, 2) * \
            np.matlib.repmat(self.terrain.dem_size, num_craters_to_sample, 1)

        # Add generated craters to existing list to get total craters
        if read_existing_craters:
            self.diameters_m = np.concatenate((self.diameters_m, self.exist_diameters_m))
            self.positions_xy = np.concatenate((self.positions_xy, self.exist_positions_xy))
            self.is_new = np.concatenate((np.ones((num_craters_to_sample,1)),
                                            np.zeros((len(self.exist_diameters_m),1))))
            
            # NOTE: we currently have no intel on the age of detected craters
            #   so for now we use the same distribution for generated + existing
            #   instead of concating things:
            #       self.ages = cat(1, self.ages(:), self.exist_ages(:))
            self.ages = Craters.generateCraterAgeDiameterWise(self.diameters_m)
            combined_ages = np.concatenate((np.zeros(num_craters_to_sample), self.exist_ages))
            self.ages = np.where(combined_ages == 0, self.ages, combined_ages)
        else:
            self.is_new = np.ones((num_craters_to_sample, 1))
            self.ages = Craters.generateCraterAgeDiameterWise(self.diameters_m)
        
        # Sort by age (oldest first)
        idx = np.flip(np.argsort(self.ages)) # TODO: Don't duplicate work
        self.ages = np.flip(np.sort(self.ages))
        #[self.ages, idx] = self.ages # descending order
        self.diameters_m = self.diameters_m[idx]
        self.positions_xy = self.positions_xy[idx, :]
        self.is_new = self.is_new[idx]
        

        idx = self.createPositionsFilter(self.positions_xy)
        self.diameters_m = self.diameters_m[idx]
        self.positions_xy = self.positions_xy[idx, :]
        self.ages = self.ages[idx]
        self.is_new = self.is_new[idx]
        
        # Select ejecta craters
        # Start with craters of diameter BLOCK_GEN_CSIZE and larger.
        # Calculate their combined ejecta area. If it is less than 5#
        # of the total terrain area, let progressively smaller craters
        # have ejecta until 5# of total area is covered by ejecta.
        cum_ejecta_area_sq_m = 0
        tmp_gen_csize = self.BLOCK_GEN_CSIZE
        while cum_ejecta_area_sq_m < self.MIN_EJECTA_AREA_PERCENT * self.terrain.area_sq_m:
            self.ejecta_crater_indices = np.any(self.diameters_m >= tmp_gen_csize).nonzero()
            cum_ejecta_area_sq_m = 0
            for i in range(0,len(self.ejecta_crater_indices)):
                crater_radius_m = self.diameters_m(self.ejecta_crater_indices(i)) / 2.0
                cur_ejecta_radius_m = self.EJECTA_RADIUS_SCALAR * crater_radius_m
                cum_ejecta_area_sq_m = cum_ejecta_area_sq_m + np.pi*cur_ejecta_radius_m^2
            tmp_gen_csize = tmp_gen_csize - 1

        if self.ejecta_crater_indices:
            print('\nNumber of craters producing ejecta: #d (largest D=#gm)',
                    len(self.ejecta_crater_indices), max(self.diameters_m))
            for i in range(0,len(self.ejecta_crater_indices)):
                print('\nEjecta crater %d diameter = %fm' %
                    (i, self.diameters_m(self.ejecta_crater_indices[i])))

        self.writeXml()

    #------------------------------------------
    # @param self: 
    #
    def plotExistingLocations(self):
        fig = plt.figure(1)
        ax = fig.add_subplot(111)
        ax.clear()
        ax.plot(self.exist_positions_xy[:,0], self.exist_positions_xy[:,1], '+')
        ax.set_title('Existing Crater Locations')
        ax.set_xlabel('Terrain X (m)')
        ax.set_ylabel('Terrain Y (m)')
        ax.set_xlim([0, self.terrain.dem_size[0]])
        ax.set_ylim([0, self.terrain.dem_size[1]])

    #------------------------------------------
    # @param self: 
    # @param ideal_distribution:
    # @param sample_distribution:
    #
    def plotDiameterDistribution(self,
            ideal_distribution,
            sample_distribution):
        fig = plt.figure(2)
        ax = fig.add_subplot(111)
        ax.clear()
        print(self.diameter_range_m[:-1])
        print(ideal_distribution)
        ax.loglog(self.diameter_range_m[:-1], ideal_distribution, 'r+')
        ax.loglog(self.diameter_range_m[:-1], sample_distribution, 'bo')
        ax.set_xlabel('Crater Diameter (m)')
        ax.set_ylabel('Crater Count')
        ax.set_title('Ideal vs Sampled Crater Diameter Distributions')
        ax.legend(['Ideal', 'Sampled'])
    
    #------------------------------------------
    # Read existing craters from XML input file
    # 
    # @param self: 
    # @param filename: existing crater filename
    #
    def readExistingCraterFile(self, filename):

        tree = ET.parse(filename)
        root = tree.getroot()


        self.exist_positions_xy = None
        self.exist_diameters_m = []
        self.exist_ages = []

        for x in root:
            xy = np.array([[float(x.attrib['x']), float(x.attrib['y'])]])
            if self.exist_positions_xy is not None:
                self.exist_positions_xy = np.concatenate((self.exist_positions_xy, xy), axis=0)
            else:
                self.exist_positions_xy = xy
            self.exist_diameters_m.append(float(x.attrib['rimRadius']) * 2)
            self.exist_ages.append(float(x.attrib['freshness']))
        self.exist_diameters_m = np.array(self.exist_diameters_m)
        self.exist_ages = np.array(self.exist_ages)
#             isGen = x.attrib['isGenerated']
    
    #------------------------------------------
    # Write the output XML crater distribution file
    # NOTE: crater_size is given as diameter,
    #       though xml file expects radius
    #
    # @param self:
    #
    def writeXml(self):
        if not self.OUTPUT_FILE:
            return
        root = ET.Element("CraterList", name="UserCraters")

        for i in range(0, len(self.diameters_m)):
            #note we divide crater size by 2 to convert between diameter and radius
            c = ET.Element("CraterData",
                            x=str(self.positions_xy[i,0] + self.terrain.origin[0]),
                            y=str(self.positions_xy[i,1] + self.terrain.origin[1]),
                            rimRadius=str(self.diameters_m[i]/2),
                            freshness=str(self.ages[i]),
                            isGenerated=str(self.is_new[i][0]>0))
            root.append(c)
        tree = ET.ElementTree(root)
        ET.indent(tree, space="  ", level=0)
        try:
            print('Writing: ' + self.OUTPUT_FILE)
            with open(self.OUTPUT_FILE, 'wb') as xfile:
                tree.write(xfile)
        except Exception as e:
            print('\nUnable to write file %s\n' % self.OUTPUT_FILE)
            print(e)
            return


    # STATIC
    
    #------------------------------------------
    # Crater density def
    # from VIPER-MSE-SPEC-001 (2/13/2020)
    # 
    # @param dimameter: crater diameter(s) in meters
    # @return num_craters_per_square_km:
    #         the number of craters with diameters
    #         greater than or equal to the input
    #         argument per square KILOMETER
    #
    def calculateDensity(diameter_m):

        if any(diameter_m) <= 0:
            raise Exception('Crater diameter cannot be <= 0')

        dl = diameter_m[diameter_m <= 80]
        dh = diameter_m[diameter_m >  80]

        nl_km = np.power(29174 * dl, -1.92)
        nh_km = np.power(156228 * dh, -2.389)

        num_craters_per_square_km = np.append(nl_km, nh_km)
        return num_craters_per_square_km

    #------------------------------------------
    # sampleDepthToDiameterForSLCs
    # Refer pages 93, 94 VIPER-MSE-SPEC-001 (3/11/2021)
    # 
    # @param sample_count: number of crater ages to generate from the
    # depth-to-diameter distribution for SLC craters
    #
    def sampleDepthToDiameterForSLCs(sample_count):
        # Mean and Std can be found in Table 39. d/D Parameters for SLCs
        xs = np.linspace(0, 1, 1000)
        fresh_prob_fit = norm.pdf(xs, 0.14, 0.035)
        fresh_prob_fit = fresh_prob_fit / np.sum(fresh_prob_fit)
        Y = np.random.choice(len(fresh_prob_fit), sample_count, True, fresh_prob_fit)
        dD = xs[Y]
        return dD

    #------------------------------------------
    # mapDepthToDiameterToFreshness
    # Refer to freshness_dD_mapping.md for more details about self def
    # 
    # @param dD: A value representing the depth-to-diameter ratio of a crater
    #
    def mapDepthToDiameterToFreshness(dD):
        # Computed mapping:
        # | freshness |   d/D     |
        # |-----------|-----------|
        # | 0.0       |   0.0000  |
        # | 0.1       |   0.0241  |
        # | 0.2       |   0.0576  |
        # | 0.3       |   0.0905  |
        # | 0.4       |   0.1152  |
        # | 0.5       |   0.1468  |
        # | 0.6       |   0.1676  |
        # | 0.7       |   0.2077  |
        # | 0.8       |   0.2197  |
        # | 0.9       |   0.2441  |
        # | 1.0       |   0.2731  |
        #
        # For simplicity the mapping is assumed to be linear
        inner = np.where(1.0 / 0.2731 * dD > 0, 1.0 / 0.2731 * dD, 0)
        freshness = np.where(inner < 1.0, inner, 1.0)
        print(freshness)
        print(freshness.shape)
        return freshness

    #------------------------------------------
    # mapSlopeTypeToDepthToDiameter
    # Refer to Table 40. Crater morphological characteristics (for D<1 km) in
    # the environmental spec VIPER-MSE-SPEC-001 for details on self def
    # 
    # @param slope_type: one of five categorical values {'very steep slope',
    #   steep slope', 'moderate slope', 'gentle slope', 'very gentle slope'}
    #
    def mapSlopeTypeToDepthToDiameter(slope_type):
        dD = xmin + np.rand(1) * (xmax - xmin)
        if slope_type == 'very steep slope':
            xmin = 0.12
            xmax = 0.20    # TODO: Env Spec specifies 0.2+ but it isn't clear the meaning of it
        elif slope_type == 'steep slope':
            xmin = 0.12
            xmax = 0.20
        elif slope_type == 'moderate slope':
            xmin = 0.10
            xmax = 0.15
        elif slope_type == 'gentle slope':
            xmin = 0.07
            xmax = 0.10
        elif slope_type == 'very gentle slope':
            xmin = 0.0
            xmax = 0.07
        else:
            print('WARNING: Unexpected slope type: ' + slope_type)
            dD = 0
        return dD

    #------------------------------------------
    # sampleDepthToDiameterForNonSLCs
    # Refer to Table 40. Crater morphological characteristics (for D<1 km) in
    # the environmental spec VIPER-MSE-SPEC-001 for details on self def.
    # 
    # @param sample_count: number of crater ages to generate from the
    # depth-to-diameter distribution for non-SLC craters
    #
    def sampleDepthToDiameterForNonSLCs(sample_count):
        # Refer to Table 40. Crater morphological characteristics (for D<1 km) in
        # the environmental spec VIPER-MSE-SPEC-001 
        
        slope_types = [
            'very steep slope',
            'steep slope',
            'moderate slope',
            'gentle slope',
            'very gentle slope']
        
        slope_weights = [0.5, 2.5, 17, 30, 50]
        slope_weights = slope_weights / np.sum(slope_weights)
        
        sampled_slope_types = np.random.choice(
                                slope_types,
                                sample_count,
                                True,
                                slope_weights)

        # convert sampled_slope_types to a d/D
        dD = [mapSlopeTypeToDepthToDiameter(x) for x in sampled_slope_types]
        return dD
    
    #------------------------------------------
    # generates a plot of the distribution of d/D values for sampled SLC craters
    # The generated plot should be very similar to Figure 48 in the doc
    # VIPER-MSE-SPEC-001 (9/24/2020) (for a large enough sample)
    def plotDepthToDiameterForSLCs(data):
        fig = plt.figure(3)
        ax = fig.add_subplot(111)
        ax.clear()

        #[f,xi] = ksdensity(data)
        print(data)
        kde = scipy.stats.gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), 100)
        f = kde(x)

        ax.plot(f)
        ax.set_title('Small crater (diameter < 100m) depth-to-diameter distributions')
        ax.set_xlabel('d/D')
        ax.set_ylabel('# Probability')
        ax.set_xlim([0, 0.4])
        m = np.mean(data) # should be equal to ~0.14
        s = np.std(data)  # should be equal to ~0.035
        l = [m-3.0*s, m-2.0*s, m-1.0*s, m+1.0*s, m+2.0*s, m+3.0*s]
        b = ['3σ low', '', '', '', '', '3σ high']
        for i in range(0,len(l)):
            ax.axvline(l[i], color='r', ls='--', label=b[i])


    #------------------------------------------
    # generate crater freshness/age info
    # Refer pages 93, 94 and 95 of the Environmental Specification Document
    # No: VIPER-MSE-SPEC-001 (9/24/2020) for details on self def.
    #
    # @param sample_count: number of crater ages to generate 
    #
    # @remark: The term 'non-SLC' is not mentioned in the environmental spec
    # it is merely an interpretation.
    #
    def generateCraterAgeDiameterWise(diameter_m):
        print('generateCraterAgeDiameterWise(%d)\n' % len(diameter_m))
        ages = np.zeros((len(diameter_m), 1))

        # determine the number of craters that are considered SLC (d < 100m)
        slcs_idx = (diameter_m < 100).nonzero()[0]
        print(slcs_idx)
        slcs_count = len(slcs_idx)
        print('Total SLCs: %d\n' % slcs_count)
        sampled_slcs_dD = Craters.sampleDepthToDiameterForSLCs(slcs_count)
        result = Craters.mapDepthToDiameterToFreshness(sampled_slcs_dD)
        for r, i in zip(result, slcs_idx):
            ages[i] = r
        Craters.plotDepthToDiameterForSLCs(sampled_slcs_dD)

        # Only an upper limit is defined for non-SLCs but is not needed
        # since we don't typically generate craters larger than 500 m.
        # A lower limit, however, is needed for the implementation
        non_slcs_idx = np.any(diameter_m >= 100).nonzero()[0] # and diameter_m < 1 km
        non_slcs_count = len(non_slcs_idx)
        print('Total non-SLCs: %d\n' % non_slcs_count)
        if non_slcs_count > 0:
            sampled_non_slcs_dD = Craters.sampleDepthToDiameterForNonSLCs(non_slcs_count)
            print(sampled_non_slcs_dD)
            result = Craters.mapDepthToDiameterToFreshness(sampled_non_slcs_dD)
            for r, i in zip(result, non_slcs_idx):
                ages[i] = r
        return ages.squeeze()
