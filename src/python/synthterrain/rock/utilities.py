#!/usr/bin/env python3

import math
import numpy as np

# Utility defs for synthetic terrain 
# generation

#------------------------------------------
# @param rev_cdf: reverse cumulative 
#              distribution def
# @return cdf: cumulative distribution 
#              def
#
def revCDF_2_CDF(rev_cdf):
    cdf = rev_cdf(1) - rev_cdf
    return cdf

#------------------------------------------
# construct cdf from histogram
# 
# @param h: histogram 
# @return cdf: cdf from hist
#
def histogram_2_revCDF(h):
    cdf = np.zeros(1, len(h))
    for i in range(0, len(cdf)):
        cdf[i] = np.sum(h[i:])
    return cdf

#------------------------------------------
# construct histogram from cdf
# 
# @return pdf: histogram
# 
def revCDF_2_histogram(rev_cdf):
    h = rev_cdf[0:-1] - rev_cdf[1:]
    h = [h, 0]
    return h

#------------------------------------------
# @param rev_cdf: reverse cumulative 
#              distribution def
# @return pdf: probability density def
# 
def revCDF_2_PDF(rev_cdf):
    pdf = rev_cdf[0:-1] - rev_cdf[1:]
    pdf = [pdf, 0] 
    pdf = pdf / np.nansum(pdf)
    return pdf

#------------------------------------------
# @param location_probability_map: 
# @param amount:
# @return location_probability_map:
#
def addGradientNoise(location_probability_map, amount):
    
    [gx, gy] = np.gradient(location_probability_map)
    hfilt = np.fspecial('disk', 11)
    gx = np.imfilter(gx, hfilt)
    gy = np.imfilter(gy, hfilt)

    # Re-integrate the noise derivatives to 
    # generate the multiplicative displacement
    location_probability_map = frankotchellappa(gx, gy)
    location_probability_map = rescale(
        location_probability_map, amount) #[0 1.5]
    return location_probability_map

#------------------------------------------
# Downsample using fastest possible method with no regard
# for oversampling dense areas. 
#
# @param input: [N x M] matrix, 
#            where N is the number of points, 
#            and M is the number of attributes per point
# @param num_points: number of points in the downsampled matrix
#            1 < num_points < N
# @return output: [num_points x M]   
# @return idx:
# @author uyw (Uland Wong)
#
def downSample(input, num_points):
    num_points = min(num_points, input.size()[0])
    idx = np.random.choice(input.size()[0], num_points, False)
    output = input[idx, :]
    return [output, idx]

#------------------------------------------
# Scales the matrix into the range defined 
# by limits. For example, converting a depth 
# map into the range 0-255 (uint8) for display. 
#
# @param limits: [low high]
# @param matrix: output will have minimum value 
#                low and maximum value high
# @return matrix:
#
def rescale(matrix, limits):
    if not (len(limits)==2 and limits(1) > limits(0)):
        raise Exception('limits must be [low high]')

    if len(matrix.size()) != 2:
        raise Exception('image must be a 2.5D heightfield')

    matrix_min = matrix.min()
    matrix = matrix - matrix_min
    matrix_max = matrix.max()
    range_i = matrix_max
    range_l = limits[1] - limits[0]
    matrix = matrix / range_i * range_l + limits[0]
    return matrix

#------------------------------------------
# FRANKOTCHELLAPPA  - Generates integrable surface from gradients
#
# An implementation of Frankot and Chellappa'a algorithm for constructing
# an integrable surface from gradient information.
#
# Usage:      z = frankotchellappa(dzdx,dzdy)
#
# Arguments:  dzdx,  - 2D matrices specifying a grid of gradients of z
#             dzdy     with respect to x and y.
#
# Returns:    z      - Inferred surface heights.
#
# Reference:
#
# Robert T. Frankot and Rama Chellappa
# A Method for Enforcing Integrability in Shape from Shading
# IEEE PAMI Vol 10, No 4 July 1988. pp 439-451
#
# Note this code just implements the surface integration component of the
# paper (Equation 21 in the paper).  It does not implement their shape from
# shading algorithm.
#
# Copyright (c) 2004 Peter Kovesi
# School of Computer Science & Software Engineering
# The University of Western Australia
# http://www.csse.uwa.edu.au/
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# The Software is provided "as is", without warranty of any kind.
#
# October 2004
#
def frankotchellappa(dzdx,dzdy):

    if not dzdx.size() == dzdy.size():
      raise Exception('Gradient matrices must match')

    [rows,cols] = dzdx.size()

    # The following sets up matrices specifying frequencies in the x and y
    # directions corresponding to the Fourier transforms of the gradient
    # data.  They range from -0.5 cycles/pixel to + 0.5 cycles/pixel. The
    # fiddly bits in the line below give the appropriate result depending on
    # whether there are an even or odd number of rows and columns

    [wx, wy] = np.meshgrid((range(0,cols) - (np.fix(cols/2))) / (cols-np.mod(cols,2)), 
                           (range(0,rows) - (np.fix(rows/2))) / (rows-np.mod(rows,2)))

    # Quadrant shift to put zero frequency at the appropriate edge
    wx = np.ifftshift(wx)
    wy = np.ifftshift(wy)

    DZDX = np.fft2(dzdx)   # Fourier transforms of gradients
    DZDY = np.fft2(dzdy)

    # Integrate in the frequency domain by phase shifting by pi/2 and
    # weighting the Fourier coefficients by their frequencies in x and y and
    # then dividing by the squared frequency.  eps is added to the
    # denominator to avoid division by 0.
    eps = 2.2204e-16
    Z = (-1j*wx*DZDX -1j*wy*DZDY)/(wx^2 + wy^2 + eps)  # Equation 21

    z = np.real(np.ifft2(Z))  # Reconstruction
    return z

#------------------------------------------
# @param dem:
# @param dem_limits:
# @param crater_pos:
# @param crater_diam:
# @param size_thresh:
# @param age:
# @return total_disp:
#
def addCraterHeights(
        dem,
        dem_limits,
        crater_pos,
        crater_diam,
        size_thresh = None,
        age=None):

    if not size_thresh:
        size_thresh = 1

    dem_size = dem.size()

    # Filter by crater size first, don't process craters under size_thresh
    idx = crater_diam >= size_thresh
    crater_pos = crater_pos[idx, :]
    crater_diam = crater_diam(idx)
    # Don't process craters outside of region
    idx = crater_pos[:,0] < 1 | crater_pos[:,0] > dem_limits[1] | crater_pos[:,1] < 1 | crater_pos[:,1] > dem_limits[3]
    crater_pos = crater_pos[not idx, :]
    crater_diam = crater_diam[not idx]

    [xy_vox, unused, scale] = dem2voxel_position(dem_size, dem_limits, crater_pos)

    total_disp = np.zeros(dem_size) 
    disp_map = np.zeros(dem_size)

    #create linear age based on order
    #we start with the oldest and work our way to the fresh craters
    if not age:
        age = range(0,len(crater_diam)) / len(crater_diam)

    #crater diameter in voxels
    D = crater_diam * scale 
    sizeXY = round(1.25 * D)    #template size

    #start with minimum crater diameter
    T = len(crater_diam)
    print(1, '\n\nPlacing #d craters\n' + str(T))

    for i in range(0,T):
        #fprintf('making crater #d of #d\n', i, T)

        temp = createCraterTemplate(D[i], [sizeXY[i] sizeXY[i]], age[i])

        # if size(temp, 1) > 50
        # 	1
        # end

        # Scale the heights back to meters instead of voxels
        temp = temp/scale

        # Use random angle? 

        # Get the offset to the center of the template
        temp_pos_x = round(temp.size()[1]/2)
        temp_pos_y = round(temp.size()[0]/2)

        # Note: addTemplates flips the y-dim so we 
        # don't have to do it manually
        disp_map = addTemplates(
            total_disp,
            temp,
            xy_vox[i,0]-temp_pos_x,
            xy_vox[i,1]-temp_pos_y)

        total_disp = total_disp + disp_map

        # replace small old craters with newer large craters
        # this code currently causes weird edge effects at the rims of craters
        #absT = abs(total_disp)
        #absD = abs(disp_map)
        #
        #replace_idx = (absT < absD)
        #additive_idx = (absT >= absD)

        #total_disp(replace_idx) = total_disp(replace_idx)+ disp_map(replace_idx)
        #total_disp(additive_idx) = total_disp(additive_idx) + disp_map(additive_idx) 
    return total_disp

#------------------------------------------
# Positions in xy_vox are not rounded and 
# will give subvoxel locations. Note that
# this does *not* calculate the y-axis flip
# necessary to convert between texture coords
# and cartesian coordinates.
#
# Indices in idx_vox are rounded and can be
# used to directly access a matrix. Note:
# idx_vox is calculated so that the y-axis
# dimension flip *is* performed
#
# @param dem_size:
# @param dem_limits:
# @param xy_cart:
# @return vy_vox:
# @return idx_vox:
# @return scale:
#
def dem2voxel_position(dem_size, dem_limits, xy_cart):

    dem_range_x = dem_limits[1] - dem_limits[0]
    dem_range_y = dem_limits[3] - dem_limits[2]

    if not xy_cart:
        x = xy_cart[:,0]
        y = xy_cart[:,1]

        xs = (x - dem_limits[0]) / dem_range_x
        xs = (xs *(dem_size[1] - 1 )) + 1 # TODO check +1

        ys = (y - dem_limits[2]) / dem_range_y
        ys = (ys *(dem_size[0] - 1 )) + 1 

        #x,y position as matrix row/columns
        xy_vox[:,0] = xs
        xy_vox[:,1] = ys

        #indices into dem matrix
        idx_vox = np.ravel_multi_index(dem_size, math.floor(ys), math.floor(xs))
        #should we flip this?
        #idx_vox = sub2ind(dem_size, dem_size(1) - floor(ys) + 1, floor(xs))
    else:
        xy_vox = []
        idx_vox = []

    #overall voxel to physical unit scaling
    scale_x = dem_size[0] / dem_range_x    #voxels / meter
    scale_y = dem_size[1] / dem_range_y 

    if scale_x != scale_y:
        raise Exception('pixel x and y scales do not match')
    scale = scale_x

    return [xy_vox, idx_vox, scale]

#------------------------------------------
# diameter - pixels
# depth - pixels
# @param sizeXY: [x_dim_pixels, y_dim_pixels]
# @return out:
#
def createCraterTemplate(D, sizeXY, age):

    #dx = gpuArray.linspace( 1, sizeXY(1), sizeXY(1) )
    #dy = gpuArray.linspace( 1, sizeXY(2), sizeXY(2))

    [x,y] = np.meshgrid(range(0,sizeXY[1]), range(1,sizeXY[0]))

    x_loc = x - (sizeXY(1)+1)/2
    y_loc = y - (sizeXY(2)+1)/2

    r = math.sqrt(x_loc^2 + y_loc^2)

    # Normalize radial distance center to rim distance
    xn = 2*r / D -1
    xn[xn < -1 & x > 1] = 0 

    H0 = 0.196 * D ^ 1.01 #fresh crater depth
    #Hr0 = 0.036 * D ^ 1.01 #fresh crater rimheight
    Hr0 = 0

    H = H0 * age 
    Hr = Hr0 * age 

    Tr = 0 #height of terrain features 
    Pr = 0 # height of ground plane

    Emin = 0

    alpha = calc_alpha()
    beta = calc_beta()

    h1 = poly_h1(xn)
    h2 = poly_h2(xn)
    h3 = poly_h3(xn)
    #h4 = poly_h4(xn)

    out = h1+h2+h3

    def poly_h1(x):
        mask = x >= -1 & x < alpha
        out = (Hr0 - Hr + H) * x^2 + 2 * (Hr0 - Hr + H) * x + Hr0
        out = np.where(mask, out, 0)
        return out

    def poly_h2(x):
        mask = x >= alpha & x < 0
        out = (Hr0 - Hr + H) * (alpha + 1) / alpha * x^2 + Hr + Tr - Pr 
        out = np.where(mask, out, 0)
        return out

    def calc_alpha():
        out = (Hr + Tr -Pr - Hr0) / (Hr0 - Hr + H) 
        return out

    def poly_h3(x):
        mask = x >= 0 & x < beta
        out = (-2 * (Hr0 - Hr + H) / (3 * beta^2)  ) * x^3 \
            + ((Hr0 - Hr + H) + 2*(Hr0 - Hr + H) / beta) * x^2 \
            + (Hr + Tr - Pr) 
        out = np.where(mask, out, 0)
        return out

    def calc_beta():
        out = 1.5 * (Hr + Tr - Pr - Hr0) /  (Hr0 - Hr + H)
        return out

    def poly_h4(x):
        mask = x >= beta & x < 1
        Fc = (Emin + Tr - Pr) * x + 2 * (Pr - Tr) - Emin
        out = 0.14 * (D/2) ^ 0.74 * (x+1) ^ (-3) + Fc 
        out = np.where(mask, out, 0)
        return out

#------------------------------------------
# @param base_im:
# @param temp:
# @param x:
# @param y:
# @return disp_map:
#
def addTemplates(base_im, temp, x, y):
    dim = base_im.size()

    #this code is much faster than imwarp if we arent trying to rotate the template
    x = np.round(x)+1
    y = np.round(y)+1

    disp_map = np.zeros(dim, 'single')

    dim2 = temp.size()

    if x < 1 or y < 1:
        #hack, bottom left corner, don't add template
        pass
    elif x + dim2[1] > dim[1] or y + dim2[0] > dim[0]:
        #template extends past max limit of image, add partial template 
        add_x = min(dim2[1], dim[1] - x+1) 
        add_y = min(dim2[0], dim[0] - y+1)    

        disp_map[y:y+add_y-1, x:x+add_x-1] = temp[1:add_y, 1:add_x]
    else:
        disp_map[y:y+dim2[0]-1, x:x+dim2[1]-1] = temp  #ok in the middle of the image
    return disp_map
