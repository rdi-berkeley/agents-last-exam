################################################################################
# Copyright 2022-2024 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
#
# This file contains preprocessing algorithms for CT projection data
# If one does not have enough CPU memory to process the entire data set, then we
# recommend splitting the data across projections for the outlierCorrection
# algorithms and splitting the data across detector rows for the ringRemoval
# algorithms.
# For beam hardening correction, see the demo_leapctype/test_beam_hardening.py
################################################################################

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from leapctype import *
leapct_sweep = tomographicModels() # used in parameter_sweep function

def gain_correction(leapct, g, air_scan, dark_scan, calibration_scans=None, ROI=None, badPixelMap=None, flux_response=None):
    r""" Performs gain correction
    
    This function processes raw radiographs by subtracting off the dark current and
    correcting for the pixel-to-pixel gain variations which reduces ring artifacts
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (C contiguous float32 numpy array or torch tensor): radiograph data
        air_scan (C contiguous float32 numpy array or torch tensor): air scan radiograph data; if not given, assumes that the input data is transmission data
        dark_scan (C contiguous float32 numpy array or torch tensor): dark scan radiograph data; if not given assumes the inputs have already been dark subtracted
        calibrartion_scans (C contiguous 3D float32 numpy array or torch tensor): calibration scan data
        ROI (4-element integer array): specifies a bounding box ([first row, last row, first column, last column]) for which to estimate a mean value for flux correction
        badPixelMap (C contiguous float32 numpy array or torch tensor): 2D bad pixel map (numRows x numCols) where a value of 1.0 marks a pixel as bad
        flux_response (C contiguous 1D float32 numpy array or torch tensor): transfer function of dark subtracted raw data
    """
    if dark_scan is None or air_scan is None:
        print('Error: must specify air_scan and dark_scan')
        return False
    if calibration_scans is not None and len(calibration_scans.shape) != 3:
        print('Error: calibration_scans should be a 3D array')
        return False
        
    # Check Inputs
    if g is None:
        print('Error: no data given')
        return False
    if len(g.shape) != 3:
        print('Error: input data must by 3D')
        return False
    if dark_scan is not None:
        if isinstance(dark_scan, int) or isinstance(dark_scan, float):
            pass
        elif len(dark_scan.shape) != 2 or  g.shape[1] != dark_scan.shape[0] or g.shape[2] != dark_scan.shape[1]:
            print('Error: dark scan image size is invalid')
            return False
    if air_scan is not None:
        if isinstance(air_scan, int) or isinstance(air_scan, float):
            pass
        elif len(air_scan.shape) != 2 or g.shape[1] != air_scan.shape[0] or g.shape[2] != air_scan.shape[1]:
            print('Error: air scan image size is invalid')
            return False
    if ROI is not None:
        if ROI[0] < 0 or ROI[2] < 0 or ROI[1] < ROI[0] or ROI[3] < ROI[2] or ROI[1] >= g.shape[1] or ROI[3] >= g.shape[2]:
            print('Error: invalid ROI')
            return False
    
    if leapct is None:
        leapct = leapct_sweep

    # Subtract off dark
    if isinstance(dark_scan, int) or isinstance(dark_scan, float):
        g[:] = g[:] - dark_scan
        if isinstance(air_scan, int) or isinstance(air_scan, float):
            air_scan = air_scan - dark_scan
        else:
            air_scan[:] = air_scan[:] - dark_scan
    else:
        g[:] = g[:] - dark_scan[None,:,:]
        air_scan[:] = air_scan[:] - dark_scan[:]

    if isinstance(air_scan, int) or isinstance(air_scan, float):
        pass
    else:
        leapct.MedianFilter2D(air_scan, threshold=0.03, windowSize=5)
    
    if calibration_scans is None or calibration_scans.shape[0] < 3:
        pass
    else:
        # do actual calibration
        if calibration_scans.shape[0] == 3:
            M = calibration_scans[1,:,:] - calibration_scans[0,:,:]
            L = calibration_scans[2,:,:] - calibration_scans[0,:,:]

            leapct.MedianFilter2D(M, threshold=0.1, windowSize=5)
            leapct.MedianFilter2D(L, threshold=0.1, windowSize=5)
            
            if has_torch == True and type(L) is torch.Tensor:
                minL = torch.min(L[L>0.0])
                minM = torch.min(M[M>0.0])
            else:
                minL = np.min(L[L>0.0])
                minM = np.min(M[M>0.0])
            M[M<=0.0] = minM
            L[L<=0.0] = minL
            
            #leapct.MedianFilter2D(M)
            #leapct.MedianFilter2D(L)
            medM = np.median(M)
            medL = np.median(L)
            
            gainM = medM/M
            gainL = (medL - medM) / (L - M)
            
            """
            Although this does not work in python, here is the basic math
            
            For g <= M:
                g = gainM * g
            For g > M:
                g = gainL * (g-M) + medM
            
            This is basically the way Jerel was doing it
            g[:] = g[:] * gainM[None,:,:]
            air_scan[:] = air_scan[:] * gainM[:]
            gainL[:] = gainL[:] / gainM[:]
            
            g[:] = leapct.minimum(g[:], medM) + gainL[None,:,:] * (leapct.maximum(g[:], medM) - medM)
            """

            if has_torch == True and type(g) is torch.Tensor:
                g[:] = torch.heaviside(M[None,:,:]-g[:], 0.0) * ((gainM[None,:,:]-gainL[None,:,:])*g[:] + gainL[None,:,:]*M[None,:,:] - medM) + (gainL[None,:,:]*(g[:]-M[None,:,:]) + medM)
            else:
                g[:] = np.heaviside(M[None,:,:]-g[:], 0.0) * ((gainM[None,:,:]-gainL[None,:,:])*g[:] + gainL[None,:,:]*M[None,:,:] - medM) + (gainL[None,:,:]*(g[:]-M[None,:,:]) + medM)
            
            if isinstance(air_scan, int) or isinstance(air_scan, float):
                pass
            else:
                air_scan[air_scan<=M] = gainM[air_scan<=M]*air_scan[air_scan<=M]
                air_scan[air_scan>M] = gainL[air_scan>M]*(air_scan[air_scan>M] - M[air_scan>M]) + medM
            
            #return True
            
        else:
            print('Error: current implementation only works for 3 calibration scans!')
            return False
            
    # Perform Flux Correction
    if ROI is not None:
        if isinstance(air_scan, int) or isinstance(air_scan, float):
            postageStamp_air = float(air_scan)
        elif has_torch == True and type(air_scan) is torch.Tensor:
            postageStamp_air = torch.mean(air_scan[ROI[0]:ROI[1]+1, ROI[2]:ROI[3]+1])
        else:
            postageStamp_air = np.mean(air_scan[ROI[0]:ROI[1]+1, ROI[2]:ROI[3]+1])
    
        if has_torch == True and type(g) is torch.Tensor:
            postageStamp = torch.mean(g[:,ROI[0]:ROI[1]+1, ROI[2]:ROI[3]+1], axis=(1,2))
        else:
            postageStamp = np.mean(g[:,ROI[0]:ROI[1]+1, ROI[2]:ROI[3]+1], axis=(1,2))
            
        postageStamp = postageStamp / postageStamp_air
        print('ROI mean: ' + str(np.mean(postageStamp)) + ', standard deviation: ' + str(np.std(postageStamp)))
        #g[:,:,:] = g[:,:,:] / postageStamp[:,None,None]
        g[:] = g[:] / postageStamp[:,None,None]
        
    badPixelCorrection(leapct, g, None, None, badPixelMap, 5, isAttenuationData=False)
        
    return True

def makeAttenuationRadiographs(leapct, g, air_scan=None, dark_scan=None, ROI=None, isAttenuationData=False):
    r"""Converts data to attenuation radiographs (flat fielding and negative log)

    .. math::
       \begin{eqnarray}
         transmission\_data &=& (raw - dark\_scan) / (air\_scan - dark\_scan) \\
         attenuation\_data &=& -log(transmission\_data)
       \end{eqnarray}

    This function assumes the input data is never attenuation data.  See argument descriptions below for how
    the input data is treated.
       
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (C contiguous float32 numpy array or torch tensor): radiograph data
        air_scan (C contiguous float32 numpy array or torch tensor): air scan radiograph data; if not given, assumes that the input data is transmission data
        dark_scan (C contiguous float32 numpy array or torch tensor): dark scan radiograph data; if not given assumes the inputs have already been dark subtracted
        ROI (4-element integer array): specifies a bounding box ([first row, last row, first column, last column]) for which to estimate a mean value for flux correction
    
    Returns:
        True if successful, False otherwise
    """

    # Check Inputs
    if g is None:
        print('Error: no data given')
        return False
    if len(g.shape) != 3:
        print('Error: input data must by 3D')
        return False
    if dark_scan is not None:
        if isinstance(dark_scan, int) or isinstance(dark_scan, float):
            #print('dark is constant')
            pass
        elif len(dark_scan.shape) != 2 or g.shape[1] != dark_scan.shape[0] or g.shape[2] != dark_scan.shape[1]:
            print('Error: dark scan image size is invalid')
            return False
    if air_scan is not None:
        if isinstance(air_scan, int) or isinstance(air_scan, float):
            #print('air is constant')
            pass
        elif len(air_scan.shape) != 2 or g.shape[1] != air_scan.shape[0] or g.shape[2] != air_scan.shape[1]:
            print('Error: air scan image size is invalid')
            return False
    if ROI is not None:
        if ROI[0] < 0 or ROI[2] < 0 or ROI[1] < ROI[0] or ROI[3] < ROI[2] or ROI[1] >= g.shape[1] or ROI[3] >= g.shape[2]:
            print('Error: invalid ROI')
            return False
    
    if leapct is None:
        leapct = leapct_sweep

    #"""
    if has_torch == True and type(air_scan) is torch.Tensor:
        minAir = torch.min(air_scan[air_scan>0.0])
        air_scan[air_scan<=0.0] = minAir
    elif type(air_scan) is np.ndarray:
        minAir = np.min(air_scan[air_scan>0.0])
        air_scan[air_scan<=0.0] = minAir
    #"""
    
    # The input may already be attenuation data
    # Then this algorithm may just apply the flux normalization
    # So in this case transform to transmission data
    if isAttenuationData:
        if ROI is None:
            return True
        else:
            leapct.expNeg(g)
    
    # Perform Flat Fielding
    
    if dark_scan is not None:
        if air_scan is not None:
            if isinstance(dark_scan, int) or isinstance(dark_scan, float):
                air_scan = air_scan - dark_scan
                if isinstance(air_scan, int) or isinstance(air_scan, float):
                    g[:] = (g[:] - dark_scan) / air_scan
                else:
                    g[:] = (g[:] - dark_scan) / air_scan[None,:,:]
            else:
                g[:] = (g[:] - dark_scan[None,:,:]) / (air_scan - dark_scan)[None,:,:]
        else:
            if isinstance(dark_scan, int) or isinstance(dark_scan, float):
                g[:] = g[:] - dark_scan
            else:
                g[:] = g[:] - dark_scan[None,:,:]
    else:
        if isinstance(air_scan, int) or isinstance(air_scan, float):
            g[:] = g[:] / air_scan
        elif air_scan is not None:
            g[:] = g[:] / air_scan[None,:,:]
    
    # Perform Flux Correction
    if ROI is not None:
        if has_torch == True and type(g) is torch.Tensor:
            postageStamp = torch.mean(g[:,ROI[0]:ROI[1]+1, ROI[2]:ROI[3]+1], axis=(1,2))
        else:
            postageStamp = np.mean(g[:,ROI[0]:ROI[1]+1, ROI[2]:ROI[3]+1], axis=(1,2))
        print('ROI mean: ' + str(np.mean(postageStamp)) + ', standard deviation: ' + str(np.std(postageStamp)))
        #g[:,:,:] = g[:,:,:] / postageStamp[:,None,None]
        g[:] = g[:] / postageStamp[:,None,None]
        
    # Convert to attenuation
    if np.isnan(g).any():
        print('some nans exist')
    g[g<=0.0] = 2.0**-16
    leapct.negLog(g)
    
    return True

def badPixelCorrection(leapct, g, air_scan=None, dark_scan=None, badPixelMap=None, windowSize=3, isAttenuationData=True):
    r"""Removes bad pixels from CT projections
    
    LEAP CT geometry parameters must be set prior to running this function 
    and can be applied to any CT geometry type.
    This algorithm processes each projection independently
    and removes bad pixels specified by the user using a median filter
    
    If no bad pixel map is provided, this routine will estimate it from the average of all projections.
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation or transmission projection data
        badPixelMap (C contiguous float32 numpy array or torch tensor): 2D bad pixel map (numRows x numCols) where a value of 1.0 marks a pixel as bad
        windowSize (int): the window size; can be 3, 5, or 7
        isAttenuationData (bool): True if g is attenuation data, False otherwise
        
    Returns:
        True if successful, False otherwise
    """
    
    if g is None:
        return False
    
    if leapct is None:
        leapct = leapct_sweep

    # This algorithm processes each transmission
    if isAttenuationData:
        leapct.expNeg(g)
        
    if badPixelMap is None:
        if has_torch == True and type(g) is torch.Tensor:
            g_mean = torch.mean(g, axis=0)
        else:
            g_mean = np.mean(g, axis=0)
        g_mean_filtered = leapct.copyData(g_mean)
        #leapct.MedianFilter2D(g_mean_filtered, threshold=0.03, windowSize=3)
        leapct.MedianFilter2D(g_mean_filtered, threshold=0.1, windowSize=5)
        ind = g_mean != g_mean_filtered
        g_mean[ind] = 1.0
        g_mean[~ind] = 0.0
        badPixelMap = g_mean
        #print('Estimated ' + str(np.sum(badPixelMap)) + ' bad pixels')
        
    leapct.badPixelCorrection(g, badPixelMap, windowSize)
    if air_scan is not None:
        leapct.badPixelCorrection(air_scan, badPixelMap, windowSize)
    if dark_scan is not None:
        leapct.badPixelCorrection(dark_scan, badPixelMap, windowSize)
    if isAttenuationData:
        leapct.negLog(g)
    return True

def outlierCorrection(leapct, g, threshold=0.03, windowSize=3, isAttenuationData=True):
    r"""Removes outliers (zingers) from CT projections
    
    No LEAP parameters need to be set for this function to work 
    and can be applied to any CT geometry type.
    This algorithm processes each projection independently
    and removes outliers by a thresholded median filter
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation or transmission projection data
        threshold (float): A pixel will be replaced by the median of its neighbors if \|g - median(g)\|/median(g) > threshold
        windowSize (int): The window size of the median filter applied is windowSize x windowSize
        isAttenuationData (bool): True if g is attenuation data, False otherwise
        
    Returns:
        True if successful, False otherwise
    """
    
    if g is None:
        return False
    if leapct is None:
        leapct = leapct_sweep
    
    # This algorithm processes each transmission
    if isAttenuationData:
        leapct.expNeg(g)
    leapct.MedianFilter2D(g, threshold, windowSize)
    if isAttenuationData:
        leapct.negLog(g)
    return True

def outlierCorrection_highEnergy(leapct, g, isAttenuationData=True):
    """Removes outliers (zingers) from CT projections
    
    No LEAP parameters need to be set for this function to work 
    and can be applied to any CT geometry type.
    This algorithm processes each projection independently
    and removes outliers by a series of three thresholded median filters
    
    This outlier correction algorithm should most be used for MV CT or neutron CT
    where outliers effect a larger neighborhood of pixels.  This function uses a three-stage
    thresholded median filter to remove outliers.
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation or transmission projection data
        isAttenuationData (bool): True if g is attenuation data, False otherwise
        
    Returns:
        True if successful, False otherwise
    """
    
    if leapct is None:
        leapct = leapct_sweep
    
    if isAttenuationData:
        leapct.expNeg(g)
    leapct.MedianFilter2D(g, 0.08, 7)
    leapct.MedianFilter2D(g, 0.024, 5)
    leapct.MedianFilter2D(g, 0.0032, 3)
    if isAttenuationData:
        leapct.negLog(g)
    return True

def LowSignalCorrection(leapct, g, threshold=0.03, windowSize=3, signalThreshold=0.001, isAttenuationData=True):
    r"""Corrects detector pixels that have very low transmission (photon starvation)
    
    No LEAP parameters need to be set for this function to work 
    and can be applied to any CT geometry type.
    This algorithm processes each projection independently
    and removes low signal errors by a double-thresholded median filter
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation or transmission projection data
        threshold (float): A pixel will be replaced by the median of its neighbors if \|g - median(g)\|/median(g) > threshold
        windowSize (int): The window size of the median filter applied is windowSize x windowSize
        signalThreshold (float): threshold where only values less than this parameter will be corrected
        isAttenuationData (bool): True if g is attenuation data, False otherwise
        
    Returns:
        True if successful, False otherwise
    """
    
    if g is None:
        return False
    if leapct is None:
        leapct = leapct_sweep

    # This algorithm processes each transmission
    if isAttenuationData:
        leapct.expNeg(g)
    leapct.LowSignalCorrection2D(g, threshold, windowSize, signalThreshold)
    if isAttenuationData:
        leapct.negLog(g)
    return True    

def detectorDeblur_FourierDeconv(leapct, g, H, WienerParam=0.0, isAttenuationData=True):
    """Removes detector blur by fourier deconvolution
    
    Args:
        g (contiguous float32 numpy array or torch tensor): attenuation or transmission projection data
        H (2D contiguous float32 numpy array or torch tensor): Magnitude of the frequency response of blurring psf, DC is at [0,0]
        WienerParam (float): Parameter for Wiener deconvolution, number should be between 0.0 and 1.0
        isAttenuationData (bool): True if g is attenuation data, False otherwise
    
    Returns:
        True if successful, False otherwise
    """
    if leapct is None:
        print('Error: must provide leapct object')
        return False
    if has_torch == True and type(H) is torch.Tensor:
        H = H.cpu().detach().numpy()
    if np.min(np.abs(H)) < 1.0/100.0:
        WienerParam = max(2.5e-5, WienerParam)
    if 0 < WienerParam and WienerParam <= 1.0:
        H = (1.0+WienerParam)*H/(H*H+WienerParam)
    else:
        H = 1.0 / H
    H = H / H[0,0]
    leapct.transmission_filter(g, H, isAttenuationData)
    return True
    
def detectorDeblur_RichardsonLucy(leapct, g, H, numIter=10, isAttenuationData=True):
    r"""Removes detector blur by Richardson-Lucy iterative deconvolution
    
    Richardson-Lucy iterative deconvolution is developed for Poisson-distributed data
    and inherently preserve the non-negativity of the input.  It uses the following update step,
    where t = transmission data
    
    .. math::
       \begin{eqnarray}
         t_{n+1} &=& t_n H^T\left[ \frac{t_0}{Ht_n} \right]
       \end{eqnarray}
    
    Args:
        g (contiguous float32 numpy array or torch tensor): attenuation or transmission projection data
        H (2D contiguous float32 numpy array or torch tensor): Magnitude of the frequency response of blurring psf, DC is at [0,0]
        numIter (int): Number of iterations
        isAttenuationData (bool): True if g is attenuation data, False otherwise
    
    Returns:
        True if successful, False otherwise
    """
    if leapct is None:
        print('Error: must provide leapct object')
        return False
    H = H / H[0,0]
    if isAttenuationData:
        leapct.expNeg(g)
    t = g
    t_0 = leapct.copyData(t)
    Ht = leapct.copyData(t)
    for n in range(numIter):
        Ht[:] = t[:]
        Ht = leapct.transmission_filter(Ht, H, False)
        Ht[:] = t_0[:] / Ht[:]
        Ht = leapct.transmission_filter(Ht, H, False)
        t[:] = t[:] * Ht[:]
    if isAttenuationData:
        leapct.negLog(t)
    return True

def ringRemoval_fast(leapct, g, delta=0.01, beta=1.0e3, numIter=30, maxChange=0.05, average_in_transmission_space=False):
    r"""Removes detector pixel-to-pixel gain variations that cause ring artifacts in reconstructed images
    
    This algorithm estimates the rings by first averaging all projections.  Then denoises this
    signal by minimizing the TV norm.  Finally the gain correction map to correct the data
    is estimated by the difference of the TV-smoothed and averaged projection data (these are all 2D signals).
    This is summarized by the math equations below.
    
    .. math::
       \begin{eqnarray}
         \overline{g} &:=& \frac{1}{numAngles}\sum_{angles} g \\
         gain\_correction &:=& argmin_x \; \left[\frac{1}{2} \|x - \overline{g}\|^2 + \beta TV(x) \right] - \overline{g} \\
         g\_corrected &:=& g + gain\_correction
       \end{eqnarray}
    
    Assumes the input data is in attenuation space.
    No LEAP parameters need to be set for this function to work 
    and can be applied to any CT geometry type.
    This algorithm is effective at removing ring artifacts and runs fast, but can
    sometimes create new rings of tangents of sharp transitions.
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation projection data
        delta (float): The delta parameter of the Total Variation Functional
        numIter (int): Number of iterations
        maxChange (float): An upper limit on the maximum difference that can be applied to a detector pixels
        beta (float): The strength of the regularization
    
    Returns:
        True if successful, False otherwise
    """
    
    """
    if has_torch == True and type(g) is torch.Tensor:
        g_sum = torch.zeros((1,g.shape[1],g.shape[2]), dtype=torch.float32)
        g_sum = g_sum.to(g.get_device())
        g_sum[0,:] = torch.mean(g,axis=0)
    else:
        g_sum = np.zeros((1,g.shape[1],g.shape[2]), dtype=np.float32)
        g_sum[0,:] = np.mean(g,axis=0)
    """
    if leapct is None:
        leapct = leapct_sweep

    if average_in_transmission_space:
        leapct.expNeg(g)
    if has_torch == True and type(g) is torch.Tensor:
        g_sum = torch.mean(g,axis=0)
    else:
        g_sum = np.mean(g,axis=0)
    if average_in_transmission_space:
        leapct.negLog(g_sum)
        leapct.negLog(g)
    g_sum_save = leapct.copyData(g_sum)
    minValue = np.min(g_sum_save)
    minValue = min(minValue, 0.0)
    
    numNeighbors = leapct.get_numTVneighbors()
    leapct.set_numTVneighbors(6)
    #leapct.diffuse(g_sum, delta, numIter, 1.0)
    leapct.TV_denoise(g_sum, delta, beta, numIter, 1.0)
    g_sum[g_sum<minValue] = minValue
    leapct.set_numTVneighbors(numNeighbors)
    
    gainMap = g_sum - g_sum_save
    
    gainMap[gainMap>maxChange] = maxChange
    gainMap[gainMap<-maxChange] = -maxChange
    g[:] = g[:] + gainMap[None,:,:]
    return True

def ringRemoval_median(leapct, g, threshold=0.0, windowSize=5, numIter=1):
    r"""Removes detector pixel-to-pixel gain variations that cause ring artifacts in reconstructed images
    
    .. math::
       \begin{eqnarray}
         gain\_correction &:=& \frac{1}{numAngles}\sum_{angles} [MedianFilter2D(g) - g] \\
         g\_corrected &:=& g + gain\_correction
       \end{eqnarray}
    
    Assumes the input data is in attenuation space.
    No LEAP parameters need to be set for this function to work 
    and can be applied to any CT geometry type.
    This algorithm is effective at removing ring artifacts without creating new ring artifacts.
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation projection data
        threshold (float): The threshold for the thresholded median filter, where a pixel will be replaced by the median of its neighbors if \|g - median(g)\|/median(g) > threshold
        windowSize (int): The window size of the median filter applied is windowSize x windowSize
        numIter (int): Number of iterations
    
    Returns:
        True if successful, False otherwise
    """
    
    if leapct is None:
        leapct = leapct_sweep
    Dg = leapct.copyData(g)
    for n in range(numIter):
        if n > 0:
            Dg[:] = g[:]
        leapct.MedianFilter2D(Dg, threshold, windowSize)
        Dg[:] = g[:] - Dg[:] # noise only
        if has_torch == True and type(g) is torch.Tensor:
            Dg_sum = torch.mean(Dg,axis=0)
        else:
            Dg_sum = np.mean(Dg,axis=0)

        g[:] = g[:] - Dg_sum[None,:,:]
    return True

def ringRemoval(leapct, g, delta=0.01, beta=1.0e1, numIter=30, maxChange=0.05):
    r"""Removes detector pixel-to-pixel gain variations that cause ring artifacts in reconstructed images
    
    This algorithm estimates the gain correction necessary to remove ring artifacts by solving denoising
    the data by minimizing the TV norm with the gradient step determined by averaging the gradients over
    all angles.
    
    Assumes the input data is in attenuation space.
    No LEAP parameters need to be set for this function to work and can be applied to any CT geometry type.
    This algorithm is effective at removing ring artifacts without creating new ring artifacts, 
    but is more computationally expensive than ringRemoval_fast.

    This algorithm works by minimizing the standard TV denoising cost function with gradient descent
    except we replace the gradient by averaging it over all projections.  This ensures that the
    projections are smoothed by a function that is constant over projection angles.
    The cost function is given by

    .. math::
       \begin{eqnarray}
       \frac{1}{2} \|x - g\|^2 + \beta TV(x)
       \end{eqnarray}
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation projection data
        delta (float): The delta parameter of the Total Variation Functional
        beta (float): The strength of the regularization
        numIter (int): Number of iterations
    
    Returns:
        True if successful, False otherwise
    """
    if leapct is None:
        leapct = leapct_sweep
    numNeighbors = leapct.get_numTVneighbors()
    leapct.set_numTVneighbors(6)
    """
    g_0 = leapct.copyData(g)
    for n in range(numIter):
        Dg = leapct.TVgradient(g, delta, beta)
        if has_torch == True and type(Dg) is torch.Tensor:
            Dg_sum = torch.mean(Dg,axis=0)
        else:
            Dg_sum = np.mean(Dg,axis=0)
        Dg[:] = Dg_sum[None,:,:]

        Dg[:] = g[:] - g_0[:] + Dg[:]
        
        num = leapct.innerProd(Dg,Dg)
        denom = leapct.TVquadForm(g, Dg, delta, beta)
        
        stepSize = num / (num+denom)
        
        g[:] = g[:] - stepSize*Dg[:]
    """
    leapct.TV_denoise(g, delta, beta, numIter, p=1.0, meanOverFirstDim=True)
    
    '''
    gainMap = g - g_0
    gainMap[gainMap>maxChange] = maxChange
    gainMap[gainMap<-maxChange] = -maxChange
    g = g_0 + gainMap
    #'''
    leapct.set_numTVneighbors(numNeighbors)
    
    return True

def transmission_shift(leapct, g, shift, isAttenuationData=True):
    r""" Subtracts constant from transmission data which is a simple method for scatter correction
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation projection data
        shift (float): the amount to subtract from the transmission data
        isAttenuationData (bool): True if g is attenuation data, False otherwise
    
    Returns:
        True is successful, False otherwise
    
    """
    if leapct is None:
        leapct = leapct_sweep
    if not isinstance(shift, float):
        raise TypeError
    if shift >= 1.0:
        return False
    minTransmission = 2.0**-16
    if isAttenuationData:
        leapct.expNeg(g)
    scale = 1.0 / (1.0 - shift)
    #g = (g - shift) / (1.0 - shift)
    g -= shift
    g *= scale
    g[g<minTransmission] = minTransmission
    
    if isAttenuationData:
        leapct.negLog(g)
    return True
    
def quadratic_extrema(x_0, x_1, x_2, y_0, y_1, y_2):
    optimal_value = x_1
    metric_value = y_1
    a = y_0/((x_0-x_1)*(x_0-x_2)) + y_1/((x_1-x_0)*(x_1-x_2)) + y_2/((x_2-x_0)*(x_2-x_1))
    b = y_0*(-x_1-x_2)/((x_0-x_1)*(x_0-x_2)) + y_1*(-x_0-x_2)/((x_1-x_0)*(x_1-x_2)) + y_2*(-x_0-x_1)/((x_2-x_0)*(x_2-x_1))
    c = y_0*x_1*x_2/((x_0-x_1)*(x_0-x_2)) + y_1*x_0*x_2/((x_1-x_0)*(x_1-x_2)) + y_2*x_0*x_1/((x_2-x_0)*(x_2-x_1))
    if a != 0.0:
        optimal_value_interp = -b/(2.0*a)
        if x_0 <= optimal_value_interp and optimal_value_interp <= x_2:
            optimal_value = optimal_value_interp
        metric_value = a*optimal_value**2 + b*optimal_value + c
    return optimal_value, metric_value

def find_centerCol_and_tiltAngle(leapct, g, centerCols, tilts, method=None, iz=None):
    return geometric_calibration(leapct, g, centerCols, tilts, 'centerCol', method, iz)

def find_tau_and_tiltAngle(leapct, g, taus, tilts, method=None, iz=None):
    return geometric_calibration(leapct, g, taus, tilts, 'tau', method, iz)

def geometric_calibration(leapct, g, shifts, tilts, param='centerCol', method=None, iz=None):
    r"""Automatic estimation of tiltAngle and centerCol or tau
    
    The CT geometry parameters and the CT volume parameters must be set prior to running this function.
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation projection data
        shifts (list of floats): the values of centerCol or tau to consider
        tilts (list of floats): the values of tiltAngle to consider
        param (string): specify centerCol or tau
        method (string): which metric to use: can be inconsistency or bowtie 
        iz (integer): the z-slice index to perform the reconstruction; if not given, uses the central slice
        
    Returns:
        the value of the metric at the optimal value
    """
    if param != 'centerCol' and param != 'tau':
        print('Error: please specify centerCol or tau')
        return None

    if method is None:
        s_max = leapct.get_pixelWidth()*0.5*(leapct.get_numCols()-1)
        dFOV_nominal = 2.0*leapct.get_sod()/leapct.get_sdd()*s_max / np.sqrt(1.0 + (s_max/leapct.get_sdd())**2)
        dFOV_min = leapct.get_diameterFOV_min()
        if dFOV_min <= 0.1*dFOV_nominal:
            method = 'bowtie'
        else:
            method = 'inconsistency'

    metrics = np.zeros_like(tilts)
    centerCol_est = np.zeros_like(tilts)
    for m in range(tilts.size):
        print('tiltAngle = ' + str(tilts[m]))
        leapct.set_tiltAngle(tilts[m])
        if method == 'inconsistency':
            dont_care, opt = parameter_sweep(leapct, g, shifts, param, iz, algorithmName='inconsistency', set_optimal=True)
        else:
            opt = find_centerCol_or_tau_bowtie(leapct, g, shifts, iRow=iz)
        metrics[m] = opt
        if param == 'centerCol':
            centerCol_est[m] = leapct.get_centerCol()
            print('estimated centerCol = ' + str(leapct.get_centerCol()))
        else:
            centerCol_est[m] = leapct.get_tau()
            print('estimated tau = ' + str(leapct.get_tau()))
        print('optimal error = ' + str(opt))
        
    values = tilts
    ind_best = np.argmin(metrics)
    optimal_value = values[ind_best]
    metric_value = metrics[ind_best]

    if 0 < ind_best and ind_best < metrics.size-1:
        optimal_value, metric_value = quadratic_extrema(values[ind_best-1], values[ind_best], values[ind_best+1], metrics[ind_best-1], metrics[ind_best], metrics[ind_best+1])

        best_tilt = optimal_value
        leapct.set_tiltAngle(best_tilt)
        if method == 'inconsistency':
            dont_care, opt = parameter_sweep(leapct, g, shifts, param, iz, algorithmName='inconsistency', set_optimal=True)
        else:
            opt = find_centerCol_or_tau_bowtie(leapct, g, shifts, param, iRow=iz)
        return opt
    else:
        best_tilt = tilts[ind_best]
        best_centerCol = centerCol_est[ind_best]
        if param == 'centerCol':
            leapct.set_centerCol(best_centerCol)
        else:
            leapct.set_tau(best_centerCol)
        leapct.set_tiltAngle(best_tilt)
        return metric_value
    
def find_centerCol_or_tau_bowtie(leapct, g, values, param='centerCol', iRow=-1):
    if param != 'centerCol' and param != 'tau':
        print('Error: please specify centerCol or tau')
        return None
    if iRow is None:
        iRow = max(0, min(leapct.get_numRows()-1, int(leapct.get_centerRow()+0.5)))
    #centerCol_best = leapct.get_centerCol()
    #metric_best = bowtie_alignment_metric(leapct, g, iRow)
    metrics = np.zeros(len(values))
    for i in range(len(values)):
        if param == 'centerCol':
            leapct.set_centerCol(values[i])
        else:
            leapct.set_tau(values[i])
        metrics[i] = bowtie_alignment_metric(leapct, g, iRow)
    
    ind_best = np.argmin(metrics)
    optimal_value = values[ind_best]
    metric_value = metrics[ind_best]
    
    if 0 < ind_best and ind_best < metrics.size-1:
        optimal_value, metric_value = quadratic_extrema(values[ind_best-1], values[ind_best], values[ind_best+1], metrics[ind_best-1], metrics[ind_best], metrics[ind_best+1])
            
    if param == 'centerCol':
        leapct.set_centerCol(optimal_value)
    else:
        leapct.set_tau(optimal_value)

    return metric_value

def bowtie_alignment_metric(leapct, g, iRow=-1, doPlot=False):
    if leapct.get_geometry() == 'CONE' or leapct.get_geometry() == 'FAN':
        if leapct.get_offsetScan():
            sino_180 = leapct.rebin_parallel_sinogram(g, 6, iRow)
            sino = np.concatenate((sino_180, np.flip(sino_180, axis=1)))
        else:
            sino = leapct.rebin_parallel_sinogram(g, 6, iRow)
    else:
        if iRow < 0:
            iRow = leapct.get_numRows()//2
        sino = np.squeeze(g[:,iRow,:])
    bowtie = np.abs(np.fft.fftshift(np.fft.fft2(sino)))
        
    nu = 0.05
    slope = (1.0/np.pi)*sino.shape[0]/sino.shape[1]
    phi = np.array(range(sino.shape[0]),dtype=np.float32)
    s = np.array(range(sino.shape[1]),dtype=np.float32)
    phi -= np.mean(phi)
    s -= np.mean(s)
    phi /= np.max(phi)
    s /= np.max(s)
    s,phi = np.meshgrid(s,phi)
    mask = np.zeros_like(bowtie)
    mask[np.logical_and(np.abs(phi)>nu, (1.0-2.0*nu)*slope*np.abs(phi)>np.abs(s))] = 1.0

    if doPlot:
        print(iRow)
        plt.imshow(sino, cmap='gray')
        plt.show()

    return np.sum(mask*bowtie)/np.sum(mask)

def parameter_sweep(leapct, g, values, param='centerCol', iz=None, algorithmName='FBP', set_optimal=False, isFiltered=False):
    r"""Performs single-slice reconstructions of several values of a given parameter
    
    The CT geometry parameters and the CT volume parameters must be set prior to running this function.
    
    The parameters to sweep are all standard LEAP CT geometry parameter names, except 'tilt' which is only available for cone- and modular-beam data.
    (note that the data g is not rotated, just the model of the detector orientation which is better because no interpolation is necessary).
    
    Args:
        leapct (tomographicModels object): This is just needed to access LEAP algorithms
        g (contiguous float32 numpy array or torch tensor): attenuation projection data
        values (list of floats): the values to reconstruct with
        param (string): the name of the parameter to sweep; can be 'centerCol', 'centerRow', 'tau', 'sod', 'sdd', 'tilt', 'vertical_shift', 'horizontal_shift'
        iz (integer): the z-slice index to perform the reconstruction; if not given, uses the central slice
        algorithmName (string): the name of the algorithm to use for reconstruction; can be 'FBP' or 'inconsistencyReconstruction'
        set_optimal (bool): if true, sets the given parameter to the value that optimizes the metric
        
    Returns:
        stack of single-slice reconstructions (i.e., 3D numpy array or torch tensor) for all parameter values
        if set_optimal is True, then also returns the value of the metric at the optimal value
    """
    
    if param == 'tiltAngle':
        param = 'tilt'
    values = np.array(values)
    values = np.unique(values)
    
    if leapct.ct_geometry_defined() == False or leapct.ct_volume_defined() == False:
        print('Error: CT geometry and CT volume parameters must be set before running this function!')
        return None
    valid_params = ['centerCol', 'centerRow', 'tau', 'sod', 'sdd', 'tilt', 'vertical_shift', 'horizontal_shift']
    if param == None:
        param = 'centerCol'
    if any(name in param for name in valid_params) == False:
        print('Error: Invalid parameter, must be one of: ' + str(valid_params))
        return None
    if iz is None:
        iz = np.argmin(np.abs(leapct.z_samples()))
        #iz = int(leapct.get_numZ()//2)
    if iz < 0 or iz >= leapct.get_numZ():
        print('Error: Slice index is out of bounds for current volume specification.')
        return None
    if param == 'tilt' and leapct.get_geometry() == 'FAN':
        print('Error: Detector tilt can cannot be applied to fan-beam data.')
        return None
    if param == 'tau' and leapct.get_geometry() == 'PARALLEL':
        print('Error: tau does not apply to parallel-beam data.')
        return None
    if leapct.get_geometry() == 'MODULAR':
        if param == 'tau' or param == 'centerCol' or param == 'centerRow':
            print('Error: centerCol, centerRow, and tau do not apply to modular-beam data.')
            return None
        
    g_sweep = g
    #leapct_sweep = tomographicModels()
    leapct_sweep.copy_parameters(leapct)
    if set_optimal == True:
        leapct_sweep.set_offsetScan(False)
        dFOV = leapct_sweep.get_diameterFOV_min()
        
        if param == 'centerCol':
            leapct_sweep.set_centerCol(values[0])
            dFOV = min(dFOV, leapct_sweep.get_diameterFOV_min())
            leapct_sweep.set_centerCol(values[-1])
            dFOV = min(dFOV, leapct_sweep.get_diameterFOV_min())
            leapct_sweep.set_centerCol(leapct.get_centerCol())
        elif param == 'tau':
            leapct_sweep.set_tau(values[0])
            dFOV = min(dFOV, leapct_sweep.get_diameterFOV_min())
            leapct_sweep.set_tau(values[-1])
            dFOV = min(dFOV, leapct_sweep.get_diameterFOV_min())
            leapct_sweep.set_tau(leapct.get_tau())
        dFOV -= leapct.get_voxelWidth()
        
        leapct_sweep.set_diameterFOV(dFOV)
        leapct_sweep.set_offsetX(0.0)
        leapct_sweep.set_offsetY(0.0)
        leapct_sweep.set_numX(int(dFOV/leapct_sweep.get_voxelWidth()))
        leapct_sweep.set_numY(int(dFOV/leapct_sweep.get_voxelWidth()))

    if param == 'tilt':
        if leapct.get_geometry() != 'CONE':
            leapct_sweep.convert_to_modularbeam()
        if leapct.get_geometry() == 'CONE':
            leapct_sweep.set_tiltAngle(0.0)
    elif leapct_sweep.get_geometry() == 'FAN' or leapct_sweep.get_geometry() == 'PARALLEL':
        leapct_sweep.set_numRows(1)
        if has_torch == True and type(g) is torch.Tensor:
            g_sweep = torch.zeros((leapct.get_numAngles(), 1, leapct.get_numCols()), dtype=torch.float32)
            g_sweep = g_sweep.to(g.get_device())
            g_sweep[:,0,:] = g[:,iz,:]
        else:
            g_sweep = np.zeros((leapct.get_numAngles(), 1, leapct.get_numCols()), dtype=np.float32)
            g_sweep[:,0,:] = g[:,iz,:]
    
    z = leapct_sweep.z_samples()
    leapct_sweep.set_numZ(1)
    offsetZ = z[iz]
    leapct_sweep.set_offsetZ(offsetZ)
    
    if (param == 'tau' or param == 'centerCol') and (leapct.get_geometry() == 'CONE' or leapct.get_geometry() == 'MODULAR' or leapct.get_geometry() == 'CONE_PARALLEL'):
        rowRange = leapct_sweep.rowRangeNeededForBackprojection(0)
        if 0 < rowRange[0] or rowRange[1] < leapct_sweep.get_numRows()-1:
            g_sweep = leapct_sweep.cropProjections(rowRange, None, g_sweep)
        else:
            g_sweep = g_sweep.copy()

        """ This improves speed, but not sure it is a good idea
        if algorithmName == 'inconsistency':
            leapct_sweep.filterProjections(g_sweep, g_sweep, True)
        else:
            leapct_sweep.filterProjections(g_sweep, g_sweep, False)
        isFiltered = True
        #"""
    
    if has_torch == True and type(g) is torch.Tensor:
        f_stack = torch.zeros((len(values), leapct_sweep.get_numY(), leapct_sweep.get_numX()), dtype=torch.float32)
        f_stack = f_stack.to(f.get_device())
    else:
        f_stack = np.zeros((len(values), leapct_sweep.get_numY(), leapct_sweep.get_numX()), dtype=np.float32)
    f = leapct_sweep.allocate_volume()
    
    metrics = np.zeros(len(values))
    last_value = 0.0
    for n in range(len(values)):
        print(str(n) + ': ' + str(param) + ' = ' + str(values[n]))
        if param == 'centerCol':
            #col_shift = (leapct_sweep.get_centerCol() - values[n])*leapct_sweep.get_pixelWidth()
            #leapct_sweep.shift_detector(0.0, col_shift)
            leapct_sweep.set_centerCol(values[n])
        elif param == 'centerRow':
            if leapct_sweep.get_geometry() == 'CONE' or leapct_sweep.get_geometry() == 'CONE-PARALLEL':
                z_shift = (leapct_sweep.get_centerRow() - values[n])*leapct_sweep.get_pixelHeight()*leapct_sweep.get_sod()/leapct_sweep.get_sdd()
                leapct_sweep.set_offsetZ(leapct_sweep.get_offsetZ() + z_shift)
            #row_shift = (leapct_sweep.get_centerRow() - values[n])*leapct_sweep.get_pixelHeight()
            #leapct_sweep.shift_detector(row_shift, 0.0)
            leapct_sweep.set_centerRow(values[n])
        elif param == 'tau':
            delta_tau = values[n] - leapct_sweep.get_tau()
            #nativeVoxelSize = leapct_sweep.get_pixelWidth() * leapct_sweep.get_sod() / leapct_sweep.get_sdd()
            #new_centerCol = leapct_sweep.get_centerCol()-delta_tau/nativeVoxelSize
            #print('    and centerCol = ' + str(new_centerCol))
            #leapct_sweep.set_centerCol(new_centerCol)
            leapct_sweep.set_tau(values[n])
        elif param == 'sod':
            leapct_sweep.set_sod(values[n])
        elif param == 'sdd':
            leapct_sweep.set_sdd(values[n])
        elif param == 'vertical_shift':
            leapct_sweep.shift_detector(values[n]-last_value, 0.0)
        elif param == 'horizontal_shift':
            leapct_sweep.shift_detector(0.0, values[n]-last_value)
        elif param == 'tilt':
            if leapct_sweep.get_geometry() == 'CONE':
                leapct_sweep.set_tiltAngle(values[n])
            else:
                leapct_sweep.rotate_detector(values[n]-last_value)
        
        if algorithmName == 'inconsistencyReconstruction' or algorithmName == 'inconsistency':
            if isFiltered:
                leapct_sweep.weightedBackproject(g_sweep, f)
            else:
                leapct_sweep.inconsistencyReconstruction(g_sweep, f)
            metrics[n] = leapct_sweep.sum(f**2)
            print('   inconsistency metric: ' + str(metrics[n]))
        else:
            if isFiltered:
                leapct_sweep.weightedBackproject(g_sweep, f)
            else:
                leapct_sweep.FBP(g_sweep, f)
            metrics[n] = entropy(f)
            print('   entropy metric: ' + str(metrics[n]))
                
        f_stack[n,:,:] = f[0,:,:]
        last_value = values[n]
    
    if set_optimal:
        if algorithmName ==' FBP':
            ind_best = np.argmax(metrics)
        else:
            ind_best = np.argmin(metrics)
        optimal_value = values[ind_best]
        metric_value = metrics[ind_best]
        
        if 0 < ind_best and ind_best < metrics.size-1:
            optimal_value, metric_value = quadratic_extrema(values[ind_best-1], values[ind_best], values[ind_best+1], metrics[ind_best-1], metrics[ind_best], metrics[ind_best+1])
                   
        if param == 'centerCol':
            leapct.set_centerCol(optimal_value)
        elif param == 'centerRow':
            leapct.set_centerRow(optimal_value)
        elif param == 'tau':
            leapct.set_tau(optimal_value)
        elif param == 'sod':
            leapct.set_sod(optimal_value)
        elif param == 'sdd':
            leapct.set_sdd(optimal_value)
        elif param == 'vertical_shift':
            leapct.shift_detector(optimal_value, 0.0)
        elif param == 'horizontal_shift':
            leapct.shift_detector(0.0, optimal_value)
        elif param == 'tilt':
            if leapct.get_geometry() == 'CONE':
                leapct.set_tiltAngle(optimal_value)
            elif leapct.get_geometry() == 'MODULAR':
                leapct.rotate_detector(optimal_value-last_value)
        return f_stack, metric_value
    else:
        return f_stack
    
def entropy(x):
    marg = np.histogramdd(np.ravel(x), bins = int(np.sqrt(x.size)))[0]/x.size
    marg = list(filter(lambda p: p > 0, np.ravel(marg)))
    return -np.sum(np.multiply(marg, np.log2(marg)))
    
def MTF(leapct, f, r, center=None, getEdgeResponse=False, oversamplingRate=4):
    if leapct.ct_volume_defined() == False:
        return None
    x = leapct.x_samples()
    y = leapct.y_samples()
    z = leapct.z_samples()

    if center is None:
        x_c = 0.0
        y_c = 0.0
        z_c = 0.0
    else:
        x_c = center[0]
        y_c = center[1]
        z_c = center[2]
    
    iz = np.argmin(np.abs(z-z_c))
    iz = max(0, min(leapct.get_numZ()-1, iz))
    f_slice = np.squeeze(f[iz,:,:])
    y,x = np.meshgrid(y,x, indexing='ij')
    dist = np.sqrt((x-x_c)**2 + (y-y_c)**2)
    ind = dist < 2.0*r
    mus = f_slice[ind]
    distanceFromCenter = dist[ind]
    
    M = oversamplingRate # the oversampling factor
    T_samples = leapct.get_voxelWidth() / float(M)
    N_samples = 2*M * int(r / (2*M * T_samples)) # N*T/2 <= r
    r_0 = r - float(N_samples) / 2.0 * T_samples
    
    #float r_min = (float(i) - 0.5) * T_samples + r_0;
    #float r_max = r_min + T_samples;
    radial_bins = np.array(range(N_samples))*T_samples + r_0 - 0.5*T_samples

    mus_equispaced = np.zeros(N_samples)
    for i in range(N_samples-1, -1, -1):
        r_min = (float(i) - 0.5) * T_samples + r_0
        r_max = r_min + T_samples
        
        ind = np.logical_and(r_min < distanceFromCenter , distanceFromCenter <= r_max)
        if np.any(ind):
            mus_equispaced[i] = np.mean(mus[ind])
        else:
            if i + 1 < N_samples:
                mus_equispaced[i] = mus_equispaced[i + 1]
            else:
                mus_equispaced[i] = 0.0

    if getEdgeResponse:
        return mus_equispaced

    #LSF[i] = -1.0 * (mus_equispaced[i + 1] - mus_equispaced[i - 1]) / (2.0 * T_samples);
    LSF = (np.roll(mus_equispaced, -1) - np.roll(mus_equispaced, 1)) / (2.0 * T_samples)
    LSF[0] = 0.0
    LSF[-1] = 0.0
    
    abs_FFT_LSF = np.abs(np.fft.fft(LSF))
    DCvalue = abs_FFT_LSF[0]
    abs_FFT_LSF = abs_FFT_LSF[0:N_samples//2]
    
    #import matplotlib.pyplot as plt
    #plt.plot(abs_FFT_LSF, 'k-*')
    #plt.show()
    
    omega = np.array(range(N_samples//2)) * 2.0 / float(N_samples)
    MTF = abs_FFT_LSF / DCvalue / np.sinc(omega)
    MTF = MTF[0:N_samples//8]
    
    return MTF
    
    
class ball_phantom_calibration:
    def __init__(self, leapct, ballSpacing, g, segmentation_threshold=None):
        if leapct.ct_geometry_defined() == False:
            print('Must define initial guess of CT geometry!')
            return
        if g is None:
            print('Must define ball phantom scan data!')
            return
        if ballSpacing is None or ballSpacing <= 0.0:
            return('Must define spacing (mm) between balls')
            return
            
        self.T_u = leapct.get_pixelWidth()
        self.T_v = leapct.get_pixelHeight()
        self.numCols = leapct.get_numCols()
        self.numRows = leapct.get_numRows()
        
        us = self.T_u*(np.array(range(self.numCols)) - 0.5*(self.numCols-1.0))
        vs = self.T_v*(np.array(range(self.numRows)) - 0.5*(self.numRows-1.0))
        
        self.numAngles = g.shape[0]
        self.T_z = ballSpacing
        self.projectionAngles = leapct.get_angles()
        self.leapct = leapct
        
        numBalls = 0
        self.N_z = 0
        self.ball_projection_locations = None
        
        ### PERFORM CONNECTED COMPONENTS SEGMENTATION
        # ball_projection_locations is numAngles x numBalls x 2 numpy array
        # which gives the COM of each projection of each ball on the detector
        us, vs = np.meshgrid(us, vs)
        g_labeled = self.connected_components(g, segmentation_threshold)
        if g_labeled is None:
            return
        numBalls = np.max(g_labeled)
        self.N_z = numBalls
        self.ball_projection_locations = np.zeros((g.shape[0], numBalls, 2), dtype=np.float32)
        for i in range(g.shape[0]):
            aProj = np.squeeze(g[i,:,:])
            aProj_labeled = np.squeeze(g_labeled[i,:,:])
            for k in range(numBalls):
                ind = aProj_labeled == k+1
                self.ball_projection_locations[i,k,0] = np.sum(us[ind]*aProj[ind]) / np.sum(aProj[ind])
                self.ball_projection_locations[i,k,1] = np.sum(vs[ind]*aProj[ind]) / np.sum(aProj[ind])
        del g_labeled
        
    def initial_guess(self, g=None):
        if self.N_z <= 0:
            return None
        sod = self.leapct.get_sod()
        sdd = self.leapct.get_sdd()
        odd = sdd - sod
        
        if self.N_z >= 3:
            spread = np.zeros(self.N_z, dtype=np.float32)
            for k in range(self.N_z):
                spread[k] = np.max(self.ball_projection_locations[:, k, 1]) - np.min(self.ball_projection_locations[:, k, 1])
            k_arg = np.argmin(spread)
            v_c = np.mean(self.ball_projection_locations[:, k_arg, 1])
            #v_c = self.T_v * (x[3] - 0.5*(self.numRows-1.0))
            centerRow = v_c / self.T_v + 0.5*(self.numRows-1.0)
            self.leapct.set_centerRow(centerRow)
        
        r = 0.5*(np.max(self.ball_projection_locations[:, :, 0]) - np.min(self.ball_projection_locations[:, :, 0])) * sod / sdd
        phase = 0.0
        
        #mean_z = 0.5*T_z*(N_z-1) + z_0
        #mean_z = np.mean(self.ball_projection_locations[:, :, 1])
        #z_0 = 0.0*mean_z - 0.5*self.T_z*(self.N_z-1)
        z_0 = -self.T_z*0.5*(self.N_z-1.0)
        
        if g is not None:
            g_sum = np.sum(g,axis=1)
            g_copy = g.copy()
            g_copy[:,:,:] = g_sum[:,None,:]
            self.leapct.find_centerCol(g_copy)
            del g_copy
            del g_sum
        x = [self.leapct.get_centerRow(), self.leapct.get_centerCol(), self.leapct.get_sod(), self.leapct.get_sdd()-self.leapct.get_sod(), 0.0, 0.0, 0.0, r, z_0, 0.0]
        return x
    
    def estimate_locations(self, x):
        if self.N_z <= 0:
            return None
        deg_to_rad = np.pi/180.0
        #x = [centerRow, centerCol, sod, odd, psi, theta, phi, r, z_0, phase]
        
        v_c = self.T_v * (x[0] - 0.5*(self.numRows-1.0))
        u_c = self.T_u * (x[1] - 0.5*(self.numCols-1.0))
        
        sod = x[2]
        odd = x[3]
        psi = x[4]
        theta = x[5]
        phi = x[6]
        
        #psi = np.clip(psi, -5.0, 5.0)
        #theta = np.clip(theta, -5.0, 5.0)
        #phi = np.clip(phi, -5.0, 5.0)
        
        A = R.from_euler('xyz', [psi, theta, phi], degrees=True).as_matrix()
        detNormal = A[:,1]
        u_vec = A[:,0]
        v_vec = A[:,2]
        
        sourcePos = np.array([0.0, sod, 0.0])
        detectorCenter = np.array([u_c, -odd, v_c])
        c_minus_s_dot_n = np.dot(detectorCenter - sourcePos, detNormal)
        
        r = x[7]
        z_0 = x[8]
        phase = x[9]
        
        retVal = np.zeros((self.numAngles, self.N_z, 2), dtype=np.float32)
        for k in range(self.N_z):
            u_est = np.zeros(self.numAngles)
            v_est = np.zeros(self.numAngles)
            for i in range(self.numAngles):
                ball = np.array([r*np.cos((self.projectionAngles[i]-phase)*deg_to_rad), r*np.sin((self.projectionAngles[i]-phase)*deg_to_rad), self.T_z*k+z_0])
                traj = ball - sourcePos
                
                t = c_minus_s_dot_n / np.dot(traj, detNormal)
                
                hitPosition = sourcePos + t * traj
                
                u_coord = np.dot(hitPosition, u_vec) + u_c
                v_coord = np.dot(hitPosition, v_vec) + v_c
                
                retVal[i,k,0] = u_coord
                retVal[i,k,1] = v_coord
        
        return retVal
        
    def do_plot(self, x):
        if self.N_z <= 0:
            return
        estimated = self.estimate_locations(x)
        plt.plot(self.ball_projection_locations[:, :, 0], self.ball_projection_locations[:, :, 1], 'ko')
        plt.plot(estimated[:,:,0], estimated[:,:,1], 'ro')
        plt.title('Ball Center Locations (black from data, red are estimated)')
        plt.xlabel('detector column dimension (mm)')
        plt.ylabel('detector row dimension (mm)')
        plt.show()
        
    def residuals(self, x):
        if self.N_z <= 0:
            return 0.0
        estimated = self.estimate_locations(x)
        return (estimated - self.ball_projection_locations).flatten()
        
    def cost(self, x):
        if self.N_z <= 0:
            return 0.0
        estimated = self.estimate_locations(x)
        return np.sum((estimated - self.ball_projection_locations)**2)
        
    def optimize(self, x=None, set_parameters=False):
        if self.N_z <= 0:
            return None
        if x is None:
            x = self.initial_guess()
        res = least_squares(self.residuals, x)
        if set_parameters:
            self.leapct.set_centerRow(res.x[0])
            self.leapct.set_centerCol(res.x[1])
            self.leapct.set_sod(np.abs(res.x[2]))
            self.leapct.set_sdd(np.abs(res.x[2]+res.x[3]))
            """
            if np.abs(res.x[4]) < 0.1:
                res.x[4] = 0.0
            if np.abs(res.x[6]) < 0.1:
                res.x[6] = 0.0
            if np.abs(res.x[5]) < 0.05:
                res.x[5] = 0.0
            #"""
            if res.x[4] != 0.0 or res.x[5] != 0.0 or res.x[6] != 0.0:
                A = R.from_euler('xyz', [res.x[4], res.x[5], res.x[6]], degrees=True).as_matrix()
                #self.leapct.convert_to_modularbeam()
                #self.leapct.rotate_detector(-res.x[5])
                #self.leapct.rotate_detector(A.T)
                self.leapct.set_tiltAngle(np.clip(res.x[4], -5.0, 5.0))
        return res

    def connected_components(self, g, threshold=None, FWHM=0.0, connectivity=2):
        if threshold is None:
            threshold = np.max(g)/4.0
        g_labeled = np.zeros((g.shape[0], g.shape[1], g.shape[2]), dtype=np.int8)
        import skimage as ski
        for i in range(g.shape[0]):
            I = np.squeeze(g[i,:,:])
            if FWHM > 1.0:
                self.BlurFilter2D(I, FWHM)
            binary_mask = I > threshold
            
            # perform connected component analysis, connectivity=2 includes diagonals
            labeled_image, count = ski.measure.label(binary_mask, connectivity=connectivity, return_num=True)
            
            # Filter out small objects
            count = 0
            min_size = 10  # Minimum desired object size (in pixels)
            filtered_image = np.zeros_like(labeled_image)
            for region in ski.measure.regionprops(labeled_image):
                if region.area >= min_size:
                    count += 1
                    filtered_image[labeled_image == region.label] = count
            labeled_image = filtered_image
            
            if i > 0:
                if count != count_last:
                    print('Error: inconsistent number of balls found across projections!')
                    print('Try running connected_components segmentation with different parameters or inspect your data.')
                    print(count)
                    print(count_last)
                    return None
            if count == 0:
                print('Error: no balls found!')
                print('Try running connected_components segmentation with different parameters or inspect your data.')
                return None
            g_labeled[i,:,:] = labeled_image[:,:]
            count_last = count
        print('found ' + str(count) + ' balls')
        return g_labeled
