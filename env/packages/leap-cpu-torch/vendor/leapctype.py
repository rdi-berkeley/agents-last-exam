################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
# ctype tomographicModels class
################################################################################

import ctypes
import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import site
import glob
import imageio
from sys import platform as _platform
from numpy.ctypeslib import ndpointer
import numpy as np
try:
    import torch
    has_torch = True
except:
    has_torch = False
from leap_filter_sequence import *
#testFS = filterSequence()

class tomographicModels:
    """ Python class for tomographicModels bindings
    Usage Example:
    from leapctype import *
    leapct = tomographicModels()
    leapct.set_conebeam(...)
    leapct.set_default_volume(...)
    ...
    leapct.project(g,f)
    """

    def __init__(self, param_id=None, lib_dir=""):
        """Constructor

        The functions in this class can take as input and output data that is either on the CPU or the GPU.
        Note that all input and ouput data for all functions must lie either on a specific GPU or on the CPU.
        You cannot have some data on the CPU and some on a GPU.
        
        If the data is on the CPU (works for both numpy arrays or pytorch tensors):
            1) and one wishes the computations to take place on the CPU, then run the following
               command: set_gpu(-1).
            2) and one wishes the computations to take place on one of more GPUs, then run
               the following command: set_gpus(list_of_gpu_indices), e.g., set_gpus([0,1])
               The default setting is for LEAP to use all GPUs, so if this is what you want
               there is no need to run the set_gpus function
        If the data is on the GPU (only possible with torch tensors):
            then the computations must also take place on this particular GPU and
            you must run the following command: set_gpu(index), where index is GPU index
            of where the data resides.
            
        Providing data that is already on a GPU is best for small data sets, where data
        transfers to and from the GPU are more time consuming than the computations
        for the actual data processing.
        
        Providing data that is on the CPU is best for medium to large data sets because this
        allows for multi-GPU processing and LEAP will automatically divide up the data
        into small enough chunks so that is does not exceed the GPU memory.
        
        The LEAP library has the ability to generate and track several parameter sets.
        By default every new object instance will create a new parameter set.
        Otherwise one can use the param_id parameter so that the object will utilize a parameter set
        that is also shared by another object of this class.  But this parameter index must already be in use.
        Do not use this argument if you want to create a NEW parameter set. See the param_id argument description below.
        
        Args:
            param_id (int): If no value is given, then a new parameter set is generated, otherwise one can specify a parameter set index to use, but this parameter index must already be in use
            lib_dir (string): Path to the LEAP dynamic library, default value is the same path as this file
        
        """
        if len(lib_dir) > 0:
            current_dir = lib_dir
        else:
            current_dir = os.path.abspath(os.path.dirname(__file__))

        if _platform == "linux" or _platform == "linux2":
            import readline
            from ctypes import cdll

            #libdir = site.getsitepackages()[0]
            #libname = glob.glob(os.path.join(libdir, "leapct*.so"))
            libname = glob.glob(os.path.join(current_dir, "*leapct*.so"))
            if len(libname) == 0:
                fullPath = os.path.join(current_dir, 'libleapct.so')
                fullPath_backup = os.path.join(current_dir, '../build/lib/libleapct.so')
            elif len(libname) == 1:
                fullPath = libname[0]
                fullPath_backup = ""
            elif len(libname) >= 2:
                fullPath = libname[0]
                fullPath_backup = libname[1]
            
            if os.path.isfile(fullPath):
                self.libprojectors = cdll.LoadLibrary(fullPath)
            elif os.path.isfile(fullPath_backup):
                self.libprojectors = cdll.LoadLibrary(fullPath_backup)
            else:
                print('Error: could not find LEAP dynamic library at')
                print(fullPath)
                print('or')
                print(fullPath_backup)
                self.libprojectors = None
            
        elif _platform == "win32":
            from ctypes import windll
        
            libname = glob.glob(os.path.join(current_dir, "*leapct*.dll"))
            if len(libname) == 0:
                fullPath = os.path.join(current_dir, 'libleapct.dll')
                fullPath_backup = os.path.join(current_dir, r'..\win_build\bin\Release\libleapct.dll')
            elif len(libname) == 1:
                fullPath = libname[0]
                fullPath_backup = ""
            elif len(libname) >= 2:
                fullPath = libname[0]
                fullPath_backup = libname[1]
        
            if os.path.isfile(fullPath):
                try:
                    self.libprojectors = windll.LoadLibrary(fullPath)
                except:
                    self.libprojectors = ctypes.CDLL(fullPath, winmode=0)
            elif os.path.isfile(fullPath_backup):
                try:
                    self.libprojectors = windll.LoadLibrary(fullPath_backup)
                except:
                    self.libprojectors = ctypes.CDLL(fullPath_backup, winmode=0)
            else:
                print('Error: could not find LEAP dynamic library at')
                print(fullPath)
                print('or')
                print(fullPath_backup)
                self.libprojectors = None
        
        elif _platform == "darwin":  # Darwin is the name for MacOS in Python's platform module
            from ctypes import cdll
            
            libname = glob.glob(os.path.join(current_dir, "*leapct*.dylib"))
            if len(libname) == 0:
                fullPath = os.path.join(current_dir, 'libleapct.dylib')
                fullPath_backup = os.path.join(current_dir, '../build/lib/libleapct.dylib')
            elif len(libname) == 1:
                fullPath = libname[0]
                fullPath_backup = ""
            elif len(libname) >= 2:
                fullPath = libname[0]
                fullPath_backup = libname[1]
            
            if os.path.isfile(fullPath):
                self.libprojectors = cdll.LoadLibrary(fullPath)
            elif os.path.isfile(fullPath_backup):
                self.libprojectors = cdll.LoadLibrary(fullPath_backup)
            else:
                print('Error: could not find LEAP dynamic library at')
                print(fullPath)
                print('or')
                print(fullPath_backup)
                self.libprojectors = None
        
        if self.libprojectors is None:
            self.param_id = -1
        else:
            if param_id is not None:
                self.param_id = param_id
            else:
                self.param_id = self.create_new_model()
            self.set_model()
        self.print_cost = False
        self.print_warnings = True
        self.volume_mask = None
        
        self.file_dtype = np.float32
        self.wmin = 0.0
        self.wmax = None

    def test_script(self):
        self.libprojectors.test_script()

    def set_model(self, i=None):
        """ This should be considered a private class function """
        self.libprojectors.set_model.restype = ctypes.c_bool
        self.libprojectors.set_model.argtypes = [ctypes.c_int]
        if i is None:
            return self.libprojectors.set_model(self.param_id)
        else:
            return self.libprojectors.set_model(i)
            
    def set_log_error(self):
        """Sets logging level to logERROR
        
        This logging level prints out the fewest statements (only error statements)
        """
        self.libprojectors.set_log_error()
        self.print_cost = False
        self.print_warnings = False
        
    def set_log_warning(self):
        """Sets logging level to logWARNING
        
        This logging level prints out the second fewest statements (only error and warning statements)
        and is the default setting.  It includes iterative reconstruction warnings and iteration number.
        """
        self.libprojectors.set_log_warning()
        self.print_cost = False
        self.print_warnings = True
            
    def set_log_status(self):
        """Sets logging level to logSTATUS
        
        This logging level prints out the second most statements, including iterative reconstruction
        cost at every iteration (these extra computations will slow down processing)
        """
        self.libprojectors.set_log_status()
        self.print_cost = True
        self.print_warnings = True
        
    def set_log_debug(self):
        """Sets logging level to logDEBUG
        
        This logging level prints out the most statements
        """
        self.libprojectors.set_log_debug()
        self.print_cost = True
        self.print_warnings = True
        
    def set_fileIO_parameters(self, dtype=np.float32, wmin=0.0, wmax=None):
        r""" This function sets parameters dealing with how tiff stacks are saved
        
        If dtype is np.float32, the data is not clipped
        
        Args:
            dtype: the data type to use, can be: np.float32, np.uint8, or np.uint16
            wmin (float): the low value for clipping the data (default is 0.0)
            wmax (float): the high value for clipping the data (default is the max of the 3D data)
        
        Returns:
            True if the dtype is valid, False otherwise
        
        """
        if dtype == np.float32 or dtype == np.uint8 or dtype == np.uint16:
            self.file_dtype = dtype
            self.wmin = wmin
            self.wmax = wmax
        else:
            print('Error: invalid dtype; must be np.float32, np.uint8, or np.uint16')
            return False
    
    def set_maxSlicesForChunking(self, N):
        """This function effects how forward and backprojection jobs are divided into multiple processing jobs on the GPU

        Smaller numbers use less GPU memory, but may slow down processing.  Only use this function if you know what you are doing.
        
        For forward projection it specifies the maximum number of detector rows used per job.
        For backprojection it specifies the maximum number of CT volume z-slices used per job.
        
        Args:
            N (int): the chunk size
        """
        self.libprojectors.set_maxSlicesForChunking.restype = ctypes.c_bool
        self.libprojectors.set_maxSlicesForChunking.argtypes = [ctypes.c_int]
        self.set_model()
        return self.libprojectors.set_maxSlicesForChunking(N)

    def create_new_model(self):
        self.libprojectors.create_new_model.restype = ctypes.c_int
        return self.libprojectors.create_new_model()
        
    def copy_parameters(self, leapct):
        """Copies the parameters from another instance of this class"""
        
        self.print_cost = leapct.print_cost
        self.print_warnings = leapct.print_warnings
        self.volume_mask = leapct.volume_mask
        
        self.set_model()
        self.libprojectors.copy_parameters.restype = ctypes.c_bool
        self.libprojectors.copy_parameters.argtypes = [ctypes.c_int]
        return self.libprojectors.copy_parameters(leapct.param_id)

    def reset(self):
        """reset
        Resets and clears all parameters
        """
        self.set_model()
        return self.libprojectors.reset()
        
    def include_cufft(self):
        """Returns True if LEAP is using CUFFT, False otherwise"""
        self.libprojectors.include_cufft.restype = ctypes.c_bool
        return self.libprojectors.include_cufft()

    def about(self):
        """prints info about LEAP, including the version number"""
        self.set_model()
        self.libprojectors.about()
        
    def version(self):
        """Returns version number string"""
        try:
            versionText = ctypes.create_string_buffer(16)
            self.libprojectors.version(versionText)
            if sys.version_info[0] == 3:
                return versionText.value.decode("utf-8")
            else:
                return versionText.value
        except:
            return "unknown"

    def printParameters(self):
        """printParameters
        prints all CT geometry and CT volume parameters to the screen
        """
        return self.print_parameters()

    def print_param(self):
        """printParameters
        prints all CT geometry and CT volume parameters to the screen
        """
        return self.print_parameters()

    def print_parameters(self):
        """print_parameters
        prints all CT geometry and CT volume parameters to the screen
        """
        self.libprojectors.print_parameters.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.print_parameters()
        
    def all_defined(self):
        """Returns True if all CT geometry and CT volume parameters are defined, False otherwise"""
        self.set_model()
        self.libprojectors.all_defined.restype = ctypes.c_bool
        return self.libprojectors.all_defined()
    
    def ct_geometry_defined(self):
        """Returns True if all CT geometry parameters are defined, False otherwise"""
        self.set_model()
        self.libprojectors.ct_geometry_defined.restype = ctypes.c_bool
        return self.libprojectors.ct_geometry_defined()
        
    def ct_volume_defined(self):
        """Returns True if all CT volume parameters are defined, False otherwise"""
        self.set_model()
        self.libprojectors.ct_volume_defined.restype = ctypes.c_bool
        return self.libprojectors.ct_volume_defined()
        
    def verify_inputs(self, g, f):
        """ Verifies that the projection data (g) and the volume data (f) are compatible with the specified parameters """
        #if f is None:
        #    f = g
    
        # check they are the same type
        if type(g) != type(f):
            print('Error: projection and volume data must be the same type')
            return False
            
        # check they are numpy array or torch.tensor
        if has_torch:
            if type(g) is not np.ndarray and type(g) is not torch.Tensor:
                print('Error: projection and volume data must be either numpy arrays or torch tensors')
                return False
                
            if type(g) is torch.Tensor:
                if g.is_cuda != f.is_cuda:
                    print('Error: projection and volume data must either both be on the CPU or both be on the GPU')
                    return False
                
        elif type(g) is not np.ndarray:
            print('Error: projection and volume data must be either numpy arrays or torch tensors')
            return False
            
        # check they are float32
        if has_torch and type(g) is torch.Tensor:
            if g.dtype != torch.float32 or f.dtype != torch.float32:
                print('Error: projection and volume data must be float32 data type')
                return False
            if g.is_contiguous() == False or f.is_contiguous() == False:
                print('Error: projection and volume data must be contiguous')
                return False
        else:
            if g.dtype != np.float32 or f.dtype != np.float32:
                print('Error: projection and volume data must be float32 data type')
                return False
            if g.data.c_contiguous == False or f.data.c_contiguous == False:
                print('Error: projection and volume data must be contiguous')
                return False
        
        # check are they 3D arrays
        if len(g.shape) != 3 or len(f.shape) != 3:
            print('Error: projection and volume data must be 3D arrays')
            return False
        
        # check size
        self.libprojectors.verify_input_sizes.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.libprojectors.verify_input_sizes.restype = ctypes.c_bool
        self.set_model()
        retVal = self.libprojectors.verify_input_sizes(g.shape[0], g.shape[1], g.shape[2], f.shape[0], f.shape[1], f.shape[2])
        if retVal == False:
            print('Error: projection and/ or volume data shapes do not match specified LEAP settings')
        return retVal

    def optimalFFTsize(self, N):
        self.libprojectors.getOptimalFFTsize.argtypes = [ctypes.c_int]
        self.libprojectors.getOptimalFFTsize.restype = ctypes.c_int
        return self.libprojectors.getOptimalFFTsize(N)

    def extraColumnsForOffsetScan(self):
        """Get the number of extra columns that need to be padded in order to do an offset scan FBP reconstruction
        
        We don't recommend users to use this function.  It is just a utility function for the filterProjections function.
        
        """
        if self.get_offsetScan() == False:
            return 0
        else:
            self.set_model()
            self.libprojectors.extraColumnsForOffsetScan.restype = ctypes.c_int
            return self.libprojectors.extraColumnsForOffsetScan()

    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS SET THE CT SCANNER GEOMETRY PARAMETERS
    ###################################################################################################################
    ###################################################################################################################
    def set_coneparallel(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0, helicalPitch=0.0):
        r"""Sets the parameters for a cone-parallel CT geometry
        
        The origin of the coordinate system is always at the center of rotation.  The forward (P) and back (P*) projection operators are given by
        
        .. math::
           \begin{eqnarray*}
           Pf(s, \varphi, \nu) &=& \int_\mathbb{R} f\left(s\boldsymbol{\theta}^\perp(\varphi) + \sqrt{R^2-s^2}\boldsymbol{\theta}(\varphi) + \Delta\left(\varphi + \alpha(s) \right)\widehat{\boldsymbol{z}}  + \frac{l}{\sqrt{1+\nu^2}}\left[-\boldsymbol{\theta} + \nu\widehat{\boldsymbol{z}}\right]\right) \, dl \\
           P^*g(\boldsymbol{x}) &=& \int \frac{\sqrt{R^2 + \nu^2(\boldsymbol{x},\varphi)}}{\sqrt{R^2-(\boldsymbol{x}\cdot\boldsymbol{\theta}^\perp(\varphi))^2}  - \boldsymbol{x}\cdot\boldsymbol{\theta}(\varphi)} g\left(\boldsymbol{x}\cdot\boldsymbol{\theta}^\perp(\varphi), \varphi, \nu(\boldsymbol{x},\varphi) \right) \, d\varphi \\
           \nu(\boldsymbol{x},\varphi) &:=& \frac{x_3 - \Delta\left(\varphi + \alpha(\boldsymbol{x}\cdot\boldsymbol{\theta}^\perp(\varphi)) \right)}{\sqrt{R^2-(\boldsymbol{x}\cdot\boldsymbol{\theta}^\perp(\varphi))^2}  - \boldsymbol{x}\cdot\boldsymbol{\theta}(\varphi)} \\
           \alpha(s) &:=& \sin^{-1}\left(\frac{s}{R}\right) + \sin^{-1}\left(\frac{\tau}{R}\right)
           \end{eqnarray*}
           
        Here, we have used :math:`R` for sod, :math:`\tau` for tau, :math:`\Delta` for helicalPitch, and v = t/sdd.
        
        Args:
            numAngles (int): number of projection angles
            numRows (int): number of rows in the x-ray detector
            numCols (int): number of columns in the x-ray detector
            pixelHeight (float): the detector pixel pitch (i.e., pixel size) between detector rows, measured in mm
            pixelWidth (float): the detector pixel pitch (i.e., pixel size) between detector columns, measured in mm
            centerRow (float): the detector pixel row index for the ray that passes from the source, through the origin, and hits the detector
            centerCol (float): the detector pixel column index for the ray that passes from the source, through the origin, and hits the detector
            phis (float32 numpy array):  a numpy array for specifying the angles of each projection, measured in degrees
            sod (float): source to object distance, measured in mm; this can also be viewed as the source to center of rotation distance
            sdd (float): source to detector distance, measured in mm
            tau (float): center of rotation offset
            helicalPitch (float): the helical pitch (mm/radians)
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        
        self.libprojectors.set_coneparallel.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.libprojectors.set_coneparallel.restype = ctypes.c_bool
        if has_torch and type(phis) is torch.Tensor:
            phis = phis.cpu().detach().numpy()
        elif type(phis) is not np.ndarray:
            angularRange = float(phis)
            phis = self.setAngleArray(numAngles, angularRange)
            
        if phis.size != numAngles:
            print('Error: phis.size != numAngles')
            return False
            
        self.set_model()
        return self.libprojectors.set_coneparallel(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau, helicalPitch)
    
    def set_conebeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0, helicalPitch=0.0, tiltAngle=0.0):
        r"""Sets the parameters for a cone-beam CT geometry
        
        The origin of the coordinate system is always at the center of rotation.  The forward (P) and back (P*) projection operators are given by

        .. math::
           \begin{eqnarray*}
           Pf(u,\varphi,v) &:=& \int_\mathbb{R} f\left(R\boldsymbol{\theta}(\varphi) - \tau\boldsymbol{\theta}^\perp(\varphi) + \Delta\varphi\widehat{\boldsymbol{z}}  + \frac{l}{\sqrt{1+u^2+v^2}}\left[-\boldsymbol{\theta}(\varphi)+u\boldsymbol{\theta}^\perp(\varphi) + v\widehat{\boldsymbol{z}} \right] \right) \, dl \\
           P^*g(\boldsymbol{x}) &=& \int \frac{\sqrt{1+ u^2(\boldsymbol{x},\varphi) +v^2(\boldsymbol{x},\varphi)}}{(R-\boldsymbol{x}\cdot\boldsymbol{\theta}(\varphi))^2} g\left( u(\boldsymbol{x},\varphi), \varphi, v(\boldsymbol{x},\varphi)\right) \, d\varphi \\
           u(\boldsymbol{x},\varphi) &:=& \frac{\boldsymbol{x}\cdot \boldsymbol{\theta}^\perp(\varphi) + \tau}{R - \boldsymbol{x}\cdot\boldsymbol{\theta}(\varphi)} \\
           v(\boldsymbol{x},\varphi) &:=& \frac{x_3 - \Delta\varphi}{R - \boldsymbol{x}\cdot\boldsymbol{\theta}(\varphi)}
           \end{eqnarray*}

        for flat-panel cone-beam and
        
        .. math::
           \begin{eqnarray*}
           Pf(\alpha,\varphi,\nu) &=& \int_\mathbb{R} f\left(R\boldsymbol{\theta}(\varphi) - \tau\boldsymbol{\theta}^\perp(\varphi) + \Delta\varphi\widehat{\boldsymbol{z}} + \frac{l}{\sqrt{1+\nu^2}}\left[-\boldsymbol{\theta}(\varphi-\alpha) + \nu\widehat{\boldsymbol{z}} \right] \right) \, dl \\
           P^*g(\boldsymbol{x}) &=& \int \frac{\sqrt{1+\nu^2(\boldsymbol{x},\varphi)}}{\| R\boldsymbol{\theta}(\varphi) - \tau\boldsymbol{\theta}^\perp(\varphi) - \boldsymbol{x} \|^2}  g\left(\alpha(\boldsymbol{x},\varphi), \varphi, \nu(\boldsymbol{x},\varphi)\right) \, d\varphi \\
           \alpha(\boldsymbol{x},\varphi) &:=& \tan^{-1}\left( \frac{\boldsymbol{x}\cdot \boldsymbol{\theta}^\perp(\varphi) + \tau}{R - \boldsymbol{x}\cdot\boldsymbol{\theta}(\varphi)} \right) \\
           \nu(\boldsymbol{x},\varphi) &:=& \frac{x_3 - \Delta\varphi}{\|R\boldsymbol{\theta}(\varphi) - \tau\boldsymbol{\theta}^\perp(\varphi) - \boldsymbol{x} \|}
           \end{eqnarray*}
        
        for curved detector cone-beam data.  Here, we have used :math:`R` for sod, :math:`\tau` for tau, :math:`\Delta` for helicalPitch, u = s/sdd, and v = t/sdd.
        Note that :math:`u = \tan\alpha` and :math:`v = \nu \sqrt{1+u^2}`.
        
        To switch between flat and curved detectors, use the set_flatDetector() and set_curvedDetector() functions.  Flat detectors are the default setting.
        
        Args:
            numAngles (int): number of projection angles
            numRows (int): number of rows in the x-ray detector
            numCols (int): number of columns in the x-ray detector
            pixelHeight (float): the detector pixel pitch (i.e., pixel size) between detector rows, measured in mm
            pixelWidth (float): the detector pixel pitch (i.e., pixel size) between detector columns, measured in mm
            centerRow (float): the detector pixel row index for the ray that passes from the source, through the origin, and hits the detector
            centerCol (float): the detector pixel column index for the ray that passes from the source, through the origin, and hits the detector
            phis (float32 numpy array):  a numpy array for specifying the angles of each projection, measured in degrees
            sod (float): source to object distance, measured in mm; this can also be viewed as the source to center of rotation distance
            sdd (float): source to detector distance, measured in mm
            tau (float): center of rotation offset
            helicalPitch (float): the helical pitch (mm/radians)
            tiltAngle (float) the rotation of the detector around the optical axis (degrees)
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        self.libprojectors.set_conebeam.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.libprojectors.set_conebeam.restype = ctypes.c_bool
        if has_torch and type(phis) is torch.Tensor:
            phis = phis.cpu().detach().numpy()
        elif type(phis) is not np.ndarray:
            angularRange = float(phis)
            phis = self.setAngleArray(numAngles, angularRange)
            
        if phis.size != numAngles:
            print('Error: phis.size != numAngles')
            return False
            
        self.set_model()
        return self.libprojectors.set_conebeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau, tiltAngle, helicalPitch)
    
    def set_coneBeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0, helicalPitch=0.0, tiltAngle=0.0):
        """Alias for set_conebeam
        """
        return self.set_conebeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau, helicalPitch, tiltAngle)
    
    def set_fanbeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0):
        r"""Sets the parameters for a fan-beam CT geometry
        
        The origin of the coordinate system is always at the center of rotation.  The forward (P) and back (P*) projection operators are given by
        
        .. math::
           \begin{eqnarray*}
           Pf(u,\varphi,x_3) &:=& \int_\mathbb{R} f\left(R\boldsymbol{\theta}(\varphi) - \tau\boldsymbol{\theta}^\perp(\varphi) - \frac{l}{\sqrt{1+u^2}}\left[\boldsymbol{\theta}(\varphi) - u\boldsymbol{\theta}^\perp(\varphi) \right] + x_3\widehat{\boldsymbol{z}} \right) \, dl \\
           P^*g(\boldsymbol{x}) &=& \int \frac{1}{R-\boldsymbol{x}\cdot\boldsymbol{\theta}(\varphi)} \sqrt{1 + u^2(\boldsymbol{x},\varphi)} g\left( u(\boldsymbol{x},\varphi), \varphi, x_3\right) \, d\varphi \\
           u(\boldsymbol{x},\varphi) &:=& \frac{\boldsymbol{x}\cdot \boldsymbol{\theta}^\perp(\varphi) + \tau}{R - \boldsymbol{x}\cdot\boldsymbol{\theta}(\varphi)}
           \end{eqnarray*}
        
        Args:
            numAngles (int): number of projection angles
            numRows (int): number of rows in the x-ray detector
            numCols (int): number of columns in the x-ray detector
            pixelHeight (float): the detector pixel pitch (i.e., pixel size) between detector rows, measured in mm
            pixelWidth (float): the detector pixel pitch (i.e., pixel size) between detector columns, measured in mm
            centerRow (float): the detector pixel row index for the ray that passes from the source, through the origin, and hits the detector
            centerCol (float): the detector pixel column index for the ray that passes from the source, through the origin, and hits the detector
            phis (float32 numpy array):  a numpy array for specifying the angles of each projection, measured in degrees
            sod (float): source to object distance, measured in mm; this can also be viewed as the source to center of rotation distance
            sdd (float): source to detector distance, measured in mm
            tau (float): center of rotation offset
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        self.libprojectors.set_fanbeam.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.libprojectors.set_fanbeam.restype = ctypes.c_bool
        if has_torch and type(phis) is torch.Tensor:
            phis = phis.cpu().detach().numpy()
        elif type(phis) is not np.ndarray:
            angularRange = float(phis)
            phis = self.setAngleArray(numAngles, angularRange)
            
        if phis.size != numAngles:
            print('Error: phis.size != numAngles')
            return False
            
        self.set_model()
        return self.libprojectors.set_fanbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau)
        
    def set_fanBeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0):
        """Alias for set_fanbeam
        """
        return self.set_fanbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau)

    def set_parallelbeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis):
        r"""Sets the parameters for a parallel-beam CT geometry
        
        The origin of the coordinate system is always at the center of rotation.  The forward (P) and back (P*) projection operators are given by
        
        .. math::
           \begin{eqnarray*}
           Pf(s, \varphi, x_3) &:=& \int_\mathbb{R} f(s\boldsymbol{\theta}^\perp(\varphi) - l\boldsymbol{\theta}(\varphi) + x_3\widehat{\boldsymbol{z}}) \, dl \\
           P^*g(\boldsymbol{x}) &=& \int g(\boldsymbol{x}\cdot\boldsymbol{\theta}^\perp(\varphi), \varphi, x_3) \, d\varphi
           \end{eqnarray*}
        
        Args:
            numAngles (int): number of projection angles
            numRows (int): number of rows in the x-ray detector
            numCols (int): number of columns in the x-ray detector
            pixelHeight (float): the detector pixel pitch (i.e., pixel size) between detector rows, measured in mm
            pixelWidth (float): the detector pixel pitch (i.e., pixel size) between detector columns, measured in mm
            centerRow (float): the detector pixel row index for the ray that passes from the source, through the origin, and hits the detector
            centerCol (float): the detector pixel column index for the ray that passes from the source, through the origin, and hits the detector
            phis (float32 numpy array):  a numpy array for specifying the angles of each projection, measured in degrees
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        
        self.libprojectors.set_parallelbeam.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        self.libprojectors.set_parallelbeam.restype = ctypes.c_bool
        if has_torch and type(phis) is torch.Tensor:
            phis = phis.cpu().detach().numpy()
        elif type(phis) is not np.ndarray:
            angularRange = float(phis)
            phis = self.setAngleArray(numAngles, angularRange)
            
        if phis.size != numAngles:
            print('Error: phis.size != numAngles')
            return False
            
        self.set_model()
        return self.libprojectors.set_parallelbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis)

    def set_parallelBeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis):
        """Alias for set_parallelbeam
        """
        return self.set_parallelbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis)

    def set_modularbeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors):
        r"""Sets the parameters for a modular-beam CT geometry
        
        The origin of the coordinate system is always at the center of rotation.  The forward (P) and back (P*) projection operators are given by
        
        .. math::
           \begin{eqnarray*}
           Pf(s,t) &:=& \int_{\mathbb{R}} f\left( \boldsymbol{y} + \frac{l}{\sqrt{D^2 + s^2 + t^2}}\left[ \boldsymbol{c}-\boldsymbol{y} + s\boldsymbol{\widehat{u}} + t\boldsymbol{\widehat{v}} \right] \right) \, dl, \\
           P^*g(\boldsymbol{x}) &=& <\boldsymbol{c} - \boldsymbol{y}, \boldsymbol{n}> \frac{\| \boldsymbol{c} - \boldsymbol{y} + s\boldsymbol{u} + t\boldsymbol{v} \|}{<\boldsymbol{x} - \boldsymbol{y}, \boldsymbol{n}>^2} g\left(s(\boldsymbol{x}),t(\boldsymbol{x})\right)
           \end{eqnarray*}
           
        where
        
        .. math::
           \begin{eqnarray*}
           s(\boldsymbol{x}) &:=& \frac{<\boldsymbol{c}-\boldsymbol{y},\boldsymbol{n}>}{<\boldsymbol{x}-\boldsymbol{y},\boldsymbol{n}>}<\boldsymbol{x}-\boldsymbol{y},\boldsymbol{u}> - <\boldsymbol{c}-\boldsymbol{y},\boldsymbol{u}> \\
           t(\boldsymbol{x}) &:=& \frac{<\boldsymbol{c}-\boldsymbol{y},\boldsymbol{n}>}{<\boldsymbol{x}-\boldsymbol{y},\boldsymbol{n}>}<\boldsymbol{x}-\boldsymbol{y},\boldsymbol{v}> - <\boldsymbol{c}-\boldsymbol{y},\boldsymbol{v}>
           \end{eqnarray*}
       
        and :math:`\boldsymbol{y}` be a location of sourcePositions, :math:`\boldsymbol{c}` be a location of moduleCenters,
        :math:`\boldsymbol{\widehat{v}}` be a rowVectors instance, :math:`\boldsymbol{\widehat{u}}` be a colVectors instance,
        :math:`D := \|\boldsymbol{c}-\boldsymbol{y}\|`, and :math:`\boldsymbol{n} := \boldsymbol{\widehat{u}} \times \boldsymbol{\widehat{v}}`.
       
        Args:
            numAngles (int): number of projection angles
            numRows (int): number of rows in the x-ray detector
            numCols (int): number of columns in the x-ray detector
            pixelHeight (float): the detector pixel pitch (i.e., pixel size) between detector rows, measured in mm
            pixelWidth (float): the detector pixel pitch (i.e., pixel size) between detector columns, measured in mm
            sourcePositions ((numAngles X 3) numpy array): the (x,y,z) position of each x-ray source
            moduleCenters ((numAngles X 3) numpy array): the (x,y,z) position of the center of the front face of the detectors
            rowVectors ((numAngles X 3) numpy array):  the (x,y,z) unit vector pointing along the positive detector row direction
            colVectors ((numAngles X 3) numpy array):  the (x,y,z) unit vector pointing along the positive detector column direction
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        
        if sourcePositions.shape[0] != numAngles:
            print('Error: sourcePositions.shape[0] != numAngles')
            return False
        if moduleCenters.shape[0] != numAngles:
            print('Error: moduleCenters.shape[0] != numAngles')
            return False
        if rowVectors.shape[0] != numAngles:
            print('Error: rowVectors.shape[0] != numAngles')
            return False
        if colVectors.shape[0] != numAngles:
            print('Error: colVectors.shape[0] != numAngles')
            return False
            
        if sourcePositions.shape[1] != 3:
            print('Error: sourcePositions.shape[1] != 3')
            return False
        if moduleCenters.shape[1] != 3:
            print('Error: moduleCenters.shape[1] != 3')
            return False
        if rowVectors.shape[1] != 3:
            print('Error: rowVectors.shape[1] != 3')
            return False
        if colVectors.shape[1] != 3:
            print('Error: colVectors.shape[1] != 3')
            return False
        
        
        self.libprojectors.set_modularbeam.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
        self.libprojectors.set_modularbeam.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_modularbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors)
    
    def set_modularBeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors):
        """Alias for set_modularbeam
        """
        return self.set_modularbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors)
    
    def set_geometry(self, which):
        """Sets the CT geometry type parameter"""
        self.libprojectors.set_geometry.argtypes = [ctypes.c_int]
        self.libprojectors.set_geometry.restype = ctypes.c_bool
        self.set_model()
        if isinstance(which, int):
            return self.set_geometry(which)
        elif isinstance(which, str):
            if which == 'CONE':
                return self.libprojectors.set_geometry(0)
            elif which == 'PARALLEL':
                return self.libprojectors.set_geometry(1)
            elif which == 'FAN':
                return self.libprojectors.set_geometry(2)
            elif which == 'MODULAR':
                return self.libprojectors.set_geometry(3)
            elif which == 'CONE-PARALLEL':
                return self.libprojectors.set_geometry(4)
            else:
                return False
        else:
            return False
    
    def set_tau(self, tau):
        """Set the tau parameter
        
        Args:
            tau (float): center of rotation offset in mm (fan- and cone-beam data only)
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        self.libprojectors.set_tau.argtypes = [ctypes.c_float]
        self.libprojectors.set_tau.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_tau(tau)
        
    def set_tiltAngle(self, tiltAngle):
        """Set the tiltAngle parameter
        
        Args:
            tiltAngle (float): the rotation of the detector around the optical axis (degrees)
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        self.libprojectors.set_tiltAngle.argtypes = [ctypes.c_float]
        self.libprojectors.set_tiltAngle.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_tiltAngle(tiltAngle)
    
    def set_helicalPitch(self, helicalPitch):
        r"""Set the helicalPitch parameter
        
        This function sets the helicalPitch parameter which is measured in mm/radians.  Sometimes the helical pitch is specified in a normalized fashion.
        If so, please use the set_normalizedHelicalPitch function.
        
        If we denote the helical pitch by :math:`h` and the normalized helical pitch by :math:`\widehat{h}`, then they are related by
        
        .. math::
           \begin{eqnarray*}
           h = \frac{numRows * pixelHeight \frac{sod}{sdd}}{2\pi} \widehat{h}
           \end{eqnarray*}
        
        Args:
            helicalPitch (float): the helical pitch (mm/radians) (cone-beam and cone-parallel data only)
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        self.libprojectors.set_helicalPitch.argtypes = [ctypes.c_float]
        self.libprojectors.set_helicalPitch.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_helicalPitch(helicalPitch)
        
    def set_normalizedHelicalPitch(self, normalizedHelicalPitch):
        r"""Set the normalized helicalPitch parameter
        
        This function sets the helicalPitch parameter by specifying the normalized helical pitch value.
        
        If we denote the helical pitch by :math:`h` and the normalized helical pitch by :math:`\widehat{h}`, then they are related by
        
        .. math::
           \begin{eqnarray*}
           h = \frac{numRows * pixelHeight \frac{sod}{sdd}}{2\pi} \widehat{h}
           \end{eqnarray*}
        
        Args:
            normalizedHelicalPitch (float): the normalized helical pitch (unitless) (cone-beam and cone-parallel data only)
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        self.libprojectors.set_normalizedHelicalPitch.argtypes = [ctypes.c_float]
        self.libprojectors.set_normalizedHelicalPitch.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_normalizedHelicalPitch(normalizedHelicalPitch)
        
    def get_normalizedHelicalPitch(self):
        """Get the normalized helical pitch"""
        #self.libprojectors.get_normalizedHelicalPitch.argtypes = []
        self.libprojectors.get_normalizedHelicalPitch.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_normalizedHelicalPitch()
        
    def set_flatDetector(self):
        """Set the detectorType to FLAT"""
        self.set_model()
        self.libprojectors.set_flatDetector.restype = ctypes.c_bool
        return self.libprojectors.set_flatDetector()
        
    def set_curvedDetector(self):
        """Set the detectorType to CURVED (only for cone-beam data)"""
        self.set_model()
        self.libprojectors.set_curvedDetector.restype = ctypes.c_bool
        return self.libprojectors.set_curvedDetector()
        
    def get_detectorType(self):
        """Get the detectorType"""
        self.set_model()
        self.libprojectors.get_detectorType.restype = ctypes.c_int
        if self.libprojectors.get_detectorType() == 0:
            return 'FLAT'
        else:
            return 'CURVED'
            
    def set_centerCol(self, centerCol):
        """Set centerCol parameter"""
        self.set_model()
        self.libprojectors.set_centerCol.restype = ctypes.c_bool
        self.libprojectors.set_centerCol.argtypes = [ctypes.c_float]
        return self.libprojectors.set_centerCol(centerCol)
        
    def find_centerCol(self, g, iRow=-1, searchBounds=None):
        r"""Find the centerCol parameter

        This function works by minimizing the difference of conjugate rays, by changing the detector column sample locations. The cost functions
        for parallel-beam and fan-beam are given by
        
        .. math::
           \begin{eqnarray*}
           &&\int \int \left[g(s,\varphi) - g(-s,\varphi \pm \pi)\right]^2 \, ds \, d\varphi \\
           &&\int \int \left[g(u,\varphi) - g\left(\frac{-u+\frac{2\tau R}{R^2-\tau^2}}{1+u\left(\frac{2\tau R}{R^2-\tau^2}\right)},\varphi -2\tan^{-1}u + \tan^{-1}\left(\frac{2\tau R}{R^2-\tau^2}\right) \pm \pi\right)\right]^2 \, du \, d\varphi, \\
           \end{eqnarray*}
           
        respectively.  For rays near the mid-plane, one can also use these cost functions for cone-parallel and cone-beam coordinates as well.

        Note that this only works for parallel-, fan-, and cone-beam CT geometry types (i.e., everything but modular-beam)
        and one may not get an accurate estimate if the projections are truncated on the right and/or left sides.
        If you have any bad edge detectors, these must be cropped out before running this algorithm.
        If this function does not return a good estimate, try changing the iRow parameter value or try using the
        inconsistencyReconstruction function is this class.
        
        See d12_geometric_calibration.py for a working example that uses this function.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            iRow (int): The detector row index to be used for the estimation, if no value is given, uses the row closest to the centerRow parameter
            searchBounds (2-element array): optional argument to specify the interval for which to perform the search
            
        Returns:
            the error metric value
        
        """
        if iRow is None:
            iRow = -1
        if searchBounds is None:
            searchBounds = -1.0*np.ones(2, dtype=np.float32)
        else:
            searchBounds = np.array(searchBounds, dtype=np.float32)
        self.libprojectors.find_centerCol.restype = ctypes.c_float
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.find_centerCol.argtypes = [ctypes.c_void_p, ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            return self.libprojectors.find_centerCol(g.data_ptr(), iRow, searchBounds, g.is_cuda == False)
        else:
            self.libprojectors.find_centerCol.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            return self.libprojectors.find_centerCol(g, iRow, searchBounds, True)
        
    def find_tau(self, g, iRow=-1, searchBounds=None):
        r"""Find the tau parameter

        This function works by minimizing the difference of conjugate rays, by changing the horizontal source position shift (equivalent to rotation stage shifts).
        The cost function for fan-beam is given by
        
        .. math::
           \begin{eqnarray*}
           &&\int \int \left[g(u,\varphi) - g\left(\frac{-u+\frac{2\tau R}{R^2-\tau^2}}{1+u\left(\frac{2\tau R}{R^2-\tau^2}\right)},\varphi -2\tan^{-1}u + \tan^{-1}\left(\frac{2\tau R}{R^2-\tau^2}\right) \pm \pi\right)\right]^2 \, du \, d\varphi, \\
           \end{eqnarray*}
           
        For rays near the mid-plane, one can also use these cost functions for cone-beam coordinates as well.

        Note that this only works for fan- and cone-beam CT geometry types
        and one may not get an accurate estimate if the projections are truncated on the right and/or left sides.
        If you have any bad edge detectors, these must be cropped out before running this algorithm.
        If this function does not return a good estimate, try changing the iRow parameter value or try using the
        inconsistencyReconstruction function is this class.
        
        See d12_geometric_calibration.py for a working example that uses this function.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            iRow (int): The detector row index to be used for the estimation, if no value is given, uses the row closest to the centerRow parameter
            searchBounds (2-element array): optional argument to specify the interval for which to perform the search
            
        Returns:
            the error metric value
        
        """
        if iRow is None:
            iRow = -1
        if searchBounds is None:
            searchBounds = -1.0*np.ones(2, dtype=np.float32)
        else:
            searchBounds = np.array(searchBounds, dtype=np.float32)
        self.libprojectors.find_tau.restype = ctypes.c_float
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.find_tau.argtypes = [ctypes.c_void_p, ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            return self.libprojectors.find_tau(g.data_ptr(), iRow, searchBounds, g.is_cuda == False)
        else:
            self.libprojectors.find_tau.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            return self.libprojectors.find_tau(g, iRow, searchBounds, True)
        
    def estimate_tilt(self, g):
        """Estimates the tilt angle (around the optical axis) of the detector

        This algorithm works by minimizing the difference between conjugate projections (those projections separated by 180 degrees).
        If the input data is fan-beam or cone-beam it first rebins the data to parallel-beam or cone-parallel coordinates first
        and then calculates the difference of conjugate projections.  This algorithm works best if centerCol is properly specified
        before running this algorithm.
        
        Note that this function does not update any CT geometry parameters.
        See also the conjugate_difference function.
        
        Example Usage:
        gamma = leapct.estimate_tilt(g)
        leapct.set_tiltAngle(gamma)

        Note that it only works for parallel-beam, fan-beam, cone-beam, and cone-parallel CT geometry types (i.e., everything but modular-beam)
        and one may not get an accurate estimate if the projections are truncated on the right and/or left sides.
        If you have any bad edge detectors, these must be cropped out before running this algorithm.
        If this function does not return a good estimate, try using the
        inconsistencyReconstruction function is this class.
        
        See d12_geometric_calibration.py for a working example that uses this function.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            
        Returns:
            the tilt angle (in degrees)
        
        """
        self.libprojectors.estimate_tilt.restype = ctypes.c_float
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.estimate_tilt.argtypes = [ctypes.c_void_p, ctypes.c_bool]
            return self.libprojectors.estimate_tilt(g.data_ptr(), g.is_cuda == False)
        else:
            self.libprojectors.estimate_tilt.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            return self.libprojectors.estimate_tilt(g, True)
            
    def conjugate_difference(self, g, alpha=0.0, centerCol=None):
        """Calculates the difference of conjugate projections with optional detector rotation and detector shift

        This algorithm calculates the difference between conjugate projections (those projections separated by 180 degrees).
        If the input data is fan-beam or cone-beam it first rebins the data to parallel-beam or cone-parallel coordinates first
        and then calculates the difference of conjugate projections.  The purpose of this function is to provide a metric
        for estimating detector tilt (rotation around the optical axis) and horizonal detector shifts (centerCol).

        Note that it only works for parallel-beam, fan-beam, cone-beam, and cone-parallel CT geometry types (i.e., everything but modular-beam)
        and one may not get an accurate estimate if the projections are truncated on the right and/or left sides.
        If you have any bad edge detectors, these must be cropped out before running this algorithm.
        If this function does not return a good estimate, try using the
        inconsistencyReconstruction function is this class.
        
        See d12_geometric_calibration.py for a working example that uses this function.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            alpha (float): detector rotation (around the optical axis) in degrees
            centerCol (float): the centerCol parameter to use (if unspecified uses the current value of centerCol)
            
        Returns:
            2D numpy array of the difference of two conjugate projections
        
        """
        numRows = self.get_numRows()
        numCols = self.get_numCols()
        if numRows <= 0 and numCols <= 0:
            print('Error: must get CT geometry parameters before using this function!')
            return None
        if centerCol is None:
            centerCol = self.get_centerCol()
        
        self.libprojectors.conjugate_difference.restype = ctypes.c_float
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            diff = torch.zeros((numRows, numCols), dtype=torch.float32)
            self.libprojectors.conjugate_difference.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_void_p, ctypes.c_bool]
            self.libprojectors.conjugate_difference(g.data_ptr(), alpha, centerCol, diff.data_ptr(), g.is_cuda == False)
        else:
            diff = np.zeros((numRows, numCols), dtype=np.float32)
            self.libprojectors.conjugate_difference.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.conjugate_difference(g, alpha, centerCol, diff, True)
        return diff
        
    def consistency_cost(self, g, Delta_centerRow=0.0, Delta_centerCol=0.0, Delta_tau=0.0, Delta_tilt=0.0):
        r"""Calculates a cost metric for the given CT geometry perturbations

        Note that it only works for the axial flat-panel cone-beam CT geometry type
        and the projections cannot be truncated on the right or left sides.
        If you have any bad edge detectors, these must be cropped out before running this algorithm.
        
        This function implements the algorithm in the following paper:
        Lesaint, Jerome, Simon Rit, Rolf Clackdoyle, and Laurent Desbat.
        Calibration for circular cone-beam CT based on consistency conditions.
        IEEE Transactions on Radiation and Plasma Medical Sciences 1, no. 6 (2017): 517-526.
        
        See d12_geometric_calibration.py for a working example that uses this function.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            Delta_centerRow (float): detector shift (detector row pixel index) in the row direction
            Delta_centerCol (float): detector shift (detector column pixel index) in the column direction
            Delta_tau (float): horizonal shift (mm) of the detector; can also be used to model detector rotations along the vector pointing across the detector rows
            Delta_tilt (float): detector rotation (degrees) around the optical axis
            
        Returns:
            the cost value of the metric
        
        """
        if type(Delta_centerRow) is list or type(Delta_centerRow) is np.ndarray:
            if len(Delta_centerRow) == 2:
                Delta_centerCol = Delta_centerRow[1]
                Delta_centerRow = Delta_centerRow[0]
            elif len(Delta_centerRow) == 3:
                Delta_tilt = Delta_centerRow[2]
                #Delta_tau = Delta_centerRow[2]
                Delta_centerCol = Delta_centerRow[1]
                Delta_centerRow = Delta_centerRow[0]
            elif len(Delta_centerRow) == 4:
                Delta_tilt = Delta_centerRow[3]
                Delta_tau = Delta_centerRow[2]
                Delta_centerCol = Delta_centerRow[1]
                Delta_centerRow = Delta_centerRow[0]
        self.libprojectors.consistency_cost.restype = ctypes.c_float
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.consistency_cost.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.consistency_cost(g.data_ptr(), Delta_centerRow, Delta_centerCol, Delta_tau, Delta_tilt, g.is_cuda == False)
        else:
            self.libprojectors.consistency_cost.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.consistency_cost(g, Delta_centerRow, Delta_centerCol, Delta_tau, Delta_tilt, True)
        
    def set_centerRow(self, centerRow):
        """Set centerRow parameter"""
        self.set_model()
        self.libprojectors.set_centerRow.restype = ctypes.c_bool
        self.libprojectors.set_centerRow.argtypes = [ctypes.c_float]
        return self.libprojectors.set_centerRow(centerRow)
        
    def set_sod(self, sod):
        """Set sod parameter"""
        self.set_model()
        self.libprojectors.set_sod.restype = ctypes.c_bool
        self.libprojectors.set_sod.argtypes = [ctypes.c_float]
        return self.libprojectors.set_sod(sod)
        
    def set_sdd(self, sdd):
        """Set sdd parameter"""
        self.set_model()
        self.libprojectors.set_sdd.restype = ctypes.c_bool
        self.libprojectors.set_sdd.argtypes = [ctypes.c_float]
        return self.libprojectors.set_sdd(sdd)
        
    def convert_to_modularbeam(self):
        """Converts parallel- or cone-beam data to a modular-beam format for extra customization of the scanning geometry"""
        #dFOV = self.get_diameterFOV()
        #self.set_diameterFOV(dFOV)
        if self.get_geometry() == 'PARALLEL':
            return self.convert_parallelbeam_to_modularbeam()
        elif self.get_geometry() == 'CONE':
            return self.convert_conebeam_to_modularbeam()
        elif self.get_geometry() == 'MODULAR':
            return True
        else:
            print('Error: transformations to modular-beam not defined for this geometry')
            return False
        
    def convert_conebeam_to_modularbeam(self):
        """sets modular-beam parameters from a cone-beam specification"""
        self.set_model()
        self.libprojectors.convert_conebeam_to_modularbeam.restype = ctypes.c_bool
        return self.libprojectors.convert_conebeam_to_modularbeam()
        
    def convert_parallelbeam_to_modularbeam(self):
        """sets modular-beam parameters from a parallel-beam specification"""
        self.set_model()
        self.libprojectors.convert_parallelbeam_to_modularbeam.restype = ctypes.c_bool
        return self.libprojectors.convert_parallelbeam_to_modularbeam()
        
    def rotate_detector(self, alpha):
        """Rotates modular-beam detector by updating the modular-beam CT geometry specification
        
        The CT geometry parameters must be defined before running this function and the 
        CT geometry must be modular-beam.  If it is not modular-beam, one may use the
        convert_to_modularbeam() function.
        
        Note that there are two forms for the argument of this function.  If the argument is a scalar, a
        rotation is performed around the optical axis, otherwise the argument can be specified as a 3x3
        rotation matrix.  A good method to specify a rotation matrix is the following:
        from scipy.spatial.transform import Rotation as R
        A = R.from_euler('xyz', [psi, theta, phi], degrees=True).as_matrix()

        Args:
            alpha (float or 3x3 numpy array): if alpha is a scalar, a rotation is performed around the optical axis, otherwise alpha can be specified as a 3x3 rotation matrix
        
        Returns:
            True if the operation was successful, False overwise
        
        """
        geom_str = self.get_geometry()
        
        if geom_str != 'MODULAR' and geom_str != 'CONE':
            print('Error: can only rotate cone- and modular-beam detectors')
            print('Use convert_to_modularbeam() first')
            return False
        if type(alpha) is np.ndarray:
            if geom_str != 'MODULAR':
                print('Error: can only perform arbitrary rotations with modular-beam geometries')
                print('Use convert_to_modularbeam() first')
                return False
            if alpha.size == 3:
                from scipy.spatial.transform import Rotation as R
                A = R.from_euler('xyz', alpha, degrees=True).as_matrix()
            elif len(alpha.shape) != 2 or alpha.shape[0] != 3 or alpha.shape[1] != 3:
                print('Error: input argument must be scalar or a 3x3 numpy array')
                return False
            else:
                A = np.ascontiguousarray(alpha.copy(), dtype=np.float32)
            if np.linalg.det(A) == 0.0:
                print('Error: rotation matrix must be nonsingular')
                return False
            A = A / np.linalg.det(A) # just make sure people aren't doing anything weird
            
            self.set_model()
            rowVecs = self.get_rowVectors()
            colVecs = self.get_colVectors()
            B = A.copy()
            for n in range(rowVecs.shape[0]):
                u_vec = colVecs[n,:]
                v_vec = rowVecs[n,:]
                n_vec = np.cross(u_vec, v_vec)
                B[:,1] = n_vec
                B[:,0] = u_vec
                B[:,2] = v_vec
                #rowVecs[n,:] = np.matmul(A, np.matmul(rowVecs[n,:], A.T))
                #colVecs[n,:] = np.matmul(A, np.matmul(colVecs[n,:], A.T))
                B = np.matmul(B,A.T)
                colVecs[n,:] = B[:,0]
                rowVecs[n,:] = B[:,2]
            
            return self.set_modularBeam(self.get_numAngles(), self.get_numRows(), self.get_numCols(), self.get_pixelHeight(), self.get_pixelWidth(), self.get_sourcePositions(), self.get_moduleCenters(), rowVecs, colVecs)
            
        else:
            self.set_model()
            if geom_str == 'MODULAR':
                self.libprojectors.rotate_detector.restype = ctypes.c_bool
                self.libprojectors.rotate_detector.argtypes = [ctypes.c_float]
                return self.libprojectors.rotate_detector(alpha)
            else:
                return self.set_tiltAngle(self.get_tiltAngle()+alpha)
        
    def rotate_coordinate_system(self, R):
        """Rotates the coordinate system
        
        This main purpose of this algorithm is to enable reconstruction on arbitrary voxel grid.
        This is only possible for modular-beam geometries, so if your geometry is not modular-beam
        then use the convert_to_modularbeam function first.
        Note that if after the rotation, the vector pointing along across the detector rows isn't
        aligned within 5 degrees of the z-axis, then one will not be able to perform FBP reconstructions
        
        Args:
            R (3X3 numpy array): rotation matrix
            
        Returns:
            true if operation was sucessful, false otherwise
        """
        if self.get_numAngles() <= 0:
            print('Error: must specify CT geometry first')
            return False
        if self.get_geometry() != 'MODULAR':
            print('Error: can only rotate coordinate system when using modular-beam geometry')
            print('Use convert_to_modularbeam first')
            return False
        if type(R) is not np.ndarray or len(R.shape) != 2 or R.shape[0] != 3 or R.shape[1] != 3:
            print('Error: must provide a 3X3 numpy array')
            return False
        self.set_model()
        sourcePositions = self.get_sourcePositions()
        moduleCenters = self.get_moduleCenters()
        rowVecs = self.get_rowVectors()
        colVecs = self.get_colVectors()
        for n in range(sourcePositions.shape[0]):
            sourcePositions[n,:] = np.matmul(R, sourcePositions[n,:])
            moduleCenters[n,:] = np.matmul(R, moduleCenters[n,:])
            rowVecs[n,:] = np.matmul(R, rowVecs[n,:])
            colVecs[n,:] = np.matmul(R, colVecs[n,:])
        
        numAngles = self.get_numAngles()
        numRows = self.get_numRows()
        numCols = self.get_numCols()
        pixelHeight = self.get_pixelHeight()
        pixelWidth = self.get_pixelWidth()
        self.set_modularBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVecs, colVecs)
        return True
        
    def shift_detector(self, r, c):
        """Shifts the detector by r mm in the row direction and c mm in the column direction by updating the CT geometry parameters accordingly"""
        self.set_model()
        self.libprojectors.shift_detector.restype = ctypes.c_bool
        self.libprojectors.shift_detector.argtypes = [ctypes.c_float, ctypes.c_float]
        return self.libprojectors.shift_detector(r, c)
        
    def rebin_curved(self, g, fanAngles, order=6):
        """ rebin data from a curved array of detector modules
        
        Note that if your data is already sampled on a curved detector, you do not need to use this function.
        Real curved detectors are composed of a series of detector modules curved around a polygon shape.
        There are often gaps between modules that need to be accounted for and that is what this function does.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            fanAngles (C contiguous float32 numpy array): array of the fan angle (degrees) of every detector pixel in a row
            order (int): the order of the interpolating polynomial
        
        """
        self.set_model()
        self.libprojectors.rebin_curved.restype = ctypes.c_bool
        if has_torch == True and type(g) is torch.Tensor:
        
            if g.is_cuda == True:
                print('Error: rebin_curved only implemented for data on CPU!')
                print('If this feature is of interest please submit a feature request.')
                return False
                
            if type(fanAngles) is torch.Tensor:
                fanAngles = fanAngles.cpu().detach().numpy()
        
            self.libprojectors.rebin_curved.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int]
            return self.libprojectors.rebin_curved(g.data_ptr(), fanAngles, order)
        else:
            self.libprojectors.rebin_curved.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int]
            return self.libprojectors.rebin_curved(g, fanAngles, order)
        
    def rebin_parallel(self, g, order=6):
        """ rebin data from fan-beam to parallel-beam or cone-beam to cone-parallel
        
        The CT geometry parameters must be defined before running this function.
        After the completion of this algorithm, the CT geometry parameters will be
        modified as necessary and the number of projections may reduce.  If the number
        of projections does reduce most LEAP functions will work correctly, but we
        recommend that users resize their arrays.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            order (int): the order of the interpolating polynomial
        
        """
        self.set_model()
        self.libprojectors.rebin_parallel.restype = ctypes.c_bool
        
        if has_torch == True and type(g) is torch.Tensor:
        
            if g.is_cuda == True:
                print('Error: rebin_parallel only implemented for data on CPU!')
                print('If this feature is of interest please submit a feature request.')
                return False
        
            self.libprojectors.rebin_parallel.argtypes = [ctypes.c_void_p, ctypes.c_int]
            return self.libprojectors.rebin_parallel(g.data_ptr(), order)
        else:
            self.libprojectors.rebin_parallel.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int]
            return self.libprojectors.rebin_parallel(g, order)
    
    def rebin_parallel_sinogram(self, g, order=6, iRow=-1):
        r""" rebin a single sinogram to parallel-beam or cone-parallel coordinates
        
        The CT geometry parameters must be defined before running this function.
        This function does not rebin the whole data set.  The purpose of this function
        is to generate a parallel-beam sinogram that can be used for geometric calibration.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            order (int): the order of the interpolating polynomial
            iRow (int): detector row index to perform rebinning
            
        Returns:
            2D array of the rebinned sinogram
        
        """

        self.set_model()
        self.libprojectors.rebin_parallel_sinogram.restype = ctypes.c_int
        
        if has_torch == True and type(g) is torch.Tensor:
        
            if g.is_cuda == True:
                print('Error: rebin_parallel_sinogram only implemented for data on CPU!')
                print('If this feature is of interest please submit a feature request.')
                return None

            sino = torch.zeros((self.get_numAngles(), self.get_numCols()), dtype=torch.float32)

            self.libprojectors.rebin_parallel_sinogram.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
            numRays = self.libprojectors.rebin_parallel_sinogram(g.data_ptr(), sino.data_ptr(), order, iRow)
            if numRays != self.get_numCols():
                sino = torch.reshape(sino, (self.get_numAngles()//2, 2*self.get_numCols()))
            return sino
        else:
            sino = np.zeros((self.get_numAngles(), self.get_numCols()), dtype=np.float32)

            self.libprojectors.rebin_parallel_sinogram.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int]
            numRays = self.libprojectors.rebin_parallel_sinogram(g, sino, order, iRow)
            if numRays != self.get_numCols():
                sino = np.reshape(sino, (self.get_numAngles()//2, 2*self.get_numCols()))
            return sino

    def sinogram_replacement(self, g, priorSinogram, metalTrace, windowSize=None):
        """ replaces specified region in projection data with other projection data
        
        This routine provides a robust solution to metal artifact reduction (MAR).
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data to alter
            priorSinogram (C contiguous float32 numpy array or torch tensor): projection data to use for patching
            metalTrace (C contiguous float32 numpy array or torch tensor): projection mask showing where to do the patching
            windowSize (3-element int array): window size in each of the three dimensions, default is [30, 1, 50]
            
        Returns:
            true if operation was sucessful, false otherwise
        """
        
        if windowSize is None:
            windowSize = np.array([3, 1, 50], dtype=np.int32)
            
        self.set_model()
        self.libprojectors.sinogram_replacement.restype = ctypes.c_bool
        if has_torch == True and type(g) is torch.Tensor:
        
            if g.is_cuda == True:
                print('Error: sinogram replacement only implemented for data on CPU!')
                print('If this feature is of interest please submit a feature request.')
                return False
                
            if type(windowSize) is torch.Tensor:
                windowSize = windowSize.cpu().detach().numpy()
        
            self.libprojectors.sinogram_replacement.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
            return self.libprojectors.sinogram_replacement(g.data_ptr(), priorSinogram.data_ptr(), metalTrace.data_ptr(), windowSize)
        else:
            self.libprojectors.sinogram_replacement.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
            return self.libprojectors.sinogram_replacement(g, priorSinogram, metalTrace, windowSize)
            
    def down_sample(self, factors, I, dims=None):
        """down-samples the given 3D array
        
        Prior to down-sampling, an anti-alias filter is applied to each dimension.  This anti-aliasing filter
        is the same one used in the LowPassFilter function.
        
        Args:
            factors: 3-element array of down-sampling factors
            I (C contiguous float32 numpy array or torch tensor): data to down-sample
            
        Returns:
            down-sampled array
        """
        
        self.set_model()
        self.libprojectors.down_sample.restype = ctypes.c_bool
        if has_torch == True and type(I) is torch.Tensor:
            if type(factors) is torch.Tensor:
                factors = factors.cpu().detach().numpy()
            factors = np.array(factors, dtype=np.float32)
            #elif type(factors) is not np.ndarray:
            #    factors = np.array(factors, dtype=np.float32)
            if factors.size != 3:
                return None
            
            if dims is None:
                I_dn = np.zeros((int(I.shape[0]/factors[0]), int(I.shape[1]/factors[1]), int(I.shape[2]/factors[2])), dtype=np.float32)
            else:
                I_dn = np.zeros((dims[0], dims[1], dims[2]), dtype=np.float32)
            device = torch.device("cuda:" + str(self.get_gpu()))
            I_dn = torch.from_numpy(I_dn).to(device)
            self.libprojectors.down_sample.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_void_p, ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.down_sample(I.data_ptr(), np.array(I.shape, dtype=np.int32), I_dn.data_ptr(), np.array(I_dn.shape, dtype=np.int32), factors, I.is_cuda == False)
        else:
            #if type(factors) is not np.ndarray:
            #    factors = np.array(factors, dtype=np.float32)
            factors = np.array(factors, dtype=np.float32)
            if factors.size != 3:
                return None
                
            if dims is None:
                I_dn = np.zeros((int(I.shape[0]/factors[0]), int(I.shape[1]/factors[1]), int(I.shape[2]/factors[2])), dtype=np.float32)
            else:
                I_dn = np.zeros((dims[0], dims[1], dims[2]), dtype=np.float32)
            self.libprojectors.down_sample.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.down_sample(I, np.array(I.shape, dtype=np.int32), I_dn, np.array(I_dn.shape, dtype=np.int32), factors, True)
        return I_dn
        
    def up_sample(self, factors, I, dims=None):
        """up-samples the given 3D array
        
        Up-sampling is performed using tri-linear interpolation.
        
        Args:
            factors: 3-element array of up-sampling factors
            I (C contiguous float32 numpy array or torch tensor): data to up-sample
            
        Returns:
            up-sampled array
        """
        
        self.set_model()
        self.libprojectors.up_sample.restype = ctypes.c_bool
        if has_torch == True and type(I) is torch.Tensor:
            if type(factors) is torch.Tensor:
                factors = factors.cpu().detach().numpy()
            factors = np.array(factors, dtype=np.float32)
            #elif type(factors) is not np.ndarray:
            #    factors = np.array(factors, dtype=np.float32)
            if factors.size != 3:
                return None
            
            if dims is None:
                I_up = np.zeros((int(I.shape[0]*factors[0]), int(I.shape[1]*factors[1]), int(I.shape[2]*factors[2])), dtype=np.float32)
            else:
                I_up = np.zeros((dims[0], dims[1], dims[2]), dtype=np.float32)
            device = torch.device("cuda:" + str(self.get_gpu()))
            I_up = torch.from_numpy(I_up).to(device)
            self.libprojectors.up_sample.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_void_p, ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.up_sample(I.data_ptr(), np.array(I.shape, dtype=np.int32), I_up.data_ptr(), np.array(I_up.shape, dtype=np.int32), factors, I.is_cuda == False)
        else:
            if type(factors) is not np.ndarray:
                factors = np.array(factors, dtype=np.float32)
            if factors.size != 3:
                return None
                
            if dims is None:
                I_up = np.zeros((int(I.shape[0]*factors[0]), int(I.shape[1]*factors[1]), int(I.shape[2]*factors[2])), dtype=np.float32)
            else:
                I_up = np.zeros((dims[0], dims[1], dims[2]), dtype=np.float32)
            self.libprojectors.up_sample.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.up_sample(I, np.array(I.shape, dtype=np.int32), I_up, np.array(I_up.shape, dtype=np.int32), factors, True)
        return I_up
        
    def scatter_model(self, f, source, energies, detector, sigma, scatterDist, jobType=-1):
        """simulates first order scatter through an object composed of a single material type (but variable density)

        This is a complicated function and one should refer to the demo script: d31_scatter_correction.py
        for a full working demonstration.
        
        This routine requires that one down-sample the CT detector pixels, volume voxels, and number of source spectra samples.
        The volume should be no larger than 200^3 voxels
        The projection data should be no larger than 256^2
        The source spectra should have no more than 20 samples (we recommend about 10 samples)
        
        Note that source and energies are defined on a course grid and the number of samples has an almost
        linear effect on the computation time of these routines.
        But detector, sigma, scatterDist are sampled from 1 keV to int(ceil(energies[-1])) in 1 keV bins
        These three arguments MUST be specified like this and the fine sampling here does not effect computation time
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): volume data in mass density units
            source (numpy array): source spectra (including response due to filters)
            energies (numpy array): energy samples for source spectra
            detector (numpy array): detector response (sampled from 1 keV to int(ceil(energies[-1])) in 1 keV bins)
            sigma (2D numpy array): sigma[0,:] = sigmaPE, sigma[1,:] = sigmaCS, sigma[2,:] = sigmaRS, energy samples same as detector response
            scatterDist (3D numpy array): incoherent and coherent scatter distributions [2 x number of energy bins x 181]
            jobType (int): if -1 estimates scatter correction gain factor, if 1 estimates scatter simulation (adds scatter to data) gain factor, and if 0 estimates scatter transmission
        
        Returns:
            Returns gain factor or scatter transmission numpy array based on the jobType parameter
        """
        
        if self.get_geometry() != 'MODULAR':
            print('Error: this function only works for modular-beam geometries.  Please use the function\"convert_to_modularbeam()\" prior to running this algorithm to convert your geometry to modular-beam.')
            return None
        if len(f.shape) != 3:
            print('Error: mass density argument must be a 3D numpy or torch tensor.')
            return None
        if f.shape[0]*f.shape[1]*f.shape[2] > 200**3:
            print('Error: number of voxels must be less than 200 x 200 x 200.  Please down-sample volume and try again.')
            return None
        if self.get_numCols() * self.get_numRows() > 256**2:
            print('Error: number of detector pixels must be less than 256 x 256.  Please down-sample projections and try again.')
            return None
        if source.size > 20:
            print('Error: number of energy bins in source spectra must be less than 20.  Please down spectra and try again.')
            return None
        if source.size != energies.size:
            print('Error: \"source\" spectra and \"energies\" must be the same size.')
            return None
        maxEnergy = int(np.ceil(energies[-1]))
        if detector.size != maxEnergy:
            print('Error: \"detector\" must have exactly ' + str(maxEnergy) + ' bins')
            return None
        if sigma.shape[0] != 3 or sigma.shape[1] != maxEnergy:
            print('Error: \"sigma\" must have exactly 3 x ' + str(maxEnergy) + ' bins')
            return None
        if scatterDist.shape[0] != 2 or scatterDist.shape[1] != maxEnergy or scatterDist.shape[2] != 181:
            print('Error: \"scatterDist\" must have exactly 2 x ' + str(maxEnergy) + ' x 181 bins')
            return None
                
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            g = self.allocate_projections(0.0, True)
            
            if f.is_cuda:
                g = g.to(f.get_device())
            
            if type(source) is torch.Tensor:
                source = source.cpu().detach().numpy()
            if type(energies) is torch.Tensor:
                energies = energies.cpu().detach().numpy()
            if type(detector) is torch.Tensor:
                detector = detector.cpu().detach().numpy()
            if type(sigma) is torch.Tensor:
                sigma = sigma.cpu().detach().numpy()
            if type(scatterDist) is torch.Tensor:
                scatterDist = scatterDist.cpu().detach().numpy()
            
            self.libprojectors.scatter_model.restype = ctypes.c_bool
            self.libprojectors.scatter_model.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_int]
            self.libprojectors.scatter_model(g.data_ptr(), f.data_ptr(), source, energies, energies.size, detector, sigma, scatterDist, g.is_cuda == False, jobType)
            
        else:
            g = self.allocate_projections()
            self.libprojectors.scatter_model.restype = ctypes.c_bool
            self.libprojectors.scatter_model.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_int]
            self.libprojectors.scatter_model(g, f, source, energies, energies.size, detector, sigma, scatterDist, True, jobType)
        return g
    
    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS SET THE CT VOLUME PARAMETERS
    ###################################################################################################################
    ###################################################################################################################
    def set_volume(self, numX, numY, numZ, voxelWidth=None, voxelHeight=None, offsetX=None, offsetY=None, offsetZ=None):
        r"""Set the CT volume parameters
        
        The samples are given by
        
        .. math::
           \begin{eqnarray*}
           x[i] &:=& voxelWidth\left(i - \frac{numX-1}{2}\right) + offsetX, \qquad i = 0,1,\dots,numX-1 \\
           y[j] &:=& voxelWidth\left(j - \frac{numY-1}{2}\right) + offsetY, \qquad j = 0,1,\dots,numY-1 \\
           z[k] &:=& voxelHeight\left(k - \frac{numZ-1}{2}\right) + offsetZ, \qquad k = 0,1,\dots,numZ-1
           \end{eqnarray*}
        
        For convenience we also provide an automated volume specification function: set_default_volume().
        See also the set_diameterFOV() function which affects the circular mask applied to the z-slices of the reconstruction.
        
        Args:
            numX (int): number of voxels in the x-dimension
            numY (int): number of voxels in the y-dimension
            numZ (int): number of voxels in the z-dimension
            voxelWidth (float): voxel pitch (size) in the x and y dimensions
            voxelHeight (float): voxel pitch (size) in the z dimension
            offsetX (float): shift the volume in the x-dimension, measured in mm
            offsetY (float): shift the volume in the y-dimension, measured in mm
            offsetZ (float): shift the volume in the z-dimension, measured in mm
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        self.libprojectors.set_volume.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.libprojectors.set_volume.restype = ctypes.c_bool
        if voxelWidth is None:
            voxelWidth = 0.0
        if voxelHeight is None:
            voxelHeight = voxelWidth
        if offsetX is None:
            offsetX = 0.0
        if offsetY is None:
            offsetY = 0.0
        if offsetZ is None:
            offsetZ = 0.0
        self.set_model()
        return self.libprojectors.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ)
        
    def set_default_volume(self,scale=1.0):
        """Sets the default volume parameters
        
        The default volume parameters are those that fill the field of view of the CT system and use the native voxel sizes.
        Note that the CT geometry parameters must be specified before running this function.
        
        Args:
            scale (float): this value scales the voxel size by this value to create denser or sparser voxel representations (not recommended for fast reconstruction)
        
        Returns:
            True if the operation was successful, false otherwise (this usually happens if the CT geometry has not yet been set)
        """
        self.libprojectors.set_default_volume.argtypes = [ctypes.c_float]
        self.libprojectors.set_default_volume.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_default_volume(scale)
    
    def set_defaultVolume(self,scale=1.0):
        """Alias for set_default_volume
        """
        return self.set_default_volume(scale)
    
    def set_volumeDimensionOrder(self,which):
        """Sets the order of the dimensions of the volume
        
        WARNING: multi-GPU processing only works for ZYX order
        
        Args:
            setVolumeDimensionOrder (int): 0 sets the order to XYZ, 1 sets the order to ZYX (this is the default value)
            
        Returns:
            True if the operation was successful, false otherwise
        """
        self.libprojectors.set_volumeDimensionOrder.argtypes = [ctypes.c_int]
        self.libprojectors.set_volumeDimensionOrder.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_volumeDimensionOrder(which)
        
    def get_volumeDimensionOrder(self):
        """Alias for set_volumeDimensionOrder
        """
        return self.libprojectors.get_volumeDimensionOrder()
        
    def set_numZ(self, numZ):
        """Set the number of voxels in the z-dimension"""
        self.libprojectors.set_numZ.argtypes = [ctypes.c_int]
        self.libprojectors.set_numZ.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_numZ(numZ)
        
    def set_numY(self, numY):
        """Set the number of voxels in the y-dimension"""
        self.libprojectors.set_numY.argtypes = [ctypes.c_int]
        self.libprojectors.set_numY.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_numY(numY)
        
    def set_numX(self, numX):
        """Set the number of voxels in the x-dimension"""
        self.libprojectors.set_numX.argtypes = [ctypes.c_int]
        self.libprojectors.set_numX.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_numX(numX)
        
    def set_offsetX(self, offsetX):
        """Set offsetX parameter which defines the central z-slice location (mm) of the volume"""
        self.libprojectors.set_offsetX.argtypes = [ctypes.c_float]
        self.libprojectors.set_offsetX.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_offsetX(offsetX)
        
    def set_offsetY(self, offsetY):
        """Set offsetY parameter which defines the central z-slice location (mm) of the volume"""
        self.libprojectors.set_offsetY.argtypes = [ctypes.c_float]
        self.libprojectors.set_offsetY.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_offsetY(offsetY)
        
    def set_offsetZ(self, offsetZ):
        """Set offsetZ parameter which defines the central z-slice location (mm) of the volume"""
        self.libprojectors.set_offsetZ.argtypes = [ctypes.c_float]
        self.libprojectors.set_offsetZ.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_offsetZ(offsetZ)
        
    def set_voxelHeight(self, H):
        """Sets the voxel height (mm) parameter"""
        self.libprojectors.set_voxelHeight.argtypes = [ctypes.c_float]
        self.libprojectors.set_voxelHeight.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_voxelHeight(H)
        
    def set_voxelWidth(self, W):
        """Sets the voxel width (mm) parameter"""
        self.libprojectors.set_voxelWidth.argtypes = [ctypes.c_float]
        self.libprojectors.set_voxelWidth.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_voxelWidth(W)
        
    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS PROVIDE CONVENIENT ROUTINES TO MAKE THE PROJECTION DATA AND VOLUME DATA NUMPY ARRAYS
    ###################################################################################################################
    ###################################################################################################################
    def allocate_projections(self, val=0.0, astensor=False, forOffsetScanFilteringStep=False):
        """Allocates projection data
        
        It is not necessary to use this function. It is included simply for convenience.

        Args:
            val (float): value to fill the array with
            astensor (bool): if true turns array into a pytorch tensor
            
        Returns:
            numpy array/ pytorch tensor if numAngles, numRows, and numCols are all positive, None otherwise
        """
        return self.allocateProjections(val, astensor, forOffsetScanFilteringStep)
    
    def allocateProjections(self, val=0.0, astensor=False, forOffsetScanFilteringStep=False):
        """Allocates projection data
        
        It is not necessary to use this function. It is included simply for convenience.

        Args:
            val (float): value to fill the array with
            astensor (bool): if true turns array into a pytorch tensor
            
        Returns:
            numpy array/ pytorch tensor if numAngles, numRows, and numCols are all positive, None otherwise
        """
        N_phis = self.get_numAngles()
        N_rows = self.get_numRows()
        N_cols = self.get_numCols()
        if forOffsetScanFilteringStep:
            N_cols += self.extraColumnsForOffsetScan()
        if N_phis > 0 and N_rows > 0 and N_cols > 0:
            if val == 0.0:
                g = np.ascontiguousarray(np.zeros((N_phis,N_rows,N_cols),dtype=np.float32), dtype=np.float32)
            else:
                g = np.ascontiguousarray(val*np.ones((N_phis,N_rows,N_cols),dtype=np.float32), dtype=np.float32)
            if has_torch and astensor:
                g = torch.from_numpy(g)
            return g
        else:
            return None
            
    def allocateProjections_gpu(self, val=0.0, forOffsetScanFilteringStep=False):
        """Allocates projection data as a pytorch tensor on the gpu
        
        It is not necessary to use this function. It is included simply for convenience.

        Args:
            val (float): value to fill the array with
            
        Returns:
            pytorch tensor if numAngles, numRows, and numCols are all positive, None otherwise
        """
        if has_torch == False:
            print('Error: you must have pytorch installed for this function')
            return None
        N_phis = self.get_numAngles()
        N_rows = self.get_numRows()
        N_cols = self.get_numCols()
        if forOffsetScanFilteringStep:
            N_cols += self.extraColumnsForOffsetScan()
        if N_phis > 0 and N_rows > 0 and N_cols > 0:
            if val == 0.0:
                g = np.ascontiguousarray(np.zeros((N_phis,N_rows,N_cols),dtype=np.float32), dtype=np.float32)
            else:
                g = np.ascontiguousarray(val*np.ones((N_phis,N_rows,N_cols),dtype=np.float32), dtype=np.float32)
            if has_torch:
                #g = torch.from_numpy(g).float().to(max(0,self.get_gpu()))
                g = self.copy_to_device(g)
            return g
        else:
            return None
        
        
    def get_volume_dim(self):
        """Get dimension sizes of volume data
        
        It is not necessary to use this function. It is included simply for convenience.

        Returns:
            numpy array of the dimension sizes of the volume
        """
        N_x = self.get_numX()
        N_y = self.get_numY()
        N_z = self.get_numZ()
        if self.get_volumeDimensionOrder() == 0:
            return np.array([N_x,N_y,N_z],dtype=np.int32)
        else:
            return np.array([N_z,N_y,N_x],dtype=np.int32)
            
    def get_projection_dim(self):
        """Get dimension sizes of projection data
        
        It is not necessary to use this function. It is included simply for convenience.

        Returns:
            numpy array of the dimension sizes of the projections
        """
        return np.array([self.get_numAngles(),self.get_numRows(),self.get_numCols()],dtype=np.int32)
        
    def allocate_volume(self, val=0.0, astensor=False):
        """Allocates reconstruction volume data
        
        It is not necessary to use this function. It is included simply for convenience.

        Args:
            val (float): value to fill the array with
            
        Returns:
            numpy array/ pytorch tensor if numAngles, numRows, and numCols are all positive, None otherwise
        """
        return self.allocateVolume(val, astensor)
        
    def allocateVolume(self, val=0.0, astensor=False):
        """Allocates reconstruction volume data
        
        It is not necessary to use this function. It is included simply for convenience.

        Args:
            val (float): value to fill the array with
            
        Returns:
            numpy array/ pytorch tensor if numAngles, numRows, and numCols are all positive, None otherwise
        """
        dim1, dim2, dim3 = self.get_volume_dim()
        if dim1 > 0 and dim2 > 0 and dim3 > 0:
            if val == 0.0:
                f = np.ascontiguousarray(np.zeros((dim1,dim2,dim3),dtype=np.float32), dtype=np.float32)
            else:
                f = np.ascontiguousarray(val*np.ones((dim1,dim2,dim3),dtype=np.float32), dtype=np.float32)
            if has_torch and astensor:
                f = torch.from_numpy(f)
            return f
        else:
            return None
            
    def allocateVolume_gpu(self, val=0.0):
        """Allocates reconstruction volume data as a pytorch tensor on the gpu
        
        It is not necessary to use this function. It is included simply for convenience.

        Args:
            val (float): value to fill the array with
            
        Returns:
            pytorch tensor if numAngles, numRows, and numCols are all positive, None otherwise
        """
        if has_torch == False:
            print('Error: you must have pytorch installed for this function')
            return None
            
        dim1, dim2, dim3 = self.get_volume_dim()
        if dim1 > 0 and dim2 > 0 and dim3 > 0:
            if val == 0.0:
                f = np.ascontiguousarray(np.zeros((dim1,dim2,dim3),dtype=np.float32), dtype=np.float32)
            else:
                f = np.ascontiguousarray(val*np.ones((dim1,dim2,dim3),dtype=np.float32), dtype=np.float32)
            if has_torch:
                #f = torch.from_numpy(f).float().to(max(0,self.get_gpu()))
                f = self.copy_to_device(f)
            return f
        else:
            return None
            
    def allocateData(self, x, val=0.0):
        if has_torch == True and type(x) is torch.Tensor:
            if x.is_cuda:
                if val == 0.0:
                    y = torch.zeros([x.shape[0], x.shape[1], x.shape[2]], dtype=torch.float32, device=torch.device('cuda:'+str(self.get_gpu())))
                else:
                    y = val*torch.ones([x.shape[0], x.shape[1], x.shape[2]], dtype=torch.float32, device=torch.device('cuda:'+str(self.get_gpu())))
            else:
                if val == 0.0:
                    y = torch.zeros([x.shape[0], x.shape[1], x.shape[2]], dtype=torch.float32)
                else:
                    y = val*torch.ones([x.shape[0], x.shape[1], x.shape[2]], dtype=torch.float32)
        else:
            if val == 0.0:
                y = np.ascontiguousarray(np.zeros((x.shape[0], x.shape[1], x.shape[2]),dtype=np.float32), dtype=np.float32)
            else:
                y = np.ascontiguousarray(val*np.ones((x.shape[0], x.shape[1], x.shape[2]),dtype=np.float32), dtype=np.float32)
        return y
            
    def setAngleArray(self,numAngles,angularRange):
        """Sets the angle array for equi-spaced projection angles, i.e., phis which specifies the projection angles for parallel-, fan-, and cone-beam data
        
        It is not necessary to use this function. It is included simply for convenience.
        LEAP allows one to specify non-equispaced projection angles for all geometries
        If one wishes to do this, do not use this function, but specify them yourself in a numpy array as an argument to set_parallelbeam, set_fanbeam, or set_coneBeam.
        In any case, angles must be strictly monotonic increasing or monotonic decreasing.
        If doing multiple revolutions, angles should just keep going past 360.0.

        Args:
            numAngles (int): number of projections
            angularRange (float): the angular range of the projection angles (degrees)
            
        Returns:
            numpy array of the projection angles (in degrees)
        """
        return np.array(range(numAngles)).astype(np.float32) * angularRange/float(numAngles)
        
    def copyData(self, x):
        if has_torch == True and type(x) is torch.Tensor:
            x_copy = x.clone()
            #x_copy.to(x.get_device())
            return x_copy
        else:
            return x.copy()
            
    def copy_to_device(self, x):
        """Copies the given argument to a torch tensor on the primary gpu
        
        This function is just provided for convenience and is not necessary for any LEAP process.
        
        Args:
            x (numpy array or torch tensor): data to copy to device number defined by get_gpu()
            
        Returns:
            torch tensor on the specified GPU
        
        """
        if has_torch == False:
            print('Error: pytorch not installed')
            return x
        else:
            device = torch.device("cuda:" + str(self.get_gpu()))
            if type(x) is torch.Tensor:
                x = x.to(device)
            else:
                x = torch.from_numpy(x).to(device)
            return x
            
    def copy_to_host(self, x):
        """Copies the given argument to a numpy array (on the cpu)"""
        if has_torch == True and type(x) is torch.Tensor:
            x = x.cpu().detach().numpy()
        return x
            
            
    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS EXECUTE THE MAIN CPU/GPU ROUTINES IN LEAP
    ###################################################################################################################
    ###################################################################################################################
    def project(self, g, f, param_id=None):
        """Calculate the forward projection of f and stores the result in g
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument g and returns the same g.
        Returning g is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            param_id (int): optional parameter to project a particular parameter index
            
        Returns:
            g, the same as the input with the same name
        """
        #self.apply_volume_mask(f)
        if self.volume_mask_is_valid(f):
            self.libprojectors.project_with_mask.restype = ctypes.c_bool
            self.set_model(param_id)
            if has_torch == True and type(g) is torch.Tensor:
                self.libprojectors.project_with_mask.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
                self.libprojectors.project_with_mask(g.data_ptr(), f.data_ptr(), self.volume_mask.data_ptr(), g.is_cuda == False)
            else:
                self.libprojectors.project_with_mask.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
                self.libprojectors.project_with_mask(g, f, self.volume_mask, True)
            return g
        else:
            self.libprojectors.project.restype = ctypes.c_bool
            self.set_model(param_id)
            if has_torch == True and type(g) is torch.Tensor:
                self.libprojectors.project.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
                self.libprojectors.project(g.data_ptr(), f.data_ptr(), g.is_cuda == False)
            else:
                self.libprojectors.project.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
                self.libprojectors.project(g, f, True)
            return g
        
    def project_cpu(self, g, f, param_id=None):
        """Calculate the forward projection of f and stores the result in g
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument g and returns the same g.
        Returning g is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data on the CPU
            f (C contiguous float32 numpy array or torch tensor): volume data on the CPU
            param_id (int): optional parameter to project a particular parameter index
            
        Returns:
            g, the same as the input with the same name
        """
        if self.volume_mask_is_valid(f):
            self.libprojectors.project_with_mask_cpu.restype = ctypes.c_bool
            self.set_model(param_id)
            if has_torch == True and type(g) is torch.Tensor:
                if g.is_cuda:
                    print('Error: project_cpu requires that the data be on the CPU')
                else:
                    self.libprojectors.project_with_mask_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                    self.libprojectors.project_with_mask_cpu(g.data_ptr(), f.data_ptr(), self.volume_mask.data_ptr())
            else:
                self.libprojectors.project_with_mask_cpu.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
                self.libprojectors.project_with_mask_cpu(g, f, self.volume_mask)
            return g
        else:
            self.libprojectors.project_cpu.restype = ctypes.c_bool
            self.set_model(param_id)
            if has_torch == True and type(g) is torch.Tensor:
                if g.is_cuda:
                    print('Error: project_cpu requires that the data be on the CPU')
                else:
                    self.libprojectors.project_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                    self.libprojectors.project_cpu(g.data_ptr(), f.data_ptr())
            else:
                self.libprojectors.project_cpu.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
                self.libprojectors.project_cpu(g, f)
            return g
        
    def project_gpu(self, g, f, param_id=None):
        """Calculate the forward projection of f and stores the result in g
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument g and returns the same g.
        Returning g is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 torch tensor): projection data on a GPU
            f (C contiguous float32 torch tensor): volume data on a GPU
            param_id (int): optional parameter to project a particular parameter index
            
        Returns:
            g, the same as the input with the same name
        """
        if self.volume_mask_is_valid(f):
            self.libprojectors.project_with_mask_gpu.restype = ctypes.c_bool
            self.set_model(param_id)
            if has_torch == True and type(g) is torch.Tensor:
                if g.is_cuda == False:
                    print('Error: project_gpu requires that the data be on the GPU')
                else:
                    self.libprojectors.project_with_mask_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
                    self.libprojectors.project_with_mask_gpu(g.data_ptr(), f.data_ptr(), self.volume_mask.data_ptr())
            else:
                print('Error: project_gpu requires that the data be pytorch tensors on the GPU')
            return g
        else:
            self.libprojectors.project_gpu.restype = ctypes.c_bool
            self.set_model(param_id)
            if has_torch == True and type(g) is torch.Tensor:
                if g.is_cuda == False:
                    print('Error: project_gpu requires that the data be on the GPU')
                else:
                    self.libprojectors.project_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                    self.libprojectors.project_gpu(g.data_ptr(), f.data_ptr())
            else:
                print('Error: project_gpu requires that the data be pytorch tensors on the GPU')
            return g
            
    def backproject(self, g, f, param_id=None):
        """Calculate the backprojection (adjoint of the forward projection) of g and stores the result in f
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            param_id (int): optional parameter to project a particular parameter index
            
        Returns:
            f, the same as the input with the same name
        """
        self.libprojectors.backproject.restype = ctypes.c_bool
        self.set_model(param_id)
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.backproject.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            self.libprojectors.backproject(g.data_ptr(), f.data_ptr(), g.is_cuda == False)
        else:
            self.libprojectors.backproject.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.backproject(g, f, True)
        self.apply_volume_mask(f)
        return f
        
    def backproject_cpu(self, g, f, param_id=None):
        """Calculate the backprojection (adjoint of the forward projection) of g and stores the result in f
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            param_id (int): optional parameter to project a particular parameter index
            
        Returns:
            f, the same as the input with the same name
        """
        self.libprojectors.backproject_cpu.restype = ctypes.c_bool
        self.set_model(param_id)
        if has_torch == True and type(g) is torch.Tensor:
            if g.is_cuda:
                print('Error: backproject_cpu requires that the data be on the CPU')
            else:
                self.libprojectors.backproject_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                self.libprojectors.backproject_cpu(g.data_ptr(), f.data_ptr())
                self.apply_volume_mask(f)
        else:
            self.libprojectors.backproject_cpu.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.backproject_cpu(g, f)
            self.apply_volume_mask(f)
        return f
        
    def backproject_gpu(self, g, f, param_id=None):
        """Calculate the backprojection (adjoint of the forward projection) of g and stores the result in f
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 torch tensor): projection data
            f (C contiguous float32 torch tensor): volume data
            param_id (int): optional parameter to project a particular parameter index
            
        Returns:
            f, the same as the input with the same name
        """
        self.libprojectors.backproject_gpu.restype = ctypes.c_bool
        self.set_model(param_id)
        if has_torch == True and type(g) is torch.Tensor:
            if g.is_cuda == False:
                print('Error: backproject_gpu requires that the data be on the GPU')
            else:
                self.libprojectors.backproject_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
                self.libprojectors.backproject_gpu(g.data_ptr(), f.data_ptr())
                self.apply_volume_mask(f)
        else:
            print('Error: backproject_gpu requires that the data be pytorch tensors on the GPU')
        return f
        
    def filterProjections(self, g, g_out=None):
        r"""Filters the projection data, g, so that its (weighted) backprojection results in an FBP reconstruction.
        
        More specifically, the same results as the FBP function can be achieved by running the following functions
        
        .. line-block::
           filterProjections(g)
           weightedBackproject(g,f)
        
        The CT geometry parameters must be set prior to running this function.
        This function take the argument g and returns the same g.
        Returning g is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            
        Returns:
            g, the same as the input with the same name
        """
        
        if g_out is None:
            if self.extraColumnsForOffsetScan() > 0:
                if has_torch == True and type(g) is torch.Tensor:
                    if g.is_cuda:
                        g_out = self.allocateProjections_gpu(forOffsetScanFilteringStep=True)
                    else:
                        g_out = self.allocate_projections(astensor=True, forOffsetScanFilteringStep=True)
                else:
                    g_out = self.allocate_projections(forOffsetScanFilteringStep=True)
            else:
                g_out = g
        
        self.libprojectors.filterProjections.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.filterProjections.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            self.libprojectors.filterProjections(g.data_ptr(), g_out.data_ptr(), g.is_cuda == False)
        else:
            self.libprojectors.filterProjections.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.filterProjections(g, g_out, True)
        return g_out
        
    def filterProjections_cpu(self, g):
        self.libprojectors.filterProjections_cpu.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            if g.is_cuda:
                print('Error: filterProjections_cpu requires the input be on the cpu')
            else:
                self.libprojectors.filterProjections_cpu.argtypes = [ctypes.c_void_p]
                self.libprojectors.filterProjections_cpu(g.data_ptr())
        else:
            self.libprojectors.filterProjections_cpu.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.filterProjections_cpu(g)
        return g
        
    def filterProjections_gpu(self, g):
        self.libprojectors.filterProjections_gpu.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor and g.is_cuda == True:
            self.libprojectors.filterProjections_gpu.argtypes = [ctypes.c_void_p]
            self.libprojectors.filterProjections_gpu(g.data_ptr())
        else:
            print('Error: filterProjections_gpu requires the input be a torch tensor on a gpu')
        return g

    def preRampFiltering(self, g):
        r"""Applying pre-ramp filter weighting to the projection data, g, for FBP reconstruction
        
        More specifically, the same results as the FBP function can be achieved by running the following functions
        
        .. line-block::
           preRampFiltering(g)
           rampFilterProjections(g, get_FBPscalar())
           postRampFiltering(g)
           weightedBackproject(g,f)
        
        The CT geometry parameters must be set prior to running this function.
        This function take the argument g and returns the same g.
        Returning g is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            
        Returns:
            g, the same as the input with the same name
        """
        self.libprojectors.preRampFiltering.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.preRampFiltering.argtypes = [ctypes.c_void_p, ctypes.c_bool]
            self.libprojectors.preRampFiltering(g.data_ptr(), g.is_cuda == False)
        else:
            self.libprojectors.preRampFiltering.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.preRampFiltering(g, True)
        return g

    def postRampFiltering(self, g):
        r"""Applying post-ramp filter weighting to the projection data, g, for FBP reconstruction
        
        More specifically, the same results as the FBP function can be achieved by running the following functions
        
        .. line-block::
           preRampFiltering(g)
           rampFilterProjections(g, get_FBPscalar())
           postRampFiltering(g)
           weightedBackproject(g,f)
        
        The CT geometry parameters must be set prior to running this function.
        This function take the argument g and returns the same g.
        Returning g is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            
        Returns:
            g, the same as the input with the same name
        """
        self.libprojectors.postRampFiltering.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.postRampFiltering.argtypes = [ctypes.c_void_p, ctypes.c_bool]
            self.libprojectors.postRampFiltering(g.data_ptr(), g.is_cuda == False)
        else:
            self.libprojectors.postRampFiltering.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.postRampFiltering(g, True)
        return g
        
    def rampFilterProjections(self, g, scalar=1.0):
        """Applies the ramp filter to the projection data, g, which is a subset of the operations in the filterProjections function.
        
        The CT geometry parameters must be set prior to running this function.
        This function take the argument g and returns the same g.
        Returning g is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            scalar: optional scalar to be applied (defaults to 1.0)
            
        Returns:
            g, the same as the input with the same name
        """
        self.libprojectors.rampFilterProjections.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.rampFilterProjections.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_float]
            self.libprojectors.rampFilterProjections(g.data_ptr(), g.is_cuda == False, scalar)
        else:
            self.libprojectors.rampFilterProjections.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_float]
            self.libprojectors.rampFilterProjections(g, True, scalar)
        return g
        
    def HilbertFilterProjections(self, g):
        """Applies the Hilbert filter to the projection data, g, which is a subset of the operations in some reconstruction algorithms
        
        The CT geometry parameters must be set prior to running this function.
        This function take the argument g and returns the same g.
        Returning g is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            
        Returns:
            g, the same as the input with the same name
        """
        self.libprojectors.HilbertFilterProjections.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.HilbertFilterProjections.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_float]
            self.libprojectors.HilbertFilterProjections(g.data_ptr(), g.is_cuda == False, 1.0)
        else:
            self.libprojectors.HilbertFilterProjections.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool, ctypes.c_float]
            self.libprojectors.HilbertFilterProjections(g, True, 1.0)
        return g
    
    def weightedBackproject(self,g,f):
        """Calculate the weighted backprojection of g and stores the result in f
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        If preceeded by the filterProjections function, this function can produce and FBP reconstruction.
        Some geometries require a weighted backprojection for FBP reconstruction,
        such as fan-beam, helical cone-beam, Attenuated Radon Transform, and symmetric objects.
        Those geometries do not require a weighted backprojection (i.e., backprojection would suffice)
        for FBP reconstruction, we still recommend using this function to perform two-step FBP reconstructions
        because it performs some subtle operations that are only appropriate for FBP reconstruction; for example,
        using extrapolation in the row direction for axial cone-beam FBP (FDK).
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            
        Returns:
            f, the same as the input with the same name
        """
        self.libprojectors.weightedBackproject.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.weightedBackproject.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            self.libprojectors.weightedBackproject(g.data_ptr(), f.data_ptr(), g.is_cuda == False)
        else:
            self.libprojectors.weightedBackproject.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.weightedBackproject(g, f, True)
        self.apply_volume_mask(f)
        return f
        
    def rampFilterVolume(self, f):
        """Applies the 2D ramp filter to the volume data, f, for each z-slice
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): volume data
            
        Returns:
            f, the same as the input with the same name
        """
        
        if self.include_cufft():
            self.libprojectors.rampFilterVolume.restype = ctypes.c_bool
            self.set_model()
            if has_torch == True and type(f) is torch.Tensor:
                self.libprojectors.rampFilterVolume.argtypes = [ctypes.c_void_p, ctypes.c_bool]
                self.libprojectors.rampFilterVolume(f.data_ptr(), f.is_cuda == False)
            else:
                self.libprojectors.rampFilterVolume.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
                self.libprojectors.rampFilterVolume(f, True)
            return f
        else:
            # Get volume dimensions
            N_z = f.shape[0]
            N_y = f.shape[1]
            N_x = f.shape[2]
            
            # Calculate optimal FFT size
            N_Y = self.optimalFFTsize(2*max(N_y,N_x))
            N_X = N_Y

            # Set frequency samples
            T = 2.0 * np.pi / float(N_X);
            Y = np.fft.fftshift((np.array(range(N_Y))-0.5*N_Y)*T)
            X = np.fft.fftshift((np.array(range(N_X))-0.5*N_X)*T)
            Y,X = np.meshgrid(Y,X)
                
            # Set 2D ramp filter frequency response
            H_X = 2*np.abs(np.sin(0.5*X))
            H_Y = 2*np.abs(np.sin(0.5*Y))
            H_2D = np.sqrt(H_X**2 + H_Y**2 - (0.5*H_X*H_Y)**2) / self.get_voxelWidth()
            H = np.repeat(np.expand_dims(H_2D, axis=0), N_z, axis=0)
            
            # Apply ramp filter
            if has_torch and type(f) is torch.Tensor:
                #print('using torch')
                H = torch.from_numpy(H).to(f.get_device())
                f[:,:,:] = torch.real(torch.fft.ifftn(torch.fft.fftn(f, s=(N_Y, N_X), dim=(1,2))*H, dim=(1,2)))[:,0:N_y,0:N_x]
            else:
                #print('using numpy')
                f[:,:,:] = np.real(np.fft.ifftn(np.fft.fftn(f, s=(N_Y, N_X), axes=(1,2))*H, axes=(1,2)))[:,0:N_y,0:N_x]
            
            return f
        
    def Laplacian(self, g, numDims=2, smoothLaplacian=False):
        """Applies a Laplacian operation to each projection
        
        The CT geometry parameters must be set prior to running this function.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            numDims (int): if 2 calculates the sum of the second derivatives along the rows and columns, if 1 calculates the second derivative along the columns
            smoothLaplacian (bool): if true, applies an extra low pass filter (given by [0.25, 0.5, 0.25]) to smooth the result
            
        Returns:
            g, the same as the input
        
        """
        self.libprojectors.Laplacian.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.Laplacian.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
            self.libprojectors.Laplacian(g.data_ptr(), numDims, smoothLaplacian, g.is_cuda == False)
        else:
            self.libprojectors.Laplacian.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
            self.libprojectors.Laplacian(g, numDims, smoothLaplacian, True)
        return g
        
    def transmission_filter(self, g, H, isAttenuationData=True):
        """Applies a 2D Filter to each transmission projection
        
        The CT geometry parameters must be set prior to running this function.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            H (C contiguous float32 2D numpy array): frequency response of filter (must be real-valued)
            isAttenuationData (bool): true if the input data is attenuation data, false if it is not (e.g., transmission, raw)
        
        Returns:
            g, the same as the input
        
        """
        self.libprojectors.transmissionFilter.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            if has_torch == True and type(H) is torch.Tensor:
                H = H.cpu().detach().numpy()
            self.libprojectors.transmissionFilter.argtypes = [ctypes.c_void_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
            self.libprojectors.transmissionFilter(g.data_ptr(), H, H.shape[0], H.shape[1], isAttenuationData, g.is_cuda == False)
        else:
            self.libprojectors.transmissionFilter.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
            self.libprojectors.transmissionFilter(g, H, H.shape[0], H.shape[1], isAttenuationData, True)
        return g
        
    def AzimuthalBlur(self, f, FWHM):
        """Applies an low pass filter to the volume data in the azimuthal direction, f, for each z-slice
        
        The CT volume parameters must be set prior to running this function.
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): volume data
            FWHM (float): full width at half maximum of the filter (in degrees)
            
        Returns:
            f, the same as the input with the same name
        """
        self.libprojectors.AzimuthalBlur.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.AzimuthalBlur.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_bool]
            self.libprojectors.AzimuthalBlur(f.data_ptr(), FWHM, f.is_cuda == False)
        else:
            self.libprojectors.AzimuthalBlur.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_bool]
            self.libprojectors.AzimuthalBlur(f, FWHM, True)
        return f

    def get_FBPscalar(self):
        """Returns the FBP scaling factor
        """
        self.libprojectors.get_FBPscalar.argtypes = []
        self.libprojectors.get_FBPscalar.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_FBPscalar()

    def fbp(self, g, f=None, inplace=False):
        """Alias for FBP"""
        return self.FBP(g, f, inplace)

    def FBP_slice(self, g, islice=None, coord='z'):
        r""" Performs FBP on a single slice
        
        The CT geometry and CT volume parameters must be set prior to running this function.
        This slice index is the index within the current volume specification.
        This function does not change any LEAP parameters.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            islice (int): the index of the slice to reconstruct
            coord (string): specifies which axis to reconstruct: can be 'x', 'y', or 'z'; 'z' is the default
            
        Returns:
            reconstructed slice of the same type as the projection data
        """
        if g is None:
            print('Error: must provide projection data!')
            return None
        
        if coord == 'x':
            ix = islice
            if ix is None or ix < 0 or ix >= self.get_numX():
                ix = self.get_numX()//2
            offsetX_save = self.get_offsetX()
            numX_save = self.get_numX()
            
            self.set_offsetX(self.x_samples()[ix])
            self.set_numX(1)
            
            f = self.FBP(g)
            
            self.set_offsetX(offsetX_save)
            self.set_numX(numX_save)
            
            return f
            
        elif coord == 'y':
            iy = islice
            if iy is None or iy < 0 or iy >= self.get_numY():
                iy = self.get_numY()//2
            offsetY_save = self.get_offsetY()
            numY_save = self.get_numY()
            
            self.set_offsetY(self.y_samples()[iy])
            self.set_numY(1)
            
            f = self.FBP(g)
            
            self.set_offsetY(offsetY_save)
            self.set_numY(numY_save)
            
            return f
            
        else:
            iz = islice
            if iz is None or iz < 0 or iz >= self.get_numZ():
                iz = self.get_numZ()//2
            
            offsetZ_save = self.get_offsetZ()
            numZ_save = self.get_numZ()
            numRows_save = self.get_numRows()
            centerRow_save = self.get_centerRow()
            
            if self.get_geometry() == 'PARALLEL' or self.get_geometry() == 'FAN':
                g_chunk = self.cropProjections([iz, iz], None, g)
                
                #self.set_numRows(rowRange[1]-rowRange[0]+1)
                #self.shift_detector(self.get_pixelHeight()*rowRange[0], 0.0)
                detectorShift = self.get_pixelHeight()*iz
                
            else:
                rowRange = self.rowRangeNeededForBackprojection(iz)
                g_chunk = self.cropProjections(rowRange, None, g)
                self.set_offsetZ(self.z_samples()[iz])
                self.set_numZ(1)
                
                #self.set_numRows(rowRange[1]-rowRange[0]+1)
                #self.shift_detector(self.get_pixelHeight()*rowRange[0], 0.0)
                detectorShift = self.get_pixelHeight()*rowRange[0]

            f = self.FBP(g_chunk, None, True)
            del g_chunk
            
            self.set_offsetZ(offsetZ_save)
            self.set_numZ(numZ_save)
            self.set_numRows(numRows_save)
            self.shift_detector(-detectorShift, 0.0)
            #self.set_centerRow(centerRow_save)
            
            return f

    def FBP(self, g, f=None, inplace=False):
        """Performs a Filtered Backprojection (FBP) reconstruction of the projection data, g, and stores the result in f
        
        This function performs analytic reconstruction (i.e., FBP) of nearly all LEAP geometries: parallel-, fan-, cone-, and (axially-aligned) modular-beam geometries,
        including both flat and curved detectors, axial or helical scans, Attenuated Radon Transform, symmetric object, etc.
        Note that FDK is an FBP-type algorithm, so for simplicity we just called it FBP in LEAP.  The same goes for other analytic reconstructions.
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            inplace(bool): if true, then the filtering operations will be done in-place (i.e., the value in g will be altered) to save on memory usage
            
        Returns:
            f, the same as the input with the same name
        """
        
        # Make a copy of g if necessary
        delete_q = False
        if has_torch == True and type(g) is torch.Tensor:
            if inplace == False:
                q = self.copyData(g)
                delete_q = True
            else:
                q = g
        else:
            if self.get_gpu() < 0 and inplace == False:
                q = self.copyData(g)
                delete_q = True
            else:
                q = g
        
        self.libprojectors.FBP.restype = ctypes.c_bool
        if has_torch == True and type(q) is torch.Tensor:
            if f is None:
                f = self.allocateVolume(0.0,True)
                f = f.to(g.get_device())
            self.libprojectors.FBP.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            self.set_model()
            isSuccessful = self.libprojectors.FBP(q.data_ptr(), f.data_ptr(), q.is_cuda == False)
            if isSuccessful == False:
                #self.filterProjections(q)
                #self.weightedBackproject(q,f)
                return None
        else:
            if f is None:
                f = self.allocateVolume()
            self.libprojectors.FBP.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.set_model()
            isSuccessful = self.libprojectors.FBP(q, f, True)
            if isSuccessful == False:
                #self.filterProjections(q)
                #self.weightedBackproject(q,f)
                return None
        if delete_q:
            del q
        return f
        
    def fbp_cpu(self, g, f, inplace=False):
        return self.FBP_cpu(g, f, inplace)
        
    def fbp_gpu(self, g, f, inplace=False):
        return self.FBP_gpu(g, f, inplace)
        
    def FBP_cpu(self, g, f, inplace=False):
        """Performs a Filtered Backprojection (FBP) reconstruction of the projection data, g, and stores the result in f
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            inplace(bool): if true, then the filtering operations will be done in-place (i.e., the value in g will be altered) to save on memory usage
            
        Returns:
            f, the same as the input with the same name
        """
        
        # First make validation checks that the data is on the CPU
        if has_torch == True and type(g) is torch.Tensor and g.is_cuda == True:
            print('Error: FBP_cpu requires that the data be on the CPU')
            return f

        # Make a copy of g if necessary
        delete_q = False
        if inplace == False:
            q = self.copyData(g)
            delete_q = True
        else:
            q = g
            
        self.libprojectors.FBP_cpu.restype = ctypes.c_bool
        if has_torch == True and type(q) is torch.Tensor:
            self.libprojectors.FBP_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.set_model()
            self.libprojectors.FBP_cpu(q.data_ptr(), f.data_ptr())
        else:
            self.libprojectors.FBP_cpu.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.set_model()
            self.libprojectors.FBP_cpu(q, f)
        if delete_q:
            del q
        return f
        
    def FBP_gpu(self, g, f, inplace=False):
        """Performs a Filtered Backprojection (FBP) reconstruction of the projection data, g, and stores the result in f
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 torch tensor): projection data
            f (C contiguous float32 torch tensor): volume data
            inplace(bool): if true, then the filtering operations will be done in-place (i.e., the value in g will be altered) to save on memory usage
            
        Returns:
            f, the same as the input with the same name
        """
        
        # First make validation checks that the data is on the GPU
        if has_torch == True and type(g) is torch.Tensor:
            if g.is_cuda == False:
                print('Error: FBP_gpu requires that the data be on the GPU')
                return f
        else:
            print('Error: FBP_gpu requires that the data be pytorch tensors on the GPU')
            return f

        # Make a copy of g if necessary
        delete_q = False
        if inplace == False:
            q = self.copyData(g)
            delete_q = True
        else:
            q = g
        
        if has_torch == True and type(q) is torch.Tensor:
            self.libprojectors.FBP_gpu.restype = ctypes.c_bool
            self.libprojectors.FBP_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            self.set_model()
            self.libprojectors.FBP_gpu(q.data_ptr(), f.data_ptr())
        if delete_q:
            del q
        return f
    
    def fbp_adjoint(self, g, f):
        return self.FBP_adjoint(g, f)
    
    def FBP_adjoint(self, g, f):
        """Performs the adjoint of the Filtered Backprojection (FBP) reconstruction of the volume data, f, and stores the result in g
        
        This function performs the adjoint of analytic reconstruction (i.e., adjoint of FBP) of nearly all LEAP geometries: parallel-, fan-, cone-, and (axially-aligned) modular-beam geometries,
        including both flat and curved detectors, axial or helical scans, Attenuated Radon Transform, symmetric object, etc.
        Note that FDK is an FBP-type algorithm, so for simplicity we just called it FBP in LEAP.  The same goes for other analytic reconstructions.
        
        This function will not provide an exact adjoint of FBP for fan-beam or helical scans.
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument g and returns the same g.
        Returning g is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            
        Returns:
            g, the same as the input with the same name
        """
        self.project(g,f)
        self.filterProjections(g)
        return g
        
    def fbp_adjoint_cpu(self, g, f):
        # First make validation checks that the data is on the CPU
        if has_torch == True and type(f) is torch.Tensor and f.is_cuda == True:
            print('Error: fbp_adjoint_cpu requires that the data be on the CPU')
            return g

        self.project_cpu(g,f)
        self.filterProjections_cpu(g)
        return g
        
    def fbp_adjoint_gpu(self, g, f):
        # First make validation checks that the data is on the GPU
        if has_torch == True and type(f) is torch.Tensor:
            if f.is_cuda == False:
                print('Error: fbp_adjoint_gpu requires that the data be on the GPU')
                return g
        else:
            print('Error: fbp_adjoint_gpu requires that the data be pytorch tensors on the GPU')
            return g

        self.project_gpu(g,f)
        self.filterProjections_gpu(g)
        return g
    
    def LT(self, g, f=None, inplace=False):
        """Performs a Lambda/Local Tomography (LT) reconstruction of the projection data, g, and stores the result in f
        
        This function performs Lambda/Local Tomography (LT) reconstruction of nearly all LEAP geometries: parallel-, fan-, cone-, and (axially-aligned) modular-beam geometries,
        including both flat and curved detectors, axial or helical scans, Attenuated Radon Transform, symmetric object, etc.
        LT reconstructions work even when the projections are truncated and reconstruct the 2D ramp filtered volume which is essentially an edge map.
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            inplace(bool): if true, then the filtering operations will be done in-place (i.e., the value in g will be altered) to save on memory usage
            
        Returns:
            f, the same as the input with the same name
        """
        
        # Make a copy of g if necessary
        if has_torch == True and type(g) is torch.Tensor:
            if inplace == False:
                q = self.copyData(g)
            else:
                q = g
        else:
            if self.get_gpu() < 0 and inplace == False:
                q = self.copyData(g)
            else:
                q = g
        
        self.libprojectors.lambdaTomography.restype = ctypes.c_bool
        if has_torch == True and type(q) is torch.Tensor:
            if f is None:
                f = self.allocateVolume(0.0,True)
                f = f.to(g.get_device())
            self.libprojectors.lambdaTomography.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            self.set_model()
            self.libprojectors.lambdaTomography(q.data_ptr(), f.data_ptr(), q.is_cuda == False)
        else:
            if f is None:
                f = self.allocateVolume()
            self.libprojectors.lambdaTomography.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.set_model()
            self.libprojectors.lambdaTomography(q, f, True)
        return f
    
    def inconsistencyReconstruction(self, g, f=None, inplace=False):
        """Performs an Inconsistency Reconstruction of the projection data, g, and stores the result in f
        
        An Inconsistency Reconstruction is an FBP reconstruction except it replaces the ramp filter with
        a derivative.  For scans with angular ranges of 360 or more this will result in a pure noise
        reconstruction if the geometry is calibrated and there are no biases in the data.  This can
        be used as a robust way to find the centerCol parameter or estimate detector tilt.
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            inplace(bool): if true, then the filtering operations will be done in-place (i.e., the value in g will be altered) to save on memory usage
            
        Returns:
            f, the same as the input with the same name
        """
        
        # Make a copy of g if necessary
        if inplace == False:
            q = self.copyData(g)
        else:
            q = g
        
        self.libprojectors.inconsistencyReconstruction.restype = ctypes.c_bool
        if has_torch == True and type(q) is torch.Tensor:
            if f is None:
                f = self.allocateVolume(0.0,True)
                f = f.to(g.get_device())
            self.libprojectors.inconsistencyReconstruction.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            self.set_model()
            self.libprojectors.inconsistencyReconstruction(q.data_ptr(), f.data_ptr(), q.is_cuda == False)
        else:
            if f is None:
                f = self.allocateVolume()
            self.libprojectors.inconsistencyReconstruction.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.set_model()
            self.libprojectors.inconsistencyReconstruction(q, f, True)
        return f
        
    def BPF(self, g, f):
        """Performs a Backprojection Filtration (BPF) reconstruction of the projection data, g, and stores the result in f
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function take the argument f and returns the same f.
        Returning f is just there for nesting several algorithms.
        This reconstruction only works for parallel-beam data
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): projection data
            f (C contiguous float32 numpy array or torch tensor): volume data
            
        Returns:
            f, the same as the input with the same name
        """
        if self.get_geometry() == 'PARALLEL':
            self.backproject(g,f)
            self.rampFilterVolume(f)
            f *= self.get_FBPscalar()
            return f
        else:
            return None
        
    def sensitivity(self, f=None):
        """Performs a calculation of the sensitivity of a CT geometry, i.e., the backprojection of data that is all ones
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        One can get the same result by backprojecting an array of projection data with all entries equal to one.
        The benefit of this function is that it is faster and uses less memory.
        
        In a volume is provided, the result will be stored there, otherwise a new volume will be allocated.
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): (optional argument) volume data to store the result
            
        Returns:
            f, the same as the input
        """
        self.libprojectors.sensitivity.restype = ctypes.c_bool
        if has_torch == True and f is not None and type(f) is torch.Tensor:
            #if f is None:
            #    f = self.allocateVolume(0.0, True)
            self.libprojectors.sensitivity.argtypes = [ctypes.c_void_p, ctypes.c_bool]
            self.set_model()
            self.libprojectors.sensitivity(f.data_ptr(), f.is_cuda == False)
        else:
            if f is None:
                f = self.allocateVolume()
            self.libprojectors.sensitivity.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.set_model()
            self.libprojectors.sensitivity(f, True)
        return f
    
    def windowFOV(self, f):
        """Sets all voxels outside the field of view to zero
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): volume data to operate on
            
        Returns:
            f, the same as the input
        """
        self.libprojectors.windowFOV.restype = ctypes.c_bool
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.windowFOV.argtypes = [ctypes.c_void_p, ctypes.c_bool]
            self.set_model()
            self.libprojectors.windowFOV(f.data_ptr(), f.is_cuda == False)
        else:
            self.libprojectors.windowFOV.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.set_model()
            self.libprojectors.windowFOV(f, True)
        return f
    
    def rowRangeNeededForBackprojection(self, iz=None):
        r"""Calculates the detector rows necessary to reconstruct the current volume specification
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        For anything but cone-beam or axially-aligned modular-beam data this function will return np.array([0, numRows-1]).
        For cone-beam or axially-aligned modular-beam data, the function can be used to inform the user of the only detector row indies that
        are necessary to reconstruct the volume.  This can be used to reduce the input data size which can
        be important to speed up calculations or reduce the CPU and/or GPU memory necessary to perform reconstruction.
        
        Returns:
            rowsNeeded, a 2X1 numpy array where the values are the first and last detector row index needed to reconstruct the volume.
        
        """
        
        rowsNeeded = np.zeros(2,dtype=np.int32)
        rowsNeeded[1] = self.get_numRows()-1
        
        if self.get_geometry() == 'PARALLEL' or self.get_geometry() == 'FAN':
            if iz is not None:
                if iz < 0 or iz >= self.get_numZ():
                    iz = self.get_numZ()//2
                rowsNeeded[0] = iz
                rowsNeeded[1] = iz
            return rowsNeeded
                
        self.libprojectors.rowRangeNeededForBackprojection.argtypes = [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
        self.libprojectors.rowRangeNeededForBackprojection.restype = ctypes.c_bool
        if iz is None:
            self.set_model()
            self.libprojectors.rowRangeNeededForBackprojection(rowsNeeded)
            return rowsNeeded
        else:
            if iz < 0 or iz >= self.get_numZ():
                iz = self.get_numZ()//2
            offsetZ_save = self.get_offsetZ()
            numZ_save = self.get_numZ()
            
            self.set_offsetZ(self.z_samples()[iz])
            self.set_numZ(1)
            
            self.set_model()
            self.libprojectors.rowRangeNeededForBackprojection(rowsNeeded)
            
            self.set_offsetZ(offsetZ_save)
            self.set_numZ(numZ_save)
            
            return rowsNeeded
        
    def viewRangeNeededForBackprojection(self):
        r"""Calculates the detector projections necessary to reconstruct the current volume specification
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        For anything but helical cone-beam data this function will return np.array([0, numAngles-1]).
        For helical cone-beam data, the function can be used to inform the user of the only projection angle indies that
        are necessary to reconstruct the volume.  This can be used to reduce the input data size which can
        be important to speed up calculations or reduce the CPU and/or GPU memory necessary to perform reconstruction.
        
        Returns:
            viewsNeeded, a 2X1 numpy array where the values are the first and last projection index needed to reconstruct the volume.
        
        """
        viewsNeeded = np.zeros(2,dtype=np.int32)
        viewsNeeded[1] = self.get_numAngles()-1
        self.libprojectors.viewRangeNeededForBackprojection.argtypes = [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
        self.libprojectors.viewRangeNeededForBackprojection.restype = ctypes.c_bool
        self.set_model()
        self.libprojectors.viewRangeNeededForBackprojection(viewsNeeded)
        return viewsNeeded
        
    def numRowsRequiredForBackprojectingSlab(self, numSlicesPerChunk):
        r"""Determines the maximum number of detector rows required to backproject a sub-volume with a specified number of z-slices
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            numSlicesPerChunk (int): the number of z-slices in a sub-volume
            
        Returns:
            the maximum number of rows required to backproject a sub-volume of the specified number of slices
        """
        self.libprojectors.numRowsRequiredForBackprojectingSlab.argtypes = [ctypes.c_int]
        self.libprojectors.numRowsRequiredForBackprojectingSlab.restype = ctypes.c_bool
        return self.libprojectors.numRowsRequiredForBackprojectingSlab(numSlicesPerChunk)
        
    def sliceRangeNeededForProjection(self, doClip=True):
        r"""Calculates the volume z-slices necessary to forward project the view seen by the detectors
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        For anything but cone-beam or axially-aligned modular-beam data this function will return np.array([0, numZ-1]).
        For cone-beam or axially-aligned modular-beam data, the function can be used to inform the user of the only z-slices indices that
        are necessary to forward project the volume (does not include slices which would fall outside all detectors.
        This can be used to reduce the volume data size which can
        be important to speed up calculations or reduce the CPU and/or GPU memory necessary to perform reconstruction.

        Args:
            doClip (boolean): if True, clips the return values to be between 0 and numZ-1
        
        Returns:
            slicesNeeded, a 2X1 numpy array where the values are the first and last z-slice index needed to project the volume.
        
        """
        slicesNeeded = np.zeros(2,dtype=np.int32)
        slicesNeeded[1] = self.get_numZ()-1
        self.libprojectors.sliceRangeNeededForProjection.argtypes = [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_bool]
        self.libprojectors.sliceRangeNeededForProjection.restype = ctypes.c_bool
        self.set_model()
        self.libprojectors.sliceRangeNeededForProjection(slicesNeeded, doClip)
        return slicesNeeded
        
    def cropCols(self, colRange, g=None):
        return self.crop_cols(colRange, g)
        
    def crop_cols(self, colRange, g=None):
        """Crops columns from projection data
        
        This function crops columns from the projection data.
        The appropriate CT geometry parameters are updated (e.g., centerCol, numCols, moduleCenters, etc)
        and if the input projection data is given, the selected projection columns are removed
        
        Args:
            colRange (2-element numpy array): the range of detector column indices to keep, all other columns will be removed
            g (C contiguous float32 numpy array or torch tensor): projection data to operate on (optional)
            
        Returns:
            If g is given, then a new numpy array is returned with the selected columns (the 3rd dimension) removed from the data
        
        """
        if colRange is None:
            return None
        if len(colRange) != 2 or colRange[0] < 0 or colRange[1] < colRange[0] or colRange[1] > self.get_numCols()-1:
            print('Error: cropCols invalid argument!')
            return None
        numCols = self.get_numCols()
        self.set_numCols(colRange[1]-colRange[0]+1)
        self.shift_detector(0.0, self.get_pixelWidth()*colRange[0])
        if g is not None:

            dim3 = colRange[1]-colRange[0]+1        
            if has_torch == True and type(g) is torch.Tensor:
                if g.is_cuda:
                    g_crop = torch.zeros([g.shape[0], g.shape[1], dim3], dtype=torch.float32, device=torch.device('cuda:'+str(self.get_gpu())))
                else:
                    g_crop = torch.zeros([g.shape[0], g.shape[1], dim3], dtype=torch.float32)
                g_crop[:,:,:] = g[:, :, colRange[0]:colRange[1]+1]
            else:
                g_crop = np.empty([g.shape[0], g.shape[1], dim3], dtype=np.float32)
                g_crop[:,:,:] = g[:, :, colRange[0]:colRange[1]+1]
        else:
            g_crop = None
        return g_crop
    
    def cropRows(self, rowRange, g=None):
        return self.crop_rows(rowRange, g)
    
    def crop_rows(self, rowRange, g=None):
        """Crops rows from projection data
        
        This function crops rows from the projection data.
        The appropriate CT geometry parameters are updated (e.g., centerRow, numRows, moduleCenters, etc)
        and if the input projection data is given, the selected projection rows are removed
        
        Args:
            rowRange (2-element numpy array): the range of detector column indices to keep, all other rows will be removed
            g (C contiguous float32 numpy array or torch tensor): projection data to operate on (optional)
            
        Returns:
            If g is given, then a new numpy array is returned with the selected rows (the 2nd dimension) removed from the data
        
        """
        if rowRange is None:
            return None
        if len(rowRange) != 2 or rowRange[0] < 0 or rowRange[1] < rowRange[0] or rowRange[1] > self.get_numRows()-1:
            print('Error: cropRows invalid argument!')
            return None
        numRows = self.get_numRows()
        self.set_numRows(rowRange[1]-rowRange[0]+1)
        self.shift_detector(self.get_pixelHeight()*rowRange[0], 0.0)
        if self.get_geometry() == 'PARALLEL' or self.get_geometry() == 'FAN':
            self.set_numZ(self.get_numRows())
            self.set_offsetZ(0.0)
        if g is not None:
        
            if has_torch == True and type(g) is torch.Tensor:
                dim2 = rowRange[1]-rowRange[0]+1
                if g.is_cuda:
                    g_crop = torch.zeros([g.shape[0], dim2, g.shape[2]], dtype=torch.float32, device=torch.device('cuda:'+str(self.get_gpu())))
                else:
                    g_crop = torch.zeros([g.shape[0], dim2, g.shape[2]], dtype=torch.float32)
                g_crop[:,:,:] = g[:, rowRange[0]:rowRange[1]+1, :]
            else:
                g_crop = np.ascontiguousarray(g[:, rowRange[0]:rowRange[1]+1, :], np.float32)
        else:
            g_crop = None
        return g_crop
        
    def cropProjections(self, rowRange, colRange=None, g=None):
        return self.crop_projections(rowRange, colRange, g)
        
    def crop_projections(self, rowRange, colRange=None, g=None):
        """Crops rows and columns from projection data
        
        This function crops rows and columns from the projection data.
        The appropriate CT geometry parameters are updated (e.g., centerRow, centerCol, numRows, numCols, moduleCenters, etc)
        and if the input projection data is given, the selected projection rows and columns are removed
        
        Args:
            colRange (2-element numpy array): the range of detector column indices to keep, all other rows will be removed
            rowRange (2-element numpy array): the range of detector column indices to keep, all other rows will be removed
            g (C contiguous float32 numpy array or torch tensor): projection data to operate on (optional)
            
        Returns:
            If g is given, then a new numpy array is returned with the selected rows and columns (the 2nd and 3rd dimensions) removed from the data
        
        """
        if colRange is None:
            return self.cropRows(rowRange, g)
        if rowRange is None:
            return self.cropCols(colRange, g)
        numCols = self.get_numCols()
        numRows = self.get_numRows()
        if len(colRange) != 2 or colRange[0] < 0 or colRange[1] < colRange[0] or colRange[1] > self.get_numCols()-1:
            print('Error: cropProjections invalid argument!')
            return None
        if len(rowRange) != 2 or rowRange[0] < 0 or rowRange[1] < rowRange[0] or rowRange[1] > self.get_numRows()-1:
            print('Error: cropProjections invalid argument!')
            return None
        self.cropRows(rowRange)
        self.cropCols(colRange)
        if g is not None:
            #g_crop = g[:, rowRange[0]:rowRange[1]+1, colRange[0]:colRange[1]+1]
            if has_torch == True and type(g) is torch.Tensor:
                dim2 = rowRange[1]-rowRange[0]+1
                dim3 = colRange[1]-colRange[0]+1
                if g.is_cuda:
                    g_crop = torch.zeros([g.shape[0], dim2, dim3], dtype=torch.float32, device=torch.device('cuda:'+str(self.get_gpu())))
                else:
                    g_crop = torch.zeros([g.shape[0], dim2, dim3], dtype=torch.float32)
                g_crop[:,:,:] = g[:, rowRange[0]:rowRange[1]+1, colRange[0]:colRange[1]+1]
            else:
                g_crop = np.ascontiguousarray(g[:, rowRange[0]:rowRange[1]+1, colRange[0]:colRange[1]+1], np.float32)
        else:
            g_crop = None
        return g_crop
    
    def down_sample_projections(self, factors, g=None):
        """down-samples the given projection data

        This function applies an anti-aliasing filter and down-samples projection data and updates the CT geometry parameters accordingly.
        This anti-aliasing filter is the same one used in the LowPassFilter function.
        
        Args:
            factors: 3-element array of down-sampling factors
            g (C contiguous float32 numpy array or torch tensor): projection data to down-sample
            
        Returns:
            down-sampled array (if projection data was provided in the arguments)
        """
        if factors[0] != 1.0:
            print('Error: cannot down-sample the projection angle dimension')
            return None

        pixelHeight = self.get_pixelHeight()*factors[1]
        pixelWidth = self.get_pixelWidth()*factors[2]        
        if g is not None:
            g_dn = self.down_sample(factors, g)
            numRows = g_dn.shape[1]
            numCols = g_dn.shape[2]
        else:
            g_dn = None
            numRows = int(self.get_numRows()/factors[1])
            numCols = int(self.get_numCols()/factors[2])

        row_shift = (self.get_centerRow() - 0.5*float(self.get_numRows()-1))*self.get_pixelHeight()
        col_shift = (self.get_centerCol() - 0.5*float(self.get_numCols()-1))*self.get_pixelWidth()

        centerRow = row_shift/pixelHeight + 0.5*float(numRows-1)
        centerCol = col_shift/pixelWidth + 0.5*float(numCols-1)
        
        self.set_pixelHeight(pixelHeight)        
        self.set_pixelWidth(pixelWidth)
        self.set_numRows(numRows)
        self.set_numCols(numCols)
        
        self.set_centerRow(centerRow)
        self.set_centerCol(centerCol)
        
        return g_dn
        
    def up_sample_projections(self, factors, g=None, dims=None):
        """up-samples the given projection data

        This function up-samples projection data and updates the CT geometry parameters accordingly.
        The up-sampling is performed using trilinear interpolation.
        
        Args:
            factors: 3-element array of up-sampling factors
            g (C contiguous float32 numpy array or torch tensor): projection data to up-sample
            
        Returns:
            up-sampled array (if projection data was provided in the arguments)
        """
        if factors[0] != 1.0:
            print('Error: cannot up-sample the projection angle dimension')
            return None

        pixelHeight = self.get_pixelHeight()/factors[1]
        pixelWidth = self.get_pixelWidth()/factors[2]
        if g is not None:
            g_dn = self.up_sample(factors, g, dims)
            numRows = g_dn.shape[1]
            numCols = g_dn.shape[2]
        else:
            g_dn = None
            numRows = int(self.get_numRows()*factors[1])
            numCols = int(self.get_numCols()*factors[2])
            
        row_shift = (self.get_centerRow() - 0.5*float(self.get_numRows()-1))*self.get_pixelHeight()
        col_shift = (self.get_centerCol() - 0.5*float(self.get_numCols()-1))*self.get_pixelWidth()

        centerRow = row_shift/pixelHeight + 0.5*float(numRows-1)
        centerCol = col_shift/pixelWidth + 0.5*float(numCols-1)
        
        self.set_pixelHeight(pixelHeight)        
        self.set_pixelWidth(pixelWidth)
        self.set_numRows(numRows)
        self.set_numCols(numCols)
        if self.get_geometry() != 'MODULAR':
            self.set_centerRow(centerRow)
            self.set_centerCol(centerCol)
        
        return g_dn
        
    def down_sample_volume(self, factors, f=None):
        """down-samples the given volume data

        This function applies an anti-aliasing filter and down-samples volume data and updates the CT volume parameters accordingly.
        This anti-aliasing filter is the same one used in the LowPassFilter function.
        
        Args:
            factors: 3-element array of down-sampling factors
            f (C contiguous float32 numpy array or torch tensor): volume data to down-sample
            
        Returns:
            down-sampled array (if volume data was provided in the arguments)
        """
        if factors[2] != factors[1]:
            print('Error: voxel pitch must be the same in x and y dimensions')
            return None
        #geomText = self.get_geometry()
        #if factors[2] != 1.0 and geomText == 'FAN' or geomText == 'PARALLEL':
        #    print('Error: cannot change the voxel pitch in the z dimension for parallel- and fan-beam')
        #    return None
        
        voxelHeight = self.get_voxelHeight()*factors[0]
        voxelWidth = self.get_voxelWidth()*factors[1]
        if f is not None:
            f_dn = self.down_sample(factors, f)
            numZ = f_dn.shape[0]
            numY = f_dn.shape[1]
            numX = f_dn.shape[2]
        else:
            f_dn = None
            numZ = int(self.get_numZ()/factors[0])
            numY = int(self.get_numY()/factors[1])
            numX = int(self.get_numX()/factors[2])
        
        self.set_voxelHeight(voxelHeight)        
        self.set_voxelWidth(voxelWidth)
        self.set_numZ(numZ)
        self.set_numY(numY)
        self.set_numX(numX)
        
        return f_dn
        
    def up_sample_volume(self, factors, f=None, dims=None):
        """up-samples the given volume data

        This function up-samples volume data and updates the CT volume parameters accordingly.
        The up-sampling is performed using trilinear interpolation.
        
        Args:
            factors: 3-element array of up-sampling factors
            f (C contiguous float32 numpy array or torch tensor): volume data to up-sample
            
        Returns:
            up-sampled array (if volume data was provided in the arguments)
        """
        if factors[2] != factors[1]:
            print('Error: voxel pitch must be the same in x and y dimensions')
            return None
        #geomText = self.get_geometry()
        #if factors[2] != 1.0 and geomText == 'FAN' or geomText == 'PARALLEL':
        #    print('Error: cannot change the voxel pitch in the z dimension for parallel- and fan-beam')
        #    return None
        
        voxelHeight = self.get_voxelHeight()/factors[0]
        voxelWidth = self.get_voxelWidth()/factors[1]
        if f is not None:
            f_up = self.up_sample(factors, f, dims)
            numZ = f_dn.shape[0]
            numY = f_dn.shape[1]
            numX = f_dn.shape[2]
        else:
            f_up = None
            numZ = int(self.get_numZ()*factors[0])
            numY = int(self.get_numY()*factors[1])
            numX = int(self.get_numX()*factors[2])
        
        self.set_voxelHeight(voxelHeight)        
        self.set_voxelWidth(voxelWidth)
        self.set_numZ(numZ)
        self.set_numY(numY)
        self.set_numX(numX)
        
        return f_up
        
    def set_volume_mask(self, vol_mask):
        r"""Sets the tomographicModels member variable called volume_mask
        
        The purpose of the volume mask is to provide user-specified regions of which voxels to include in
        forward and back projection algorithms.  This mask must by a 3D numpy or torch tensor of the same
        size as the reconstruction volume. This array must have just ones and zeros, where the zeros mark
        the voxels that should not be considered in the forward or backprojection algorithms.
        This masking can remove unwanted regions from the reconstruction or potentially improve the quality of
        iterative reconstruction algorithms by removing unknowns from the reconstruction problem.
        Using this mask does not improve the speed of forward or backprojection algorithms.
        This mask is applied after performing a backprojection and before performing a projection.
        Note that masks applied to the projection data can be applied as arguments to the various
        iterative reconstruction algorithms (see "W" or "mask" arguments).
        
        Args:
            vol_mask (C contiguous float32 numpy array or torch tensor): the mask to be applied to the volume
            
        Returns:
            no return value
        
        """
        self.volume_mask = vol_mask
        
    def clear_volume_mask(self):
        r"""Clears the tomographicModels member variable called volume_mask
        
        After running this function, the volume masking will no longer be performed because volume_mask is set to None
        
        """
        self.volume_mask = None
        
    def volume_mask_is_valid(self, f):
        if self.volume_mask is not None and f is not None and type(f) == type(self.volume_mask) and f.shape == self.volume_mask.shape:
            return True
        else:
            return False
        
    def apply_volume_mask(self, f):
        r"""Multiplies the given input by the tomographicModels member variable called volume_mask

        This function performs the following: f = f * self.volume_mask        
        If the volume_mask has not been set (see set_volume_mask) or does not match the size or type of the input, this function does nothing
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): reconstruction volume
        
        Returns:
            no return value
        """
        if self.volume_mask_is_valid(f):
            f[:,:,:] = f[:,:,:] * self.volume_mask[:,:,:]
    
    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION PROVIDES ITERATIVE RECONSTRUCTION ALGORITHM THAT USE THE LEAP FORWARD AND BACKPROJECTION OPERATIONS
    ###################################################################################################################
    ###################################################################################################################
    def isAllZeros(self, f):
        if has_torch == True and type(f) is torch.Tensor:
            if torch.count_nonzero(f) == 0:
                return True
            else:
                return False
        else:
            if not np.any(f):
                return True
            else:
                return False
        
    def innerProd(self, x, y, w=None):
        if has_torch == True and type(x) is torch.Tensor:
            if w is None:
                return torch.sum(x*y)
            else:
                return torch.sum(x*y*w)
        else:
            if w is None:
                return np.sum(x*y)
            else:
                return np.sum(x*y*w)
                
    def sum(self, x):
        if has_torch == True and type(x) is torch.Tensor:
            return torch.sum(x)
        else:
            return np.sum(x)
        
    def abs(self, x):
        if has_torch == True and type(x) is torch.Tensor:
            return torch.abs(x)
        else:
            return np.abs(x)
            
    def minimum(self, x, y):
        if has_torch == True and type(x) is torch.Tensor:
            return torch.minimum(x, y)
        else:
            return np.minimum(x, y)
            
    def maximum(self, x, y):
        if has_torch == True and type(x) is torch.Tensor:
            return torch.maximum(x, y)
        else:
            return np.maximum(x, y)
            
    def sign(self, x):
        if has_torch == True and type(x) is torch.Tensor:
            return torch.sign(x)
        else:
            return np.sign(x)
                
    def expNeg(self, x):
        """ Returns exp(-x), converting attenuation data to transmission data """
        if has_torch == True and type(x) is torch.Tensor:
            torch.exp(-x, out=x)
        else:
            if len(x.shape) == 1:
                N_1 = 1
                N_2 = 1
                N_3 = x.shape[0]
            elif len(x.shape) == 2:
                N_1 = 1
                N_2, N_3 = x.shape
            elif len(x.shape) == 3:
                N_1, N_2, N_3 = x.shape
            else:
                np.exp(-x, out=x)
                return x

            self.libprojectors.expNeg.restype = ctypes.c_bool
            self.set_model()
            self.libprojectors.expNeg.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int]
            self.libprojectors.expNeg(x, N_1, N_2, N_3)

        return x
            
    def negLog(self, x, gray_value = 1.0):
        """ Returns -log(x), converting transmission data to attenuation data """
        if has_torch == True and type(x) is torch.Tensor:
            torch.log(x/gray_value, out=x)
            x *= -1.0
        else:

            if len(x.shape) == 1:
                N_1 = 1
                N_2 = 1
                N_3 = x.shape[0]
            elif len(x.shape) == 2:
                N_1 = 1
                N_2, N_3 = x.shape
            elif len(x.shape) == 3:
                N_1, N_2, N_3 = x.shape
            else:
                np.log(x/gray_value, out=x)
                x *= -1.0
                return x

            self.libprojectors.negLog.restype = ctypes.c_bool
            self.set_model()
            self.libprojectors.negLog.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
            self.libprojectors.negLog(x, N_1, N_2, N_3, gray_value)

        return x
    
    def breakIntoSubsets(self, g, numSubsets):
        if numSubsets <= 0 or len(g.shape) != 3:
            return None
        else:
            N = g.shape[0]
            g_subsets = []
            for m in range(numSubsets):
                if m == g.shape[0]-1:
                    if has_torch == True and type(g) is torch.Tensor:
                        if g.is_cuda:
                            g_subset = torch.zeros([1, g.shape[1], g.shape[2]], dtype=torch.float32, device=torch.device('cuda:'+str(self.get_gpu())))
                        else:
                            g_subset = torch.zeros([1, g.shape[1], g.shape[2]], dtype=torch.float32)
                    else:
                        g_subset = np.zeros((1,g.shape[1],g.shape[2]),dtype=np.float32)
                    g_subset[0,:,:] = g[m,:,:]
                    g_subsets.append(g_subset)
                else:
                    if has_torch == True and type(g) is torch.Tensor:
                        dim1 = g[m:N:numSubsets,0,0].shape[0]
                        if g.is_cuda:
                            g_subset = torch.zeros([dim1, g.shape[1], g.shape[2]], dtype=torch.float32, device=torch.device('cuda:'+str(self.get_gpu())))
                        else:
                            g_subset = torch.zeros([dim1, g.shape[1], g.shape[2]], dtype=torch.float32)
                        g_subset[:,:,:] = g[m:N:numSubsets,:,:]
                    else:
                        g_subset = np.ascontiguousarray(g[m:N:numSubsets,:,:], np.float32)
                    g_subsets.append(g_subset)
            return g_subsets
    
    def space_carving(self, projection_mask, vol_mask):
        r"""Space Carving Segmentation-Reconstruction Algorithm
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        The volume mask will be set to zero where the rays of the zero values of the projection_mask input argument are located.
        The space carving reconstruction algorithm is given by
        
        .. math::
           \begin{eqnarray}
             f_{mask} &:=& 1 - u\left( P^T(1-g_{mask}) \right)
           \end{eqnarray}
        
        where :math:`g_{mask}` is the projection mask, :math:`f_{mask}` is the volume mask, and :math:`u(\cdot)` is the step or heaviside function.
        
        Args:
            projection_mask (C contiguous float32 numpy or torch array): projection mask/segmentation data (all values should be 0 or 1)
            vol_mask (C contiguous float32 numpy or torch array): volume mask/segmentation data
            
        Returns:
            vol_mask, the same as the input with the same name
        
        """
        projection_mask[:,:,:] = 1.0 - projection_mask[:,:,:]
        self.backproject(projection_mask, vol_mask)
        projection_mask[:,:,:] = 1.0 - projection_mask[:,:,:]
        if type(vol_mask) is np.ndarray:
            np.heaviside(vol_mask, 0.0, out=vol_mask)
        elif has_torch and type(vol_mask) is torch.Tensor:
            torch.heaviside(vol_mask, vol_mask, out=vol_mask)
        vol_mask[:,:,:] = 1.0 - vol_mask[:,:,:]
        self.windowFOV(vol_mask)
        return vol_mask
        
    
    def MLEM(self, g, f, numIter, filters=None, mask=None):
        r"""Maximum Likelihood-Expectation Maximization reconstruction
        
        This algorithm performs reconstruction with the following update equation
        
        .. math::
           \begin{eqnarray}
             f_{n+1} &:=& \frac{f_n}{P^T 1 + R'(f_n)} P^T\left[ \frac{g}{Pf_n} \right]
           \end{eqnarray}
           
        where R'(f) is the gradient of the regularization term(s).
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This reconstruction algorithms assumes the projection data, g, is Poisson distributed which is the
        correct model for SPECT data.
        CT projection data is not Poisson distributed because of the application of the -log
        
        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            filters (filterSequence object): list of differentiable regularization filters
            mask (C contiguous float32 numpy or torch array): projection data to mask out bad data, etc.
            
        
        Returns:
            f, the same as the input with the same name
        """
        #if has_torch == True and type(f) is torch.Tensor:
        #    print('ERROR: Iterative reconstruction algorithms not implemented for torch tensors!')
        #    print('Please convert to numpy array prior to running this algorithm.')
        #    return f
        if mask is not None and mask.shape != g.shape:
            print('Error: mask must be the same shape as the projection data!')
            return None
        if self.verify_inputs(g,f) == False:
            return None
        if self.isAllZeros(f) == True:
            f[:] = 1.0
        else:
            f[f<0.0] = 0.0
 
        if mask is not None:
            Pstar1 = self.copyData(f)
            self.backproject(mask,Pstar1)
        else:
            Pstar1 = self.sensitivity(self.copyData(f))
        Pstar1[Pstar1==0.0] = 1.0
        d = self.allocateData(f)
        Pd = self.allocateData(g)

        for n in range(numIter):
            if self.print_warnings:
                print('MLEM iteration ' + str(n+1) + ' of ' + str(numIter))
            self.project(Pd,f)
            ind = Pd != 0.0
            Pd[ind] = g[ind]/Pd[ind]
            if mask is not None:
                Pd[:] = Pd[:] * mask[:]
            self.backproject(Pd,d)
            if filters is None:
                f *= d/Pstar1
            else:
                if filters.anyDifferentiable():
                    f *= d/self.maximum(0.1*Pstar1, Pstar1 + filters.gradient(f))
                else:
                    f *= d/Pstar1
                    if filters.beta > 0.0 and filters.beta < 1.0:
                        f_save = self.copyData(f)
                        filters.apply(f)
                        f[:] = filters.beta*f[:] + (1.0-filters.beta)*f_save[:]
                        f[f<0.0] = 0.0
                    elif filters.beta > 0.0:
                        filters.apply(f)
                        f[f<0.0] = 0.0
                
        return f
    
    def OSEM(self, g, f, numIter, numSubsets=1, filters=None, mask=None):
        """Ordered Subsets-Expectation Maximization reconstruction
        
        The OSEM algorithm is performed using two nested loops.  The inner loop is performed like
        the MLEM algorithm (see MLEM algorithm documentation for a description of the algorithm),
        but the successive updates of the reconstructed volume are done with a subset of the projection
        angles.  Once every subset of is complete, the process starts over again.  Using these ordered
        subsets reduces the time it takes for this algorithm to converge.
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This reconstruction algorithms assumes the projection data, g, is Poisson distributed which is the
        correct model for SPECT data.
        CT projection data is not Poisson distributed because of the application of the -log
        
        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            numSubsets (int): number of subsets
            filters (filterSequence object): list of differentiable regularization filters
            mask (C contiguous float32 numpy or torch array): projection data to mask out bad data, etc.
        
        Returns:
            f, the same as the input with the same name
        """
        #if has_torch == True and type(f) is torch.Tensor:
        #    print('ERROR: Iterative reconstruction algorithms not implemented for torch tensors!')
        #    print('Please convert to numpy array prior to running this algorithm.')
        #    return f
        if mask is not None and mask.shape != g.shape:
            print('Error: mask must be the same shape as the projection data!')
            return None
        if self.verify_inputs(g,f) == False:
            return None
        if self.isAllZeros(f) == True:
            f[:] = 1.0
        else:
            f[f<0.0] = 0.0
 
        numSubsets = min(numSubsets, self.get_numAngles())
        #if self.get_geometry() == 'MODULAR' and numSubsets > 1:
        #    print('WARNING: Subsets not yet implemented for modular-beam geometry, setting to 1.')
        #    numSubsets = 1
        if numSubsets <= 1:
            return self.MLEM(g, f, numIter, mask)
        else:
        
            # divide g and phis
            subsetParams = subsetParameters(self, numSubsets)
            g_subsets = self.breakIntoSubsets(g, numSubsets)
            if mask is not None:
                mask_subsets = self.breakIntoSubsets(mask, numSubsets)
            else:
                mask_subsets = None
                
            d = self.allocateData(f)
            for n in range(numIter):
                if self.print_warnings:
                    print('OSEM iteration ' + str(n+1) + ' of ' + str(numIter))
                for m in range(numSubsets):
                
                    # set angle array
                    #self.set_angles(phis_subsets[m])
                    subsetParams.setSubset(m)
                    
                    if mask is not None:
                        Pstar1 = self.copyData(f)
                        self.backproject(mask_subsets[m],Pstar1)
                        Pstar1[Pstar1==0.0] = 1.0
                    else:
                        Pstar1 = self.sensitivity(self.copyData(f))
                    #Pstar1[Pstar1==0.0] = 1.0

                    Pd = self.allocateData(g_subsets[m])
                    self.project(Pd,f)
                    ind = Pd != 0.0
                    Pd[ind] = g_subsets[m][ind]/Pd[ind]
                    if mask_subsets is not None:
                        Pd[:] = Pd[:] * mask_subsets[m][:]
                    self.backproject(Pd,d)
                    
                    if filters is None:
                        f *= d/Pstar1
                    else:
                        f *= d/self.maximum(0.1*Pstar1, Pstar1 + filters.gradient(f)/float(numSubsets))
                    
            subsetParams.setSubset(-1)
            return f
        
    def SIRT(self, g, f, numIter, mask=None):
        r"""Simultaneous Iterative Reconstruction Technique reconstruction

        This is the same algorithm as a SART reconstruction with one subset.
        The SIRT algorithm is performed using the following update equation
                
        .. math::
           \begin{eqnarray}
             f_{n+1} &:=& f_n + \frac{0.9}{P^T 1} P^T\left[ \frac{Pf_n - g}{P1} \right]
           \end{eqnarray}
           
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            mask (C contiguous float32 numpy or torch array): projection data to mask out bad data, etc. (zero values indicate projection data pixels not to use)
        
        Returns:
            f, the same as the input with the same name
        
        """
        return self.SART(g, f, numIter, 1, mask)
        
    def SART(self, g, f, numIter, numSubsets=1, mask=None, nonnegativityConstraint=True):
        r"""Simultaneous Algebraic Reconstruction Technique reconstruction
        
        The SART algorithm is performed using two nested loops.  The inner loop is performed like
        the SIRT algorithm (see SIRT algorithm documentation for a description of the algorithm),
        but the successive updates of the reconstructed volume are done with a subset of the projection
        angles.  Once every subset of is complete, the process starts over again.  Using these ordered
        subsets reduces the time it takes for this algorithm to converge.
        
        If one wishes to combine this algorithm with regularization, (e.g., TV), please see ASDPOCS.
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            numSubsets (int): number of subsets
            mask (C contiguous float32 numpy or torch array): projection data to mask out bad data, etc. (zero values indicate projection data pixels not to use)
            nonnegativityConstraint (bool): if true constrains values of reconstruction to be nonnegative
        
        Returns:
            f, the same as the input with the same name
        """
        #if has_torch == True and type(f) is torch.Tensor:
        #    print('ERROR: Iterative reconstruction algorithms not implemented for torch tensors!')
        #    print('Please convert to numpy array prior to running this algorithm.')
        #    return f
        if mask is not None and mask.shape != g.shape:
            print('Error: mask must be the same shape as the projection data!')
            return None
        if self.verify_inputs(g,f) == False:
            return None
        numSubsets = min(numSubsets, self.get_numAngles())
        if numSubsets <= 1:
            P1 = self.allocateData(g)
            self.project(P1,self.allocateData(f,1.0))
            P1[P1<=0.0] = 1.0
            
            Pstar1 = self.sensitivity(self.allocateData(f))
            Pstar1[Pstar1<=0.0] = 1.0
            
            Pd = self.allocateData(g)
            d = self.allocateData(f)

            for n in range(numIter):
                if self.print_warnings:
                    print('SART iteration ' + str(n+1) + ' of ' + str(numIter))
                self.project(Pd,f)
                if mask is not None:
                    Pd = (g-Pd) / P1 * mask
                else:
                    Pd = (g-Pd) / P1
                if self.print_cost:
                    print('\tcost = ' + str(0.5*self.innerProd(Pd,Pd)))
                self.backproject(Pd,d)
                f += 0.9*d / Pstar1
                if nonnegativityConstraint:
                    f[f<0.0] = 0.0
            return f
        else:
            P1 = self.allocateData(g)
            self.project(P1,self.allocateData(f,1.0))
            P1[P1<=0.0] = 1.0
            
            # divide g, P1, and phis
            subsetParams = subsetParameters(self, numSubsets)
            g_subsets = self.breakIntoSubsets(g, numSubsets)
            P1_subsets = self.breakIntoSubsets(P1, numSubsets)
            if mask is not None:
                mask_subsets = self.breakIntoSubsets(mask, numSubsets)
            else:
                mask_subsets = None
            
            d = self.allocateData(f)
            for n in range(numIter):
                if self.print_warnings:
                    print('SART iteration ' + str(n+1) + ' of ' + str(numIter))
                for m in range(numSubsets):
                
                    # set angle array
                    #self.set_angles(phis_subsets[m])
                    subsetParams.setSubset(m)
                    #self.print_parameters()
                    
                    Pstar1 = self.sensitivity(self.allocateData(f))
                    #Pstar1[Pstar1<=0.0] = 1.0

                    Pd = self.allocateData(g_subsets[m])
                    #print(self.get_numAngles())
                    #print(Pd.shape)
                    self.project(Pd,f)
                    
                    if mask_subsets is not None:
                        Pd = (g_subsets[m]-Pd) / P1_subsets[m] * mask_subsets[m]
                    else:
                        Pd = (g_subsets[m]-Pd) / P1_subsets[m]
                    self.backproject(Pd,d)
                    #print('P1 range: ' + str(np.min(P1_subsets[m])) + ' to ' + str(np.max(P1_subsets[m])))
                    #print('d range: ' + str(np.min(d)) + ' to ' + str(np.max(d)))
                    f += 0.9*d / Pstar1
                    if nonnegativityConstraint:
                        f[f<0.0] = 0.0
            subsetParams.setSubset(-1)
            return f
            
    def ASDPOCS(self, g, f, numIter, numSubsets, numTV, filters=None, mask=None, nonnegativityConstraint=True):
        r"""Adaptive Steepest Descent-Projection onto Convex Subsets reconstruction
        
        This algorithm combines SART with regularization (e.g., TV; see \"filters\" argument).  See SART and SIRT documentation
        for more information.  This algorithm solves the following optimization problem
        
        .. math::
           \begin{eqnarray}
             \text{minimize } R(f) \text{ subject to } \|Pf - g\|_2^2 < \varepsilon
           \end{eqnarray}
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        This function actually implements the iTV reconstruction method which is a slight varition to ASDPOCS
        which we find works slightly better.
        
        Here is the reference
        Ritschl, Ludwig, and Marc Kachelriess.
        "Improved total variation regularized image reconstruction (iTV) applied to clinical CT data."
        In Medical Imaging 2011: Physics of Medical Imaging, vol. 7961, pp. 786-798. SPIE, 2011.
        
        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            numSubsets (int): number of subsets
            numTV (int): number of TV diffusion steps, larger numbers perform stronger regularization/ smoothing
            filters (filterSequence object): list of regularization filters
            mask (C contiguous float32 numpy or torch array): projection data to mask out bad data, etc. (zero values indicate projection data pixels not to use)
            nonnegativityConstraint (bool): if true constrains values of reconstruction to be nonnegative
        
        Returns:
            f, the same as the input with the same name
        """
        #if has_torch == True and type(f) is torch.Tensor:
        #    print('ERROR: Iterative reconstruction algorithms not implemented for torch tensors!')
        #    print('Please convert to numpy array prior to running this algorithm.')
        #    return f
        if mask is not None and mask.shape != g.shape:
            print('Error: mask must be the same shape as the projection data!')
            return None
        if self.verify_inputs(g,f) == False:
            return None
        if numTV <= 0:
            return self.SART(g,f,numIter,numSubsets,mask)
            
        if filters is None:
            filters = filterSequence()
            filters.append(TV(self, delta=0.0))
            #print('None?')
        elif isinstance(filters, (int, float)):
            delta = filters
            filters = filterSequence()
            filters.append(TV(self, delta))
            #print('float?')
            
        numSubsets = min(numSubsets, self.get_numAngles())
        omega = 0.8
        P1 = self.allocateData(g)
        self.project(P1,self.allocateData(f,1.0))
        P1[P1==0.0] = 1.0

        subsetParams = subsetParameters(self, numSubsets)
        g_subsets = []
        P1_subsets = []
        mask_subsets = None
        if numSubsets > 1:
            g_subsets = self.breakIntoSubsets(g, numSubsets)
            P1_subsets = self.breakIntoSubsets(P1, numSubsets)
            if mask is not None:
                mask_subsets = self.breakIntoSubsets(mask, numSubsets)
        else:
            Pstar1 = self.sensitivity(self.allocateData(f))
            Pstar1[Pstar1==0.0] = 1.0
        
        #Pd = self.allocateProjections()
        Pf_minus_g = self.allocateData(g)
        Pf_TV_minus_g = self.allocateData(g)
        d = self.allocateData(f)
        f_TV = self.allocateData(f)

        self.project(Pf_minus_g, f)
        Pf_minus_g -= g
        
        curCost = self.innerProd(Pf_minus_g, Pf_minus_g)
        
        for n in range(numIter):
            if self.print_warnings:
                print('ASDPOCS iteration ' + str(n+1) + ' of ' + str(numIter))
            
            # SART Update
            if numSubsets <= 1:
                if mask is not None:
                    Pf_minus_g = Pf_minus_g / P1 * mask
                else:
                    Pf_minus_g = Pf_minus_g / P1
                self.backproject(Pf_minus_g,d)
                f -= 0.9*d / Pstar1
                if nonnegativityConstraint:
                    f[f<0.0] = 0.0
            else:
                for m in range(numSubsets):
                    #self.set_angles(phis_subsets[m])
                    subsetParams.setSubset(m)
                    Pstar1 = self.sensitivity(self.allocateData(f))
                    #Pstar1[Pstar1==0.0] = 1.0

                    Pd = self.allocateData(g_subsets[m])
                    self.project(Pd,f)
                    if mask_subsets is not None:
                        Pd = (g_subsets[m]-Pd) / P1_subsets[m] * mask_subsets[m]
                    else:
                        Pd = (g_subsets[m]-Pd) / P1_subsets[m]
                    self.backproject(Pd,d)
                    f += 0.9*d / Pstar1
                    if nonnegativityConstraint:
                        f[f<0.0] = 0.0
                subsetParams.setSubset(-1)

            # Calculate SART error sinogram and calculate cost            
            self.project(Pf_minus_g, f)
            Pf_minus_g = Pf_minus_g - g
            
            epsilon_SART = self.innerProd(Pf_minus_g, Pf_minus_g)

            #'''            
            # TV step(s)
            f_TV[:] = f[:]
            #self.diffuse(f_TV, delta, numTV)
            for m in range(numTV):
                f_TV = filters.apply(f_TV)
            #self.displayVolume(f_TV)
            self.project(Pf_TV_minus_g, f_TV)
            Pf_TV_minus_g = Pf_TV_minus_g - g
            
            epsilon_TV = self.innerProd(Pf_TV_minus_g, Pf_TV_minus_g)
            
            # Combine SART and TV Steps
            temp = self.innerProd(Pf_minus_g, Pf_TV_minus_g)
            a = epsilon_SART - 2.0 * temp + epsilon_TV
            b = temp - epsilon_SART
            c = epsilon_SART - ((1.0 - omega) * epsilon_SART + omega * curCost)
            
            if has_torch == True and type(f) is torch.Tensor:
                a = a.cpu().detach().numpy()
                b = b.cpu().detach().numpy()
                c = c.cpu().detach().numpy()
            
            disc = b * b - a * c
            
            alpha = 0.0
            alpha_1 = 0.0
            alpha_2 = 0.0
            if disc < 0.0:
                if self.print_warnings:
                    print("  Unable to estimate step size, setting lambda to zero!")
                alpha = 0.0
            else:
                if a != 0.0:
                    alpha_1 = (-b + np.sqrt(disc)) / a
                    alpha_2 = (-b - np.sqrt(disc)) / a
                if a == 0.0:
                    alpha = 1.0
                elif a > 0.0:
                    alpha = alpha_1
                else:
                    alpha = alpha_2
            alpha = min(1.0, alpha)
            alpha = max(0.0, min(1.0, alpha))  # force lambda to be a valid number
            if self.print_warnings:
                print('  lambda = ' + str(np.round(1000.0*alpha)/1000.0) + ' (' + str(np.round(1000.0*alpha_1)/1000.0) + ', ' + str(np.round(1000.0*alpha_2)/1000.0) + ')')
            if self.print_cost:
                print('\tcost = ' + str(epsilon_SART))
            if alpha < 0.0:
                if self.print_warnings:
                    print("  Stopping criteria met, stopping iterations.")
                break
                
                
            # Do Update
            f[:] = (1.0-alpha)*f[:] + alpha*f_TV[:]
            Pf_minus_g[:] = (1.0-alpha)*Pf_minus_g[:] + alpha*Pf_TV_minus_g[:]
            
            curCost = self.innerProd(Pf_minus_g, Pf_minus_g)
            #'''
                        
        return f
        
        
    def LS(self, g, f, numIter, preconditioner=None, nonnegativityConstraint=True):
        r"""Least Squares reconstruction

        This function minimizes the Least Squares cost function using Preconditioned Conjugate Gradient.
        The optional preconditioner is the Separable Quadratic Surrogate for the Hessian of the cost function
        which is given by (P*P1)^-1, where 1 is a volume of all ones, P is forward projection, and P* is backprojection.
        The Least Squares cost function is given by the following
        
        .. math::
           \begin{eqnarray}
             C_{LS}(f) &:=& \frac{1}{2} \| Pf - g \|^2
           \end{eqnarray}
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            preconditioner (string): specifies the preconditioner as 'SQS', 'RAMP', or 'SARR'
            nonnegativityConstraint (bool): if true constrains values of reconstruction to be nonnegative
        
        Returns:
            f, the same as the input with the same name
        """
        return self.RWLS(g, f, numIter, None, 1.0, preconditioner, nonnegativityConstraint)
        
    def WLS(self, g, f, numIter, W=None, preconditioner=None, nonnegativityConstraint=True):
        r"""Weighted Least Squares reconstruction
        
        This function minimizes the Weighted Least Squares cost function using Preconditioned Conjugate Gradient.
        The optional preconditioner is the Separable Quadratic Surrogate for the Hessian of the cost function
        which is given by (P*WP1)^-1, where 1 is a volume of all ones, W are the weights, P is forward projection, and P* is backprojection.
        The Weighted Least Squares cost function is given by the following
        
        .. math::
           \begin{eqnarray}
             C_{WLS}(f) &:=& \frac{1}{2} (Pf - g)^T W (Pf - g)
           \end{eqnarray}
           
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            W (C contiguous float32 numpy array): weights, should be the same size as g, if not given, W=exp(-g); can also be used to mask out bad data
            preconditioner (string): specifies the preconditioner as 'SQS', 'RAMP', or 'SARR'
            nonnegativityConstraint (bool): if true constrains values of reconstruction to be nonnegative
        
        Returns:
            f, the same as the input with the same name
        """
        return self.RWLS(g, f, numIter, None, W, preconditioner, nonnegativityConstraint)
        
    def RLS(self, g, f, numIter, filters=None, preconditioner=None, nonnegativityConstraint=True):
        r"""Regularized Least Squares reconstruction
        
        This function minimizes the Regularized Least Squares cost function using Preconditioned Conjugate Gradient.
        The optional preconditioner is the Separable Quadratic Surrogate for the Hessian of the cost function
        which is given by (P*P1)^-1, where 1 is a volume of all ones, P is forward projection, and P* is backprojection.
        The Regularized Least Squares cost function is given by the following
        
        .. math::
           \begin{eqnarray}
             C_{RLS}(f) &:=& \frac{1}{2} \| Pf - g \|^2 + R(f)
           \end{eqnarray}

        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            filters (filterSequence object): list of differentiable regularization filters
            preconditioner (string): specifies the preconditioner as 'SQS', 'RAMP', or 'SARR'
            nonnegativityConstraint (bool): if true constrains values of reconstruction to be nonnegative
        
        Returns:
            f, the same as the input with the same name
        """
        return self.RWLS(g, f, numIter, filters, 1.0, preconditioner, nonnegativityConstraint)
       
    def RWLS(self, g, f, numIter, filters=None, W=None, preconditioner=None, nonnegativityConstraint=True):
        r"""Regularized Weighted Least Squares reconstruction
        
        This function minimizes the Regularized Weighted Least Squares cost function using Preconditioned Conjugate Gradient.
        The optional preconditioner is the Separable Quadratic Surrogate for the Hessian of the cost function
        which is given by (P*WP1)^-1, where 1 is a volume of all ones, W are the weights, P is forward projection, and P* is backprojection.
        The Regularized Weighted Least Squares cost function is given by the following
        
        .. math::
           \begin{eqnarray}
             C_{RWLS}(f) &:=& \frac{1}{2} (Pf - g)^T W (Pf - g) + R(f)
           \end{eqnarray}

        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            filters (filterSequence object): list of differentiable regularization filters
            W (C contiguous float32 numpy array): weights, should be the same size as g, if not given, W:=exp(-g); can also be used to mask out bad data
            preconditioner (string): specifies the preconditioner as 'SQS', 'RAMP', or 'SARR'
            nonnegativityConstraint (bool): if true constrains values of reconstruction to be nonnegative
        
        Returns:
            f, the same as the input with the same name
        """
        
        if filters is None:
            filters = filterSequence(0.0)
        elif isinstance(filters, (int, float)):
            print('Error: invalid inputs!  Note that algorithm syntax has changed, please see documentation.')
            return None
        elif filters.count() == 0:
            print('Warning: filterSequence given, but has no terms!')
            filters.beta = 0.0
        
        if self.verify_inputs(g,f) == False:
            return None
        
        conjGradRestart = 50
        if W is None:
            W = self.copyData(g)
            self.BlurFilter2D(W,3.0)
            self.expNeg(W)
            
        Pf = self.copyData(g)
        if self.isAllZeros(f) == False:
            # fix scaling
            if nonnegativityConstraint:
                f[f<0.0] = 0.0
            self.project(Pf,f)
            Pf_dot_Pf = self.innerProd(Pf,Pf)
            g_dot_Pf = self.innerProd(g,Pf)
            if Pf_dot_Pf > 0.0 and g_dot_Pf > 0.0:
                f *= g_dot_Pf / Pf_dot_Pf
                Pf *= g_dot_Pf / Pf_dot_Pf
        else:
            Pf[:] = 0.0
        Pf_minus_g = Pf
        Pf_minus_g -= g
        
        grad = self.allocateData(f)
        u = self.allocateData(f)
        #Pu = self.allocateData(g)
        
        d = self.allocateData(f)
        Pd = self.allocateData(g)
        
        grad_old_dot_grad_old = 0.0
        grad_old = self.allocateData(f)
        
        if preconditioner == True:
            preconditioner = 'SQS'
        if preconditioner == 'SQS':
            # Calculate the SQS preconditioner
            # Reuse some of the memory allocated above
            #Q = 1.0 / P*WP1
            Q = self.allocateData(f)
            Q[:] = 1.0
            self.project(Pd,Q)
            Pd *= W
            self.backproject(Pd,Q)
            Q[Q==0.0] = 1.0
            Q = 1.0 / Q
        else:
            Q = 1.0
        
        for n in range(numIter):
            if self.print_warnings:
                print('RWLS iteration ' + str(n+1) + ' of ' + str(numIter))
            #WPf_minus_g = self.copyData(Pf_minus_g)
            WPf_minus_g = Pf_minus_g
            if W is not None:
                WPf_minus_g *= W
            if preconditioner == 'SARR':
                self.FBP(WPf_minus_g, grad)
            else:
                self.backproject(WPf_minus_g, grad)
            if filters.beta > 0.0:
                #Sf1 = self.TVgradient(f, delta, beta)
                Sf1 = filters.gradient(f)
                if preconditioner == 'SARR':
                    self.rampFilterVolume(Sf1)
                    self.windowFOV(Sf1)
                    pitchHat = self.get_normalizedHelicalPitch()
                    if pitchHat > 0.0:
                        Sf1 *= self.get_FBPscalar() * 0.5*pitchHat
                    else:
                        Sf1 *= self.get_FBPscalar() * 180.0/self.get_angularRange()
                elif preconditioner == 'SQS':
                    self.windowFOV(Sf1)
                grad[:] += Sf1[:]

                #f[:] = grad[:] # FIXME
                #return f # FIXME
                
            u[:] = grad[:]
            u = Q*u
            if preconditioner == 'RAMP':
                self.rampFilterVolume(u)
                self.windowFOV(u)
            #self.project(Pu, u)
            
            if n == 0 or (n % conjGradRestart) == 0:
                d[:] = u[:]
                #Pd[:] = Pu[:]
            else:
                gamma = (self.innerProd(u,grad) - self.innerProd(u,grad_old)) / grad_old_dot_grad_old

                d = u + gamma*d
                #Pd = Pu + gamma*Pd

                if self.innerProd(d,grad) <= 0.0:
                    if self.print_warnings:
                        print('\tRLWS-CG: CG descent condition violated, must use GD descent direction')
                    d[:] = u[:]
                    #Pd[:] = Pu[:]
            
            grad_old_dot_grad_old = self.innerProd(u,grad)
            grad_old[:] = grad[:]
            
            self.project(Pd, d)
            
            num = 0.0
            if preconditioner == 'SARR':
                num = self.innerProd(Pd,WPf_minus_g)
                if filters.beta > 0.0:
                    num += self.innerProd(d,Sf1)
            else:
                num = self.innerProd(d,grad)
            stepSize = self.RWLSstepSize(f, grad, d, Pd, W, filters, num)
            if stepSize <= 0.0:
                if self.print_warnings:
                    print('invalid step size; quitting!')
                break
            
            f[:] = f[:] - stepSize*d[:]
            if nonnegativityConstraint:
                f[f<0.0] = 0.0
                self.project(Pf,f)
            else:
                Pf[:] = Pf[:] - stepSize*Pd[:]
            Pf_minus_g[:] = Pf[:] - g[:]
            if self.print_cost:
                dataFidelity = 0.5*self.innerProd(Pf_minus_g,Pf_minus_g,W)
                if has_torch == True and type(dataFidelity) is torch.Tensor:
                    dataFidelity = dataFidelity.cpu().detach().numpy()
                if filters.beta > 0.0:
                    #regularizationCost = self.TVcost(f, delta, beta)
                    regularizationCost = filters.cost(f)
                    if has_torch == True and type(regularizationCost) is torch.Tensor:
                        regularizationCost = regularizationCost.cpu().detach().numpy()
                    print('\tcost = ' + str(dataFidelity+regularizationCost) + ' = ' + str(dataFidelity) + ' + ' + str(regularizationCost))
                else:
                    print('\tcost = ' + str(dataFidelity))
                
        return f

    def RWLSstepSize(self, f, grad, d, Pd, W, filters, num=None):
        """Calculates the step size for an RWLS iteration

        Args:
            f (C contiguous float32 numpy or torch array): volume data
            grad (C contiguous float32 numpy or torch array): gradient of the RWLS cost function
            d (C contiguous float32 numpy or torch array): descent direction of the RWLS cost function
            Pd (C contiguous float32 numpy or torch array): forward projection of d
            W (C contiguous float32 numpy or torch array): weights, should be the same size as g, if not given, assumes is all ones
            filters (filterSequence object): list of filters to use as a regularizer terms
        
        Returns:
            step size (float)
        """
        if num is None:
            num = self.innerProd(d,grad)
        if W is not None:
            denomA = self.innerProd(Pd,Pd,W)
        else:
            denomA = self.innerProd(Pd,Pd)
        denomB = 0.0
        if filters.beta > 0.0:
            #denomB = self.TVquadForm(f, d, delta, beta)
            denomB = filters.quadForm(f, d)
            #print('denomB = ' + str(denomA))
        denom = denomA + denomB
        if has_torch == True and type(denom) is torch.Tensor:
            denom = denom.cpu().detach().numpy()
            num = num.cpu().detach().numpy()

        stepSize = 0.0
        if np.abs(denom) > 1.0e-16:
            stepSize = num / denom
        if self.print_warnings:
            print('\tlambda = ' + str(stepSize))
        return stepSize
        
    def DLS(self, g, f, numIter, preconditionerFWHM=1.0, nonnegativityConstraint=False, dimDeriv=2):
        """Derivative Least Squares reconstruction
        
        See documentation for RDLS because this is the same algorithm without the regularization.
        
        """
        return self.RDLS(g, f, numIter, 0.0, 0.0, preconditionerFWHM, nonnegativityConstraint, dimDeriv)
        
    def RDLS(self, g, f, numIter, filters=None, preconditionerFWHM=1.0, nonnegativityConstraint=False, dimDeriv=1):
        r"""Regularized Derivative Least Squares reconstruction
        
        This function minimizes the Regularized Derivative Least Squares cost function using Preconditioned Conjugate Gradient.
        The optional preconditioner is a 2D blurring for each z-slice.
        The Regularized Weighted Least Squares cost function is given by the following
        
        .. math::
           \begin{eqnarray}
             C_{RDLS}(f) &:=& \frac{1}{2} (Pf - g)^T \Delta (Pf - g) + R(f)
           \end{eqnarray}
           
        where :math:`\Delta` is the Laplacian operator.        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            filters (filterSequence object): list of differentiable regularization filters
            preconditionerFWHM (float): specifies the FWHM of the blur preconditioner
            nonnegativityConstraint (bool): whether to apply a nonnegativity constraint
            dimDeriv (int): number of dimensions (1 or 2) to apply the Laplacian derivative
        
        Returns:
            f, the same as the input with the same name
        """
        
        if filters is None:
            filters = filterSequence(0.0)
        elif isinstance(filters, (int, float)):
            print('Error: invalid inputs!  Note that algorithm syntax has changed, please see documentation.')
            return None
        
        if self.verify_inputs(g,f) == False:
            return None
        
        if preconditionerFWHM > 1.0:
            smoothLaplacian = True
            #print('The preconditionerFWHM feature does not seem to be working, so this parameter will be disabled until we resolve the issue.')
        else:
            smoothLaplacian = False
        #preconditionerFWHM = 1.0
        conjGradRestart = 50
        Pf = self.copyData(g)
        if self.isAllZeros(f) == False:
            # fix scaling
            if nonnegativityConstraint:
                f[f<0.0] = 0.0
            self.project(Pf,f)
            Pf_dot_Pf = self.innerProd(Pf,Pf)
            g_dot_Pf = self.innerProd(g,Pf)
            if Pf_dot_Pf > 0.0 and g_dot_Pf > 0.0:
                f *= g_dot_Pf / Pf_dot_Pf
                Pf *= g_dot_Pf / Pf_dot_Pf
        else:
            Pf[:] = 0.0
        Pf_minus_g = Pf
        Pf_minus_g[:] -= g[:]
        
        LPf_minus_g = self.copyData(Pf)
        
        grad = self.allocateData(f)
        u = self.allocateData(f)
        #Pu = self.allocateData(g)
        
        d = self.allocateData(f)
        Pd = self.allocateData(g)
        
        grad_old_dot_grad_old = 0.0
        grad_old = self.allocateData(f)
                
        for n in range(numIter):
            if self.print_warnings:
                print('RDLS iteration ' + str(n+1) + ' of ' + str(numIter))
            LPf_minus_g[:] = Pf_minus_g[:]
            self.Laplacian(LPf_minus_g, dimDeriv, smoothLaplacian)
            LPf_minus_g *= -1.0
            self.backproject(LPf_minus_g, grad)
            if filters.beta > 0.0:
                #Sf1 = self.TVgradient(f, delta, beta)
                Sf1 = filters.gradient(f)
                grad[:] += Sf1[:]

            u[:] = grad[:]
            #if preconditionerFWHM > 1.0:
            #    self.BlurFilter(u, preconditionerFWHM)
            #self.project(Pu, u)
            
            if n == 0 or (n % conjGradRestart) == 0:
                d[:] = u[:]
                #Pd[:] = Pu[:]
            else:
                gamma = (self.innerProd(u,grad) - self.innerProd(u,grad_old)) / grad_old_dot_grad_old

                d[:] = u[:] + gamma*d[:]
                #Pd[:] = Pu[:] + gamma*Pd[:]

                if self.innerProd(d,grad) <= 0.0:
                    if self.print_warnings:
                        print('\tRDLS-CG: CG descent condition violated, must use gradient descent direction')
                    d[:] = u[:]
                    #Pd[:] = Pu[:]
            
            grad_old_dot_grad_old = self.innerProd(u,grad)
            grad_old[:] = grad[:]
            
            self.project(Pd, d)
            
            stepSize = self.RDLSstepSize(f, grad, d, Pd, filters, dimDeriv, smoothLaplacian)
            #if stepSize <= 0.0:
            #    print('invalid step size; quitting!')
            #    break
            
            f[:] = f[:] - stepSize*d[:]
            if nonnegativityConstraint:
                f[f<0.0] = 0.0
                self.project(Pf,f)
                Pf_minus_g[:] = Pf[:] - g[:]
            else:
                Pf_minus_g[:] = Pf_minus_g[:] - stepSize*Pd[:]
        return f

    def RDLSstepSize(self, f, grad, d, Pd, filters, dimDeriv, smoothLaplacian):
        """Calculates the step size for an RDLS iteration

        Args:
            f (C contiguous float32 numpy or torch array): volume data
            grad (C contiguous float32 numpy or torch array): gradient of the RWLS cost function
            d (C contiguous float32 numpy or torch array): descent direction of the RWLS cost function
            Pd (C contiguous float32 numpy or torch array): forward projection of d
            filters (filterSequence object): list of differentiable regularization filters
        
        Returns:
            step size (float)
        """
        num = self.innerProd(d,grad)
        LPd = self.copyData(Pd)
        self.Laplacian(LPd, dimDeriv, smoothLaplacian)
        LPd *= -1.0
        denomA = self.innerProd(LPd,Pd)
        denomB = 0.0;
        if filters.beta > 0.0:
            #denomB = self.TVquadForm(f, d, delta, beta)
            denomB = filters.quadForm(f, d)
            #print('denomB = ' + str(denomA))
        denom = denomA + denomB

        if has_torch == True and type(denom) is torch.Tensor:
            denom = denom.cpu().detach().numpy()
            num = num.cpu().detach().numpy()

        stepSize = 0.0
        if np.abs(denom) > 1.0e-16:
            stepSize = num / denom
        if self.print_warnings:
            if stepSize < 0.0:
                print('\tlambda = ' + str(stepSize) + ' = ' + str(num) + ' / ' + str(denom))
            else:
                print('\tlambda = ' + str(stepSize))
        return stepSize

    def MLTR(self, g, f, numIter, numSubsets=1, filters=None, mask=None):
        r"""Maximum Likelihood Transmission reconstruction
        
        This function maximizes the Maximum Likelihood function of CT transmission data which assumes a Poisson noise model.
        This algorithm best models the noise for very low transmission/ low count rate data.
        The MLTR cost function is given by the following
        
        .. math::
           \begin{eqnarray}
             C_{MLTR}(f) &:=& \left< -t\log\left(e^{-Pf}\right) + e^{-Pf} , 1 \right>
           \end{eqnarray}

        where :math:`t = e^{-g}` is the transmission data and \"1\" is a vector of all ones.  The inner product notation is
        just use for simplicity, but all it is really doing is performing a sum over all the elements.
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.

        Args:
            g (C contiguous float32 numpy or torch array): projection data
            f (C contiguous float32 numpy or torch array): volume data
            numIter (int): number of iterations
            numSubsets (int): number of subsets (reduces the time it takes for this algorithm to converge)
            filters (filterSequence object): list of differentiable regularization filters
            mask (C contiguous float32 numpy or torch array): projection data to mask out bad data, etc. (zero values indicate projection data pixels not to use)
        
        Returns:
            f, the same as the input with the same name
        """
        #if has_torch == True and type(f) is torch.Tensor:
        #    print('ERROR: MLTR reconstruction algorithms not implemented for torch tensors!')
        #    print('Please convert to numpy array prior to running this algorithm.')
        #    return f
        if mask is not None and mask.shape != g.shape:
            print('Error: mask must be the same shape as the projection data!')
            return None
            
        if self.verify_inputs(g,f) == False:
            return None
            
        if filters is None:
            filters = filterSequence(0.0)
        elif isinstance(filters, (int, float)):
            print('Error: invalid inputs!  Note that algorithm syntax has changed, please see documentation.')
            return None
        elif filters.count() == 0:
            print('Warning: filterSequence given, but has no terms!')
            filters.beta = 0.0
        
        numSubsets = max(1,numSubsets)
        filters.beta = max(0.0, filters.beta/float(numSubsets))
    
        self.expNeg(g)
        t = g

        if self.isAllZeros(f) == False:
            f[f<=0.0] = 0.0
        
        d = self.allocateData(f)
        
        d[:] = 1.0
        P1 = self.allocateData(g)
        self.project(P1,d)
        P1[P1<=0.0] = 1.0

        SQS = self.allocateData(f)
        if numSubsets <= 1:
            Pf = self.allocateData(g)
            
            transDiff = self.allocateData(g)
            for n in range(numIter):
                if self.print_warnings:
                    print('ML-TR iteration ' + str(n+1) + ' of ' + str(numIter))
                self.project(Pf,f)
                
                transDiff[:] = Pf[:]
                self.expNeg(transDiff)
                
                if mask is not None:
                    transDiff[:] = transDiff[:] * P1[:] * mask[:]
                else:
                    transDiff[:] = transDiff[:] * P1[:]
                self.backproject(transDiff, SQS)
                SQS[SQS<=0.0] = 1.0
                SQS[:] = 1.0 / SQS[:]
                
                if mask is not None:
                    transDiff[:] = transDiff[:]/P1[:] - t[:]*mask[:]
                else:
                    transDiff[:] = transDiff[:]/P1[:] - t[:]
                self.backproject(transDiff, d)
                
                # Regularizer and divide by SQS
                stepMultiplier = 1.0
                if filters.beta > 0.0:
                    #Sf1 = self.TVgradient(f, delta, beta)
                    Sf1 = filters.gradient(f)
                    d[:] -= Sf1[:]
                    grad_dot_descent = self.innerProd(d,d,SQS)
                
                    d[:] = d[:] * SQS[:]
                
                    #stepMultiplier = grad_dot_descent / (grad_dot_descent + self.TVquadForm(f,d, delta, beta))
                    stepMultiplier = grad_dot_descent / (grad_dot_descent + filters.quadForm(f,d))
                else:
                    d[:] = d[:] * SQS[:]
                
                f[:] = f[:] + stepMultiplier*d[:]
                f[f<0.0] = 0.0
                
        else:
            subsetParams = subsetParameters(self, numSubsets)
            t_subsets = self.breakIntoSubsets(t, numSubsets)
            P1_subsets = self.breakIntoSubsets(P1, numSubsets)
            if mask is not None:
                mask_subsets = self.breakIntoSubsets(mask, numSubsets)
            else:
                mask_subsets = None
            for n in range(numIter):
                if self.print_warnings:
                    print('ML-TR iteration ' + str(n+1) + ' of ' + str(numIter))
                for m in range(numSubsets):
                    subsetParams.setSubset(m)
                    transDiff = self.allocateData(t_subsets[m])
                    Pf = self.allocateData(t_subsets[m])
                    
                    self.project(Pf,f)
                    
                    transDiff[:] = Pf[:]
                    self.expNeg(transDiff)
                    
                    if mask_subsets is not None:
                        transDiff[:] = transDiff[:] * P1_subsets[m][:] * mask_subsets[m][:]
                    else:
                        transDiff[:] = transDiff[:] * P1_subsets[m][:]
                    self.backproject(transDiff, SQS)
                    SQS[SQS<=0.0] = 1.0
                    SQS[:] = 1.0 / SQS[:]
                    
                    if mask_subsets is not None:
                        transDiff[:] = transDiff[:]/P1_subsets[m][:] - t_subsets[m][:] * mask_subsets[m][:]
                    else:
                        transDiff[:] = transDiff[:]/P1_subsets[m][:] - t_subsets[m][:]
                    self.backproject(transDiff, d)
                    
                    # Regularizer and divide by SQS
                    stepMultiplier = 1.0
                    if filters.beta > 0.0:
                        #Sf1 = self.TVgradient(f, delta, beta)
                        Sf1 = filters.gradient(f)
                        d[:] -= Sf1[:]
                        grad_dot_descent = self.innerProd(d,d,SQS)
                    
                        d[:] = d[:] * SQS[:]
                    
                        #stepMultiplier = grad_dot_descent / (grad_dot_descent + self.TVquadForm(f,d, delta, beta))
                        stepMultiplier = grad_dot_descent / (grad_dot_descent + filters.quadForm(f,d))
                    else:
                        d[:] = d[:] * SQS[:]
                    
                    f[:] = f[:] + stepMultiplier*d[:]
                    f[f<0.0] = 0.0
        
            subsetParams.setSubset(-1)
        
        g = t
        self.negLog(g)
        filters.beta = filters.beta*float(numSubsets)
        
        return f
            

    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS EXECUTE LEAP'S GPU DENOISING FILTERS
    ###################################################################################################################
    ###################################################################################################################
    def applyTransferFunction(self, x,  LUT, sampleRate, firstSample=0.0):
        """Applies a transfer function to arbitrary 3D data, i.e., x = LUT(x)
        
        Args:
            x (2D or 3D C contiguous float32 numpy array or torch tensor): 3D data (input and output)
            LUT (C contiguous float32 numpy array or torch tensor): lookup table with transfer function values
            sampleRate (float): the step size between samples
            firstSample (float): the value of the first sample in the lookup table

        Returns:            
            true if operation  was sucessful, false otherwise
        """
        
        if len(x.shape) == 2:
            numAngles = 1
            numRows, numCols = x.shape
        else:
            numAngles, numRows, numCols = x.shape
        
        #bool applyTransferFunction(float* x, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, bool data_on_cpu)
        self.libprojectors.applyTransferFunction.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(x) is torch.Tensor:
            self.libprojectors.applyTransferFunction.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.applyTransferFunction(x.data_ptr(), numAngles, numRows, numCols, LUT.data_ptr(), firstSample, sampleRate, LUT.size, f.is_cuda == False)
        else:
            self.libprojectors.applyTransferFunction.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.applyTransferFunction(x, numAngles, numRows, numCols, LUT, firstSample, sampleRate, LUT.size, True)
    
    def beam_hardening_heel_effect(self, g, anode_normal, LUT, takeOffAngles, sampleRate, firstSample=0.0):
       r""" This function performs beam hardening/ beam hardening correction for variable take-off angles
       
       The CT geometry parameters must be set prior to running this function.
       The anode normal is in a reference frame where the detector normal is the y-axis
       
       Args:
           g (C contiguous float32 numpy array or torch tensor): 3D data of first component (input and output)
           anode_normal (3-element array): unit vector normal to the anode
           LUT (2D contiguous float32 numpy array or torch tensor): lookup table with transfer function values
           takeOffAngles (1D numpy array or torch tensor): the takeoff angles (degrees) modeled in the lookup table
           sampleRate (float): the step size between samples
           firstSample (float): the value of the first sample in the lookup table
       
       """
       if has_torch == True and type(g) is torch.Tensor:
           print('Error: beam_hardening_heel_effect not yet implemented for torch tensors!')
           return False
       else:
           self.libprojectors.beam_hardening_heel_effect.restype = ctypes.c_bool
           self.libprojectors.beam_hardening_heel_effect.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
           self.set_model()
           return self.libprojectors.beam_hardening_heel_effect(g, anode_normal, LUT, takeOffAngles, LUT.shape[1], LUT.shape[0], sampleRate, firstSample, True)
    
    def applyDualTransferFunction(self, x, y,  LUT, sampleRate, firstSample=0.0):
        """Applies a 2D transfer function to arbitrary 3D data pair, i.e., x,y = LUT(x,y)
        
        Args:
            x (2D or 3D C contiguous float32 numpy array or torch tensor): 3D data of first component (input and output)
            y (2D or 3D C contiguous float32 numpy array or torch tensor): 3D data of second component (input and output)
            LUT (C contiguous float32 numpy array or torch tensor): lookup table with transfer function values
            sampleRate (float): the step size between samples
            firstSample (float): the value of the first sample in the lookup table

        Returns:            
            true if operation  was sucessful, false otherwise
        """
        
        if len(LUT.shape) == 2:
            LUT_2d = LUT
            LUT = np.zeros((2,LUT.shape[0],LUT.shape[1]),dtype=np.float32)
            if has_torch == True and type(LUT_2d) is torch.Tensor:
                LUT = torch.from_numpy(LUT)
                if LUT_2d.is_cuda:
                    LUT = LUT.float().to(LUT_2d.get_device())
            LUT[0,:,:] = LUT_2d[:,:]
            LUT[1,:,:] = LUT_2d[:,:]
        
        if len(x.shape) == 2:
            numAngles = 1
            numRows, numCols = x.shape
        else:
            numAngles, numRows, numCols = x.shape
        
        #bool applyDualTransferFunction(float* x, float* y, int N_1, int N_2, int N_3, float* LUT, float firstSample, float sampleRate, int numSamples, bool data_on_cpu)
        self.libprojectors.applyDualTransferFunction.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(x) is torch.Tensor:
            self.libprojectors.applyDualTransferFunction.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.applyDualTransferFunction(x.data_ptr(), y.data_ptr(), numAngles, numRows, numCols, LUT.data_ptr(), firstSample, sampleRate, LUT.shape[1], x.is_cuda == False)
        else:
            self.libprojectors.applyDualTransferFunction.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.applyDualTransferFunction(x, y, numAngles, numRows, numCols, LUT, firstSample, sampleRate, LUT.shape[1], True)
    
    def convertToRhoeZe(self, f_L, f_H, sigma_L, sigma_H):
        """transforms a low and high energy pair to electron density and effective atomic number
        
        Args:
            f_L (2D or 3D C contiguous float32 numpy array or torch tensor): low energy volume in LAC units
            f_H (2D or 3D C contiguous float32 numpy array or torch tensor): high energy volume in LAC units
            sigma_L (3D C contiguous float32 numpy array or torch tensor): mass cross section values for elements 1-100 at the low energy
            sigma_H (3D C contiguous float32 numpy array or torch tensor): mass cross section values for elements 1-100 at the high energy
        
        Returns:
            the Ze and rho volumes
        
        """
        
        if len(f_L.shape) == 2:
            numZ = 1
            numY, numX = f_L.shape
        else:
            numZ, numY, numX = f_L.shape
        
        #bool convertToRhoeZe(float* f_L, float* f_H, int N_1, int N_2, int N_3, float* sigma_L, float* sigma_H, bool data_on_cpu)
        self.libprojectors.convertToRhoeZe.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f_L) is torch.Tensor:
            self.libprojectors.convertToRhoeZe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_bool]
            self.libprojectors.convertToRhoeZe(f_L.data_ptr(), f_H.data_ptr(), numZ, numY, numX, sigma_L.data_ptr(), sigma_H.data_ptr(), f_L.is_cuda == False)
        else:
            self.libprojectors.convertToRhoeZe.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            self.libprojectors.convertToRhoeZe(f_L, f_H, numZ, numY, numX, sigma_L, sigma_H, True)
        return f_L, f_H
    
    def synthesize_symmetry(self, f_radial):
        """Converts symmetric volume to a 3D volume
        
        The CT volume parameters must be set prior to running this function and must be specified as symmetric.
        
        Args:
            f_radial (C contiguous float32 numpy array): symmetric volume numpy array
            
        Returns:
            3D volume
        """
        
        dim1, dim2, dim3 = self.get_volume_dim()
        if dim1 <= 0 or dim2 <= 0 and dim3 <= 0:
            print('Error: CT volume parameters not set')
            return None
        
        if has_torch == True and type(f_radial) is torch.Tensor:
            print('Error: not yet implemented for pytorch tensors')
            return None
        else:
            f = np.ascontiguousarray(np.zeros((dim1,dim2,dim2),dtype=np.float32), dtype=np.float32)
            
            self.libprojectors.synthesize_symmetry.restype = ctypes.c_bool
            self.set_model()
            self.libprojectors.synthesize_symmetry.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.synthesize_symmetry(f_radial, f)
            return f
    
    def LowPassFilter(self, f, FWHM=2.0):
        """Alias for BlurFilter"""
        return self.BlurFilter(f, FWHM)
    
    def BlurFilter(self, f, FWHM=2.0):
        """Applies a blurring filter to the provided numpy array or torch tensor
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        The filter is given by cos^2(pi/(2*FWHM) * i), i = -ceil(FWHM), ..., ceil(FWHM)
        This filter is very simular to a Gaussian filter, but is a FIR
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): numpy array to smooth
            FWHM (float): the full width at half maximum (in number of pixels) of the filter
        
        Returns:
            f, the same as the input
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        #bool BlurFilter(float* f, int, int, int, float FWHM);
        self.libprojectors.BlurFilter.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.BlurFilter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.BlurFilter(f.data_ptr(), N_1, N_2, N_3, FWHM, f.is_cuda == False)
        else:
            self.libprojectors.BlurFilter.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.BlurFilter(f, N_1, N_2, N_3, FWHM, True)

    def HighPassFilter(self, f, FWHM=2.0):
        """Applies a high pass filter to the provided numpy array or torch tensor
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array or torch tensor of any size
        The filter is given by delta[i] - cos^2(pi/(2*FWHM) * i), i = -ceil(FWHM), ..., ceil(FWHM)
        This filter is very simular to a Gaussian filter, but is a FIR
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): numpy array to sharpen
            FWHM (float): the full width at half maximum (in number of pixels) of the filter
        
        Returns:
            f, the same as the input
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        #bool HighPassFilter(float* f, int, int, int, float FWHM);
        self.libprojectors.HighPassFilter.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.HighPassFilter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.HighPassFilter(f.data_ptr(), N_1, N_2, N_3, FWHM, f.is_cuda == False)
        else:
            self.libprojectors.HighPassFilter.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.HighPassFilter(f, N_1, N_2, N_3, FWHM, True)

    def LowPassFilter2D(self, f, FWHM=2.0):
        """Alias for BlurFilter2D"""
        return self.BlurFilter2D(f, FWHM)
            
    def BlurFilter2D(self, f, FWHM=2.0):
        """Applies a 2D blurring filter to the provided numpy array or torch tensor
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        The filter is given by cos^2(pi/(2*FWHM) * i), i = -ceil(FWHM), ..., ceil(FWHM)
        This filter is very simular to a Gaussian filter, but is a FIR
        
        Args:
            f (C contiguous float32 numpy array): numpy array to smooth
            FWHM (float): the full width at half maximum (in number of pixels) of the filter
        
        Returns:
            f, the same as the input
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        #bool BlurFilter2D(float* f, int, int, int, float FWHM);
        self.libprojectors.BlurFilter2D.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.BlurFilter2D.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.BlurFilter2D(f.data_ptr(), N_1, N_2, N_3, FWHM, f.is_cuda == False)
        else:
            self.libprojectors.BlurFilter2D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.BlurFilter2D(f, N_1, N_2, N_3, FWHM, True)
            
    def HighPassFilter2D(self, f, FWHM=2.0):
        """Applies a 2D high pass filter to the provided numpy array or torch tensor
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array or torch tensor of any size
        The filter is given by delta[i] - cos^2(pi/(2*FWHM) * i), i = -ceil(FWHM), ..., ceil(FWHM)
        This filter is very simular to a Gaussian filter, but is a FIR
        
        Args:
            f (C contiguous float32 numpy array): numpy array to smooth
            FWHM (float): the full width at half maximum (in number of pixels) of the filter
        
        Returns:
            f, the same as the input
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        #bool HighPassFilter2D(float* f, int, int, int, float FWHM);
        self.libprojectors.HighPassFilter2D.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.HighPassFilter2D.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.HighPassFilter2D(f.data_ptr(), N_1, N_2, N_3, FWHM, f.is_cuda == False)
        else:
            self.libprojectors.HighPassFilter2D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.HighPassFilter2D(f, N_1, N_2, N_3, FWHM, True)
            
    def MeanFilter(self, x, windowRadius=1):
        r"""Applies a 3D mean filter to the provided numpy array
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        This algorithm performs a 3D (2*r+1)^3 mean around each data value
        
        Args:
            x (C contiguous float32 numpy array or torch tensor): array to filter
            windowRadius (int): the radius of the window
        
        Returns:
            x, the same as the input
        """
        
        if len(x.shape) == 3:
            N_1 = x.shape[0]
            N_2 = x.shape[1]
            N_3 = x.shape[2]
        elif len(x.shape) == 2:
            N_1 = 1
            N_2 = x.shape[0]
            N_3 = x.shape[1]
        
        self.libprojectors.MeanOrVarianceFilter.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(x) is torch.Tensor:
            self.libprojectors.MeanOrVarianceFilter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.MeanOrVarianceFilter(x.data_ptr(), N_1, N_2, N_3, windowRadius, 1, x.is_cuda == False)
        else:
            self.libprojectors.MeanOrVarianceFilter.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.MeanOrVarianceFilter(x, N_1, N_2, N_3, windowRadius, 1, True)
            
    def VarianceFilter(self, x, windowRadius=1):
        r"""Applies a 3D variance filter to the provided numpy array
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        This algorithm performs a 3D (2*r+1)^3 variance around each data value
        
        Args:
            x (C contiguous float32 numpy array or torch tensor): array to filter
            windowRadius (int): the radius of the window
        
        Returns:
            x, the same as the input
        """
        
        if len(x.shape) == 3:
            N_1 = x.shape[0]
            N_2 = x.shape[1]
            N_3 = x.shape[2]
        elif len(x.shape) == 2:
            N_1 = 1
            N_2 = x.shape[0]
            N_3 = x.shape[1]
        
        self.libprojectors.MeanOrVarianceFilter.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(x) is torch.Tensor:
            self.libprojectors.MeanOrVarianceFilter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.MeanOrVarianceFilter(x.data_ptr(), N_1, N_2, N_3, windowRadius, 2, x.is_cuda == False)
        else:
            self.libprojectors.MeanOrVarianceFilter.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.MeanOrVarianceFilter(x, N_1, N_2, N_3, windowRadius, 2, True)
    
    def LowSignalCorrection(self, f, threshold=0.0, windowSize=3, signalThreshold=0.001):
        r"""Same as MedianFilter, but only filters those values that are lower than the specified threshold
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        #bool MedianFilter(float* f, int, int, int, float threshold);
        self.libprojectors.MedianFilter.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.MedianFilter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.MedianFilter(f.data_ptr(), N_1, N_2, N_3, threshold, windowSize, signalThreshold, f.is_cuda == False)
        else:
            self.libprojectors.MedianFilter.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.MedianFilter(f, N_1, N_2, N_3, threshold, windowSize, signalThreshold, True)
    
    def MedianFilter(self, f, threshold=0.0, windowSize=3):
        r"""Applies a thresholded 3D median filter (3x3x3 or 3x5x5) to the provided array
        
        The provided input does not have to be projection or volume data. It can be any 3D array of any size
        This algorithm performs a 3D (3x3x3 or 3x5x5) median around each data value and then replaces this value only if
        \|original value - median value\| >= threshold*\|median value\|
        Note that if threshold is zero, then this is simply a median filter
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): 3D array to filter
            threshold (float): the threshold of whether to use the filtered value or not
            windowSize (int): the window size; can be 3 or 5
        
        Returns:
            f, the same as the input
        """
        return self.LowSignalCorrection(f, threshold, windowSize, 0.0)
    
    def LowSignalCorrection2D(self, f, threshold=0.0, windowSize=3, signalThreshold=0.001):
        r"""Same as MedianFilter2D, but only filters those values that are lower than the specified threshold
        """
        #bool MedianFilter2D(float* f, int, int, int, float threshold, int windowSize);
        self.libprojectors.MedianFilter2D.restype = ctypes.c_bool
        self.set_model()
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.MedianFilter2D.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.MedianFilter2D(f.data_ptr(), N_1, N_2, N_3, threshold, windowSize, signalThreshold, f.is_cuda == False)
        else:
            self.libprojectors.MedianFilter2D.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.MedianFilter2D(f, N_1, N_2, N_3, threshold, windowSize, signalThreshold, True)
    
    def MedianFilter2D(self, f, threshold=0.0, windowSize=3):
        r"""Applies a thresholded 2D median filter (windowSize x windowSize) to the provided array
        
        The provided input does not have to be projection or volume data. It can be any 3D array of any size
        This algorithm performs a 2D (windowSize x windowSize) median around each data value and then replaces this value only if
        \|original value - median value\| >= threshold*\|median value\|
        Note that if threshold is zero, then this is simply a median filter
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): 3D array to denoise
            threshold (float): the threshold of whether to use the filtered value or not
            windowSize (int): the window size; can be 3, 5, or 7
        
        Returns:
            True is successful, False otherwise
        """
        return self.LowSignalCorrection2D(f, threshold, windowSize, 0.0)
    
    def badPixelCorrection(self, g, badPixelMap, windowSize=3):
        r"""Bad Pixel Correction
        
        The provided input must be projection data and the CT geometry parameters must be set.
        This algorithm processes each projection independently
        and removes bad pixels specified by the user using a median filter.
        
        Args:
            g (C contiguous float32 numpy array or torch tensor): 3D projection data array
            badPixelMap (C contiguous float32 numpy array or torch tensor): 2D bad pixel map (numRows x numCols)
            windowSize (int): the window size; can be 3, 5, or 7
        
        Returns:
            True is successful, False otherwise
        """
        
        if len(g.shape) == 3:
            numAngles = g.shape[0]
            numRows = g.shape[1]
            numCols = g.shape[2]
        elif len(g.shape) == 2:
            numAngles = 1
            numRows = g.shape[0]
            numCols = g.shape[1]
        
        #if len(g.shape) != 3 or g.shape[0] != self.get_numAngles() or g.shape[1] != self.get_numRows() or g.shape[2] != self.get_numCols():
        #    print('Error: input data dimensions do not match CT data dimensions')
        #    return False
        if len(badPixelMap.shape) != 2 or numRows != badPixelMap.shape[0] or numCols != badPixelMap.shape[1]:
            print('Error: bad pixel map dimensions do not match CT data dimensions')
            return False
        if type(g) != type(badPixelMap):
            print('Error: projection data and bad pixel map must be the same type')
            return False
        
        #bool MedianFilter2D(float* f, int, int, int, float threshold, int windowSize);
        self.libprojectors.badPixelCorrection.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
        
            if g.is_cuda != badPixelMap.is_cuda:
                print('Error: projection data and bad pixel map must both be on the cpu or both be on the same gpu')
                return False
        
            self.libprojectors.badPixelCorrection.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.badPixelCorrection(g.data_ptr(), numAngles, numRows, numCols, badPixelMap.data_ptr(), windowSize, g.is_cuda == False)
        else:
            self.libprojectors.badPixelCorrection.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.badPixelCorrection(g, numAngles, numRows, numCols, badPixelMap, windowSize, True)
    
    def PriorBilateralFilter(self, f, spatialFWHM, intensityFWHM, prior=None):
        """Performs 3D Bilateral Filter (BLF) denoising method where the intensity distance is measured against a prior image
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): 3D array to denoise
            spatialFWHM (float): the FWHM (in number of pixels) of the spatial closeness term of the BLF
            intensityFWHM (float): the FWHM of the intensity closeness terms of the BLF
            prior (C contiguous float32 numpy array or torch tensor): 3D data prior used for the intensity distance
        
        Returns:
            f, the same as the input
        
        """
        if prior is None or isinstance(prior, (int, float)):
            return self.BilateralFilter(f, spatialFWHM, intensityFWHM, prior)
            
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        self.libprojectors.PriorBilateralFilter.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.PriorBilateralFilter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_void_p, ctypes.c_bool]
            return self.libprojectors.PriorBilateralFilter(f.data_ptr(), N_1, N_2, N_3, spatialFWHM, intensityFWHM, prior.data_ptr(), f.is_cuda == False)
        else:
            self.libprojectors.PriorBilateralFilter.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_bool]
            return self.libprojectors.PriorBilateralFilter(f, N_1, N_2, N_3, spatialFWHM, intensityFWHM, prior, True)
    
    def BilateralFilter(self, f, spatialFWHM, intensityFWHM, scale=1.0):
        """Performs 3D (Scaled) Bilateral Filter (BLF) denoising method
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): 3D array to denoise
            spatialFWHM (float): the FWHM (in number of pixels) of the spatial closeness term of the BLF
            intensityFWHM (float): the FWHM of the intensity closeness terms of the BLF
            scale (float): an optional argument to used a blurred volume (and this parameter specifies the FWHM of the blurring) to calculate the intensity closeness term
        
        Returns:
            f, the same as the input
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        self.libprojectors.BilateralFilter.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.BilateralFilter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.BilateralFilter(f.data_ptr(), N_1, N_2, N_3, spatialFWHM, intensityFWHM, scale, f.is_cuda == False)
        else:
            self.libprojectors.BilateralFilter.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.BilateralFilter(f, N_1, N_2, N_3, spatialFWHM, intensityFWHM, scale, True)
            
    def GuidedFilter(self, f, r, epsilon, numIter=1):
        """Performs 3D Guided Filter denoising method
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): 3D array to denoise
            r (int): the window radius (in number of pixels)
            epsilon (float): the degree of smoothing
            numIter (int): the number of iterations
        
        Returns:
            f, the same as the input
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        r = min(r,10)
        self.libprojectors.GuidedFilter.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.GuidedFilter.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.GuidedFilter(f.data_ptr(), N_1, N_2, N_3, r, epsilon, numIter, f.is_cuda == False)
        else:
            self.libprojectors.GuidedFilter.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.GuidedFilter(f, N_1, N_2, N_3, r, epsilon, numIter, True)
    
    def DictionaryDenoising(self, f, dictionary, sparsityThreshold=8, epsilon=0.0):
        """Represents 3D data by a sparse representation of an overcomplete dictionary, effectively denoising the data
        
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): 3D array to denoise
            dictionary (C contiguous float32 numpy array): 4D array of dictionary patches
            sparsityThreshold (int): the maximum number of dictionary elements to use to represent a patch in the volume
            epsilon (float): the L^2 residual threshold to decide of the sparse dictionary representation is close enough (larger numbers perform stronger denoising)
        
        Returns:
            f, the same as the input
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        #bool dictionaryDenoising(float* f, int N_1, int N_2, int N_3, float* dictionary, int numElements, int N_d1, int N_d2, int N_d3, float epsilon, int sparsityThreshold, bool data_on_cpu);
        self.libprojectors.dictionaryDenoising.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.dictionaryDenoising.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.dictionaryDenoising(f.data_ptr(), N_1, N_2, N_3, dictionary, dictionary.shape[0], dictionary.shape[1], dictionary.shape[2], dictionary.shape[3], epsilon, sparsityThreshold, f.is_cuda == False)
        else:
            self.libprojectors.dictionaryDenoising.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            return self.libprojectors.dictionaryDenoising(f, N_1, N_2, N_3, dictionary, dictionary.shape[0], dictionary.shape[1], dictionary.shape[2], dictionary.shape[3], epsilon, sparsityThreshold, True)
    
    def get_numTVneighbors(self):
        """Gets the number of neighboring voxels to use for 3D TV"""
        self.libprojectors.get_numTVneighbors.restype = ctypes.c_int
        self.set_model()
        #self.libprojectors.get_numTVneighbors.argtypes = [ctypes.c_int]
        return self.libprojectors.get_numTVneighbors()
        
    def set_numTVneighbors(self, N):
        """Sets the number of neighboring voxels to use for 3D TV
        
        Args:
            N (int): the number of neighbors to use for 3D TV calculations (can be 6 or 26)
        
        """
        self.libprojectors.set_numTVneighbors.restype = ctypes.c_bool
        self.set_model()
        self.libprojectors.set_numTVneighbors.argtypes = [ctypes.c_int]
        return self.libprojectors.set_numTVneighbors(N)
    
    def TVcost(self, f, delta, beta=0.0, p=1.2):
        r"""Calculates the anisotropic Total Variation (TV) functional, i.e., cost of the provided numpy array
        
        This function uses a Huber-like loss function applied to the differences of neighboring samples (in 3D).
        One can switch between using 6 or 26 neighbors using the \"set_numTVneighbors\" function.
        The aTV functional with Huber-like loss function is given by
        
        .. math::
           \begin{eqnarray}
             R(x) &:=& \sum_{\boldsymbol{i}} \sum_{\boldsymbol{j} \in N_{\boldsymbol{i}}} \|\boldsymbol{i} - \boldsymbol{j}\|^{-1} h(x_\boldsymbol{i} - x_\boldsymbol{j}) \\
             h(t) &:=& \begin{cases} \frac{1}{2}t^2, & \text{if } |t| \leq delta \\ \frac{delta^{2 - p}}{p}|t|^p + delta^2\left(\frac{1}{2} - \frac{1}{p}\right), & \text{if } |t| > delta \end{cases}
           \end{eqnarray}

        where :math:`N_{\boldsymbol{i}}` is a neighborhood around the 3D pixel index :math:`\boldsymbol{i} = (i_1, i_2, i_3)`.
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size.
        
        Args:
            f (C contiguous float32 numpy array): 3D numpy array
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): TV multiplier (sometimes called the regularizaion strength)
            p (float): the exponent for the Huber-like loss function used in TV
        
        Returns:
            TV functional value
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        #float TVcost(float* f, int N_1, int N_2, int N_3, float delta, float beta);
        self.libprojectors.TVcost.restype = ctypes.c_float
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.TVcost.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.TVcost(f.data_ptr(), N_1, N_2, N_3, delta, beta, p, f.is_cuda == False)
        else:
            self.libprojectors.TVcost.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.TVcost(f, N_1, N_2, N_3, delta, beta, p, True)
        
    def TVgradient(self, f, delta, beta=0.0, p=1.2):
        r"""Calculates the gradient of the anisotropic Total Variation (TV) functional of the provided numpy array
        
        This function uses a Huber-like loss function applied to the differences of neighboring samples (in 3D).
        One can switch between using 6 or 26 neighbors using the \"set_numTVneighbors\" function.
        The aTV functional with Huber-like loss function is given by
        
        .. math::
           \begin{eqnarray}
             R(x) &:=& \sum_{\boldsymbol{i}} \sum_{\boldsymbol{j} \in N_{\boldsymbol{i}}} \|\boldsymbol{i} - \boldsymbol{j}\|^{-1} h(x_\boldsymbol{i} - x_\boldsymbol{j}) \\
             h(t) &:=& \begin{cases} \frac{1}{2}t^2, & \text{if } |t| \leq delta \\ \frac{delta^{2 - p}}{p}|t|^p + delta^2\left(\frac{1}{2} - \frac{1}{p}\right), & \text{if } |t| > delta \end{cases} \\
             h'(t) &=& \begin{cases} t, & \text{if } |t| \leq delta \\ delta^{2 - p}sgn(t)|t|^{p-1}, & \text{if } |t| > delta \end{cases}
           \end{eqnarray}

        where :math:`N_{\boldsymbol{i}}` is a neighborhood around the 3D pixel index :math:`\boldsymbol{i} = (i_1, i_2, i_3)`.
        The provided input does not have to be projection or volume data. It can be any 3D numpy array of any size
        
        Args:
            f (C contiguous float32 numpy array): 3D numpy array
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): TV multiplier (sometimes called the regularizaion strength)
            p (float): the exponent for the Huber-like loss function used in TV
        
        Returns:
            Df (C contiguous float32 numpy array): the gradient of the TV functional applied to the input
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        #bool TVgradient(float* f, float* Df, int N_1, int N_2, int N_3, float delta, float beta);
        self.libprojectors.TVgradient.restype = ctypes.c_bool
        
        if has_torch == True and type(f) is torch.Tensor:
            Df = f.clone()
            self.set_model()
            self.libprojectors.TVgradient.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            self.libprojectors.TVgradient(f.data_ptr(), Df.data_ptr(), N_1, N_2, N_3, delta, beta, p, f.is_cuda == False)
            return Df
        else:
            Df = np.ascontiguousarray(np.zeros(f.shape,dtype=np.float32), dtype=np.float32)
            self.set_model()
            self.libprojectors.TVgradient.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            self.libprojectors.TVgradient(f, Df, N_1, N_2, N_3, delta, beta, p, True)
            return Df
    
    def TVquadForm(self, f, d, delta, beta=0.0, p=1.2):
        r"""Calculates the quadratic form of the anisotropic Total Variation (TV) functional of the provided numpy arrays
        
        The provided inputs does not have to be projection or volume data. It can be any 3D numpy array of any size
        This function calculates the following inner product <d, R''(f)d>, where R'' is the Hessian of the TV functional
        The quadraitc surrogate is used here, so this function can be used to calculate the step size of a cost function
        that includes a TV regularization term.
        See the same  cost in the diffuse function below for an example of its usage.
        
        This function uses a Huber-like loss function applied to the differences of neighboring samples (in 3D).
        One can switch between using 6 or 26 neighbors using the \"set_numTVneighbors\" function.
        The aTV functional with Huber-like loss function is given by
        
        .. math::
           \begin{eqnarray}
             R(x) &:=& \sum_{\boldsymbol{i}} \sum_{\boldsymbol{j} \in N_{\boldsymbol{i}}} \|\boldsymbol{i} - \boldsymbol{j}\|^{-1} h(x_\boldsymbol{i} - x_\boldsymbol{j}) \\
             h(t) &:=& \begin{cases} \frac{1}{2}t^2, & \text{if } |t| \leq delta \\ \frac{delta^{2 - p}}{p}|t|^p + delta^2\left(\frac{1}{2} - \frac{1}{p}\right), & \text{if } |t| > delta \end{cases}
           \end{eqnarray}

        where :math:`N_{\boldsymbol{i}}` is a neighborhood around the 3D pixel index :math:`\boldsymbol{i} = (i_1, i_2, i_3)`.
        To make this calculate a quadraitc surrogate (upper bound), LEAP uses h'(t)/t instead of h''(t).
        
        Args:
            f (C contiguous float32 numpy array): 3D numpy array
            d (C contiguous float32 numpy array): 3D numpy array
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): TV multiplier (sometimes called the regularizaion strength)
            p (float): the exponent for the Huber-like loss function used in TV
        
        Returns:
            Df (C contiguous float32 numpy array): the gradient of the TV functional applied to the input
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        #float TVquadForm(float* f, float* d, int N_1, int N_2, int N_3, float delta, float beta);
        self.libprojectors.TVquadForm.restype = ctypes.c_float
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.TVquadForm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.TVquadForm(f.data_ptr(), d.data_ptr(), N_1, N_2, N_3, delta, beta, p, f.is_cuda == False)
        else:
            self.libprojectors.TVquadForm.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
            return self.libprojectors.TVquadForm(f, d, N_1, N_2, N_3, delta, beta, p, True)
        
    def diffuse(self, f, delta, numIter, p=1.2):
        r"""Performs anisotropic Total Variation (TV) smoothing to the provided 3D numpy array
        
        The provided inputs does not have to be projection or volume data. It can be any 3D numpy array of any size.
        This function performs a specifies number of iterations of minimizing the aTV functional using gradient descent.
        The step size calculation uses the method of Separable Quadratic Surrogate (see also TVquadForm).
        
        Args:
            f (C contiguous float32 numpy array): 3D numpy array
            delta (float): parameter for the Huber-like loss function used in TV
            numIter (int): number of iterations
            p (float): the exponent for the Huber-like loss function used in TV
        
        Returns:
            f, the same array as the input denoised
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        self.libprojectors.Diffuse.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.Diffuse.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            self.libprojectors.Diffuse(f.data_ptr(), N_1, N_2, N_3, delta, p, numIter, f.is_cuda == False)
        else:
            self.libprojectors.Diffuse.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_bool]
            self.libprojectors.Diffuse(f, N_1, N_2, N_3, delta, p, numIter, True)
        return f
        ''' Here is equivalent code to run this algorithm using the TV functions above
        for n in range(N):
            d = self.TVgradient(f, delta, p)
            num = np.sum(d**2)
            denom = self.TVquadForm(f, d, delta, p)
            if denom <= 1.0e-16:
                break
            stepSize = num / denom
            f -= stepSize * d
        return f
        '''
        
    def TV_denoise(self, f, delta, beta, numIter, p=1.2, meanOverFirstDim=False):
        r"""Performs anisotropic Total Variation (TV) denoising to the provided 3D numpy array
        
        The provided inputs does not have to be projection or volume data. It can be any 3D numpy array of any size.
        This function performs a specifies number of iterations of minimizing the sum of an L2 loss and aTV functional using gradient descent.
        The step size calculation uses the method of Separable Quadratic Surrogate (see also TVquadForm).
        
        Args:
            f (C contiguous float32 numpy array): 3D numpy array
            delta (float): parameter for the Huber-like loss function used in TV
            beta (float): regularization strength
            numIter (int): number of iterations
            p (float): the exponent for the Huber-like loss function used in TV
        
        Returns:
            f, the same array as the input denoised
        """
        
        if len(f.shape) == 3:
            N_1 = f.shape[0]
            N_2 = f.shape[1]
            N_3 = f.shape[2]
        elif len(f.shape) == 2:
            N_1 = 1
            N_2 = f.shape[0]
            N_3 = f.shape[1]
        
        self.libprojectors.TV_denoise.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.TV_denoise.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
            self.libprojectors.TV_denoise(f.data_ptr(), N_1, N_2, N_3, delta, beta, p, numIter, meanOverFirstDim, f.is_cuda == False)
        else:
            self.libprojectors.TV_denoise.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_bool, ctypes.c_bool]
            self.libprojectors.TV_denoise(f, N_1, N_2, N_3, delta, beta, p, numIter, meanOverFirstDim, True)
        return f

    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS SET AND GET VARIOUS PARAMETERS, INCLUDING THOSE THAT SET HOW LEAP IS TO BE RUN
    ###################################################################################################################
    ###################################################################################################################
    def number_of_gpus(self):
        self.libprojectors.number_of_gpus.restype = ctypes.c_int
        return self.libprojectors.number_of_gpus()
    
    def set_gpu(self, which):
        """Set which GPU to use, use -1 to do CPU calculations"""
        return self.set_GPU(which)
    
    def set_GPU(self, which):
        """Set which GPU to use, use -1 to do CPU calculations"""
        self.libprojectors.set_GPU.argtypes = [ctypes.c_int]
        self.libprojectors.set_GPU.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_GPU(which)

    def set_all_gpus(self):
        return self.set_gpus(list(range(self.number_of_gpus())))

    def set_gpus(self, listOfGPUs):
        """Set which GPUs to use when doing multi-GPU calculations"""
        return self.set_GPUs(listOfGPUs)
        
    def set_GPUs(self, listOfGPUs):
        """Set which GPUs to use when doing multi-GPU calculations"""
        if len(listOfGPUs) == 0:
            self.set_model()
            return self.libprojectors.set_gpu(-1)
        else:
            self.libprojectors.set_GPUs.argtypes = [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ctypes.c_int]
            self.libprojectors.set_GPUs.restype = ctypes.c_bool
            listOfGPUs = np.ascontiguousarray(listOfGPUs, dtype=np.int32)
            self.set_model()
            return self.libprojectors.set_GPUs(listOfGPUs, int(listOfGPUs.size))
    
    def get_gpus(self):
        """Get the index of all of the GPUs being used"""
        
        self.libprojectors.get_gpus.argtypes = [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS")]
        self.libprojectors.get_gpus.restype = ctypes.c_int
        self.set_model()
        
        if self.number_of_gpus() <= 0:
            return []
        
        possible_gpu_list = -1*np.ones(self.number_of_gpus(), dtype=np.int32)
        N = self.libprojectors.get_gpus(possible_gpu_list)
        gpu_list = np.zeros(N, dtype=np.int32)
        gpu_list[:] = possible_gpu_list[0:N]
        return gpu_list
    
    def get_gpu(self):
        """Get the index of the primary GPU that is being used"""
        return self.get_GPU()
        
    def get_GPU(self):
        """Get the index of the primary GPU that is being used"""
        self.libprojectors.get_GPU.restype = ctypes.c_int
        self.set_model()
        return self.libprojectors.get_GPU()
        
    def set_diameterFOV(self, d):
        """Set the diameterFOV parameter

        This function sets the parameter that specifies the diameter of the circular mask on the reconstruction z-slices.
        If this parameter is not specified, then the mask diameter will be calculated based on the size of the
        reconstructable field of view as determined by the CT geometry parameters (and setting of the offsetScan parameter).
        Applying this mask removes artifacts outside the field of view that can be distracting.  It also provides
        speed improvements and for cone-beam geometries can help algorithms to use less memory.
        If one does not want any masking applied, just provide a very large number to this function's argument.
        
        
        Args:
            d (float): the diameter (mm) of the circular mask on the reconstruction z-slices
        
        """
        self.libprojectors.set_rFOV.argtypes = [ctypes.c_float]
        self.libprojectors.set_rFOV.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_rFOV(0.5*d)
        
    def get_diameterFOV_min(self):
        """Gets the diameter of the reconstructable field of view for non offset scans

        """
        #self.libprojectors.get_rFOV_min.argtypes = [ctypes.c_float]
        self.libprojectors.get_rFOV_min.restype = ctypes.c_float
        self.set_model()
        return 2.0*self.libprojectors.get_rFOV_min()
        
    def get_diameterFOV(self):
        """Gets the diameterFOV parameter

        """
        #self.libprojectors.get_rFOV.argtypes = [ctypes.c_float]
        self.libprojectors.get_rFOV.restype = ctypes.c_float
        self.set_model()
        return 2.0*self.libprojectors.get_rFOV()
        
    def set_truncatedScan(self, aFlag):
        """Set the truncatedScan parameter
        
        One should perform a truncated scan FBP reconstruction, when the object being imaged extends
        past both the right and left sides of the detector, i.e., the projections are truncated.
        In this case, you should use the command: leapct.set_truncatedScan(True) prior to executing an FBP reconstruction,
        so that it uses extrapolation of the signal instead of zero-padding when applying the ramp filter
        this reduces cupping artifacts and other truncation artifacts.
        
        Args:
            aFlag (bool): Set to True to perform a truncated FBP reconstruction, whenever the FBP function is called
        
        """
        self.libprojectors.set_truncatedScan.argtypes = [ctypes.c_bool]
        self.libprojectors.set_truncatedScan.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_truncatedScan(aFlag)
        
    def set_offsetScan(self, aFlag):
        """Set the offsetScan parameter
        
        This function is used to perform an FBP reconstruction where the projections are truncated
        on either the left or the right side (i.e., the object extends past the detector on the left or right side)
        In this case, you should use the command: leapct.set_offsetScan(True)
        This can happen if the detector is shifted horizontally (do this with the centerCol parameter) and/or
        the source is shifted horizontally (do this with the tau parameter).
        This is sometimes refered to as a half-fan or half-cone or half-scan.
        Sometimes this is not on purpose, but in most cases this is done deliberately because it enables one
        to nearly double the diameter of the field of view which is needed for large objects.
        
        Args:
            aFlag (bool): Set to True to perform an offset scan FBP reconstruction, whenever the FBP function is called
        
        """
        self.libprojectors.set_offsetScan.argtypes = [ctypes.c_bool]
        self.libprojectors.set_offsetScan.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_offsetScan(aFlag)
        
    def get_offsetScan(self):
        """Gets the offsetScan setting (True or False)"""
        self.libprojectors.get_offsetScan.argtypes = []
        self.libprojectors.get_offsetScan.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.get_offsetScan()
    
    def set_axisOfSymmetry(self,val):
        """Set the axisOfSymmetry parameter"""
        self.libprojectors.set_axisOfSymmetry.argtypes = [ctypes.c_float]
        self.libprojectors.set_axisOfSymmetry.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_axisOfSymmetry(val)
        
    def get_axisOfSymmetry(self):
        """Gets the axisOfSymmetry parameter"""
        #self.libprojectors.get_axisOfSymmetry.argtypes = []
        self.libprojectors.get_axisOfSymmetry.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_axisOfSymmetry()
        
    def clear_axisOfSymmetry(self):
        """Clears the axisOfSymmetry parameter (revert back to voxelized volume models)"""
        self.libprojectors.clear_axisOfSymmetry.argtypes = []
        self.libprojectors.clear_axisOfSymmetry.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.clear_axisOfSymmetry()
        
    def set_projector(self, which='SF'):
        r"""Set which projector model to use

        Note that all forward projectors use the modified separable footprint model,
        this function only changes the backprojection model.  Voxel-driven backprojection
        is faster, but less accurate

        Args:
            which (string): 'SF' for modified Separable Footprint, 'VD' for Voxel-Driven
        
        Returns:
            True is the input was valid, False otherwise
        """
        
        if isinstance(which, str):
            if which == 'SF':
                which = 2
            elif which == 'VD':
                which = 3
            else:
                return False
        
        self.libprojectors.set_projector.argtypes = [ctypes.c_int]
        self.libprojectors.set_projector.restype = ctypes.c_bool
        self.set_model()
        
        return self.libprojectors.set_projector(which)
        
    def get_projector(self):
        r"""Get which projector model is currently being used
        
        Returns:
            'SF' or 'VD'
        """
        
        self.libprojectors.get_projector.argtypes = []
        self.libprojectors.get_projector.restype = ctypes.c_int
        self.set_model()
        if self.libprojectors.get_projector() == 2:
            return 'SF'
        else:
            return 'VD'
        
    def set_rampFilter(self,which):
        """Set the ramp filter to use: 0, 2, 4, 6, 8, 10, or 12
        
        Args:
            which (int): the order of the finite difference used in the ramp filter, higher numbers produce a sharper reconstruction. Shepp-Logan filter is the default value (2) and Ram-Lak is 12.
            
        Returns:
            True is the input was valid.
        
        """
        self.libprojectors.set_rampID.argtypes = [ctypes.c_int]
        self.libprojectors.set_rampID.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_rampID(which)
        
    def get_rampFilter(self):
        """Gets the ramp filter type, i.e., 0, 2, 4, 6, 8, or 10
        """
        self.libprojectors.get_rampID.argtypes = []
        self.libprojectors.get_rampID.restype = ctypes.c_int
        self.set_model()
        return self.libprojectors.get_rampID()
    
    def set_FBPlowpass(self, W=1.0):
        """Applies a low-pass filter of the specified FWHM to the ramp filter
        
        Args:
            W (float): the FWHM (in detector pixels) of a low pass filter applied to the ramp filter; must be >= 2.0
        """
        self.libprojectors.set_FBPlowpass.argtypes = [ctypes.c_float]
        self.libprojectors.set_FBPlowpass.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_FBPlowpass(W)
        
    def get_FBPlowpass(self, W=1.0):
        """Get the parameter that applies a low-pass filter of the specified FWHM to the ramp filter
        
        Returns:
            W (float): the FWHM (in detector pixels) of a low pass filter applied to the ramp filter
        """
        self.libprojectors.get_FBPlowpass.argtypes = []
        self.libprojectors.get_FBPlowpass.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_FBPlowpass()
    
    def set_attenuationMap(self, mu):
        """Set the voxelized attenuation map for Attenuated Radon Transform calculations"""
        self.libprojectors.set_attenuationMap.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(mu) is torch.Tensor:
            self.libprojectors.set_attenuationMap.argtypes = [ctypes.c_void_p]
            return self.libprojectors.set_attenuationMap(mu.data_ptr())
        else:
            self.libprojectors.set_attenuationMap.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            return self.libprojectors.set_attenuationMap(mu)
            
    def muSpecified(self):
        """Returns a boolean of whether or not the attenuation model has been set for the Attenuated Radon Transform
        
        Returns:
            true if (muCoeff != 0 and muRadius > 0) or mu != NULL, false otherwise
        """
        self.set_model()
        self.libprojectors.muSpecified.restype = ctypes.c_bool
        return self.libprojectors.muSpecified()
        
    def flipAttenuationMapSign(self):
        """Changes the sign of the attenuation model has been set for the Attenuated Radon Transform

        WARNING: does not work if mu is stored on a GPU!
        
        If muCoeff != 0.0, muCoeff *= -1
        if mu != NULL, mu *= -1
        """
        self.set_model()
        self.libprojectors.flipAttenuationMapSign.restype = ctypes.c_bool
        self.libprojectors.flipAttenuationMapSign.argtypes = [ctypes.c_bool]
        return self.libprojectors.flipAttenuationMapSign(True)
        
    def set_cylindircalAttenuationMap(self, c, R):
        """Set the parameters for a cylindrical attenuation map for Attenuated Radon Transform calculations"""
        self.libprojectors.set_cylindircalAttenuationMap.argtypes = [ctypes.c_float, ctypes.c_float]
        self.libprojectors.set_cylindircalAttenuationMap.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_cylindircalAttenuationMap(c, R)
        
    def clear_attenuationMap(self):
        """Clears the attenuation map parameters and reverts back to using the X-ray Transform"""
        self.libprojectors.clear_attenuationMap.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.clear_attenuationMap()
    
    def angles_are_defined(self):
        #self.libprojectors.angles_are_defined.argtypes = []
        self.libprojectors.angles_are_defined.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.angles_are_defined()
        
    def angles_are_equispaced(self):
        #self.libprojectors.angles_are_equispaced.argtypes = []
        self.libprojectors.angles_are_equispaced.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.angles_are_equispaced()
    
    def get_angles(self):
        """Get a numpy array of the projection angles"""
        if self.get_numAngles() > 0:
            phis = np.ascontiguousarray(np.zeros(self.get_numAngles()).astype(np.float32), dtype=np.float32)
            self.libprojectors.get_angles.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.get_angles.restype = ctypes.c_bool
            self.set_model()
            self.libprojectors.get_angles(phis)
            return phis
        else:
            return None

        
    def get_angularRange(self):
        """Get the angular range of the projection angles (degrees)"""
        #self.libprojectors.get_angularRange.argtypes = []
        self.libprojectors.get_angularRange.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_angularRange()
        
    def set_phis(self,phis):
        """Set the projection angles"""
        return self.set_angles(phis)
        
    def set_angles(self,phis):
        """Set the projection angles"""
        self.libprojectors.set_angles.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int]
        self.libprojectors.set_angles.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_angles(phis, int(phis.size))
        
    def set_numCols(self, numCols):
        """Set the number of detector columns"""
        self.libprojectors.set_numCols.argtypes = [ctypes.c_int]
        self.libprojectors.set_numCols.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_numCols(numCols)
        
    def set_numRows(self, numRows):
        """Set the number of detector rows"""
        self.libprojectors.set_numRows.argtypes = [ctypes.c_int]
        self.libprojectors.set_numRows.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_numRows(numRows)
        
    def set_numAngles(self, numAngles):
        """Set the number of projection angles"""
        self.libprojectors.set_numAngles.argtypes = [ctypes.c_int]
        self.libprojectors.set_numAngles.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_numAngles(numAngles)
        
    def set_pixelHeight(self, H):
        """Sets the detector pixel height (mm)"""
        self.libprojectors.set_pixelHeight.argtypes = [ctypes.c_float]
        self.libprojectors.set_pixelHeight.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_pixelHeight(H)
        
    def set_pixelWidth(self, W):
        """Sets the detector pixel width (mm)"""
        self.libprojectors.set_pixelWidth.argtypes = [ctypes.c_float]
        self.libprojectors.set_pixelWidth.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.set_pixelWidth(W)

    ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION RETRIEVES THE VARIOUS CT GEOMETRY AND VOLUME PARAMETERS THAT HAS BEEN SET IN LEAP
    ###################################################################################################################
    ###################################################################################################################
    def get_geometry(self):
        """Get the CT geometry type"""
        self.libprojectors.get_geometry.restype = ctypes.c_int
        self.set_model()
        geometryType = self.libprojectors.get_geometry()
        if geometryType == 0:
            return 'CONE'
        elif geometryType == 1:
            return 'PARALLEL'
        elif geometryType == 2:
            return 'FAN'
        elif geometryType == 3:
            return 'MODULAR'
        elif geometryType == 4:
            return 'CONE-PARALLEL'
        else:
            return 'UNKNOWN'
            
    def get_sod(self):
        """Get the sod parameter"""
        self.libprojectors.get_sod.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_sod()
        
    def get_sdd(self):
        """Get the sdd parameter"""
        self.libprojectors.get_sdd.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_sdd()
        
    def get_helicalPitch(self):
        """Get the helicalPitch parameter"""
        self.libprojectors.get_helicalPitch.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_helicalPitch()
        
    def get_z_source_offset(self):
        """Get the source position z-coordinate for the first projection"""
        self.libprojectors.get_z_source_offset.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_z_source_offset()
    
    def get_numAngles(self):
        """Get the numAngles parameter"""
        self.set_model()
        return self.libprojectors.get_numAngles()
        
    def get_numRows(self):
        """Get the numRows parameter"""
        self.set_model()
        return self.libprojectors.get_numRows()
        
    def get_numCols(self):
        """Get the numCols parameter"""
        self.set_model()
        return self.libprojectors.get_numCols()
        
    def get_pixelHeight(self):
        """Get the pixelHeight parameter"""
        self.libprojectors.get_pixelHeight.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_pixelHeight()
        
    def get_pixelWidth(self):
        """Get the pixelWidth parameter"""
        self.libprojectors.get_pixelWidth.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_pixelWidth()
        
    def get_centerRow(self):
        """Get the centerRow parameter"""
        self.libprojectors.get_centerRow.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_centerRow()
        
    def get_centerCol(self):
        """Get the centerCol parameter"""
        self.libprojectors.get_centerCol.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_centerCol()
        
    def get_tau(self):
        """Get the tau parameter"""
        self.libprojectors.get_tau.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_tau()
        
    def get_tiltAngle(self):
        """Get the tiltAngle parameter"""
        self.libprojectors.get_tiltAngle.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_tiltAngle()
        
    def get_sourcePositions(self):
        """Get the sourcePositions parameter (modular-beam only)"""
        #bool get_sourcePositions(float*);
        if self.get_numAngles() <= 0:
            return None
        else:
            x = np.ascontiguousarray(np.zeros((self.get_numAngles(),3)).astype(np.float32), dtype=np.float32)
            self.libprojectors.get_sourcePositions.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.get_sourcePositions.restype = ctypes.c_bool
            self.set_model()
            self.libprojectors.get_sourcePositions(x)
            return x
        
    def get_moduleCenters(self):
        """Get the moduleCenters parameter (modular-beam only)"""
	    #bool get_moduleCenters(float*);
        if self.get_numAngles() <= 0:
            return None
        else:
            x = np.ascontiguousarray(np.zeros((self.get_numAngles(),3)).astype(np.float32), dtype=np.float32)
            self.libprojectors.get_moduleCenters.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.get_moduleCenters.restype = ctypes.c_bool
            self.set_model()
            self.libprojectors.get_moduleCenters(x)
            return x
        
    def get_rowVectors(self):
        """Get the rowVectors parameter (modular-beam only)"""
	    #bool get_rowVectors(float*);
        if self.get_numAngles() <= 0:
            return None
        else:
            x = np.ascontiguousarray(np.zeros((self.get_numAngles(),3)).astype(np.float32), dtype=np.float32)
            self.libprojectors.get_rowVectors.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.get_rowVectors.restype = ctypes.c_bool
            self.set_model()
            self.libprojectors.get_rowVectors(x)
            return x
        
    def get_colVectors(self):
        """Get the colVectors parameter (modular-beam only)"""
	    #bool get_colVectors(float*);
        if self.get_numAngles() <= 0:
            return None
        else:
            x = np.ascontiguousarray(np.zeros((self.get_numAngles(),3)).astype(np.float32), dtype=np.float32)
            self.libprojectors.get_colVectors.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            self.libprojectors.get_colVectors.restype = ctypes.c_bool
            self.set_model()
            self.libprojectors.get_colVectors(x)
            return x
        
    def get_numX(self):
        """Get the numX parameter"""
        self.set_model()
        return self.libprojectors.get_numX()
    
    def get_numY(self):
        """Get the numY parameter"""
        self.set_model()
        return self.libprojectors.get_numY()
    
    def get_numZ(self):
        """Get the numZ parameter"""
        self.set_model()
        return self.libprojectors.get_numZ()
        
    def get_voxelHeight(self):
        """Get the voxelHeight parameter"""
        self.libprojectors.get_voxelHeight.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_voxelHeight()
        
    def get_voxelWidth(self):
        """Get the voxelWidth parameter"""
        self.libprojectors.get_voxelWidth.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_voxelWidth()
        
    def get_offsetX(self):
        """Get the offsetX parameter"""
        self.libprojectors.get_offsetX.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_offsetX()
        
    def get_offsetY(self):
        """Get the offsetY parameter"""
        self.libprojectors.get_offsetY.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_offsetY()
        
    def get_offsetZ(self):
        """Get the offsetZ parameter"""
        self.libprojectors.get_offsetZ.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_offsetZ()
        
    def get_z0(self):
        """Get the z-coordinate of the first voxel"""
        self.libprojectors.get_z0.restype = ctypes.c_float
        self.set_model()
        return self.libprojectors.get_z0()

    
    ###################################################################################################################
    ###################################################################################################################
    # UTILITY FUNCTIONS
    ###################################################################################################################
    ###################################################################################################################
    def x_samples(self,centerCoords=False):
        """Returns an array of the x sample locations"""
        if centerCoords:
            x_0 = -0.5*(self.get_numX()-1)*self.get_voxelWidth()
        else:
            x_0 = self.get_offsetX() - 0.5*(self.get_numX()-1)*self.get_voxelWidth()
        return np.array(range(self.get_numX()),dtype=np.float32)*self.get_voxelWidth() + x_0
        
    def y_samples(self,centerCoords=False):
        """Returns an array of the y sample locations"""
        if centerCoords:
            y_0 = -0.5*(self.get_numY()-1)*self.get_voxelWidth()
        else:
            y_0 = self.get_offsetY() - 0.5*(self.get_numY()-1)*self.get_voxelWidth()
        return np.array(range(self.get_numY()),dtype=np.float32)*self.get_voxelWidth() + y_0
        
    def z_samples(self,centerCoords=False):
        """Returns an array of the z sample locations"""
        if self.ct_volume_defined() == False:
            return None
        if centerCoords:
            z_0 = -0.5*(self.get_numZ()-1)*self.get_voxelHeight()
        else:
            z_0 = self.get_z0()
        return np.array(range(self.get_numZ()),dtype=np.float32)*self.get_voxelHeight() + z_0
    
    def voxelSamples(self,centerCoords=False):
        """Returns 3D meshgrid of the voxel x,y,z sample locations"""
        x = self.x_samples(centerCoords)
        y = self.y_samples(centerCoords)
        z = self.z_samples(centerCoords)
        if self.get_volumeDimensionOrder() == 0:
            x,y,z = np.meshgrid(x,y,z, indexing='ij')
        else:
            z,y,x = np.meshgrid(z,y,x, indexing='ij')
        return x,y,z

    def display(self,vol):
        """Uses napari to display the provided 3D data
        """
        self.displayVolume(vol)

    def display_volume(self,vol):
        """Uses napari to display the provided 3D data
        """
        self.displayVolume(vol)

    def displayVolume(self,vol):
        """Uses napari to display the provided 3D data
        """
        if has_torch == True and type(vol) is torch.Tensor:
            x = vol.cpu().detach().numpy()
        else:
            x = vol
        
        try:
            import napari
            if len(x.shape) == 3 and (x.shape[0] == 1 or x.shape[1] == 1 or x.shape[2] == 1):
                viewer = napari.view_image(np.squeeze(x), rgb=False)
            else:
                viewer = napari.view_image(x, rgb=False)
            napari.run()
        except:
            print('Cannot load napari, to install run this command:')
            print('pip install napari[all]')
            
    def sketchSystem(self,whichView=None):
        """Alias for sketch_system
        """
        self.sketch_system(whichView)
            
    def sketch_system(self,whichView=None):
        """ Uses matplot lib to sketch the CT geometry and CT volume
        
        The CT geometry parameters and the CT volume parameters must be set prior to running this function.
        
        Args:
            whichView (int): if provided displays the source and detector at the specified view index
        """
        if self.get_numAngles() <= 0 or self.get_numRows() <= 0 or self.get_numCols() <= 0:
            print('CT geometry not set!')
            return False
    
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if whichView is None or isinstance(whichView, int):
            self.drawCT(ax,whichView)
        else:
            for i in range(len(whichView)):
                self.drawCT(ax,whichView[i])
        self.drawVolume(ax)
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5*max([x_range, y_range, z_range])
        print(plot_radius)

        ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
        
        ax.view_init(90, -90)
        #ax.figure.set_size_inches(8, 8)
        plt.show()
    
    def drawCT(self, ax, whichView=None):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        import matplotlib.pyplot as plt
        
        pixelWidth = self.get_pixelWidth()
        pixelHeight = self.get_pixelHeight()
        numCols = self.get_numCols()
        numRows = self.get_numRows()
        centerRow = self.get_centerRow()
        centerCol = self.get_centerCol()
        tau = 0.0
        tiltAngle = 0.0
        
        geometryText = self.get_geometry()
        
        if geometryText == 'CONE' or geometryText == 'FAN':
            if geometryText == 'CONE':
                tiltAngle = self.get_tiltAngle()
            sod = self.get_sod()
            sdd = self.get_sdd()
            tau = self.get_tau()
            R = sod
            D = sdd
            odd = D-R
            detectorWidth = numCols*pixelWidth
            detectorHeight = numRows*pixelHeight
            detectorLeft = -centerCol*pixelWidth
            detectorRight = (numCols-1-centerCol)*pixelWidth
            detectorLeft *= -1.0
            detectorRight *= -1.0
            detectorBottom = -centerRow*pixelHeight
            detectorTop = (numRows-1-centerRow)*pixelHeight
            xs = np.array([detectorLeft, detectorRight, detectorRight, detectorLeft, detectorLeft])
            ys = np.array([-odd, -odd, -odd, -odd, -odd])
            zs = np.array([detectorBottom, detectorBottom, detectorTop, detectorTop, detectorBottom])
            
            if tiltAngle != 0.0:
                tiltAngle *= np.pi/180.0
                A = np.zeros((2,2))
                A[0,0] = np.cos(tiltAngle)
                A[0,1] = np.sin(tiltAngle)
                A[1,0] = -np.sin(tiltAngle)
                A[1,1] = np.cos(tiltAngle)
                xs_rot = A[0,0]*xs + A[0,1]*zs
                zs_rot = A[1,0]*xs + A[1,1]*zs
                xs = xs_rot
                zs = zs_rot
            xs += tau
            
            ax.plot(xs,ys,zs,color='black')

            if geometryText == 'CONE':
                ax.plot([tau, tau], [R, -odd], [0, 0], color='green') # pxcenter line
                #ax.plot([tau, tau-tau*(R+odd)/R], [R, -odd], [0, 0], color='green') # pxmidoff line
                #ax.plot([tau, detectorLeft, tau, detectorRight, tau, detectorLeft, tau, detectorRight], [R, -odd, R, -odd, R, -odd, R, -odd], [0, detectorBottom, 0, detectorBottom, 0, detectorTop, 0, detectorTop],color='red')
                ax.plot([tau, xs[0], tau, xs[1], tau, xs[3], tau, xs[2]], [R, -odd, R, -odd, R, -odd, R, -odd], [0, zs[0], 0, zs[1], 0, zs[3], 0, zs[2]],color='red')
            else:
                ax.plot([tau, tau], [R, -odd], [0, 0], color='green')  # pxcenter line
                #ax.plot([tau, tau - tau * (R + odd) / R], [R, -odd], [0, 0], color='green')  # pxmidoff line
                ax.plot([tau, detectorLeft, detectorRight, tau], [R, -odd, -odd, R], [detectorBottom, detectorBottom, detectorBottom, detectorBottom], color='red')
                ax.plot([tau, detectorLeft, detectorRight, tau], [R, -odd, -odd, R], [0, 0, 0, 0], color='red')
                ax.plot([tau, detectorLeft, detectorRight, tau], [R, -odd, -odd, R], [detectorTop, detectorTop, detectorTop, detectorTop], color='red')

            #topLeft = np.array([detectorLeft, ys[0], detectorTop])
            #topRight = np.array([detectorRight, ys[1], detectorTop])
            #bottomLeft = np.array([detectorLeft, ys[2], detectorBottom])
            #bottomRight = np.array([detectorRight, ys[3], detectorBottom])
            
            topLeft = np.array([xs[3], ys[0], zs[3]])
            topRight = np.array([xs[2], ys[1], zs[2]])
            bottomLeft = np.array([xs[0], ys[2], zs[0]])
            bottomRight = np.array([xs[1], ys[3], zs[1]])
            
            Z = np.squeeze(np.array([topLeft.tolist(), topRight.tolist(), bottomRight.tolist(), bottomLeft.tolist(), bottomLeft.tolist(), bottomRight.tolist(), topRight.tolist(), topLeft.tolist()]))
            ax.scatter3D(Z[:,0], Z[:,1], Z[:,2])
            verts = [[Z[0],Z[1],Z[2],Z[3]],
            [Z[4],Z[5],Z[6],Z[7]],
            [Z[1],Z[2],Z[5],Z[6]],
            [Z[0],Z[3],Z[4],Z[7]],
            [Z[0],Z[1],Z[6],Z[7]],
            [Z[2],Z[3],Z[4],Z[5]]]
            ax.add_collection3d(Poly3DCollection(verts, facecolors='black', linewidths=1, edgecolors='k', alpha=.20))

        elif geometryText == 'PARALLEL':
            tau = 0.0
            detectorWidth = numCols*pixelWidth
            detectorHeight = numRows*pixelHeight
            detectorLeft = -centerCol*pixelWidth + tau
            detectorRight = (numCols-1-centerCol)*pixelWidth + tau
            detectorBottom = -centerRow*pixelHeight
            detectorTop = (numRows-1-centerRow)*pixelHeight

            sod = detectorWidth
            sdd = 2*sod
            odd = sdd - sod

            xs = [detectorLeft, detectorRight, detectorRight, detectorLeft, detectorLeft]
            ys = [-odd, -odd, -odd, -odd, -odd]
            zs = [detectorBottom, detectorBottom, detectorTop, detectorTop, detectorBottom]
            ax.plot(xs, ys, zs, color='black')

            ax.plot([0, 0], [sod, -odd], [0, 0], color='green')
            ax.plot([detectorLeft, detectorLeft, detectorRight, detectorRight], [sod, -odd, -odd, sod], [detectorBottom, detectorBottom, detectorBottom, detectorBottom], color='red')
            ax.plot([detectorLeft, detectorLeft, detectorRight, detectorRight], [sod, -odd, -odd, sod], [0, 0, 0, 0], color='red')
            ax.plot([detectorLeft, detectorLeft, detectorRight, detectorRight], [sod, -odd, -odd, sod], [detectorTop, detectorTop, detectorTop, detectorTop], color='red')

            topLeft = np.array([detectorLeft, ys[0], detectorTop])
            topRight = np.array([detectorRight, ys[1], detectorTop])
            bottomLeft = np.array([detectorLeft, ys[2], detectorBottom])
            bottomRight = np.array([detectorRight, ys[3], detectorBottom])
            Z = np.squeeze(np.array([topLeft.tolist(), topRight.tolist(), bottomRight.tolist(), bottomLeft.tolist(), bottomLeft.tolist(), bottomRight.tolist(), topRight.tolist(), topLeft.tolist()]))
            ax.scatter3D(Z[:,0], Z[:,1], Z[:,2])
            verts = [[Z[0],Z[1],Z[2],Z[3]],
            [Z[4],Z[5],Z[6],Z[7]],
            [Z[1],Z[2],Z[5],Z[6]],
            [Z[0],Z[3],Z[4],Z[7]],
            [Z[0],Z[1],Z[6],Z[7]],
            [Z[2],Z[3],Z[4],Z[5]]]
            ax.add_collection3d(Poly3DCollection(verts, facecolors='black', linewidths=1, edgecolors='k', alpha=.20))
            
        elif geometryText == 'MODULAR':
            sourcePositions = self.get_sourcePositions()
            moduleCenters = self.get_moduleCenters()
            rowVecs = self.get_rowVectors()
            colVecs = self.get_colVectors()
            
            ind_lo = 0
            ind_hi = self.get_numAngles()-1
            if whichView is not None:
                ind_lo = whichView
                ind_hi = whichView
            
            for n in range(ind_lo,ind_hi+1):
                sourcePos = sourcePositions[n,:]
                detPos = moduleCenters[n,:]
                rowVec = rowVecs[n,:]
                colVec = colVecs[n,:]
                
                rowMin = -0.5*(numRows-1)*pixelHeight*rowVec
                rowMax = 0.5*(numRows-1)*pixelHeight*rowVec
                colMin = -0.5*(numCols-1)*pixelWidth*colVec
                colMax = 0.5*(numCols-1)*pixelWidth*colVec
                
                topLeft = detPos + rowMax + colMin
                topRight = detPos + rowMax + colMax
                bottomLeft = detPos + rowMin + colMin
                bottomRight = detPos + rowMin + colMax
                
                ax.plot([sourcePos[0], topLeft[0]], [sourcePos[1], topLeft[1]], [sourcePos[2], topLeft[2]], color='red')
                ax.plot([sourcePos[0], topRight[0]], [sourcePos[1], topRight[1]], [sourcePos[2], topRight[2]], color='red')
                ax.plot([sourcePos[0], bottomLeft[0]], [sourcePos[1], bottomLeft[1]], [sourcePos[2], bottomLeft[2]], color='red')
                ax.plot([sourcePos[0], bottomRight[0]], [sourcePos[1], bottomRight[1]], [sourcePos[2], bottomRight[2]], color='red')
                
                Z = np.squeeze(np.array([topLeft.tolist(), topRight.tolist(), bottomRight.tolist(), bottomLeft.tolist(), bottomLeft.tolist(), bottomRight.tolist(), topRight.tolist(), topLeft.tolist()]))
                ax.scatter3D(Z[:,0], Z[:,1], Z[:,2])
                verts = [[Z[0],Z[1],Z[2],Z[3]],
                [Z[4],Z[5],Z[6],Z[7]],
                [Z[1],Z[2],Z[5],Z[6]],
                [Z[0],Z[3],Z[4],Z[7]],
                [Z[0],Z[1],Z[6],Z[7]],
                [Z[2],Z[3],Z[4],Z[5]]]
                ax.add_collection3d(Poly3DCollection(verts, facecolors='black', linewidths=1, edgecolors='k', alpha=.20))
                
        if geometryText == 'PARALLEL' or geometryText == 'FAN' or geometryText == 'CONE':
            if geometryText == 'CONE':
                pitch = self.get_helicalPitch()
                z_source_offset = self.get_z_source_offset()
            else:
                pitch = 0.0
                z_source_offset = 0.0
                
            phis = np.pi/180.0*self.get_angles() - 0.5*np.pi
            ax.plot(sod*np.cos(phis) + tau*np.sin(phis), sod*np.sin(phis)-tau*np.cos(phis), (pitch*phis+z_source_offset), '.', color='green')
    
    def drawVolume(self, ax):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
        import matplotlib.pyplot as plt
        N_x = self.get_numX()
        N_y = self.get_numY()
        N_z = self.get_numZ()
        if N_x > 0 and N_y > 0 and N_z > 0:
            x_c = self.get_offsetX()
            y_c = self.get_offsetY()
            z_c = self.get_offsetZ()
            T = self.get_voxelWidth()
            T_z = self.get_voxelHeight()
            x_min = -0.5*float(N_x-1)*T + x_c
            x_max = 0.5*float(N_x-1)*T + x_c
            y_min = -0.5*float(N_y-1)*T + y_c
            y_max = 0.5*float(N_y-1)*T + y_c
            z_min = self.get_z0()
            z_max = float(N_z-1)*T_z + z_min

            topLeft_front = np.array([x_min, y_max, z_max])
            topRight_front = np.array([x_max, y_max, z_max])
            bottomLeft_front = np.array([x_min, y_min, z_max])
            bottomRight_front = np.array([x_max, y_min, z_max])
            topLeft_back = np.array([x_min, y_max, z_min])
            topRight_back = np.array([x_max, y_max, z_min])
            bottomLeft_back = np.array([x_min, y_min, z_min])
            bottomRight_back = np.array([x_max, y_min, z_min])
            Z = np.squeeze(np.array([topLeft_front.tolist(), topRight_front.tolist(), bottomRight_front.tolist(), bottomLeft_front.tolist(), bottomLeft_back.tolist(), bottomRight_back.tolist(), topRight_back.tolist(), topLeft_back.tolist()]))
            ax.scatter3D(Z[:,0], Z[:,1], Z[:,2])
            verts = [[Z[0],Z[1],Z[2],Z[3]],
            [Z[4],Z[5],Z[6],Z[7]],
            [Z[1],Z[2],Z[5],Z[6]],
            [Z[0],Z[3],Z[4],Z[7]],
            [Z[0],Z[1],Z[6],Z[7]],
            [Z[2],Z[3],Z[4],Z[5]]]
            ax.add_collection3d(Poly3DCollection(verts, facecolors='magenta', linewidths=1, edgecolors='k', alpha=.20))
    

    ###################################################################################################################
    ###################################################################################################################
    # FILE I/O FUNCTIONS
    ###################################################################################################################
    ###################################################################################################################
    def parse_param_dic(self, param_fn):
        pdic = {}
        not_scalar_params = ['proj_geometry', 'proj_phis', 'proj_srcpos', 'proj_modcenter', 'proj_rowvec', 'proj_colvec', 'geometry', 'phis', 'sourcePositions', 'moduleCenters', 'rowVectors', 'colVectors']
        with open(param_fn, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) == 0 or line[0] == '#':
                    continue
                key = line.split('=')[0].strip()
                value = line.split('=')[1].strip()
                #if key == 'proj_phis' or key == 'proj_geometry' or key == 'proj_srcpos' or key == 'proj_modcenter' or key == 'proj_rowvec' or key == 'proj_colvec':
                if any(item == key for item in not_scalar_params): 
                    pdic[key] = value
                else:
                    pdic[key] = float(value)
        return pdic

    def load_parameters(self, parameters_fileName, param_type=0): # param_type 0: cfg, 1: dict
        """Load the CT volume and CT geometry parameters from file
        
        Args:
            parameters_fileName (string): file name of the parameters file to load
            param_type (int): if 0, assumes that parameters_fileName is a file; if 1, assumes that parameters_fileName is the actual content of a parameters file
        """
        return self.load_param(parameters_fileName, param_type)

    def load_param(self, param_fn, param_type=0): # param_type 0: cfg, 1: dict
        pdic = {}
        if param_type == 0:
            pdic = self.parse_param_dic(param_fn)
        elif param_type == 1:
            pdic = param_fn

        if 'geometry' not in pdic.keys():
            # Could not load a basic parameter
            # Try loading the legacy parameters
            return self.load_param_legacy(param_fn, param_type)

        if 'phis' not in pdic.keys():
            phis_str = ''
        else:
            phis_str = pdic['phis']
        if len(phis_str) > 0:
            phis = np.array([float(x.strip()) for x in phis_str.split(',')]).astype(np.float32)
        elif 'angularRange' in pdic.keys():
            phis = self.setAngleArray(int(pdic['numAngles']), pdic['angularRange'])
        else:
            if pdic['geometry'] == 'parallel' or pdic['geometry'] == 'fan' or pdic['geometry'] == 'cone':
                print('ERROR: invalid LEAP parameter file')
                return False
        if pdic['geometry'] == 'modular':
            if 'sourcePositions' not in pdic.keys():
                print('ERROR: invalid LEAP parameter file')
                return False
        
        self.set_volume(int(pdic['numX']), int(pdic['numY']), int(pdic['numZ']),
                        pdic['voxelWidth'], pdic['voxelHeight'], 
                        pdic['offsetX'], pdic['offsetY'], pdic['offsetZ'])
        if pdic['geometry'] == 'parallel':
            self.set_parallelbeam(int(pdic['numAngles']), int(pdic['numRows']), int(pdic['numCols']), 
                                   pdic['pixelHeight'], pdic['pixelWidth'], 
                                   pdic['centerRow'], pdic['centerCol'], phis)
            if 'muCoeff' in pdic.keys() and 'muRadius' in pdic.keys():
                self.set_cylindircalAttenuationMap(pdic['muCoeff'], pdic['muRadius'])
        elif pdic['geometry'] == 'fan':
            self.set_fanbeam(int(pdic['numAngles']), int(pdic['numRows']), int(pdic['numCols']), 
                               pdic['pixelHeight'], pdic['pixelWidth'], 
                               pdic['centerRow'], pdic['centerCol'], 
                               phis, pdic['sod'], pdic['sdd'], pdic['tau'])
        elif pdic['geometry'] == 'cone':
            self.set_conebeam(int(pdic['numAngles']), int(pdic['numRows']), int(pdic['numCols']), 
                               pdic['pixelHeight'], pdic['pixelWidth'], 
                               pdic['centerRow'], pdic['centerCol'], 
                               phis, pdic['sod'], pdic['sdd'], pdic['tau'], pdic['helicalPitch'], pdic['tiltAngle'])
        elif pdic['geometry'] == 'modular':
        
            sourcePositions = np.array([float(x.strip()) for x in pdic['sourcePositions'].split(',')]).astype(np.float32)
            moduleCenters = np.array([float(x.strip()) for x in pdic['moduleCenters'].split(',')]).astype(np.float32)
            rowVectors = np.array([float(x.strip()) for x in pdic['rowVectors'].split(',')]).astype(np.float32)
            colVectors = np.array([float(x.strip()) for x in pdic['colVectors'].split(',')]).astype(np.float32)
            self.set_modularbeam(int(pdic['numAngles']), int(pdic['numRows']), int(pdic['numCols']), 
                                  pdic['pixelHeight'], pdic['pixelWidth'], 
                                  sourcePositions, moduleCenters, rowVectors, colVectors)

        self.set_flatDetector()
        if 'detectorType' in pdic.keys():
            if pdic['detectorType'] == 'curved':
                self.set_curvedDetector()
        
        if 'axisOfSymmetry' in pdic.keys():
            self.set_axisOfSymmetry(pdic['axisOfSymmetry'])
        
        return True

    def load_param_legacy(self, param_fn, param_type=0): # param_type 0: cfg, 1: dict
        pdic = {}
        if param_type == 0:
            pdic = self.parse_param_dic(param_fn)
        elif param_type == 1:
            pdic = param_fn

        if 'proj_geometry' not in pdic.keys():
            print('ERROR: invalid LEAP parameter file')
            return False

        if 'proj_phis' not in pdic.keys():
            phis_str = ''
        else:
            phis_str = pdic['proj_phis']
        if len(phis_str) > 0:
            #phis = torch.from_numpy(np.array([float(x.strip()) for x in phis_str.split(',')]).astype(np.float32))
            phis = np.array([float(x.strip()) for x in phis_str.split(',')]).astype(np.float32)
        elif 'proj_arange' in pdic.keys():
            phis = self.setAngleArray(int(pdic['proj_nangles']), pdic['proj_arange'])
            #phis = torch.from_numpy(phis)
        else:
            if pdic['proj_geometry'] == 'parallel' or pdic['proj_geometry'] == 'fan' or pdic['proj_geometry'] == 'cone':
                print('ERROR: invalid LEAP parameter file')
                return False
        if pdic['proj_geometry'] == 'modular':
            if 'proj_srcpos' not in pdic.keys():
                print('ERROR: invalid LEAP parameter file')
                return False
        
        self.set_volume(int(pdic['img_dimx']), int(pdic['img_dimy']), int(pdic['img_dimz']),
                        pdic['img_pwidth'], pdic['img_pheight'], 
                        pdic['img_offsetx'], pdic['img_offsety'], pdic['img_offsetz'])
        if pdic['proj_geometry'] == 'parallel':
            self.set_parallelbeam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
                                   pdic['proj_pheight'], pdic['proj_pwidth'], 
                                   pdic['proj_crow'], pdic['proj_ccol'], phis)
        elif pdic['proj_geometry'] == 'fan':
            self.set_fanbeam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
                               pdic['proj_pheight'], pdic['proj_pwidth'], 
                               pdic['proj_crow'], pdic['proj_ccol'], 
                               phis, pdic['proj_sod'], pdic['proj_sdd'])
        elif pdic['proj_geometry'] == 'cone':
            self.set_conebeam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
                               pdic['proj_pheight'], pdic['proj_pwidth'], 
                               pdic['proj_crow'], pdic['proj_ccol'], 
                               phis, pdic['proj_sod'], pdic['proj_sdd'])
        elif pdic['proj_geometry'] == 'modular':
            self.set_modularbeam(int(pdic['proj_nangles']), int(pdic['proj_nrows']), int(pdic['proj_ncols']), 
                                  pdic['proj_pheight'], pdic['proj_pwidth'], 
                                  pdic['proj_srcpos'], pdic['proj_modcenter'], pdic['proj_rowvec'], pdic['proj_colvec'])
        
        return True
    
    def save_parameters(self, fileName):
        """Save the CT volume and CT geometry parameters to the provided file name"""
        return self.save_param(fileName)
    
    def save_param(self, fileName):
        """Save the CT volume and CT geometry parameters to file"""
        if sys.version_info[0] == 3:
            fileName = bytes(str(fileName), 'ascii')
        self.libprojectors.saveParamsToFile.restype = ctypes.c_bool
        self.set_model()
        return self.libprojectors.saveParamsToFile(fileName)
        
        return self.leapct.save_param(fileName)
    
    def save_projections(self, fileName, g, sequence_offset=0, axis_split=0):
        """Save projection data to file (tif sequence, nrrd, or npy)
        
        Args:
            fileName (string): the file name to save the projection data to
            g (C contiguous float32 numpy array or torch tensor): projection data
            sequence_offset (int): if saving as a tif/tiff sequence, this specifies the index of the first file
        
        """
        return self.saveProjections(fileName, g, sequence_offset, axis_split=axis_split)
        
    def saveProjections(self, fileName, g, sequence_offset=0, axis_split=0):
        """Save projection data to file (tif sequence, nrrd, or npy)"""
        if self.get_numAngles() > 0 and self.get_numRows() > 0 and self.get_numCols() > 0:
            pixelWidth = self.get_pixelWidth()
            pixelHeight = self.get_pixelHeight()
            #numCols = self.get_numCols()
            #numRows = self.get_numRows()
            centerRow = self.get_centerRow()
            centerCol = self.get_centerCol()
            phis = self.get_angles()
        
            phi_0 = phis[0]
            row_0 = -centerRow*pixelHeight
            col_0 = -centerCol*pixelWidth
            T = pixelWidth
        else:
            phi_0 = 0.0
            row_0 = 0.0
            col_0 = 0.0
            T = 1.0
        return self.save_data(fileName, g, T, phi_0, row_0, col_0, sequence_offset, axis_split=axis_split)
    
    def save_volume(self, fileName, f, sequence_offset=0):
        """Save volume data to file (tif sequence, nrrd, or npy)
        
        Args:
            fileName (string): the file name to save the projection data to
            f (C contiguous float32 numpy array or torch tensor): volume data
            sequence_offset (int): if saving as a tif/tiff sequence, this specifies the index of the first file
        
        """
        return self.saveVolume(fileName, f, sequence_offset)
    
    def saveVolume(self, fileName, f, sequence_offset=0):
        """Save volume data to file (tif sequence, nrrd, or npy)"""
        if self.get_numX() > 0 and self.get_numY() > 0 and self.get_numZ() > 0:
            z_0 = self.z_samples()[0]
            y_0 = self.y_samples()[0]
            x_0 = self.x_samples()[0]
            T = self.get_voxelWidth()
        else:
            x_0 = 0.0
            y_0 = 0.0
            z_0 = 0.0
            T = 1.0
        return self.save_data(fileName, f, T, x_0, y_0, z_0, sequence_offset)
    
    def load_tif_python(self, fileName, x=None, rowRange=None, colRange=None):
        if os.path.isfile(fileName) == False:
            return None

        try:
            anImage = np.array(imageio.imread(fileName), dtype=np.float32)
            
            numRows = anImage.shape[0]
            numCols = anImage.shape[1]
            
            if rowRange is not None:
                if rowRange[0] < 0 or rowRange[0] > rowRange[1] or rowRange[1] > anImage.shape[0]-1:
                    print('Invalid rowRange')
                    return None
                numRows = rowRange[1]-rowRange[0]+1
            if colRange is not None:
                if colRange[0] < 0 or colRange[0] > colRange[1] or colRange[1] > anImage.shape[1]-1:
                    print('Invalid colRange')
                    return None
                numCols = colRange[1]-colRange[0]+1
            
            if x is not None:
                x = np.zeros((numRows, numCols), dtype=np.float32)
            elif x.shape[0] != numRows or x.shape[1] != numCols:
                return None
            
            if rowRange is not None:
                if colRange is not None:
                    x[:,:] = anImage[rowRange[0]:rowRange[1]+1, colRange[0]:colRange[1]+1]
                else:
                    x[:,:] = anImage[rowRange[0]:rowRange[1]+1,:]
            else:
                if colRange is not None:
                    x[:,:] = anImage[:,colRange[0]:colRange[1]+1]
                else:
                    x[:,:] = anImage[:,:]
            return x
        except:
            print('error reading: ' + str(fileName))
            return None
    
    def load_tif(self, fileName, x=None, rowRange=None, colRange=None):
        if os.path.isfile(fileName) == False:
            return None
            
        if sys.version_info[0] == 3:
            path = bytes(str(fileName), 'ascii')
        else:
            path = fileName

        if x is None:
            shape = np.zeros(2, dtype=np.int32)
            size = np.zeros(2, dtype=np.float32)
            slope_and_offset = size.copy()
            
            self.libprojectors.read_tif_header.restype = ctypes.c_bool
            self.libprojectors.read_tif_header.argtypes = [ctypes.c_char_p, ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
            success = self.libprojectors.read_tif_header(path, shape, size, slope_and_offset)
            if success == False:
                try:
                    return self.load_tif_python(fileName, None, rowRange, colRange)
                    #x = np.array(imageio.imread(fileName), dtype=np.float32)
                    #return x
                except:
                    return None
            else:
                numRows = shape[0]
                numCols = shape[1]
                if rowRange is not None:
                    if rowRange[0] < 0 or rowRange[1] > shape[0]-1 or rowRange[0] > rowRange[1]:
                        return None
                    else:
                        numRows = rowRange[1] - rowRange[0] + 1
                if colRange is not None:
                    if colRange[0] < 0 or colRange[1] > shape[1]-1 or colRange[0] > colRange[1]:
                        return None
                    else:
                        numCols = colRange[1] - colRange[0] + 1
                    
                x = np.zeros((numRows, numCols), dtype=np.float32)
        
        else:
            # x is given; make sure it is the correct size
            if rowRange is not None:
                #if rowRange[0] == 0 and rowRange[1] == x.shape[0]-1:
                #    rowRange = None
                if rowRange[0] < 0 or rowRange[1]-rowRange[0]+1 > x.shape[0] or rowRange[0] > rowRange[1]:
                    print('Invalid rowRange')
                    return None
            if colRange is not None:
                #if colRange[0] == 0 and colRange[1] == x.shape[1]-1:
                #    colRange = None
                if colRange[0] < 0 or colRange[1]-colRange[0]+1 > x.shape[1] or colRange[0] > colRange[1]:
                    print('Invalid colRange')
                    return None
                
        if rowRange is None:
            if colRange is None:
                self.libprojectors.read_tif.restype = ctypes.c_bool
                self.libprojectors.read_tif.argtypes = [ctypes.c_char_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
                if self.libprojectors.read_tif(path, x):
                    return x
                else:
                    return None
            else:
                self.libprojectors.read_tif_cols.restype = ctypes.c_bool
                self.libprojectors.read_tif_cols.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
                if self.libprojectors.read_tif_cols(path, colRange[0], colRange[1], x):
                    return x
                else:
                    return None
        else:
            if colRange is None:
                self.libprojectors.read_tif_rows.restype = ctypes.c_bool
                self.libprojectors.read_tif_rows.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
                if self.libprojectors.read_tif_rows(path, rowRange[0], rowRange[1], x):
                    return x
                else:
                    return None
            else:
                self.libprojectors.read_tif_roi.restype = ctypes.c_bool
                self.libprojectors.read_tif_roi.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS")]
                if self.libprojectors.read_tif_roi(path, rowRange[0], rowRange[1], colRange[0], colRange[1], x):
                    return x
                else:
                    return None
        
    def save_tif(self, fileName, x, T=[1.0,1.0]):
        if sys.version_info[0] == 3:
            fileName = bytes(str(fileName), 'ascii')
        self.libprojectors.save_tif.restype = ctypes.c_bool
        #bool save_tif(char* fileName, float* data, int numRows, int numCols, float pixelHeight, float pixelWidth, int dtype, float wmin, float wmax);
        self.libprojectors.save_tif.argtypes = [ctypes.c_char_p, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float, ctypes.c_float]
        self.set_model()
        
        if self.file_dtype == np.uint8:
            dtype = 0
        elif self.file_dtype == np.uint16:
            dtype = 1
        else:
            dtype = 3
        if dtype == 3:
            wmin = 0.0
            wmax = 1.0
        else:
            if self.wmin is None:
                wmin = np.min(x)
            else:
                wmin = self.wmin
            if self.wmax is None:
                wmax = np.max(x)
            else:
                wmax = self.wmax
            if wmax <= wmin:
                wmax = wmin + 1.0
        
        return self.libprojectors.save_tif(fileName, x, x.shape[0], x.shape[1], T[0], T[1], dtype, wmin, wmax)
    
    
    def save_data(self, fileName, x, T=1.0, offset_0=0.0, offset_1=0.0, offset_2=0.0, sequence_offset=0, axis_split=0):
        """Save 3D data to file (tif sequence, nrrd, or npy)"""
        volFilePath, dontCare = os.path.split(fileName)
        if os.path.isdir(volFilePath) == False or os.access(volFilePath, os.W_OK) == False:
            print('Folder to save data either does not exist or not accessible!')
            return False
            
        if has_torch == True and type(x) is torch.Tensor:
            x = x.cpu().detach().numpy()
            
        if fileName.endswith('.npy'):
            np.save(fileName, x)
            return True
        elif fileName.endswith('.nrrd'):
            try:
                import nrrd
                
                # https://pynrrd.readthedocs.io/en/latest/examples.html
                header = {'units': ['mm', 'mm', 'mm'], 'spacings': [T, T, T], 'axismins': [offset_0, offset_1, offset_2], 'thicknesses': [T, T, T],}
                nrrd.write(fileName, x, header)
                return True
            except:
                print('Error: Failed to load nrrd library!')
                print('To install this package do: pip install pynrrd')
                return False
        elif fileName.endswith('.tif') or fileName.endswith('.tiff'):
            baseName, fileExtension = os.path.splitext(fileName)
            if self.file_dtype != np.uint8 and self.file_dtype != np.uint16 and self.file_dtype != np.float32:
                print('Can only save as uint8, uint16, or float32!')
                return False
                
            if len(x.shape) <= 2:
                self.save_tif(baseName + fileExtension, x)
            else:
                if axis_split == 0:
                    for i in range(x.shape[0]):
                        im = x[i,:,:]
                        self.save_tif(baseName + '_' + str(int(i)+sequence_offset) + fileExtension, im)
                else:
                    for i in range(x.shape[1]):
                        im = x[:,i,:]
                        self.save_tif(baseName + '_' + str(int(i)+sequence_offset) + fileExtension, im)
            return True
            """
            try:
                #from PIL import Image
                import imageio
                
                baseName, fileExtension = os.path.splitext(fileName)
                if self.file_dtype != np.uint8 and self.file_dtype != np.uint16 and self.file_dtype != np.float32:
                    print('Can only save as uint8, uint16, or float32!')
                    return False
                
                if self.file_dtype != x.dtype:
                    wmin = self.wmin
                    if self.wmax is None:
                        wmax = np.max(x)
                    else:
                        wmax = self.wmax
                    if wmax <= wmin:
                        wmax = wmin + 1.0
                    if self.file_dtype == np.uint8:
                        max_dtype = 255.0
                    else:
                        max_dtype = 65535.0
                
                if len(x.shape) <= 2:
                    im = x
                    #im.save(baseName + '_' + str(int(i)) + fileExtension)
                    imageio.imwrite(baseName + fileExtension, im)
                else:
                    if self.file_dtype == x.dtype:
                        for i in range(x.shape[0]):
                            im = x[i,:,:]
                            imageio.imwrite(baseName + '_' + str(int(i)+sequence_offset) + fileExtension, im)
                    else:
                        for i in range(x.shape[0]):
                            im = x[i,:,:]
                            np.clip(im, wmin, wmax, out=im)
                            im = np.array((im - wmin) / (wmax - wmin) * max_dtype, dtype=self.file_dtype)
                            imageio.imwrite(baseName + '_' + str(int(i)+sequence_offset) + fileExtension, im)
                        
                return True
             
            except:
                #print('Error: Failed to load PIL library!')
                #print('To install this package do: pip install Pillow')
                print('Error: Failed to load imageio library!')
                print('To install PIL do: pip install imageio')
                return False
            """
        else:
            print('Error: must be a tif, npy, or nrrd file!')
            return False
            
    def loadVolume(self, fileName, x=None, fileRange=None, rowRange=None, colRange=None):
        """Load 3D volume data from file (tif sequence, nrrd, or npy)"""
        return self.load_data(fileName, x, fileRange, rowRange, colRange)
        
    def load_volume(self, fileName, x=None, fileRange=None, rowRange=None, colRange=None):
        """Load 3D volume data from the given file name provided (tif sequence, nrrd, or npy)
        
        See load_data for more information
        """
        return self.load_data(fileName, x, fileRange, rowRange, colRange)
        
    def loadProjections(self, fileName, x=None, fileRange=None, rowRange=None, colRange=None, axis_split=0):
        """Load 3D projection data from file (tif sequence, nrrd, or npy)"""
        return self.load_data(fileName, x, fileRange, rowRange, colRange, axis_split)
        
    def load_projections(self, fileName, x=None, fileRange=None, rowRange=None, colRange=None, axis_split=0):
        """Load 3D projection data from the given file name provided (tif sequence, nrrd, or npy)
        
        See load_data for more information
        """
        return self.load_data(fileName, x, fileRange, rowRange, colRange, axis_split)
        
    def loadData(self, fileName, x=None, fileRange=None, rowRange=None, colRange=None, axis_split=0):
        """Load 3D data from the given file name provided (tif sequence, nrrd, or npy)"""
        return self.load_data(fileName, x, fileRange, rowRange, colRange, axis_split)
        
    def get_file_list(self, fileName, fileRange=None):
        asterisk_ind = fileName.find('*')
        if asterisk_ind != -1:
            fileList = glob.glob(fileName)
            if len(fileList) == 0:
                print(fileName)
                print('file sequence does not exist')
                return None
            _, fileExt = os.path.splitext(fileName)
            baseFileName = fileName[0:asterisk_ind]
            sequence_separator = ""

        else:
            
            sequence_separator = "_"
        
            #baseFileName, fileExt = os.path.splitext(os.path.basename(fileName))
            baseFileName, fileExt = os.path.splitext(fileName)
            templateFile = baseFileName + '_*' + fileExt
            fileList = glob.glob(templateFile)
            if len(fileList) == 0:
                sequence_separator = ""
                templateFile = baseFileName + '*' + fileExt
                fileList = glob.glob(os.path.split(templateFile)[1])
                if len(fileList) == 0:
                    print('file sequence does not exist')
                    return None
                
        if fileRange is not None:
            # prune fileList
            fileList_pruned = []
            for i in range(len(fileList)):
                digit = int(fileList[i].replace(baseFileName+sequence_separator,'').replace(fileExt,''))
                if fileRange[0] <= digit and digit <= fileRange[1]:
                    fileList_pruned.append(fileList[i])
            fileList = fileList_pruned
            
        justDigits = []
        for i in range(len(fileList)):
            digitStr = fileList[i].replace(baseFileName+sequence_separator,'').replace(fileExt,'')
            justDigits.append(int(digitStr))
        ind = np.argsort(justDigits)
        fileList = [fileList[i] for i in ind]
        return fileList
    
    def load_data(self, fileName, x=None, fileRange=None, rowRange=None, colRange=None, axis_split=0):
        """Load 3D data from file (tif sequence, nrrd, or npy)

        This function reads 3D data and stores it in a 3D numpy array.  We officially support
        nrrd, npy, or a a sequence of tif/tiff files.  Note that fileRange, rowRange, and colRange arguments
        only apply to tif sequences.
        
        A tif sequences must be in the following form: basename_XXXX.tif or (tiff).  The XXXX are the sequence numbers
        which can be padded with zeros or not.  When calling this function be sure to specify the input as basename.tif,
        i.e., do not include the underscore and digits when providing this file name to this function.
        
        Args:
            fileName (string): full path to npy or nrrd file or sequence of tif files.
            x (3D float32 numpy array): place to store the data (this argument is optional)
            fileRange (list with two integers): the first and last files to read of a tif sequence
            rowRange (list with two integers): the first and last rows to read in a tif sequence
            colRange (list with two integers): the first and last columns to read in a tif sequence
        
        Returns:
            3D numpy array of the data in the file(s); if x is given, just returns x
        """
        
        if fileRange is not None:
            if len(fileRange) != 2 or fileRange[0] > fileRange[1] or fileRange[0] < 0 or fileRange[1] < 0:
                print('Error: fileRange must be a list of two positive numbers')
                return None
        if rowRange is not None:
            if len(rowRange) != 2 or rowRange[0] > rowRange[1] or rowRange[0] < 0 or rowRange[1] < 0:
                print('Error: rowRange must be a list of two positive numbers')
                return None
        if colRange is not None:
            if len(colRange) != 2 or colRange[0] > colRange[1] or colRange[0] < 0 or colRange[1] < 0:
                print('Error: colRange must be a list of two positive numbers')
                return None
        
        if fileName.endswith('.npy'):
            if os.path.isfile(fileName) == False:
                print('file does not exist')
                return None
            else:
                if x is not None:
                    x[:] = np.load(fileName)
                    return x
                else:
                    return np.load(fileName)
        elif fileName.endswith('.nrrd'):
            if os.path.isfile(fileName) == False:
                print('file does not exist')
                return None
            try:
                import nrrd
                if x is not None:
                    x[:], header = nrrd.read(fileName)
                else:
                    x, header = nrrd.read(fileName)
                #T_fromFile = header['spacings'][0]
                return x
            except:
                print('Error: Failed to load nrrd library!')
                print('To install this package do: pip install pynrrd')
                return None
        elif fileName.endswith('.tif') or fileName.endswith('.tiff'):
            
            fileList = self.get_file_list(fileName, fileRange)
            if fileList is None or len(fileList) == 0:
                return None

            firstImg = self.load_tif(fileList[0], None, rowRange, colRange)
            if firstImg is None:
                firstImg = self.load_tif_python(fileList[0], None, rowRange, colRange)
                if firstImg is None:
                    return None
                use_python_readers = True
                #print('using python readers')
            else:
                use_python_readers = False
                #print('using LEAP readers')
            
            numRows = firstImg.shape[0]
            numCols = firstImg.shape[1]
            anImage = np.zeros((numRows, numCols), dtype=np.float32)
            if x is not None:
                if axis_split == 1:
                    if len(x.shape) != 3 or x.shape[1] != len(fileList) or x.shape[0] != numRows or x.shape[2] != numCols:
                        print('Error: given array size does not match size of data in files!')
                        return None
                else:
                    if len(x.shape) != 3 or x.shape[0] != len(fileList) or x.shape[1] != numRows or x.shape[2] != numCols:
                        print('Error: given array size does not match size of data in files!')
                        return None
            else:
                if axis_split == 1:
                    x = np.zeros((numRows, len(fileList), numCols), dtype=np.float32)
                else:
                    x = np.zeros((len(fileList), numRows, numCols), dtype=np.float32)
            print('reading ' + str(len(fileList)) + ' images of size ' + str(numRows) + ' x ' + str(firstImg.shape[1]))
            for i in range(len(fileList)):
                curFile = fileList[i]
                if use_python_readers:
                    if self.load_tif_python(curFile, anImage, rowRange, colRange) is None:
                        return None
                    else:
                        if axis_split == 1:
                            x[:,i,:] = anImage[:,:]
                        else:
                            x[i,:,:] = anImage[:,:]
                else:
                    if self.load_tif(curFile, anImage, rowRange, colRange) is None:
                        return None
                    else:
                        if axis_split == 1:
                            x[:,i,:] = anImage[:,:]
                        else:
                            x[i,:,:] = anImage[:,:]
            
            """
            firstImg = np.array(imageio.imread(fileList[0]), dtype=np.float32)
            
            numRows = firstImg.shape[0]
            numCols = firstImg.shape[1]
            if rowRange is not None:
                if rowRange[1] > numRows-1:
                    print('Error: row range is out range')
                    return None
                numRows = rowRange[1] - rowRange[0] + 1
            if colRange is not None:
                if colRange[1] > numCols-1:
                    print('Error: col range is out range')
                    return None
                numCols = colRange[1] - colRange[0] + 1
            
            if x is not None:
                if len(x.shape) != 3 or x.shape[0] != len(fileList) or x.shape[1] != numRows or x.shape[2] != numCols:
                    print('Error: given array size does not match size of data in files!')
                    return None
            else:
                x = np.zeros((len(fileList), numRows, numCols), dtype=np.float32)
            print('found ' + str(x.shape[0]) + ' images of size ' + str(firstImg.shape[0]) + ' x ' + str(firstImg.shape[1]))
            for i in range(len(fileList)):
                if ind is None:
                    anImage = np.array(imageio.imread(fileList[i]), dtype=np.float32)
                else:
                    anImage = np.array(imageio.imread(fileList[ind[i]]), dtype=np.float32)
                if rowRange is not None:
                    if colRange is not None:
                        x[i,:,:] = anImage[rowRange[0]:rowRange[1]+1,colRange[0]:colRange[1]+1]
                    else:
                        x[i,:,:] = anImage[rowRange[0]:rowRange[1]+1,:]
                else:
                    if colRange is not None:
                        x[i,:,:] = anImage[:,colRange[0]:colRange[1]+1]
                    else:
                        x[i,:,:] = anImage[:,:]
            
            if currentWorkingDirectory is not None:
                os.chdir(currentWorkingDirectory)
            """
            return x
            
        else:
            print('Error: must be a tif, npy, or nrrd file!')
            return None
    
    
    ###################################################################################################################
    ###################################################################################################################
    # PHANTOM SPECIFICATION FUNCTIONS
    ###################################################################################################################
    ###################################################################################################################
    def rayTrace(self, g=None, oversampling=1):
        """Performs analytic ray-tracing simulation through a phantom composed of geometrical objects

        See the addObject function for how to build the phantom description
        The CT geometry parameters must be specified prior to running this functions
        
        Args:
            g (C contiguous float32 numpy array): CT projection data
            oversampling (int): the oversampling factor for each ray
            
        Returns:
            g
        
        """
        if g is None:
            g = self.allocate_projections()
        self.set_model()
        if has_torch == True and type(g) is torch.Tensor:
            self.libprojectors.rayTrace.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]
            self.libprojectors.rayTrace(g.data_ptr(), int(oversampling), g.is_cuda == False)
        else:
            self.libprojectors.rayTrace.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ctypes.c_bool]
            self.libprojectors.rayTrace(g, int(oversampling), True)
        return g
    
    def voxelize(self, f, oversampling=1):
        r"""Voxelizes a phantom defined by geometric objects.
        
        One must have a phantom already defined before running this function.
        
        Args:
            f (C contiguous float32 numpy array or torch tensor): volume data
            oversampling (int): the oversampling factor of the voxelization
        """
        self.libprojectors.addObject.restype = ctypes.c_bool
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.voxelize.argtypes = [ctypes.c_void_p, ctypes.c_int]
            return self.libprojectors.voxelize(f.data_ptr(), oversampling)
        else:
            self.libprojectors.voxelize.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int]
            return self.libprojectors.voxelize(f, oversampling)
    
    def addObject(self, f, typeOfObject, c, r, val, A=None, clip=None, oversampling=1):
        """Adds a geometric object to the phantom
        
        This function operates in two modes: (1) specifying a voxelized phantom and (2) specifying a phantom
        to be used in an analytic ray-tracing simulation (see rayTrace).
        If a volume is given (first argument of this function), then the specified object will be added to the voxel data.
        If a volume is not given then the object will be added to a stack of objects that define a phantom to be used for
        an analytic ray-tracing simulation.
        
        The order in which multiple object are defined is important.  Background objects must be specified first
        and foreground objects defined last.  If you reverse the order, then the foreground objects will be effectively overriden
        by the background objects.
        
        The CT volume or CT geometry parameters must be specified prior to running this functions.
        
        Args:
            f (C contiguous float32 numpy array): CT volume
            type (int): ELLIPSOID=0, PARALLELEPIPED=1, CYLINDER_X=2, CYLINDER_Y=3, CYLINDER_Z=4, CONE_X=5, CONE_Y=6, CONE_Z=7
            c (3X1 numpy array): x,y,z coordinates of the center of the object
            r (3X1 numpy array): radii in the x,y,z directions of the object
            val (float): the values to ascribe inside this object
            A (3x3 numpy array): rotation matrix to rotate the object about its center
            clip (3X1 numpy array): specifies the clipping planes, if any
        """
        self.libprojectors.addObject.restype = ctypes.c_bool
        if A is None:
            A = np.zeros((3,3),dtype=np.float32)
            A[0,0] = 1.0
            A[1,1] = 1.0
            A[2,2] = 1.0
        if clip is None:
            clip = np.zeros(3,dtype=np.float32)
        
        if isinstance(c, int) or isinstance(c, float):
            c = [c, c, c]
        if isinstance(r, int) or isinstance(r, float):
            r = [r, r, r]
        
        c = np.ascontiguousarray(c, dtype=np.float32)
        r = np.ascontiguousarray(r, dtype=np.float32)
        A = np.ascontiguousarray(A, dtype=np.float32)
        clip = np.ascontiguousarray(clip, dtype=np.float32)
        self.set_model()
        if has_torch == True and type(f) is torch.Tensor:
            self.libprojectors.addObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int]
            return self.libprojectors.addObject(f.data_ptr(), int(typeOfObject), c, r, float(val), A, clip, oversampling)
        elif f is None:
            self.libprojectors.addObject.argtypes = [ctypes.c_void_p, ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int]
            return self.libprojectors.addObject(f, int(typeOfObject), c, r, float(val), A, clip, oversampling)
        else:
            self.libprojectors.addObject.argtypes = [ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_float, ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"), ctypes.c_int]
            return self.libprojectors.addObject(f, int(typeOfObject), c, r, float(val), A, clip, oversampling)
        
    def set_FORBILD(self, f=None, includeEar=True, oversampling=1):
        """Sets the FORBILD head phantom
        
        This function operates in two modes: (1) specifying a voxelized phantom and (2) specifying a phantom
        to be used in an analytic ray-tracing simulation (see rayTrace).
        If a volume is given (first argument of this function), then the specified object will be added to the voxel data.
        If a volume is not given then all the objects of the FORBILD head phantom will be added to a stack of objects
        that define a phantom to be used for an analytic ray-tracing simulation.
        
        Note that the values of the FORBILD head phantom are all scaled by 0.02
        which is the LAC of water at around 60 keV.  The FOV is about [-96, 96, -120, 120, -125, 125]
        
        Args:
            f (C contiguous float32 numpy array): CT volume
            includeEar (boolean): specifies whether the air bubbles in the ear are to be included or not
            oversampling (int): the oversampling factor of the voxelization
        
        """
        has_scipy = False
        try:
            from scipy.spatial.transform import Rotation as R
            has_scipy = True
        except:
            print('Warning: scipy package cannot be found, so not included rotated objects')
            has_scipy = False
        self.addObject(f, 0, 10.0*np.array([0.0, 0.0, 0.0]), 10.0*np.array([9.6, 12.0, 12.5]), 1.800*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([0.0, 0.0, 0.0]), 10.0*np.array([9.0, 11.4, 11.9]), 1.050*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([-4.7, 4.3, 0.872]), 10.0*np.array([2.0, 2.0, 2.0]), 1.060*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([4.7, 4.3, 0.872]), 10.0*np.array([2.0, 2.0, 2.0]), 1.060*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([-1.08, -9, 0.0]), 10.0*np.array([0.4, 0.4, 0.4]), 1.0525*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([1.08, -9, 0.0]), 10.0*np.array([0.4, 0.4, 0.4]), 1.0475*0.02, None, None, oversampling)
        self.addObject(f, 0, 10.0*np.array([0.0, 8.4, 0.0]), 10.0*np.array([1.8, 3.0, 3.0]), 0.0, None, None, oversampling)
        if has_scipy:
            self.addObject(f, 0, 10.0*np.array([-1.9, 5.4, 0.0]), 10.0*np.array([1.206483*np.cos(15*np.pi/180.0), 0.420276*np.cos(15*np.pi/180.0), 3.0]), 1.800*0.02, R.from_euler("xyz", [0, 0, -120], degrees=True).as_matrix(), None, oversampling)
            self.addObject(f, 0, 10.0*np.array([1.9, 5.4, 0.0]), 10.0*np.array([1.2*np.cos(15*np.pi/180.0), 0.42*np.cos(15*np.pi/180.0), 3.0]), 1.800*0.02, R.from_euler("xyz", [0, 0, 120], degrees=True).as_matrix(), None, oversampling)
            self.addObject(f, 4, 10.0*np.array([-4.3, 6.8, -1.0]), 10.0*np.array([1.8, 0.24, 2.0]), 1.800*0.02, R.from_euler("xyz", [0, 0, -150], degrees=True).as_matrix(), None, oversampling)
            self.addObject(f, 4, 10.0*np.array([4.3, 6.8, -1.0]), 10.0*np.array([1.8, 0.24, 2.0]), 1.800*0.02, R.from_euler("xyz", [0, 0, -30], degrees=True).as_matrix(), None, oversampling)
        self.addObject(f, 0, 10.0*np.array([0.0, -3.6, 0.0]), 10.0*np.array([1.8, 3.6, 3.6]), 1.045*0.02, None, None, oversampling)
        if has_scipy:
            self.addObject(f, 0, 10.0*np.array([6.393945, -6.393945, 0.0]), 10.0*np.array([1.2, 0.42, 1.2]), 1.055*0.02, R.from_euler("xyz", [0, 0, -58.1], degrees=True).as_matrix(), None, oversampling)
            self.addObject(f, 4, 10.0*np.array([0.0, 3.6, 0.0]), 10.0*np.array([1.2, 4.0, 0.25*np.cos(15*np.pi/180.0)]), 1.800*0.02, R.from_euler("xyz", [60, 0, 0], degrees=True).as_matrix(), None, oversampling)
            self.addObject(f, 2, 10.0*np.array([0.0, 9.6, 0.0]), 10.0*np.array([0.525561/2.0, 2.0, 0.4]), 1.800*0.02, R.from_euler("xyz", [-60, 0, 0], degrees=True).as_matrix(), None, oversampling)
        self.addObject(f, 6, 10.0*np.array([0.0, -11.15, -0.2]), 10.0*np.array([0.5, 0.75, 0.2]), 1.800*0.02, None, None, oversampling)
        self.addObject(f, 6, 10.0*np.array([0.0, -11.15, 0.2]), 10.0*np.array([0.5, 0.75, 0.2]), 1.800*0.02, None, None, oversampling)
        if self.get_numAngles() == 1 and self.get_numX() == 1:
            pass
        else:
            #self.addObject(f, 0, 10.0*np.array([9.1, 0.0, 0.0]), 10.0*np.array([4.2, 1.8, 1.8]), 1.800*0.02, None, np.array([1.0, 0.0, 0.0]), oversampling)
            self.addObject(f, 0, 10.0*np.array([9.1, 0.0, 0.0]), 10.0*np.array([4.2, 1.8, 1.8]), 1.800*0.02, None, np.array([91.0, 0.0, 0.0]), oversampling)

        #'''
        if includeEar:
            xyzs = np.array([8.80, -1.03920, -1.03920,  8.40, -1.03920, -1.03920,  8.0, -1.03920, -1.03920,  7.60, -1.03920, -1.03920,  8.60, -0.69280, -1.03920,  8.20, -0.69280, -1.03920,  7.80, -0.69280, -1.03920,  7.40, -0.69280, -1.03920,  7.0, -0.69280, -1.03920,  8.80, -0.34640, -1.03920,  8.40, -0.34640, -1.03920,  8.0, -0.34640, -1.03920,  7.60, -0.34640, -1.03920,  7.20, -0.34640, -1.03920,  6.80, -0.34640, -1.03920,  8.80, 1.03920, -1.03920,  8.40, 1.03920, -1.03920,  8.0, 1.03920, -1.03920,  7.60, 1.03920, -1.03920,  8.60, 0.69280, -1.03920,  8.20, 0.69280, -1.03920,  7.80, 0.69280, -1.03920,  7.40, 0.69280, -1.03920,  7.0, 0.69280, -1.03920,  8.80, 0.34640, -1.03920,  8.40, 0.34640, -1.03920,  8.0, 0.34640, -1.03920,  7.60, 0.34640, -1.03920,  7.20, 0.34640, -1.03920,  6.80, 0.34640, -1.03920,  8.60, 0.0, -1.03920,  8.20, 0.0, -1.03920,  7.80, 0.0, -1.03920,  7.40, 0.0, -1.03920,  7.0, 0.0, -1.03920,  6.60, 0.0, -1.03920,  8.80, -1.03920, 1.03920,  8.40, -1.03920, 1.03920,  8.0, -1.03920, 1.03920,  7.60, -1.03920, 1.03920,  8.60, -0.69280, 1.03920,  8.20, -0.69280, 1.03920,  7.80, -0.69280, 1.03920,  7.40, -0.69280, 1.03920,  7.0, -0.69280, 1.03920,  8.80, -0.34640, 1.03920,  8.40, -0.34640, 1.03920,  8.0, -0.34640, 1.03920,  7.60, -0.34640, 1.03920,  7.20, -0.34640, 1.03920,  6.80, -0.34640, 1.03920,  8.80, 1.03920, 1.03920,  8.40, 1.03920, 1.03920,  8.0, 1.03920, 1.03920,  7.60, 1.03920, 1.03920,  8.60, 0.69280, 1.03920,  8.20, 0.69280, 1.03920,  7.80, 0.69280, 1.03920,  7.40, 0.69280, 1.03920,  7.0, 0.69280, 1.03920,  8.80, 0.34640, 1.03920,  8.40, 0.34640, 1.03920,  8.0, 0.34640, 1.03920,  7.60, 0.34640, 1.03920,  7.20, 0.34640, 1.03920,  6.80, 0.34640, 1.03920,  8.60, 0.0, 1.03920,  8.20, 0.0, 1.03920,  7.80, 0.0, 1.03920,  7.40, 0.0, 1.03920,  7.0, 0.0, 1.03920,  6.60, 0.0, 1.03920,  8.60, -1.03920, -0.69280,  8.20, -1.03920, -0.69280,  7.80, -1.03920, -0.69280,  7.40, -1.03920, -0.69280,  7.0, -1.03920, -0.69280,  8.80, -0.69280, -0.69280,  8.40, -0.69280, -0.69280,  8.0, -0.69280, -0.69280,  7.60, -0.69280, -0.69280,  7.20, -0.69280, -0.69280,  6.80, -0.69280, -0.69280,  6.40, -0.69280, -0.69280,  8.60, -0.34640, -0.69280,  8.20, -0.34640, -0.69280,  7.80, -0.34640, -0.69280,  7.40, -0.34640, -0.69280,  7.0, -0.34640, -0.69280,  6.60, -0.34640, -0.69280,  6.20, -0.34640, -0.69280,  8.60, 1.03920, -0.69280,  8.20, 1.03920, -0.69280,  7.80, 1.03920, -0.69280,  7.40, 1.03920, -0.69280,  7.0, 1.03920, -0.69280,  8.80, 0.69280, -0.69280,  8.40, 0.69280, -0.69280,  8.0, 0.69280, -0.69280,  7.60, 0.69280, -0.69280,  7.20, 0.69280, -0.69280,  6.80, 0.69280, -0.69280,  6.40, 0.69280, -0.69280,  8.60, 0.34640, -0.69280,  8.20, 0.34640, -0.69280,  7.80, 0.34640, -0.69280,  7.40, 0.34640, -0.69280,  7.0, 0.34640, -0.69280,  6.60, 0.34640, -0.69280,  6.20, 0.34640, -0.69280,  8.80, 0.0, -0.69280,  8.40, 0.0, -0.69280,  8.0, 0.0, -0.69280,  7.60, 0.0, -0.69280,  7.20, 0.0, -0.69280,  6.80, 0.0, -0.69280,  6.40, 0.0, -0.69280,  6.0, 0.0, -0.69280,  8.60, -1.03920, 0.69280,  8.20, -1.03920, 0.69280,  7.80, -1.03920, 0.69280,  7.40, -1.03920, 0.69280,  7.0, -1.03920, 0.69280,  8.80, -0.69280, 0.69280,  8.40, -0.69280, 0.69280,  8.0, -0.69280, 0.69280,  7.60, -0.69280, 0.69280,  7.20, -0.69280, 0.69280,  6.80, -0.69280, 0.69280,  6.40, -0.69280, 0.69280,  8.60, -0.34640, 0.69280,  8.20, -0.34640, 0.69280,  7.80, -0.34640, 0.69280,  7.40, -0.34640, 0.69280,  7.0, -0.34640, 0.69280,  6.60, -0.34640, 0.69280,  6.20, -0.34640, 0.69280,  8.60, 1.03920, 0.69280,  8.20, 1.03920, 0.69280,  7.80, 1.03920, 0.69280,  7.40, 1.03920, 0.69280,  7.0, 1.03920, 0.69280,  8.80, 0.69280, 0.69280,  8.40, 0.69280, 0.69280,  8.0, 0.69280, 0.69280,  7.60, 0.69280, 0.69280,  7.20, 0.69280, 0.69280,  6.80, 0.69280, 0.69280,  6.40, 0.69280, 0.69280,  8.60, 0.34640, 0.69280,  8.20, 0.34640, 0.69280,  7.80, 0.34640, 0.69280,  7.40, 0.34640, 0.69280,  7.0, 0.34640, 0.69280,  6.60, 0.34640, 0.69280,  6.20, 0.34640, 0.69280,  8.80, 0.0, 0.69280,  8.40, 0.0, 0.69280,  8.0, 0.0, 0.69280,  7.60, 0.0, 0.69280,  7.20, 0.0, 0.69280,  6.80, 0.0, 0.69280,  6.40, 0.0, 0.69280,  6.0, 0.0, 0.69280,  8.80, -1.03920, -0.34640,  8.40, -1.03920, -0.34640,  8.0, -1.03920, -0.34640,  7.60, -1.03920, -0.34640,  7.20, -1.03920, -0.34640,  6.80, -1.03920, -0.34640,  8.60, -0.69280, -0.34640,  8.20, -0.69280, -0.34640,  7.80, -0.69280, -0.34640,  7.40, -0.69280, -0.34640,  7.0, -0.69280, -0.34640,  6.60, -0.69280, -0.34640,  6.20, -0.69280, -0.34640,  8.80, -0.34640, -0.34640,  8.40, -0.34640, -0.34640,  8.0, -0.34640, -0.34640,  7.60, -0.34640, -0.34640,  7.20, -0.34640, -0.34640,  6.80, -0.34640, -0.34640,  6.40, -0.34640, -0.34640,  6.0, -0.34640, -0.34640,  8.80, 1.03920, -0.34640,  8.40, 1.03920, -0.34640,  8.0, 1.03920, -0.34640,  7.60, 1.03920, -0.34640,  7.20, 1.03920, -0.34640,  6.80, 1.03920, -0.34640,  8.60, 0.69280, -0.34640,  8.20, 0.69280, -0.34640,  7.80, 0.69280, -0.34640,  7.40, 0.69280, -0.34640,  7.0, 0.69280, -0.34640,  6.60, 0.69280, -0.34640,  6.20, 0.69280, -0.34640,  8.80, 0.34640, -0.34640,  8.40, 0.34640, -0.34640,  8.0, 0.34640, -0.34640,  7.60, 0.34640, -0.34640,  7.20, 0.34640, -0.34640,  6.80, 0.34640, -0.34640,  6.40, 0.34640, -0.34640,  6.0, 0.34640, -0.34640,  8.60, 0.0, -0.34640,  8.20, 0.0, -0.34640,  7.80, 0.0, -0.34640,  7.40, 0.0, -0.34640,  7.0, 0.0, -0.34640,  6.60, 0.0, -0.34640,  6.20, 0.0, -0.34640,  5.80, 0.0, -0.34640,  8.80, -1.03920, 0.34640,  8.40, -1.03920, 0.34640,  8.0, -1.03920, 0.34640,  7.60, -1.03920, 0.34640,  7.20, -1.03920, 0.34640,  6.80, -1.03920, 0.34640,  8.60, -0.69280, 0.34640,  8.20, -0.69280, 0.34640,  7.80, -0.69280, 0.34640,  7.40, -0.69280, 0.34640,  7.0, -0.69280, 0.34640,  6.60, -0.69280, 0.34640,  6.20, -0.69280, 0.34640,  8.80, -0.34640, 0.34640,  8.40, -0.34640, 0.34640,  8.0, -0.34640, 0.34640,  7.60, -0.34640, 0.34640,  7.20, -0.34640, 0.34640,  6.80, -0.34640, 0.34640,  6.40, -0.34640, 0.34640,  6.0, -0.34640, 0.34640,  8.80, 1.03920, 0.34640,  8.40, 1.03920, 0.34640,  8.0, 1.03920, 0.34640,  7.60, 1.03920, 0.34640,  7.20, 1.03920, 0.34640,  6.80, 1.03920, 0.34640,  8.60, 0.69280, 0.34640,  8.20, 0.69280, 0.34640,  7.80, 0.69280, 0.34640,  7.40, 0.69280, 0.34640,  7.0, 0.69280, 0.34640,  6.60, 0.69280, 0.34640,  6.20, 0.69280, 0.34640,  8.80, 0.34640, 0.34640,  8.40, 0.34640, 0.34640,  8.0, 0.34640, 0.34640,  7.60, 0.34640, 0.34640,  7.20, 0.34640, 0.34640,  6.80, 0.34640, 0.34640,  6.40, 0.34640, 0.34640,  6.0, 0.34640, 0.34640,  8.60, 0.0, 0.34640,  8.20, 0.0, 0.34640,  7.80, 0.0, 0.34640,  7.40, 0.0, 0.34640,  7.0, 0.0, 0.34640,  6.60, 0.0, 0.34640,  6.20, 0.0, 0.34640,  5.80, 0.0, 0.34640,  8.60, -1.03920, 0.0,  8.20, -1.03920, 0.0,  7.80, -1.03920, 0.0,  7.40, -1.03920, 0.0,  7.0, -1.03920, 0.0,  6.60, -1.03920, 0.0,  8.80, -0.69280, 0.0,  8.40, -0.69280, 0.0,  8.0, -0.69280, 0.0,  7.60, -0.69280, 0.0,  7.20, -0.69280, 0.0,  6.80, -0.69280, 0.0,  6.40, -0.69280, 0.0,  6.0, -0.69280, 0.0,  8.60, -0.34640, 0.0,  8.20, -0.34640, 0.0,  7.80, -0.34640, 0.0,  7.40, -0.34640, 0.0,  7.0, -0.34640, 0.0,  6.60, -0.34640, 0.0,  6.20, -0.34640, 0.0,  5.80, -0.34640, 0.0,  8.60, 1.03920, 0.0,  8.20, 1.03920, 0.0,  7.80, 1.03920, 0.0,  7.40, 1.03920, 0.0,  7.0, 1.03920, 0.0,  6.60, 1.03920, 0.0,  8.80, 0.69280, 0.0,  8.40, 0.69280, 0.0,  8.0, 0.69280, 0.0,  7.60, 0.69280, 0.0,  7.20, 0.69280, 0.0,  6.80, 0.69280, 0.0,  6.40, 0.69280, 0.0,  6.0, 0.69280, 0.0,  8.60, 0.34640, 0.0,  8.20, 0.34640, 0.0,  7.80, 0.34640, 0.0,  7.40, 0.34640, 0.0,  7.0, 0.34640, 0.0,  6.60, 0.34640, 0.0,  6.20, 0.34640, 0.0,  5.80, 0.34640, 0.0,  8.80, 0.0, 0.0,  8.40, 0.0, 0.0,  8.0, 0.0, 0.0,  7.60, 0.0, 0.0,  7.20, 0.0, 0.0,  6.80, 0.0, 0.0,  6.40, 0.0, 0.0,  6.0, 0.0, 0.0,  5.60, 0.0, 0.0])
            for n in range(xyzs.size//3):
                x = xyzs[3*n+0]
                y = xyzs[3*n+1]
                z = xyzs[3*n+2]
                self.addObject(f, 0, 10.0*np.array([x, y, z]), 10.0*np.array([0.15, 0.15, 0.15]), 0.0, None, None, oversampling)
        #'''
        
    def clearPhantom(self):
        """Clears all phantom objects"""
        self.set_model()
        self.libprojectors.clearPhantom()
        
    def scalePhantom(self, c):
        r"""Scales the size of the phantom by the provided factor
        
        One must have a phantom already defined before running this function.
        
        Args:
            c (float or list of three float): the scaling values (values greater than one make the phantom larger)
        """
        if isinstance(c, int) or isinstance(c, float):
            c_x = c
            c_y = c
            c_z = c
        elif isinstance(c, np.ndarray) and c.size == 3:
            c_x = c[0]
            c_y = c[1]
            c_z = c[2]
        else:
            return False
            
        self.libprojectors.scalePhantom.restype = ctypes.c_bool
        self.libprojectors.scalePhantom.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.set_model()
        return self.libprojectors.scalePhantom(c_x, c_y, c_z)
        
class subsetParameters:
    def __init__(self, ctModel, numSubsets):
        self.numSubsets = numSubsets
        self.ctModel = ctModel
        
        self.phis_subsets = []
        self.sourcePositions_subsets = []
        self.moduleCenters_subsets = []
        self.rowVectors_subsets = []
        self.colVectors_subsets = []
        
        self.phis = None
        self.sourcePositions = None
        self.moduleCenters = None
        self.rowVectors = None
        self.colVectors = None

        # First set parameters for all angles
        if self.ctModel.get_geometry() == 'MODULAR':
            self.sourcePositions = self.ctModel.get_sourcePositions()
            self.moduleCenters = self.ctModel.get_moduleCenters()
            self.rowVectors = self.ctModel.get_rowVectors()
            self.colVectors = self.ctModel.get_colVectors()
        else:
            self.phis = self.ctModel.get_angles()
            
        # Now set parameters for each subset
        N = self.ctModel.get_numAngles()
        if self.ctModel.get_geometry() == 'MODULAR':
            for m in range(self.numSubsets):
                if m == self.sourcePositions.shape[0]-1:
                    sourcePositions_subset = np.zeros((1,3),dtype=np.float32)
                    sourcePositions_subset[0,:] = self.sourcePositions[m,:]
                    self.sourcePositions_subsets.append(sourcePositions_subset)
                    
                    moduleCenters_subset = np.zeros((1,3),dtype=np.float32)
                    moduleCenters_subset[0,:] = self.moduleCenters[m,:]
                    self.moduleCenters_subsets.append(moduleCenters_subset)

                    rowVectors_subset = np.zeros((1,3),dtype=np.float32)
                    rowVectors_subset[0,:] = self.rowVectors[m,:]
                    self.rowVectors_subsets.append(rowVectors_subset)
                    
                    colVectors_subset = np.zeros((1,3),dtype=np.float32)
                    colVectors_subset[0,:] = self.colVectors[m,:]
                    self.colVectors_subsets.append(colVectors_subset)

                else:
                    sourcePositions_subset = np.ascontiguousarray(self.sourcePositions[m:N:self.numSubsets,:], np.float32)
                    self.sourcePositions_subsets.append(sourcePositions_subset)
                    
                    moduleCenters_subset = np.ascontiguousarray(self.moduleCenters[m:N:self.numSubsets,:], np.float32)
                    self.moduleCenters_subsets.append(moduleCenters_subset)
                    
                    rowVectors_subset = np.ascontiguousarray(self.rowVectors[m:N:self.numSubsets,:], np.float32)
                    self.rowVectors_subsets.append(rowVectors_subset)
                    
                    colVectors_subset = np.ascontiguousarray(self.colVectors[m:N:self.numSubsets,:], np.float32)
                    self.colVectors_subsets.append(colVectors_subset)
        else:
            phis_subsets = []
            for m in range(self.numSubsets):
                if m == self.phis.size-1:
                    phis_subset = np.zeros((1,1),dtype=np.float32)
                    phis_subset[0,0] = self.phis[m]
                    self.phis_subsets.append(phis_subset)
                else:
                    phis_subset = np.ascontiguousarray(self.phis[m:N:self.numSubsets], np.float32)
                    self.phis_subsets.append(phis_subset)
        
    def setSubset(self, isubset):
        if 0 <= isubset and isubset < self.numSubsets:
            if self.ctModel.get_geometry() == 'MODULAR':
                numAngles = self.sourcePositions_subsets[isubset].shape[0]
                numRows = self.ctModel.get_numRows()
                numCols = self.ctModel.get_numCols()
                pixelHeight = self.ctModel.get_pixelHeight()
                pixelWidth = self.ctModel.get_pixelWidth()
                sourcePositions = self.sourcePositions_subsets[isubset]
                moduleCenters = self.moduleCenters_subsets[isubset]
                rowVectors = self.rowVectors_subsets[isubset]
                colVectors = self.colVectors_subsets[isubset]
                self.ctModel.set_modularBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors)
                #print(sourcePositions)
            else:
                self.ctModel.set_angles(self.phis_subsets[isubset])
        else:
            if self.ctModel.get_geometry() == 'MODULAR':
                numAngles = self.sourcePositions.shape[0]
                numRows = self.ctModel.get_numRows()
                numCols = self.ctModel.get_numCols()
                pixelHeight = self.ctModel.get_pixelHeight()
                pixelWidth = self.ctModel.get_pixelWidth()
                sourcePositions = self.sourcePositions
                moduleCenters = self.moduleCenters
                rowVectors = self.rowVectors
                colVectors = self.colVectors
                self.ctModel.set_modularBeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, moduleCenters, rowVectors, colVectors)
            else:
                self.ctModel.set_angles(self.phis)
