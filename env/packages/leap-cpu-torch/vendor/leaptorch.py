################################################################################
# Copyright 2022-2023 Lawrence Livermore National Security, LLC and other 
# LEAP project developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# LivermorE AI Projector for Computed Tomography (LEAP)
# PyTorch projector class
################################################################################

import numpy as np
import torch
from leapctype import *
lct = tomographicModels()

# Note:
# Image tensor format: [Batch, ImageZ, ImageY, ImageX]
# Projection tensor format: [Batch, Views, Detector_Row, Detector_Col]

# CPU Projector for forward and backward propagation
class ProjectorFunctionCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: image, output: projection (sinogram)
        for batch in range(input.shape[0]):
            f = input[batch]
            g = proj[batch]
            lct.project_cpu(g, f, param_id.item()) # compute proj (g) from input (f)
        ctx.save_for_backward(input, vol, param_id)
        return proj

    @staticmethod
    def backward(ctx, grad_output): # grad_output: projection (sinogram) grad_input: image
        input, vol, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            f = vol[batch]
            g = grad_output[batch]
            lct.backproject_cpu(g, f, param_id.item()) # compute input (f) from proj (g)
        return vol, None, None, None

# GPU Projector for forward and backward propagation
class ProjectorFunctionGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: image, output: projection (sinogram)
        for batch in range(input.shape[0]):
            f = input[batch]
            g = proj[batch]
            lct.project_gpu(g, f, param_id.item()) # compute proj (g) from input (f)
        ctx.save_for_backward(input, vol, param_id)
        return proj

    @staticmethod
    def backward(ctx, grad_output): # grad_output: projection (sinogram) grad_input: image
        input, vol, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            f = vol[batch]
            g = grad_output[batch]
            lct.backproject_gpu(g, f, param_id.item()) # compute input (f) from proj (g)
        return vol, None, None, None


# CPU BackProjector for forward and backward propagation
class BackProjectorFunctionCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: projection (sinogram), output: image
        for batch in range(input.shape[0]):
            f = vol[batch]
            g = input[batch]
            lct.backproject_cpu(g, f, param_id.item()) # compute input (f) from proj (g)
            #vol[batch] = f
        ctx.save_for_backward(input, proj, param_id)
        return vol

    @staticmethod
    def backward(ctx, grad_output): # grad_output: image, grad_input: projection (sinogram)
        input, proj, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            f = grad_output[batch]
            g = proj[batch]
            lct.project_cpu(g, f, param_id.item()) # compute proj (g) from input (f)
            #proj[batch] = g
        return proj, None, None, None

# GPU BackProjector for forward and backward propagation
class BackProjectorFunctionGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: projection (sinogram), output: image
        for batch in range(input.shape[0]):
            f = vol[batch]
            g = input[batch]
            lct.backproject_gpu(g, f, param_id.item()) # compute input (f) from proj (g)
            #vol[batch] = f
        ctx.save_for_backward(input, proj, param_id)
        return vol
        
    @staticmethod
    def backward(ctx, grad_output): # grad_output: image, grad_input: projection (sinogram)
        input, proj, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            f = grad_output[batch]
            g = proj[batch]
            lct.project_gpu(g, f, param_id.item()) # compute proj (g) from input (f)
            #proj[batch] = g
        return proj, None, None, None




# CPU FBP for forward and backward propagation
class FBPFunctionCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: projection (sinogram), output: image
        for batch in range(input.shape[0]):
            f = vol[batch]
            g = input[batch]
            lct.fbp_cpu(g, f) # compute input (f) from proj (g)
        ctx.save_for_backward(input, proj, param_id)
        return vol
        
    @staticmethod
    def backward(ctx, grad_output): # grad_output: image, grad_input: projection (sinogram)
        input, proj, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            f = grad_output[batch]
            g = proj[batch]
            lct.fbp_adjoint_cpu(g, f) # compute proj (g) from input (f) -> needs to be replaced!!!
        return proj, None, None, None


# GPU FBP for forward and backward propagation
class FBPFunctionGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: projection (sinogram), output: image
        for batch in range(input.shape[0]):
            f = vol[batch]
            g = input[batch]
            lct.fbp_gpu(g, f) # compute input (f) from proj (g)
        ctx.save_for_backward(input, proj, param_id)
        return vol
        
    @staticmethod
    def backward(ctx, grad_output): # grad_output: image, grad_input: projection (sinogram)
        input, proj, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            f = grad_output[batch]
            g = proj[batch]
            lct.fbp_adjoint_gpu(g, f) # compute proj (g) from input (f) -> needs to be replaced!!!
        return proj, None, None, None



# CPU reverse FBP for forward and backward propagation
class FBPReverseFunctionCPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: image, output: projection (sinogram)
        for batch in range(input.shape[0]):
            f = input[batch]
            g = proj[batch]
            lct.fbp_adjoint_cpu(g, f) # compute proj (g) from input (f)
        ctx.save_for_backward(input, vol, param_id)
        return proj

    @staticmethod
    def backward(ctx, grad_output): # grad_output: projection (sinogram) grad_input: image
        input, vol, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            f = vol[batch]
            g = grad_output[batch]
            lct.fbp_cpu(g, f) # compute input (f) from proj (g)
        return vol, None, None, None

# GPU reverse FBP for forward and backward propagation
class FBPReverseFunctionGPU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, proj, vol, param_id): # input: image, output: projection (sinogram)
        for batch in range(input.shape[0]):
            f = input[batch]
            g = proj[batch]
            lct.fbp_adjoint_gpu(g, f) # compute proj (g) from input (f)
        ctx.save_for_backward(input, vol, param_id)
        return proj

    @staticmethod
    def backward(ctx, grad_output): # grad_output: projection (sinogram) grad_input: image
        input, vol, param_id = ctx.saved_tensors
        for batch in range(input.shape[0]):
            f = vol[batch]
            g = grad_output[batch]
            lct.fbp_gpu(g, f) # compute input (f) from proj (g)
        return vol, None, None, None    
    

# base abstract Projector class
class BaseProjector(torch.nn.Module):
    def __init__(self, use_static=False, use_gpu=False, gpu_device=None, batch_size=1):
        super(BaseProjector, self).__init__()

        self.use_static = use_static
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.batch_size = batch_size

        if self.use_static:
            self.param_id = 0
            self.leapct = tomographicModels(self.param_id)
        else:
            self.leapct = tomographicModels()
            self.param_id = self.leapct.param_id
        lct.param_id = self.param_id
        self.param_id_t = torch.tensor(self.param_id).to(self.gpu_device) if self.use_gpu else torch.tensor(self.param_id)
        if self.use_gpu:
            self.set_gpu(self.gpu_device)
        self.batch_size = batch_size
        self.vol_data = None
        self.proj_data = None

    def forward(self, input):
        return None

    def set_volume(self, numX, numY, numZ, voxelWidth, voxelHeight, offsetX=0.0, offsetY=0.0, offsetZ=0.0):
        """Set the CT volume parameters
        
        This function is the same as leapct.tomographicModels.set_volume, except that it also
        allocates the batch data for the volume (see also allocate_batch_data)
        
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
        self.leapct.set_volume(numX, numY, numZ, voxelWidth, voxelHeight, offsetX, offsetY, offsetZ)
        vol_np = np.ascontiguousarray(np.zeros((self.batch_size, numZ, numY, numX),dtype=np.float32), dtype=np.float32)
        self.vol_data = torch.from_numpy(vol_np)
        if self.use_gpu:
            self.vol_data = self.vol_data.float().to(self.gpu_device)
            
    def set_default_volume(self, scale=1.0):
        """Sets the default volume parameters
        
        The default volume parameters are those that fill the field of view of the CT system and use the native voxel sizes.
        This function is the same as leapct.tomographicModels.set_default_volume, except that it also
        allocates the batch data for the volume (see also allocate_batch_data)
        
        Args:
            scale (float): this value scales the voxel size by this value to create denser or sparser voxel representations (not recommended for fast reconstruction)
        
        Returns:
            True if the operation was successful, false otherwise (this usually happens if the CT geometry has not yet been set)
        """
        self.leapct.set_defaultVolume(scale)
        dim1, dim2, dim3 = self.get_volume_dim()
        vol_np = np.ascontiguousarray(np.zeros((self.batch_size, dim1, dim2, dim3),dtype=np.float32), dtype=np.float32)
        self.vol_data = torch.from_numpy(vol_np)
        if self.use_gpu:
            self.vol_data = self.vol_data.float().to(self.gpu_device)

    def set_parallelbeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis):
        """Sets the parameters for a parallel-beam CT geometry
        
        The origin of the coordinate system is always at the center of rotation.
        This function is the same as leapct.tomographicModels.set_parallelbeam, except that it also
        allocates the batch data for the projections (see also allocate_batch_data)
        
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
        if type(phis) is torch.Tensor:
            phis = phis.numpy()
        self.leapct.set_parallelbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis)
        proj_np = np.ascontiguousarray(np.zeros((self.batch_size, numAngles, numRows, numCols),dtype=np.float32), dtype=np.float32)
        self.proj_data = torch.from_numpy(proj_np)
        if self.use_gpu:
            self.proj_data = self.proj_data.float().to(self.gpu_device)
            
    def set_fanbeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0):
        """Sets the parameters for a fan-beam CT geometry
        
        The origin of the coordinate system is always at the center of rotation.
        This function is the same as leapct.tomographicModels.set_fanbeam, except that it also
        allocates the batch data for the projections (see also allocate_batch_data)
        
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
        if type(phis) is torch.Tensor:
            phis = phis.numpy()
        self.leapct.set_fanbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau)
        proj_np = np.ascontiguousarray(np.zeros((self.batch_size, numAngles, numRows, numCols),dtype=np.float32), dtype=np.float32)
        self.proj_data = torch.from_numpy(proj_np)
        if self.use_gpu:
            self.proj_data = self.proj_data.float().to(self.gpu_device)

    def set_conebeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau=0.0, helicalPitch=0.0):
        """Sets the parameters for a cone-beam CT geometry
        
        The origin of the coordinate system is always at the center of rotation.
        This function is the same as leapct.tomographicModels.set_conebeam, except that it also
        allocates the batch data for the projections (see also allocate_batch_data)
        
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
        if type(phis) is torch.Tensor:
            phis = phis.numpy()
        self.leapct.set_conebeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, centerRow, centerCol, phis, sod, sdd, tau, helicalPitch)
        proj_np = np.ascontiguousarray(np.zeros((self.batch_size, numAngles, numRows, numCols),dtype=np.float32), dtype=np.float32)
        self.proj_data = torch.from_numpy(proj_np)
        if self.use_gpu:
            self.proj_data = self.proj_data.float().to(self.gpu_device)
    
    def set_modularbeam(self, numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, detectorCenters, rowVec, colVec):
        """Sets the parameters for a modular-beam CT geometry
        
        The origin of the coordinate system is always at the center of rotation.
        This function is the same as leapct.tomographicModels.set_modularbeam, except that it also
        allocates the batch data for the projections (see also allocate_batch_data)
        
        Args:
            numAngles (int): number of projection angles
            numRows (int): number of rows in the x-ray detector
            numCols (int): number of columns in the x-ray detector
            pixelHeight (float): the detector pixel pitch (i.e., pixel size) between detector rows, measured in mm
            pixelWidth (float): the detector pixel pitch (i.e., pixel size) between detector columns, measured in mm
            sourcePositions ((numAngles X 3) numpy array): the (x,y,z) position of each x-ray source
            moduleCenters ((numAngles X 3) numpy array): the (x,y,z) position of the center of the front face of the detectors
            rowVectors ((numAngles X 3) numpy array):  the (x,y,z) unit vector point along the positive detector row direction
            colVectors ((numAngles X 3) numpy array):  the (x,y,z) unit vector point along the positive detector column direction
            
        Returns:
            True if the parameters were valid, false otherwise
        """
        if type(sourcePositions) is torch.Tensor:
            sourcePositions = sourcePositions.numpy()
        if type(detectorCenters) is torch.Tensor:
            detectorCenters = detectorCenters.numpy()
        if type(rowVec) is torch.Tensor:
            rowVec = rowVec.numpy()
        if type(colVec) is torch.Tensor:
            colVec = colVec.numpy()
        self.leapct.set_modularbeam(numAngles, numRows, numCols, pixelHeight, pixelWidth, sourcePositions, detectorCenters, rowVec, colVec)
        proj_np = np.ascontiguousarray(np.zeros((self.batch_size, numAngles, numRows, numCols),dtype=np.float32), dtype=np.float32)
        self.proj_data = torch.from_numpy(proj_np)
        if self.use_gpu:
            self.proj_data = self.proj_data.float().to(self.gpu_device)
        
    def get_volume_dim(self):
        """Returns the shape of the CT volume dimensions"""
        return self.leapct.get_volume_dim()
        
    def get_projection_dim(self):
        """Returns the shape of the CT projection dimensions"""
        return self.leapct.get_projection_dim()
        
    def allocate_batch_data(self):
        """Allocates the projection and volume batch data which is data that is used within this class and should be considered a private member variable"""
        vol_dim1, vol_dim2, vol_dim3 = self.get_volume_dim()
        proj_dim1, proj_dim2, proj_dim3 = self.get_projection_dim()
        
        if vol_dim1 > 0 and vol_dim2 > 0 and vol_dim3 > 0 and proj_dim1 > 0 and proj_dim2 > 0 and proj_dim3 > 0:
            self.vol_data = torch.from_numpy(np.ascontiguousarray(np.zeros((self.batch_size, vol_dim1, vol_dim2, vol_dim3),dtype=np.float32), dtype=np.float32))
            self.proj_data = torch.from_numpy(np.ascontiguousarray(np.zeros((self.batch_size, proj_dim1, proj_dim2, proj_dim3),dtype=np.float32), dtype=np.float32))
            if self.use_gpu:
                self.vol_data = self.vol_data.float().to(self.gpu_device)
                self.proj_data = self.proj_data.float().to(self.gpu_device)
                
    def load_param(self, param_fn, param_type=0): # param_type 0: cfg, 1: dict
        """Loads the LEAP parameters from file; same as leapct.tomographicModels.load_param"""
        if self.leapct.load_param(param_fn, param_type) == True:
            self.allocate_batch_data()

    def save_param(self, param_fn):
        """Saves the LEAP parameters to file; same as leapct.tomographicModels.save_param"""
        return self.leapct.save_param(param_fn)

    def set_gpu(self, which):
        """Sets the primary GPU number to be used by LEAP"""
        self.gpu_device = which
        return self.leapct.set_gpu(self.gpu_device.index)
        
    def set_gpus(self, listofgpus):
        """Sets all list of GPUs (by number) to be used by LEAP"""
        self.gpu_device = listofgpus[0]
        return self.leapct.set_gpus(listofgpus)
        
    def print_parameters(self):
        """Prints the CT geometry and CT volume parameters to the screen"""
        self.leapct.print_parameters()

   ###################################################################################################################
    ###################################################################################################################
    # THIS SECTION OF FUNCTIONS ARE ALIASES OF FUNCTIONS ABOVE INCLUDED FOR BACKWARD COMPATIBILITY
    ###################################################################################################################
    ###################################################################################################################
    def print_param(self):
        self.print_parameters()

    def set_GPU(self,which):
        return self.set_gpu(which)
        
    def set_GPUs(self,listofgpus):
        return self.set_gpus(listofgpus)


# Pytorch Projector class
class Projector(BaseProjector):
    """ Python class for PyTorch binding of LEAP
    
    Note that leapct is a member variable of this class which is an object of the leapctype.tomographicModels class.
    
    Thus all tomography functions can be access by (object of this class).leapct.XXX
    
    Usage Example:
    
    from leaptorch import Projector
    
    proj = Projector(forward_project=True, use_static=True, use_gpu=use_cuda, gpu_device=device)
    
    proj.set_conebeam(...)
    
    proj.set_default_volume(...)
    ...
    """

    def __init__(self, forward_project=True, use_static=False, use_gpu=False, gpu_device=None, batch_size=1):
        super(Projector, self).__init__(use_static, use_gpu, gpu_device, batch_size)
        self.forward_project = forward_project
    
    def fbp(self, input): # input is projection data (g batch)
        """Performs Filtered Backprojection (FBP) reconstruction of any CT geometry on the batch data"""
        for batch in range(input.shape[0]):
            if self.use_gpu:
                self.leapct.FBP_gpu(input[batch], self.vol_data[batch])
            else:
                self.leapct.FBP_cpu(input[batch], self.vol_data[batch])
        return self.vol_data

    def forward(self, input):
        """Performs the forward model on the batch data (forward projection if forward_project=True, backprojection otherwise)"""
        if self.forward_project:
            if self.use_gpu:
                return ProjectorFunctionGPU.apply(input, self.proj_data, self.vol_data, self.param_id_t)
            else:
                return ProjectorFunctionCPU.apply(input, self.proj_data, self.vol_data, self.param_id_t)
        else:
            if self.use_gpu:
                return BackProjectorFunctionGPU.apply(input, self.proj_data, self.vol_data, self.param_id_t)
            else:
                return BackProjectorFunctionCPU.apply(input, self.proj_data, self.vol_data, self.param_id_t)



class FBP(BaseProjector):

    def __init__(self, forward_FBP=True, use_static=False, use_gpu=False, gpu_device=None, batch_size=1):
        super(FBP, self).__init__(use_static, use_gpu, gpu_device, batch_size)
        self.forward_FBP = forward_FBP

    def forward(self, input):
        if self.forward_FBP:
            if self.use_gpu:
                return FBPFunctionGPU.apply(input, self.proj_data, self.vol_data, self.param_id_t)
            else:
                return FBPFunctionCPU.apply(input, self.proj_data, self.vol_data, self.param_id_t)
        else:
            if self.use_gpu:
                return FBPReverseFunctionGPU.apply(input, self.proj_data, self.vol_data, self.param_id_t)
            else:
                return FBPReverseFunctionCPU.apply(input, self.proj_data, self.vol_data, self.param_id_t)

