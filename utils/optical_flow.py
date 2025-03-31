import cv2
import torch

class OpticalFlowTransform(object):
    """
    Optical Flow Transform.
    Applies a transformation for each data sample while considering the previous sample.
    """

    def __init__(self, algorithm="tv-l1", flow_scale=1.0, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2):
        self.opencv_cuda = self.check_opencv_cuda()
        self.flow_scale = flow_scale
        self.pyr_scale = pyr_scale
        self.levels = levels
        self.winsize = winsize
        self.iterations = iterations
        self.poly_n = poly_n
        self.poly_sigma = poly_sigma
        
        if (algorithm == "tv-l1"):
            self.transform_func = self.optical_flow_tvl1
        elif (algorithm == "farneback"):
            self.transform_func = self.optical_flow_farneback
        else:
            raise ValueError(f"The optical flow options are either 'tv-l1' or 'farneback', got {algorithm}")

    def __call__(self, sequence):
        transformed_sequences = []
        transformed_data = None
        for i in range(len(sequence)):
            curr_data = sequence[i]
            prev_data = None if i == 0 else sequence[i-1]
            transformed_data = self.transform_func(curr_data, prev_data)
            transformed_sequences.append(transformed_data)
        transformed_sequences = torch.stack(transformed_sequences, dim=0)
        return transformed_sequences
    
    def optical_flow_tvl1(self, curr_data, prev_data, flow=None):
        """ Using TV-L1 Algorithm """
        if (prev_data == None):
            prev_data = torch.zeros_like(curr_data)
        curr_frame = curr_data[0,:,:].numpy()
        prev_frame = prev_data[0,:,:].numpy()
        
        if (self.opencv_cuda):
            gpu_flow = cv2.cuda_OpticalFlowDual_TVL1.create()
            gpu_prev_frame = cv2.cuda_GpuMat()
            gpu_curr_frame = cv2.cuda_GpuMat()
            gpu_prev_frame.upload(prev_frame)
            gpu_curr_frame.upload(curr_frame)
            gpu_flow_result = gpu_flow.calc(gpu_prev_frame, gpu_curr_frame, None)
            gpu_prev_frame.release()
            gpu_curr_frame.release()
            flow = gpu_flow_result.download()  # Download the result back to CPU
            gpu_flow_result.release()
        else:
            optical_flow = cv2.optflow.createOptFlow_DualTVL1()
            flow = optical_flow.calc(prev_frame, curr_frame, None)
        
        flow = torch.from_numpy(flow).float() * self.flow_scale
        flow = flow.permute(2, 0, 1) # Change to (C, H, W) format

        return flow
    
    def optical_flow_farneback(self, curr_data, prev_data, flow=None):
        """ Using farneback Algorithm """
        if (prev_data == None):
            prev_data = torch.zeros_like(curr_data)
        curr_frame = curr_data[0,:,:].numpy()
        prev_frame = prev_data[0,:,:].numpy()
        
        flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, None, self.pyr_scale, self.levels, self.winsize,
                                                self.iterations, self.poly_n, self.poly_sigma, 0)
        
        flow = torch.from_numpy(flow).float() * self.flow_scale
        flow = flow.permute(2, 0, 1) # Change to (C, H, W) format

        return flow
    
    def check_opencv_cuda(self):
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            print("CUDA is enabled in OpenCV.")
            print("Number of CUDA devices:", cv2.cuda.getCudaEnabledDeviceCount())
            device_id = cv2.cuda.getDevice()
            print(f"current used cuda device id: {device_id}")
            cv2.cuda.printShortCudaDeviceInfo(device_id)
            return True
        else:
            print("CUDA is NOT enabled in OpenCV.")
            return False