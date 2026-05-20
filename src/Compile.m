cd .

arch = strrep(gpuDevice().ComputeCapability, '.', '');  
nvcc_flag = ['NVCCFLAGS=-gencode=arch=compute_' arch ',code=sm_' arch ];
tic
mexcuda(nvcc_flag, 'simFDTD.cu');
toc
