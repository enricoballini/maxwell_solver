cd .

pathmex='./MexFunc ';
arch = strrep(gpuDevice().ComputeCapability, '.', '');  
nvcc_flag = ['NVCCFLAGS=-gencode=arch=compute_' arch ',code=sm_' arch];

mexcuda('-output',pathmex,nvcc_flag, 'CreateSimulation.cu');
mexcuda('-output',pathmex,nvcc_flag, 'SetConst.cu');
mexcuda('-output',pathmex,nvcc_flag, 'ReadySimulation.cu');
mexcuda('-output',pathmex,nvcc_flag, 'InitField.cu');
mexcuda('-output',pathmex,nvcc_flag, 'SetAntennaLocations.cu');
mexcuda('-output',pathmex,nvcc_flag, 'SetSource.cu'); 

mexcuda('-output',pathmex,nvcc_flag, 'DeleteSimulation.cu');
mexcuda('-output',pathmex,nvcc_flag, 'ResetFieldAll_1.cu'); 

mexcuda('-output',pathmex,nvcc_flag, 'OutputAnt.cu');
mexcuda('-output',pathmex,nvcc_flag, 'OutputVis.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'InitAntCond.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'UpdateAntCond.cu'); 

mexcuda('-output',pathmex,nvcc_flag, 'Init1D.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'InitAdjoint.cu'); 
mexcuda('-output',pathmex,nvcc_flag,'CXXFLAGS=$CXXFLAGS -fopenmp', 'LDFLAGS=$LDFLAGS -fopenmp','Gradient_calculation_all.cu');
%mexcuda Gradient_calculation_all.cu   'NVCCFLAGS=-gencode=arch=compute_86,code=sm_86'  CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" 
%mexcuda Gradient_calculation_all.cu  CXXFLAGS="\$CXXFLAGS -Xcompiler -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" 'NVCCFLAGS=-gencode=arch=compute_86,code=sm_86' 

mexcuda('-output',pathmex,nvcc_flag, 'UpdateMaterial.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'impulseResponse.cu'); 

mexcuda('-output',pathmex,nvcc_flag, 'UpdateH_All.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'UpdateH_PML.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'UpdateE_All.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'UpdateE_PML.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'CoaxUpdate.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'CreateMu.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'RecordForwardField.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'UpdateH_PlaneWave.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'UpdateE_PlaneWave.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'GradientGPU.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'RecordGPUField.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'InitDesignGPU.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'InitAntCondML.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'UpdateAntCondML.cu'); 

mexcuda('-output',pathmex,nvcc_flag, 'OutputVisDouble.cu'); 
mexcuda('-output',pathmex,nvcc_flag, 'UpdateCuda.cu'); 

% mex CreateSimulation.cu  gpu.obj -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64" -lcudart
% mex -largeArrayDims CreateSimulation.cpp gpu.obj -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64" -lcudart
