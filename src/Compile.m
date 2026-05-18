cd .

arch = strrep(gpuDevice().ComputeCapability, '.', '');  
nvcc_flag = ['NVCCFLAGS=-gencode=arch=compute_' arch ',code=sm_' arch ];

mexcuda(nvcc_flag, 'simFDTD.cu');



mexcuda(nvcc_flag, 'ReadySimulation.cu');
mexcuda(nvcc_flag, 'InitField.cu');
mexcuda(nvcc_flag, 'SetSource.cu'); 

mexcuda(nvcc_flag, 'DeleteSimulation.cu');
mexcuda(nvcc_flag, 'ResetFieldAll_1.cu'); 

mexcuda(nvcc_flag, 'OutputAnt.cu');
mexcuda(nvcc_flag, 'OutputVis.cu'); 

mexcuda(nvcc_flag, 'Init1D.cu'); 
mexcuda( nvcc_flag,'CXXFLAGS=$CXXFLAGS -fopenmp', 'LDFLAGS=$LDFLAGS -fopenmp','Gradient_calculation_all.cu');
%mexcuda Gradient_calculation_all.cu   'NVCCFLAGS=-gencode=arch=compute_86,code=sm_86'  CXXFLAGS="\$CXXFLAGS -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" 
%mexcuda Gradient_calculation_all.cu  CXXFLAGS="\$CXXFLAGS -Xcompiler -fopenmp" LDFLAGS="\$LDFLAGS -fopenmp" 'NVCCFLAGS=-gencode=arch=compute_86,code=sm_86' 

mexcuda(nvcc_flag, 'UpdateMaterial.cu'); 

mexcuda(nvcc_flag, 'UpdateH_All.cu'); 
mexcuda(nvcc_flag, 'UpdateH_PML.cu'); 
mexcuda(nvcc_flag, 'UpdateE_All.cu'); 
mexcuda(nvcc_flag, 'UpdateE_PML.cu'); 
mexcuda(nvcc_flag, 'CoaxUpdate.cu'); 
mexcuda(nvcc_flag, 'CreateMu.cu'); 
mexcuda(nvcc_flag, 'RecordForwardField.cu'); 
mexcuda(nvcc_flag, 'UpdateH_PlaneWave.cu'); 
mexcuda(nvcc_flag, 'UpdateE_PlaneWave.cu'); 
mexcuda(nvcc_flag, 'RecordGPUField.cu'); 

mexcuda(nvcc_flag, 'OutputVisDouble.cu'); 
mexcuda(nvcc_flag, 'UpdateCuda.cu'); 


