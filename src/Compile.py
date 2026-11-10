import os
import subprocess

os.chdir('.')

arch = subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader','--id=0'],text=True).strip().replace('.', '')
nvcc_flag = '-gencode=arch=compute_' + arch + ',code=sm_' + arch

# mexcuda(nvcc_flag, 'simFDTD.cu'): same nvcc options as mexcuda (nvcc_g++.xml), but builds
# the ctypes library libsimFDTD.so from simFDTD_py.cu instead of a MEX file
subprocess.run(['nvcc', nvcc_flag, '-shared', '-std=c++11', '-O2', '-DNDEBUG',
                '--compiler-options=-D_GNU_SOURCE,-fexceptions,-fPIC,-fno-omit-frame-pointer,-pthread,-fwrapv',
                '-Ipython_include', '-o', 'libsimFDTD.so', 'simFDTD_py.cu'], check=True)
