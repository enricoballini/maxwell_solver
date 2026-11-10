import os
import shutil
import subprocess
import sysconfig

os.chdir('.')

arch = subprocess.check_output(['nvidia-smi','--query-gpu=compute_cap','--format=csv,noheader','--id=0'],text=True).strip().replace('.', '')
nvcc_flag = '-gencode=arch=compute_' + arch + ',code=sm_' + arch

# nvcc on PATH (system CUDA toolkit, HPC module, container), otherwise the one pip installs from requirements.txt
nvcc = shutil.which('nvcc') or os.path.join(sysconfig.get_paths()['purelib'], 'nvidia', 'cu13', 'bin', 'nvcc')
print('nvcc:', nvcc, flush=True)

# mexcuda(nvcc_flag, 'simFDTD.cu'): same nvcc options as mexcuda (nvcc_g++.xml), but builds
# the ctypes library libsimFDTD.so from simFDTD_py.cu instead of a MEX file
subprocess.run([nvcc, nvcc_flag, '-shared', '-std=c++11', '-O2', '-DNDEBUG',
                '--compiler-options=-D_GNU_SOURCE,-fexceptions,-fPIC,-fno-omit-frame-pointer,-pthread,-fwrapv',
                '-o', 'libsimFDTD.so', 'simFDTD_py.cu'], check=True)
