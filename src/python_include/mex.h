// Empty stand-in for MATLAB's mex.h, used only when Compile.py builds libsimFDTD.so
// for Python. simulation.h includes <mex.h> but uses nothing from it, and MATLAB's
// mex.h would turn printf into mexPrintf, which does not exist outside MATLAB.
