#include "simulation.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    int *flag;
    simulation * sim = (simulation *) (*(uint64_T*) mxGetData(prhs[0])); 
    flag = (int *) mxGetPr(prhs[1]);
    sim -> computeGradientGPU(flag);
}