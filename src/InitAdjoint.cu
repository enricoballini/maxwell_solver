#include "simulation.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    float *source;
    simulation * sim = (simulation *) (*(uint64_T*) mxGetData(prhs[0]));   
    source = (float *) mxGetPr(prhs[1]);
    sim -> InitAdjoint(source);
}