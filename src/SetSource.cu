#include "matlabconfig.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    float *SourceIn;
    simFDTD * sim = (simFDTD *) (*(uint64_T*) mxGetData(prhs[0]));   
    SourceIn = (float *) mxGetPr(prhs[1]);
    
    sim -> setSource(SourceIn);
}