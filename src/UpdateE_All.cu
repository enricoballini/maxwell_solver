#include "matlabconfig.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    
    simFDTD * sim = (simFDTD *) (*(uint64_T*) mxGetPr(prhs[0]));   
    sim -> updateE_All();
}