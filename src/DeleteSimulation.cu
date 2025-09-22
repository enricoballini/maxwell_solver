#include "matlabconfig.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    
    simFDTD * sim = (simFDTD *) (*(uint64_T*) mxGetData(prhs[0]));   
    sim -> ~simFDTD();
}