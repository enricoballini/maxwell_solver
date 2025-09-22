#include "matlabconfig.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    mwSize     ndim = 2;
    
    float *antData;
    
    simFDTD * sim = (simFDTD *) (*(uint64_T*) mxGetData(prhs[0]));   
    
    unsigned int OutAntSize;
    OutAntSize = *( (unsigned int*) mxGetData(prhs[1]));
    const mwSize dims[] = {1,OutAntSize};
    
    plhs[0] = mxCreateNumericArray(ndim,dims,mxSINGLE_CLASS, mxREAL);
    antData = (float *) mxGetPr(plhs[0]);
    
    sim -> outputAnt(antData);
}