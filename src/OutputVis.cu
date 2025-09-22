#include "matlabconfig.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    mwSize     ndim = 2;
    
    float *Data;
    int *flag;
    simFDTD * sim = (simFDTD *) (*(uint64_T*) mxGetData(prhs[0]));   
    
    unsigned int OutAntSize;
    OutAntSize = *( (unsigned int*) mxGetData(prhs[1]));
    flag = (int *) mxGetPr(prhs[2]);
    const mwSize dims[] = {1,OutAntSize};
    
    plhs[0] = mxCreateNumericArray(ndim,dims,mxSINGLE_CLASS, mxREAL);
    Data = (float *) mxGetPr(plhs[0]);

    sim -> outputVis(Data,flag);
}