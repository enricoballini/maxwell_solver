#include "simulation.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    mwSize     ndim = 2;
    
    double *Data;
    int *flag;
    simulation * sim = (simulation *) (*(uint64_T*) mxGetData(prhs[0]));   
    
    unsigned int OutAntSize;
    OutAntSize = *( (unsigned int*) mxGetData(prhs[1]));
    flag = (int *) mxGetPr(prhs[2]);
    const mwSize dims[] = {1,OutAntSize};
    
    plhs[0] = mxCreateNumericArray(ndim,dims,mxDOUBLE_CLASS, mxREAL);
    Data = (double *) mxGetPr(plhs[0]);

    sim -> outputVisDouble(Data,flag[0]);
}