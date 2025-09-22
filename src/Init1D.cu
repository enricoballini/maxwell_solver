#include "matlabconfig.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float *rate;
    float *PW_source;
    simFDTD *sim = (simFDTD *)(*(uint64_T *)mxGetData(prhs[0]));
    rate = (float *)mxGetPr(prhs[1]);
    PW_source = (float *)mxGetPr(prhs[2]);
    sim->init1D(rate, PW_source);
}