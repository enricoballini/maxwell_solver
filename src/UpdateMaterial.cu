#include "simulation.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float *values;
    int *flag;
    simulation * sim = (simulation *) (*(uint64_T*) mxGetData(prhs[0]));
    flag = (int *) mxGetPr(prhs[1]);
    values = (float *) mxGetPr(prhs[2]);
    sim -> updateMaterial(flag, values);
}