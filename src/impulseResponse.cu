#include "simulation.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    float *data;
    int * num;
    simulation * sim = (simulation *) (*(uint64_T*) mxGetData(prhs[0]));
    data = (float *) mxGetPr(prhs[1]);
    num = (int *) mxGetPr(prhs[2]);
    sim -> impulse_response(data,num);
}