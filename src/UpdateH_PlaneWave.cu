#include "matlabconfig.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int *time;
    simFDTD *sim = (simFDTD *)(*(uint64_T *)mxGetData(prhs[0]));
    time = (int *)mxGetPr(prhs[1]);
    sim->updateH_PlaneWave(time);
}