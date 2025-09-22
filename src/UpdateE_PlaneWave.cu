#include "simulation.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int *time;
    simulation *sim = (simulation *)(*(uint64_T *)mxGetData(prhs[0]));
    time = (int *)mxGetPr(prhs[1]);
    sim->updateE_PlaneWave(time);
}