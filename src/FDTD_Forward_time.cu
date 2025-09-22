#include "simulation.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    int *time;
    int *mode;
    simulation * sim = (simulation *) (*(uint64_T*) mxGetData(prhs[0]));
    mode = (int *) mxGetPr(prhs[2]);
    time = (int *) mxGetPr(prhs[1]);

    sim -> FDTD_Forward_time(time,mode);
}