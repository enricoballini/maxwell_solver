#include "simulation.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 2) {
        mexErrMsgIdAndTxt("UpdateField:InvalidInput",
                          "Usage: UpdateField(simPtr, mode);");
    }

    simulation *sim = (simulation *) (*(uint64_T*) mxGetPr(prhs[0]));
    int mode = (int) mxGetScalar(prhs[1]);

    if (mode == 1) {
        sim->updateE_All();
    }
    else if (mode == 2) {
        sim->updateE_PML();
    }
    else if (mode == 3) {
        sim->updateH_All();
    }
    else {
        mexErrMsgIdAndTxt("UpdateField:BadMode",
                          "Invalid mode %d. Use 1, 2 or 3.", mode);
    }
}