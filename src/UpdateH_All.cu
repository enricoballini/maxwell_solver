#include "simulation.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    
    simulation * sim = (simulation *) (*(uint64_T*) mxGetPr(prhs[0]));   
    sim -> updateH_All();
}