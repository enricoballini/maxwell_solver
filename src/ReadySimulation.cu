#include "simulation.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    
    simulation * sim = (simulation *) (*(uint64_T*) mxGetData(prhs[0]));   
    sim -> readySimulation();
}