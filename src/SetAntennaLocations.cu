#include "simulation.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    int *Locations;
    
    simulation * sim = (simulation *) (*(uint64_T*) mxGetData(prhs[0]));
    Locations = (int *) mxGetPr(prhs[1]);
    
    sim -> setAntennaLocations(Locations);
}