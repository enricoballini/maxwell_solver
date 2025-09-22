#include "matlabconfig.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    int *Locations;
    
    simFDTD * sim = (simFDTD *) (*(uint64_T*) mxGetData(prhs[0]));
    Locations = (int *) mxGetPr(prhs[1]);
    
    sim -> setAntennaLocations(Locations);
}