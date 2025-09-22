#include "simulation.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    float *sigma_anty;
    float *sigma_antz;
    int * k;
    simulation * sim = (simulation *) (*(uint64_T*) mxGetData(prhs[0]));
    sigma_anty = (float *) mxGetPr(prhs[1]);
    sigma_antz = (float *) mxGetPr(prhs[2]);
    k = (int *) mxGetPr(prhs[3]);
    
    sim -> updateAntCondML(sigma_anty,sigma_antz,k);
}