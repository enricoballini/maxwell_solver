#include "simulation.h"
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    float  *mu_x;
    float  *mu_y;
    float  *mu_z;
    
    simulation * sim = (simulation *) (*(uint64_T*) mxGetData(prhs[0]));
    mu_x = (float *) mxGetPr(prhs[1]);
    mu_y = (float *) mxGetPr(prhs[2]);
    mu_z = (float *) mxGetPr(prhs[3]);
    
    sim -> createMu(mu_x,mu_y,mu_z);
}