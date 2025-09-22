#include "simulation.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    float *sigma;
    int *ant_pos;
    int *dim_ant;
    sigma = (float *) mxGetPr(prhs[3]);
    dim_ant = (int *) mxGetPr(prhs[2]);
    ant_pos = (int *) mxGetPr(prhs[1]);
    simulation * sim = (simulation *) (*(uint64_T*) mxGetData(prhs[0]));
    sim -> initAntCond(ant_pos, dim_ant, sigma);
}