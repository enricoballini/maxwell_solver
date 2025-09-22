#include "matlabconfig.h"
// //// *const: [c,mu0,eps0,EpsS,eta]
// ///  *freqRes: [fctr, delX, delT, tauP]
// ///  *CPML_param: [alpha_m, sig_max, alpha_max, kappa_max]
// ///  *TimeDim: [max_TimeSteps, I_size, J_size, K_size, pml_size, poly_m]
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{ 
    float  *TopfreqRes;
    int *TopTimeDim;
    
    simFDTD * sim = (simFDTD *) (*(uint64_T*) mxGetData(prhs[0]));
    TopfreqRes = (float *) mxGetPr(prhs[1]);
    TopTimeDim = (int *) mxGetPr(prhs[2]);
    
    sim -> setConst(TopfreqRes, TopTimeDim);
}