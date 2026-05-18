#define _USE_MATH_DEFINES
#include "mex.h"
#include <cmath>
#include <cstring>

void doCreateSignal(int nlhs, mxArray *plhs[], int nrhs,
                    const mxArray *prhs[]) {
  if (nrhs != 6)
    mexErrMsgIdAndTxt("FunLib:CreateSignal",
                      "Usage: FunLib('CreateSignal', fctr, delT, PercentageBW, "
                      "NumOfSidLobes, timeSteps)");

  double fctr = mxGetScalar(prhs[1]);
  double delT = mxGetScalar(prhs[2]);
  double PercentageBW = mxGetScalar(prhs[3]);
  double NumOfSidLobes = mxGetScalar(prhs[4]);
  size_t timeSteps = (size_t)mxGetScalar(prhs[5]);

  double fmaxx = fctr * PercentageBW / 2.0;
  plhs[0] = mxCreateDoubleScalar(fmaxx);

  // Create source output (1 x timeSteps)
  plhs[1] = mxCreateDoubleMatrix(1, timeSteps, mxREAL);
  double *source = mxGetPr(plhs[1]);

  double spread = 1.0 / (2.0 * delT * fmaxx);
  double tto = NumOfSidLobes * spread;

  for (size_t n = 0; n < timeSteps; ++n) {
    double nn = (double)n + 1.0;
    double value = 0.0;

    if (nn < 2.0 * tto) {
      double arg = 2.0 * M_PI * fmaxx * (nn - tto) * delT;
      double denom = (fabs(arg) < 1e-14) ? 1e-14 : arg;
      double sinc = sin(arg) / denom;

      value = sinc * sin(2.0 * M_PI * fctr * (nn - tto) * delT);
    }

    // Window
    value *= cos(((nn - 1.0) / (double)timeSteps) * (M_PI / 2.0));
    source[n] = value;
  }
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
  if (nrhs < 2) {
    mexErrMsgIdAndTxt("MyMex:InputError", "At least two inputs required.");
  }
  if (!mxIsChar(prhs[0])) {
    mexErrMsgIdAndTxt("MyMex:TypeError", "First input must be a char.");
  }
  char command[64];
  mxGetString(prhs[0], command, sizeof(command));
  if (strcmp(command, "CreateSignal") == 0 || strcmp(command, "Signal") == 0) {
    mexPrintf("Create Signal!\n");
    doCreateSignal(nlhs, plhs, nrhs, prhs);
  } else {
    mexErrMsgIdAndTxt("MyMex:CommandError",
                      "Unknown command '%s'. Valid options: 'Init', 'Reset', "
                      "'Update', 'Delete', 'Const', 'Source', 'Ant'. ",
                      command);
  }
}
