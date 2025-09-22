#include "matlabconfig.h"



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    simFDTD *sim = new simFDTD;
    //Set the dimensions for the output
    const mwSize dimensions[] = {1, 1};

    //Create mxArray output object
    mxArray *output = mxCreateNumericArray(2, dimensions, mxUINT64_CLASS, mxREAL);
    //Create a pointer to the output data
    uint64_T *data = (uint64_T *) mxGetData(output);

    //Put the pointer in the data
    data[0] = (uint64_T) sim;

    //set ouput
    plhs[0] = output;
}