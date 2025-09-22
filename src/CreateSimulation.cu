
#include "simulation.h"
// #include "mex.hpp"
// #include "mexAdapter.hpp"
// using namespace matlab::data;
// using namespace matlab::mex;
// class MexFunction : public matlab::mex::Function
// {
// public:
//     void operator()(ArgumentList outputs, ArgumentList inputs)
//     {
//         matlab::data::ArrayFactory factory;

//         // Create a new simulation object
//         simulation *sim = new simulation;

//         // Wrap the pointer into a uint64 scalar
//         TypedArray<uint64_t> outArray = factory.createScalar(reinterpret_cast<uint64_t>(sim));

//         // Assign to output
//         outputs[0] = outArray;
//     }
// };

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    simulation *sim = new simulation;
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