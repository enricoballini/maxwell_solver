#include "simulation.h"

static char toLowerAscii(char value)
{
    if (value >= 'A' && value <= 'Z')
        return value + ('a' - 'A');
    return value;
}

static void lowerAsciiInPlace(char *value)
{
    while (*value != '\0')
    {
        *value = toLowerAscii(*value);
        ++value;
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    if (nrhs < 1)
    {
        mexErrMsgIdAndTxt("simFDTD:InvalidInput", "At least one input required: operation string.");
        return;
    }

    char *op = mxArrayToString(prhs[0]);
    lowerAsciiInPlace(op);

    if (strcmp(op, "createsimulation") == 0)
    {
        simulation *sim = new simulation;
        const mwSize dimensions[] = {1, 1};
        mxArray *output = mxCreateNumericArray(2, dimensions, mxUINT64_CLASS, mxREAL);
        uint64_T *data = (uint64_T *)mxGetData(output);
        data[0] = (uint64_T)sim;
        plhs[0] = output;
    }
    else
    {
        if (nrhs < 2)
        {
            mexErrMsgIdAndTxt("simFDTD:InvalidInput", "Simulation pointer required for this operation.");
            mxFree(op);
            return;
        }
        simulation *sim = (simulation *)(*(uint64_T *)mxGetData(prhs[1]));

        if (strcmp(op, "deletesimulation") == 0)
        {
            delete sim;
        }
        else if (strcmp(op, "updatee_all") == 0)
        {
            sim->updateE_All();
        }
        else if (strcmp(op, "updateh_all") == 0)
        {
            sim->updateH_All();
        }
        else if (strcmp(op, "updatee_pml") == 0)
        {
            sim->updateE_PML();
        }
        else if (strcmp(op, "updateh_pml") == 0)
        {
            sim->updateH_PML();
        }
        else if (strcmp(op, "updatecuda") == 0)
        {
            sim->updateCuda();
        }
        else if (strcmp(op, "initfield") == 0)
        {
            sim->initField();
        }
        else if (strcmp(op, "addcurrents") == 0)
        {
            if (nrhs < 3)
                mexErrMsgIdAndTxt("simFDTD:InvalidInput", "time and indices required.");
            int *time = (int *)mxGetPr(prhs[2]);
            int *ind = (int *)mxGetPr(prhs[3]);
            sim->addCurrents(time, ind);
        }
        else if (strcmp(op, "readysimulation") == 0)
        {
            sim->readySimulation();
        }
        else if (strcmp(op, "resetfieldall") == 0)
        {
            sim->ResetFieldAll();
        }
        else if (strcmp(op, "updatee_planewave") == 0)
        {
            if (nrhs < 3)
                mexErrMsgIdAndTxt("simFDTD:InvalidInput", "time required.");
            int *time = (int *)mxGetPr(prhs[2]);
            sim->updateE_PlaneWave(time);
        }
        else if (strcmp(op, "updateh_planewave") == 0)
        {
            if (nrhs < 3)
                mexErrMsgIdAndTxt("simFDTD:InvalidInput", "time required.");
            int *time = (int *)mxGetPr(prhs[2]);
            sim->updateH_PlaneWave(time);
        }
        else if (strcmp(op, "setconst") == 0)
        {
            if (nrhs < 4)
                mexErrMsgIdAndTxt("simFDTD:InvalidInput", "freqRes and TimeDim required.");
            float *TopfreqRes = (float *)mxGetPr(prhs[2]);
            int *TopTimeDim = (int *)mxGetPr(prhs[3]);
            sim->setConst(TopfreqRes, TopTimeDim);
        }
        else if (strcmp(op, "setsource") == 0)
        {
            if (nrhs < 3)
                mexErrMsgIdAndTxt("simFDTD:InvalidInput", "SourceIn required.");
            float *SourceIn = (float *)mxGetPr(prhs[2]);
            if (nrhs > 3)
            {
                mexPrintf("Update Source Mode!\n");
                sim->updateSource(SourceIn);
            }
            else
            {
                sim->setSource(SourceIn);
            }
        }
        else if (strcmp(op, "setantennalocations") == 0)
        {
            if (nrhs < 3)
                mexErrMsgIdAndTxt("simFDTD:InvalidInput", "Locations required.");
            int *Locations = (int *)mxGetPr(prhs[2]);
            sim->setAntennaLocations(Locations);
        }
        else if (strcmp(op, "createmu") == 0)
        {
            if (nrhs < 5)
                mexErrMsgIdAndTxt("simFDTD:InvalidInput", "mu_x, mu_y, mu_z required.");
            float *mu_x = (float *)mxGetPr(prhs[2]);
            float *mu_y = (float *)mxGetPr(prhs[3]);
            float *mu_z = (float *)mxGetPr(prhs[4]);
            sim->createMu(mu_x, mu_y, mu_z);
        }
        else if (strcmp(op, "init1d") == 0)
        {
            if (nrhs < 4)
                mexErrMsgIdAndTxt("simFDTD:InvalidInput", "rate and PW_source required.");
            float *rate = (float *)mxGetPr(prhs[2]);
            float *PW_source = (float *)mxGetPr(prhs[3]);
            sim->init1D(rate, PW_source);
        }
        else if (strcmp(op, "updatematerial") == 0)
        {
            if (nrhs < 4)
                mexErrMsgIdAndTxt("simFDTD:InvalidInput", "flag, values required.");
            int *flag = (int *)mxGetPr(prhs[2]);
            float *values = (float *)mxGetPr(prhs[3]);
            sim->updateMaterial(flag, values);
        }
       
        else if (strcmp(op, "outputant") == 0)
        {
            if (nrhs < 3)
                mexErrMsgIdAndTxt("simFDTD:InvalidInput", "OutAntSize required.");
            unsigned int OutAntSize = *((unsigned int *)mxGetData(prhs[2]));
            const mwSize dims[] = {1, OutAntSize};
            plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
            float *antData = (float *)mxGetPr(plhs[0]);
            sim->outputAnt(antData);
        }
        else if (strcmp(op, "outputvis") == 0)
        {
            if (nrhs < 4)
                mexErrMsgIdAndTxt("simFDTD:InvalidInput", "OutAntSize and flag required.");
            unsigned int OutAntSize = *((unsigned int *)mxGetData(prhs[2]));
            int *flag = (int *)mxGetPr(prhs[3]);
            const mwSize dims[] = {1, OutAntSize};
            plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
            float *Data = (float *)mxGetPr(plhs[0]);
            sim->outputVis(Data, flag);
        }
        else
        {
            mexErrMsgIdAndTxt("simFDTD:UnknownOperation", "Unknown operation: %s", op);
        }
    }

    mxFree(op);
}

