// C entry points for Python (ctypes): one function per operation of the MEX gateway
// simFDTD.cu, calling the same simulation methods. The op-string dispatch and the
// argument checks are in simFDTD.py. Built into libsimFDTD.so by Compile.py.
#include "simulation.h"

extern "C"
{
    void *simFDTD_createsimulation() { return new simulation; }
    void simFDTD_deletesimulation(void *sim) { delete (simulation *)sim; }
    void simFDTD_updatee_all(void *sim) { ((simulation *)sim)->updateE_All(); }
    void simFDTD_updateh_all(void *sim) { ((simulation *)sim)->updateH_All(); }
    void simFDTD_updatee_pml(void *sim) { ((simulation *)sim)->updateE_PML(); }
    void simFDTD_updateh_pml(void *sim) { ((simulation *)sim)->updateH_PML(); }
    void simFDTD_updatecuda(void *sim) { ((simulation *)sim)->updateCuda(); }
    // InitField: removed from simFDTD.cu in commit 48b7151, still called by highfrequency.m/.py
    void simFDTD_initfield(void *sim) { ((simulation *)sim)->initField(); }
    void simFDTD_addcurrents(void *sim, int *time, int *ind) { ((simulation *)sim)->addCurrents(time, ind); }
    void simFDTD_readysimulation(void *sim) { ((simulation *)sim)->readySimulation(); }
    void simFDTD_resetfieldall(void *sim) { ((simulation *)sim)->ResetFieldAll(); }
    void simFDTD_updatee_planewave(void *sim, int *time) { ((simulation *)sim)->updateE_PlaneWave(time); }
    void simFDTD_updateh_planewave(void *sim, int *time) { ((simulation *)sim)->updateH_PlaneWave(time); }
    void simFDTD_setconst(void *sim, float *TopfreqRes, int *TopTimeDim) { ((simulation *)sim)->setConst(TopfreqRes, TopTimeDim); }
    void simFDTD_setsource(void *sim, float *SourceIn) { ((simulation *)sim)->setSource(SourceIn); }
    void simFDTD_updatesource(void *sim, float *SourceIn) { ((simulation *)sim)->updateSource(SourceIn); }
    void simFDTD_setantennalocations(void *sim, int *Locations) { ((simulation *)sim)->setAntennaLocations(Locations); }
    void simFDTD_createmu(void *sim, float *mu_x, float *mu_y, float *mu_z) { ((simulation *)sim)->createMu(mu_x, mu_y, mu_z); }
    void simFDTD_init1d(void *sim, float *rate, float *PW_source) { ((simulation *)sim)->init1D(rate, PW_source); }
    void simFDTD_updatematerial(void *sim, int *flag, float *values) { ((simulation *)sim)->updateMaterial(flag, values); }
    void simFDTD_outputant(void *sim, float *antData) { ((simulation *)sim)->outputAnt(antData); }
    void simFDTD_outputvis(void *sim, float *Data, int *flag) { ((simulation *)sim)->outputVis(Data, flag); }

    // for reset(gpuDevice) in the scripts
    void simFDTD_devicereset() { cudaDeviceReset(); }
}
