# Python counterpart of the MEX gateway simFDTD.cu: same operation strings (plus InitField,
# see below), same argument checks, same outputs. It calls the functions in libsimFDTD.so
# (built by Compile.py from simFDTD_py.cu) through ctypes.
#
# Like the MEX gateway, the arrays are passed to C as raw pointers without type conversion,
# so the caller must pass int32/single arrays where the gateway expects int/float
# (the scripts do this with TypeInt/TypeReal).
import ctypes
import os

import numpy as np

_lib = ctypes.CDLL(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "libsimFDTD.so")
)
_lib.simFDTD_createsimulation.restype = ctypes.c_void_p
for _name, _nptr in [
    ("deletesimulation", 1),
    ("updatee_all", 1),
    ("updateh_all", 1),
    ("updatee_pml", 1),
    ("updateh_pml", 1),
    ("updatecuda", 1),
    ("initfield", 1),
    ("addcurrents", 3),
    ("readysimulation", 1),
    ("resetfieldall", 1),
    ("updatee_planewave", 2),
    ("updateh_planewave", 2),
    ("setconst", 3),
    ("setsource", 2),
    ("updatesource", 2),
    ("setantennalocations", 2),
    ("createmu", 4),
    ("init1d", 3),
    ("updatematerial", 3),
    ("outputant", 2),
    ("outputvis", 3),
]:
    getattr(_lib, "simFDTD_" + _name).argtypes = [ctypes.c_void_p] * _nptr
    getattr(_lib, "simFDTD_" + _name).restype = None
_lib.simFDTD_devicereset.argtypes = []
_lib.simFDTD_devicereset.restype = None


def reset_gpuDevice():
    # reset(gpuDevice)
    _lib.simFDTD_devicereset()


def _mexErrMsgIdAndTxt(errorid, errormsg):
    raise RuntimeError(errorid + ": " + errormsg)


def _data(x):
    # like mxGetData/mxGetPr: the array's data in MATLAB (column-major) memory order
    return np.asfortranarray(x)


def simFDTD(*prhs):
    nrhs = len(prhs)
    if nrhs < 1:
        _mexErrMsgIdAndTxt(
            "simFDTD:InvalidInput", "At least one input required: operation string."
        )

    op = prhs[0].lower()

    if op == "createsimulation":
        return np.uint64(_lib.simFDTD_createsimulation())

    if nrhs < 2:
        _mexErrMsgIdAndTxt(
            "simFDTD:InvalidInput", "Simulation pointer required for this operation."
        )
    sim = int(
        _data(prhs[1]).view(np.uint64)[0]
    )  # (simulation *)(*(uint64_T *)mxGetData(prhs[1]))

    if op == "deletesimulation":
        _lib.simFDTD_deletesimulation(sim)
    elif op == "updatee_all":
        _lib.simFDTD_updatee_all(sim)
    elif op == "updateh_all":
        _lib.simFDTD_updateh_all(sim)
    elif op == "updatee_pml":
        _lib.simFDTD_updatee_pml(sim)
    elif op == "updateh_pml":
        _lib.simFDTD_updateh_pml(sim)
    elif op == "updatecuda":
        _lib.simFDTD_updatecuda(sim)
    elif op == "initfield":
        # Removed from the MEX gateway in commit 48b7151, when ReadySimulation started to call
        # initField() itself. Kept here so that highfrequency.py runs as highfrequency.m did with
        # a MEX built before that commit; after ReadySimulation it only prints that the field
        # arrays are already initialized.
        _lib.simFDTD_initfield(sim)
    elif op == "addcurrents":
        if nrhs < 4:
            _mexErrMsgIdAndTxt("simFDTD:InvalidInput", "time and indices required.")
        time = _data(prhs[2])
        ind = _data(prhs[3])
        _lib.simFDTD_addcurrents(sim, time.ctypes.data, ind.ctypes.data)
    elif op == "readysimulation":
        _lib.simFDTD_readysimulation(sim)
    elif op == "resetfieldall":
        _lib.simFDTD_resetfieldall(sim)
    elif op == "updatee_planewave":
        if nrhs < 3:
            _mexErrMsgIdAndTxt("simFDTD:InvalidInput", "time required.")
        time = _data(prhs[2])
        _lib.simFDTD_updatee_planewave(sim, time.ctypes.data)
    elif op == "updateh_planewave":
        if nrhs < 3:
            _mexErrMsgIdAndTxt("simFDTD:InvalidInput", "time required.")
        time = _data(prhs[2])
        _lib.simFDTD_updateh_planewave(sim, time.ctypes.data)
    elif op == "setconst":
        if nrhs < 4:
            _mexErrMsgIdAndTxt("simFDTD:InvalidInput", "freqRes and TimeDim required.")
        TopfreqRes = _data(prhs[2])
        TopTimeDim = _data(prhs[3])
        _lib.simFDTD_setconst(sim, TopfreqRes.ctypes.data, TopTimeDim.ctypes.data)
    elif op == "setsource":
        if nrhs < 3:
            _mexErrMsgIdAndTxt("simFDTD:InvalidInput", "SourceIn required.")
        SourceIn = _data(prhs[2])
        if nrhs > 3:
            print("Update Source Mode!")
            _lib.simFDTD_updatesource(sim, SourceIn.ctypes.data)
        else:
            _lib.simFDTD_setsource(sim, SourceIn.ctypes.data)
    elif op == "setantennalocations":
        if nrhs < 3:
            _mexErrMsgIdAndTxt("simFDTD:InvalidInput", "Locations required.")
        Locations = _data(prhs[2])
        _lib.simFDTD_setantennalocations(sim, Locations.ctypes.data)
    elif op == "createmu":
        if nrhs < 5:
            _mexErrMsgIdAndTxt("simFDTD:InvalidInput", "mu_x, mu_y, mu_z required.")
        mu_x = _data(prhs[2])
        mu_y = _data(prhs[3])
        mu_z = _data(prhs[4])
        _lib.simFDTD_createmu(sim, mu_x.ctypes.data, mu_y.ctypes.data, mu_z.ctypes.data)
    elif op == "init1d":
        if nrhs < 4:
            _mexErrMsgIdAndTxt("simFDTD:InvalidInput", "rate and PW_source required.")
        rate = _data(prhs[2])
        PW_source = _data(prhs[3])
        _lib.simFDTD_init1d(sim, rate.ctypes.data, PW_source.ctypes.data)
    elif op == "updatematerial":
        if nrhs < 4:
            _mexErrMsgIdAndTxt("simFDTD:InvalidInput", "flag, values required.")
        flag = _data(prhs[2])
        values = _data(prhs[3])
        _lib.simFDTD_updatematerial(sim, flag.ctypes.data, values.ctypes.data)
    elif op == "outputant":
        if nrhs < 3:
            _mexErrMsgIdAndTxt("simFDTD:InvalidInput", "OutAntSize required.")
        OutAntSize = int(
            _data(prhs[2]).view(np.uint32)[0]
        )  # *((unsigned int *)mxGetData(prhs[2]))
        antData = np.zeros((1, OutAntSize), dtype=np.single, order="F")
        _lib.simFDTD_outputant(sim, antData.ctypes.data)
        return antData
    elif op == "outputvis":
        if nrhs < 4:
            _mexErrMsgIdAndTxt("simFDTD:InvalidInput", "OutAntSize and flag required.")
        OutAntSize = int(
            _data(prhs[2]).view(np.uint32)[0]
        )  # *((unsigned int *)mxGetData(prhs[2]))
        flag = _data(prhs[3])
        Data = np.zeros((1, OutAntSize), dtype=np.single, order="F")
        _lib.simFDTD_outputvis(sim, Data.ctypes.data, flag.ctypes.data)
        return Data
    else:
        _mexErrMsgIdAndTxt("simFDTD:UnknownOperation", "Unknown operation: " + op)
