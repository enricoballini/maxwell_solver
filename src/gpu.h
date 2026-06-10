#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math_constants.h>
template <typename T>
__global__ void GPU_updateH_Full(T *__restrict__ Hx, T *__restrict__ Hy, T *__restrict__ Hz, const T *__restrict__ Ex, const T *__restrict__ Ey, const T *__restrict__ Ez,
                                 const T *__restrict__ Chxe, const T *__restrict__ Chye, const T *__restrict__ Chze, const size_t Imax, const size_t Jmax, const size_t Kmax,
                                 const size_t dim)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idz = ti / dim;
    size_t idy = (ti - idz * dim) / Imax;
    size_t idx = ti - idz * dim - idy * Imax;
    size_t idEz1 = ti + Imax;
    size_t idEy1 = ti + dim;
    size_t idEx2 = ti + dim;
    size_t idEz2 = ti + 1;
    size_t idEy3 = ti + 1;
    size_t idEx3 = ti + Imax;

    if ((idz < Kmax - 1) && (idx < Imax - 1) && (idy < Jmax - 1))
    {
        T Ez_ti = Ez[ti];
        T Ey_ti = Ey[ti];
        T Ex_ti = Ex[ti];

        Hx[ti] = Hx[ti] + Chxe[ti] * (-(Ez[idEz1] - Ez_ti) + (Ey[idEy1] - Ey_ti));
        Hy[ti] = Hy[ti] + Chye[ti] * (-(Ex[idEx2] - Ex_ti) + (Ez[idEz2] - Ez_ti));
        Hz[ti] = Hz[ti] + Chze[ti] * (-(Ey[idEy3] - Ey_ti) + (Ex[idEx3] - Ex_ti));
    }
}

template <typename T>
__global__ void copy1dinv(T *__restrict__ E_inv_GPU, const T *__restrict__ Ez_GPU, const size_t Imax, const size_t time)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti < Imax)
        E_inv_GPU[Imax - 1 - ti + time] = Ez_GPU[ti];
}

template <typename T>
__global__ void copy1d(T *__restrict__ E_inv_GPU, const T *__restrict__ Ez_GPU, const size_t Imax, const size_t time)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti < Imax)
        E_inv_GPU[ti + time] = Ez_GPU[ti];
}

template <typename T>
__global__ void AddSource_GPU_1D(T *__restrict__ Ez_GPU, const size_t i, const T source)
{

    Ez_GPU[i] = Ez_GPU[i] + source;
}

template <typename T>
__global__ void AddSource_GPU_3D_1DThread(T *__restrict__ E_GPU, const T *__restrict__ Cexe_GPU, const T *__restrict__ Cexh_GPU, const T delX, const size_t Imax, const size_t Jmax, const size_t Kmax,
                                          const size_t i0, const size_t i1, const size_t j0, const size_t j1,
                                          const size_t k0, const size_t k1, const T source)
{
    if ((i1 < i0) || (j1 < j0) || (k1 < k0))
        return;

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t ni = i1 - i0 + 1;
    const size_t nj = j1 - j0 + 1;
    const size_t nk = k1 - k0 + 1;
    const size_t total = ni * nj * nk;

    if (tid < total)
    {
        const size_t i = i0 + tid % ni;
        const size_t j = j0 + (tid / ni) % nj;
        const size_t k = k0 + tid / (ni * nj);

        if ((i < Imax) && (j < Jmax) && (k < Kmax))
        {
            const size_t index = i + j * Imax + k * Imax * Jmax;
            E_GPU[index] = Cexe_GPU[index] * E_GPU[index] - Cexh_GPU[index] * source * delX / delX / delX; ///*0.5*(1-cos(2*CUDART_PI_F*i/ni));
            /// E_GPU[index] = - source ;
        }
    }
}

template <typename T>
__global__ void GPU_updateHPml_1D(const T *__restrict__ E, T *__restrict__ H, T *__restrict__ psi_Hyx_1D_GPU, const T *__restrict__ b_Hx_1D,
                                  const T *__restrict__ c_Hx_1D, const T Che, const T dx, const size_t Imax, const size_t nxPML_1)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    bool cond1 = (ti < nxPML_1);
    bool cond2 = (ti < 2 * nxPML_1 - 1) && (ti >= nxPML_1);
    size_t index = cond1 ? ti : (cond2 ? ti + Imax - 2 * nxPML_1 : 0);
    if (cond1 || cond2)
    {
        psi_Hyx_1D_GPU[ti] = b_Hx_1D[ti] * psi_Hyx_1D_GPU[ti] + c_Hx_1D[ti] * (E[index + 1] - E[index]) / dx;
        H[index] = H[index] + Che * psi_Hyx_1D_GPU[ti] * dx;
    }
    // if (cond1)
    // {
    //     psi_Hyx_1D_GPU[ti] = b_Hx_1D[ti] * psi_Hyx_1D_GPU[ti] + c_Hx_1D[ti] * (E[ti + 1] - E[ti]) / dx;
    //     H[ti] = H[ti] + Che * psi_Hyx_1D_GPU[ti] * dx;
    // }
    // if (cond2)
    // {
    //     psi_Hyx_1D_GPU[ti] = b_Hx_1D[ti] * psi_Hyx_1D_GPU[ti] + c_Hx_1D[ti] * (E[ti + Imax - 2 * nxPML_1 + 1] - E[ti + Imax - 2 * nxPML_1]) / dx;
    //     H[ti + Imax - 2 * nxPML_1] = H[ti + Imax - 2 * nxPML_1] + Che * psi_Hyx_1D_GPU[ti] * dx;
    // }
}

template <typename T>
__global__ void GPU_updateEPml_1D(const T *__restrict__ H, T *__restrict__ E, T *__restrict__ psi_Ezx_1D, const T *__restrict__ b_Ex_1D,
                                  const T *__restrict__ c_Ex_1D, const T Ceh, const T dx, const size_t Imax, const size_t nxPML_1)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    bool cond1 = (ti > 0) && (ti < nxPML_1);
    bool cond2 = (ti < 2 * nxPML_1) && (ti >= nxPML_1);
    size_t index = cond1 ? ti : (cond2 ? ti + Imax - 2 * nxPML_1 : 0);
    if (cond1 || cond2)
    {
        psi_Ezx_1D[ti] = b_Ex_1D[ti] * psi_Ezx_1D[ti] + c_Ex_1D[ti] * (H[index] - H[index - 1]) / dx;
        E[index] = E[index] + Ceh * psi_Ezx_1D[ti] * dx;
    }
    // if (ti < nxPML_1)
    // {
    //     psi_Ezx_1D[ti] = b_Ex_1D[ti] * psi_Ezx_1D[ti] + c_Ex_1D[ti] * (H[ti] - H[ti - 1]) / dx;
    //     E[ti] = E[ti] + Ceh * psi_Ezx_1D[ti] * dx;
    // }
    // if ((ti < 2 * nxPML_1 ) && (ti >= nxPML_1))
    // {
    //     psi_Ezx_1D[ti] = b_Ex_1D[ti] * psi_Ezx_1D[ti] + c_Ex_1D[ti] * (H[ti + Imax - 2 * nxPML_1] - H[ti + Imax - 2 * nxPML_1 - 1]) / dx;
    //     E[ti + Imax - 2 * nxPML_1] = E[ti + Imax - 2 * nxPML_1] + Ceh * psi_Ezx_1D[ti] * dx;
    // }
}

template <typename T>
__global__ void GPU_updateH_1D(T *__restrict__ H, const T *__restrict__ E, const T den_h, const T Che, const size_t Imax)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti < Imax - 1)
        H[ti] = H[ti] + Che * (E[ti + 1] - E[ti]);
}

template <typename T>
__global__ void GPU_updateE_1D(T *__restrict__ E, const T *__restrict__ H, const T den_e,
                               const T Ceh, const size_t Imax)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti > 0 && ti < Imax)
        E[ti] = E[ti] + Ceh * (H[ti] - H[ti - 1]);
}

////x-direction plane wave
template <typename T>
__global__ void GPU_updateHx_PlaneWaveX_Z(T *__restrict__ Hx, const T *__restrict__ E, const T c,
                                          const T Chxe, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    size_t tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    size_t dim = Imax * Jmax;

    if ((ti <= i1) && (tk <= k1 - 1))
    {
        Hx[ti + (j0 - 1) * Imax + tk * dim] = Hx[ti + (j0 - 1) * Imax + tk * dim] + Chxe * E[time + ti] * c;
        Hx[ti + j1 * Imax + tk * dim] = Hx[ti + j1 * Imax + tk * dim] - Chxe * E[time + ti] * c;
    }
}

template <typename T>
__global__ void GPU_updateHy_PlaneWaveX_Z(T *__restrict__ Hy, const T *__restrict__ E, const T c,
                                          const T Chye, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{
    size_t tj = blockIdx.x * blockDim.x + threadIdx.x + j0;
    size_t tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    size_t dim = Imax * Jmax;

    if ((tj <= j1) && (tk <= k1 - 1))
    {
        Hy[i0 - 1 + tj * Imax + tk * dim] = Hy[i0 - 1 + tj * Imax + tk * dim] - Chye * E[time + i0] * c;
        Hy[i1 + tj * Imax + tk * dim] = Hy[i1 + tj * Imax + tk * dim] + Chye * E[time + i1] * c;
    }
}

template <typename T>
__global__ void GPU_updateEx_PlaneWaveX_Z(T *__restrict__ Ex, const T *__restrict__ H, const T c,
                                          const T *__restrict__ Cexh, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    size_t tj = blockIdx.y * blockDim.y + threadIdx.y + j0;
    size_t dim = Imax * Jmax;

    if ((ti <= i1 - 1) && (tj <= j1))
    {
        Ex[ti + tj * Imax + k0 * dim] = Ex[ti + tj * Imax + k0 * dim] + Cexh[ti + tj * Imax + (k0 - 1) * dim] * H[ti + time] * c;
        Ex[ti + tj * Imax + k1 * dim] = Ex[ti + tj * Imax + k1 * dim] - Cexh[ti + tj * Imax + (k1)*dim] * H[ti + time] * c;
    }
}

template <typename T>
__global__ void GPU_updateEz_PlaneWaveX_Z(T *__restrict__ Ez, const T *__restrict__ H, const T c,
                                          const T *__restrict__ Cezh, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{
    size_t tj = blockIdx.x * blockDim.x + threadIdx.x + j0;
    size_t tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    size_t dim = Imax * Jmax;

    if ((tj <= j1) && (tk <= k1 - 1))
    {
        Ez[i0 + tj * Imax + tk * dim] = Ez[i0 + tj * Imax + tk * dim] - Cezh[i0 - 1 + tj * Imax + tk * dim] * H[time + i0 - 1] * c;
        Ez[i1 + tj * Imax + tk * dim] = Ez[i1 + tj * Imax + tk * dim] + Cezh[i1 + tj * Imax + tk * dim] * H[time + i1] * c;
    }
}

////x-direction y-polarized
template <typename T>
__global__ void GPU_updateHx_PlaneWaveX_Y(T *__restrict__ Hx, const T *__restrict__ E, const T c,
                                          const T Chxe, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    size_t tj = blockIdx.y * blockDim.y + threadIdx.y + j0;
    size_t dim = Imax * Jmax;

    if ((ti <= i1) && (tj <= j1 - 1))
    {
        Hx[ti + tj * Imax + (k0 - 1) * dim] = Hx[ti + tj * Imax + (k0 - 1) * dim] - Chxe * E[ti + time] * c;
        Hx[ti + tj * Imax + (k1)*dim] = Hx[ti + tj * Imax + (k1)*dim] + Chxe * E[ti + time] * c;
    }
}

template <typename T>
__global__ void GPU_updateHz_PlaneWaveX_Y(T *__restrict__ Hz, const T *__restrict__ E, const T c,
                                          const T Chze, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{
    size_t tj = blockIdx.x * blockDim.x + threadIdx.x + j0;
    size_t tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    size_t dim = Imax * Jmax;

    if ((tj <= j1 - 1) && (tk <= k1))
    {
        Hz[i0 - 1 + tj * Imax + tk * dim] = Hz[i0 - 1 + tj * Imax + tk * dim] + Chze * E[i0 + time] * c;
        Hz[i1 + tj * Imax + tk * dim] = Hz[i1 + tj * Imax + tk * dim] - Chze * E[i1 + time] * c;
    }
}

template <typename T>
__global__ void GPU_updateEx_PlaneWaveX_Y(T *__restrict__ Ex, const T *__restrict__ H, const T c,
                                          const T *__restrict__ Cexh, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{

    // int ti = blockIdx.x * blockDim.x + threadIdx.x;
    // int dim = Imax * Jmax;
    // int idx = ti % (i1 - 1 - i0 + 1) + i0;
    // int idz = ti / (i1 - 1 - i0 + 1) + k0;
    // int id1 = idx + j0 * Imax + idz * dim;
    // int id2 = idx + j1 * Imax + idz * dim;
    // if ((idx <= i1 - 1) && (idz <= k1))
    // {
    //     Ex[id1] = Ex[id1] - Cexh[id1] * H[idx + time] * c;
    //     Ex[id2] = Ex[id2] + Cexh[id2] * H[idx + time] * c;
    // }
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    size_t tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    size_t dim = Imax * Jmax;

    if ((ti <= i1 - 1) && (tk <= k1))
    {
        Ex[ti + j0 * Imax + tk * dim] = Ex[ti + j0 * Imax + tk * dim] - Cexh[ti + j0 * Imax + tk * dim] * H[ti + time] * c;
        Ex[ti + j1 * Imax + tk * dim] = Ex[ti + j1 * Imax + tk * dim] + Cexh[ti + j0 * Imax + tk * dim] * H[ti + time] * c;
    }
}

template <typename T>
__global__ void GPU_updateEy_PlaneWaveX_Y(T *__restrict__ Ey, const T *__restrict__ H, const T c,
                                          const T *__restrict__ Ceyh, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{
    size_t tj = blockIdx.x * blockDim.x + threadIdx.x + j0;
    size_t tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    size_t dim = Imax * Jmax;

    if ((tj <= j1 - 1) && (tk <= k1))
    {
        Ey[i0 + tj * Imax + tk * dim] = Ey[i0 + tj * Imax + tk * dim] + Ceyh[i0 - 1 + tj * Imax + tk * dim] * H[i0 - 1 + time] * c;
        Ey[i1 + tj * Imax + tk * dim] = Ey[i1 + tj * Imax + tk * dim] - Ceyh[i1 + tj * Imax + tk * dim] * H[i1 + time] * c;
    }
}

////y-direction x-polarization
template <typename T>
__global__ void GPU_updateHy_PlaneWaveY_X(T *__restrict__ Hy, const T *__restrict__ E, const T c,
                                          const T Chye, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    size_t tj = blockIdx.y * blockDim.y + threadIdx.y + j0;
    size_t dim = Imax * Jmax;

    if ((ti <= i1 - 1) && (tj <= j1))
    {
        Hy[ti + tj * Imax + (k0 - 1) * dim] = Hy[ti + tj * Imax + (k0 - 1) * dim] + Chye * E[tj] * c;
        Hy[ti + tj * Imax + (k1)*dim] = Hy[ti + tj * Imax + (k1)*dim] - Chye * E[tj] * c;
    }
}

template <typename T>
__global__ void GPU_updateHz_PlaneWaveY_X(T *__restrict__ Hz, const T *__restrict__ E, const T c,
                                          const T Chze, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    size_t tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    size_t dim = Imax * Jmax;

    if ((ti <= i1 - 1) && (tk <= k1))
    {
        Hz[ti + (j0 - 1) * Imax + tk * dim] = Hz[ti + (j0 - 1) * Imax + tk * dim] - Chze * E[j0] * c;
        Hz[ti + (j1)*Imax + tk * dim] = Hz[ti + (j1)*Imax + tk * dim] + Chze * E[j1] * c;
    }
}

template <typename T>
__global__ void GPU_updateEx_PlaneWaveY_X(T *__restrict__ Ex, const T *__restrict__ H, const T c,
                                          const T *__restrict__ Cexh, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    size_t tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    size_t dim = Imax * Jmax;

    if ((ti <= i1 - 1) && (tk <= k1))
    {
        Ex[ti + j0 * Imax + tk * dim] = Ex[ti + j0 * Imax + tk * dim] - Cexh[ti + (j0 - 1) * Imax + tk * dim] * H[j0 - 1] * c;
        Ex[ti + j1 * Imax + tk * dim] = Ex[ti + j1 * Imax + tk * dim] + Cexh[ti + (j1)*Imax + tk * dim] * H[j1] * c;
    }
}

template <typename T>
__global__ void GPU_updateEy_PlaneWaveY_X(T *__restrict__ Ey, const T *__restrict__ H, const T c,
                                          const T *__restrict__ Ceyh, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1)
{
    size_t tj = blockIdx.x * blockDim.x + threadIdx.x + j0;
    size_t tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    size_t dim = Imax * Jmax;

    if ((tj <= j1 - 1) && (tk <= k1))
    {
        Ey[i0 + tj * Imax + tk * dim] = Ey[i0 + tj * Imax + tk * dim] + Ceyh[i0 - 1 + tj * Imax + tk * dim] * H[tj] * c;
        Ey[i1 + tj * Imax + tk * dim] = Ey[i1 + tj * Imax + tk * dim] - Ceyh[i1 + tj * Imax + tk * dim] * H[tj] * c;
    }
}

////%%%y-direction plane wave z-polarized
template <typename T>
__global__ void GPU_updateHx_PlaneWaveY_Z(T *__restrict__ Hx, const T *__restrict__ E, const T c,
                                          const T Chxe, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    size_t tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    size_t dim = Imax * Jmax;

    if ((ti <= i1) && (tk <= k1 - 1))
    {
        Hx[ti + (j0 - 1) * Imax + tk * dim] = Hx[ti + (j0 - 1) * Imax + tk * dim] + Chxe * E[j0] * c;
        Hx[ti + j1 * Imax + tk * dim] = Hx[ti + j1 * Imax + tk * dim] - Chxe * E[j1] * c;
    }
}

template <typename T>
__global__ void GPU_updateHy_PlaneWaveY_Z(T *__restrict__ Hy, const T *__restrict__ E, const T c,
                                          const T Chye, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1)
{
    size_t tj = blockIdx.x * blockDim.x + threadIdx.x + j0;
    size_t tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    size_t dim = Imax * Jmax;

    if ((tj <= j1) && (tk <= k1 - 1))
    {
        Hy[i0 - 1 + tj * Imax + tk * dim] = Hy[i0 - 1 + tj * Imax + tk * dim] - Chye * E[tj] * c;
        Hy[i1 + tj * Imax + tk * dim] = Hy[i1 + tj * Imax + tk * dim] + Chye * E[tj] * c;
    }
}

template <typename T>
__global__ void GPU_updateEy_PlaneWaveY_Z(T *__restrict__ Ey, const T *__restrict__ H, const T c,
                                          const T *__restrict__ Ceyh, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    size_t tj = blockIdx.y * blockDim.y + threadIdx.y + j0;
    size_t dim = Imax * Jmax;

    if ((ti <= i1) && (tj <= j1 - 1))
    {
        Ey[ti + tj * Imax + k0 * dim] = Ey[ti + tj * Imax + k0 * dim] - Ceyh[ti + tj * Imax + (k0 - 1) * dim] * H[tj] * c;
        Ey[ti + tj * Imax + k1 * dim] = Ey[ti + tj * Imax + k1 * dim] + Ceyh[ti + tj * Imax + (k1)*dim] * H[tj] * c;
    }
}

template <typename T>
__global__ void GPU_updateEz_PlaneWaveY_Z(T *__restrict__ Ez, const T *__restrict__ H, const T c,
                                          const T *__restrict__ Cezh, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    size_t tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    size_t dim = Imax * Jmax;

    if ((ti <= i1) && (tk <= k1 - 1))
    {
        Ez[ti + j0 * Imax + tk * dim] = Ez[ti + j0 * Imax + tk * dim] + Cezh[ti + (j0 - 1) * Imax + tk * dim] * H[j0 - 1] * c;
        Ez[ti + j1 * Imax + tk * dim] = Ez[ti + j1 * Imax + tk * dim] - Cezh[ti + j1 * Imax + tk * dim] * H[j1] * c;
    }
}

/////z-direction plane wave y-polarization
template <typename T>
__global__ void GPU_updateHx_PlaneWaveZ_Y(T *__restrict__ Hx, const T *__restrict__ E, const T c,
                                          const T Chxe, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dim = Imax * Jmax;
    size_t idx = ti % (i1 - i0 + 1) + i0;
    size_t idy = ti / (i1 - i0 + 1) + j0;
    size_t id1 = (k0 - 1) * dim + idx + idy * Imax;
    size_t id2 = k1 * dim + idx + idy * Imax;
    if ((idx <= i1) && (idy <= j1 - 1))
    {
        Hx[id1] = Hx[id1] - Chxe * E[k0 + time] * c;
        Hx[id2] = Hx[id2] + Chxe * E[k1 + time] * c;
    }
    // int ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    // int tj = blockIdx.y * blockDim.y + threadIdx.y + j0;
    // int dim = Imax * Jmax;
    //
    // if ((ti<=i1) && (tj<=j1-1))
    // {Hx[ti + tj * Imax + (k0-1) * dim] = Hx[ti + tj * Imax + (k0-1) * dim]- Chxe * E[k0+time]*c;
    //  Hx[ti + tj * Imax + (k1) * dim] = Hx[ti + tj * Imax + (k1) * dim]+Chxe * E[k1+time]*c;
    // }
}

template <typename T>
__global__ void GPU_updateHz_PlaneWaveZ_Y(T *__restrict__ Hz, const T *__restrict__ E, const T c,
                                          const T Chze, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{
    size_t dim = Imax * Jmax;
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;

    size_t idy = ti % (j1 - 1 - j0 + 1) + j0;
    size_t idz = ti / (j1 - 1 - j0 + 1) + k0;
    size_t id1 = idz * dim + i0 - 1 + idy * Imax;
    size_t id2 = idz * dim + i1 + idy * Imax;
    if ((idy <= j1 - 1) && (idz <= k1))
    {
        Hz[id1] = Hz[id1] + Chze * E[idz + time] * c;
        Hz[id2] = Hz[id2] - Chze * E[idz + time] * c;
    }

    //
    // int tj = blockIdx.x * blockDim.x + threadIdx.x + j0;
    // int tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    // int dim = Imax * Jmax;
    //
    // if ((tj<=j1-1) && (tk<=k1))
    // {Hz[i0-1 + tj * Imax + tk * dim] = Hz[i0-1 + tj * Imax + tk * dim]+Chze * E[tk+time]*c;
    //  Hz[i1 + tj * Imax + tk * dim] = Hz[i1 + tj * Imax + tk * dim]- Chze * E[tk+time]*c;
    // }
}

template <typename T>
__global__ void GPU_updateEy_PlaneWaveZ_Y(T *__restrict__ Ey, const T *__restrict__ H, const T c,
                                          const T *__restrict__ Ceyh, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    size_t tj = blockIdx.y * blockDim.y + threadIdx.y + j0;
    size_t dim = Imax * Jmax;

    if ((ti <= i1) && (tj <= j1 - 1))
    {
        Ey[ti + tj * Imax + k0 * dim] = Ey[ti + tj * Imax + k0 * dim] - Ceyh[ti + tj * Imax + (k0 - 1) * dim] * H[k0 - 1 + time] * c;
        Ey[ti + tj * Imax + k1 * dim] = Ey[ti + tj * Imax + k1 * dim] + Ceyh[ti + tj * Imax + (k1 - 1) * dim] * H[k1 + time] * c;
    }
}

template <typename T>
__global__ void GPU_updateEz_PlaneWaveZ_Y(T *__restrict__ Ez, const T *__restrict__ H, const T c,
                                          const T *__restrict__ Cezh, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    size_t tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    size_t dim = Imax * Jmax;

    if ((ti <= i1) && (tk <= k1 - 1))
    {
        Ez[ti + j0 * Imax + tk * dim] = Ez[ti + j0 * Imax + tk * dim] + Cezh[ti + (j0 - 1) * Imax + tk * dim] * H[tk + time] * c;
        Ez[ti + (j1)*Imax + tk * dim] = Ez[ti + (j1)*Imax + tk * dim] - Cezh[ti + (j1)*Imax + tk * dim] * H[tk + time] * c;
    }
}

/////z-direction plane wave
template <typename T>
__global__ void GPU_updateHz_PlaneWaveZ_X(T *__restrict__ Hz, const T *__restrict__ E, const T c,
                                          const T Chze, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    size_t tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    size_t dim = Imax * Jmax;

    if ((ti <= i1 - 1) && (tk <= k1))
    {
        Hz[ti + (j0 - 1) * Imax + tk * dim] = Hz[ti + (j0 - 1) * Imax + tk * dim] - Chze * E[tk + time] * c;
        Hz[ti + j1 * Imax + tk * dim] = Hz[ti + j1 * Imax + tk * dim] + Chze * E[tk + time] * c;
    }
}
template <typename T>
__global__ void GPU_updateHy_PlaneWaveZ_X(T *__restrict__ Hy, const T *__restrict__ E, const T c,
                                          const T Chye, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    size_t tj = blockIdx.y * blockDim.y + threadIdx.y + j0;
    size_t dim = Imax * Jmax;

    if ((ti <= i1 - 1) && (tj <= j1))
    {
        Hy[ti + tj * Imax + (k0 - 1) * dim] = Hy[ti + tj * Imax + (k0 - 1) * dim] + Chye * E[k0 + time] * c;
        Hy[ti + tj * Imax + k1 * dim] = Hy[ti + tj * Imax + k1 * dim] - Chye * E[k1 + time] * c;
    }
}

template <typename T>
__global__ void GPU_updateEx_PlaneWaveZ_X(T *__restrict__ Ex, const T *__restrict__ H, const T c,
                                          const T *__restrict__ Cexh, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + i0;
    size_t tj = blockIdx.y * blockDim.y + threadIdx.y + j0;
    size_t dim = Imax * Jmax;

    if ((ti <= i1 - 1) && (tj <= j1))
    {
        Ex[ti + tj * Imax + k0 * dim] = Ex[ti + tj * Imax + k0 * dim] + Cexh[ti + tj * Imax + (k0 - 1) * dim] * H[k0 - 1 + time] * c;
        Ex[ti + tj * Imax + k1 * dim] = Ex[ti + tj * Imax + k1 * dim] - Cexh[ti + tj * Imax + (k1 - 1) * dim] * H[k1 + time] * c;
    }
}

template <typename T>
__global__ void GPU_updateEz_PlaneWaveZ_X(T *__restrict__ Ez, const T *__restrict__ H, const T c,
                                          const T *__restrict__ Cezh, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t i0, const size_t i1,
                                          const size_t j0, const size_t j1, const size_t k0, const size_t k1, const size_t time)
{
    size_t tj = blockIdx.x * blockDim.x + threadIdx.x + j0;
    size_t tk = blockIdx.y * blockDim.y + threadIdx.y + k0;
    size_t dim = Imax * Jmax;

    if ((tj <= j1) && (tk <= k1 - 1))
    {
        Ez[i0 + tj * Imax + tk * dim] = Ez[i0 + tj * Imax + tk * dim] - Cezh[i0 - 1 + tj * Imax + tk * dim] * H[tk + time] * c;
        Ez[i1 + tj * Imax + tk * dim] = Ez[i1 + tj * Imax + tk * dim] + Cezh[i1 + tj * Imax + tk * dim] * H[tk + time] * c;
    }
}

template <typename T>
__global__ void GPU_updateH(T *__restrict__ Hx, T *__restrict__ Hy, T *__restrict__ Hz, const T *__restrict__ Ex, const T *__restrict__ Ey, const T *__restrict__ Ez,
                            const T Chxh, const T Chxe, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t dim)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;

    size_t idz = ti / dim;
    size_t rem = ti - idz * dim;
    size_t idy = rem / Imax;
    size_t idx = rem - idy * Imax;
    size_t idEz1 = ti + Imax;
    size_t idEy1 = ti + dim;
    size_t idEx2 = ti + dim;
    size_t idEz2 = ti + 1;
    size_t idEy3 = ti + 1;
    size_t idEx3 = ti + Imax;

    if ((idz < Kmax - 1) && (idx < Imax - 1) && (idy < Jmax - 1))
    {
        T Ez_ti = Ez[ti];
        T Ey_ti = Ey[ti];
        T Ex_ti = Ex[ti];

        Hx[ti] = Chxh * Hx[ti] + Chxe * (-(Ez[idEz1] - Ez_ti) + (Ey[idEy1] - Ey_ti));
        Hy[ti] = Chxh * Hy[ti] + Chxe * (-(Ex[idEx2] - Ex_ti) + (Ez[idEz2] - Ez_ti));
        Hz[ti] = Chxh * Hz[ti] + Chxe * (-(Ey[idEy3] - Ey_ti) + (Ex[idEx3] - Ex_ti));
    }
}

template <typename T>
__global__ void GPU_updateHPmlX(T *__restrict__ Hy, T *__restrict__ Hz, const T *__restrict__ Ey, const T *__restrict__ Ez, T *__restrict__ psi_Hyx_GPU,
                                T *__restrict__ psi_Hzx_GPU, const T *__restrict__ b_H_1,
                                const T *__restrict__ c_H_1, const size_t nxPML, const T Che, const T dx, const size_t Imax,
                                const size_t Jmax, const size_t Kmax, const size_t dim)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dim_s = 2 * nxPML * Jmax;
    size_t idz = ti / dim_s;
    size_t idy = (ti - idz * dim_s) / (2 * nxPML);
    size_t idx = ti - idz * dim_s - idy * 2 * nxPML;
    bool cond1 = (idz < Kmax - 1) && (idx < nxPML) && (idy < Jmax - 1);
    bool cond2 = (idz < Kmax - 1) && (idy < Jmax - 1) && (idx >= nxPML) && (idx < 2 * nxPML - 1);
    size_t id = cond1 ? idx + idy * Imax + idz * dim : cond2 ? Imax - 2 * nxPML + idx + idy * Imax + idz * dim
                                                             : 0;
    if (cond1 || cond2)
    {
        psi_Hyx_GPU[ti] = b_H_1[idx] * psi_Hyx_GPU[ti] + c_H_1[idx] * (Ez[id + 1] - Ez[id]) / dx;
        Hy[id] = Hy[id] + Che * psi_Hyx_GPU[ti] * dx;
        psi_Hzx_GPU[ti] = b_H_1[idx] * psi_Hzx_GPU[ti] + c_H_1[idx] * (Ey[id + 1] - Ey[id]) / dx;
        Hz[id] = Hz[id] - Che * psi_Hzx_GPU[ti] * dx;
    }
    // if ((idz < Kmax - 1) && (idx < nxPML - 1) && (idy < Jmax - 1))
    // {
    //     int id = idx + idy * Imax + idz * dim;
    //     psi_Hyx_GPU[ti] = b_H_1[idx] * psi_Hyx_GPU[ti] + c_H_1[idx] * (Ez[id + 1] - Ez[id]) / dx;
    //     Hy[id] = Hy[id] + Che * psi_Hyx_GPU[ti] * dx;
    //     psi_Hzx_GPU[ti] = b_H_1[idx] * psi_Hzx_GPU[ti] + c_H_1[idx] * (Ey[id + 1] - Ey[id]) / dx;
    //     Hz[id] = Hz[id] - Che * psi_Hzx_GPU[ti] * dx;
    // }
    // else if ((idz < Kmax - 1) && (idy < Jmax - 1) && (idx >= nxPML) && (idx < 2 * nxPML - 1))
    // {
    //     int id1 = Imax - 2 * nxPML + idx + idy * Imax + idz * dim;
    //     psi_Hyx_GPU[ti] = b_H_1[idx] * psi_Hyx_GPU[ti] + c_H_1[idx] * (Ez[id1 + 1] - Ez[id1]) / dx;
    //     Hy[id1] = Hy[id1] + Che * psi_Hyx_GPU[ti] * dx;
    //     psi_Hzx_GPU[ti] = b_H_1[idx] * psi_Hzx_GPU[ti] + c_H_1[idx] * (Ey[id1 + 1] - Ey[id1]) / dx;
    //     Hz[id1] = Hz[id1] - Che * psi_Hzx_GPU[ti] * dx;
    // }
}

template <typename T>
__global__ void GPU_updateHPmlY(T *__restrict__ Hx, T *__restrict__ Hz, const T *__restrict__ Ex, const T *__restrict__ Ez, T *__restrict__ psi_Hxy_1,
                                T *__restrict__ psi_Hzy_1, const T *__restrict__ b_H_1, const T *__restrict__ c_H_1, const size_t nyPML, const T Che, const T dy, const size_t Imax,
                                const size_t Jmax, const size_t Kmax, const size_t dim)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dim_s = Imax * 2 * nyPML;
    size_t idz = ti / dim_s;
    size_t idy = (ti - idz * dim_s) / Imax;
    size_t idx = ti - idz * dim_s - idy * Imax;
    bool cond1 = (idz < Kmax - 1) && (idx < Imax - 1) && (idy < nyPML);
    bool cond2 = (idz < Kmax - 1) && (idx < Imax - 1) && (idy >= nyPML) && (idy < 2 * nyPML - 1);
    size_t id = cond1 ? idx + idy * Imax + idz * dim : cond2 ? idx + (idy + Jmax - 2 * nyPML) * Imax + idz * dim
                                                             : 0;
    if (cond1 || cond2)
    {
        psi_Hxy_1[ti] = b_H_1[idy] * psi_Hxy_1[ti] + c_H_1[idy] * (Ez[id + Imax] - Ez[id]) / dy;
        Hx[id] = Hx[id] - Che * psi_Hxy_1[ti] * dy;

        psi_Hzy_1[ti] = b_H_1[idy] * psi_Hzy_1[ti] + c_H_1[idy] * (Ex[id + Imax] - Ex[id]) / dy;
        Hz[id] = Hz[id] + Che * psi_Hzy_1[ti] * dy;
    }
    // if ((idz < Kmax - 1) && (idx < Imax - 1) && (idy < nyPML - 1))
    // {
    //     int id = idx + idy * Imax + idz * dim;
    //     psi_Hxy_1[ti] = b_H_1[idy] * psi_Hxy_1[ti] + c_H_1[idy] * (Ez[id + Imax] - Ez[id]) / dy;
    //     Hx[id] = Hx[id] - Che * psi_Hxy_1[ti] * dy;

    //     psi_Hzy_1[ti] = b_H_1[idy] * psi_Hzy_1[ti] + c_H_1[idy] * (Ex[id + Imax] - Ex[id]) / dy;
    //     Hz[id] = Hz[id] + Che * psi_Hzy_1[ti] * dy;
    // }
    // else if ((idz < Kmax - 1) && (idx < Imax - 1) && (idy >= nyPML) && (idy < 2 * nyPML - 1))
    // {
    //     int id1 = idx + (idy + Jmax - 2 * nyPML) * Imax + idz * dim;

    //     psi_Hxy_1[ti] = b_H_1[idy] * psi_Hxy_1[ti] + c_H_1[idy] * (Ez[id1 + Imax] - Ez[id1]) / dy;
    //     Hx[id1] = Hx[id1] - Che * psi_Hxy_1[ti] * dy;

    //     psi_Hzy_1[ti] = b_H_1[idy] * psi_Hzy_1[ti] + c_H_1[idy] * (Ex[id1 + Imax] - Ex[id1]) / dy;
    //     Hz[id1] = Hz[id1] + Che * psi_Hzy_1[ti] * dy;
    // }
}

template <typename T>
__global__ void GPU_updateHPmlZ(T *__restrict__ Hx, T *__restrict__ Hy, const T *__restrict__ Ex, const T *__restrict__ Ey, T *__restrict__ psi_Hxz_GPU, T *__restrict__ psi_Hyz_1_GPU,
                                const T *__restrict__ b_H_1,
                                const T *__restrict__ c_H_1, const size_t nzPML, const T Che, const T dz, const size_t Imax,
                                const size_t Jmax, const size_t Kmax, const size_t dim)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dim_s = Imax * Jmax;
    size_t idz = ti / dim_s;
    size_t idy = (ti - idz * dim_s) / Imax;
    size_t idx = ti - idz * dim_s - idy * Imax;
    bool cond1 = (idz < nzPML) && (idx < Imax - 1) && (idy < Jmax - 1);
    bool cond2 = (idz < 2 * nzPML - 1) && (idz > nzPML - 1) && (idx < Imax - 1) && (idy < Jmax - 1);
    size_t id = cond1 ? idx + idy * Imax + idz * dim : cond2 ? idx + idy * Imax + (idz + Kmax - 2 * nzPML) * dim
                                                             : 0;
    if (cond1 || cond2)
    {
        psi_Hxz_GPU[ti] = b_H_1[idz] * psi_Hxz_GPU[ti] + c_H_1[idz] * (Ey[id + dim] - Ey[id]) / dz;
        Hx[id] = Hx[id] + Che * psi_Hxz_GPU[ti] * dz;
        psi_Hyz_1_GPU[ti] = b_H_1[idz] * psi_Hyz_1_GPU[ti] + c_H_1[idz] * (Ex[id + dim] - Ex[id]) / dz;
        Hy[id] = Hy[id] - Che * psi_Hyz_1_GPU[ti] * dz;
    }
    // if ((idz < nzPML - 1) && (idx < Imax - 1) && (idy < Jmax - 1))
    // {
    //     int id = idx + idy * Imax + idz * dim;
    //     psi_Hxz_GPU[ti] = b_H_1[idz] * psi_Hxz_GPU[ti] + c_H_1[idz] * (Ey[id + dim] - Ey[id]) / dz;
    //     Hx[id] = Hx[id] + Che * psi_Hxz_GPU[ti] * dz;
    //     psi_Hyz_1_GPU[ti] = b_H_1[idz] * psi_Hyz_1_GPU[ti] + c_H_1[idz] * (Ex[id + dim] - Ex[id]) / dz;
    //     Hy[id] = Hy[id] - Che * psi_Hyz_1_GPU[ti] * dz;
    // }
    // else if ((idz < 2 * nzPML - 1) && (idz > nzPML - 1) && (idx < Imax - 1) && (idy < Jmax - 1))
    // {
    //     int id1 = idx + idy * Imax + (idz + Kmax - 2 * nzPML) * dim;
    //     psi_Hxz_GPU[ti] = b_H_1[idz] * psi_Hxz_GPU[ti] + c_H_1[idz] * (Ey[id1 + dim] - Ey[id1]) / dz;
    //     Hx[id1] = Hx[id1] + Che * psi_Hxz_GPU[ti] * dz;
    //     psi_Hyz_1_GPU[ti] = b_H_1[idz] * psi_Hyz_1_GPU[ti] + c_H_1[idz] * (Ex[id1 + dim] - Ex[id1]) / dz;
    //     Hy[id1] = Hy[id1] - Che * psi_Hyz_1_GPU[ti] * dz;
    // }
}

template <typename T>
__global__ void GPU_updateE(T *__restrict__ Ex, T *__restrict__ Ey, T *__restrict__ Ez, const T *__restrict__ Cexe, const T *__restrict__ Cexh, const T *__restrict__ Ceye, const T *__restrict__ Ceyh,
                            const T *__restrict__ Ceze, const T *__restrict__ Cezh, const T *__restrict__ Hx, const T *__restrict__ Hy,
                            const T *__restrict__ Hz, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t dim)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idz = ti / dim;

    size_t rem = ti - idz * dim;
    size_t idy = rem / Imax;
    size_t idx = rem - idy * Imax;
    size_t idHy1 = ti - dim;
    size_t idHz1 = ti - Imax;
    size_t idHx2 = ti - dim;
    size_t idHz2 = ti - 1;
    size_t idHx3 = ti - Imax;
    size_t idHy3 = ti - 1;
    T Hx_ti = Hx[ti];
    T Hy_ti = Hy[ti];
    T Hz_ti = Hz[ti];
    if ((idz < Kmax) && (idx < Imax) && (idy < Jmax) && (idx >= 0) && (idy >= 0) && (idz >= 0))
    {
        if (idy != 0 && idz != 0)
        {
            Ex[ti] = Cexe[ti] * Ex[ti] + Cexh[ti] * (+(Hz_ti - Hz[idHz1]) - (Hy_ti - Hy[idHy1]));
        }
        if (idx != 0 && idz != 0)
        {
            Ey[ti] = Ceye[ti] * Ey[ti] + Ceyh[ti] * (+(Hx_ti - Hx[idHx2]) - (Hz_ti - Hz[idHz2]));
        }
        if (idx != 0 && idy != 0)
        {
            Ez[ti] = Ceze[ti] * Ez[ti] + Cezh[ti] * (+(Hy_ti - Hy[idHy3]) - (Hx_ti - Hx[idHx3]));
        }
    }
}

template <typename T>
__global__ void GPU_updateEPmlX(const T *__restrict__ Hy, const T *__restrict__ Hz, T *__restrict__ Ey, T *__restrict__ Ez, T *__restrict__ psi_Eyx_1, T *__restrict__ psi_Ezx_1,
                                const T *__restrict__ b_Ex_1, const T *__restrict__ c_Ex_1, const T *__restrict__ Ceyh, const T *__restrict__ Cezh,
                                const T dx, const size_t nxPML, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t dim)
{

    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dim_s = 2 * nxPML * Jmax;
    size_t idz = ti / dim_s;
    size_t idy = (ti - idz * dim_s) / (2 * nxPML);
    size_t idx = ti - idz * dim_s - idy * 2 * nxPML;
    bool cond1 = (idz < Kmax - 1) && (idz >= 1) && (idx > 0) && (idx < nxPML) && (idy > 0) && (idy < Jmax - 1);
    bool cond2 = (idz < Kmax - 1) && (idz >= 1) && (idx < 2 * nxPML) && (idx >= nxPML) && (idy < Jmax - 1) && (idy > 0);
    size_t id = cond1 ? idx + idy * Imax + idz * dim : cond2 ? idx + Imax - 2 * nxPML + idy * Imax + idz * dim
                                                             : 1;
    if (cond1 || cond2)
    {
        psi_Eyx_1[ti] = b_Ex_1[idx] * psi_Eyx_1[ti] + c_Ex_1[idx] * (Hz[id] - Hz[id - 1]) / dx;
        Ey[id] = Ey[id] - Ceyh[id] * psi_Eyx_1[ti] * dx;

        psi_Ezx_1[ti] = b_Ex_1[idx] * psi_Ezx_1[ti] + c_Ex_1[idx] * (Hy[id] - Hy[id - 1]) / dx;
        Ez[id] = Ez[id] + Cezh[id] * psi_Ezx_1[ti] * dx;
    }
}

template <typename T>
__global__ void GPU_updateEPmlY(const T *__restrict__ Hx, const T *__restrict__ Hz, T *__restrict__ Ex, T *__restrict__ Ez, T *__restrict__ psi_Exy_1,
                                T *__restrict__ psi_Ezy_1, const T *__restrict__ b_Ey_1, const T *__restrict__ c_Ey_1, const T *__restrict__ Cexh, const T *__restrict__ Cezh,
                                const T dy, const size_t nyPML, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t dim)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dim_s = Imax * 2 * nyPML;
    size_t idz = ti / dim_s;
    size_t idy = (ti - idz * dim_s) / Imax;
    size_t idx = ti - idz * dim_s - idy * Imax;
    bool cond1 = (idz < Kmax - 1) && (idz >= 1) && (idx < Imax - 1) && (idx > 0) && (idy > 0) && (idy < nyPML);
    bool cond2 = (idz < Kmax - 1) && (idz >= 1) && (idx < Imax - 1) && (idx > 0) && (idy >= nyPML) && (idy < 2 * nyPML);
    size_t id = cond1 ? idx + idy * Imax + idz * dim : cond2 ? idx + (idy + Jmax - 2 * nyPML) * Imax + idz * dim
                                                             : Imax;
    if (cond1 || cond2)
    {
        psi_Exy_1[ti] = b_Ey_1[idy] * psi_Exy_1[ti] + c_Ey_1[idy] * (Hz[id] - Hz[id - Imax]) / dy;
        Ex[id] = Ex[id] + Cexh[id] * psi_Exy_1[ti] * dy;
        psi_Ezy_1[ti] = b_Ey_1[idy] * psi_Ezy_1[ti] + c_Ey_1[idy] * (Hx[id] - Hx[id - Imax]) / dy;
        Ez[id] = Ez[id] - Cezh[id] * psi_Ezy_1[ti] * dy;
    }
    // if ((idz < Kmax - 1) && (idz >= 1) && (idx < Imax - 1) && (idx > 0) && (idy > 0) && (idy <= nyPML - 1))
    // {
    //     int id = idx + idy * Imax + idz * dim;
    //     psi_Exy_1[ti] = b_Ey_1[idy] * psi_Exy_1[ti] + c_Ey_1[idy] * (Hz[id] - Hz[id - Imax]) / dy;
    //     Ex[id] = Ex[id] + Cexh[id] * psi_Exy_1[ti] * dy;
    //     psi_Ezy_1[ti] = b_Ey_1[idy] * psi_Ezy_1[ti] + c_Ey_1[idy] * (Hx[id] - Hx[id - Imax]) / dy;
    //     Ez[id] = Ez[id] - Cezh[id] * psi_Ezy_1[ti] * dy;
    // }
    // else if ((idz < Kmax - 1) && (idz >= 1) && (idx < Imax - 1) && (idx > 0) && (idy > nyPML - 1) && (idy < 2 * nyPML - 1))
    // {
    //     int id = idx + (idy + Jmax - 2 * nyPML) * Imax + idz * dim;
    //     psi_Exy_1[ti] = b_Ey_1[idy] * psi_Exy_1[ti] + c_Ey_1[idy] * (Hz[id] - Hz[id - Imax]) / dy;
    //     Ex[id] = Ex[id] + Cexh[id] * psi_Exy_1[ti] * dy;
    //     psi_Ezy_1[ti] = b_Ey_1[idy] * psi_Ezy_1[ti] + c_Ey_1[idy] * (Hx[id] - Hx[id - Imax]) / dy;
    //     Ez[id] = Ez[id] - Cezh[id] * psi_Ezy_1[ti] * dy;
    // }
}

template <typename T>
__global__ void GPU_updateEPmlZ(const T *__restrict__ Hx, const T *__restrict__ Hy, T *__restrict__ Ex, T *__restrict__ Ey, T *__restrict__ psi_Exz_1,
                                T *__restrict__ psi_Eyz_1, const T *__restrict__ b_Ez_1, const T *__restrict__ c_Ez_1, const T *__restrict__ Cexh, const T *__restrict__ Ceyh,
                                const T dz, const size_t nzPML, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t dim)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dim_s = dim;
    size_t idz = ti / dim_s;
    size_t idy = (ti - idz * dim_s) / Imax;
    size_t idx = ti - idz * dim_s - idy * Imax;
    bool cond1 = (idz < nzPML) && (idz >= 1) && (idx < Imax - 1) && (idx > 0) && (idy < Jmax - 1) && (idy > 0);
    bool cond2 = (idz < 2 * nzPML) && (idz >= nzPML) && (idx < Imax - 1) && (idx > 0) && (idy < Jmax - 1) && (idy > 0);
    size_t id = cond1 ? idx + idy * Imax + idz * dim : cond2 ? idx + idy * Imax + (idz + Kmax - 2 * nzPML) * dim
                                                             : dim;
    if (cond1 || cond2)
    {
        psi_Exz_1[ti] = b_Ez_1[idz] * psi_Exz_1[ti] + c_Ez_1[idz] * (Hy[id] - Hy[id - dim]) / dz;
        Ex[id] = Ex[id] - Cexh[id] * psi_Exz_1[ti] * dz;

        psi_Eyz_1[ti] = b_Ez_1[idz] * psi_Eyz_1[ti] + c_Ez_1[idz] * (Hx[id] - Hx[id - dim]) / dz;
        Ey[id] = Ey[id] + Ceyh[id] * psi_Eyz_1[ti] * dz;
    }
    // if ((idz <= nzPML - 1) && (idz >= 1) && (idx < Imax - 1) && (idx > 0) && (idy < Jmax - 1) && (idy > 0))
    // {
    //     int id = idx + idy * Imax + idz * dim;
    //     psi_Exz_1[ti] = b_Ez_1[idz] * psi_Exz_1[ti] + c_Ez_1[idz] * (Hy[id] - Hy[id - dim]) / dz;
    //     Ex[id] = Ex[id] - Cexh[id] * psi_Exz_1[ti] * dz;

    //     psi_Eyz_1[ti] = b_Ez_1[idz] * psi_Eyz_1[ti] + c_Ez_1[idz] * (Hx[id] - Hx[id - dim]) / dz;
    //     Ey[id] = Ey[id] + Ceyh[id] * psi_Eyz_1[ti] * dz;
    // }
    // else if ((idz < 2 * nzPML - 1) && (idz > nzPML - 1) && (idx < Imax - 1) && (idx > 0) && (idy < Jmax - 1) && (idy > 0))
    // {
    //     int id = idx + idy * Imax + (idz + Kmax - 2 * nzPML) * dim;
    //     psi_Exz_1[ti] = b_Ez_1[idz] * psi_Exz_1[ti] + c_Ez_1[idz] * (Hy[id] - Hy[id - dim]) / dz;
    //     Ex[id] = Ex[id] - Cexh[id] * psi_Exz_1[ti] * dz;
    //     psi_Eyz_1[ti] = b_Ez_1[idz] * psi_Eyz_1[ti] + c_Ez_1[idz] * (Hx[id] - Hx[id - dim]) / dz;
    //     Ey[id] = Ey[id] + Ceyh[id] * psi_Eyz_1[ti] * dz;
    // }
}
