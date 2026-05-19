#include "cuda_runtime.h"
#include "device_launch_parameters.h"
template <typename T>
__device__ __forceinline__ T fma_custom(T a, T b, T c);

// Specialization for float
template <>
__device__ __forceinline__ float fma_custom<float>(float a, float b, float c)
{
    return __fmaf_rn(a, b, c); // fused multiply-add, round to nearest
}

// Specialization for double
template <>
__device__ __forceinline__ double fma_custom<double>(double a, double b, double c)
{
    return __fma_rn(a, b, c); // fused multiply-add, round to nearest
}

template <unsigned int blockSize, typename T>
__device__ void warpReduce(volatile T *sdata, const size_t tid)
{
    if (blockSize >= 64)
        sdata[tid] += sdata[tid + 32];
    __syncthreads();
    if (blockSize >= 32)
        sdata[tid] += sdata[tid + 16];
    __syncthreads();
    if (blockSize >= 16)
        sdata[tid] += sdata[tid + 8];
    __syncthreads();
    if (blockSize >= 8)
        sdata[tid] += sdata[tid + 4];
    __syncthreads();
    if (blockSize >= 4)
        sdata[tid] += sdata[tid + 2];
    __syncthreads();
    if (blockSize >= 2)
        sdata[tid] += sdata[tid + 1];
    __syncthreads();
}
template <unsigned int blockSize, typename T>
__global__ void sum_reductions(const T *__restrict__ input, T *__restrict__ output, const size_t size)
{
    __shared__ T sdata[blockSize];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    size_t gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;
    while (i < size)
    {
        if (i + blockDim.x < size)
            sdata[tid] += input[i] + input[i + blockDim.x];
        else
            sdata[tid] += input[i];
        i += gridSize;
    }
    __syncthreads();
    if (tid < 512)
        sdata[tid] += sdata[tid + 512];
    __syncthreads();
    if (tid < 256)
        sdata[tid] += sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
        sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (tid < 64)
        sdata[tid] += sdata[tid + 64];
    __syncthreads();
    if (tid < 32)
        warpReduce<blockSize>(sdata, tid);
    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

template <unsigned int blockSize, typename T>
__global__ void sum_reduce(const T *__restrict__ input, T *__restrict__ output, const size_t index, const size_t size)
{
    __shared__ T sdata[blockSize];
    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    size_t gridSize = blockSize * 2 * gridDim.x;
    sdata[tid] = 0;
    while (i < size)
    {
        if (i + blockDim.x < size)
            sdata[tid] += input[i] + input[i + blockDim.x];
        else
            sdata[tid] += input[i];
        i += gridSize;
    }
    __syncthreads();
    if (tid < 512)
        sdata[tid] += sdata[tid + 512];
    __syncthreads();
    if (tid < 256)
        sdata[tid] += sdata[tid + 256];
    __syncthreads();
    if (tid < 128)
        sdata[tid] += sdata[tid + 128];
    __syncthreads();
    if (tid < 64)
        sdata[tid] += sdata[tid + 64];
    __syncthreads();
    if (tid < 32)
        warpReduce<blockSize>(sdata, tid);
    if (tid == 0)
        output[index] = sdata[0];
}

template <typename T1, typename T2>
__global__ void computeGradientGPU_Mul(const T1 *__restrict__ E1_ant, const T1 *__restrict__ E1_ant_AD, T2 *__restrict__ output, const size_t ti, const size_t size, const size_t max_TimeSteps)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t index = (max_TimeSteps - tid - 1) * size + ti;
    if (tid < max_TimeSteps - 1)
    {
        output[tid] = E1_ant[index] * (E1_ant_AD[tid * size + ti] + E1_ant_AD[(tid + 1) * size + ti]) / 2;
    }
    else
    {
        output[tid] = 0;
    }
}

template <unsigned int blockSize, typename T>
__global__ void PortCurrentGPU(const T *__restrict__ Hx_GPU, const T *__restrict__ Hy_GPU, const size_t ind, const size_t Imax, const size_t Jmax, const T delX,
                               const size_t depth_coax0, const size_t depth_coax1, const size_t width_coax0, const size_t width_coax1, T *__restrict__ current)
{
    __shared__ T sdata[blockSize];
    size_t tid = threadIdx.x;
    if (tid < width_coax0 + width_coax1 + 1)
    {
        sdata[tid] = (Hy_GPU[ind + depth_coax1 + (tid - width_coax0) * Imax] - Hy_GPU[ind - depth_coax0 - 1 + (tid - width_coax0) * Imax]) * delX;
    }
    else
    {
        sdata[tid] = 0.0;
    }
    __syncthreads();
    if (tid < depth_coax0 + depth_coax1 + 1)
    {
        sdata[tid] = sdata[tid] + (Hx_GPU[ind + (tid - depth_coax0) * Imax - (width_coax0 + 1) * Imax] - Hx_GPU[ind + (tid - depth_coax0) * Imax + width_coax1 * Imax]) * delX;
    }
    __syncthreads();
    if (tid < 512)
    {
        sdata[tid] += sdata[tid + 512];
    }
    __syncthreads();
    if (tid < 256)
    {
        sdata[tid] += sdata[tid + 256];
    }
    __syncthreads();
    if (tid < 128)
    {
        sdata[tid] += sdata[tid + 128];
    }
    __syncthreads();
    if (tid < 64)
    {
        sdata[tid] += sdata[tid + 64];
    }
    __syncthreads();
    if (tid < 32)
        warpReduce<blockSize>(sdata, tid);
    if (tid == 0)
        current[0] = sdata[0];
}

template <typename T>
__global__ void PortEGPU(T *__restrict__ PortSignal_GPU, T *__restrict__ Ex_GPU, T *__restrict__ Ey_GPU, const T *__restrict__ current,
                         const T *__restrict__ source, const size_t time, const size_t ind, const size_t Imax, const size_t Jmax, const T delX,
                         const T Z_c, const int depth_coax0, const int depth_coax1, const int width_coax0, const int width_coax1, const int mode)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    T tep = (mode == 1) ? (source[time] - current[0] * Z_c) / delX : -current[0] * Z_c / delX;
    if (tid < width_coax0 + width_coax1 + 1)
    {
        Ex_GPU[ind + (tid - width_coax0) * Imax + depth_coax1] = tep;
        Ex_GPU[ind - 1 + (tid - width_coax0) * Imax - depth_coax0] = -tep;
    }
    if (tid < depth_coax0 + depth_coax1 + 1)
    {
        Ey_GPU[ind + width_coax1 * Imax + (tid - depth_coax0)] = tep;
        Ey_GPU[ind - (1 + width_coax0) * Imax + (tid - depth_coax0)] = -tep;
    }
    if (tid == 0)
        PortSignal_GPU[time] = -Z_c * current[0] + tep * delX;
}

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

        Hx[ti] = fma_custom(Chxe[ti], (-(Ez[idEz1] - Ez_ti) + (Ey[idEy1] - Ey_ti)), Hx[ti]);
        Hy[ti] = fma_custom(Chye[ti], (-(Ex[idEx2] - Ex_ti) + (Ez[idEz2] - Ez_ti)), Hy[ti]);
        Hz[ti] = fma_custom(Chze[ti], (-(Ey[idEy3] - Ey_ti) + (Ex[idEx3] - Ex_ti)), Hz[ti]);

        // Hx[ti] = Hx[ti] + Chxe[ti] * (-(Ez[idEz1] - Ez_ti) + (Ey[idEy1] - Ey_ti));
        // Hy[ti] = Hy[ti] + Chye[ti] * (-(Ex[idEx2] - Ex_ti) + (Ez[idEz2] - Ez_ti));
        // Hz[ti] = Hz[ti] + Chze[ti] * (-(Ey[idEy3] - Ey_ti) + (Ex[idEx3] - Ex_ti));
    }
}

template <typename T>
__global__ void coaxUpdateGPU(T *__restrict__ PortSignal_GPU, T *__restrict__ Ex_GPU, T *__restrict__ Ey_GPU, const T *__restrict__ Hx_GPU,
                              const T *__restrict__ Hy_GPU, const T *__restrict__ source, const size_t time,
                              const size_t ind, const size_t Imax, const size_t Jmax, const T delX,
                              const T Z_c, const int depth_coax0, const int depth_coax1, const int width_coax0, const int width_coax1, const int mode)
{
    // current[0] = Hx_GPU[ind - (width_coax0 + 1) * Imax] - Hx_GPU[ind + width_coax1 * Imax];
    // for (int i = -width_coax0; i <= width_coax1; i++)
    // {
    //     current[0] = current[0] + Hy_GPU[ind + i * Imax] - Hy_GPU[ind - 1 + i * Imax];
    // }
    // current[0] = current[0] * delX;

    T current_temp = 0.0;
    for (size_t i = -width_coax0; i <= width_coax1; i++)
    {
        current_temp = current_temp + (Hy_GPU[ind + depth_coax1 + i * Imax] - Hy_GPU[ind - depth_coax0 - 1 + i * Imax]) * delX;
    }
    for (size_t i = -depth_coax0; i <= depth_coax1; i++)
    {
        current_temp = current_temp + (Hx_GPU[ind + i - (width_coax0 + 1) * Imax] - Hx_GPU[ind + i + width_coax1 * Imax]) * delX;
    }
    T tep;

    tep = (mode == 1) ? (source[time] - current_temp * Z_c) / delX : -current_temp * Z_c / delX;
    for (size_t i = -width_coax0; i <= width_coax1; i++)
    {
        Ex_GPU[ind + i * Imax + depth_coax1] = tep;
        Ex_GPU[ind - 1 + i * Imax - depth_coax0] = -tep;
    }
    for (size_t j = -depth_coax0; j <= depth_coax1; j++)
    {
        Ey_GPU[ind + width_coax1 * Imax + j] = tep;
        Ey_GPU[ind - (1 + width_coax0) * Imax + j] = -tep;
    }

    PortSignal_GPU[time] = -Z_c * current_temp + tep * delX;
}

// template <typename T>
// __device__ T current_calculation(const T *__restrict__ Hx, const T *__restrict__ Hy, T delX, size_t ind, int Imax)
// {
//     return 3.1415926535897 / 4.0 * (Hy[ind] - Hy[ind - 1] + Hx[ind - Imax] - Hx[ind]) * delX;
// }

// template <typename T>
// __global__ void UpdateSource(T *__restrict__ PortSignal_GPU, T *__restrict__ Ex_GPU, T *__restrict__ Ey_GPU, T *__restrict__ source, int time,
//                              T *__restrict__ current, size_t ind, int Imax, int Jmax, T delX,
//                              T Z_c, int width_coax0, int width_coax1, int mode)
// {
//     // for(int i= -width_coax0;i<=width_coax1;i++){
//     //     Ex_GPU[Ax + (Ay+i) * Imax + Az * (Imax*Jmax)] = (source[time]-current[0]*Z_c)/delX;
//     //     Ex_GPU[Ax -1 + (Ay+i)* Imax + Az * (Imax*Jmax)] = -Ex_GPU[Ax + (Ay-width_coax0) * Imax + Az * (Imax*Jmax)];}
//     // for(int i= -width_coax0;i<=width_coax1-1;i++){
//     //     Ey_GPU[Ax + (Ay+i)* Imax + Az * (Imax*Jmax)] = 0.0f;
//     // }
//     // Ey_GPU[Ax + (Ay +width_coax1) * Imax + Az * (Imax*Jmax)] = Ex_GPU[Ax + Ay * Imax + Az * (Imax*Jmax)];
//     // Ey_GPU[Ax + (Ay - 1-width_coax0) * Imax + Az * (Imax*Jmax)] = -Ex_GPU[Ax + Ay * Imax + Az * (Imax*Jmax)];
//     T tep;
//     tep = (mode == 1) ? (source[time] - current[0] * Z_c) / delX : -current[0] * Z_c / delX;
//     Ex_GPU[ind] = tep;
//     Ex_GPU[ind - 1] = -tep;
//     Ey_GPU[ind] = tep;
//     Ey_GPU[ind - Imax] = -tep;
//     PortSignal_GPU[time] = -Z_c * current[0] + tep * delX;
// }

// template <typename T>
// __global__ void PortUpdate(T *__restrict__ PortSignal_GPU, T *__restrict__ Ex_GPU, T *__restrict__ Ey_GPU, const T *__restrict__ Hx_GPU,
//                            const T *__restrict__ Hy_GPU, T *__restrict__ source, int time,
//                            T *__restrict__ current, size_t ind, int Imax, int Jmax, T delX,
//                            T Z_c, int width_coax0, int width_coax1, int mode)
// {
//     current[0] = 3.1415926535897 / 4.0 * (Hy_GPU[ind] - Hy_GPU[ind - 1] + Hx_GPU[ind - Imax] - Hx_GPU[ind]) * delX;

//     // for(int i= -width_coax0;i<=width_coax1;i++){
//     //     Ex_GPU[Ax + (Ay+i) * Imax + Az * (Imax*Jmax)] = (source[time]-current[0]*Z_c)/delX;
//     //     Ex_GPU[Ax -1 + (Ay+i)* Imax + Az * (Imax*Jmax)] = -Ex_GPU[Ax + (Ay-width_coax0) * Imax + Az * (Imax*Jmax)];}
//     // for(int i= -width_coax0;i<=width_coax1-1;i++){
//     //     Ey_GPU[Ax + (Ay+i)* Imax + Az * (Imax*Jmax)] = 0.0f;
//     // }
//     // Ey_GPU[Ax + (Ay +width_coax1) * Imax + Az * (Imax*Jmax)] = Ex_GPU[Ax + Ay * Imax + Az * (Imax*Jmax)];
//     // Ey_GPU[Ax + (Ay - 1-width_coax0) * Imax + Az * (Imax*Jmax)] = -Ex_GPU[Ax + Ay * Imax + Az * (Imax*Jmax)];
//     T tep;
//     tep = (mode == 1) ? (source[time] - current[0] * Z_c) / delX : -current[0] * Z_c / delX;

//     Ex_GPU[ind] = tep;
//     Ex_GPU[ind - 1] = -tep;
//     Ey_GPU[ind] = tep;
//     Ey_GPU[ind - Imax] = -tep;
//     PortSignal_GPU[time] = -Z_c * current[0] + tep * delX;
// }

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
__global__ void AddSource_GPU_3D_1DThread(T *__restrict__ E_GPU, const size_t Imax, const size_t Jmax, const size_t Kmax,
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
            E_GPU[index] = E_GPU[index] + source;
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

template <typename T>
__global__ void AdjointSource(T *__restrict__ source_adjoint_GPU, const T *__restrict__ PortSignal_GPU,
                              const size_t max_TimeSteps, const size_t indAnt)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti < max_TimeSteps)
        source_adjoint_GPU[ti+indAnt*max_TimeSteps] = PortSignal_GPU[indAnt*max_TimeSteps + max_TimeSteps - ti - 1];
}

template <typename T>
__global__ void GPU_ReadAndSum_Field(T *__restrict__ E_record, T *__restrict__ E_temp, const T *__restrict__ E, const size_t WidthDesign,
                                     const size_t HeightDesign, const size_t Ax, const size_t Ay, const size_t Az, const size_t Imax, const size_t Jmax, const size_t time)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tk = tid / WidthDesign;
    size_t tj = tid - tk * WidthDesign;
    size_t index = Ax + (Ay + tj) * Imax + (Az + tk) * (Imax * Jmax);
    if (tid < WidthDesign * HeightDesign)
    {
        E_record[tid + time * WidthDesign * HeightDesign] = (E[index] + E_temp[tid]) / 2;
        E_temp[tid] = E[index];
    }
    // int tj=blockIdx.x * blockDim.x + threadIdx.x;
    // int tk=blockIdx.y * blockDim.y + threadIdx.y;
    // if((tj<WidthDesign)&&(tk<HeightDesign)) {
    //     E_record[tj+tk*WidthDesign+time*WidthDesign*HeightDesign]=E[Ax + (Ay+tj) * Imax + (Az+tk) * (Imax*Jmax)];}
}

// template <typename T>
// __global__ void GPU_ComputeGrad(T *__restrict__ E_record, T *__restrict__ temp, const T *__restrict__ E, const int size, const int max_TimeSteps, const int i)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     if (ti < size)
//     {
//         int index = (max_TimeSteps - i - 1) * size + ti;
//         temp[ti] = Ey_ant[index] * (Ey_ant_AD[ti + i * size] + Ey_ant_AD[ti + (i + 1) * size]) / 2.0;
//     }
// }

template <typename T>
__global__ void GPU_Read_Ant_Field(T *__restrict__ E_record, const T *__restrict__ E, const size_t WidthDesign,
                                   const size_t HeightDesign, const size_t Ax, const size_t Ay, const size_t Az, const size_t Imax, const size_t Jmax, const size_t time)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t tj = tid / WidthDesign;
    size_t ti = tid - tj * WidthDesign;
    size_t index = Ax + ti + (Ay + tj) * Imax + Az * (Imax * Jmax);
    if (tid < WidthDesign * HeightDesign)
    {
        E_record[tid + time * WidthDesign * HeightDesign] = E[index];
    }
    // int tj=blockIdx.x * blockDim.x + threadIdx.x;
    // int tk=blockIdx.y * blockDim.y + threadIdx.y;
    // if((tj<WidthDesign)&&(tk<HeightDesign)) {
    //     E_record[tj+tk*WidthDesign+time*WidthDesign*HeightDesign]=E[Ax + (Ay+tj) * Imax + (Az+tk) * (Imax*Jmax)];}
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

// template <typename T>
// __global__ void ReadCoax(T *__restrict__ PortSignal_GPU, const T *__restrict__ Ex_GPU, size_t ind,
//                          const int time, const T *__restrict__ current, const T delX,
//                          T Z_c, T *__restrict__ voltage)
// {
//     PortSignal_GPU[time] = -Z_c * current[0] + Ex_GPU[ind] * delX;
// }
// //

// template <typename T>
// __global__ void Current_coax(T *__restrict__ current, const T *__restrict__ Hx_GPU, const T *__restrict__ Hy_GPU, size_t ind, int Imax, int Jmax,
//                              T delX, int width_coax0, int width_coax1)
// {
//     // current[0]=0;
//     // for(int i= -width_coax0;i<=width_coax1;i++){
//     //     current[0]=current[0]+Hy_GPU[Ax+ (Ay + i)* Imax + Az * (Imax*Jmax)]-Hy_GPU[(Ax-1) + (Ay +i)* Imax + Az * (Imax*Jmax)];
//     // }
//     // current[0]=current[0]+Hx_GPU[Ax + (Ay-1-width_coax0) * Imax + Az * (Imax*Jmax)]-Hx_GPU[Ax + (Ay+width_coax1) * Imax + Az * (Imax*Jmax)];
//     // current[0]=current[0]*3.1415927/4*delX;
//     current[0] = 3.1415926535897 / 4.0 * (Hy_GPU[ind] - Hy_GPU[ind - 1] + Hx_GPU[ind - Imax] - Hx_GPU[ind]) * delX;
// }

// template <typename T>
// __global__ void GPU_ResetField(T *__restrict__ F, const int size)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     if (ti < size)
//         F[ti] = 0.0f;
// }

// template <typename T>
// __global__ void GPU_updateHx1(T *__restrict__ Hx, const T *__restrict__ Ez, const T *__restrict__ Ey,
//                               const T Chxh, const T Chxe, const int Imax, const int Jmax, const int Kmax, const int dim)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     int idz = ti / dim;
//     int idy = (ti - idz * dim) / Imax;
//     int idx = ti - idz * dim - idy * Imax;

//     __shared__ T Chxh_deivce;
//     Chxh_deivce = Chxh;
//     __shared__ T Chxe_device;
//     Chxe_device = Chxe;
//     int idEz = ti + Imax;
//     int idEy = ti + (Imax * Jmax);
//     if ((idz < Kmax - 1) && (idx < Imax - 1) && (idy < Jmax - 1))
//         Hx[ti] = Chxh_deivce * Hx[ti] + Chxe_device * (-(Ez[idEz] - Ez[ti]) + (Ey[idEy] - Ey[ti]));
// }
// template <typename T>
// __global__ void GPU_updateHy1(T *__restrict__ Hy, const T *__restrict__ Ex, const T *__restrict__ Ez,
//                               const T Chyh, const T Chye, const int Imax, const int Jmax, const int Kmax, const int dim)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     int idz = ti / dim;
//     int idy = (ti - idz * dim) / Imax;
//     int idx = ti - idz * dim - idy * Imax;
//     int idEx = ti + Imax * Jmax;
//     int idEz = ti + 1;
//     if ((idz < Kmax - 1) && (idx < Imax - 1) && (idy < Jmax - 1))
//         Hy[ti] = Chyh * Hy[ti] + Chye * (-(Ex[idEx] - Ex[ti]) + (Ez[idEz] - Ez[ti]));
// }

// template <typename T>
// __global__ void GPU_updateHz1(T *__restrict__ Hz, const T *__restrict__ Ey, const T *__restrict__ Ex,
//                               const T Chzh, const T Chze, const int Imax, const int Jmax, const int Kmax, const int dim)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     int idz = ti / dim;
//     int idy = (ti - idz * dim) / Imax;
//     int idx = ti - idz * dim - idy * Imax;
//     int idEy = ti + 1;
//     int idEx = ti + Imax;
//     __shared__ T Chzh_device;
//     Chzh_device = Chzh;
//     __shared__ T Chze_device;
//     Chze_device = Chze;
//     __syncthreads();
//     if ((idz < Kmax - 1) && (idx < Imax - 1) && (idy < Jmax - 1))
//         Hz[ti] = Chzh_device * Hz[ti] + Chze_device * (-(Ey[idEy] - Ey[ti]) + (Ex[idEx] - Ex[ti]));
// }

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

        // Hx[ti] = Chxh * Hx[ti] + Chxe * (-(Ez[idEz1] - Ez_ti) + (Ey[idEy1] - Ey_ti));
        // Hy[ti] = Chxh * Hy[ti] + Chxe * (-(Ex[idEx2] - Ex_ti) + (Ez[idEz2] - Ez_ti));
        // Hz[ti] = Chxh * Hz[ti] + Chxe * (-(Ey[idEy3] - Ey_ti) + (Ex[idEx3] - Ex_ti));

        Hx[ti] = fma_custom(Chxh, Hx[ti], Chxe * (-(Ez[idEz1] - Ez_ti) + (Ey[idEy1] - Ey_ti)));
        Hy[ti] = fma_custom(Chxh, Hy[ti], Chxe * (-(Ex[idEx2] - Ex_ti) + (Ez[idEz2] - Ez_ti)));
        Hz[ti] = fma_custom(Chxh, Hz[ti], Chxe * (-(Ey[idEy3] - Ey_ti) + (Ex[idEx3] - Ex_ti)));
    }
}

// template <typename T>
// __global__ void GPU_updateHPmlBx1(T *__restrict__ H, const T *__restrict__ E, T *__restrict__ psi_H_1, const T *__restrict__ b_H_1,
//                                   const T *__restrict__ c_H_1, const int I, const int J, const int K, const T Che, const T d, const int Imax,
//                                   const int Jmax, const int Kmax)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     int dim = Imax * Jmax;
//     int dim_s = I * J;
//     int idz = ti / dim_s;
//     int idy = (ti - idz * dim_s) / I;
//     int idx = ti - idz * dim_s - idy * I;
//     int id = idx + idy * Imax + idz * dim;
//     __shared__ T den_device;
//     den_device = d;
//     __shared__ T Che_device;
//     Che_device = Che;
//     if ((idz < K - 1) && (idx < I - 1) && (idy < J - 1))
//     {
//         psi_H_1[ti] = b_H_1[idx] * psi_H_1[ti] + c_H_1[idx] * (E[id + 1] - E[id]) / den_device;
//         H[id] = H[id] - Che_device * psi_H_1[ti] * den_device;
//     }
// }

// template <typename T>
// __global__ void GPU_updateHPmlBy1(T *__restrict__ H, const T *__restrict__ E, T *__restrict__ psi_H_1, const T *__restrict__ b_H_1,
//                                   const T *__restrict__ c_H_1, const int I, const int J, const int K, const T Che, const T d, const int Imax,
//                                   const int Jmax, const int Kmax)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     int dim = Imax * Jmax;
//     int dim_s = I * J;
//     int idz = ti / dim_s;
//     int idy = (ti - idz * dim_s) / I;
//     int idx = ti - idz * dim_s - idy * I;
//     int id = idx + idy * Imax + idz * dim;

//     __shared__ T den_device;
//     den_device = d;
//     __shared__ T Che_device;
//     Che_device = Che;

//     if ((idz < K - 1) && (idx < I - 1) && (idy < J - 1))
//     {
//         psi_H_1[ti] = b_H_1[idy] * psi_H_1[ti] + c_H_1[idy] * (E[id + Imax] - E[id]) / den_device;
//         H[id] = H[id] - Che_device * psi_H_1[ti] * den_device;
//     }
// }

// template <typename T>
// __global__ void GPU_updateHPmlBz1(T *__restrict__ H, const T *__restrict__ E, T *__restrict__ psi_H_1, const T *__restrict__ b_H_1,
//                                   const T *__restrict__ c_H_1, const int I, const int J, const int K, const T Che, const T d, const int Imax,
//                                   const int Jmax, const int Kmax)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     int dim = Imax * Jmax;
//     int dim_s = I * J;
//     int idz = ti / dim_s;
//     int idy = (ti - idz * dim_s) / I;
//     int idx = ti - idz * dim_s - idy * I;
//     int id = idx + idy * Imax + idz * dim;
//     __shared__ T den_device;
//     den_device = d;
//     __shared__ T Che_device;
//     Che_device = Che;
//     if ((idz < K - 1) && (idx < I - 1) && (idy < J - 1))
//     {
//         psi_H_1[ti] = b_H_1[idz] * psi_H_1[ti] + c_H_1[idz] * (E[id + dim] - E[id]) / den_device;
//         H[id] = H[id] - Che_device * psi_H_1[ti] * den_device;
//     }
// }

// template <typename T>
// __global__ void GPU_updateHPmlTx1(T *__restrict__ Hy, const T *__restrict__ Ez, T *__restrict__ psi_Hyx_2, const T *__restrict__ b_Hx_2,
//                                   const T *__restrict__ c_Hx_2, const T Chye, const T dx, const int nxPML_2, const int Imax, const int Jmax,
//                                   const int Kmax)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     int dim = Imax * Jmax;
//     int dim_s = nxPML_2 * Jmax;
//     int idz = ti / dim_s;
//     int idy = (ti - idz * dim_s) / nxPML_2;
//     int idx = ti - idz * dim_s - idy * nxPML_2;
//     int id = Imax - nxPML_2 + idx + idy * Imax + idz * dim;
//     if ((idz < Kmax - 1) && (idx < nxPML_2 - 1) && (idy < Jmax - 1))
//     {
//         psi_Hyx_2[ti] = b_Hx_2[idx + 1] * psi_Hyx_2[ti] + c_Hx_2[idx + 1] * (Ez[id + 1] - Ez[id]) / dx;
//         Hy[id] = Hy[id] + Chye * psi_Hyx_2[ti] * dx;
//     }
// }

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

// template <typename T>
// __global__ void GPU_updateHPmlTy1(T *__restrict__ Hx, const T *__restrict__ Ez, T *__restrict__ psi_Hxy_2, const T *__restrict__ b_Hy_2,
//                                   const T *__restrict__ c_Hy_2, const T Chxe, const T dy, const int nyPML_2, const int Imax, const int Jmax,
//                                   const int Kmax)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     int dim = Imax * Jmax;
//     int dim_s = nyPML_2 * Imax;
//     int idz = ti / dim_s;
//     int idy = (ti - idz * dim_s) / Imax;
//     int idx = ti - idz * dim_s - idy * Imax;
//     int id = idx + (idy + Jmax - nyPML_2) * Imax + idz * dim;
//     if ((idz < Kmax - 1) && (idx < Imax - 1) && (idy < nyPML_2 - 1))
//     {
//         psi_Hxy_2[ti] = b_Hy_2[idy + 1] * psi_Hxy_2[ti] + c_Hy_2[idy + 1] * (Ez[id + Imax] - Ez[id]) / dy;
//         Hx[id] = Hx[id] - Chxe * psi_Hxy_2[ti] * dy;
//     }
// }

// template <typename T>
// __global__ void GPU_updateHPmlTz1(T *__restrict__ Hx, const T *__restrict__ Ey, T *__restrict__ psi_Hxz_2, const T *__restrict__ b_Hz_2,
//                                   const T *__restrict__ c_Hz_2, const T Chxe, const T dz, const int nzPML_2, const int Imax, const int Jmax,
//                                   const int Kmax)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     int dim = Imax * Jmax;
//     int dim_s = dim;
//     int idz = ti / dim_s;
//     int idy = (ti - idz * dim_s) / Imax;
//     int idx = ti - idz * dim_s - idy * Imax;
//     int id = idx + idy * Imax + (idz + Kmax - nzPML_2) * dim;
//     if ((idz < nzPML_2 - 1) && (idx < Imax - 1) && (idy < Jmax - 1))
//     {
//         psi_Hxz_2[ti] = b_Hz_2[idz + 1] * psi_Hxz_2[ti] + c_Hz_2[idz + 1] * (Ey[id + dim] - Ey[id]) / dz;
//         Hx[id] = Hx[id] + Chxe * psi_Hxz_2[ti] * dz;
//     }
// }

template <typename T>
__global__ void GPU_updateE1(T *__restrict__ Ex, T *__restrict__ Ey, T *__restrict__ Ez, const T *__restrict__ Cexe, const T *__restrict__ Cexh, const T *__restrict__ Ceye, const T *__restrict__ Ceyh,
                             const T *__restrict__ Ceze, const T *__restrict__ Cezh, const T *__restrict__ Hx, const T *__restrict__ Hy,
                             const T *__restrict__ Hz, const size_t Imax, const size_t Jmax, const size_t Kmax, const size_t dim, T *__restrict__ psi_Eyx_1, T *__restrict__ psi_Ezx_1,
                             T *__restrict__ psi_Exy_1, T *__restrict__ psi_Ezy_1, T *__restrict__ psi_Exz_1, T *__restrict__ psi_Eyz_1,
                             const T dx, const T *__restrict__ b_Ex_1, const T *__restrict__ c_Ex_1, const size_t nxPML)
{
    size_t ti = blockIdx.x * blockDim.x + threadIdx.x + 1 + Imax * Jmax + Imax;
    size_t idz = ti / dim;
    size_t idy = (ti - idz * dim) / Imax;
    size_t idx = ti - idz * dim - idy * Imax;

    bool cond = (idz < Kmax - 1) && (idx < Imax - 1) && (idy < Jmax - 1) && (idx > 0) && (idy > 0);
    if (cond)
    {
        T dy = dx;
        T dz = dx;

        size_t nyPML = nxPML;
        size_t nzPML = nxPML;
        T Hx_ti = Hx[ti];
        T Hy_ti = Hy[ti];
        T Hz_ti = Hz[ti];

        T Hzx_temp = Hz_ti - Hz[ti - 1];
        T Hyx_temp = Hy_ti - Hy[ti - 1];
        T Hzy_temp = Hz_ti - Hz[ti - Imax];
        T Hxy_temp = Hx_ti - Hx[ti - Imax];
        T Hyz_temp = Hy_ti - Hy[ti - dim];
        T Hxz_temp = Hx_ti - Hx[ti - dim];

        Ex[ti] = fma(Cexe[ti], Ex[ti], Cexh[ti] * (+Hzy_temp - Hyz_temp));
        Ey[ti] = fma(Ceye[ti], Ey[ti], Ceyh[ti] * (+Hxz_temp - Hzx_temp));
        Ez[ti] = fma(Ceze[ti], Ez[ti], Cezh[ti] * (+Hyx_temp - Hxy_temp));

        if (idx <= nxPML - 1)
        {
            size_t id = idx + idy * 2 * nxPML + idz * 2 * nxPML * Jmax;
            psi_Eyx_1[id] = b_Ex_1[idx] * psi_Eyx_1[id] + c_Ex_1[idx] * Hzx_temp / dx;
            Ey[ti] = Ey[ti] - Ceyh[ti] * psi_Eyx_1[id] * dx;

            psi_Ezx_1[id] = b_Ex_1[idx] * psi_Ezx_1[id] + c_Ex_1[idx] * Hyx_temp / dx;
            Ez[ti] = Ez[ti] + Cezh[ti] * psi_Ezx_1[id] * dx;
        }
        else if (idx > Imax - nxPML - 1)
        {
            size_t id1 = idx - (Imax - 2 * nxPML) + idy * 2 * nxPML + idz * 2 * nxPML * Jmax;
            psi_Eyx_1[id1] = b_Ex_1[idx - (Imax - 2 * nxPML)] * psi_Eyx_1[id1] + c_Ex_1[idx - (Imax - 2 * nxPML)] * Hzx_temp / dx;
            Ey[ti] = Ey[ti] - Ceyh[ti] * psi_Eyx_1[id1] * dx;

            psi_Ezx_1[id1] = b_Ex_1[idx - (Imax - 2 * nxPML)] * psi_Ezx_1[id1] + c_Ex_1[idx - (Imax - 2 * nxPML)] * Hyx_temp / dx;
            Ez[ti] = Ez[ti] + Cezh[ti] * psi_Ezx_1[id1] * dx;
        }
        if (idy <= nyPML - 1)
        {
            size_t id = idx + idy * Imax + idz * Imax * 2 * nyPML;
            psi_Exy_1[id] = b_Ex_1[idy] * psi_Exy_1[id] + c_Ex_1[idy] * Hzy_temp / dy;
            Ex[ti] = Ex[ti] + Cexh[ti] * psi_Exy_1[id] * dy;
            psi_Ezy_1[id] = b_Ex_1[idy] * psi_Ezy_1[id] + c_Ex_1[idy] * Hxy_temp / dy;
            Ez[ti] = Ez[ti] - Cezh[ti] * psi_Ezy_1[id] * dy;
        }
        else if (idy > Jmax - nyPML - 1)
        {
            size_t id = idx + (idy - (Jmax - 2 * nyPML)) * Imax + idz * Imax * 2 * nyPML;
            psi_Exy_1[id] = b_Ex_1[idy - (Jmax - 2 * nyPML)] * psi_Exy_1[id] + c_Ex_1[idy - (Jmax - 2 * nyPML)] * Hzy_temp / dy;
            Ex[ti] = Ex[ti] + Cexh[ti] * psi_Exy_1[id] * dy;
            psi_Ezy_1[id] = b_Ex_1[idy - (Jmax - 2 * nyPML)] * psi_Ezy_1[id] + c_Ex_1[idy - (Jmax - 2 * nyPML)] * Hxy_temp / dy;
            Ez[ti] = Ez[ti] - Cezh[ti] * psi_Ezy_1[id] * dy;
        }

        if (idz <= nzPML - 1)
        {
            size_t id = idx + idy * Imax + idz * dim;
            psi_Exz_1[id] = b_Ex_1[idz] * psi_Exz_1[id] + c_Ex_1[idz] * Hyz_temp / dz;
            Ex[ti] = Ex[ti] - Cexh[ti] * psi_Exz_1[id] * dz;

            psi_Eyz_1[id] = b_Ex_1[idz] * psi_Eyz_1[id] + c_Ex_1[idz] * Hxz_temp / dz;
            Ey[ti] = Ey[ti] + Ceyh[ti] * psi_Eyz_1[id] * dz;
        }
        else if (idz > Kmax - nzPML - 1)
        {
            size_t id = idx + idy * Imax + (idz - (Kmax - 2 * nzPML)) * dim;
            psi_Exz_1[id] = b_Ex_1[idz - (Kmax - 2 * nzPML)] * psi_Exz_1[id] + c_Ex_1[idz - (Kmax - 2 * nzPML)] * Hyz_temp / dz;
            Ex[ti] = Ex[ti] - Cexh[ti] * psi_Exz_1[id] * dz;
            psi_Eyz_1[id] = b_Ex_1[idz - (Kmax - 2 * nzPML)] * psi_Eyz_1[id] + c_Ex_1[idz - (Kmax - 2 * nzPML)] * Hxz_temp / dz;
            Ey[ti] = Ey[ti] + Ceyh[ti] * psi_Eyz_1[id] * dz;
        }
    }
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
    if ((idz < Kmax) && (idx < Imax) && (idy < Jmax) && (idx > 0) && (idy > 0) && (idz > 0))
    {
        T Hx_ti = Hx[ti];
        T Hy_ti = Hy[ti];
        T Hz_ti = Hz[ti];

        Ex[ti] = fma_custom(Cexe[ti], Ex[ti], Cexh[ti] * ((Hz_ti - Hz[idHz1]) - (Hy_ti - Hy[idHy1])));
        Ey[ti] = fma_custom(Ceye[ti], Ey[ti], Ceyh[ti] * (+(Hx_ti - Hx[idHx2]) - (Hz_ti - Hz[idHz2])));
        Ez[ti] = fma_custom(Ceze[ti], Ez[ti], Cezh[ti] * (+(Hy_ti - Hy[idHy3]) - (Hx_ti - Hx[idHx3])));
        // Ex[ti] = Cexe[ti] * Ex[ti] + Cexh[ti] * (+(Hz_ti - Hz[idHz1]) - (Hy_ti - Hy[idHy1]));
        // Ey[ti] = Ceye[ti] * Ey[ti] + Ceyh[ti] * (+(Hx_ti - Hx[idHx2]) - (Hz_ti - Hz[idHz2]));
        // Ez[ti] = Ceze[ti] * Ez[ti] + Cezh[ti] * (+(Hy_ti - Hy[idHy3]) - (Hx_ti - Hx[idHx3]));
    }
}

// template <typename T>
// __global__ void GPU_updateEx_basic(T *__restrict__ Ex, const T *__restrict__ Cexe, const T *__restrict__ Cexh, const T *__restrict__ Hz,
//                                    const T *__restrict__ Hy, const int Imax, const int Jmax, const int Kmax, const int dim)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x + 1 + Imax * Jmax + Imax;
//     int idz = ti / dim;
//     int idy = (ti - idz * dim) / Imax;
//     int idx = ti - idz * dim - idy * Imax;
//     int idHy = ti - dim;
//     int idHz = ti - Imax;
//     if ((idz < Kmax - 1) && (idx < Imax - 1) && (idy < Jmax - 1) && (idx > 0) && (idy > 0))
//     {
//         Ex[ti] = Cexe[ti] * Ex[ti] + Cexh[ti] * (+(Hz[ti] - Hz[idHz]) - (Hy[ti] - Hy[idHy]));
//     }
// }

// template <typename T>
// __global__ void GPU_updateEy_basic(T *__restrict__ Ey, const T *__restrict__ Ceye, const T *__restrict__ Ceyh,
//                                    const T *__restrict__ Hx, const T *__restrict__ Hz, const int Imax, const int Jmax, const int Kmax, const int dim)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x + 1 + Imax * Jmax + Imax;
//     int idz = ti / dim;
//     int idy = (ti - idz * dim) / Imax;
//     int idx = ti - idz * dim - idy * Imax;
//     int idHx = ti - dim;
//     int idHz = ti - 1;
//     if ((idz < Kmax - 1) && (idx < Imax - 1) && (idy < Jmax - 1) && (idx > 0) && (idy > 0))
//     {
//         Ey[ti] = Ceye[ti] * Ey[ti] + Ceyh[ti] * (+(Hx[ti] - Hx[idHx]) - (Hz[ti] - Hz[idHz]));
//     }
// }
// template <typename T>
// __global__ void GPU_updateEz_basic(T *__restrict__ Ez, const T *__restrict__ Ceze, const T *__restrict__ Cezh, const T *__restrict__ Hy,
//                                    const T *__restrict__ Hx, const int Imax, const int Jmax, const int Kmax, const int dim)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x + 1 + Imax * Jmax + Imax;
//     int idz = ti / dim;
//     int idy = (ti - idz * dim) / Imax;
//     int idx = ti - idz * dim - idy * Imax;
//     int idHx = ti - Imax;
//     int idHy = ti - 1;
//     if ((idz < Kmax - 1) && (idx < Imax - 1) && (idy < Jmax - 1) && (idx > 0) && (idy > 0))
//     {
//         Ez[ti] = Ceze[ti] * Ez[ti] + Cezh[ti] * (+(Hy[ti] - Hy[idHy]) - (Hx[ti] - Hx[idHx]));
//     }
// }

// template <typename T>
// __global__ void GPU_updateEPmlBx1(const T *__restrict__ Hz, T *__restrict__ Ey, T *__restrict__ psi_Eyx_1, const T *__restrict__ b_Ex_1,
//                                   const T *__restrict__ c_Ex_1, const T *__restrict__ Ceyh,
//                                   const T dx, const int I, const int J, const int K, const T c, const int Imax, const int Jmax, const int Kmax)
// {

//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     int dim = Imax * Jmax;
//     int dim_s = I * J;
//     int idz = ti / dim_s;
//     int idy = (ti - idz * dim_s) / I;
//     int idx = ti - idz * dim_s - idy * I;
//     int id = idx + idy * Imax + idz * dim;
//     if ((idz < K - 1) && (idz >= 1) && (idx > 0) && (idx <= I - 1) && (idy > 0) && (idy < J - 1))
//     {
//         psi_Eyx_1[ti] = b_Ex_1[idx] * psi_Eyx_1[ti] + c_Ex_1[idx] * (Hz[id] - Hz[id - 1]) / dx;
//         Ey[id] = Ey[id] - c * Ceyh[id] * psi_Eyx_1[ti] * dx;
//     }
// }

// template <typename T>
// __global__ void GPU_updateEPmlBy1(const T *__restrict__ Hz, T *__restrict__ Ex, T *__restrict__ psi_Exy_1,
//                                   const T *__restrict__ b_Ey_1, const T *__restrict__ c_Ey_1, const T *__restrict__ Cexh,
//                                   const T dy, const int I, const int J, const int K, const int c, const int Imax, const int Jmax, const int Kmax)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     int dim = Imax * Jmax;
//     int dim_s = I * J;
//     int idz = ti / dim_s;
//     int idy = (ti - idz * dim_s) / I;
//     int idx = ti - idz * dim_s - idy * I;
//     int id = idx + idy * Imax + idz * dim;
//     if ((idz < K - 1) && (idz >= 1) && (idx < I - 1) && (idx > 0) && (idy > 0) && (idy <= J - 1))
//     {
//         psi_Exy_1[ti] = b_Ey_1[idy] * psi_Exy_1[ti] + c_Ey_1[idy] * (Hz[id] - Hz[id - Imax]) / dy;
//         Ex[id] = Ex[id] + c * Cexh[id] * psi_Exy_1[ti] * dy;
//     }
// }

// template <typename T>
// __global__ void GPU_updateEPmlBz1(const T *__restrict__ Hy, T *__restrict__ Ex, T *__restrict__ psi_Exz_1,
//                                   const T *__restrict__ b_Ez_1, const T *__restrict__ c_Ez_1, const T *__restrict__ Cexh,
//                                   const T dz, const int I, const int J, const int K, const int c, const int Imax, const int Jmax, const int Kmax)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     int dim = Imax * Jmax;
//     int dim_s = I * J;
//     int idz = ti / dim_s;
//     int idy = (ti - idz * dim_s) / I;
//     int idx = ti - idz * dim_s - idy * I;
//     int id = idx + idy * Imax + idz * dim;
//     if ((idz <= K - 1) && (idz >= 1) && (idx < I - 1) && (idx > 0) && (idy < J - 1) && (idy > 0))
//     {
//         psi_Exz_1[ti] = b_Ez_1[idz] * psi_Exz_1[ti] + c_Ez_1[idz] * (Hy[id] - Hy[id - dim]) / dz;
//         Ex[id] = Ex[id] - c * Cexh[id] * psi_Exz_1[ti] * dz;
//     }
// }

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

// template <typename T>
// __global__ void GPU_updateEPmlTx1(const T *__restrict__ Hz, T *__restrict__ Ey, T *__restrict__ psi_Eyx_2, const T *__restrict__ b_Ex_2,
//                                   const T *__restrict__ c_Ex_2, const T *__restrict__ Ceyh, const T dx, const int nxPML_2, const int c,
//                                   const int Imax, const int Jmax, const int Kmax)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     int dim = Imax * Jmax;
//     int dim_s = nxPML_2 * Jmax;
//     int idz = ti / dim_s;
//     int idy = (ti - idz * dim_s) / nxPML_2;
//     int idx = ti - idz * dim_s - idy * nxPML_2;
//     int id = idx + Imax - nxPML_2 + idy * Imax + idz * dim;

//     if ((idz < Kmax - 1) && (idz >= 1) && (idx < nxPML_2 - 1) && (idy < Jmax - 1) && (idy > 0))
//     {
//         psi_Eyx_2[ti] = b_Ex_2[idx] * psi_Eyx_2[ti] + c_Ex_2[idx] * (Hz[id] - Hz[id - 1]) / dx;
//         Ey[id] = Ey[id] - c * Ceyh[id] * psi_Eyx_2[ti] * dx;
//     }
// }

// template <typename T>
// __global__ void GPU_updateEPmlTy1(const T *__restrict__ Hz, T *__restrict__ Ex, T *__restrict__ psi_Exy_2, const T *__restrict__ b_Ey_2,
//                                   const T *__restrict__ c_Ey_2, const T *__restrict__ Cexh, const T dy, const int nyPML_2, const int c,
//                                   const int Imax, const int Jmax, const int Kmax)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     int dim = Imax * Jmax;
//     int dim_s = Imax * nyPML_2;
//     int idz = ti / dim_s;
//     int idy = (ti - idz * dim_s) / Imax;
//     int idx = ti - idz * dim_s - idy * Imax;
//     int id = idx + (idy + Jmax - nyPML_2) * Imax + idz * dim;
//     if ((idz < Kmax - 1) && (idz >= 1) && (idx < Imax - 1) && (idx > 0) && (idy < nyPML_2 - 1))
//     {
//         psi_Exy_2[ti] = b_Ey_2[idy] * psi_Exy_2[ti] + c_Ey_2[idy] * (Hz[id] - Hz[id - Imax]) / dy;
//         Ex[id] = Ex[id] + c * Cexh[id] * psi_Exy_2[ti] * dy;
//     }
// }

// template <typename T>
// __global__ void GPU_updateEPmlTz1(const T *__restrict__ Hy, T *__restrict__ Ex, T *__restrict__ psi_Exz_2, const T *__restrict__ b_Ez_2,
//                                   const T *__restrict__ c_Ez_2, const T *__restrict__ Cexh, const T dz, const int nzPML_2,
//                                   const int c, const int Imax, const int Jmax, const int Kmax)
// {
//     int ti = blockIdx.x * blockDim.x + threadIdx.x;
//     int dim = Imax * Jmax;
//     int dim_s = dim;
//     int idz = ti / dim_s;
//     int idy = (ti - idz * dim_s) / Imax;
//     int idx = ti - idz * dim_s - idy * Imax;
//     int id = idx + idy * Imax + (idz + Kmax - nzPML_2) * dim;
//     if ((idz < nzPML_2 - 1) && (idx < Imax - 1) && (idx > 0) && (idy < Jmax - 1) && (idy > 0))
//     {
//         psi_Exz_2[ti] = b_Ez_2[idz] * psi_Exz_2[ti] + c_Ez_2[idz] * (Hy[id] - Hy[id - dim]) / dz;
//         Ex[id] = Ex[id] - c * Cexh[id] * psi_Exz_2[ti] * dz;
//     }
// }

// template <typename T>
// __global__ void AddSource_coax1(T *__restrict__ Ex_GPU, T *__restrict__ Ey_GPU, T source, T *__restrict__ current,
//                                 int Ax, int Ay, int Az, int Imax, int Jmax, T delX,
//                                 T Z_c, int width_coax0, int width_coax1, T *__restrict__ voltage, T cons, T *PortSignal_GPU, int time)
// {
//     // voltage[1]=(source-current[0]*Z_c+voltage[0]*(-0.5f+Z_c*cons))/(0.5f+Z_c*cons);
//     // for(int i= -width_coax0;i<=width_coax1;i++){
//     //     Ex_GPU[Ax + (Ay+i) * Imax + Az * (Imax*Jmax)] =voltage[1]/delX;
//     //     Ex_GPU[Ax -1 + (Ay+i) * Imax + Az * (Imax*Jmax)] = -Ex_GPU[Ax + (Ay-width_coax0) * Imax + Az * (Imax*Jmax)];}
//     // for(int i= -width_coax0;i<=width_coax1-1;i++){
//     //     Ey_GPU[Ax + (Ay+i)* Imax + Az * (Imax*Jmax)] = 0.0f;
//     // }
//     // Ey_GPU[Ax + (Ay +width_coax1)* Imax + Az * (Imax*Jmax)] = Ex_GPU[Ax + Ay * Imax + Az * (Imax*Jmax)];
//     // Ey_GPU[Ax + (Ay - 1-width_coax0) * Imax + Az * (Imax*Jmax)] = -Ex_GPU[Ax + Ay * Imax + Az * (Imax*Jmax)];
//     // PortSignal_GPU[time]= (voltage[1]+voltage[0])/2.0f-Z_c*(current[0]+cons*(voltage[1]-voltage[0]));
//     //     voltage[0]=voltage[1];
//     voltage[1] = (source - current[0] * Z_c + voltage[0] * (-0.5f + Z_c * cons)) / (0.5f + Z_c * cons);
//     Ex_GPU[Ax + Ay * Imax + Az * Imax * Jmax] = voltage[1] / delX;
//     Ex_GPU[Ax - 1 + Ay * Imax + Az * Imax * Jmax] = -Ex_GPU[Ax + Ay * Imax + Az * Imax * Jmax];

//     Ey_GPU[Ax + Ay * Imax + Az * (Imax * Jmax)] = Ex_GPU[Ax + Ay * Imax + Az * (Imax * Jmax)];
//     Ey_GPU[Ax + (Ay - 1) * Imax + Az * (Imax * Jmax)] = -Ex_GPU[Ax + Ay * Imax + Az * (Imax * Jmax)];
//     PortSignal_GPU[time] = (voltage[1] + voltage[0]) / 2.0f - Z_c * (current[0] + cons * (voltage[1] - voltage[0]));
//     voltage[0] = voltage[1];
// }

// template <typename T>
// __global__ void AddSource_coax(T *__restrict__ Ex_GPU, T *__restrict__ Ey_GPU, T source, T *__restrict__ current, int ind, int Imax, int Jmax, T delX,
//                                T Z_c, int width_coax0, int width_coax1, T *__restrict__ voltage)
// {
//     voltage[0] = source - current[0] * Z_c;
//     // for(int i= -width_coax0;i<=width_coax1;i++){
//     //     Ex_GPU[Ax + (Ay+i) * Imax + Az * (Imax*Jmax)] =(source-current[0]*Z_c)/delX;
//     //     Ex_GPU[Ax -1 + (Ay+i) * Imax + Az * (Imax*Jmax)] = -Ex_GPU[Ax + (Ay-width_coax0) * Imax + Az * (Imax*Jmax)];}
//     // for(int i= -width_coax0;i<=width_coax1-1;i++){
//     //     Ey_GPU[Ax + (Ay+i)* Imax + Az * (Imax*Jmax)] = 0.0f;
//     // }
//     // Ey_GPU[Ax + (Ay +width_coax1)* Imax + Az * (Imax*Jmax)] = Ex_GPU[Ax + Ay * Imax + Az * (Imax*Jmax)];
//     // Ey_GPU[Ax + (Ay - 1-width_coax0) * Imax + Az * (Imax*Jmax)] = -Ex_GPU[Ax + Ay * Imax + Az * (Imax*Jmax)];
//     Ex_GPU[ind] = voltage[0] / delX;
//     Ex_GPU[ind - 1] = -Ex_GPU[ind];
//     Ey_GPU[ind] = Ex_GPU[ind];
//     Ey_GPU[ind - Imax] = -Ex_GPU[ind];
// }

// template <typename T>
// __global__ void AddSource_coax_GPU(T *__restrict__ Ex_GPU, T *__restrict__ Ey_GPU, T *__restrict__ source, int time,
//                                    T *__restrict__ current, size_t ind, int Imax, int Jmax, T delX,
//                                    T Z_c, int width_coax0, int width_coax1)
// {
//     // for(int i= -width_coax0;i<=width_coax1;i++){
//     //     Ex_GPU[Ax + (Ay+i) * Imax + Az * (Imax*Jmax)] = (source[time]-current[0]*Z_c)/delX;
//     //     Ex_GPU[Ax -1 + (Ay+i)* Imax + Az * (Imax*Jmax)] = -Ex_GPU[Ax + (Ay-width_coax0) * Imax + Az * (Imax*Jmax)];}
//     // for(int i= -width_coax0;i<=width_coax1-1;i++){
//     //     Ey_GPU[Ax + (Ay+i)* Imax + Az * (Imax*Jmax)] = 0.0f;
//     // }
//     // Ey_GPU[Ax + (Ay +width_coax1) * Imax + Az * (Imax*Jmax)] = Ex_GPU[Ax + Ay * Imax + Az * (Imax*Jmax)];
//     // Ey_GPU[Ax + (Ay - 1-width_coax0) * Imax + Az * (Imax*Jmax)] = -Ex_GPU[Ax + Ay * Imax + Az * (Imax*Jmax)];
//     Ex_GPU[ind] = (source[time] - current[0] * Z_c) / delX;
//     Ex_GPU[ind - 1] = -Ex_GPU[ind];
//     Ey_GPU[ind] = Ex_GPU[ind];
//     Ey_GPU[ind - Imax] = -Ex_GPU[ind];
// }
