
#define FDTD_UNIFORM_MESH 1

#ifndef SIMULATION_H
#define SIMULATION_H
#include <math.h>
#include <mex.h>
#include <stdio.h>
#include "gpu.h"
#include <iostream>
#include <omp.h>
#ifndef M_PI
#define M_PI 3.1415926535897
#endif
typedef float real;
constexpr size_t real_size = sizeof(float);
constexpr size_t pml_size = 10;
constexpr size_t NumThreads = 1024;
constexpr double c = 299792458;
constexpr double mu0 = 4 * M_PI * 1e-7;
constexpr double eps0 = 1 / c / c / mu0;
constexpr double sigma_max = 5.8e7;

class simulation
{
public:
    ///////set coaxial paramters
    void createMu(real *mu_x, real *mu_y, real *mu_z)
    {
        init_mu = 1;
        real *Cx;
        real *Cy;
        real *Cz;
        cudaMallocHost((void **)&Cx, IJK_Size * real_size);
        cudaMallocHost((void **)&Cy, IJK_Size * real_size);
        cudaMallocHost((void **)&Cz, IJK_Size * real_size);
        cudaMalloc((void **)&Chxe_GPU, IJK_Size * real_size);
        cudaMalloc((void **)&Chye_GPU, IJK_Size * real_size);
        cudaMalloc((void **)&Chze_GPU, IJK_Size * real_size);

        for (size_t i = 0; i < IJK_Size; i++)
        {
            Cx[i] = delT / mu0 / mu_x[i] / (1 + sigma_m * delT / 2 / mu0 / mu_x[i]) * den_hx[i];
        }
        cudaMemcpyAsync(Chxe_GPU, Cx, IJK_Size * real_size, cudaMemcpyHostToDevice, stream);
        for (size_t i = 0; i < IJK_Size; i++)
        {
            Cy[i] = delT / mu0 / mu_y[i] / (1 + sigma_m * delT / 2 / mu0 / mu_y[i]) * den_hx[i];
        }
        cudaMemcpyAsync(Chye_GPU, Cy, IJK_Size * real_size, cudaMemcpyHostToDevice, stream);
        for (size_t i = 0; i < IJK_Size; i++)
        {
            Cz[i] = delT / mu0 / mu_z[i] / (1 + sigma_m * delT / 2 / mu0 / mu_z[i]) * den_hx[i];
        }
        cudaMemcpyAsync(Chze_GPU, Cz, IJK_Size * real_size, cudaMemcpyHostToDevice, stream);

        cudaFreeHost(Cx);
        cudaFreeHost(Cy);
        cudaFreeHost(Cz);
    }

    //////Initialize simulation
    simulation()
    {
        init_impulse = 0;
        init_1D = 0;
        init_mu = 0;
        init_design = 0;
        block_size = 32;
        cudaStreamCreate(&stream);
    }
    ////////////////////set constant parameters
    void setConst(real *freqRes, int *TimeDim)
    {
        fctr = freqRes[0];
        delX = freqRes[1];
        delT = freqRes[2];
        poly_m = 3;
        alpha_m = 1.0;
        sig_max = 0.8 * (poly_m + 1.0) / (sqrt(mu0 / eps0) * delX);
        alpha_max = 0.08;
        kappa_max = 1.0;
        max_TimeSteps = TimeDim[0];
        I_size = TimeDim[1];
        J_size = TimeDim[2];
        K_size = TimeDim[3];
        numAnt = TimeDim[4];
        /////////////////////////////////////////////
        // PML thickness in each direction
        Imax = I_size + 2 * pml_size;
        Jmax = J_size + 2 * pml_size;
        Kmax = K_size + 2 * pml_size;
        IJ_Size = Imax * Jmax;
        IJK_Size = Imax * Jmax * Kmax;
        NumBlocks = (IJK_Size + NumThreads - 1) / NumThreads;

        Z_c = 50.0;
        size_t width = 15;
        i0 = width;
        i1 = Imax - width - 1;
        j0 = width;
        j1 = Jmax - width - 1;
        k0 = width;
        k1 = Kmax - width - 1;

        DesignY = 100;
        DesignZ = 100;
        width_coax0 = TimeDim[5];
        width_coax1 = TimeDim[6];
        depth_coax0 = TimeDim[7];
        depth_coax1 = TimeDim[8];
        size_x = pml_size * Jmax * Kmax;
        size_y = Imax * pml_size * Kmax;
        size_z = Imax * Jmax * pml_size;
        gridsize_x = (2 * size_x + NumThreads - 1) / NumThreads;
        gridsize_y = (2 * size_y + NumThreads - 1) / NumThreads;
        gridsize_z = (2 * size_z + NumThreads - 1) / NumThreads;

        init_const = 1;
        cudaMallocHost((void **)&eps_infx, IJK_Size * real_size);
        cudaMallocHost((void **)&eps_infy, IJK_Size * real_size);
        cudaMallocHost((void **)&eps_infz, IJK_Size * real_size);
        cudaMallocHost((void **)&sigma_sx, IJK_Size * real_size);
        cudaMallocHost((void **)&sigma_sy, IJK_Size * real_size);
        cudaMallocHost((void **)&sigma_sz, IJK_Size * real_size);

        std::fill(eps_infx, eps_infx + IJK_Size, 1.0);
        std::fill(eps_infy, eps_infy + IJK_Size, 1.0);
        std::fill(eps_infz, eps_infz + IJK_Size, 1.0);
        std::fill(sigma_sx, sigma_sx + IJK_Size, 0.0);
        std::fill(sigma_sy, sigma_sy + IJK_Size, 0.0);
        std::fill(sigma_sz, sigma_sz + IJK_Size, 0.0);
    }

    ////Initialize 1D wave propagation
    void init1D(real *rate, real *PW_source)
    {
        init_1D = 1;
        size_1D = max(max(Imax, Jmax), Kmax);
        size_t NumBlock_1D = NumThreads;
        size_t NumGrid_1D = 1;
        if (size_1D < NumThreads)
            NumBlock_1D = size_1D;
        else
            NumGrid_1D = (size_1D + NumThreads - 1) / NumThreads;
        real Ceh_1D = delT / eps0 / delX;
        real *b_H_1D;
        real *c_H_1D;
        real *b_E_1D;
        real *c_E_1D;

        cudaMallocHost(&Ez_1D, max_TimeSteps * (size_1D)*real_size); // Pinned memory for asynchronous copy
        std::fill(Ez_1D, Ez_1D + max_TimeSteps * (size_1D), 0);

        real *Hy_1D_temp;
        cudaMalloc(&Hy_1D_temp, size_1D * real_size);
        cudaMemsetAsync(Hy_1D_temp, 0, size_1D * real_size, stream);

        real *psi_Ezx_1D;
        real *psi_Hyx_1D;
        cudaMalloc(&psi_Ezx_1D, 2 * pml_size * real_size);
        cudaMalloc(&psi_Hyx_1D, 2 * pml_size * real_size);
        cudaMemsetAsync(psi_Ezx_1D, 0, 2 * pml_size * real_size, stream);
        cudaMemsetAsync(psi_Hyx_1D, 0, 2 * pml_size * real_size, stream);

        cudaMalloc(&b_H_1D, 2 * pml_size * real_size);
        cudaMalloc(&c_H_1D, 2 * pml_size * real_size);
        cudaMalloc(&b_E_1D, 2 * pml_size * real_size);
        +cudaMalloc(&c_E_1D, 2 * pml_size * real_size);
        cudaMemcpyAsync(b_H_1D, b_Hx_1, 2 * pml_size * real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(c_H_1D, c_Hx_1, 2 * pml_size * real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(b_E_1D, b_Ex_1, 2 * pml_size * real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(c_E_1D, c_Ex_1, 2 * pml_size * real_size, cudaMemcpyHostToDevice, stream);
        cudaMalloc(&Ez_1D_GPU, max_TimeSteps * (size_1D)*real_size);
        cudaMalloc(&Hy_1D_GPU, max_TimeSteps * (size_1D)*real_size);
        cudaMemsetAsync(Ez_1D_GPU, 0, max_TimeSteps * (size_1D)*real_size, stream);
        cudaMemsetAsync(Hy_1D_GPU, 0, max_TimeSteps * (size_1D)*real_size, stream);
        real *Ez_1d_temp;
        cudaMalloc(&Ez_1d_temp, size_1D * real_size);
        cudaMemsetAsync(Ez_1d_temp, 0, size_1D * real_size, stream);
        cudaMalloc(&E_inv_1D_GPU, max_TimeSteps * (size_1D)*real_size);
        cudaMalloc(&H_inv_1D_GPU, max_TimeSteps * (size_1D)*real_size);
        cudaMemsetAsync(E_inv_1D_GPU, 0, max_TimeSteps * (size_1D)*real_size, stream);
        cudaMemsetAsync(H_inv_1D_GPU, 0, max_TimeSteps * (size_1D)*real_size, stream);
        size_t time;
        real source_rate = rate[0];
        real *PW_source_1D = new real[max_TimeSteps];
        for (size_t i = 0; i < max_TimeSteps; i++)
        {
            PW_source_1D[i] = PW_source[i] * source_rate;
        }
        for (size_t n = 0; n < max_TimeSteps; n++)
        {
            time = n * size_1D;
            copy1d<<<NumGrid_1D, NumBlock_1D, 0, stream>>>(Ez_1D_GPU, Ez_1d_temp, size_1D, time);
            GPU_updateH_1D<<<NumGrid_1D, NumBlock_1D, 0, stream>>>(Hy_1D_temp, Ez_1d_temp, 1 / delX, Che, size_1D);
            GPU_updateHPml_1D<<<1, 2 * pml_size, 0, stream>>>(Ez_1d_temp, Hy_1D_temp, psi_Hyx_1D, b_H_1D, c_H_1D, Che, delX, size_1D, pml_size);
            GPU_updateE_1D<<<NumGrid_1D, NumBlock_1D, 0, stream>>>(Ez_1d_temp, Hy_1D_temp, 1 / delX, Ceh_1D, size_1D);
            GPU_updateEPml_1D<<<1, 2 * pml_size, 0, stream>>>(Hy_1D_temp, Ez_1d_temp, psi_Ezx_1D, b_E_1D, c_E_1D, Ceh_1D, delX, size_1D, pml_size);
            AddSource_GPU_1D<<<1, 1, 0, stream>>>(Ez_1d_temp, size_1D - 13, PW_source_1D[n]);
            copy1d<<<NumGrid_1D, NumBlock_1D, 0, stream>>>(Hy_1D_GPU, Hy_1D_temp, size_1D, time);
        }
        cudaMemsetAsync(psi_Ezx_1D, 0, 2 * pml_size * real_size, stream);
        cudaMemsetAsync(psi_Hyx_1D, 0, 2 * pml_size * real_size, stream);
        cudaMemsetAsync(Hy_1D_temp, 0, size_1D * real_size, stream);
        cudaMemsetAsync(Ez_1d_temp, 0, size_1D * real_size, stream);
        for (size_t n = 0; n < max_TimeSteps; n++)
        {
            time = n * size_1D;
            copy1d<<<NumGrid_1D, NumBlock_1D, 0, stream>>>(E_inv_1D_GPU, Ez_1d_temp, size_1D, time);
            GPU_updateH_1D<<<NumGrid_1D, NumBlock_1D, 0, stream>>>(Hy_1D_temp, Ez_1d_temp, 1 / delX, Che, size_1D);

            GPU_updateHPml_1D<<<1, 2 * pml_size, 0, stream>>>(Ez_1d_temp, Hy_1D_temp, psi_Hyx_1D, b_H_1D, c_H_1D, Che, delX, size_1D, pml_size);

            GPU_updateE_1D<<<NumGrid_1D, NumBlock_1D, 0, stream>>>(Ez_1d_temp, Hy_1D_temp, 1 / delX, Ceh_1D, size_1D);

            GPU_updateEPml_1D<<<1, 2 * pml_size, 0, stream>>>(Hy_1D_temp, Ez_1d_temp, psi_Ezx_1D, b_E_1D, c_E_1D, Ceh_1D, delX, size_1D, pml_size);

            AddSource_GPU_1D<<<1, 1, 0, stream>>>(Ez_1d_temp, 13, PW_source_1D[n]);
            copy1d<<<NumGrid_1D, NumBlock_1D, 0, stream>>>(H_inv_1D_GPU, Hy_1D_temp, size_1D, time);
        }
        delete[] PW_source_1D;
        cudaFree(Ez_1d_temp);
        cudaFree(psi_Ezx_1D);
        cudaFree(psi_Hyx_1D);
        cudaFree(Hy_1D_temp);
        cudaFree(b_H_1D);
        cudaFree(c_H_1D);
        cudaFree(b_E_1D);
        cudaFree(c_E_1D);
    }

    void computeG_ant(const real *E1_ant, const real *E1_ant_AD, double *G1_ant, size_t size, size_t max_TimeSteps)
    {
#pragma omp parallel for
        for (size_t ti = 0; ti < size; ++ti)
        {
            double sum = 0.0;
            for (size_t i = 0; i < max_TimeSteps - 1; ++i)
            {
                size_t index = (max_TimeSteps - i - 1) * size + ti;
                sum += static_cast<double>(E1_ant[index]) * (static_cast<double>(E1_ant_AD[ti + i * size]) + static_cast<double>(E1_ant_AD[ti + (i + 1) * size])) / 2.0;
            }
            G1_ant[ti] = sum;
        }
        if (NumML > 1)
        {
            for (size_t ind = 1; ind < NumML; ind++)
            {
#pragma omp parallel for
                for (size_t ti = 0; ti < size; ++ti)
                {
                    double sum = 0.0;
                    for (size_t i = 0; i < max_TimeSteps - 1; ++i)
                    {
                        size_t index = (max_TimeSteps - i - 1) * size + ti;
                        sum += static_cast<double>(E1_ant[index + ind * max_TimeSteps * size]) * (static_cast<double>(E1_ant_AD[ti + i * size + ind * max_TimeSteps * size]) + static_cast<double>(E1_ant_AD[ti + (i + 1) * size + ind * max_TimeSteps * size])) / 2.0;
                    }
                    G1_ant[ti + ind * size] = sum;
                }
            }
        }
    }
    void updateMaterial(int *flag, real *values)
    {
        if (flag[0] == 1)
            for (size_t i = 0; i < IJK_Size; i++)
            {
                sigma_sx[i] = values[i];
            }
        else if (flag[0] == 2)
            for (size_t i = 0; i < IJK_Size; i++)
            {
                sigma_sy[i] = values[i];
            }
        else if (flag[0] == 3)
        {
            for (size_t i = 0; i < IJK_Size; i++)
            {
                sigma_sz[i] = values[i];
            }
        }
        else if (flag[0] == 4)
        {
            for (size_t i = 0; i < IJK_Size; i++)
            {
                eps_infx[i] = values[i];
            }
        }
        else if (flag[0] == 5)
        {
            for (size_t i = 0; i < IJK_Size; i++)
            {
                eps_infy[i] = values[i];
            }
        }
        else if (flag[0] == 6)
        {
            for (size_t i = 0; i < IJK_Size; i++)
            {
                eps_infz[i] = values[i];
            }
        }
    }

    void updateH_PlaneWave(int *time)
    {
        size_t nstep = time[0] * size_1D;
        if (time[1] == 1)
        {
            updateH_PW_ZY(Ez_1D_GPU, 1, nstep + size_1D - Kmax);
        }
        else if (time[1] == 2)
        {
            updateH_PW_ZX(Ez_1D_GPU, 1, nstep + size_1D - Kmax);
        }
        else if (time[1] == 3)
        {
            updateH_PW_XZ(Ez_1D_GPU, 1, nstep + size_1D - Imax);
        }
        else if (time[1] == 4)
        {
            updateH_PW_XY(E_inv_1D_GPU, 1, nstep + size_1D - Imax);
        }
    }
    void updateE_PlaneWave(int *time)
    {
        size_t nstep = time[0] * size_1D;
        if (time[1] == 1)
        {
            updateE_PW_ZY(Hy_1D_GPU, 1, nstep + size_1D - Kmax);
        }
        else if (time[1] == 2)
        {
            updateE_PW_ZX(Hy_1D_GPU, 1, nstep + size_1D - Kmax);
        }
        else if (time[1] == 3)
        {
            updateE_PW_XZ(Hy_1D_GPU, 1, nstep + size_1D - Imax);
        }
        else if (time[1] == 4)
        {
            updateE_PW_XY(H_inv_1D_GPU, 1, nstep + size_1D - Imax);
        }
    }
    void coaxUpdate(int *time)
    {
        int i = time[0];
        int mode = time[1];
        int sourcemode = time[2];
        if (mode == 1) ////adjoint
        {
            coaxUpdateGPU<<<1, 1, 0, stream>>>(PortSignal_GPU, Ex_GPU, Ey_GPU, Hx_GPU, Hy_GPU, source_adjoint_GPU, i, coax_ind[0], Imax, Jmax, delX, Z_c, depth_coax0, depth_coax1, width_coax0, width_coax1, sourcemode);
        }
        else if (mode == 0) ////forward
        {
            coaxUpdateGPU<<<1, 1, 0, stream>>>(PortSignal_GPU, Ex_GPU, Ey_GPU, Hx_GPU, Hy_GPU, source_GPU, i, coax_ind[0], Imax, Jmax, delX, Z_c, depth_coax0, depth_coax1, width_coax0, width_coax1, sourcemode);
        }
        else if (mode == -1) ////forward with reduction
        {
            PortCurrentGPU<NumThreads><<<1, NumThreads, 0, stream>>>(Hx_GPU, Hy_GPU, coax_ind[0], Imax, Jmax, delX, depth_coax0, depth_coax1, width_coax0, width_coax1, current);
            PortEGPU<<<1, NumThreads, 0, stream>>>(PortSignal_GPU, Ex_GPU, Ey_GPU, current, source, i, coax_ind[0], Imax, Jmax, delX, Z_c, depth_coax0, depth_coax1, width_coax0, width_coax1, 1);
        }
    }
    void recordForwardField(int *time)
    {
        int i = time[0];
        GPU_Read_Ant_Field<<<DesignX *(DesignY + 1) / NumThreads + 1, NumThreads, 0, stream>>>(E1_ant_GPU, Ex_GPU, DesignX, DesignY + 1, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z, Imax, Jmax, i);
        GPU_Read_Ant_Field<<<(DesignX + 1) * DesignY / NumThreads + 1, NumThreads, 0, stream>>>(E2_ant_GPU, Ey_GPU, DesignX + 1, DesignY, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z, Imax, Jmax, i);
        if (NumML > 1)
        {
            for (size_t ind = 1; ind < NumML; ind++)
            {
                const size_t E1ind = ind * max_TimeSteps;
                GPU_Read_Ant_Field<<<DesignX *(DesignY + 1) / NumThreads + 1, NumThreads, 0, stream>>>(E1_ant_GPU, Ex_GPU, DesignX, DesignY + 1, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z + Ind_ML[ind], Imax, Jmax, i + E1ind);
                const size_t E2ind = ind * max_TimeSteps;
                GPU_Read_Ant_Field<<<(DesignX + 1) * DesignY / NumThreads + 1, NumThreads, 0, stream>>>(E2_ant_GPU, Ey_GPU, DesignX + 1, DesignY, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z + Ind_ML[ind], Imax, Jmax, i + E2ind);
            }
        }
    }

    void recordGPUField(int *input)
    {
        int i = input[0];    /////time
        int mode = input[1]; ////mode
        if (mode == 0)       ////receiving mode
        {
            GPU_Read_Ant_Field<<<DesignX *(DesignY + 1) / NumThreads + 1, NumThreads, 0, stream>>>(E1_ant_GPU, Ex_GPU, DesignX, DesignY + 1, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z, Imax, Jmax, i);
            GPU_Read_Ant_Field<<<(DesignX + 1) * DesignY / NumThreads + 1, NumThreads, 0, stream>>>(E2_ant_GPU, Ey_GPU, DesignX + 1, DesignY, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z, Imax, Jmax, i);
        }
        else if (mode == 1) ////transimission mode
        {
            GPU_Read_Ant_Field<<<DesignX *(DesignY + 1) / NumThreads + 1, NumThreads, 0, stream>>>(E1_ant_GPU_2, Ex_GPU, DesignX, DesignY + 1, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z, Imax, Jmax, i);
            GPU_Read_Ant_Field<<<(DesignX + 1) * DesignY / NumThreads + 1, NumThreads, 0, stream>>>(E2_ant_GPU_2, Ey_GPU, DesignX + 1, DesignY, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z, Imax, Jmax, i);
        }
        else if (mode == 2) ////adjoint mode
        {
            GPU_Read_Ant_Field<<<DesignX *(DesignY + 1) / NumThreads + 1, NumThreads, 0, stream>>>(E1_ant_AD_GPU, Ex_GPU, DesignX, DesignY + 1, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z, Imax, Jmax, i);
            GPU_Read_Ant_Field<<<(DesignX + 1) * DesignY / NumThreads + 1, NumThreads, 0, stream>>>(E2_ant_AD_GPU, Ey_GPU, DesignX + 1, DesignY, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z, Imax, Jmax, i);
        }
    }

    void computeGradientGPU(int *input)
    {
        size_t newblock = (max_TimeSteps + 2 * NumThreads - 1) / (2 * NumThreads);
        int mode = input[0];
        int size1 = DesignX * (DesignY + 1);
        int size2 = (DesignX + 1) * DesignY;
        if (mode == 0) ////receiving mode
        {
            for (size_t i = 0; i < size1; i++)
            {
                computeGradientGPU_Mul<<<(max_TimeSteps + NumThreads - 1) / NumThreads, NumThreads, 0, stream>>>(E1_ant_GPU, E1_ant_AD_GPU, G_temp, i, size1, max_TimeSteps);
                sum_reductions<NumThreads><<<newblock, NumThreads, 0, stream>>>(G_temp, G_output, max_TimeSteps);
                sum_reduce<NumThreads><<<1, NumThreads, 0, stream>>>(G_output, G1_ant_GPU, i, newblock);
            }
            for (size_t i = 0; i < size2; i++)
            {
                computeGradientGPU_Mul<<<(max_TimeSteps + NumThreads - 1) / NumThreads, NumThreads, 0, stream>>>(E2_ant_GPU, E2_ant_AD_GPU, G_temp, i, size2, max_TimeSteps);
                sum_reductions<NumThreads><<<newblock, NumThreads, 0, stream>>>(G_temp, G_output, max_TimeSteps);
                sum_reduce<NumThreads><<<1, NumThreads, 0, stream>>>(G_output, G2_ant_GPU, i, newblock);
            }
        }
        else if (mode == 1) ////transmitting mode
        {
            for (size_t i = 0; i < size1; i++)
            {
                computeGradientGPU_Mul<<<(max_TimeSteps + NumThreads - 1) / NumThreads, NumThreads, 0, stream>>>(E1_ant_GPU_2, E1_ant_AD_GPU, G_temp, i, size1, max_TimeSteps);
                sum_reductions<NumThreads><<<newblock, NumThreads, 0, stream>>>(G_temp, G_output, max_TimeSteps);
                sum_reduce<NumThreads><<<1, NumThreads, 0, stream>>>(G_output, G1_ant_GPU, i, newblock);
            }

            for (size_t i = 0; i < size2; i++)
            {
                computeGradientGPU_Mul<<<(max_TimeSteps + NumThreads - 1) / NumThreads, NumThreads, 0, stream>>>(E2_ant_GPU_2, E2_ant_AD_GPU, G_temp, i, size2, max_TimeSteps);
                sum_reductions<NumThreads><<<newblock, NumThreads, 0, stream>>>(G_temp, G_output, max_TimeSteps);
                sum_reduce<NumThreads><<<1, NumThreads, 0, stream>>>(G_output, G2_ant_GPU, i, newblock);
            }
        }
        cudaMemcpyAsync(G1_ant, G1_ant_GPU, sizeof(double) * size1, cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(G2_ant, G2_ant_GPU, sizeof(double) * size2, cudaMemcpyDeviceToHost, stream);
        cudaMemsetAsync(G1_ant_GPU, 0, sizeof(double) * size1, stream);
        cudaMemsetAsync(G2_ant_GPU, 0, sizeof(double) * size2, stream);
        cudaMemsetAsync(G_output, 0, sizeof(double) * max_TimeSteps, stream);
        cudaMemsetAsync(G_temp, 0, sizeof(double) * max_TimeSteps, stream);
    }

    void impulse_response(real *data, int *num)
    {
        init_impulse = 1;
        ht = new real[num[0] * max_TimeSteps];
        for (size_t i = 0; i < num[0] * max_TimeSteps; i++)
            ht[i] = data[i];
        cudaMalloc((void **)&ht_GPU, num[0] * max_TimeSteps * real_size);
        cudaMemcpyAsync(ht_GPU, ht, num[0] * max_TimeSteps * real_size, cudaMemcpyHostToDevice, stream);
        cudaMalloc((void **)&W_plus, num[0] * max_TimeSteps * real_size);
        cudaMemcpyAsync(W_plus, ht, num[0] * max_TimeSteps * real_size, cudaMemcpyHostToDevice, stream);
    }
    void updateAntCond(real *sigma_antx, real *sigma_anty)
    {
        for (size_t i = PosDesign[0].x; i <= PosDesign[0].x + DesignX - 1; i++)
            for (size_t j = PosDesign[0].y; j <= PosDesign[0].y + DesignY; j++)
            {
                sigma_sx[i + j * Imax + PosDesign[0].z * (Imax * Jmax)] = sigma_antx[i - PosDesign[0].x + (j - PosDesign[0].y) * DesignX];
            }
        for (size_t i = PosDesign[0].x; i <= PosDesign[0].x + DesignX; i++)
            for (size_t j = PosDesign[0].y; j <= PosDesign[0].y + DesignY - 1; j++)
            {
                sigma_sy[i + j * Imax + PosDesign[0].z * (Imax * Jmax)] = sigma_anty[i - PosDesign[0].x + (j - PosDesign[0].y) * (DesignX + 1)];
            }
        updateCuda();
    }

    void updateAntCondML(real *sigma_antx, real *sigma_anty, int *k)
    {

        for (size_t i = PosDesign[0].x; i <= PosDesign[0].x + DesignX - 1; i++)
            for (size_t j = PosDesign[0].y; j <= PosDesign[0].y + DesignY; j++)
            {
                sigma_sx[i + j * Imax + (PosDesign[0].z + k[0]) * (Imax * Jmax)] = sigma_antx[i - PosDesign[0].x + (j - PosDesign[0].y) * DesignX];
            }
        for (size_t i = PosDesign[0].x; i <= PosDesign[0].x + DesignX; i++)
            for (size_t j = PosDesign[0].y; j <= PosDesign[0].y + DesignY - 1; j++)
            {
                sigma_sy[i + j * Imax + (PosDesign[0].z + k[0]) * (Imax * Jmax)] = sigma_anty[i - PosDesign[0].x + (j - PosDesign[0].y) * (DesignX + 1)];
            }

        updateCuda();
    }

    void gradient_calculation_all(int *flag)
    {
        if (flag[0] == 1)
        {
            cudaMemcpyAsync(E1_ant, E1_ant_GPU, size_t_E1_ant, cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(E2_ant, E2_ant_GPU, size_t_E2_ant, cudaMemcpyDeviceToHost, stream);
        }
        else if (flag[0] == 2)
        {
            cudaMemcpyAsync(E1_ant_AD, E1_ant_GPU, size_t_E1_ant, cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(E2_ant_AD, E2_ant_GPU, size_t_E2_ant, cudaMemcpyDeviceToHost, stream);
        }
        else if (flag[0] == 4)
        {
            cudaMemcpyAsync(E1_ant_temp, E1_ant_GPU, size_t_E1_ant, cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(E2_ant_temp, E2_ant_GPU, size_t_E2_ant, cudaMemcpyDeviceToHost, stream);
        }
        else if (flag[0] == 3)
        {
            size_t size = DesignX * (DesignY + 1);
            computeG_ant(E1_ant, E1_ant_AD, G1_ant, size, max_TimeSteps);
            // for(int ti=0;ti<size;ti++){
            //     G1_ant[ti]=0;
            //     for(int i=0;i<max_TimeSteps-1;i++){
            //         G1_ant[ti]=G1_ant[ti]+E1_ant[(max_TimeSteps-i-1-1)*size+ti]*(E1_ant_AD[ti+i*size]+E1_ant_AD[ti+(i+1)*size])/2.0;
            //     }
            // }
            size = (DesignX + 1) * DesignY;
            computeG_ant(E2_ant, E2_ant_AD, G2_ant, size, max_TimeSteps);

            // for(int ti=0;ti<size;ti++){
            //     G2_ant[ti]=0;
            //     for(int i=0;i<max_TimeSteps-1;i++){
            //         G2_ant[ti]=G2_ant[ti]+E2_ant[(max_TimeSteps-i-1-1)*size+ti]*(E2_ant_AD[ti+i*size]+E2_ant_AD[ti+(i+1)*size])/2.0;
            //     }
            // }
        }
        else if (flag[0] == 5)
        {
            size_t size = DesignX * (DesignY + 1);
            computeG_ant(E1_ant_temp, E1_ant_AD, G1_ant, size, max_TimeSteps);

            // for(int ti=0;ti<size;ti++){
            //     G1_ant[ti]=0;
            //     for(int i=0;i<max_TimeSteps-1;i++){
            //         G1_ant[ti]=G1_ant[ti]+E1_ant_temp[(max_TimeSteps-i-1-1)*size+ti]*(E1_ant_AD[ti+i*size]+E1_ant_AD[ti+(i+1)*size])/2.0;
            //     }
            // }
            size = (DesignX + 1) * DesignY;
            computeG_ant(E2_ant_temp, E2_ant_AD, G2_ant, size, max_TimeSteps);

            // for(int ti=0;ti<size;ti++){
            //     G2_ant[ti]=0;
            //     for(int i=0;i<max_TimeSteps-1;i++){
            //         G2_ant[ti]=G2_ant[ti]+E2_ant_temp[(max_TimeSteps-i-1-1)*size+ti]*(E2_ant_AD[ti+i*size]+E2_ant_AD[ti+(i+1)*size])/2.0;
            //     }
            // }
        }
    }
    void InitAdjoint(real *sourceIn)
    {
        cudaMemcpy(PortSignal_GPU, sourceIn, max_TimeSteps * real_size, cudaMemcpyHostToDevice);
        AdjointSource<<<(max_TimeSteps + NumThreads - 1) / NumThreads, NumThreads, 0>>>(source_adjoint_GPU, PortSignal_GPU, max_TimeSteps);
    }
    void updateH_All()
    {
        if (init_mu == 1)
        {
            GPU_updateH_Full<<<NumBlocks, NumThreads, 0, stream>>>(Hx_GPU, Hy_GPU, Hz_GPU, Ex_GPU, Ey_GPU, Ez_GPU, Chxe_GPU, Chye_GPU, Chze_GPU, Imax, Jmax, Kmax, IJ_Size);
        }
        else
        {
            GPU_updateH<<<NumBlocks, NumThreads, 0, stream>>>(Hx_GPU, Hy_GPU, Hz_GPU, Ex_GPU, Ey_GPU, Ez_GPU, Chxh, Chxe, Imax, Jmax, Kmax, IJ_Size);
        }
    }
    void updateE_All()
    {
        GPU_updateE<<<NumBlocks, NumThreads, 0, stream>>>(Ex_GPU, Ey_GPU, Ez_GPU, Cexe_GPU, Cexh_GPU, Ceye_GPU, Ceyh_GPU, Ceze_GPU, Cezh_GPU, Hx_GPU, Hy_GPU, Hz_GPU, Imax, Jmax, Kmax, IJ_Size);
    }
    void updateH_PML()
    {
        GPU_updateHPmlY<<<gridsize_y, NumThreads, 0, stream>>>(Hx_GPU, Hz_GPU, Ex_GPU, Ez_GPU, psi_Hxy_GPU, psi_Hzy_GPU, b_Hx_GPU, c_Hx_GPU, pml_size, Che, delX, Imax, Jmax, Kmax, IJ_Size);
        GPU_updateHPmlX<<<gridsize_x, NumThreads, 0, stream>>>(Hy_GPU, Hz_GPU, Ey_GPU, Ez_GPU, psi_Hyx_GPU, psi_Hzx_GPU, b_Hx_GPU, c_Hx_GPU, pml_size, Che, delX, Imax, Jmax, Kmax, IJ_Size);
        GPU_updateHPmlZ<<<gridsize_z, NumThreads, 0, stream>>>(Hx_GPU, Hy_GPU, Ex_GPU, Ey_GPU, psi_Hxz_GPU, psi_Hyz_GPU, b_Hx_GPU, c_Hx_GPU, pml_size, Che, delX, Imax, Jmax, Kmax, IJ_Size);
    }
    void updateE_PML()
    {
        GPU_updateEPmlY<<<gridsize_y, NumThreads, 0, stream>>>(Hx_GPU, Hz_GPU, Ex_GPU, Ez_GPU, psi_Exy_GPU, psi_Ezy_GPU, b_Ex_GPU, c_Ex_GPU, Cexh_GPU, Cezh_GPU, delX, pml_size, Imax, Jmax, Kmax, IJ_Size);
        GPU_updateEPmlZ<<<gridsize_z, NumThreads, 0, stream>>>(Hx_GPU, Hy_GPU, Ex_GPU, Ey_GPU, psi_Exz_GPU, psi_Eyz_GPU, b_Ex_GPU, c_Ex_GPU, Cexh_GPU, Ceyh_GPU, delX, pml_size, Imax, Jmax, Kmax, IJ_Size);
        GPU_updateEPmlX<<<gridsize_x, NumThreads, 0, stream>>>(Hy_GPU, Hz_GPU, Ey_GPU, Ez_GPU, psi_Eyx_GPU, psi_Ezx_GPU, b_Ex_GPU, c_Ex_GPU, Ceyh_GPU, Cezh_GPU, delX, pml_size, Imax, Jmax, Kmax, IJ_Size);
    }

    ////////////Plance wave/////////////////////////////////
    void updateH_PW_XY(real *Ez_1D_GPU, real c, size_t time)
    {
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid_HxPWX_Y((Imax / block_size) + 1, (Jmax / block_size) + 1);
        GPU_updateHx_PlaneWaveX_Y<<<dimGrid_HxPWX_Y, dimBlock, 0, stream>>>(Hx_GPU, Ez_1D_GPU, c, Chxe, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);

        dim3 dimGrid_Hz_PWX_Y((Jmax / block_size) + 1, (Kmax / block_size) + 1);
        GPU_updateHz_PlaneWaveX_Y<<<dimGrid_Hz_PWX_Y, dimBlock, 0, stream>>>(Hz_GPU, Ez_1D_GPU, c, Chze, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);
    }
    void updateH_PW_XZ(real *Ez_1D_GPU, real c, size_t time)
    {
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid_HxPWX_Z((Imax / block_size) + 1, (Kmax / block_size) + 1);
        GPU_updateHx_PlaneWaveX_Z<<<dimGrid_HxPWX_Z, dimBlock, 0, stream>>>(Hx_GPU, Ez_1D_GPU, c, Chxe, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);
        dim3 dimGrid_Hy_PWX_Z((Jmax / block_size) + 1, (Kmax / block_size) + 1);
        GPU_updateHy_PlaneWaveX_Z<<<dimGrid_Hy_PWX_Z, dimBlock, 0, stream>>>(Hy_GPU, Ez_1D_GPU, c, Chye, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);
    }
    void updateH_PW_YX(real *Ez_1D_GPU, real c)
    {
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid_Hy_PWY_X((Imax / block_size) + 1, (Jmax / block_size) + 1);
        GPU_updateHy_PlaneWaveY_X<<<dimGrid_Hy_PWY_X, dimBlock, 0, stream>>>(Hy_GPU, Ez_1D_GPU, c, Chye, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1);

        dim3 dimGrid_Hz_PWY_X((Imax / block_size) + 1, (Kmax / block_size) + 1);
        GPU_updateHz_PlaneWaveY_X<<<dimGrid_Hz_PWY_X, dimBlock, 0, stream>>>(Hz_GPU, Ez_1D_GPU, c, Chze, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1);
    }
    void updateH_PW_YZ(real *Ez_1D_GPU, real c)
    {
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid_HxPWY_Z((Imax / block_size) + 1, (Kmax / block_size) + 1);
        GPU_updateHx_PlaneWaveY_Z<<<dimGrid_HxPWY_Z, dimBlock, 0, stream>>>(Hx_GPU, Ez_1D_GPU, c, Chxe, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1);

        dim3 dimGrid_HyPWY_Z((Jmax / block_size) + 1, (Kmax / block_size) + 1);
        GPU_updateHy_PlaneWaveY_Z<<<dimGrid_HyPWY_Z, dimBlock, 0, stream>>>(Hy_GPU, Ez_1D_GPU, c, Chye, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1);
    }
    void updateH_PW_ZX(real *Ez_1D_GPU, real c, size_t time)
    {
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid_Hy_PWZ_X((Imax / block_size) + 1, (Jmax / block_size) + 1);
        GPU_updateHy_PlaneWaveZ_X<<<dimGrid_Hy_PWZ_X, dimBlock, 0, stream>>>(Hy_GPU, Ez_1D_GPU, c, Chye, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);

        dim3 dimGrid_Hz_PWZ_X((Imax / block_size) + 1, (Kmax / block_size) + 1);
        GPU_updateHz_PlaneWaveZ_X<<<dimGrid_Hz_PWZ_X, dimBlock, 0, stream>>>(Hz_GPU, Ez_1D_GPU, c, Chze, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);
    }
    void updateH_PW_ZY(real *Ez_1D_GPU, real c, size_t time)
    {
        size_t nbs = ((i1 - i0 + 1) * (j1 - 1 - j0 + 1) + NumThreads - 1) / NumThreads;
        // dim3 dimBlock(block_size, block_size);
        // dim3 dimGrid_HxPWZ_Y((Imax / block_size) + 1, (Jmax / block_size) + 1);
        // GPU_updateHx_PlaneWaveZ_Y<<<dimGrid_HxPWZ_Y, dimBlock , 0, stream >>>( Hx_GPU, Ez_1D_GPU, c, Chxe,  Imax,  Jmax,  Kmax,  i0, i1,j0,  j1,  k0,  k1, time);
        GPU_updateHx_PlaneWaveZ_Y<<<nbs, NumThreads, 0, stream>>>(Hx_GPU, Ez_1D_GPU, c, Chxe, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);
        nbs = ((j1 - 1 - j0 + 1) * (k1 - k0 + 1) + NumThreads - 1) / NumThreads;
        // dim3 dimGrid_HzPWZ_Y((Jmax / block_size) + 1, (Kmax / block_size) + 1);
        // GPU_updateHz_PlaneWaveZ_Y<<<dimGrid_HzPWZ_Y, dimBlock , 0, stream >>>( Hz_GPU, Ez_1D_GPU, c, Chze,  Imax,  Jmax,  Kmax,  i0, i1,j0,  j1,  k0,  k1,time);
        GPU_updateHz_PlaneWaveZ_Y<<<nbs, NumThreads, 0, stream>>>(Hz_GPU, Ez_1D_GPU, c, Chze, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);
    }
    void updateE_PW_XY(real *Hy_1D_GPU, real c, size_t time)
    {
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid_ExPWX_Y((Imax / block_size) + 1, (Kmax / block_size) + 1);
        GPU_updateEx_PlaneWaveX_Y<<<dimGrid_ExPWX_Y, dimBlock, 0, stream>>>(Ex_GPU, Hy_1D_GPU, -c, Cexh_GPU, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);
        // GPU_updateEx_PlaneWaveX_Y << <(i1 -1 - i0 + 1) * (k1 - k0 + 1) / NumThreads + 1, NumThreads >> > (Ex_GPU, Hy_1D_GPU, -c, Cexh_GPU, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);

        dim3 dimGrid_EyPWX_Y((Jmax / block_size) + 1, (Kmax / block_size) + 1);
        GPU_updateEy_PlaneWaveX_Y<<<dimGrid_EyPWX_Y, dimBlock, 0, stream>>>(Ey_GPU, Hy_1D_GPU, -c, Ceyh_GPU, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);
    }
    void updateE_PW_XZ(real *Hy_1D_GPU, real c, size_t time)
    {
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid_ExPWX_Z((Imax / block_size) + 1, (Jmax / block_size) + 1);
        GPU_updateEx_PlaneWaveX_Z<<<dimGrid_ExPWX_Z, dimBlock, 0, stream>>>(Ex_GPU, Hy_1D_GPU, c, Cexh_GPU, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);

        dim3 dimGrid_EzPWX_Z((Jmax / block_size) + 1, (Kmax / block_size) + 1);
        GPU_updateEz_PlaneWaveX_Z<<<dimGrid_EzPWX_Z, dimBlock, 0, stream>>>(Ez_GPU, Hy_1D_GPU, c, Cezh_GPU, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);
    }
    void updateE_PW_YX(real *Hy_1D_GPU, real c)
    {
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid_ExPWY_X((Imax / block_size) + 1, (Kmax / block_size) + 1);
        GPU_updateEx_PlaneWaveY_X<<<dimGrid_ExPWY_X, dimBlock, 0, stream>>>(Ex_GPU, Hy_1D_GPU, c, Cexh_GPU, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1);

        dim3 dimGrid_EyPWY_X((Jmax / block_size) + 1, (Kmax / block_size) + 1);
        GPU_updateEy_PlaneWaveY_X<<<dimGrid_EyPWY_X, dimBlock, 0, stream>>>(Ey_GPU, Hy_1D_GPU, c, Ceyh_GPU, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1);
    }
    void updateE_PW_YZ(real *Hy_1D_GPU, real c)
    {
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid_EyPWY_Z((Imax / block_size) + 1, (Jmax / block_size) + 1);
        GPU_updateEy_PlaneWaveY_Z<<<dimGrid_EyPWY_Z, dimBlock, 0, stream>>>(Ey_GPU, Hy_1D_GPU, -c, Ceyh_GPU, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1);

        dim3 dimGrid_EzPWY_Z((Imax / block_size) + 1, (Kmax / block_size) + 1);
        GPU_updateEz_PlaneWaveY_Z<<<dimGrid_EzPWY_Z, dimBlock, 0, stream>>>(Ez_GPU, Hy_1D_GPU, -c, Cezh_GPU, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1);
    }
    void updateE_PW_ZX(real *Hy_1D_GPU, real c, size_t time)
    {
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid_ExPWZ_X((Imax / block_size) + 1, (Jmax / block_size) + 1);
        GPU_updateEx_PlaneWaveZ_X<<<dimGrid_ExPWZ_X, dimBlock, 0, stream>>>(Ex_GPU, Hy_1D_GPU, -c, Cexh_GPU, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);

        dim3 dimGrid_EzPWZ_X((Jmax / block_size) + 1, (Kmax / block_size) + 1);
        GPU_updateEz_PlaneWaveZ_X<<<dimGrid_EzPWZ_X, dimBlock, 0, stream>>>(Ez_GPU, Hy_1D_GPU, -c, Cezh_GPU, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);
    }
    void updateE_PW_ZY(real *Hy_1D_GPU, real c, size_t time)
    {
        dim3 dimBlock(block_size, block_size);
        dim3 dimGrid_EyPWZ_Y((Imax / block_size) + 1, (Jmax / block_size) + 1);
        GPU_updateEy_PlaneWaveZ_Y<<<dimGrid_EyPWZ_Y, dimBlock, 0, stream>>>(Ey_GPU, Hy_1D_GPU, c, Ceyh_GPU, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);

        dim3 dimGrid_EzPWZ_Y((Imax / block_size) + 1, (Kmax / block_size) + 1);
        GPU_updateEz_PlaneWaveZ_Y<<<dimGrid_EzPWZ_Y, dimBlock, 0, stream>>>(Ez_GPU, Hy_1D_GPU, c, Cezh_GPU, Imax, Jmax, Kmax, i0, i1, j0, j1, k0, k1, time);
    }

    /// 1D

    void initAntCond(int *ant_pos, int *dim_ant, real *sigma_init)
    {
        init_design = 1;

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        printf("Free: %.2f GB, Total: %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0), total_mem / (1024.0 * 1024.0 * 1024.0));
        PosDesign = new Antenna[numAnt];
        PosDesign[0].x = ant_pos[0];
        PosDesign[0].y = ant_pos[1];
        PosDesign[0].z = ant_pos[2];
        DesignX = dim_ant[0];
        DesignY = dim_ant[1];
        DesignZ = dim_ant[2];
        size_t_E1_ant = static_cast<size_t>(max_TimeSteps) * DesignX * (DesignY + 1) * real_size;
        size_t_E2_ant = static_cast<size_t>(max_TimeSteps) * (DesignX + 1) * DesignY * real_size;
        size_E1_ant = static_cast<size_t>(max_TimeSteps) * DesignX * (DesignY + 1);
        size_E2_ant = static_cast<size_t>(max_TimeSteps) * (DesignX + 1) * DesignY;
        printf("size_t_E1_ant: %zu\n", size_t_E1_ant);
        printf("size_t_E2_ant: %zu\n", size_t_E2_ant);
        printf("size_E1_ant: %zu\n", size_E1_ant);
        printf("size_E2_ant: %zu\n", size_E2_ant);
        cudaMallocHost(&E1_ant, max_TimeSteps * DesignX * (DesignY + 1) * real_size);      // Pinned memory for asynchronous copy
        cudaMallocHost(&E2_ant, max_TimeSteps * (DesignX + 1) * DesignY * real_size);      // Pinned memory for asynchronous copy
        cudaMallocHost(&E1_ant_AD, max_TimeSteps * DesignX * (DesignY + 1) * real_size);   // Pinned memory for asynchronous copy
        cudaMallocHost(&E2_ant_AD, max_TimeSteps * (DesignX + 1) * DesignY * real_size);   // Pinned memory for asynchronous copy
        cudaMallocHost(&E1_ant_temp, max_TimeSteps * DesignX * (DesignY + 1) * real_size); // Pinned memory for asynchronous copy
        cudaMallocHost(&E2_ant_temp, max_TimeSteps * (DesignX + 1) * DesignY * real_size); // Pinned memory for asynchronous copy
        cudaMallocHost(&G1_ant, DesignX * (DesignY + 1) * sizeof(double));                 // Pinned memory for asynchronous copy
        cudaMallocHost(&G2_ant, (DesignX + 1) * DesignY * sizeof(double));                 // Pinned memory for asynchronous copy
        std::fill(G1_ant, G1_ant + DesignX * (DesignY + 1), 0.0);
        std::fill(G2_ant, G2_ant + (DesignX + 1) * DesignY, 0.0);
        std::fill(E1_ant, E1_ant + DesignX * (DesignY + 1) * max_TimeSteps, 0.0);
        std::fill(E1_ant_AD, E1_ant_AD + DesignX * (DesignY + 1) * max_TimeSteps, 0.0);
        std::fill(E1_ant_temp, E1_ant_temp + (DesignX + 1) * DesignY * max_TimeSteps, 0.0);
        std::fill(E2_ant, E2_ant + (DesignX + 1) * DesignY * max_TimeSteps, 0.0);
        std::fill(E2_ant_AD, E2_ant_AD + (DesignX + 1) * DesignY * max_TimeSteps, 0.0);
        std::fill(E2_ant_temp, E2_ant_temp + (DesignX + 1) * DesignY * max_TimeSteps, 0.0);

        for (size_t i = PosDesign[0].x; i <= PosDesign[0].x + DesignX - 1; i++)
        {
            for (size_t j = PosDesign[0].y; j <= PosDesign[0].y + DesignY; j++)
            {
                sigma_sx[i + j * Imax + PosDesign[0].z * (Imax * Jmax)] = sigma_init[0];
            }
        }
        for (size_t i = PosDesign[0].x; i <= PosDesign[0].x + DesignX; i++)
        {
            for (size_t j = PosDesign[0].y; j <= PosDesign[0].y + DesignY - 1; j++)
            {
                sigma_sy[i + j * Imax + PosDesign[0].z * (Imax * Jmax)] = sigma_init[0];
            }
        }
        cudaMalloc((void **)&E1_ant_GPU, max_TimeSteps * DesignX * (DesignY + 1) * real_size);
        cudaMemsetAsync(E1_ant_GPU, 0, max_TimeSteps * DesignX * (DesignY + 1) * real_size, stream);
        cudaMalloc((void **)&E2_ant_GPU, max_TimeSteps * (DesignX + 1) * DesignY * real_size);
        cudaMemsetAsync(E2_ant_GPU, 0, max_TimeSteps * (DesignX + 1) * DesignY * real_size, stream);
    }

    void initAntCondML(int *ant_pos, int *dim_ant, real *sigma_init, int *Ind)
    {
        init_design = 1;

        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);

        printf("Free: %.2f GB, Total: %.2f GB\n", free_mem / (1024.0 * 1024.0 * 1024.0), total_mem / (1024.0 * 1024.0 * 1024.0));

        PosDesign = new Antenna[numAnt];
        PosDesign[0].x = ant_pos[0];
        PosDesign[0].y = ant_pos[1];
        PosDesign[0].z = ant_pos[2];
        DesignX = dim_ant[0];
        DesignY = dim_ant[1];
        DesignZ = dim_ant[2];
        NumML = dim_ant[3];
        Ind_ML = new size_t[NumML];

        for (size_t i = 0; i < NumML; i++)
        {
            Ind_ML[i] = Ind[i];
        }
        size_t_E1_ant = static_cast<size_t>(max_TimeSteps) * DesignX * (DesignY + 1) * NumML * real_size;
        size_t_E2_ant = static_cast<size_t>(max_TimeSteps) * (DesignX + 1) * DesignY * NumML * real_size;
        size_E1_ant = static_cast<size_t>(max_TimeSteps) * DesignX * (DesignY + 1) * NumML;
        size_E2_ant = static_cast<size_t>(max_TimeSteps) * (DesignX + 1) * DesignY * NumML;
        printf("size_t_E1_ant: %.2f GB\n", size_t_E1_ant / (1024.0 * 1024.0 * 1024.0));
        printf("size_t_E2_ant: %.2f GB\n", size_t_E2_ant / (1024.0 * 1024.0 * 1024.0));
        printf("size_E1_ant: %.2f GB\n", size_E1_ant / (1024.0 * 1024.0 * 1024.0));
        printf("size_E2_ant: %.2f GB\n", size_E2_ant / (1024.0 * 1024.0 * 1024.0));
        cudaMallocHost(&E1_ant, size_t_E1_ant);                                    // Pinned memory for asynchronous copy
        cudaMallocHost(&E2_ant, size_t_E2_ant);                                    // Pinned memory for asynchronous copy
        cudaMallocHost(&E1_ant_AD, size_t_E1_ant);                                 // Pinned memory for asynchronous copy
        cudaMallocHost(&E2_ant_AD, size_t_E2_ant);                                 // Pinned memory for asynchronous copy
        cudaMallocHost(&E1_ant_temp, size_t_E1_ant);                               // Pinned memory for asynchronous copy
        cudaMallocHost(&E2_ant_temp, size_t_E2_ant);                               // Pinned memory for asynchronous copy
        cudaMallocHost(&G1_ant, DesignX * (DesignY + 1) * NumML * sizeof(double)); // Pinned memory for asynchronous copy
        cudaMallocHost(&G2_ant, (DesignX + 1) * DesignY * NumML * sizeof(double)); // Pinned memory for asynchronous copy
        std::fill(G1_ant, G1_ant + DesignX * (DesignY + 1) * NumML, 0.0);
        std::fill(G2_ant, G2_ant + (DesignX + 1) * DesignY * NumML, 0.0);
        std::fill(E1_ant, E1_ant + size_E1_ant, 0.0);
        std::fill(E1_ant_AD, E1_ant_AD + size_E1_ant, 0.0);
        std::fill(E1_ant_temp, E1_ant_temp + size_E1_ant, 0.0);
        std::fill(E2_ant, E2_ant + size_E2_ant, 0.0);
        std::fill(E2_ant_AD, E2_ant_AD + size_E2_ant, 0.0);
        std::fill(E2_ant_temp, E2_ant_temp + size_E2_ant, 0.0);
        for (size_t k = 0; k < NumML; k++)
        {
            for (size_t i = PosDesign[0].x; i <= PosDesign[0].x + DesignX - 1; i++)
            {
                for (size_t j = PosDesign[0].y; j <= PosDesign[0].y + DesignY; j++)
                {
                    sigma_sx[i + j * Imax + (PosDesign[0].z + Ind_ML[k]) * (Imax * Jmax)] = sigma_init[0];
                }
            }
            for (size_t i = PosDesign[0].x; i <= PosDesign[0].x + DesignX; i++)
            {
                for (size_t j = PosDesign[0].y; j <= PosDesign[0].y + DesignY - 1; j++)
                {
                    sigma_sy[i + j * Imax + (PosDesign[0].z + Ind_ML[k]) * (Imax * Jmax)] = sigma_init[0];
                }
            }
        }
        cudaMalloc((void **)&E1_ant_GPU, size_t_E1_ant);
        cudaMemsetAsync(E1_ant_GPU, 0, size_t_E1_ant, stream);
        cudaMalloc((void **)&E2_ant_GPU, size_t_E2_ant);
        cudaMemsetAsync(E2_ant_GPU, 0, size_t_E2_ant, stream);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }
    }

    void initDesignGPU()
    {
        init_design_GPU = 1;
        cudaMalloc((void **)&E1_ant_AD_GPU, max_TimeSteps * DesignX * (DesignY + 1) * real_size);
        cudaMemsetAsync(E1_ant_AD_GPU, 0, max_TimeSteps * DesignX * (DesignY + 1) * real_size, stream);
        cudaMalloc((void **)&E2_ant_AD_GPU, max_TimeSteps * (DesignX + 1) * DesignY * real_size);
        cudaMemsetAsync(E2_ant_AD_GPU, 0, max_TimeSteps * (DesignX + 1) * DesignY * real_size, stream);
        cudaMalloc((void **)&E1_ant_GPU_2, max_TimeSteps * DesignX * (DesignY + 1) * real_size);
        cudaMemsetAsync(E1_ant_GPU_2, 0, max_TimeSteps * DesignX * (DesignY + 1) * real_size, stream);
        cudaMalloc((void **)&E2_ant_GPU_2, max_TimeSteps * (DesignX + 1) * DesignY * real_size);
        cudaMemsetAsync(E2_ant_GPU_2, 0, max_TimeSteps * (DesignX + 1) * DesignY * real_size, stream);
        cudaMalloc((void **)&G1_ant_GPU, DesignX * (DesignY + 1) * real_size);
        cudaMemsetAsync(G1_ant_GPU, 0, DesignX * (DesignY + 1) * real_size, stream);
        cudaMalloc((void **)&G2_ant_GPU, (DesignX + 1) * DesignY * real_size);
        cudaMemsetAsync(G2_ant_GPU, 0, (DesignX + 1) * DesignY * real_size, stream);
        size_t newblock = (max_TimeSteps + 2 * NumThreads - 1) / (2 * NumThreads);
        cudaMalloc(&G_temp, sizeof(double) * max_TimeSteps);
        cudaMemsetAsync(G_temp, 0, sizeof(double) * max_TimeSteps);
        cudaMalloc(&G_output, sizeof(double) * newblock);
        cudaMemsetAsync(G_output, 0, sizeof(double) * newblock);
    }

    void outputVis(real *Data, int *flag)
    {
        if (flag[0] == 0)
        {
            for (size_t i = 0; i < IJK_Size; i++)
                Data[i] = sigma_sx[i];
        }
        if (flag[0] == -1)
        {
            for (size_t i = 0; i < IJK_Size; i++)
                Data[i] = sigma_sy[i];
        }
        if (flag[0] == -2)
        {
            for (size_t i = 0; i < IJK_Size; i++)
                Data[i] = sigma_sz[i];
        }
        if (flag[0] == 1)
        {
            cudaMemcpyAsync(Ex, Ex_GPU, (IJK_Size)*real_size, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            for (size_t i = 0; i < IJK_Size; i++)
            {
                Data[i] = Ex[i];
            }
        }
        if (flag[0] == 2)
        {
            cudaMemcpyAsync(Ey, Ey_GPU, (IJK_Size)*real_size, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            for (size_t i = 0; i < IJK_Size; i++)
            {
                Data[i] = Ey[i];
            }
        }
        if (flag[0] == 3)
        {
            cudaMemcpyAsync(Ez, Ez_GPU, (IJK_Size)*real_size, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            for (size_t i = 0; i < IJK_Size; i++)
            {
                Data[i] = Ez[i];
            }
        }
        if (flag[0] == 4)
        {

            cudaMemcpyAsync(E1_ant, E1_ant_GPU, size_t_E1_ant, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            for (size_t i = 0; i < size_E1_ant; i++)
            {
                Data[i] = E1_ant[i];
            }
        }
        if (flag[0] == 5)
        {
            cudaMemcpyAsync(E2_ant, E2_ant_GPU, size_t_E2_ant, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            for (size_t i = 0; i < size_E2_ant; i++)
            {
                Data[i] = E2_ant[i];
            }
        }
        if (flag[0] == 6)
        {
            if (NumML > 1)
            {
                for (size_t i = 0; i < NumML * DesignX * (DesignY + 1); i++)
                    Data[i] = G1_ant[i];
            }
            else
            {
                for (size_t i = 0; i < DesignX * (DesignY + 1); i++)
                    Data[i] = G1_ant[i];
            }
        }
        if (flag[0] == 7)
        {
            if (NumML > 1)
            {
                for (size_t i = 0; i < NumML * (DesignX + 1) * DesignY; i++)
                    Data[i] = G2_ant[i];
            }
            else
            {
                for (size_t i = 0; i < (DesignX + 1) * DesignY; i++)
                    Data[i] = G2_ant[i];
            }
        }
        if (flag[0] == 11)
        {
            cudaMemcpyAsync(Ez_1D, Ez_1D_GPU, max_TimeSteps * size_1D * real_size, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            for (size_t i = 0; i < max_TimeSteps * size_1D; i++)
            {
                Data[i] = Ez_1D[i];
            }
        }
        if (flag[0] == 12)
        {
            cudaMemcpyAsync(Ez_1D, E_inv_1D_GPU, (size_1D)*real_size, cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            for (size_t i = 0; i < size_1D; i++)
            {
                Data[i] = Ez_1D[i];
            }
        }
        if (flag[0] == 13)
        {
            cudaMemcpy(PortSignal, source_adjoint_GPU, (numAnt * max_TimeSteps) * real_size, cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(stream);
            for (size_t i = 0; i < numAnt * max_TimeSteps; i++)
            {
                Data[i] = PortSignal[i];
            }
        }
    }
    void outputVisDouble(double *Data, int flag)
    {
        if (flag == 6)
        {
            // Output G1_ant
            size_t total = (NumML > 1) ? NumML * DesignX * (DesignY + 1)
                                       : DesignX * (DesignY + 1);
            for (size_t i = 0; i < total; ++i)
                Data[i] = G1_ant[i];
        }
        else if (flag == 7)
        {
            // Output G2_ant
            size_t total = (NumML > 1) ? NumML * (DesignX + 1) * DesignY
                                       : (DesignX + 1) * DesignY;
            for (size_t i = 0; i < total; ++i)
                Data[i] = G2_ant[i];
        }
    }

    void outputAnt(real *antData)
    {
        cudaMemcpyAsync(PortSignal, PortSignal_GPU, (numAnt * max_TimeSteps) * real_size, cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        for (size_t i = 0; i < numAnt * max_TimeSteps; i++)
        {
            antData[i] = PortSignal[i];
        }
    }

    void readySimulation()
    {
        init_readysimulation = 1;
        //%================FDTD parameters for E and H fields====================
        ////////////initilization of PML parameters//////////////////////////////
        /////////////////////////////////////////////////////////////////////////
        cudaMallocHost(&b_Ex_1, 2 * pml_size * real_size);
        cudaMalloc((void **)&b_Ex_GPU, 2 * pml_size * real_size);
        cudaMallocHost(&b_Ex_2, pml_size * real_size);
        cudaMallocHost(&c_Ex_2, pml_size * real_size);
        cudaMallocHost(&c_Ex_1, 2 * pml_size * real_size);
        cudaMalloc((void **)&c_Ex_GPU, 2 * pml_size * real_size);

        alpha_Ex_PML_1 = new real[pml_size];
        alpha_Ex_PML_2 = new real[pml_size];
        sig_Ex_PML_1 = new real[pml_size];
        sig_Ex_PML_2 = new real[pml_size];
        kappa_Ex_PML_1 = new real[pml_size];
        kappa_Ex_PML_2 = new real[pml_size];

        cudaMallocHost(&b_Hx_1, 2 * pml_size * real_size);
        cudaMalloc((void **)&b_Hx_GPU, (2 * pml_size) * real_size);
        cudaMallocHost(&b_Hx_2, pml_size * real_size);
        cudaMallocHost(&c_Hx_1, 2 * pml_size * real_size);
        cudaMalloc((void **)&c_Hx_GPU, (2 * pml_size) * real_size);
        cudaMallocHost(&c_Hx_2, pml_size * real_size);

        alpha_Hx_PML_1 = new real[pml_size];
        alpha_Hx_PML_2 = new real[pml_size];
        sig_Hx_PML_1 = new real[pml_size];
        sig_Hx_PML_2 = new real[pml_size];
        kappa_Hx_PML_1 = new real[pml_size];
        kappa_Hx_PML_2 = new real[pml_size];

        //////////Uniform Mesh/////////
        cudaMallocHost(&den_ex, IJK_Size * real_size);
        cudaMallocHost(&den_hx, IJK_Size * real_size);

        //////////Non-Uniform Mesh/////////
        if (FDTD_UNIFORM_MESH != 1)
        {
            cudaMallocHost(&den_ey, IJK_Size * real_size);
            cudaMallocHost(&den_ez, IJK_Size * real_size);
            cudaMallocHost(&den_hy, IJK_Size * real_size);
            cudaMallocHost(&den_hz, IJK_Size * real_size);
        }


        //% E-Field
        //% X-dir
        std::fill(b_Ex_1, b_Ex_1 + 2 * pml_size, 0.0);
        std::fill(b_Ex_2, b_Ex_2 + pml_size, 0.0);
        std::fill(c_Ex_2, c_Ex_2 + pml_size, 0.0);
        std::fill(c_Ex_1, c_Ex_1 + 2 * pml_size, 0.0);
        std::fill(alpha_Ex_PML_1, alpha_Ex_PML_1 + pml_size, 0.0);
        std::fill(alpha_Ex_PML_2, alpha_Ex_PML_2 + pml_size, 0.0);
        std::fill(sig_Ex_PML_1, sig_Ex_PML_1 + pml_size, 0.0);
        std::fill(sig_Ex_PML_2, sig_Ex_PML_2 + pml_size, 0.0);
        std::fill(kappa_Ex_PML_1, kappa_Ex_PML_1 + pml_size, 0.0);
        std::fill(kappa_Ex_PML_2, kappa_Ex_PML_2 + pml_size, 0.0);
        //% H-Field
        //% x-dir
        std::fill(b_Hx_1, b_Hx_1 + 2 * pml_size, 0.0);
        std::fill(b_Hx_2, b_Hx_2 + pml_size, 0.0);
        std::fill(c_Hx_1, c_Hx_1 + 2 * pml_size, 0.0);
        std::fill(c_Hx_2, c_Hx_2 + pml_size, 0.0);
        std::fill(alpha_Hx_PML_1, alpha_Hx_PML_1 + pml_size, 0.0);
        std::fill(alpha_Hx_PML_2, alpha_Hx_PML_2 + pml_size, 0.0);
        std::fill(sig_Hx_PML_1, sig_Hx_PML_1 + pml_size, 0.0);
        std::fill(sig_Hx_PML_2, sig_Hx_PML_2 + pml_size, 0.0);
        std::fill(kappa_Hx_PML_1, kappa_Hx_PML_1 + pml_size, 0.0);
        std::fill(kappa_Hx_PML_2, kappa_Hx_PML_2 + pml_size, 0.0);
        //% Denominators for the update equations

        // Uniform Mesh
        std::fill(den_ex, den_ex + IJK_Size, 0.0);
        std::fill(den_hx, den_hx + IJK_Size, 0.0);

        // Non-Uniform Mesh
        if (FDTD_UNIFORM_MESH != 1)
        {
            std::fill(den_ey, den_ey + IJK_Size, 0.0);
            std::fill(den_ez, den_ez + IJK_Size, 0.0);
            std::fill(den_hy, den_hy + IJK_Size, 0.0);
            std::fill(den_hz, den_hz + IJK_Size, 0.0);
        }


        /////////////////////////////////////////////////////////////////////////
        /////////////////////////////////////////////////////////////////////////

        ///////////////////Set CPML coefficients///////////////////////////////
        for (size_t i = 0; i < pml_size; i++)
        {
            sig_Ex_PML_1[i] = sig_max * pow(((pml_size - i - 1) / (pml_size - 1.0)), poly_m);
            alpha_Ex_PML_1[i] = alpha_max * pow(((i + 1 - 1.0) / (pml_size - 1.0)), alpha_m);
            kappa_Ex_PML_1[i] = 1.0 + (kappa_max - 1.0) * pow(((pml_size - i - 1) / (pml_size - 1.0)), poly_m);
            b_Ex_1[i] = exp(-(sig_Ex_PML_1[i] / kappa_Ex_PML_1[i] + alpha_Ex_PML_1[i]) * delT / eps0);
            if ((sig_Ex_PML_1[i] == 0.0) && (alpha_Ex_PML_1[i] == 0.0) && ((i + 1) == (pml_size - 1)))
                c_Ex_1[i] = 0.0;
            else
                c_Ex_1[i] = sig_Ex_PML_1[i] / kappa_Ex_PML_1[i] / (sig_Ex_PML_1[i] + kappa_Ex_PML_1[i] * alpha_Ex_PML_1[i]) * (b_Ex_1[i] - 1.0);
        }

        //% Hx part
        for (size_t i = 0; i < pml_size - 1; i++)
        {
            sig_Hx_PML_1[i] = sig_max * pow(((pml_size - i - 1 - 0.5) / (pml_size - 1.0)), poly_m);
            alpha_Hx_PML_1[i] = alpha_max * pow(((i + 1 - 0.5) / (pml_size - 1.0)), alpha_m);
            kappa_Hx_PML_1[i] = 1.0 + (kappa_max - 1.0) * pow(((pml_size - i - 1 - 0.5) / (pml_size - 1.0)), poly_m);
            b_Hx_1[i] = exp(-(sig_Hx_PML_1[i] / kappa_Hx_PML_1[i] + alpha_Hx_PML_1[i]) * delT / eps0);
            c_Hx_1[i] = sig_Hx_PML_1[i] / kappa_Hx_PML_1[i] / (sig_Hx_PML_1[i] + kappa_Hx_PML_1[i] * alpha_Hx_PML_1[i]) * (b_Hx_1[i] - 1.0);
        }

        //%-------------------
        //% PML2 for x dir
        //% Ex part
        for (size_t i = 0; i < pml_size; i++)
        {
            sig_Ex_PML_2[i] = sig_max * pow(((pml_size - i - 1) / (pml_size - 1.0)), poly_m);
            alpha_Ex_PML_2[i] = alpha_max * pow(((i + 1 - 1.0) / (pml_size - 1.0)), alpha_m);
            kappa_Ex_PML_2[i] = 1.0 + (kappa_max - 1.0) * pow(((pml_size - i - 1) / (pml_size - 1.0)), poly_m);
            b_Ex_2[pml_size - i - 1] = exp(-(sig_Ex_PML_2[i] / kappa_Ex_PML_2[i] + alpha_Ex_PML_2[i]) * delT / eps0);
            if ((sig_Ex_PML_2[i] == 0.0) && (alpha_Ex_PML_2[i] == 0.0) && ((i + 1) == (pml_size - 1)))
                c_Ex_2[pml_size - i - 1] = 0.0;
            else
                c_Ex_2[pml_size - i - 1] = sig_Ex_PML_2[i] / kappa_Ex_PML_2[i] / (sig_Ex_PML_2[i] + kappa_Ex_PML_2[i] * alpha_Ex_PML_2[i]) * (b_Ex_2[pml_size - i - 1] - 1.0);
        }

        //% Hx part
        for (size_t i = 0; i < pml_size - 1; i++)
        {
            sig_Hx_PML_2[i] = sig_max * pow(((pml_size - i - 1 - 0.5) / (pml_size - 1.0)), poly_m);
            alpha_Hx_PML_2[i] = alpha_max * pow(((i + 1 - 0.5) / (pml_size - 1.0)), alpha_m);
            kappa_Hx_PML_2[i] = 1.0 + (kappa_max - 1.0) * pow(((pml_size - i - 1 - 0.5) / (pml_size - 1.0)), poly_m);
            b_Hx_2[pml_size - i - 1] = exp(-(sig_Hx_PML_2[i] / kappa_Hx_PML_2[i] + alpha_Hx_PML_2[i]) * delT / eps0);
            c_Hx_2[pml_size - i - 1] = sig_Hx_PML_2[i] / kappa_Hx_PML_2[i] / (sig_Hx_PML_2[i] + kappa_Hx_PML_2[i] * alpha_Hx_PML_2[i]) * (b_Hx_2[pml_size - i - 1] - 1.0);
        }
        for (size_t i = 0; i < pml_size; i++)
        {
            b_Ex_1[pml_size + i] = b_Ex_2[i];
            c_Ex_1[pml_size + i] = c_Ex_2[i];
        }

        for (size_t i = 0; i < pml_size - 1; i++)
        {
            b_Hx_1[pml_size + i] = b_Hx_2[i + 1];
            c_Hx_1[pml_size + i] = c_Hx_2[i + 1];
        }

        cudaMemcpyAsync(b_Hx_GPU, b_Hx_1, 2 * pml_size * real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(c_Hx_GPU, c_Hx_1, 2 * pml_size * real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(b_Ex_GPU, b_Ex_1, 2 * pml_size * real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(c_Ex_GPU, c_Ex_1, 2 * pml_size * real_size, cudaMemcpyHostToDevice, stream);

        ////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////

        cudaMallocHost(&Cexe, IJK_Size * real_size);
        cudaMallocHost(&Ceye, IJK_Size * real_size);
        cudaMallocHost(&Ceze, IJK_Size * real_size);
        cudaMallocHost(&Cexh, IJK_Size * real_size);
        cudaMallocHost(&Ceyh, IJK_Size * real_size);
        cudaMallocHost(&Cezh, IJK_Size * real_size);
        std::fill(Cexe, Cexe + IJK_Size, 0.0);
        std::fill(Ceye, Ceye + IJK_Size, 0.0);
        std::fill(Ceze, Ceze + IJK_Size, 0.0);
        std::fill(Cexh, Cexh + IJK_Size, 0.0);
        std::fill(Ceyh, Ceyh + IJK_Size, 0.0);
        std::fill(Cezh, Cezh + IJK_Size, 0.0);
        cudaMalloc((void **)&Cexe_GPU, (IJK_Size)*real_size);
        cudaMalloc((void **)&Ceye_GPU, (IJK_Size)*real_size);
        cudaMalloc((void **)&Ceze_GPU, (IJK_Size)*real_size);
        cudaMalloc((void **)&Cexh_GPU, (IJK_Size)*real_size);
        cudaMalloc((void **)&Ceyh_GPU, (IJK_Size)*real_size);
        cudaMalloc((void **)&Cezh_GPU, (IJK_Size)*real_size);
        cudaMemsetAsync(Cexe_GPU, 0, (IJK_Size)*real_size, stream);
        cudaMemsetAsync(Ceye_GPU, 0, (IJK_Size)*real_size, stream);
        cudaMemsetAsync(Ceze_GPU, 0, (IJK_Size)*real_size, stream);
        cudaMemsetAsync(Cexh_GPU, 0, (IJK_Size)*real_size, stream);
        cudaMemsetAsync(Ceyh_GPU, 0, (IJK_Size)*real_size, stream);
        cudaMemsetAsync(Cezh_GPU, 0, (IJK_Size)*real_size, stream);

        size_t ii = pml_size - 2;
        for (size_t i = 0; i < Imax - 1; i++)
        {
            if (i < pml_size - 1)
            {
                for (size_t j = 0; j < Jmax; j++)
                    for (size_t k = 0; k < Kmax; k++)
                        den_hx[i + j * Imax + k * (Imax * Jmax)] = 1.0 / (delX * kappa_Hx_PML_1[i]);
            }
            else if (i >= Imax - pml_size)
            {
                for (size_t j = 0; j < Jmax; j++)
                    for (size_t k = 0; k < Kmax; k++)
                        den_hx[i + j * Imax + k * (Imax * Jmax)] = 1.0 / (delX * kappa_Hx_PML_2[ii]);
                ii = ii - 1;
            }
            else
            {
                for (size_t j = 0; j < Jmax; j++)
                    for (size_t k = 0; k < Kmax; k++)
                        den_hx[i + j * Imax + k * (Imax * Jmax)] = 1.0 / delX;
            }
        }

        //%-------------------
        ii = pml_size - 1;
        for (size_t i = 0; i < Imax - 1; i++)
        {
            if (i < pml_size)
            {
                for (size_t j = 0; j < Jmax; j++)
                    for (size_t k = 0; k < Kmax; k++)
                        den_ex[i + j * Imax + k * (Imax * Jmax)] = 1.0 / (delX * kappa_Ex_PML_1[i]);
            }
            else if (i >= Imax - pml_size)
            {
                for (size_t j = 0; j < Jmax; j++)
                    for (size_t k = 0; k < Kmax; k++)
                        den_ex[i + j * Imax + k * (Imax * Jmax)] = 1.0 / (delX * kappa_Ex_PML_2[ii]);
                ii = ii - 1;
            }
            else
            {
                for (size_t j = 0; j < Jmax; j++)
                    for (size_t k = 0; k < Kmax; k++)
                        den_ex[i + j * Imax + k * (Imax * Jmax)] = 1.0 / delX;
            }
        }

        sigma_m = 0;
        Chh = (1 - sigma_m * delT / 2 / mu0) / (1 + sigma_m * delT / 2 / mu0);
        Che = delT / mu0 / (1 + sigma_m * delT / 2 / mu0) / delX;
        Chxh = Chh;
        Chyh = Chh;
        Chzh = Chh;
        Chxe = Che;
        Chye = Che;
        Chze = Che;

        for (size_t i = 0; i < IJK_Size; i++)
        {
            Cexe[i] = (2 * eps0 * eps_infx[i] - sigma_sx[i] * delT) / (2 * eps0 * eps_infx[i] + sigma_sx[i] * delT);
            Ceye[i] = (2 * eps0 * eps_infy[i] - sigma_sy[i] * delT) / (2 * eps0 * eps_infy[i] + sigma_sy[i] * delT);
            Ceze[i] = (2 * eps0 * eps_infz[i] - sigma_sz[i] * delT) / (2 * eps0 * eps_infz[i] + sigma_sz[i] * delT);
            Cexh[i] = 2 * delT / (2 * eps0 * eps_infx[i] + sigma_sx[i] * delT) * den_ex[i];
            Ceyh[i] = 2 * delT / (2 * eps0 * eps_infy[i] + sigma_sy[i] * delT) * den_ex[i];
            Cezh[i] = 2 * delT / (2 * eps0 * eps_infz[i] + sigma_sz[i] * delT) * den_ex[i];
        }

        cudaMemcpyAsync(Cexe_GPU, Cexe, (IJK_Size)*real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(Ceye_GPU, Ceye, (IJK_Size)*real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(Ceze_GPU, Ceze, (IJK_Size)*real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(Cexh_GPU, Cexh, (IJK_Size)*real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(Ceyh_GPU, Ceyh, (IJK_Size)*real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(Cezh_GPU, Cezh, (IJK_Size)*real_size, cudaMemcpyHostToDevice, stream);

        //////Initialize arrays for antenna record
        cudaMallocHost(&PortSignal, numAnt * max_TimeSteps * real_size); // Pinned memory for asynchronous copy
        std::fill(PortSignal, PortSignal + numAnt * max_TimeSteps, 0.0);
        cudaMalloc((void **)&PortSignal_GPU, (numAnt * max_TimeSteps) * real_size);
        cudaMalloc((void **)&source_adjoint_GPU, (numAnt * max_TimeSteps) * real_size);
        cudaMemsetAsync(source_adjoint_GPU, 0, (numAnt * max_TimeSteps) * real_size, stream);
        cudaMemsetAsync(PortSignal_GPU, 0, (numAnt * max_TimeSteps) * real_size, stream);

        ///////relase memory//////////
        delete[] alpha_Ex_PML_1;
        delete[] alpha_Ex_PML_2;
        delete[] sig_Ex_PML_1;
        delete[] sig_Ex_PML_2;
        delete[] kappa_Ex_PML_1;
        delete[] kappa_Ex_PML_2;
        delete[] alpha_Hx_PML_1;
        delete[] alpha_Hx_PML_2;
        delete[] sig_Hx_PML_1;
        delete[] sig_Hx_PML_2;
        delete[] kappa_Hx_PML_1;
        delete[] kappa_Hx_PML_2;
    }
    void initField()
    {
        init_field = 1;
        cudaMallocHost(&Ex, IJK_Size * real_size);
        cudaMallocHost(&Ey, IJK_Size * real_size);
        cudaMallocHost(&Ez, IJK_Size * real_size);
        std::fill(Ex, Ex + IJK_Size, 0.0);
        cudaMalloc((void **)&Ex_GPU, IJK_Size * real_size);
        cudaMemsetAsync(Ex_GPU, 0, IJK_Size * real_size, stream);

        std::fill(Ey, Ey + IJK_Size, 0.0);
        cudaMalloc((void **)&Ey_GPU, IJK_Size * real_size);
        cudaMemsetAsync(Ey_GPU, 0, IJK_Size * real_size, stream);

        std::fill(Ez, Ez + IJK_Size, 0.0);
        cudaMalloc((void **)&Ez_GPU, IJK_Size * real_size);
        cudaMemsetAsync(Ez_GPU, 0, IJK_Size * real_size, stream);

        cudaMalloc((void **)&Hx_GPU, IJK_Size * real_size);
        cudaMemsetAsync(Hx_GPU, 0, IJK_Size * real_size, stream);
        cudaMalloc((void **)&Hy_GPU, IJK_Size * real_size);
        cudaMemsetAsync(Hy_GPU, 0, IJK_Size * real_size, stream);
        cudaMalloc((void **)&Hz_GPU, IJK_Size * real_size);
        cudaMemsetAsync(Hz_GPU, 0, IJK_Size * real_size, stream);

        cudaMalloc((void **)&psi_Exy_GPU, 2 * size_y * real_size);
        cudaMemsetAsync(psi_Exy_GPU, 0, 2 * size_y * real_size, stream);

        cudaMalloc((void **)&psi_Exz_GPU, 2 * size_z * real_size);
        cudaMemsetAsync(psi_Exz_GPU, 0, 2 * size_z * real_size, stream);

        cudaMalloc((void **)&psi_Eyx_GPU, 2 * size_x * real_size);
        cudaMemsetAsync(psi_Eyx_GPU, 0, 2 * size_x * real_size, stream);

        cudaMalloc((void **)&psi_Eyz_GPU, 2 * size_z * real_size);
        cudaMemsetAsync(psi_Eyz_GPU, 0, 2 * size_z * real_size, stream);

        cudaMalloc((void **)&psi_Ezx_GPU, 2 * size_x * real_size);
        cudaMemsetAsync(psi_Ezx_GPU, 0, 2 * size_x * real_size, stream);

        cudaMalloc((void **)&psi_Ezy_GPU, 2 * size_y * real_size);
        cudaMemsetAsync(psi_Ezy_GPU, 0, 2 * size_y * real_size, stream);

        cudaMalloc((void **)&psi_Hxy_GPU, 2 * size_y * real_size);
        cudaMemsetAsync(psi_Hxy_GPU, 0, 2 * size_y * real_size, stream);

        cudaMalloc((void **)&psi_Hxz_GPU, 2 * size_z * real_size);
        cudaMemsetAsync(psi_Hxz_GPU, 0, 2 * size_z * real_size, stream);

        cudaMalloc((void **)&psi_Hyx_GPU, 2 * size_x * real_size);
        cudaMemsetAsync(psi_Hyx_GPU, 0, 2 * size_x * real_size, stream);

        cudaMalloc((void **)&psi_Hyz_GPU, 2 * size_z * real_size);
        cudaMemsetAsync(psi_Hyz_GPU, 0, 2 * size_z * real_size, stream);

        cudaMalloc((void **)&psi_Hzx_GPU, 2 * size_x * real_size);
        cudaMemsetAsync(psi_Hzx_GPU, 0, 2 * size_x * real_size, stream);

        cudaMalloc((void **)&psi_Hzy_GPU, 2 * size_y * real_size);
        cudaMemsetAsync(psi_Hzy_GPU, 0, 2 * size_y * real_size, stream);

        cudaMalloc((void **)&current, real_size);
        cudaMemsetAsync(current, 0, real_size, stream);

        cudaMalloc((void **)&voltage, 2 * real_size);
        cudaMemsetAsync(voltage, 0, 2 * real_size, stream);
    }

    void setAntennaLocations(int *locations)
    {
        PosCoax = new Antenna[numAnt];
        coax_ind = new size_t[numAnt];
        size_t Ax_orig, Ay_orig, Az_orig;
        for (size_t i = 0; i < numAnt; i++)
        {
            Ax_orig = locations[i + 0 * numAnt];
            Ay_orig = locations[i + 1 * numAnt];
            Az_orig = locations[i + 2 * numAnt];
            PosCoax[i].x = pml_size + Ax_orig;
            PosCoax[i].y = pml_size + Ay_orig;
            PosCoax[i].z = pml_size + Az_orig;
            coax_ind[i] = static_cast<size_t>(PosCoax[i].x) + PosCoax[i].y * Imax + PosCoax[i].z * (Imax * Jmax);
            printf("Antenna position %d: (%d, %d, %d), coax index=%zu\n", i, PosCoax[i].x, PosCoax[i].y, PosCoax[i].z, coax_ind[i]);
        }

        init_antenna = 1;
    }
    void setSource(real *SourceIn)
    {
        init_source = 1;
        cudaMallocHost(&source, max_TimeSteps * real_size);
        for (size_t i = 0; i < max_TimeSteps; i++)
            source[i] = SourceIn[i];
        cudaMalloc((void **)&source_GPU, max_TimeSteps * real_size);
        cudaMemcpyAsync(source_GPU, source, max_TimeSteps * real_size, cudaMemcpyHostToDevice, stream);
    }
    void updateCuda()
    {
        for (size_t i = 0; i < IJK_Size; i++)
        {
            Cexe[i] = (2 * eps0 * eps_infx[i] - sigma_sx[i] * delT) / (2 * eps0 * eps_infx[i] + sigma_sx[i] * delT);
            Ceye[i] = (2 * eps0 * eps_infy[i] - sigma_sy[i] * delT) / (2 * eps0 * eps_infy[i] + sigma_sy[i] * delT);
            Ceze[i] = (2 * eps0 * eps_infz[i] - sigma_sz[i] * delT) / (2 * eps0 * eps_infz[i] + sigma_sz[i] * delT);
            Cexh[i] = 2 * delT / (2 * eps0 * eps_infx[i] + sigma_sx[i] * delT) * den_ex[i];
            Ceyh[i] = 2 * delT / (2 * eps0 * eps_infy[i] + sigma_sy[i] * delT) * den_ex[i];
            Cezh[i] = 2 * delT / (2 * eps0 * eps_infz[i] + sigma_sz[i] * delT) * den_ex[i];
        }

        cudaMemcpyAsync(Cexe_GPU, Cexe, (IJK_Size)*real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(Ceye_GPU, Ceye, (IJK_Size)*real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(Ceze_GPU, Ceze, (IJK_Size)*real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(Cexh_GPU, Cexh, (IJK_Size)*real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(Ceyh_GPU, Ceyh, (IJK_Size)*real_size, cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(Cezh_GPU, Cezh, (IJK_Size)*real_size, cudaMemcpyHostToDevice, stream);
    }
    void ResetFieldAll_1()
    {
        cudaMemsetAsync(Ex_GPU, 0, IJK_Size * real_size, stream);
        cudaMemsetAsync(Ey_GPU, 0, IJK_Size * real_size, stream);
        cudaMemsetAsync(Ez_GPU, 0, IJK_Size * real_size, stream);
        cudaMemsetAsync(Hx_GPU, 0, IJK_Size * real_size, stream);
        cudaMemsetAsync(Hy_GPU, 0, IJK_Size * real_size, stream);
        cudaMemsetAsync(Hz_GPU, 0, IJK_Size * real_size, stream);

        cudaMemsetAsync(psi_Exy_GPU, 0, 2 * size_y * real_size, stream);
        cudaMemsetAsync(psi_Ezy_GPU, 0, 2 * size_y * real_size, stream);
        cudaMemsetAsync(psi_Hxy_GPU, 0, 2 * size_y * real_size, stream);
        cudaMemsetAsync(psi_Hzy_GPU, 0, 2 * size_y * real_size, stream);

        cudaMemsetAsync(psi_Exz_GPU, 0, 2 * size_z * real_size, stream);
        cudaMemsetAsync(psi_Eyz_GPU, 0, 2 * size_z * real_size, stream);
        cudaMemsetAsync(psi_Hxz_GPU, 0, 2 * size_z * real_size, stream);
        cudaMemsetAsync(psi_Hyz_GPU, 0, 2 * size_z * real_size, stream);

        cudaMemsetAsync(psi_Ezx_GPU, 0, 2 * size_x * real_size, stream);
        cudaMemsetAsync(psi_Eyx_GPU, 0, 2 * size_x * real_size, stream);
        cudaMemsetAsync(psi_Hyx_GPU, 0, 2 * size_x * real_size, stream);
        cudaMemsetAsync(psi_Hzx_GPU, 0, 2 * size_x * real_size, stream);
    }

    ~simulation()
    {
        if (init_antenna == 1)
        {
            delete[] coax_ind;
            delete[] PosCoax;
        }
        if (init_source == 1)
        {
            cudaFreeHost(source);
            cudaFree(source_GPU);
        }
        if (init_field == 1)
        {
            cudaFreeHost(Ex);
            cudaFreeHost(Ey);
            cudaFreeHost(Ez);
            cudaFree(Ex_GPU);
            cudaFree(Ey_GPU);
            cudaFree(Ez_GPU);
            cudaFree(Hx_GPU);
            cudaFree(Hy_GPU);
            cudaFree(Hz_GPU);

            cudaFree(psi_Exy_GPU);
            cudaFree(psi_Exz_GPU);
            cudaFree(psi_Eyx_GPU);

            cudaFree(psi_Eyz_GPU);
            cudaFree(psi_Ezx_GPU);
            cudaFree(psi_Ezy_GPU);

            cudaFree(psi_Hxy_GPU);
            cudaFree(psi_Hxz_GPU);
            cudaFree(psi_Hyx_GPU);

            cudaFree(psi_Hyz_GPU);
            cudaFree(psi_Hzx_GPU);
            cudaFree(psi_Hzy_GPU);
        }

        if (init_const == 1)
        {
            cudaFreeHost(eps_infx);
            cudaFreeHost(eps_infy);
            cudaFreeHost(eps_infz);

            cudaFreeHost(sigma_sx);
            cudaFreeHost(sigma_sy);
            cudaFreeHost(sigma_sz);
        }

        if (init_mu == 1)
        {
            cudaFree(Chxe_GPU);
            cudaFree(Chye_GPU);
            cudaFree(Chze_GPU);
        }
        if (init_readysimulation == 1)
        {
            cudaFreeHost(Cexe);
            cudaFreeHost(Ceye);
            cudaFreeHost(Ceze);
            cudaFreeHost(Cexh);
            cudaFreeHost(Ceyh);
            cudaFreeHost(Cezh);

            cudaFree(Cexe_GPU);
            cudaFree(Ceye_GPU);
            cudaFree(Ceze_GPU);
            cudaFree(Cexh_GPU);
            cudaFree(Ceyh_GPU);
            cudaFree(Cezh_GPU);
            cudaFree(b_Ex_1);
            cudaFree(b_Ex_2);
            cudaFree(c_Ex_2);
            cudaFree(c_Ex_1);
            cudaFree(b_Ex_GPU);
            cudaFree(c_Ex_GPU);

            cudaFree(b_Hx_1);
            cudaFree(b_Hx_2);
            cudaFree(c_Hx_1);
            cudaFree(c_Hx_2);

            cudaFree(b_Hx_GPU);
            cudaFree(c_Hx_GPU);

            // Uniform Mesh
            cudaFree(den_ex);
            cudaFree(den_hx);

            // Non-Uniform Mesh
            if (FDTD_UNIFORM_MESH != 1)
            {
                cudaFree(den_ey);
                cudaFree(den_ez);
                cudaFree(den_hy);
                cudaFree(den_hz);
            }

            cudaFree(PortSignal);
            cudaFree(PortSignal_GPU);
            cudaFree(source_adjoint_GPU);

            cudaFree(current);
        }
        /// 1D
        if (init_1D == 1)
        {
            cudaFreeHost(Ez_1D);
            cudaFree(Ez_1D_GPU);
            cudaFree(Hy_1D_GPU);
            cudaFree(H_inv_1D_GPU);
            cudaFree(E_inv_1D_GPU);
        }

        if (init_impulse == 1)
        {
            delete[] ht;
            cudaFree(ht_GPU);
            cudaFree(W_plus);
        }
        if (init_design == 1)
        {
            delete[] PosDesign;

            // cudaFree(E3_ant_GPU);
            cudaFree(E1_ant_GPU);
            cudaFree(E2_ant_GPU);

            // delete[] E3_ant;
            cudaFreeHost(E1_ant);
            cudaFreeHost(E2_ant);
            cudaFreeHost(E1_ant_AD);
            cudaFreeHost(E2_ant_AD);
            cudaFreeHost(E1_ant_temp);
            cudaFreeHost(E2_ant_temp);
            cudaFreeHost(G1_ant);
            cudaFreeHost(G2_ant);
            if (Ind_ML)
                delete[] Ind_ML;
        }
        if (init_design_GPU == 1)
        {
            delete[] PosDesign;

            cudaFree(E1_ant_AD_GPU);
            cudaFree(E2_ant_AD_GPU);
            cudaFree(E1_ant_GPU_2);
            cudaFree(E2_ant_GPU_2);
            cudaFree(G1_ant_GPU);
            cudaFree(G2_ant_GPU);
            cudaFree(G_temp);
            cudaFree(G_output);
        }
        cudaStreamDestroy(stream);
    };

private:
    struct Antenna
    {
        size_t x;
        size_t y;
        size_t z;
    } *PosDesign, *PosCoax;
    size_t *coax_ind;
    size_t *Ind_ML;
    size_t NumML = 0;

    size_t size_t_E1_ant;
    size_t size_t_E2_ant;
    size_t size_E1_ant;
    size_t size_E2_ant;

    int init_const = 0;
    int init_antenna = 0;
    int init_source = 0;
    int init_field = 0;
    int init_readysimulation = 0;
    int init_1D = 0;
    int init_mu = 0;
    int init_impulse;
    int init_design;
    int init_design_GPU;
    size_t block_size;
    real *Ex;
    real *Ey;
    real *Ez;
    real *Ex_GPU;
    real *Ey_GPU;
    real *Ez_GPU;
    real *Hx_GPU;
    real *Hy_GPU;
    real *Hz_GPU;

    real *psi_Exy_GPU;
    real *psi_Exz_GPU;
    real *psi_Eyx_GPU;

    real *psi_Eyz_GPU;
    real *psi_Ezx_GPU;
    real *psi_Ezy_GPU;

    real *psi_Hxy_GPU;
    real *psi_Hxz_GPU;
    real *psi_Hyx_GPU;

    real *psi_Hyz_GPU;
    real *psi_Hzx_GPU;
    real *psi_Hzy_GPU;

    /// Debye Parameters
    real *eps_infx;
    real *eps_infy;
    real *eps_infz;
    real *sigma_sx;
    real *sigma_sy;
    real *sigma_sz;

    real *Cexe;
    real *Ceye;
    real *Ceze;
    real *Cexh;
    real *Ceyh;
    real *Cezh;

    real *Cexe_GPU;
    real *Ceye_GPU;
    real *Ceze_GPU;
    real *Cexh_GPU;
    real *Ceyh_GPU;
    real *Cezh_GPU;

    real sigma_m;
    real Chh;
    real Chxh;
    real Chyh;
    real Chzh;
    real Che;
    real Chxe;
    real Chye;
    real Chze;

    real *Chxe_GPU;
    real *Chye_GPU;
    real *Chze_GPU;

    ////Source////
    real *source;
    real *source_GPU;

    /// constants

    real Z_c;

    /// freqRes
    real fctr;
    real delX;
    real delT;
    /// TimeDim
    size_t numAnt;
    size_t max_TimeSteps;
    size_t I_size;
    size_t J_size;
    size_t K_size;
    real poly_m; // polynomial order for pml grading
    size_t IJK_Size;
    /// CPML_param
    real alpha_m;
    real sig_max;
    real alpha_max;
    real kappa_max;

    size_t Imax;
    size_t Jmax;
    size_t Kmax;
    size_t IJ_Size;

    // //     PML Coefficients
    real *b_Ex_1;
    real *b_Ex_2;
    real *c_Ex_2;
    real *c_Ex_1;

    real *b_Ex_GPU;
    real *c_Ex_GPU;

    real *alpha_Ex_PML_1;
    real *alpha_Ex_PML_2;
    real *sig_Ex_PML_1;
    real *sig_Ex_PML_2;
    real *kappa_Ex_PML_1;
    real *kappa_Ex_PML_2;

    real *b_Hx_1;
    real *b_Hx_2;
    real *c_Hx_1;
    real *c_Hx_2;

    real *b_Hx_GPU;
    real *c_Hx_GPU;

    real *alpha_Hx_PML_1;
    real *alpha_Hx_PML_2;
    real *sig_Hx_PML_1;
    real *sig_Hx_PML_2;
    real *kappa_Hx_PML_1;
    real *kappa_Hx_PML_2;

    //////////Uniform Mesh/////////
    real *den_ex;
    real *den_hx;

    //////////Non-Uniform Mesh/////////
    real *den_ey;
    real *den_ez;
    real *den_hy;
    real *den_hz;

    real *PortSignal;
    real *PortSignal_GPU;
    real *current;

    size_t i0;
    size_t i1;
    size_t j0;
    size_t j1;
    size_t k0;
    size_t k1;

    /////1d problem

    real *Ez_1D; ////1D field in CPU for illustration.
    real *Ez_1D_GPU;
    real *Hy_1D_GPU;
    real *E_inv_1D_GPU;
    real *H_inv_1D_GPU;
    size_t size_1D;

    real *E3_ant_GPU;
    real *E1_ant_GPU;
    real *E2_ant_GPU;

    real *E1_ant_AD_GPU;
    real *E2_ant_AD_GPU;
    real *E3_ant_AD_GPU;

    real *E1_ant_GPU_2;
    real *E2_ant_GPU_2;
    real *E3_ant_GPU_2;

    double *G1_ant_GPU;
    double *G2_ant_GPU;
    double *G3_ant_GPU;

    double *G_temp;
    double *G_output;

    real *E3_ant_temp;
    real *E1_ant_temp;
    real *E2_ant_temp;

    //
    double *G3_ant;
    double *G1_ant;
    double *G2_ant;

    real *E3_ant;
    real *E1_ant;
    real *E2_ant;

    real *E3_ant_AD;
    real *E1_ant_AD;
    real *E2_ant_AD;

    size_t DesignX;
    size_t DesignY;
    size_t DesignZ;
    real *source_adjoint_GPU;
    real sigma_max;
    int width_coax1;
    int width_coax0;
    int depth_coax0;
    int depth_coax1;

    ////accelerate GPU

    size_t NumBlocks;

    int flag_Debye;
    real *voltage;
    real *ht;
    real *ht_GPU;
    real *W_plus;

    size_t size_x;
    size_t size_y;
    size_t size_z;

    size_t gridsize_x;
    size_t gridsize_y;
    size_t gridsize_z;

    cudaStream_t stream;
};
#endif // SIMULATION_H

// ///////////////
// void FDTD_Forward_time(int *time, int *mode)
//     {
//         int i = time[0];
//         int nstep = i * size_1D;
//         if (mode[0] == 0) //////Transimission mode
//         {
//             updateH_All();
//             updateH_PML();
//             updateE_All();
//             updateE_PML();
//             // PortUpdate<<<1, 1, 0, stream>>>(PortSignal_GPU, Ex_GPU, Ey_GPU, Hx_GPU, Hy_GPU, source_GPU, i, current, coax_ind[0], Imax, Jmax, delX, Z_c, width_coax0, width_coax1, 1);
//             coaxUpdateGPU<<<1, 1, 0, stream>>>(PortSignal_GPU, Ex_GPU, Ey_GPU, Hx_GPU, Hy_GPU, source_GPU, i, coax_ind[0], Imax, Jmax, delX, Z_c, depth_coax0, depth_coax1, width_coax0, width_coax1, 1);

//             // Current_coax<<<1, 1, 0, stream>>>(current, Hx_GPU, Hy_GPU, coax_ind[0], Imax, Jmax, delX, width_coax0, width_coax1);
//             //  UpdateSource << <1, 1 >> > (PortSignal_GPU,Ex_GPU, Ey_GPU, source_GPU, i, current, coax_ind[0], Imax, Jmax, delX, Z_c, width_coax0, width_coax1,1);
//             // AddSource_coax_GPU<<<1, 1, 0, stream>>>(Ex_GPU, Ey_GPU, source_GPU, i, current, coax_ind[0], Imax, Jmax, delX, Z_c, width_coax0, width_coax1);
//             // ReadCoax<<<1, 1, 0, stream>>>(PortSignal_GPU, Ex_GPU, coax_ind[0], i, current, delX, Z_c, voltage);
//             // GPU_Read_Ant_Field<<<DesignY *(DesignZ + 1) / NumThreads + 1, NumThreads, 0, stream>>>(E1_ant_GPU, Ey_GPU, DesignY, DesignZ + 1, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z, Imax, Jmax, i);
//             // GPU_Read_Ant_Field<<<(DesignY + 1) * DesignZ / NumThreads + 1, NumThreads, 0, stream>>>(E2_ant_GPU, Ez_GPU, DesignY + 1, DesignZ, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z, Imax, Jmax, i);
//         }
//         else if (mode[0] == 1)
//         {
//             updateH_All();
//             updateH_PW_ZY(Ez_1D_GPU, 1, nstep + size_1D - Kmax);
//             updateH_PW_XY(E_inv_1D_GPU, 1, nstep + size_1D - Imax);
//             updateH_PML();
//             updateE_All();
//             updateE_PW_ZY(Hy_1D_GPU, 1, nstep + size_1D - Kmax);
//             updateE_PW_XY(H_inv_1D_GPU, 1, nstep + size_1D - Imax);
//             updateE_PML();
//             PortUpdate<<<1, 1, 0, stream>>>(PortSignal_GPU, Ex_GPU, Ey_GPU, Hx_GPU, Hy_GPU, source_GPU, i, current, coax_ind[0], Imax, Jmax, delX, Z_c, width_coax0, width_coax1, 0);
//             // Current_coax<<<1, 1, 0, stream>>>(current, Hx_GPU, Hy_GPU, coax_ind[0], Imax, Jmax, delX, width_coax0, width_coax1);
//             // UpdateSource<<<1, 1, 0, stream>>>(PortSignal_GPU, Ex_GPU, Ey_GPU, source_GPU, i, current, coax_ind[0], Imax, Jmax, delX, Z_c, width_coax0, width_coax1, 0);
//             //  AddSource_coax << <1, 1 >> > (Ex_GPU, Ey_GPU, 0.0f, current, coax_ind[0], Imax, Jmax, delX, Z_c, width_coax0, width_coax1, voltage);
//             //  ReadCoax << <1, 1 >> > (PortSignal_GPU, Ex_GPU, coax_ind[0], i, current, delX, Z_c, voltage);
//             GPU_Read_Ant_Field<<<DesignY *(DesignZ + 1) / NumThreads + 1, NumThreads, 0, stream>>>(E1_ant_GPU, Ey_GPU, DesignY, DesignZ + 1, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z, Imax, Jmax, i);
//             GPU_Read_Ant_Field<<<(DesignY + 1) * DesignZ / NumThreads + 1, NumThreads, 0, stream>>>(E2_ant_GPU, Ez_GPU, DesignY + 1, DesignZ, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z, Imax, Jmax, i);
//         }

//         else if (mode[0] == 3)
//         {
//             updateH_All();
//             updateH_PML();
//             updateE_All();
//             updateE_PML();
//             Current_coax<<<1, 1, 0, stream>>>(current, Hx_GPU, Hy_GPU, coax_ind[0], Imax, Jmax, delX, width_coax0, width_coax1);
//             // UpdateSource << <1, 1 >> > (PortSignal_GPU,Ex_GPU, Ey_GPU, source_GPU, i, current, coax_ind[0], Imax, Jmax, delX, Z_c, width_coax0, width_coax1,1);
//             AddSource_coax_GPU<<<1, 1, 0, stream>>>(Ex_GPU, Ey_GPU, source_GPU, i, current, coax_ind[0], Imax, Jmax, delX, Z_c, width_coax0, width_coax1);
//             ReadCoax<<<1, 1, 0, stream>>>(PortSignal_GPU, Ex_GPU, coax_ind[0], i, current, delX, Z_c, voltage);
//         }
//         else if (mode[0] == 2)
//         {
//             updateH_All();
//             updateH_PML();
//             updateE_All();
//             updateE_PML();
//             float temp = delX / c / delT;
//             Current_coax<<<1, 1, 0, stream>>>(current, Hx_GPU, Hy_GPU, coax_ind[0], Imax, Jmax, delX, width_coax0, width_coax1);
//             AddSource_coax1<<<1, 1, 0, stream>>>(Ex_GPU, Ey_GPU, source[i], current, PosCoax[0].x, PosCoax[0].y, PosCoax[0].z, Imax, Jmax, delX, Z_c, width_coax0, width_coax1,
//                                                  voltage, temp, PortSignal_GPU, i);
//         }
//     }

// ///    void FDTD_Adjoint_time(int *time, int *mode)
//     {
//         int i = time[0];
//         updateH_All();
//         updateH_PML();
//         updateE_All();
//         updateE_PML();
//         Current_coax<<<1, 1, 0, stream>>>(current, Hx_GPU, Hy_GPU, coax_ind[0], Imax, Jmax, delX, width_coax0, width_coax1);
//         AddSource_coax_GPU<<<1, 1, 0, stream>>>(Ex_GPU, Ey_GPU, source_adjoint_GPU, i, current, coax_ind[0], Imax, Jmax, delX, Z_c, width_coax0, width_coax1);
//         ReadCoax<<<1, 1, 0, stream>>>(PortSignal_GPU, Ex_GPU, coax_ind[0], i + max_TimeSteps, current, delX, Z_c, voltage);
//         GPU_Read_Ant_Field<<<DesignY *(DesignZ + 1) / NumThreads + 1, NumThreads, 0, stream>>>(E1_ant_GPU, Ey_GPU, DesignY, DesignZ + 1, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z, Imax, Jmax, i);
//         GPU_Read_Ant_Field<<<(DesignY + 1) * DesignZ / NumThreads + 1, NumThreads, 0, stream>>>(E2_ant_GPU, Ez_GPU, DesignY + 1, DesignZ, PosDesign[0].x, PosDesign[0].y, PosDesign[0].z, Imax, Jmax, i);
//     }