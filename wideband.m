reset(gpuDevice)
cd(fileparts(mfilename("fullpath")))
clear
close all



addpath(genpath("."),genpath("../Functions"))
%%%optimization parameters initialization
clight=299792458;
mu0=4*pi*1e-7;
eps0=1/clight^2/mu0;
unit=10^3;
delX=10*unit;
delT=0.99*delX/(sqrt(3)*clight);
fctr=5.5e9;
timeSteps=300000;
distancePML=10;


Amax = 100e3;   % 100 kA
t0   = 5;      % 峰值时间
FWHM = 10;     % 宽度
sigma = FWHM / 4/2.355;
dt=delT;
t  = 0:dt:100;

g = Amax * exp(-(t - t0).^2 / (2*sigma^2));

figure;
plot(t, g/1e3, 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Current (kA)');
title('Decaying Gaussian Source (Peak at t=0)');
grid on;




%%%%save optimization parameters
constSim.delX=delX;
constSim.delT=delT;
constSim.fctr=fctr;
constSim.timeSteps=timeSteps;

constSim.Dim=[230,260,30];
constSim.AntLoc=[20,20,20];
[numAnts,~]=size(constSim.AntLoc);
%[fmaxx,constSim.source]=CreateSource(constSim);
constSim.source=g(1:timeSteps);
%%%%save path
flag_plot=1;

%--------------------project_FDTD_3D-----------------------
pml_size=10;
freqRes = [constSim.fctr delX delT];
TimeDim = [timeSteps constSim.Dim(1) constSim.Dim(2) constSim.Dim(3) numAnts];

% % % %  Extend Dielectric profile for the CPML boundary condition
estEpsInf   = ones(constSim.Dim);
estEpsDelta = 0*estEpsInf;
estCond     = 0*estEpsInf;
extend_dimX = constSim.Dim(1)+2*pml_size;
extend_dimY = constSim.Dim(2)+2*pml_size;
extend_dimZ = constSim.Dim(3)+2*pml_size;
SimOri=round([extend_dimX/2,extend_dimY/2,extend_dimZ/2]);
eps_infx = ones(extend_dimX,extend_dimY,extend_dimZ);
sigma_sx = 0.0000*ones(extend_dimX,extend_dimY,extend_dimZ);
eps_infy=eps_infx;
eps_infz=eps_infx;
sigma_sy=sigma_sx;
sigma_sz=sigma_sx;
x1=107+1;
x2=134+1;
indxy=1:16;
sigma_sx(1:x1,:,indxy)=0.01586;
sigma_sx(x2+1:end,:,indxy)=0.001;
sigma_sx(1:x1,:,17)=0.01586/2;
sigma_sx(x2+1:end,:,17)=0.001/2;

sigma_sy(1:x1,:,indxy)=0.01586;
sigma_sy(x2+1:end,:,indxy)=0.001;
sigma_sy(1:x1,:,17)=0.01586/2;
sigma_sy(x2+1:end,:,17)=0.001/2;

sigma_sz(1:x1,:,1:16)=0.01586;
sigma_sz(x2+1:end,:,1:16)=0.001;

fact=1.1;
for i=x1+1:x2
        sigma_sx(i,:,1:16)=0.01586/fact;
        sigma_sy(i,:,1:16)=0.01586/fact;
sigma_sx(i,:,17)=0.01586/fact/2;
sigma_sy(i,:,17)=0.01586/fact/2;
        sigma_sz(i,:,1:16)=0.01586/fact;
    fact=fact*1.1;
end
%FDTD
P = simFDTD('CreateSimulation');
simFDTD('SetConst',P,TypeReal(freqRes),TypeInt(TimeDim));
simFDTD('SetAntennaLocations',P,TypeInt(constSim.AntLoc));
simFDTD('SetSource',P,TypeReal(constSim.source));


sigma_max=5.8e7;
constSim.mu_r=1;


simFDTD('UpdateMaterial',P,TypeInt(1),TypeReal(sigma_sx(:)));
simFDTD('UpdateMaterial',P,TypeInt(2),TypeReal(sigma_sy(:)));
simFDTD('UpdateMaterial',P,TypeInt(3),TypeReal(sigma_sz(:)));
simFDTD('UpdateMaterial',P,TypeInt(4),TypeReal(eps_infx(:)));
simFDTD('UpdateMaterial',P,TypeInt(5),TypeReal(eps_infy(:)));
simFDTD('UpdateMaterial',P,TypeInt(6),TypeReal(eps_infz(:)));




simFDTD('ReadySimulation',P);
simFDTD('InitField',P);
if(constSim.mu_r~=1)
    simFDTD('CreateMu',P,TypeReal(mu_x),TypeReal(mu_y),TypeReal(mu_z))
end
antdatasize=numAnts*timeSteps;




simFDTD('UpdateCuda',P);

simFDTD('ResetFieldAll',P);
for i=0:timeSteps-1
    simFDTD('UpdateH_All',P);
    simFDTD('UpdateH_PML',P);
    simFDTD('UpdateE_All',P);
    simFDTD('UpdateE_PML',P);
    simFDTD('AddCurrents',P,TypeInt([i,2]),TypeInt([9,241,140,140,26,26]));

    if(mod(i,1000)==0)
        aa=simFDTD('OutputVis',P,uint32(extend_dimX*extend_dimY*extend_dimZ),int32(1));
        bb=(reshape(aa,[extend_dimX ,extend_dimY,extend_dimZ]));
        subplot(3,3,1)
        imagesc(squeeze(bb(:,:,17))')
        colorbar
        axis xy
        axis square
        title("Ex--Xsilce"+num2str(i))
        aa=simFDTD('OutputVis',P,uint32(extend_dimX*extend_dimY*extend_dimZ),int32(2));
        bb=(reshape(aa,[extend_dimX ,extend_dimY,extend_dimZ]));
        subplot(3,3,2)
        imagesc(squeeze(bb(:,:,17))')
        colorbar
        axis xy
        axis square
        title("Ex--Xsilce"+num2str(i))
        % subplot(3,3,2)
        % imagesc(squeeze(bb(:,Ant(1,2),:))')
        % colorbar
        % axis xy
        % axis square
        % title("Ex--Ysilce")
        % subplot(3,3,3)
        % imagesc(squeeze(bb(:,:,Ant(1,3)+ProbeHgt))')
        % colorbar
        % axis xy
        % axis square
        % title("Ex--Zsilce")
        % subplot(3,3,4)
        % aa=OutputVis(P,uint32(extend_dimX*extend_dimY*extend_dimZ),int32(2));
        % bb=(reshape(aa,[extend_dimX ,extend_dimY,extend_dimZ]));
        % imagesc(squeeze(bb(Ant(1,1),:,:))')
        % colorbar
        % axis xy
        % axis square
        % title("Ey--Xsilce")
        % subplot(3,3,5)
        % imagesc(squeeze(bb(:,Ant(1,2)+1,:))')
        % colorbar
        % axis xy
        % axis square
        % title("Ey--Ysilce")
        % subplot(3,3,6)
        % imagesc(squeeze(bb(:,:,Ant(1,3)+ProbeHgt))')
        % colorbar
        % axis xy
        % axis square
        % title("Ey--Zsilce")
        % subplot(3,3,7)
        % aa=OutputVis(P,uint32(extend_dimX*extend_dimY*extend_dimZ),int32(3));
        % bb=(reshape(aa,[extend_dimX ,extend_dimY,extend_dimZ]));
        % imagesc(squeeze(bb(Ant(1,1),:,:))')
        % colorbar
        % axis xy
        % axis square
        % title("Ez--Xsilce")
        % subplot(3,3,8)
        % imagesc(squeeze(bb(:,Ant(1,2),:))')
        % colorbar
        % axis xy
        % axis square
        % title("Ez--Ysilce")
        % subplot(3,3,9)
        % imagesc(squeeze(bb(:,:,Ant(1,3)+ProbeHgt))')
        % colorbar
        % axis xy
        % axis square
        % title("Ez--Zsilce")
        refresh,pause(0.1)
    end


end


simFDTD('DeleteSimulation',P);
reset(gpuDevice)



function FDTD_solver(P,timeSteps,FullGPU)
for i=0:timeSteps-1
    simFDTD('UpdateH_All',P);
    simFDTD('UpdateH_PML',P);
    simFDTD('UpdateE_All',P);
    simFDTD('UpdateE_PML',P);
end
end

function FDTD_solverRe(P,timeSteps,FullGPU)
for i=0:timeSteps-1
    simFDTD('UpdateH_All',P);
    %UpdateH_PlaneWave(P,TypeInt([i,1]));
    simFDTD('UpdateH_PlaneWave',P,TypeInt([i,2]));
    %UpdateH_PlaneWave(P,TypeInt([i,3]));

    simFDTD('UpdateH_PML',P);
    simFDTD('UpdateE_All',P);
    %UpdateE_PlaneWave(P,TypeInt([i,1]));
    simFDTD('UpdateE_PlaneWave',P,TypeInt([i,2]));
    %UpdateE_PlaneWave(P,TypeInt([i,3]));

    simFDTD('UpdateE_PML',P);

end
end





function x=TypeInt(x)
x=int32(x);
end
function x=TypeReal(x)
x=single(x);
end
