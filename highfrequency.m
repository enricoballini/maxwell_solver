reset(gpuDevice)
cd(fileparts(mfilename("fullpath")))
clear
close all



addpath(genpath("."),genpath("../Functions"))
%%%optimization parameters initialization
clight=299792458;
mu0=4*pi*1e-7;
eps0=1/clight^2/mu0;
fctr=5.5e9;
delX=0.2104/2/10^3;
delT=0.99*delX/(sqrt(3)*clight);
timeSteps=20000;
distancePML=10;
NumOfSidLobes=2;
PercentageBW =8e9/fctr;
constSim.delT=delT;
constSim.fctr=fctr;
constSim.timeSteps=timeSteps;
constSim.NumOfSidLobes = NumOfSidLobes;
constSim.PercentageBW =PercentageBW;
[fmaxx,constSim.source]=CreateSource(constSim);

figure;
plot(constSim.source, 'LineWidth', 2);
grid on;




%%%%save optimization parameters
constSim.delX=delX;
constSim.delT=delT;
constSim.fctr=fctr;
constSim.timeSteps=timeSteps;

constSim.Dim=[230,260,30];
numAnts=0;
%%%%save path
flag_plot=1;

%--------------------project_FDTD_3D-----------------------
pml_size=10;
freqRes = [constSim.fctr delX delT];
TimeDim = [timeSteps constSim.Dim(1) constSim.Dim(2) constSim.Dim(3) numAnts];

% % % %  Extend Dielectric profile for the CPML boundary condition
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
%FDTD
P = simFDTD('CreateSimulation');
simFDTD('SetConst',P,TypeReal(freqRes),TypeInt(TimeDim));
simFDTD('SetSource',P,TypeReal(constSim.source));


simFDTD('UpdateMaterial',P,TypeInt(1),TypeReal(sigma_sx(:)));
simFDTD('UpdateMaterial',P,TypeInt(2),TypeReal(sigma_sy(:)));
simFDTD('UpdateMaterial',P,TypeInt(3),TypeReal(sigma_sz(:)));
simFDTD('UpdateMaterial',P,TypeInt(4),TypeReal(eps_infx(:)));
simFDTD('UpdateMaterial',P,TypeInt(5),TypeReal(eps_infy(:)));
simFDTD('UpdateMaterial',P,TypeInt(6),TypeReal(eps_infz(:)));

simFDTD('ReadySimulation',P);
simFDTD('InitField',P);
antdatasize=numAnts*timeSteps;

simFDTD('Init1D',P,TypeReal(1),TypeReal(constSim.source));



simFDTD('UpdateCuda',P);

simFDTD('ResetFieldAll',P);
for i=0:timeSteps-1
    simFDTD('UpdateH_All',P);
    %simFDTD('UpdateH_PlaneWave',P,TypeInt([i,3]));
    simFDTD('UpdateH_PML',P);
    simFDTD('UpdateE_All',P);
    %simFDTD('UpdateE_PlaneWave',P,TypeInt([i,3]));
    simFDTD('UpdateE_PML',P);
    simFDTD('AddCurrents',P,TypeInt([i,3]),TypeInt([11,239,139,139,26,26]));

    if(mod(i,10)==0)
        % aa=simFDTD('OutputVis',P,uint32(extend_dimX*extend_dimY*extend_dimZ),int32(2));
        % bb=(reshape(aa,[extend_dimX ,extend_dimY,extend_dimZ]));
        % subplot(2,2,1)
        % imagesc(squeeze((bb(100,:,:)))')
        % colorbar
        % axis xy
        % axis square
        % title("Ex--Xsilce"+num2str(i))
        % aa=simFDTD('OutputVis',P,uint32(extend_dimX*extend_dimY*extend_dimZ),int32(2));
        % bb=(reshape(aa,[extend_dimX ,extend_dimY,extend_dimZ]));
        % subplot(2,2,2)
        % imagesc(squeeze((bb(:,:,16)))')
        % colorbar
        % axis xy
        % axis square
        % title("Ey--Xsilce"+num2str(i))

        aa=simFDTD('OutputVis',P,uint32(extend_dimX*extend_dimY*extend_dimZ),int32(1));
        bb=(reshape(aa,[extend_dimX ,extend_dimY,extend_dimZ]));
        %subplot(2,2,3)
        imagesc(squeeze(abs(bb(:,:,17)))')
        colorbar
        axis xy
        axis square
        title("Ez--Xsilce"+num2str(i))

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
