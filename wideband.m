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
timeSteps=30000;
distancePML=10;



%%%%save optimization parameters
constSim.delX=delX;
constSim.delT=delT;
constSim.fctr=fctr;
constSim.timeSteps=timeSteps;

constSim.Dim=[230,260,30];
constSim.AntLoc=[20;20;20];
[numAnts,~]=size(constSim.AntLoc);
%[fmaxx,constSim.source]=CreateSource(constSim);
constSim.source=zeros(1,timeSteps);
%%%%save path
flag_plot=1;

%--------------------project_FDTD_3D-----------------------
pml_size=10;
freqRes = [constSim.fctr delX delT];
TimeDim = [timeSteps constSim.Dim(1) constSim.Dim(2) constSim.Dim(3) numAnts 0 0 0 0];

% % % %  Extend Dielectric profile for the CPML boundary condition
estEpsInf   = ones(constSim.Dim);
estEpsDelta = 0*estEpsInf;
estCond     = 0*estEpsInf;
extend_dimX = constSim.Dim(1)+2*pml_size;
extend_dimY = constSim.Dim(2)+2*pml_size;
extend_dimZ = constSim.Dim(3)+2*pml_size;
SimOri=round([extend_dimX/2,extend_dimY/2,extend_dimZ/2]);
eps_infx = ones(extend_dimX,extend_dimY,extend_dimZ);
sigma_sx = zeros(extend_dimX,extend_dimY,extend_dimZ);
eps_infy=eps_infx;
eps_infz=eps_infx;
sigma_sy=sigma_sx;
sigma_sz=sigma_sx;
x1=107+1;
x2=134+1;
sigma_sx(1:x1,:,11:17)=0.01586;
sigma_sx(x2+1:end,:,11:17)=0.001;
fact=1.1;
for i=x1+1:x2
    sigma_sx(i,:,:)=0.01586/fact;
    fact=fact*1.1;
end

%FDTD
P = simFDTD('CreateSimulation');
simFDTD('SetConst',P,TypeReal(freqRes),TypeInt(TimeDim));
simFDTD('SetAntennaLocations',P,TypeInt(constSim.AntLoc));
simFDTD('SetSource',P,TypeReal(constSim.source));


sigma_max=5.8e7;
constSim.mu_r=1;

Ant(1,1)=constSim.AntLoc(1,1)+pml_size+1;
Ant(1,2)=constSim.AntLoc(1,2)+pml_size+1;
Ant(1,3)=constSim.AntLoc(1,3)+pml_size+1;



if(constSim.mu_r~=1)
    mu_x = ones(extend_dimX,extend_dimY,extend_dimZ);mu_y=mu_x;mu_z=mu_x;
    mu_x (Ant(1,1)-depth_coax0:Ant(1,1)+depth_coax1,Ant(1,2)-width_coax0-1:Ant(1,2)+width_coax1,Ant(1,3):Ant(1,3)+CoaxHgt-1)=constSim.mu_r;
    mu_y (Ant(1,1)-depth_coax0-1:Ant(1,1)+depth_coax1,Ant(1,2)-width_coax0:Ant(1,2)+width_coax1,Ant(1,3):Ant(1,3)+CoaxHgt-1)=constSim.mu_r;
    mu_z (Ant(1,1)-depth_coax0-1:Ant(1,1)+depth_coax1,Ant(1,2)-width_coax0-1:Ant(1,2)+width_coax1,Ant(1,3):Ant(1,3)+CoaxHgt)=constSim.mu_r;
end

simFDTD('UpdateMaterial',P,TypeInt(1),TypeReal(sigma_sx(:)));
simFDTD('UpdateMaterial',P,TypeInt(2),TypeReal(sigma_sy(:)));
simFDTD('UpdateMaterial',P,TypeInt(3),TypeReal(sigma_sz(:)));
simFDTD('UpdateMaterial',P,TypeInt(4),TypeReal(eps_infx(:)));
simFDTD('UpdateMaterial',P,TypeInt(5),TypeReal(eps_infy(:)));
simFDTD('UpdateMaterial',P,TypeInt(6),TypeReal(eps_infz(:)));





ReadySimulation(P);
InitField(P);
if(constSim.mu_r~=1)
    CreateMu(P,TypeReal(mu_x),TypeReal(mu_y),TypeReal(mu_z))
end
antdatasize=numAnts*timeSteps;



load("2025-09-17_Microstrip_MultiLayer_Sym_Thick_14_0 -2_45_45_200_200_2","xmma_record","F")

% for i=1:10:180
rho_new=xmma_record(120,:);
%
% rho_new1=xmma_record1;
% rho_new2=xmma_record1;
% rho_new3=xmma_record1;
% if(NumML==2)
% rho_new=[rho_new1,rho_new2];
% elseif(NumML==3)
% rho_new=[rho_new1,rho_new2,rho_new3];
% end
limit_rho=0.5;
rho_new(rho_new<limit_rho)=0;
rho_new(rho_new>=limit_rho)=1;
sigma_new=Fmapping(rho_new);
sigma_new(rho_new==1)=5.8e7;
sigma_new(rho_new~=1)=sigma_sub;
sigma_new=reshape(sigma_new,[],NumML);
rho_new=reshape(rho_new,[],NumML);

figure(1)
for i=1:NumML
    subplot(1,NumML,i)
    patch(xx-1,yy-1,(rho_new(:,i)),'EdgeColor','none')
    colormap jet
    colorbar
    caxis([0,1])
    axis equal
    xlim([0,DesignX])
    ylim([0,DesignY])
end

% drawnow
% pause(0.1)
% end

DesignOri=[SimOri(1)-DesignX/2,SimOri(2)-DesignY/2,Ant(1,3)+ProbeHgt+CoaxHgt];
for index=1:NumML
    sigma_sx(DesignOri(1,1):DesignOri(1,1)+DesignX-1,DesignOri(1,2):DesignOri(1,2)+DesignY,DesignOri(1,3)+IndLayer(index)) = reshape(sigma_new(1:sizeDesign1,index),DesignX,DesignY+1);
    sigma_sy(DesignOri(1,1):DesignOri(1,1)+DesignX,DesignOri(1,2):DesignOri(1,2)+DesignY-1,DesignOri(1,3)+IndLayer(index)) = reshape(sigma_new(sizeDesign1+1:end,index),DesignX+1,DesignY);
end
UpdateMaterial(P,TypeInt(1),TypeReal(sigma_sx(:)));
UpdateMaterial(P,TypeInt(2),TypeReal(sigma_sy(:)));
UpdateCuda(P);
% for index=1:NumML
%     UpdateAntCondML(P,TypeReal(sigma_new(1:sizeDesign1,index)),TypeReal(sigma_new(sizeDesign1+1:end,index)),TypeInt(IndLayer(index)));
% end

%%%%transmit
disp("Transmiting mode is running")
ResetFieldAll(P);
tT1=tic;
FDTD_solver(P,timeSteps,FullGPU)
tTransmit(1)=toc(tT1);
W_out2=OutputAnt(P,int32(antdatasize));
hold on
aa=W_out2.*cos(((0:constSim.timeSteps-1)/constSim.timeSteps)* pi/2);

N=double(constSim.timeSteps);
f=(0:10*N-1)/constSim.delT/10/N;
source_new=constSim.source;
source_new(N+1:10*N)=0;
sourceF=fft([source_new(2:end),0]);
figure(2)
bb=aa;
signal=[bb,zeros(1,9*N)];
signalF=fft(signal);
plot(f/1e9,20*log10(abs(signalF)./abs(sourceF)),'LineWidth',3);
hold on

axis([5 7 -25 0 ]);
%title("S {parameters}","FontSize",30)
set(gca,"FontSize",25)
ylabel("|S_{ij}| [dB]","fontsize",25)
xlabel("Frequency [GHz]","fontsize",25)
set(gca,'LineWidth',2)
grid on
ax=gca;
ax.LineStyleOrder=["-","--","o",'-+'];
ax.ColorOrder = [ 0 0 1;1 0 0; 0 1 0;];
title("","FontSize",25)
set(findall(gca, 'Type', 'Line'),'LineWidth',2);
legend("S_{11}^{FDTD}","S_{11}^{CST}","fontsize",25,'NumColumns',1)

legend("S_{FDTD}^{ori}","S_{FDTD}^{1e^5}","S_{FDTD}^{5.8e^7}","fontsize",20,'NumColumns',1,'BackgroundAlpha',0.1)
legend("S_{FDTD}^{ori}","S_{FDTD}^{bi}","fontsize",20,'NumColumns',1,'BackgroundAlpha',0.1)


set(gcf, 'Position',  [100, 100, 600, 500])
xticks([5 5.1 5.2 5.3 5.4 5.5 5.6 5.7 5.8 5.9 6.0])
xtickind=5:0.2:7;
xticks(xtickind)
for i=1:11
    xxlabel{i}='';
    if(mod(i,2)~=0)
        xxlabel{i}=num2str(xtickind(i));
    end
end
xticklabels(xxlabel)
xtickangle(0);



DeleteSimulation(P);
reset(gpuDevice)



function FDTD_solver(P,timeSteps,FullGPU)
for i=0:timeSteps-1
    UpdateH_All(P);
    UpdateH_PML(P);
    UpdateE_All(P);
    UpdateE_PML(P);
    CoaxUpdate(P,TypeInt([i,0,1]));
    % if(FullGPU==1)
    %     RecordGPUField(P,TypeInt([i,1]));
    % else
    %     RecordForwardField(P,TypeInt(i));
    % end
end
end

function FDTD_solverRe(P,timeSteps,FullGPU)
for i=0:timeSteps-1
    UpdateH_All(P);
    %UpdateH_PlaneWave(P,TypeInt([i,1]));
    UpdateH_PlaneWave(P,TypeInt([i,2]));
    %UpdateH_PlaneWave(P,TypeInt([i,3]));

    UpdateH_PML(P);
    UpdateE_All(P);
    %UpdateE_PlaneWave(P,TypeInt([i,1]));
    UpdateE_PlaneWave(P,TypeInt([i,2]));
    %UpdateE_PlaneWave(P,TypeInt([i,3]));

    UpdateE_PML(P);
    CoaxUpdate(P,TypeInt([i,0,0]));
    % if(FullGPU==1)
    %     RecordGPUField(P,TypeInt([i,0]));
    % else
    %     RecordForwardField(P,TypeInt(i));
    % end
end
end

function FDTD_Adjoint(P,timeSteps,FullGPU)
for i=0:timeSteps-1
    UpdateH_All(P);
    UpdateH_PML(P);
    UpdateE_All(P);
    UpdateE_PML(P);
    CoaxUpdate(P,TypeInt([i,1,1]));
    % if(FullGPU==1)
    %     RecordGPUField(P,TypeInt([i,2]));
    % else
    %     RecordForwardField(P,TypeInt(i));
    % end
end
end

function plotDesign(xx,yy,DesignX,DesignY,rhoFilter,sigma_new,rho_initial)
figure(1)
subplot(1,3,1)
patch(xx-1,yy-1,rhoFilter,'EdgeColor','none')
axis tight
axis equal
colormap jet
colorbar
xlim([-1,DesignX+1])
ylim([-1,DesignY+1])
subplot(1,3,2)
patch(xx-1,yy-1,sigma_new,'EdgeColor','none')
axis tight
axis equal
colormap jet
colorbar
xlim([-1,DesignX+1])
ylim([-1,DesignY+1])
colormap jet
colorbar
subplot(1,3,3)
patch(xx-1,yy-1,rho_initial,'EdgeColor','none')
axis tight
axis equal
colormap jet
colorbar
xlim([-1,DesignX+1])
ylim([-1,DesignY+1])
colormap jet
colorbar
end



function x=TypeInt(x)
x=int32(x);
end
function x=TypeReal(x)
x=single(x);
end
