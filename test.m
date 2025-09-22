reset(gpuDevice)
cd(fileparts(mfilename("fullpath")))
clear
close all

FullGPU=0;
if FullGPU==1
    disp("Gradient on GPU")
else
    disp("Gradient on CPU")
end

addpath(genpath("."),genpath("../Functions"))
%%%optimization parameters initialization
clight=299792458;
mu0=4*pi*1e-7;
eps0=1/clight^2/mu0;
delX=0.2104/2/10^3;
delT=0.99*delX/(sqrt(3)*clight);
fctr=5.5e9;
timeSteps=30000;
NumOfSidLobes=2;
PercentageBW =4e9/fctr;
distancePML=30;
CoaxHgt=1;
ProbeHgt=14;
NumFilter=2;
opts.FlagSym=3;
opts.offset1=45;
opts.offset2=45;
Csource1D=0.1/delX;
rho0=0.7;
Cfilter=0*1e-3;
cmapping=8;
Fmapping=@(x) 10.^(cmapping*x-(cmapping-5));
DesignX=200;
DesignY=200;
sizeDesign1=DesignX*(DesignY+1);
sizeDesign2=(DesignX+1)*DesignY;
sizeDesign=sizeDesign1+sizeDesign2;
rmin=sqrt(2)/2;
%%%%save optimization parameters
constSim.rmin=rmin;
constSim.Fmapping=Fmapping;
constSim.rho0=rho0;
constSim.DesignX=DesignX;
constSim.DesignY=DesignY;
constSim.delX=delX;
constSim.delT=delT;
constSim.fctr=fctr;
constSim.timeSteps=timeSteps;
constSim.NumOfSidLobes = NumOfSidLobes;
constSim.PercentageBW =PercentageBW;
constSim.Dim=[DesignX+distancePML*2,DesignY+distancePML*2,35+ProbeHgt];
constSim.AntLoc=[constSim.Dim(1)/2-opts.offset1,constSim.Dim(2)/2-opts.offset2,10];
constSim.Csource1D=Csource1D;
constSim.NumFilter=NumFilter;
constSim.opts=opts;
constSim.rad=10;
constSim.filtervalues=Cfilter;
[numAnts,~]=size(constSim.AntLoc);
[fmaxx,constSim.source]=CreateSource(constSim);
constSim.epscoax0=mu0/((50*pi)^2*eps0);
constSim.Z0=sqrt(mu0/eps0/constSim.epscoax0);
disp("Simulation parameters:")
disp("spatial step="+delX+", time step="+delT + ", frequency="+fctr+", timesteps="+timeSteps+", sidelobes="+NumOfSidLobes+", bandwidth="+PercentageBW);
str="antenna location";
for i=1:size(constSim.AntLoc,1)
    str=str+" Ant"+i+"=["+constSim.AntLoc(i,1)+","+constSim.AntLoc(i,2)+","+constSim.AntLoc(i,3)+"]";
end
disp(str)

%%%%save path
flag_plot=1;
datetime.setDefaultFormats('defaultdate','yyyy-MM-dd')
Forename=string(datetime("today"))+"AntOptSym_"+num2str(opts.offset1)+"_"+num2str(opts.offset2)+"_";
current_path="./temp_results";
if(~exist(current_path,"dir"))
    mkdir(current_path)
end
filenum=0;
filename=Forename+num2str(DesignX)+"_"+num2str(DesignY)+"_"+filenum+".mat";
while(isfile(current_path+"/"+filename))
    filenum=filenum+1;
    filename=Forename+num2str(DesignX)+"_"+num2str(DesignY)+"_"+filenum+".mat";
end
save(current_path+"/"+filename)



%--------------------project_FDTD_3D-----------------------
pml_size=10;
freqRes = [constSim.fctr delX delT];
TimeDim = [timeSteps constSim.Dim(1) constSim.Dim(2) constSim.Dim(3) numAnts 0 0 0 0];
width_coax0=TimeDim(6);
width_coax1=TimeDim(7);
depth_coax0=TimeDim(8);
depth_coax1=TimeDim(9);



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
eps_infx(pml_size+1:constSim.Dim(1)+pml_size,pml_size+1:constSim.Dim(2)+pml_size,pml_size+1:constSim.Dim(3)+pml_size) = estEpsInf;
sigma_sx(pml_size+1:constSim.Dim(1)+pml_size,pml_size+1:constSim.Dim(2)+pml_size,pml_size+1:constSim.Dim(3)+pml_size) = estCond;
eps_infy=eps_infx;
eps_infz=eps_infx;
sigma_sy=sigma_sx;
sigma_sz=sigma_sx;

%FDTD
P = CreateSimulation();
SetConst(P,TypeReal(freqRes),TypeInt(TimeDim));
SetAntennaLocations(P,TypeInt(constSim.AntLoc));
SetSource(P, TypeReal(constSim.source));


sigma_max=5.8e7;
N_coax=(TimeDim(6)+TimeDim(7)+1)*2+(TimeDim(8)+TimeDim(9)+1)*2;
constSim.mu_r=1;
constSim.epscoax= constSim.mu_r*mu0/(50*pi*50*pi*eps0)*(4/N_coax)^2;

Ant(1,1)=constSim.AntLoc(1,1)+pml_size+1;
Ant(1,2)=constSim.AntLoc(1,2)+pml_size+1;
Ant(1,3)=constSim.AntLoc(1,3)+pml_size+1;


UpdateMaterial(P,TypeInt(1),TypeReal(sigma_sx(:)));
UpdateMaterial(P,TypeInt(2),TypeReal(sigma_sy(:)));
UpdateMaterial(P,TypeInt(3),TypeReal(sigma_sz(:)));

UpdateMaterial(P,TypeInt(4),TypeReal(eps_infx(:)));
UpdateMaterial(P,TypeInt(5),TypeReal(eps_infy(:)));
UpdateMaterial(P,TypeInt(6),TypeReal(eps_infz(:)));
%clear eps_infx eps_infy sigma_sx sigma_sy sigma_sz

ReadySimulation(P);
InitField(P);
if(constSim.mu_r~=1)
    CreateMu(P,TypeReal(mu_x),TypeReal(mu_y),TypeReal(mu_z))
end
antdatasize=numAnts*timeSteps;

constSim.source_1D=constSim.source;
Init1D(P,TypeReal(Csource1D),TypeReal(constSim.source_1D));


disp("Transmiting mode is running")
ResetFieldAll(P);
tT1=tic;

extend_dim=[extend_dimX extend_dimY extend_dimZ];
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
    %CoaxUpdate(P,TypeInt([i,0,0]));
    % if(FullGPU==1)
    %     RecordGPUField(P,TypeInt([i,0]));
    % else
    %     RecordForwardField(P,TypeInt(i));
    % end
if(mod(i,10)==0)
    aa=OutputVis(P,uint32(extend_dim(1)*extend_dim(2)*extend_dim(3)),int32(1));
        bb=(reshape(aa,[extend_dim(1) ,extend_dim(2),extend_dim(3)]));
        subplot(3,3,1)
        imagesc(squeeze(bb(Ant(1,1),:,:))')
        colorbar
        axis xy
        axis square
        title("Ex--Xsilce"+num2str(i))
        subplot(3,3,2)
        imagesc(squeeze(bb(:,Ant(1,2),:))')
        colorbar
        axis xy
        axis square
        title("Ex--Ysilce")
        subplot(3,3,3)
        imagesc(squeeze(bb(:,:,Ant(1,3)+20))')
        colorbar
        axis xy
        axis square
        title("Ex--Zsilce")
        subplot(3,3,4)
        aa=OutputVis(P,uint32(extend_dim(1)*extend_dim(2)*extend_dim(3)),int32(2));
        bb=(reshape(aa,[extend_dim(1) ,extend_dim(2),extend_dim(3)]));
        imagesc(squeeze(bb(Ant(1,1),:,:))')
        colorbar
        axis xy
        axis square
        title("Ey--Xsilce")
        subplot(3,3,5)
        imagesc(squeeze(bb(:,Ant(1,2),:))')
        colorbar
        axis xy
        axis square
        title("Ey--Ysilce")
        subplot(3,3,6)
        imagesc(squeeze(bb(:,:,Ant(1,3)+20))')
        colorbar
        axis xy
        axis square
        title("Ey--Zsilce")
        subplot(3,3,7)
        aa=OutputVis(P,uint32(extend_dim(1)*extend_dim(2)*extend_dim(3)),int32(3));
        bb=(reshape(aa,[extend_dim(1) ,extend_dim(2),extend_dim(3)]));
        imagesc(squeeze(bb(Ant(1,1),:,:))')
        colorbar
        axis xy
        axis square
        title("Ez--Xsilce")
        subplot(3,3,8)
        imagesc(squeeze(bb(:,Ant(1,2),:))')
        colorbar
        axis xy
        axis square
        title("Ez--Ysilce")
        subplot(3,3,9)
        imagesc(squeeze(bb(:,:,Ant(1,3)+20))')
        colorbar
        axis xy
        axis square
        title("Ez--Zsilce")
        refresh,pause(0.1)
end
end



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
