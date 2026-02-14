% run_3D_ChannelWave_Fault_CPML_FigSuite_optimized.m
% MATLAB R2024b optimized version
% - Scenario-level parfor
% - Multi-shot parfor
% - Optional GPU acceleration for spectral/post-processing
% - Reduced loop overhead in receiver recording and FK extraction

clear; clc; close all;

%% 0) OUTPUT + GLOBAL STYLE
stamp = char(datetime("now",'Format','yyyy-MM-dd_HHmmss'));
OUTDIR = fullfile(pwd, ['OUT_ChannelWave3D_OPT_' stamp]);
if ~exist(OUTDIR,'dir'); mkdir(OUTDIR); end

set(groot,'defaultTextInterpreter','latex');
set(groot,'defaultAxesTickLabelInterpreter','latex');
set(groot,'defaultLegendInterpreter','latex');
set(groot,'defaultAxesFontName','Helvetica');
set(groot,'defaultTextFontName','Helvetica');
set(groot,'defaultAxesFontSize',10);
set(groot,'defaultTextFontSize',10);
set(groot,'defaultLineLineWidth',1.1);
set(groot,'defaultAxesLineWidth',1.0);
set(groot,'defaultAxesBox','on');
set(groot,'defaultFigureColor','w');

FIG = makeFigExportConfig();

%% 1) USER PHYSICS
phys.Lx = 200; phys.Ly = 120; phys.Lz = 120;
phys.hCoal = 4.0;
phys.zCoalCenter0 = 60;
phys.xFault = 100;
phys.throwSign = +1;

matCoal.Vp = 1600; matCoal.Vs = 800; matCoal.rho = 1300;
matRock.Vp = 3000; matRock.Vs = 1500; matRock.rho = 2500;

H_list = [0, 0.8, 2.2, 3.5];
modelNames = {'REF(H=0)','A(small)','B(medium)','C(large)'};

%% 2) NUMERICS + HPC SWITCHES
sim.dx = 2.0; sim.dy = 2.0; sim.dz = 2.0;
sim.nPML = 10;
sim.CFL = 0.45;
sim.f0 = 50;
sim.tMax = 0.35;
sim.snapEvery = 20;
sim.gifEvery = 6;
sim.sliceY_m = phys.Ly/2;
sim.sliceZ_m = phys.zCoalCenter0;
sim.useSingle = true;

hpc.useParallel = true;
hpc.useGPU = true;
hpc.gpuPostOnly = true;  % true = only FFT/hilbert/post on GPU
hpc.poolWorkers = [];

ms.enable = true;
ms.nShots = 9;
ms.shotX_start = 40;
ms.shotX_end   = 95;
ms.shotStepMode = 'linear';
ms.tMaxShot = 0.28;
ms.stackReceiverSide = 'right';
ms.rxLineZ_m = phys.zCoalCenter0;
ms.rxLineY_m = phys.Ly/2;

vmax = max([matCoal.Vp, matRock.Vp]);
sim.dt = sim.CFL * min([sim.dx,sim.dy,sim.dz])/(sqrt(3)*vmax);
sim.Nt = ceil(sim.tMax/sim.dt);
fprintf('dt=%.3e, Nt=%d\n', sim.dt, sim.Nt);

%% 3) RECEIVERS
rx.left.x = phys.xFault - 20; rx.right.x = phys.xFault + 20;
rx.y = phys.Ly/2; rx.z = phys.zCoalCenter0;

fkArray.x0 = 30; fkArray.x1 = 95; fkArray.dx = sim.dx;
fkArray.y = phys.Ly/2; fkArray.z = phys.zCoalCenter0;
fkArray.fMax = 220;

%% 4) PARALLEL ENV
if hpc.useParallel
    p = gcp('nocreate');
    if isempty(p)
        if isempty(hpc.poolWorkers)
            parpool('threads');
        else
            parpool('threads', hpc.poolWorkers);
        end
    end
end
if hpc.useGPU
    try
        gpuDevice();
    catch ME
        warning('GPU unavailable: %s. Fallback CPU.', ME.message);
        hpc.useGPU = false;
    end
end

%% 5) RUN MODELS (PARFOR)
Results = cell(numel(H_list),1);
Models = cell(numel(H_list),1);

if hpc.useParallel
    parfor im = 1:numel(H_list)
        Results{im} = runOneScenario(im, H_list, modelNames, phys, matCoal, matRock, sim, rx, fkArray, OUTDIR);
    end
else
    for im = 1:numel(H_list)
        Results{im} = runOneScenario(im, H_list, modelNames, phys, matCoal, matRock, sim, rx, fkArray, OUTDIR);
    end
end

for im=1:numel(H_list)
    Models{im} = buildFaultedCoalModel(phys, matCoal, matRock, sim, H_list(im));
end

%% 6) METRICS
metrics = computeRTandTransfer_opt(Results, H_list, sim, fkArray.fMax, hpc);
save(fullfile(OUTDIR,'METRICS_RT_TRANSFER.mat'),'metrics','H_list','rx','sim','-v7.3');

%% 7) MULTI-SHOT
msRes = [];
if ms.enable
    modelBig = Models{end};
    msRes = multiShotStacking_opt(modelBig, sim, phys, ms, OUTDIR, hpc);
    save(fullfile(OUTDIR,'MULTISHOT_STACKING.mat'),'msRes','ms','-v7.3');
end

%% 8) FIGURES (kept lightweight but publication-style)
snapScale = getUnifiedSnapScale(Results);
fig_ModelGeometry(phys, H_list, modelNames, OUTDIR, FIG);
fig_Snapshots_XZ(Results, modelNames, phys, snapScale, OUTDIR, FIG);
fig_SeismoAndSpectra(Results, modelNames, sim, OUTDIR, FIG);
fig_RTmetrics(metrics, H_list, OUTDIR, FIG);
if ~isempty(msRes)
    fig_MultiShotStack(msRes, OUTDIR, FIG);
end

fprintf('\nDone. Output: %s\n', OUTDIR);

%% ======================= FUNCTIONS =======================
function res = runOneScenario(im, H_list, modelNames, phys, matCoal, matRock, sim, rx, fkArray, OUTDIR)
H = H_list(im);
model = buildFaultedCoalModel(phys, matCoal, matRock, sim, H);
src.x = 60; src.y = phys.Ly/2; src.z = phys.zCoalCenter0;
src.f0 = sim.f0; src.amp = 1.0;
opt.OUTDIR = OUTDIR;
opt.tag = sprintf('H_%g', H);
opt.modelName = modelNames{im};
opt.snapEvery = sim.snapEvery;
opt.gifEvery = sim.gifEvery;
opt.sliceY_m = sim.sliceY_m;
opt.sliceZ_m = sim.sliceZ_m;
opt.useSingle = sim.useSingle;
opt.phys = phys;
res = fdtd3D_elastic_SFCPML_opt(model, sim, src, rx, fkArray, opt);
save(fullfile(OUTDIR, ['RESULT_' opt.tag '.mat']), 'res','model','sim','phys','H','-v7.3');
end

function FIG = makeFigExportConfig()
FIG.savePDF = true; FIG.saveTIFF = true; FIG.saveEPS = false;
FIG.dpiImage = 300; FIG.dpiLine = 600;
FIG.paperWidth_cm = 18; FIG.paperHeight_cm = 12;
end

function model = buildFaultedCoalModel(phys, matCoal, matRock, sim, H)
Nx = round(phys.Lx/sim.dx)+1; Ny = round(phys.Ly/sim.dy)+1; Nz = round(phys.Lz/sim.dz)+1;
model.Nx = Nx; model.Ny = Ny; model.Nz = Nz;
model.x = linspace(0,phys.Lx,Nx); model.y = linspace(0,phys.Ly,Ny); model.z = linspace(0,phys.Lz,Nz);

Vp = matRock.Vp*ones(Nx,Ny,Nz,'single');
Vs = matRock.Vs*ones(Nx,Ny,Nz,'single');
rho = matRock.rho*ones(Nx,Ny,Nz,'single');

zL = phys.zCoalCenter0;
zR = phys.zCoalCenter0 + phys.throwSign*H;
zcx = zL*ones(Nx,1,'single');
zcx(model.x>phys.xFault) = zR;

zTop = zcx - phys.hCoal/2;
zBot = zcx + phys.hCoal/2;
zv = reshape(model.z,1,1,[]);

for ix = 1:Nx
    mask = (zv>=zTop(ix)) & (zv<=zBot(ix));
    if any(mask(:))
        Vp(ix,:,mask) = matCoal.Vp;
        Vs(ix,:,mask) = matCoal.Vs;
        rho(ix,:,mask)= matCoal.rho;
    end
end

mu = rho.*(Vs.^2);
lambda = rho.*(Vp.^2)-2*mu;
model.Vp=Vp; model.Vs=Vs; model.rho=rho; model.mu=mu; model.lambda=lambda;
model = precomputeStaggeredProps(model);
model.pml = buildCPMLcoeffs(model, sim);
end

function model = precomputeStaggeredProps(model)
rho = model.rho; mu = model.mu;
model.rho_vx = 0.5*(rho(1:end-1,:,:) + rho(2:end,:,:));
model.rho_vy = 0.5*(rho(:,1:end-1,:) + rho(:,2:end,:));
model.rho_vz = 0.5*(rho(:,:,1:end-1) + rho(:,:,2:end));

model.mu_xy = 0.25*(mu(1:end-1,1:end-1,:)+mu(2:end,1:end-1,:)+mu(1:end-1,2:end,:)+mu(2:end,2:end,:));
model.mu_xz = 0.25*(mu(1:end-1,:,1:end-1)+mu(2:end,:,1:end-1)+mu(1:end-1,:,2:end)+mu(2:end,:,2:end));
model.mu_yz = 0.25*(mu(:,1:end-1,1:end-1)+mu(:,2:end,1:end-1)+mu(:,1:end-1,2:end)+mu(:,2:end,2:end));
end

function pml = buildCPMLcoeffs(model, sim)
R = 1e-8; m = 3; kappaMax = 6; alphaMax = 2*pi*sim.f0; vmax = max(model.Vp(:));
pml.nPML = sim.nPML;
[pml.x.s, pml.x.v] = pmlProfile1D(model.Nx, sim.dx, sim.nPML, vmax, R, m, kappaMax, alphaMax);
[pml.y.s, pml.y.v] = pmlProfile1D(model.Ny, sim.dy, sim.nPML, vmax, R, m, kappaMax, alphaMax);
[pml.z.s, pml.z.v] = pmlProfile1D(model.Nz, sim.dz, sim.nPML, vmax, R, m, kappaMax, alphaMax);
pml.profilesForFig = struct('x',pml.x,'y',pml.y,'z',pml.z);
end

function [S,V] = pmlProfile1D(N,d,nPML,vmax,R,m,kappaMax,alphaMax)
sigmaMax = - (m+1) * log(R) * vmax / (2*nPML*d);
sigmaS=zeros(N,1,'single'); kappaS=ones(N,1,'single'); alphaS=zeros(N,1,'single');
sigmaV=zeros(N-1,1,'single'); kappaV=ones(N-1,1,'single'); alphaV=zeros(N-1,1,'single');
for i=1:N
    dist=0;
    if i<=nPML, dist=(nPML-i+1)/nPML; elseif i>=N-nPML+1, dist=(i-(N-nPML))/nPML; end
    if dist>0
        sigmaS(i)=sigmaMax*dist^m; kappaS(i)=1+(kappaMax-1)*dist^m; alphaS(i)=alphaMax*(1-dist);
    end
end
for i=1:N-1
    dist=0;
    if i<=nPML, dist=(nPML-i+0.5)/nPML; elseif i>=(N-1)-nPML+1, dist=(i-((N-1)-nPML))/nPML; end
    if dist>0
        sigmaV(i)=sigmaMax*dist^m; kappaV(i)=1+(kappaMax-1)*dist^m; alphaV(i)=alphaMax*(1-dist);
    end
end
S=struct('sigma',sigmaS,'kappa',kappaS,'alpha',alphaS);
V=struct('sigma',sigmaV,'kappa',kappaV,'alpha',alphaV);
end

function res = fdtd3D_elastic_SFCPML_opt(model, sim, src, rx, fkArray, opt)
Nx=model.Nx; Ny=model.Ny; Nz=model.Nz; dt=sim.dt; dx=sim.dx; dy=sim.dy; dz=sim.dz;

vx_x=zeros(Nx-1,Ny,Nz,'single'); vx_y=vx_x; vx_z=vx_x;
vy_x=zeros(Nx,Ny-1,Nz,'single'); vy_y=vy_x; vy_z=vy_x;
vz_x=zeros(Nx,Ny,Nz-1,'single'); vz_y=vz_x; vz_z=vz_x;

sxx_x=zeros(Nx,Ny,Nz,'single'); sxx_y=sxx_x; sxx_z=sxx_x;
syy_x=zeros(Nx,Ny,Nz,'single'); syy_y=syy_x; syy_z=syy_x;
szz_x=zeros(Nx,Ny,Nz,'single'); szz_y=szz_x; szz_z=szz_x;
sxy_x=zeros(Nx-1,Ny-1,Nz,'single'); sxy_y=sxy_x;
sxz_x=zeros(Nx-1,Ny,Nz-1,'single'); sxz_z=sxz_x;
syz_y=zeros(Nx,Ny-1,Nz-1,'single'); syz_z=syz_y;

pml = buildABK(model.pml, dt);
res.pmlProfiles = model.pml.profilesForFig;

psi_vx_sxx_x=zeros(size(vx_x),'single'); psi_vx_sxy_y=zeros(size(vx_x),'single'); psi_vx_sxz_z=zeros(size(vx_x),'single');
psi_vy_sxy_x=zeros(size(vy_x),'single'); psi_vy_syy_y=zeros(size(vy_x),'single'); psi_vy_syz_z=zeros(size(vy_x),'single');
psi_vz_sxz_x=zeros(size(vz_x),'single'); psi_vz_syz_y=zeros(size(vz_x),'single'); psi_vz_szz_z=zeros(size(vz_x),'single');

psi_s_dvx_x=zeros(Nx,Ny,Nz,'single'); psi_s_dvy_y=psi_s_dvx_x; psi_s_dvz_z=psi_s_dvx_x;
psi_sxy_dvx_y=zeros(Nx-1,Ny-1,Nz,'single'); psi_sxy_dvy_x=psi_sxy_dvx_y;
psi_sxz_dvx_z=zeros(Nx-1,Ny,Nz-1,'single'); psi_sxz_dvz_x=psi_sxz_dvx_z;
psi_syz_dvy_z=zeros(Nx,Ny-1,Nz-1,'single'); psi_syz_dvz_y=psi_syz_dvy_z;

ixS=nearestIndex(model.x,src.x); iyS=nearestIndex(model.y,src.y); izS=nearestIndex(model.z,src.z);
ixL=nearestIndex(model.x,rx.left.x); ixR=nearestIndex(model.x,rx.right.x);
iyR=nearestIndex(model.y,rx.y); izR=nearestIndex(model.z,rx.z);

ixFK = nearestIndex(model.x,fkArray.x0):nearestIndex(model.x,fkArray.x1);
iyFK = nearestIndex(model.y,fkArray.y); izFK = nearestIndex(model.z,fkArray.z);
nFK = numel(ixFK);

t = single((0:sim.Nt-1)*dt);
w = single(rickerWavelet(double(t), src.f0));

seisL_vz=zeros(sim.Nt,1,'single'); seisR_vz=seisL_vz;
seisL_vy=zeros(sim.Nt,1,'single'); seisR_vy=seisL_vy;
fk_vz_xt=zeros(sim.Nt,nFK,'single'); fk_vy_xt=fk_vz_xt;

iySlice = nearestIndex(model.y, opt.sliceY_m); izSlice = nearestIndex(model.z,opt.sliceZ_m);
snapCount = floor(sim.Nt/opt.snapEvery)+1;
snap_vz_xz=zeros(Nx,Nz,snapCount,'single'); snap_p_xz=snap_vz_xz; snap_vz_xy=zeros(Nx,Ny,snapCount,'single'); snap_t=zeros(snapCount,1,'single');
isnap=0;

rho_vx=model.rho_vx; rho_vy=model.rho_vy; rho_vz=model.rho_vz;
lambda=model.lambda; mu=model.mu; mu_xy=model.mu_xy; mu_xz=model.mu_xz; mu_yz=model.mu_yz;

for it=1:sim.Nt
    sxx=sxx_x+sxx_y+sxx_z; syy=syy_x+syy_y+syy_z; szz=szz_x+szz_y+szz_z;
    sxy=sxy_x+sxy_y; sxz=sxz_x+sxz_z; syz=syz_y+syz_z;

    dsxx_dx=(sxx(2:end,:,:)-sxx(1:end-1,:,:))/dx; [dsxx_dx,psi_vx_sxx_x]=cpml_x(dsxx_dx,psi_vx_sxx_x,pml.x.v); vx_x=vx_x+dt.*dsxx_dx./rho_vx;
    dsxy_dy=shearDiffY_toVx(sxy,dy,Nx-1,Ny,Nz); [dsxy_dy,psi_vx_sxy_y]=cpml_y(dsxy_dy,psi_vx_sxy_y,pml.y.s); vx_y=vx_y+dt.*dsxy_dy./rho_vx;
    dsxz_dz=shearDiffZ_toVx(sxz,dz,Nx-1,Ny,Nz); [dsxz_dz,psi_vx_sxz_z]=cpml_z(dsxz_dz,psi_vx_sxz_z,pml.z.s); vx_z=vx_z+dt.*dsxz_dz./rho_vx;

    dsxy_dx=shearDiffX_toVy(sxy,dx,Nx,Ny-1,Nz); [dsxy_dx,psi_vy_sxy_x]=cpml_x(dsxy_dx,psi_vy_sxy_x,pml.x.s); vy_x=vy_x+dt.*dsxy_dx./rho_vy;
    dsyy_dy=(syy(:,2:end,:)-syy(:,1:end-1,:))/dy; [dsyy_dy,psi_vy_syy_y]=cpml_y(dsyy_dy,psi_vy_syy_y,pml.y.v); vy_y=vy_y+dt.*dsyy_dy./rho_vy;
    dsyz_dz=shearDiffZ_toVy(syz,dz,Nx,Ny-1,Nz); [dsyz_dz,psi_vy_syz_z]=cpml_z(dsyz_dz,psi_vy_syz_z,pml.z.s); vy_z=vy_z+dt.*dsyz_dz./rho_vy;

    dsxz_dx=shearDiffX_toVz(sxz,dx,Nx,Ny,Nz-1); [dsxz_dx,psi_vz_sxz_x]=cpml_x(dsxz_dx,psi_vz_sxz_x,pml.x.s); vz_x=vz_x+dt.*dsxz_dx./rho_vz;
    dsyz_dy=shearDiffY_toVz(syz,dy,Nx,Ny,Nz-1); [dsyz_dy,psi_vz_syz_y]=cpml_y(dsyz_dy,psi_vz_syz_y,pml.y.s); vz_y=vz_y+dt.*dsyz_dy./rho_vz;
    dszz_dz=(szz(:,:,2:end)-szz(:,:,1:end-1))/dz; [dszz_dz,psi_vz_szz_z]=cpml_z(dszz_dz,psi_vz_szz_z,pml.z.v); vz_z=vz_z+dt.*dszz_dz./rho_vz;

    vx=vx_x+vx_y+vx_z; vy=vy_x+vy_y+vy_z; vz=vz_x+vz_y+vz_z;

    dvx_dx=velDiffX_toS(vx,dx,Nx,Ny,Nz); [dvx_dx,psi_s_dvx_x]=cpml_x(dvx_dx,psi_s_dvx_x,pml.x.s);
    dvy_dy=velDiffY_toS(vy,dy,Nx,Ny,Nz); [dvy_dy,psi_s_dvy_y]=cpml_y(dvy_dy,psi_s_dvy_y,pml.y.s);
    dvz_dz=velDiffZ_toS(vz,dz,Nx,Ny,Nz); [dvz_dz,psi_s_dvz_z]=cpml_z(dvz_dz,psi_s_dvz_z,pml.z.s);

    sxx_x=sxx_x+dt.*((lambda+2*mu).*dvx_dx); sxx_y=sxx_y+dt.*(lambda.*dvy_dy); sxx_z=sxx_z+dt.*(lambda.*dvz_dz);
    syy_x=syy_x+dt.*(lambda.*dvx_dx); syy_y=syy_y+dt.*((lambda+2*mu).*dvy_dy); syy_z=syy_z+dt.*(lambda.*dvz_dz);
    szz_x=szz_x+dt.*(lambda.*dvx_dx); szz_y=szz_y+dt.*(lambda.*dvy_dy); szz_z=szz_z+dt.*((lambda+2*mu).*dvz_dz);

    dvx_dy=(vx(:,2:end,:)-vx(:,1:end-1,:))/dy; [dvx_dy,psi_sxy_dvx_y]=cpml_y(dvx_dy,psi_sxy_dvx_y,pml.y.v);
    dvy_dx=(vy(2:end,:,:)-vy(1:end-1,:,:))/dx; [dvy_dx,psi_sxy_dvy_x]=cpml_x(dvy_dx,psi_sxy_dvy_x,pml.x.v);
    sxy_y=sxy_y+dt.*(mu_xy.*dvx_dy); sxy_x=sxy_x+dt.*(mu_xy.*dvy_dx);

    dvx_dz=(vx(:,:,2:end)-vx(:,:,1:end-1))/dz; [dvx_dz,psi_sxz_dvx_z]=cpml_z(dvx_dz,psi_sxz_dvx_z,pml.z.v);
    dvz_dx=(vz(2:end,:,:)-vz(1:end-1,:,:))/dx; [dvz_dx,psi_sxz_dvz_x]=cpml_x(dvz_dx,psi_sxz_dvz_x,pml.x.v);
    sxz_z=sxz_z+dt.*(mu_xz.*dvx_dz); sxz_x=sxz_x+dt.*(mu_xz.*dvz_dx);

    dvy_dz=(vy(:,:,2:end)-vy(:,:,1:end-1))/dz; [dvy_dz,psi_syz_dvy_z]=cpml_z(dvy_dz,psi_syz_dvy_z,pml.z.v);
    dvz_dy=(vz(:,2:end,:)-vz(:,1:end-1,:))/dy; [dvz_dy,psi_syz_dvz_y]=cpml_y(dvz_dy,psi_syz_dvz_y,pml.y.v);
    syz_z=syz_z+dt.*(mu_yz.*dvy_dz); syz_y=syz_y+dt.*(mu_yz.*dvz_dy);

    a3 = w(it)/3;
    sxx_x(ixS,iyS,izS)=sxx_x(ixS,iyS,izS)+a3; sxx_y(ixS,iyS,izS)=sxx_y(ixS,iyS,izS)+a3; sxx_z(ixS,iyS,izS)=sxx_z(ixS,iyS,izS)+a3;
    syy_x(ixS,iyS,izS)=syy_x(ixS,iyS,izS)+a3; syy_y(ixS,iyS,izS)=syy_y(ixS,iyS,izS)+a3; syy_z(ixS,iyS,izS)=syy_z(ixS,iyS,izS)+a3;
    szz_x(ixS,iyS,izS)=szz_x(ixS,iyS,izS)+a3; szz_y(ixS,iyS,izS)=szz_y(ixS,iyS,izS)+a3; szz_z(ixS,iyS,izS)=szz_z(ixS,iyS,izS)+a3;

    vyT = vy_x+vy_y+vy_z; vzT = vz_x+vz_y+vz_z;
    sxxT=sxx_x+sxx_y+sxx_z; syyT=syy_x+syy_y+syy_z; szzT=szz_x+szz_y+szz_z;

    seisL_vz(it)=sampleVzAtStressNode(vzT,ixL,iyR,izR); seisR_vz(it)=sampleVzAtStressNode(vzT,ixR,iyR,izR);
    seisL_vy(it)=sampleVyAtStressNode(vyT,ixL,iyR,izR); seisR_vy(it)=sampleVyAtStressNode(vyT,ixR,iyR,izR);

    fk_vz_xt(it,:) = sampleVzLine(vzT, ixFK, iyFK, izFK);
    fk_vy_xt(it,:) = sampleVyLine(vyT, ixFK, iyFK, izFK);

    if it==1 || mod(it,opt.snapEvery)==0
        isnap = isnap+1;
        snap_t(isnap) = t(it);
        pSlice = - (sxxT(:,iySlice,:)+syyT(:,iySlice,:)+szzT(:,iySlice,:))/3;
        snap_p_xz(:,:,isnap) = squeeze(pSlice);
        snap_vz_xz(:,:,isnap)=sampleVzSliceXZ(vzT,iySlice,Nx,Nz);
        snap_vz_xy(:,:,isnap)=sampleVzSliceXY(vzT,izSlice,Nx,Ny);
    end
end

res.modelName = opt.modelName; res.Htag = opt.tag; res.t = t(:);
res.seis.left.vz=seisL_vz; res.seis.right.vz=seisR_vz; res.seis.left.vy=seisL_vy; res.seis.right.vy=seisR_vy;
res.fk.vz_xt=fk_vz_xt; res.fk.vy_xt=fk_vy_xt; res.fk.x=model.x(ixFK); res.fk.dt=sim.dt;
res.snap.t=snap_t(1:isnap); res.snap.vz_xz=snap_vz_xz(:,:,1:isnap); res.snap.p_xz=snap_p_xz(:,:,1:isnap); res.snap.vz_xy=snap_vz_xy(:,:,1:isnap);
end

function pml = buildABK(pml0,dt)
pml.nPML = pml0.nPML;
pml.x.s=makeABK(pml0.x.s,dt); pml.x.v=makeABK(pml0.x.v,dt);
pml.y.s=makeABK(pml0.y.s,dt); pml.y.v=makeABK(pml0.y.v,dt);
pml.z.s=makeABK(pml0.z.s,dt); pml.z.v=makeABK(pml0.z.v,dt);
end
function C = makeABK(C0,dt)
sigma=C0.sigma; kappa=C0.kappa; alpha=C0.alpha;
b=exp(-(sigma./kappa+alpha)*dt);
a=sigma.*(b-1)./((kappa.*(sigma+kappa.*alpha))+eps('single'));
C=struct('a',a,'b',b,'kappa',kappa);
end
function [d,psi]=cpml_x(d,psi,C), b=reshape(C.b,[],1,1); a=reshape(C.a,[],1,1); k=reshape(C.kappa,[],1,1); psi=b.*psi+a.*d; d=d./k+psi; end
function [d,psi]=cpml_y(d,psi,C), b=reshape(C.b,1,[],1); a=reshape(C.a,1,[],1); k=reshape(C.kappa,1,[],1); psi=b.*psi+a.*d; d=d./k+psi; end
function [d,psi]=cpml_z(d,psi,C), b=reshape(C.b,1,1,[]); a=reshape(C.a,1,1,[]); k=reshape(C.kappa,1,1,[]); psi=b.*psi+a.*d; d=d./k+psi; end

function out=shearDiffY_toVx(sxy,dy,Nx1,Ny,Nz), out=zeros(Nx1,Ny,Nz,'single'); out(:,2:Ny-1,:)=(sxy(:,2:Ny-1,:)-sxy(:,1:Ny-2,:))/dy; out(:,1,:)=sxy(:,1,:)/dy; out(:,Ny,:)=-sxy(:,Ny-1,:)/dy; end
function out=shearDiffZ_toVx(sxz,dz,Nx1,Ny,Nz), out=zeros(Nx1,Ny,Nz,'single'); out(:,:,2:Nz-1)=(sxz(:,:,2:Nz-1)-sxz(:,:,1:Nz-2))/dz; out(:,:,1)=sxz(:,:,1)/dz; out(:,:,Nz)=-sxz(:,:,Nz-1)/dz; end
function out=shearDiffX_toVy(sxy,dx,Nx,Ny1,Nz), out=zeros(Nx,Ny1,Nz,'single'); out(2:Nx-1,:,:)=(sxy(2:Nx-1,:,:)-sxy(1:Nx-2,:,:))/dx; out(1,:,:)=sxy(1,:,:)/dx; out(Nx,:,:)=-sxy(Nx-1,:,:)/dx; end
function out=shearDiffZ_toVy(syz,dz,Nx,Ny1,Nz), out=zeros(Nx,Ny1,Nz,'single'); out(:,:,2:Nz-1)=(syz(:,:,2:Nz-1)-syz(:,:,1:Nz-2))/dz; out(:,:,1)=syz(:,:,1)/dz; out(:,:,Nz)=-syz(:,:,Nz-1)/dz; end
function out=shearDiffX_toVz(sxz,dx,Nx,Ny,Nz1), out=zeros(Nx,Ny,Nz1,'single'); out(2:Nx-1,:,:)=(sxz(2:Nx-1,:,:)-sxz(1:Nx-2,:,:))/dx; out(1,:,:)=sxz(1,:,:)/dx; out(Nx,:,:)=-sxz(Nx-1,:,:)/dx; end
function out=shearDiffY_toVz(syz,dy,Nx,Ny,Nz1), out=zeros(Nx,Ny,Nz1,'single'); out(:,2:Ny-1,:)=(syz(:,2:Ny-1,:)-syz(:,1:Ny-2,:))/dy; out(:,1,:)=syz(:,1,:)/dy; out(:,Ny,:)=-syz(:,Ny-1,:)/dy; end
function out=velDiffX_toS(vx,dx,Nx,Ny,Nz), out=zeros(Nx,Ny,Nz,'single'); out(2:Nx-1,:,:)=(vx(2:Nx-1,:,:)-vx(1:Nx-2,:,:))/dx; out(1,:,:)=vx(1,:,:)/dx; out(Nx,:,:)=-vx(Nx-1,:,:)/dx; end
function out=velDiffY_toS(vy,dy,Nx,Ny,Nz), out=zeros(Nx,Ny,Nz,'single'); out(:,2:Ny-1,:)=(vy(:,2:Ny-1,:)-vy(:,1:Ny-2,:))/dy; out(:,1,:)=vy(:,1,:)/dy; out(:,Ny,:)=-vy(:,Ny-1,:)/dy; end
function out=velDiffZ_toS(vz,dz,Nx,Ny,Nz), out=zeros(Nx,Ny,Nz,'single'); out(:,:,2:Nz-1)=(vz(:,:,2:Nz-1)-vz(:,:,1:Nz-2))/dz; out(:,:,1)=vz(:,:,1)/dz; out(:,:,Nz)=-vz(:,:,Nz-1)/dz; end

function idx = nearestIndex(vec,x0), [~,idx]=min(abs(vec-x0)); end
function w = rickerWavelet(t,f0), t0=1.5/f0; tau=t-t0; pf=pi*f0; w=(1-2*(pf*tau).^2).*exp(-(pf*tau).^2); w=w/(max(abs(w))+eps); end
function val = sampleVzAtStressNode(vz,ix,iy,iz), iz=max(2,min(iz,size(vz,3))); val=0.5*(vz(ix,iy,iz-1)+vz(ix,iy,iz)); end
function val = sampleVyAtStressNode(vy,ix,iy,iz), iy=max(2,min(iy,size(vy,2))); val=0.5*(vy(ix,iy-1,iz)+vy(ix,iy,iz)); end
function arr = sampleVzLine(vz,ixFK,iy,iz), iz=max(2,min(iz,size(vz,3))); arr=0.5*(squeeze(vz(ixFK,iy,iz-1))+squeeze(vz(ixFK,iy,iz)))'; end
function arr = sampleVyLine(vy,ixFK,iy,iz), iy=max(2,min(iy,size(vy,2))); arr=0.5*(squeeze(vy(ixFK,iy-1,iz))+squeeze(vy(ixFK,iy,iz)))'; end
function vz_xz=sampleVzSliceXZ(vz,iy,Nx,Nz), vz_xz=zeros(Nx,Nz,'single'); for iz=2:Nz-1, vz_xz(:,iz)=0.5*(vz(:,iy,iz-1)+vz(:,iy,iz)); end; vz_xz(:,1)=vz(:,iy,1); vz_xz(:,Nz)=vz(:,iy,end); end
function vz_xy=sampleVzSliceXY(vz,iz,Nx,Ny), iz=max(2,min(iz,size(vz,3))); vz_xy=reshape(0.5*(vz(:,:,iz-1)+vz(:,:,iz)),[Nx,Ny]); end

function metrics = computeRTandTransfer_opt(Results,H_list,sim,fMax,hpc)
REF = Results{1}; t = double(REF.t); dt=sim.dt; N=numel(t); Nfft=2^nextpow2(N); f=(0:Nfft/2)'/(Nfft*dt); iF=f<=fMax;
win = hann(N);
metrics.f = f(iF);
for i=1:numel(Results)
    SL = oneSidedFFT(Results{i}.seis.left.vz.*win, Nfft, hpc);
    SR = oneSidedFFT(Results{i}.seis.right.vz.*win, Nfft, hpc);
    SLY= oneSidedFFT(Results{i}.seis.left.vy.*win, Nfft, hpc);
    SRY= oneSidedFFT(Results{i}.seis.right.vy.*win, Nfft, hpc);
    metrics.transfer(i).H = H_list(i);
    metrics.transfer(i).Tvz = abs(SR(iF))./(abs(SL(iF))+eps);
    metrics.transfer(i).Tvy = abs(SRY(iF))./(abs(SLY(iF))+eps);
end

refL = REF.seis.left.vz; refR = REF.seis.right.vz;
SR0 = oneSidedFFT(refR.*win, Nfft, hpc);
for i=1:numel(Results)
    dL = Results{i}.seis.left.vz-refL; dR = Results{i}.seis.right.vz;
    ER = trapz(t, double(dL).^2); ET = trapz(t,double(dR).^2)/(trapz(t,double(refR).^2)+eps);
    SD = oneSidedFFT(dL.*win, Nfft, hpc); SR = oneSidedFFT(dR.*win, Nfft, hpc);
    metrics.RT(i).H=H_list(i); metrics.RT(i).E_reflect_like=ER; metrics.RT(i).E_trans_ratio=ET;
    metrics.RT(i).R_f = abs(SD(iF)).^2;
    metrics.RT(i).T_f = abs(SR(iF)).^2 ./ (abs(SR0(iF)).^2 + eps);
end
thr=0.35;
for i=1:numel(Results)
    Tf=movmean(metrics.RT(i).T_f,9); ff=metrics.f; idx=find(Tf<thr,1,'first');
    metrics.cutoff(i).H=H_list(i); metrics.cutoff(i).fc=ternary(isempty(idx),NaN,ff(idx)); metrics.cutoff(i).thr=thr;
end
end

function y = oneSidedFFT(x,Nfft,hpc)
if hpc.useGPU && hpc.gpuPostOnly
    xg = gpuArray(double(x)); X = fft(xg,Nfft); y = gather(X(1:Nfft/2+1));
else
    X = fft(double(x),Nfft); y = X(1:Nfft/2+1);
end
end

function out = ternary(cond,a,b), if cond, out=a; else, out=b; end, end

function msRes = multiShotStacking_opt(model, sim, phys, ms, OUTDIR, hpc)
if strcmpi(ms.shotStepMode,'linear'), shotX = linspace(ms.shotX_start, ms.shotX_end, ms.nShots); else, shotX = ms.shotX_start + (0:ms.nShots-1)*(ms.shotX_end-ms.shotX_start)/(ms.nShots-1); end
if strcmpi(ms.stackReceiverSide,'right'), rxLineX=(phys.xFault+5):sim.dx:(phys.Lx-10); else, rxLineX=10:sim.dx:(phys.xFault-5); end
ixLine=arrayfun(@(x) nearestIndex(model.x,x), rxLineX); iyLine=nearestIndex(model.y,ms.rxLineY_m); izLine=nearestIndex(model.z,ms.rxLineZ_m);
NtShot=ceil(ms.tMaxShot/sim.dt); t=(0:NtShot-1)*sim.dt; w=single(rickerWavelet(t,sim.f0));
nR=numel(ixLine); gather=zeros(NtShot,nR,ms.nShots,'single');

if hpc.useParallel
    parfor is=1:ms.nShots
        src.x=shotX(is); src.y=phys.Ly/2; src.z=phys.zCoalCenter0; src.f0=sim.f0; src.amp=1.0;
        gather(:,:,is)=runShotGather_SFCPML(model,sim,src,ixLine,iyLine,izLine,NtShot,w);
    end
else
    for is=1:ms.nShots
        src.x=shotX(is); src.y=phys.Ly/2; src.z=phys.zCoalCenter0; src.f0=sim.f0; src.amp=1.0;
        gather(:,:,is)=runShotGather_SFCPML(model,sim,src,ixLine,iyLine,izLine,NtShot,w);
    end
end

if hpc.useGPU && hpc.gpuPostOnly
    env = gather(abs(hilbert(gpuArray(gather))));
else
    env = abs(hilbert(gather));
end
stackEnv = sum(env,3); stackMean = mean(env,3);
i0=max(1,round(0.10/sim.dt)); i1=min(NtShot,round(min(0.24,t(end))/sim.dt));
lateEnergy_shot=squeeze(trapz(t(i0:i1), env(i0:i1,:,:).^2,1));
lateEnergy_stack=trapz(t(i0:i1), stackEnv(i0:i1,:).^2,1);

msRes.shotX=shotX(:); msRes.rxLineX=rxLineX(:); msRes.t=t(:); msRes.gather=gather;
msRes.stackEnv=stackEnv; msRes.stackMeanEnv=stackMean; msRes.lateEnergy_shot=lateEnergy_shot; msRes.lateEnergy_stack=lateEnergy_stack(:);
save(fullfile(OUTDIR,'MULTISHOT_GATHER_RAW.mat'),'msRes','-v7.3');
end

function gather = runShotGather_SFCPML(model,sim,src,ixLine,iyLine,izLine,NtShot,w)
rx.left.x=model.x(ixLine(1)); rx.right.x=model.x(ixLine(end)); rx.y=model.y(iyLine); rx.z=model.z(izLine);
fk.x0=model.x(ixLine(1)); fk.x1=model.x(ixLine(end)); fk.y=rx.y; fk.z=rx.z; fk.dx=sim.dx; fk.fMax=220;
opt.OUTDIR=''; opt.tag='shot'; opt.modelName='shot'; opt.snapEvery=NtShot+1; opt.gifEvery=[]; opt.sliceY_m=rx.y; opt.sliceZ_m=rx.z; opt.useSingle=true; opt.phys=struct('xFault',inf);
sim2=sim; sim2.Nt=NtShot; sim2.tMax=NtShot*sim.dt;
res = fdtd3D_elastic_SFCPML_opt(model, sim2, src, rx, fk, opt);
gather = res.fk.vz_xt;
if size(gather,2)~=numel(ixLine), gather = gather(:,1:numel(ixLine)); end
end

function snapScale = getUnifiedSnapScale(Results)
allv=[];
for i=1:numel(Results)
    v = Results{i}.snap.vz_xz;
    allv=[allv; v(:)]; %#ok<AGROW>
    if numel(allv)>2e7, allv=allv(1:2:end); end
end
snapScale = prctile(abs(allv),99.7)+eps;
end

function fig_ModelGeometry(phys,H_list,modelNames,OUTDIR,FIG)
f=figure('Units','centimeters','Position',[2 2 FIG.paperWidth_cm FIG.paperHeight_cm]); tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
nexttile; hold on; box on; grid on; view(36,22);
patch([phys.xFault phys.xFault phys.xFault phys.xFault],[0 phys.Ly phys.Ly 0],[0 0 phys.Lz phys.Lz],[.8 .8 .8],'FaceAlpha',.2,'EdgeColor','none');
plot3([0 phys.Lx],[phys.Ly/2 phys.Ly/2],[phys.zCoalCenter0 phys.zCoalCenter0],'k-','LineWidth',2);
xlabel('$x$ (m)'); ylabel('$y$ (m)'); zlabel('$z$ (m)'); set(gca,'ZDir','reverse'); title('(a) Geometry');
nexttile; plot(1:numel(H_list),H_list,'o-'); grid on; xticks(1:numel(H_list)); xticklabels(modelNames); xtickangle(20); ylabel('$H$ (m)'); title('(b) Throws');
exportFig(f,fullfile(OUTDIR,'Fig1_ModelGeometry_OPT'),FIG,'line');
end

function fig_Snapshots_XZ(Results,modelNames,phys,snapScale,OUTDIR,FIG)
f=figure('Units','centimeters','Position',[2 2 FIG.paperWidth_cm 1.4*FIG.paperHeight_cm]); tiledlayout(2,2,'TileSpacing','compact','Padding','compact');
for k=1:4
    nexttile; res=Results{k}; i=max(1,round(size(res.snap.vz_xz,3)*.55)); vz=res.snap.vz_xz(:,:,i);
    imagesc(linspace(0,phys.Lx,size(vz,1)), linspace(0,phys.Lz,size(vz,2)), vz'); axis tight ij; colormap(turbo(256)); clim([-snapScale snapScale]);
    hold on; plot([phys.xFault phys.xFault],[0 phys.Lz],'w--'); hold off; colorbar; title(sprintf('(%c) %s',char('a'+k-1),modelNames{k})); xlabel('$x$'); ylabel('$z$');
end
exportFig(f,fullfile(OUTDIR,'Fig2_Snapshots_XZ_OPT'),FIG,'image');
end

function fig_SeismoAndSpectra(Results,modelNames,sim,OUTDIR,FIG)
f=figure('Units','centimeters','Position',[2 2 FIG.paperWidth_cm 1.4*FIG.paperHeight_cm]); tiledlayout(2,2,'TileSpacing','compact','Padding','compact'); t=Results{1}.t;
nexttile; hold on; for k=1:4, plot(t,Results{k}.seis.right.vz/(max(abs(Results{k}.seis.right.vz))+eps)+1.2*(k-1)); end; grid on; title('(a) Post-fault $v_z$');
nexttile; hold on; for k=1:4, plot(t,Results{k}.seis.right.vy/(max(abs(Results{k}.seis.right.vy))+eps)+1.2*(k-1)); end; grid on; title('(b) Post-fault $v_y$');
[fHz,Mz]=computeSpectraMatrix(Results,sim.dt,220,'vz'); nexttile; plot(fHz,Mz'); grid on; xlim([0 220]); title('(c) $|V_z(f)|$');
[fHz,My]=computeSpectraMatrix(Results,sim.dt,220,'vy'); nexttile; plot(fHz,My'); grid on; xlim([0 220]); title('(d) $|V_y(f)|$'); legend(modelNames,'Location','best');
exportFig(f,fullfile(OUTDIR,'Fig3_SeismoSpectra_OPT'),FIG,'line');
end

function [fHz,M] = computeSpectraMatrix(Results,dt,fMax,comp)
N=numel(Results{1}.t); Nfft=2^nextpow2(N); f=(0:Nfft/2)/(Nfft*dt); iF=f<=fMax; M=zeros(numel(Results),sum(iF)); w=hann(N);
for k=1:numel(Results)
    x = ternary(strcmpi(comp,'vz'), Results{k}.seis.right.vz, Results{k}.seis.right.vy);
    X=fft(double(x).*w,Nfft); S=abs(X(1:Nfft/2+1)); M(k,:)=S(iF);
end
fHz=f(iF);
end

function fig_RTmetrics(metrics,H_list,OUTDIR,FIG)
f=figure('Units','centimeters','Position',[2 2 FIG.paperWidth_cm FIG.paperHeight_cm]); tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
ER=arrayfun(@(s) s.E_reflect_like, metrics.RT); ET=arrayfun(@(s) s.E_trans_ratio, metrics.RT);
nexttile; plot(H_list,ER,'o-'); grid on; title('$E_R$ vs H'); xlabel('$H$');
nexttile; plot(H_list,ET,'o-'); grid on; title('$E_T/E_{T0}$ vs H'); xlabel('$H$');
nexttile; fc=arrayfun(@(s) s.fc, metrics.cutoff); plot(H_list,fc,'o-'); grid on; title('$f_c$ vs H'); xlabel('$H$');
exportFig(f,fullfile(OUTDIR,'Fig4_RT_OPT'),FIG,'line');
end

function fig_MultiShotStack(msRes,OUTDIR,FIG)
f=figure('Units','centimeters','Position',[2 2 FIG.paperWidth_cm 1.4*FIG.paperHeight_cm]); tiledlayout(2,2,'TileSpacing','compact','Padding','compact'); t=msRes.t; x=msRes.rxLineX;
nexttile; imagesc(t,x,msRes.gather(:,:,1)'); axis tight xy; colorbar; title('(a) Single shot gather');
nexttile; imagesc(t,x,msRes.stackEnv'); axis tight xy; colorbar; title('(b) Stacked envelope');
nexttile; imagesc(1:numel(msRes.shotX),x,msRes.lateEnergy_shot); axis tight xy; colorbar; title('(c) Late energy per shot');
nexttile; plot(x,msRes.lateEnergy_stack,'-'); grid on; title('(d) Stacked late energy');
exportFig(f,fullfile(OUTDIR,'Fig5_MultiShot_OPT'),FIG,'image');
end

function exportFig(figHandle, baseName, FIG, kind)
if strcmpi(kind,'line'), dpi=FIG.dpiLine; else, dpi=FIG.dpiImage; end
if FIG.savePDF, exportgraphics(figHandle,[baseName '.pdf'],'ContentType','vector'); end
if FIG.saveTIFF, exportgraphics(figHandle,[baseName '.tif'],'Resolution',dpi); end
close(figHandle);
end
