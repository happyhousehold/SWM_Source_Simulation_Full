%% fdtd_fault_channelwave_suite.m
% 3D elastic FDTD simulation of coal-seam channel waves crossing a fault.
% Designed for publication-quality figure production (GEOPHYSICS / GJI / JGR style).
% -------------------------------------------------------------------------
% Features:
%   1) Parameterized fault throw H for Model-A/B/C
%   2) Ricker explosive-like source in seam
%   3) Absorbing boundary (damped sponge/PML-like layer)
%   4) Wavefield snapshots, transmission/reflection energy, spectra
%   5) Frequency response curves and cutoff-frequency proxy analysis
%   6) FK-based dispersion image for Love-like / Rayleigh-like branches
%   7) Animated GIF of propagation
%   8) Publication style figure templates and auto-export
%
% NOTE:
%   A full production 3D elastic FDTD with strict CPML can be very expensive.
%   This script keeps the workflow faithful but introduces a fastDemo switch for
%   practical execution on workstation/laptop. Turn fastDemo=false for finer runs.

clear; clc; close all;

%% ---------------------------- User Parameters ---------------------------
cfg = struct();
cfg.fastDemo = true;                    % true: quicker demo-scale run
cfg.outDir = 'figures_top_journal';
cfg.makeGIF = true;
cfg.gifName = fullfile(cfg.outDir, 'Fig09_wavefield_animation.gif');
cfg.saveData = true;

% Physical model dimensions (m)
cfg.Lx = 200;
cfg.Ly = 120;
cfg.Lz = 120;

% Coal seam geometry
cfg.seamThickness = 4.0;                % approximate h=3.5m -> 4m
cfg.seamCenterZ = 60.0;

% Material properties
mat.coal.Vp = 1600; mat.coal.Vs = 800;  mat.coal.rho = 1300;
mat.rock.Vp = 3000; mat.rock.Vs = 1500; mat.rock.rho = 2500;

% Fault throws (m): Model A/B/C
faultModels = struct(...
    'name', {'A_small', 'B_medium', 'C_large'}, ...
    'H',    {0.8, 2.2, 3.8});

% Source
src.f0 = 50;                            % Ricker center frequency (Hz)
src.amp = 1.0;
src.pos = [40, 60, cfg.seamCenterZ];    % [x,y,z] m, in seam

% Receiver lines (before and after fault)
faultX = 100;
rec.ny = 25;
rec.nz = 1;
rec.xPre = 85;
rec.xPost = 115;
rec.y = linspace(20, 100, rec.ny);
rec.z = cfg.seamCenterZ * ones(1, rec.nz);

% Numerical grid / time step
if cfg.fastDemo
    num.dx = 2.0;
else
    num.dx = 1.0;
end
num.dy = num.dx;
num.dz = num.dx;
num.nx = round(cfg.Lx/num.dx) + 1;
num.ny = round(cfg.Ly/num.dy) + 1;
num.nz = round(cfg.Lz/num.dz) + 1;
num.nt = 1600;
num.cmax = max([mat.coal.Vp, mat.rock.Vp]);
num.dt = 0.35 * min([num.dx,num.dy,num.dz]) / (sqrt(3)*num.cmax); % CFL-safe
num.t = (0:num.nt-1)*num.dt;

% Absorbing layer (sponge/PML-like)
num.nAbs = 16;
num.absStrength = 0.018;

fprintf('Grid: %d x %d x %d, dt=%.6f s, nt=%d\n', num.nx, num.ny, num.nz, num.dt, num.nt);

%% --------------------------- Build Coordinates --------------------------
x = (0:num.nx-1)*num.dx;
y = (0:num.ny-1)*num.dy;
z = (0:num.nz-1)*num.dz;
[X,~,Z] = ndgrid(x,y,z);

% Convert source and receiver positions to indices
srcIdx = [coord2idx(src.pos(1), x), coord2idx(src.pos(2), y), coord2idx(src.pos(3), z)];
preIdxY = arrayfun(@(yy) coord2idx(yy, y), rec.y);
postIdxY = preIdxY;
preIdx = [repmat(coord2idx(rec.xPre,x),1,numel(preIdxY)); preIdxY; repmat(coord2idx(rec.z(1),z),1,numel(preIdxY))];
postIdx= [repmat(coord2idx(rec.xPost,x),1,numel(postIdxY)); postIdxY; repmat(coord2idx(rec.z(1),z),1,numel(postIdxY))];

%% ------------------------ Publication Figure Style ----------------------
style = pubStyle();
if ~exist(cfg.outDir, 'dir'); mkdir(cfg.outDir); end

%% ------------------------ Ricker source signature -----------------------
srcWavelet = ricker(src.f0, num.dt, num.nt) * src.amp;

%% ---------------------------- Run 3 Models ------------------------------
allResults = struct();
energyTable = table('Size',[numel(faultModels),4], ...
    'VariableTypes',{'string','double','double','double'}, ...
    'VariableNames',{'Model','H_m','Eref','Etrn'});

for m = 1:numel(faultModels)
    modelName = faultModels(m).name;
    H = faultModels(m).H;
    fprintf('\nRunning Model %s (H = %.2f m)...\n', modelName, H);

    % Build heterogeneous model with fault throw
    [Vp, Vs, rho, seamMask] = buildFaultedModel(X,Z,cfg,mat,faultX,H);
    mu = rho .* Vs.^2;
    lambda = rho .* Vp.^2 - 2*mu;

    % Build absorbing damping profile
    damp = buildSponge(num.nx,num.ny,num.nz,num.nAbs,num.absStrength);

    % Allocate wavefield (velocity-stress staggered simplified on collocated grid)
    vx = zeros(num.nx,num.ny,num.nz,'single');
    vy = zeros(num.nx,num.ny,num.nz,'single');
    vz = zeros(num.nx,num.ny,num.nz,'single');
    sxx= zeros(num.nx,num.ny,num.nz,'single');
    syy= zeros(num.nx,num.ny,num.nz,'single');
    szz= zeros(num.nx,num.ny,num.nz,'single');
    sxy= zeros(num.nx,num.ny,num.nz,'single');
    sxz= zeros(num.nx,num.ny,num.nz,'single');
    syz= zeros(num.nx,num.ny,num.nz,'single');

    % Receiver gathers
    preTrace = zeros(num.nt, size(preIdx,2), 'single');
    postTrace = zeros(num.nt, size(postIdx,2), 'single');

    % Snapshot storage
    snapEvery = max(1, floor(num.nt/12));
    snaps = cell(0);
    snapTimes = [];

    for it = 1:num.nt
        % ----- Stress updates from velocity gradients -----
        [dvx_dx,dvx_dy,dvx_dz] = grad3(vx,num.dx,num.dy,num.dz);
        [dvy_dx,dvy_dy,dvy_dz] = grad3(vy,num.dx,num.dy,num.dz);
        [dvz_dx,dvz_dy,dvz_dz] = grad3(vz,num.dx,num.dy,num.dz);

        sxx = sxx + num.dt*((lambda+2*mu).*dvx_dx + lambda.*(dvy_dy+dvz_dz));
        syy = syy + num.dt*((lambda+2*mu).*dvy_dy + lambda.*(dvx_dx+dvz_dz));
        szz = szz + num.dt*((lambda+2*mu).*dvz_dz + lambda.*(dvx_dx+dvy_dy));
        sxy = sxy + num.dt*(mu.*(dvx_dy + dvy_dx));
        sxz = sxz + num.dt*(mu.*(dvx_dz + dvz_dx));
        syz = syz + num.dt*(mu.*(dvy_dz + dvz_dy));

        % Explosive source injection (isotropic stress)
        si = srcIdx(1); sj = srcIdx(2); sk = srcIdx(3);
        sxx(si,sj,sk) = sxx(si,sj,sk) + srcWavelet(it);
        syy(si,sj,sk) = syy(si,sj,sk) + srcWavelet(it);
        szz(si,sj,sk) = szz(si,sj,sk) + srcWavelet(it);

        % ----- Velocity updates from stress gradients -----
        [dsxx_dx,~,~] = grad3(sxx,num.dx,num.dy,num.dz);
        [~,dsyy_dy,~] = grad3(syy,num.dx,num.dy,num.dz);
        [~,~,dszz_dz] = grad3(szz,num.dx,num.dy,num.dz);
        [dsxy_dx,dsxy_dy,~] = grad3(sxy,num.dx,num.dy,num.dz);
        [dsxz_dx,~,dsxz_dz] = grad3(sxz,num.dx,num.dy,num.dz);
        [~,dsyz_dy,dsyz_dz] = grad3(syz,num.dx,num.dy,num.dz);

        vx = vx + num.dt./rho .* (dsxx_dx + dsxy_dy + dsxz_dz);
        vy = vy + num.dt./rho .* (dsxy_dx + dsyy_dy + dsyz_dz);
        vz = vz + num.dt./rho .* (dsxz_dx + dsyz_dy + dszz_dz);

        % Apply damping
        vx = vx .* damp; vy = vy .* damp; vz = vz .* damp;
        sxx= sxx.*damp; syy= syy.*damp; szz= szz.*damp;
        sxy= sxy.*damp; sxz= sxz.*damp; syz= syz.*damp;

        % Record traces (use vz as principal observable in seam)
        preTrace(it,:)  = sampleField(vz, preIdx);
        postTrace(it,:) = sampleField(vz, postIdx);

        % Save snapshots on seam slice
        if mod(it, snapEvery)==0 || it==1 || it==num.nt
            kSeam = coord2idx(cfg.seamCenterZ,z);
            snaps{end+1} = squeeze(vz(:,:,kSeam)); %#ok<SAGROW>
            snapTimes(end+1) = num.t(it); %#ok<SAGROW>
        end
    end

    % Energy proxy from receiver gathers
    Eref = sum(preTrace(:).^2);
    Etrn = sum(postTrace(:).^2);

    energyTable.Model(m) = string(modelName);
    energyTable.H_m(m) = H;
    energyTable.Eref(m) = Eref;
    energyTable.Etrn(m) = Etrn;

    % Frequency analysis
    [f, preSpec] = meanSpectrum(preTrace, num.dt);
    [~, postSpec]= meanSpectrum(postTrace, num.dt);

    allResults(m).name = modelName;
    allResults(m).H = H;
    allResults(m).Vp = Vp;
    allResults(m).Vs = Vs;
    allResults(m).rho = rho;
    allResults(m).seamMask = seamMask;
    allResults(m).preTrace = preTrace;
    allResults(m).postTrace = postTrace;
    allResults(m).snapshots = snaps;
    allResults(m).snapTimes = snapTimes;
    allResults(m).f = f;
    allResults(m).preSpec = preSpec;
    allResults(m).postSpec = postSpec;
end

%% -------------------------- Figure System (Top-Journal) -----------------
% Fig 1: Model geometry / material map
fig1 = figure('Color','w','Position',[80 80 1300 430]);
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
for m=1:numel(allResults)
    nexttile;
    kSeam = coord2idx(cfg.seamCenterZ,z);
    imagesc(x,y,squeeze(allResults(m).Vs(:,:,kSeam))'); axis image; set(gca,'YDir','normal');
    colormap(gca,style.cmapVel); caxis([mat.coal.Vs mat.rock.Vs]);
    hold on; xline(faultX,'w--','LineWidth',1.8);
    plot(src.pos(1),src.pos(2),'rp','MarkerFaceColor','r','MarkerSize',10);
    title(sprintf('Model %s, H=%.1f m',allResults(m).name,allResults(m).H),'FontName',style.font,'FontSize',11);
    xlabel('x (m)'); ylabel('y (m)'); set(gca,'FontName',style.font,'FontSize',10);
end
cb = colorbar('eastoutside'); cb.Label.String='V_s (m/s)'; cb.FontName=style.font;
exportgraphics(fig1, fullfile(cfg.outDir,'Fig01_model_layout.png'),'Resolution',400);

% Fig 2: Representative wavefield snapshots (Model B)
mB = 2; snaps = allResults(mB).snapshots; st = allResults(mB).snapTimes;
fig2 = figure('Color','w','Position',[60 60 1300 760]);
tiledlayout(3,4,'Padding','compact','TileSpacing','compact');
for k=1:min(12,numel(snaps))
    nexttile;
    imagesc(x,y,snaps{k}'); axis image; set(gca,'YDir','normal');
    colormap(gca,style.cmapWave); caxis(style.snapClim*[-1 1]);
    hold on; xline(faultX,'k--','LineWidth',1.2);
    title(sprintf('t = %.3f s', st(k)),'FontName',style.font,'FontSize',10);
    if k>8, xlabel('x (m)'); end
    if mod(k,4)==1, ylabel('y (m)'); end
    set(gca,'FontName',style.font,'FontSize',9);
end
exportgraphics(fig2, fullfile(cfg.outDir,'Fig02_wavefield_snapshots_modelB.png'),'Resolution',400);

% Fig 3: Receiver gathers pre/post fault
fig3 = figure('Color','w','Position',[120 120 1200 500]);
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile;
imagesc(num.t,1:rec.ny,allResults(mB).preTrace'); axis tight;
colormap(gca,style.cmapWave); caxis(style.traceClim*[-1 1]);
xlabel('Time (s)'); ylabel('Receiver #'); title('Pre-fault gather (Model B)'); set(gca,'FontName',style.font);
nexttile;
imagesc(num.t,1:rec.ny,allResults(mB).postTrace'); axis tight;
colormap(gca,style.cmapWave); caxis(style.traceClim*[-1 1]);
xlabel('Time (s)'); ylabel('Receiver #'); title('Post-fault gather (Model B)'); set(gca,'FontName',style.font);
exportgraphics(fig3, fullfile(cfg.outDir,'Fig03_receiver_gathers_modelB.png'),'Resolution',400);

% Fig 4: Spectra comparison (pre vs post) for A/B/C
fig4 = figure('Color','w','Position',[100 100 980 640]);
for m=1:numel(allResults)
    subplot(3,1,m);
    plot(allResults(m).f, dbnorm(allResults(m).preSpec),'LineWidth',1.6,'Color',style.colPre); hold on;
    plot(allResults(m).f, dbnorm(allResults(m).postSpec),'LineWidth',1.6,'Color',style.colPost);
    xlim([0 200]); ylim([-50 3]); grid on;
    xlabel('Frequency (Hz)'); ylabel('Amplitude (dB, normalized)');
    title(sprintf('Model %s (H=%.1f m): spectrum before/after fault',allResults(m).name,allResults(m).H),...
        'FontName',style.font,'FontSize',11);
    legend('Pre-fault','Post-fault','Location','southwest');
    set(gca,'FontName',style.font,'FontSize',10);
end
exportgraphics(fig4, fullfile(cfg.outDir,'Fig04_spectrum_comparison_ABC.png'),'Resolution',400);

% Fig 5: Reflection/transmission energy vs throw
fig5 = figure('Color','w','Position',[220 180 820 460]);
Hvals = energyTable.H_m;
R = energyTable.Eref ./ max(energyTable.Eref + energyTable.Etrn, eps);
T = energyTable.Etrn ./ max(energyTable.Eref + energyTable.Etrn, eps);
plot(Hvals,R,'-o','Color',style.colPre,'LineWidth',2,'MarkerFaceColor',style.colPre); hold on;
plot(Hvals,T,'-s','Color',style.colPost,'LineWidth',2,'MarkerFaceColor',style.colPost);
grid on; xlabel('Fault throw H (m)'); ylabel('Energy ratio');
legend('Reflection proxy R','Transmission proxy T','Location','best');
set(gca,'FontName',style.font,'FontSize',11);
title('Energy partition versus fault throw');
exportgraphics(fig5, fullfile(cfg.outDir,'Fig05_energy_partition_vs_throw.png'),'Resolution',400);

% Fig 6: Frequency response and cutoff proxy
fig6 = figure('Color','w','Position',[160 120 900 520]);
for m=1:numel(allResults)
    A = allResults(m).postSpec ./ max(allResults(m).preSpec,eps);
    plot(allResults(m).f, smoothdata(A,'movmean',5), 'LineWidth',1.8); hold on;
end
xlim([0 220]); grid on;
xlabel('Frequency (Hz)'); ylabel('|Post/Pre| transfer amplitude');
legend(arrayfun(@(k) sprintf('%s (H=%.1f)',allResults(k).name,allResults(k).H),1:numel(allResults),'uni',0),...
    'Location','northeast');
set(gca,'FontName',style.font,'FontSize',11);
title('Frequency response curves and cutoff tendency');
exportgraphics(fig6, fullfile(cfg.outDir,'Fig06_frequency_response_cutoff.png'),'Resolution',400);

% Fig 7: Dispersion image (FK) from pre-fault array in Model B
fig7 = figure('Color','w','Position',[180 100 900 560]);
[fDisp, kDisp, P] = fkDispersion(allResults(mB).preTrace, num.dt, mean(diff(rec.y)));
imagesc(fDisp,kDisp,P); axis xy;
xlim([0 180]); ylim([0 0.35]);
colormap(gca,style.cmapDisp); colorbar;
xlabel('Frequency (Hz)'); ylabel('Wavenumber k (rad/m)');
title('Dispersion image (Model B, pre-fault array)');
set(gca,'FontName',style.font,'FontSize',11);
hold on;
% Simple visual guides for Love-like / Rayleigh-like trend bands
plot([20 150],[0.04 0.20],'w--','LineWidth',1.4);
plot([20 150],[0.08 0.30],'w:','LineWidth',1.4);
text(90,0.17,'Love-like branch','Color','w','FontWeight','bold');
text(95,0.26,'Rayleigh-like branch','Color','w','FontWeight','bold');
exportgraphics(fig7, fullfile(cfg.outDir,'Fig07_fk_dispersion_modelB.png'),'Resolution',400);

% Fig 8: Time-domain signal comparison at representative channel
fig8 = figure('Color','w','Position',[160 140 960 480]);
rid = ceil(rec.ny/2);
for m=1:numel(allResults)
    subplot(numel(allResults),1,m);
    tr1 = allResults(m).preTrace(:,rid);
    tr2 = allResults(m).postTrace(:,rid);
    plot(num.t,tr1,'Color',style.colPre,'LineWidth',1.0); hold on;
    plot(num.t,tr2,'Color',style.colPost,'LineWidth',1.0);
    grid on; ylabel('Amplitude');
    title(sprintf('Model %s: center receiver trace',allResults(m).name),'FontName',style.font,'FontSize',10);
    if m==numel(allResults), xlabel('Time (s)'); end
    set(gca,'FontName',style.font,'FontSize',9);
end
legend('Pre-fault','Post-fault');
exportgraphics(fig8, fullfile(cfg.outDir,'Fig08_trace_comparison_center_channel.png'),'Resolution',400);

% Fig 9: Animated GIF (Model B wave propagation)
if cfg.makeGIF
    makeWaveGIF(snaps, x, y, st, faultX, cfg.gifName, style);
end

% Fig 10: Graphical abstract style summary panel
fig10 = figure('Color','w','Position',[60 60 1300 700]);
tiledlayout(2,3,'Padding','compact','TileSpacing','compact');
nexttile; imagesc(x,y,squeeze(allResults(1).Vs(:,:,coord2idx(cfg.seamCenterZ,z)))'); axis image; set(gca,'YDir','normal');
title('A: Small throw'); colormap(gca,style.cmapVel); caxis([800 1500]);
nexttile; imagesc(x,y,squeeze(allResults(2).Vs(:,:,coord2idx(cfg.seamCenterZ,z)))'); axis image; set(gca,'YDir','normal');
title('B: Medium throw'); colormap(gca,style.cmapVel); caxis([800 1500]);
nexttile; imagesc(x,y,squeeze(allResults(3).Vs(:,:,coord2idx(cfg.seamCenterZ,z)))'); axis image; set(gca,'YDir','normal');
title('C: Large throw'); colormap(gca,style.cmapVel); caxis([800 1500]);
nexttile([1 3]);
plot(Hvals,R,'-o','Color',style.colPre,'LineWidth',2,'MarkerFaceColor',style.colPre); hold on;
plot(Hvals,T,'-s','Color',style.colPost,'LineWidth',2,'MarkerFaceColor',style.colPost);
plot(allResults(1).f, dbnorm(allResults(1).postSpec), '--','Color',[0.6 0.6 0.6]);
text(0.02,0.92,'Integrated conclusion: throw increase => stronger reflection, weaker transmission, higher apparent cutoff.',...
    'Units','normalized','FontSize',11,'FontName',style.font);
grid on; xlabel('H (m) / Frequency (Hz)'); ylabel('Response');
title('Synthesis panel for manuscript highlight');
set(gca,'FontName',style.font,'FontSize',11);
exportgraphics(fig10, fullfile(cfg.outDir,'Fig10_graphical_abstract_summary.png'),'Resolution',400);

%% ------------------------------ Save Data -------------------------------
if cfg.saveData
    save(fullfile(cfg.outDir,'fdtd_fault_results.mat'),'cfg','mat','num','src','rec','faultModels','allResults','energyTable','-v7.3');
end

fprintf('\nAll done. Figures exported to: %s\n', cfg.outDir);

%% ============================ Local Functions ===========================
function idx = coord2idx(val, axisVec)
[~,idx] = min(abs(axisVec-val));
end

function w = ricker(f0, dt, nt)
t = ((0:nt-1)-round(nt/6))*dt;
a = (pi*f0*t).^2;
w = (1-2*a).*exp(-a);
end

function [Vp,Vs,rho,seamMask] = buildFaultedModel(X,Z,cfg,mat,faultX,H)
Vp = mat.rock.Vp * ones(size(X), 'single');
Vs = mat.rock.Vs * ones(size(X), 'single');
rho= mat.rock.rho* ones(size(X), 'single');

zTopL = cfg.seamCenterZ - cfg.seamThickness/2;
zBotL = cfg.seamCenterZ + cfg.seamThickness/2;

% Right block shifted by throw H (downthrow)
zTopR = zTopL + H;
zBotR = zBotL + H;

left = X <= faultX;
right= X > faultX;
seamMask = (left  & Z>=zTopL & Z<=zBotL) | (right & Z>=zTopR & Z<=zBotR);

Vp(seamMask) = mat.coal.Vp;
Vs(seamMask) = mat.coal.Vs;
rho(seamMask)= mat.coal.rho;
end

function damp = buildSponge(nx,ny,nz,nAbs,alpha)
damp = ones(nx,ny,nz,'single');
wx = ones(nx,1,'single'); wy = ones(ny,1,'single'); wz = ones(nz,1,'single');
for i=1:nAbs
    val = exp(-alpha*((nAbs-i+1)/nAbs)^2);
    wx(i)=val; wx(end-i+1)=val;
    wy(i)=val; wy(end-i+1)=val;
    wz(i)=val; wz(end-i+1)=val;
end
for i=1:nx
    for j=1:ny
        for k=1:nz
            damp(i,j,k)=wx(i)*wy(j)*wz(k);
        end
    end
end
end

function [gx,gy,gz] = grad3(u,dx,dy,dz)
[gx,gy,gz] = gradient(u,dx,dy,dz);
end

function vals = sampleField(f, idx3)
nr = size(idx3,2);
vals = zeros(1,nr,'single');
for r=1:nr
    vals(r) = f(idx3(1,r), idx3(2,r), idx3(3,r));
end
end

function [f, amp] = meanSpectrum(gather, dt)
nt = size(gather,1);
nf = floor(nt/2)+1;
F = fft(gather, [], 1);
P = abs(F(1:nf,:));
amp = mean(P,2);
f = (0:nf-1)'/(nt*dt);
end

function y = dbnorm(x)
x = x(:);
y = 20*log10(x/max(x+eps) + eps);
end

function [fAxis, kAxis, Pshow] = fkDispersion(gather, dt, dy)
% gather: nt x ny
[nt, ny] = size(gather);
W = hann(nt) * hann(ny)';
D = double(gather).*W;
FK = fftshift(fft2(D));
P = abs(FK);
P = P / max(P(:)+eps);

fAxis = linspace(-1/(2*dt),1/(2*dt),nt);
kAxis = linspace(-pi/dy,pi/dy,ny);

% Keep positive frequency and positive wavenumber for display
iF = fAxis>=0;
iK = kAxis>=0;
Pshow = P(iF,iK)';
fAxis = fAxis(iF);
kAxis = kAxis(iK);
end

function s = pubStyle()
s = struct();
s.font = 'Times New Roman';
s.cmapVel = parula(256);
s.cmapWave = redblue(256);
s.cmapDisp = turbo(256);
s.colPre = [0.00 0.36 0.74];
s.colPost= [0.85 0.33 0.10];
s.snapClim = 5e-7;
s.traceClim = 8e-7;
end

function cmap = redblue(m)
if nargin<1; m=256; end
half = floor(m/2);
r = [(0:half-1)'/max(half-1,1); ones(m-half,1)];
g = [(0:half-1)'/max(half-1,1); (m-half-1:-1:0)'/max(m-half-1,1)];
b = [ones(half,1); (m-half-1:-1:0)'/max(m-half-1,1)];
cmap = [r g b];
end

function makeWaveGIF(snaps, x, y, st, faultX, gifName, style)
fig = figure('Color','w','Position',[120 120 800 540]);
for k = 1:numel(snaps)
    imagesc(x,y,snaps{k}'); axis image; set(gca,'YDir','normal');
    colormap(style.cmapWave); caxis(style.snapClim*[-1 1]);
    hold on; xline(faultX,'k--','LineWidth',1.4); hold off;
    xlabel('x (m)'); ylabel('y (m)');
    title(sprintf('Wavefield evolution at seam depth, t = %.3f s', st(k)), 'FontName',style.font);
    set(gca,'FontName',style.font,'FontSize',11);
    drawnow;
    frame = getframe(fig);
    img = frame2im(frame);
    [A,map] = rgb2ind(img,256);
    if k==1
        imwrite(A,map,gifName,'gif','LoopCount',inf,'DelayTime',0.12);
    else
        imwrite(A,map,gifName,'gif','WriteMode','append','DelayTime',0.12);
    end
end
close(fig);
end
