clear;
DataFileName = 'rgb_cm_cutoff_max_per_plot';                      % RGB center of mass
DataFileName = 'surf_norm_cm_cutoff_max_per_plot_df';             % 3D Scanner
DataFile = ['/Users/pless/data/OPEN/' DataFileName '.csv'];

%%
DD = importdata(DataFile,',');
cultivars = DD.textdata(2:end,2);
DAP_String = DD.textdata(2:end,3);   % days after planting
PLOT_String = DD.textdata(2:end,4);   % plot id
features = DD.data(2:end,:);
nF = size(features,2);
%%
[P, PA, PC] = unique(PLOT_String);
nP = max(PC);

[C, IA, IC] = unique(cultivars);                 % Find the unique cultivars.  
                                         % IC is the "cultivar number" for
                                         % each row of data.
for jx = length(DAP_String):-1:1
    DAP(jx) = str2num(DAP_String{jx});
end
DAP = DAP';

%% What dates do we have?
dapCount = []
for ix =1:max(DAP)+1  
    dapCount(ix) = sum(DAP==ix-1);
    possibleDates(ix) = ix-1;
end

validDatesMask = dapCount>0;
validDates = possibleDates(validDatesMask);
nvD = length(validDates);





%% make smoothed features
VIZ = 0;
maxdate = max(DAP(:));
newFeatures = zeros(maxdate+1,max(PC),nF);

for px = 1:max(PC)
    PDX = find(PC == px);
    dates = DAP(PDX);
    for fx = 1:nF
        vals = features(PDX,fx);
        newfeats = zeros(maxdate,1) .* nan;
        validDates = 1+(min(dates):max(dates));
        newfeats(validDates) = csaps(dates,vals,0.1,min(dates):max(dates));
        newFeatures(:,px,fx) = newfeats;
        
        if VIZ
            clf;
            plot(dates,vals,'bx','MarkerSize',10);
            hold on;
            plot((0:maxdate),newFeatures(:,px,fx),'r-');
            drawnow;
            pause(0.1);
        end
    end
    disp(px);
end
   
%% UNSMOOTHED PLOT
VIDEOOUTPUT = 1;
vidName = ['Features_' datestr(now,'yyyymmdd_HH-MM') '_' DataFileName];
vidObj = VideoWriter(vidName,'MPEG-4');
vidObj.FrameRate = 2;
vidObj.Quality = 95;
open(vidObj);

for fx = 1:128
     for icx = 1:50:max(IC)
         clf; hold all;
         % First plot ALL the cultivars over time (ignores loop variable)
         for acx = 1:max(IC)
             CDX = find(IC==acx);
             plot(DAP(CDX),features(CDX,fx),'Color',[.7 .7 .7]);
         end
         
         % then plot the lines for each plot of this cultivar
         CDX = find(IC == icx);
         plotNumsforThisCultivar = unique(PC(CDX));
         for px = 1:length(plotNumsforThisCultivar)
             PDX = find(PC==plotNumsforThisCultivar(px));
             plot(DAP(PDX),features(PDX,fx),'LineWidth', 2);
         end
         title(['Feature ' num2str(fx) ', Cultivar: ' C{icx} ' surf_norm_cm_cutoff_max_per_plot_df.csv'],'Interpreter','none');
         xlabel('Days after planting');
         ylabel('feature value');
         [vals] = sort(features(:,fx));
         ylim([vals(130) vals(end-130)]);
         if VIDEOOUTPUT
             drawnow
             F = getframe(gcf);
             writeVideo(vidObj,F);
         else
             pause;
         end
         
     end
end
if VIDEOOUTPUT, close(vidObj), end

%% SMOOTHED PLOT
VIDEOOUTPUT = 1;

if VIDEOOUTPUT
vidName = ['SmoothedFeatures_' datestr(now,'yyyymmdd_HH-MM') '_' DataFileName];
vidObj = VideoWriter(vidName,'MPEG-4');
vidObj.FrameRate = 2;
vidObj.Quality = 95;
open(vidObj);
end
        
for fx = 1:128
     for icx = 1:50:max(IC)
         clf;
         hold all;
         plot(newFeatures(:,:,fx),'Color',[.7 .7 .7]);     
         
         % then plot the lines for each plot of this cultivar
         CDX = find(IC == icx);
         plotNumsforThisCultivar = unique(PC(CDX));
         plot(newFeatures(:,plotNumsforThisCultivar,fx),'lineWidth',5);
         
         title(['Feature ' num2str(fx) ', Cultivar: ' C{icx} '  File: ' DataFileName],'Interpreter','none');
         xlabel('Days after planting');
         ylabel('feature value');
         F = newFeatures(:,:,fx);
         [vals] = sort(F(isfinite(F(:))));
         ylim([vals(130) vals(end-130)]);
         if VIDEOOUTPUT
             drawnow
             F = getframe(gcf);
             writeVideo(vidObj,F);
         else
             pause;
         end
         
     end
end
if VIDEOOUTPUT, close(vidObj), end
%% Crappy Heritability estimate:
VIDEOOUTPUT = 1;
if VIDEOOUTPUT
    vidName = ['HeritabilityVid_' datestr(now,'yyyymmdd_HH-MM') '_' DataFileName];
    vidObj = VideoWriter(vidName,'MPEG-4');
    vidObj.FrameRate = 2;
    vidObj.Quality = 95;
    open(vidObj);
end

% I need to map plot numbers to cultivars
for px = 1:nP
    PDX = find(PC == px);
    PtoC(px) = IC(PDX(1));
end
%
        %newFeatures(:,px,fx) = newfeats;
HERITABILITY = zeros(128,nF);
for fx = 1:nF
    for dxIndex = 1:length(validDates)
        dx = validDates(dxIndex);
        clf; hold all;
        P = newFeatures(dx,:,fx);
        diffSumSame = 0;
        diffSumDiff = 0;
        count = 0;
        for cx = 1:nP
            PXsame = find(PtoC == cx & isfinite(P));
            if length(PXsame)>1 
                diffSumSame = diffSumSame + (P(PXsame(1))-P(PXsame(2))).^2;
                PXdiff = find(PtoC ~= cx & isfinite(P));
                diffSumDiff = diffSumDiff + mean(   (P(PXsame(1))-P(PXdiff(:))).^2);
                count = count + 1;
            end
            plot(P(PXsame),PtoC(PXsame),'X-','MarkerSize', 3,'Color',[0.7 0.7 0.7]);
            if mod(cx,17)==0
                plot(P(PXsame),PtoC(PXsame),'X-','MarkerSize', 10,'LineWidth',3);
            end
        end
        ExpectedDiff = diffSumDiff / count;        % Expected squared diff to other cultivar
        ExpectedSame = diffSumSame / count;        % Expected squared diff to same cultivars
        CrappyHeritability = 1-ExpectedSame/ExpectedDiff; % My understanding of something related to heritability
        HERITABILITY(fx,dx) = CrappyHeritability;
        
        if count>10 & mod(dx,3)==0
            title(['Feature: ' num2str(fx) ', Day: ' num2str(dx) ', Heritability: ' num2str(CrappyHeritability,'%1.2f') '  File: ' DataFileName ],'Interpreter','none');
            ylabel('Cultivar');
            xlabel('Feature Value');
            drawnow;
            if VIDEOOUTPUT
                drawnow
                F = getframe(gcf);
                writeVideo(vidObj,F);
            else
                pause;
            end
        end
        
    end
end
close(vidObj);
%%
clf
imagesc(HERITABILITY(:,1:3:end)); 
xlabel('time (days after planting/3)');
ylabel('feature number');
colorbar
title([ 'Feature Heritability Summary ' DataFileName],'Interpreter', 'None');

F = getframe(gcf);
imageName = ['HeritabilitySummary_' datestr(now,'yyyymmdd_HH-MM') '_' DataFileName '.jpg'];
imwrite(F.cdata,imageName)
