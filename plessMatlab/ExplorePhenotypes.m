% sanity check for phenotypes data
S4 = importdata('~/data/TERRA/s4_phenotypes_ACTUALLY_FINAL.csv',',');
S6 = importdata('~/data/TERRA/s6_phenotypes_ACTUALLY_FINAL.csv',',');

%%
DataStruct = S6;
data = DataStruct.data;                          % get an array for numerical data
featureLabels = DataStruct.textdata(1,6:end);    % Get the list of extracted features.
cultivars = DataStruct.textdata(2:end,5);        % get the actual names of the cultivars (e.g.P12121...)
%%

[C, IA, IC] = unique(cultivars);                 % Find the unique cultivars.  
                                         % IC is the "cultivar number" for
                                         % each row of data.

% IC==X are the locations where you find the X-th cultivar:
cultivars(IC==4)
[jnk idx] = sort(IC);
imagesc(data(idx,:),[0 300]);            % plot stuff to see if we are seeing useful things.
imagesc(data(idx,1:50),[0 300]);

%% 

 vidObj = VideoWriter('~/Desktop/phenotypeHeritability.mp4','MPEG-4');
 vidObj.FrameRate = 2;
 vidObj.Quality = 95;
 open(vidObj);
 
for dataCol = 1:244
    P = data(:,dataCol);                              % Get this data
    
    disp(['dataCol = ' num2str(dataCol)]);            % Print loop index so you can find problems

    % Compute "heritability" as 1 - variance within cultivar/variance btw cultivars
    count = 0;                                        % init counters.
    diffSquaredSameSum = 0;
    diffSquaredDiffSum = 0;
    for ix = 1:max(IC)                                % Loop over cultivars

        thisCultivarIDX = find(IC==ix & P>-900 & isfinite(P));               % Which data comes from this cultivar?
        % -999 is used as data not available.  (and so is nan?)
                                             
        if length(thisCultivarIDX)>1                  % Check if we have multiple measurements?
            diffSquaredSame = (P(thisCultivarIDX(1)) - P(thisCultivarIDX(2)))^2;
            
            diffIdx = find(IC~=ix & P>-900);          % Find all other good measurements
%            randChoice = 1+floor(rand(1,1)*length(diffIdx));
%            diffSquaredDiff = (P(thisCultivarIDX(1)) - P(diffIdx(randChoice)))^2;

            allDiffsSquared = (P(thisCultivarIDX(1)) - P(diffIdx(:))).^2;
            diffSquaredDiff = mean(allDiffsSquared);  % How different are they?

            diffSquaredSameSum = diffSquaredSameSum + diffSquaredSame;
            diffSquaredDiffSum = diffSquaredDiffSum + diffSquaredDiff;
            count = count + 1;
        end
    end
    
    ExpectedDiff = diffSquaredDiffSum / count;        % Expected squared diff to same cultivar
    ExpectedSame = diffSquaredSameSum / count;        % Expected squared diff to other cultivars
    
    CrappyHeritability = 1-ExpectedSame/ExpectedDiff; % My understanding of something related to heritability

    % Make Plot to show      
    PlotTitle = featureLabels{dataCol};               % Get the name of the phenotype
    clf;hold on;
    %for ix = min(IC):max(IC)
    for ix = 1:1:max(IC)                              % loop over cultivars
        thisCultivarIDX = find(IC==ix & P>-900);               % find which points come from this cultivar
        if length(thisCultivarIDX)>0
            for jx = 1:length(thisCultivarIDX)
            plot(P(thisCultivarIDX(jx)),ix,'x', 'MarkerSize',10);   % plot them
        end
        plot([min(P(thisCultivarIDX)) max(P(thisCultivarIDX))], [ix, ix],'-');  % make a line
        end
    end
    
% Organize some stuff to make the plot pretty.
FullTitle = [PlotTitle ': "heritability" = ' num2str(CrappyHeritability,'%1.2f')];
    title(FullTitle,'Interpreter','none');
    ylabel('Cultivar Index (showing every cultivar)');
    xlabel('Phenotype Value');
    drawnow
    F = getframe(gcf);
    writeVideo(vidObj,F);
end
   close(vidObj);
