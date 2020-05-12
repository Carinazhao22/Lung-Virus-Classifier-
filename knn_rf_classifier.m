%% clear workspace
% clear all;
% close all;
% clc
 
%% -------Machine learning method----------- %%
%% loading training data 
trainingData=imageSet('Data\train','recursive');
trainingData=partition(trainingData,107,'randomize'); %107 samples each type
COVID=trainingData(1);
NORMAL=trainingData(2);
VIRUS=trainingData(3);
fprintf('Data loaded.\n');
%% create models
trainSample = [];
trainType=[];
 
%% covid-19
type=1;
fprintf('Extracting features to build up models....\n'); 
for i=1:COVID.Count
    img=imread(string(COVID.ImageLocation(i)));
     
    if size(img,3)>1
    img=rgb2gray(img);
    end
    img = uint8(255*mat2gray(img));
    
    % The gray-level co-occurrence matrix (GLCM) with 4 directions (0,45,90,135)
    glcm = graycomatrix(img,'Offset',[0 1; -1 1; -1 0;-1 -1]);
    feature = graycoprops(glcm,{'Contrast','Correlation','Energy','Homogeneity'});
    %normlization
        for n = 1:4
           glcm(:,:,n) = glcm(:,:,n)/sum(sum(glcm(:,:,n)));
        end
    entropy = zeros(1,4);
    IDE=zeros(1,4);
        for n = 1:4
         for f = 1:8
          for j = 1:8
            if glcm(f,j,n)~=0
            entropy(n) = -glcm(f,j,n)*log(glcm(f,j,n))+entropy(n); %%entropy
            IDE(n)=IDE(n)+(glcm(f,j,n)/(1+(f-j)^2)); %Inverse Differential Moment
            end
          end
         end
        end
 
     temp= [feature.Contrast, feature.Correlation, feature.Energy, feature.Homogeneity,entropy,IDE];
   
    trainSample = [trainSample;temp];
    trainType=[trainType;type];
end
 
%% normal
type=2;
for i=1:NORMAL.Count
    img=imread(string(NORMAL.ImageLocation(i)));
     
    if size(img,3)>1
    img=rgb2gray(img);
    end
    img = uint8(255*mat2gray(img));
    glcm = graycomatrix(img,'Offset',[0 1; -1 1; -1 0;-1 -1]);
    feature = graycoprops(glcm,{'Contrast','Correlation','Energy','Homogeneity'});
 %normlization
        for n = 1:4
           glcm(:,:,n) = glcm(:,:,n)/sum(sum(glcm(:,:,n)));
        end
 %entropy from glcm
 entropy = zeros(1,4);
 IDE=zeros(1,4);
        for n = 1:4
         for f = 1:8
          for j = 1:8
            if glcm(f,j,n)~=0
            entropy(n) = -glcm(f,j,n)*log(glcm(f,j,n))+entropy(n); 
            IDE(n)=IDE(n)+(glcm(f,j,n)/(1+(f-j)^2));
            end
          end
         end
        end
 
    temp= [feature.Contrast, feature.Correlation, feature.Energy, feature.Homogeneity,entropy,IDE];
   
    trainSample = [trainSample;temp];
    trainType=[trainType;type];
end
 
%% virus
type=3;
for i=1:VIRUS.Count
    img=imread(string(VIRUS.ImageLocation(i)));
    
    if size(img,3)>1
    img=rgb2gray(img);
    end
    img= uint8(255*mat2gray(img));
    glcm = graycomatrix(img,'Offset',[0 1; -1 1; -1 0;-1 -1]);
    feature = graycoprops(glcm,{'Contrast','Correlation','Energy','Homogeneity'});
 %normlization
        for n = 1:4
           glcm(:,:,n) = glcm(:,:,n)/sum(sum(glcm(:,:,n)));
        end
    entropy = zeros(1,4);
    IDE=zeros(1,4);
        for n = 1:4
         for f = 1:8
          for j = 1:8
            if glcm(f,j,n)~=0
            entropy(n) = -glcm(f,j,n)*log(glcm(f,j,n))+entropy(n); %%entropy
            IDE(n)=IDE(n)+(glcm(f,j,n)/(1+(f-j)^2));%inverse different moment
            end
          end
         end
        end
  
    temp= [feature.Contrast, feature.Correlation, feature.Energy, feature.Homogeneity,entropy,IDE];
   
    trainSample = [trainSample;temp];
    trainType=[trainType;type];
end
zero_degree_features=[];
for i=1:4:24
 zero_degree_features = [zero_degree_features, trainSample(:,i)];
end
zero_degree_features = [zero_degree_features, trainType];
save features zero_degree_features;
fprintf('Done wiht extracting features.\n'); 
%% load test data
fprintf('Testing step.\n');
testData=imageSet('Data\test','recursive');
%testData=partition(testData,5,'randomize');
COVID=testData(1);
NORMAL=testData(2);
VIRUS=testData(3);
test = [];
 
%% covid-19
type=1;
for i=1:COVID.Count
    img=imread(string(COVID.ImageLocation(i)));
    img=imresize(img, [227 227]);
    if size(img,3)>1
    img=rgb2gray(img);
    end
    img = uint8(255*mat2gray(img));
 
    % The gray-level co-occurrence matrix (GLCM) with 4 directions (0,45,90,135)
    glcm = graycomatrix(img,'Offset',[0 1; -1 1; -1 0;-1 -1]);
    feature = graycoprops(glcm,{'Contrast','Correlation','Energy','Homogeneity'}); 
    %normlization
        for n = 1:4
           glcm(:,:,n) = glcm(:,:,n)/sum(sum(glcm(:,:,n)));
        end
    entropy = zeros(1,4);
    IDE=zeros(1,4);
        for n = 1:4
         for f = 1:8
          for j = 1:8
            if glcm(f,j,n)~=0
            entropy(n) = -glcm(f,j,n)*log(glcm(f,j,n))+entropy(n); %%entropy
            IDE(n)=IDE(n)+(glcm(f,j,n)/(1+(f-j)^2));
            end
          end
         end
        end
     
    temp= [feature.Contrast, feature.Correlation, feature.Energy, feature.Homogeneity,entropy,IDE,type];
    test = [test;temp];
end
 
%% normal
type=2;
for i=1:NORMAL.Count
    img=imread(string(NORMAL.ImageLocation(i)));
    img=imresize(img, [227 227]);
    if size(img,3)>1
    img=rgb2gray(img);
    end
    img = uint8(255*mat2gray(img));
    glcm = graycomatrix(img,'Offset',[0 1; -1 1; -1 0;-1 -1]);
    feature = graycoprops(glcm,{'Contrast','Correlation','Energy','Homogeneity'});
    %normlization
        for n = 1:4
           glcm(:,:,n) = glcm(:,:,n)/sum(sum(glcm(:,:,n)));
        end
    entropy = zeros(1,4);
    IDE=zeros(1,4);
        for n = 1:4
         for f = 1:8
          for j = 1:8
            if glcm(f,j,n)~=0
            entropy(n) = -glcm(f,j,n)*log(glcm(f,j,n))+entropy(n); %%entropy
            IDE(n)=IDE(n)+(glcm(f,j,n)/(1+(f-j)^2));
            end
          end
         end
        end
     
  temp= [feature.Contrast, feature.Correlation, feature.Energy, feature.Homogeneity,entropy,IDE,type];
  test = [test;temp];
end
 
%% virus
type=3;
for i=1:VIRUS.Count
    img=imread(string(VIRUS.ImageLocation(i)));
    img=imresize(img, [227 227]);
    if size(img,3)>1
    img=rgb2gray(img);
    end
    img= uint8(255*mat2gray(img));
    glcm = graycomatrix(img,'Offset',[0 1; -1 1; -1 0;-1 -1]);
    feature = graycoprops(glcm,{'Contrast','Correlation','Energy','Homogeneity'});
    %normlization
        for n = 1:4
           glcm(:,:,n) = glcm(:,:,n)/sum(sum(glcm(:,:,n)));
        end
    entropy = zeros(1,4);
    IDE=zeros(1,4);
        for n = 1:4
         for f = 1:8
          for j = 1:8
            if glcm(f,j,n)~=0
            entropy(n) = -glcm(f,j,n)*log(glcm(f,j,n))+entropy(n); %%entropy
            IDE(n)=IDE(n)+(glcm(f,j,n)/(1+(f-j)^2));
            end
          end
         end
        end
    
    temp= [feature.Contrast, feature.Correlation, feature.Energy, feature.Homogeneity,entropy,IDE,type];
    test = [test;temp];
end

 
%% random forest
fprintf('Random Forest.\n');
nTrees=500;
B = TreeBagger(nTrees,trainSample,trainType,'OOBPrediction','On','Method', 'classification'); 
%view(B.Trees{1},'Mode','graph')
%view(B.Trees{2},'Mode','graph')
figure;
oobErrorBaggedEnsemble = oobError(B);
plot(oobErrorBaggedEnsemble)
xlabel('Number of grown trees');
ylabel('Out-of-bag classification error');
rffit= predict(B,test(:,1:24)); 
rffit=cell2mat(rffit); 

%% KNN
%find out the best hyperparameters in KNN (K & distance)
fprintf('KNN.\n');
rng(1)
fprintf('Looking for optimal parameters in KNN and doing predictation.\n');
Mdl = fitcknn(trainSample,trainType,'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',struct('AcquisitionFunctionName','expected-improvement-plus'));
K= fitcknn(trainSample,trainType,'NumNeighbors',Mdl.NumNeighbors,'Distance',Mdl.Distance,'Standardize',1);
knnfit= predict(K,test(:,1:24));
a=test(:,25);
 
fprintf('Printing out results.\n');
%% Confusion Matrix from knnR and rfR  
rfCorrCovid=0;
rfCorrNormal=0;
rfCorrVirus=0;
knnCorrCovid=0;
knnCorrNormal=0;
knnCorrVirus=0;
actualCovid=0;
actualNormal=0;
actualVirus=0;
knnErrAsCovid=0;
knnErrAsVirus=0;
rfErrAsCovid=0;
rfErrAsVirus=0;
 
%% normal
for i=1:size(a,1)
    if a(i)==2
        actualNormal=actualNormal+1;
        if knnfit(i)==2
            knnCorrNormal=knnCorrNormal+1;
        elseif knnfit(i)==1
            knnErrAsCovid=knnErrAsCovid+1;
        else
            knnErrAsVirus=knnErrAsVirus+1;
        end
        
        if rffit(i)=='2'
            rfCorrNormal=rfCorrNormal+1;
        elseif rffit(i)=='1'
            rfErrAsCovid=rfErrAsCovid+1;
        else
            rfErrAsVirus=rfErrAsVirus+1;
        end
        
    end
               
end
knnR=[knnCorrNormal knnErrAsCovid knnErrAsVirus,actualNormal];
rfR=[rfCorrNormal rfErrAsCovid rfErrAsVirus,actualNormal];
 
%% covid-19
knnErrAsNormal=0;
knnErrAsVirus=0;
rfErrAsNormal=0;
rfErrAsVirus=0;
 
for i=1:size(a,1)
    if a(i)==1
       actualCovid=actualCovid+1;
        if knnfit(i)==1
            knnCorrCovid=knnCorrCovid+1;
        elseif knnfit(i)==2
            knnErrAsNormal=knnErrAsNormal+1;
        else
            knnErrAsVirus=knnErrAsVirus+1;
        end
        if rffit(i)=='1'
            rfCorrCovid=rfCorrCovid+1;
        elseif rffit(i)=='2'
            rfErrAsNormal=rfErrAsNormal+1;
        else
            rfErrAsVirus=rfErrAsVirus+1;
        end        
    end               
end
knnR=[knnR;knnErrAsNormal,knnCorrCovid,knnErrAsVirus,actualCovid];
rfR=[rfR;rfErrAsNormal,rfCorrCovid,rfErrAsVirus,actualCovid];
 
%% virus
knnErrAsCovid=0;
knnErrAsNormal=0;
rfErrAsCovid=0;
rfErrAsNormal=0;
for i=1:size(a,1)
    if a(i)==3
        actualVirus=actualVirus+1;
        if knnfit(i)==3
            knnCorrVirus=knnCorrVirus+1;
        elseif knnfit(i)==1
            knnErrAsNormal=knnErrAsNormal+1;
        else
            knnErrAsCovid=knnErrAsCovid+1;
        end
        if rffit(i)=='3'
            rfCorrVirus=rfCorrVirus+1;
        elseif rffit(i)=='1'
            rfErrAsCovid=rfErrAsCovid+1;
        else
            rfErrAsCovid=rfErrAsCovid+1;
        end        
    end               
end
knnR=[knnR;knnErrAsNormal,knnErrAsCovid,knnCorrVirus,actualVirus];
rfR=[rfR;rfErrAsNormal,rfErrAsCovid,rfCorrVirus,actualVirus];

 
%% show confusion matrix
totalSumK=[sum(knnR(:,1)),sum(knnR(:,2)),sum(knnR(:,3)),sum(knnR(:,4))];
totalSumR=[sum(rfR(:,1)),sum(rfR(:,2)),sum(rfR(:,3)),sum(rfR(:,4))];
TPRk=[knnR(1,1)/totalSumK(1),knnR(2,2)/totalSumK(2),knnR(3,3)/totalSumK(3),NaN];
TPRr=[rfR(1,1)/totalSumR(1),rfR(2,2)/totalSumR(2),rfR(3,3)/totalSumR(3),NaN];
knnR=[knnR;totalSumK;TPRk];
rfR=[rfR;totalSumR;TPRr];
CMknn=array2table(knnR,'VariableNames',{'(KNN)ClassifiedNormal','ClassifiedCovid','ClassifiedVirus','ActualTotal'},'RowNames',{'NORMAL','COVID-19','VIRUS','SUM','TrueRate'});
CMknn.Properties.Description = 'Confusion Matrix for KNN Results';
CMrf=array2table(rfR,'VariableNames',{'(RF)ClassifiedNormal','ClassifiedCovid','ClassifiedVirus','ActualTotal'},'RowNames',{'NORMAL','COVID-19','VIRUS','SUM','TrueRate'});
CMrf.Properties.Description = 'Confusion Matrix for RF Results';
fig = uifigure;
uit = uitable(fig);
uit.Position=[20 200 515 150];
uit.Data = CMknn;
color_row = [1;2;3];
color_col = [1;2;3];
s = uistyle('BackgroundColor','yellow');
addStyle(uit,s,'cell',[color_row,color_col]);
hold on;
uit = uitable(fig);
uit.Position=[20 20 515 150];
uit.Data = CMrf;
hold off;
color_row = [1;2;3];
color_col = [1;2;3];
s = uistyle('BackgroundColor','yellow');
addStyle(uit,s,'cell',[color_row,color_col]);

%% show classification results （showing this figure only when using 5 test images for each set）
lim = size(a,1)/3;
lim = ceil(1.5*lim);
acturalNo=0;
acturalCo=0;
acturalVi=0;
rfNo=0;
rfCo=0;
rfVi=0;
knnNo=0;
knnCo=0;
knnVi=0;
%figure;
flag=1;
flag1=1;
flag2=1;
flag3=1;
flag4=1;
flag5=1;
flag6=1;
flag7=1;
flag8=1;
%title('Classification Result');
for i=1:size(a,1)
    if i<=COVID.Count
        file = COVID.ImageLocation(i);
	elseif i>COVID.Count && i<=COVID.Count+NORMAL.Count
        file = NORMAL.ImageLocation(i-COVID.Count);
    else
        file = VIRUS.ImageLocation(i-COVID.Count-NORMAL.Count);
    end
    
    if a(i)==1
        acturalCo=acturalCo+1;        
%         subplot(9,lim,lim+acturalCo);imshow(imread(string(file)));
%         if flag==1
%             title('Real situation COVID-19');
%             flag=0;
%         end
        
	elseif a(i)==2
        acturalNo=acturalNo+1;
%         subplot(9,lim,acturalNo);imshow(imread(string(file)));
%         if flag1==1
%             title('Real situation Normal');
%             flag1=0;
%         end

    else
        acturalVi=acturalVi+1;
%         subplot(9,lim,2*lim+acturalVi);imshow(imread(string(file)));
%         if flag2==1
%         title('Real situation Virus');
%         flag2=0;
%         end
    end
    
    if rffit(i)=='1'
        rfCo=rfCo+1;
%         subplot(9,lim,4*lim+rfCo);imshow(imread(string(file)));
%         if flag3==1
%             title('RF Covid-19');
%             flag3=0;
%         end
	elseif rffit(i)=='2'
        rfNo=rfNo+1;
%         subplot(9,lim,3*lim+rfNo);imshow(imread(string(file)));
%         if flag4==1
%             title('RF Normal');
%             flag4=0;
%         end
    else
        rfVi=rfVi+1;
%         subplot(9,lim,5*lim+rfVi);imshow(imread(string(file)));
%         if flag5==1
%             title('RF Virus');
%             flag5=0;
%         end
    end
    
    if knnfit(i)==1
        knnCo=knnCo+1;
%         subplot(9,lim,7*lim+knnCo);imshow(imread(string(file)));
%         if flag6==1
%         title('KNN Covid-19');
%         flag6=0;
%         end
	elseif knnfit(i)==2
        knnNo=knnNo+1;
%         subplot(9,lim,6*lim+knnNo);imshow(imread(string(file)));
%         if flag7==1
%         title('KNN Normal');
%         flag7=0;
%         end
    else
        knnVi=knnVi+1;
%         subplot(9,lim,8*lim+knnVi);imshow(imread(string(file)));
%         if flag8==1
%         title('KNN Virus');
%         flag8=0;
%         end
    end
end
% sgt = sgtitle('Real and Classified Image Results ','Color','Black');
% sgt.FontSize = 20;
%  
%% show bar figure
% figure;
% Y = [acturalNo, rfNo, knnNo; acturalCo, rfCo, knnCo; acturalVi, rfVi, knnVi];
% b = bar(Y);
% ylim([0 lim]);
% for i=1:3
% text(b(i).XEndPoints,b(i).YEndPoints,string(b(i).YData),'HorizontalAlignment','center','VerticalAlignment','bottom');
% end
% set(gca,'xticklabel',{'Normal','Covid-19','Virus'});
% title('The Number of Diseases Classification Results via Different Algorithms','FontWeight','bold');
% xlabel('Diseases','FontWeight','bold');
% ylabel('The Number of Each Type','FontWeight','bold');
% legend('Actual','RF','KNN','FontWeight','bold');

 