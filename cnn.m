X_RayDatasetPath = fullfile(pwd,'Data','train');
imds = imageDatastore(X_RayDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
labelCount = countEachLabel(imds);
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomize');
%%

%%
AlexNetLayers = [
    %1
    imageInputLayer([227 227 3])
    %2
    convolution2dLayer([3 3],96,'Stride',4)
    %3
    reluLayer
    %4
    crossChannelNormalizationLayer(5)
    %5
    maxPooling2dLayer([3 3],'Stride',[2 2])
    %6
    groupedConvolution2dLayer([5 5],128,2,'Stride',1,'Padding',2)
    %7
    reluLayer
    %8
    crossChannelNormalizationLayer(5)
    %9
    maxPooling2dLayer([3 3],'Stride',[2 2],'Padding',0)
    %10
    convolution2dLayer([3 3],384,'Stride',1,'Padding',1)
    %11
    reluLayer
    %12
    groupedConvolution2dLayer([3 3],192,2,'Stride',1,'Padding',1)
    %13
    reluLayer
    %14
     groupedConvolution2dLayer([3 3],192,2,'Stride',1,'Padding',1)
    %15
    reluLayer
    %16
    maxPooling2dLayer([3 3],'Stride',[2 2],'Padding',0)
    %17
    fullyConnectedLayer(1024)
    %18
    reluLayer
    %19
    dropoutLayer(0.5)
    %20
    fullyConnectedLayer(1024)
    %21
    reluLayer
    %22
    dropoutLayer(0.5)
    %23
    fullyConnectedLayer(3)
    softmaxLayer
    classificationLayer];
%%


options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',16, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');

myNet = trainNetwork(imdsTrain,AlexNetLayers,options);
YPred = classify(myNet,imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation);
save ('myNet.mat');