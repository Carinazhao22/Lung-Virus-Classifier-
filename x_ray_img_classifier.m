clear;
clc;
clearvars;
load ('myNet.mat');
fprintf('network loaded\n')

%load test images;
testDatasetPath = fullfile(pwd,'Data','test');
test_imds = imageDatastore(testDatasetPath,'IncludeSubfolders',true,'LabelSource','foldernames');
labelCount = countEachLabel(test_imds);
fprintf('dataset with %d images loaded\n',length(test_imds.Files));
%%
Normal.actual = double(labelCount{2,2});
Normal.true_covid = 0;
Normal.true_virus_pneumonia = 0;
Normal.correct = 0;
Normal.count = 0;

Covid.actual =double(labelCount{1,2});
Covid.true_normal = 0;
Covid.true_virus_pneumonia = 0;
Covid.correct = 0;
Covid.count = 0;

VIRUS_PNEUMONIA.actual = double(labelCount{3,2});
VIRUS_PNEUMONIA.true_normal = 0;
VIRUS_PNEUMONIA.true_covid = 0;
VIRUS_PNEUMONIA.correct = 0;
VIRUS_PNEUMONIA.count = 0;

CNN_normal=[];
CNN_covid19=[];
CNN_virus=[];
%%
for i = 1:length(test_imds.Files)
    if mod(i,100) == 0
        fprintf('%d images classified, %d remaining\n',i,length(test_imds.Files)-i);
    end
    x_ray_img = readimage(test_imds,i);
    if size(x_ray_img,3) ~= 3
        x_ray_img = x_ray_img(:,:,[1 1 1]);
    end
    resized_img = imresize(x_ray_img,[227 227]);
    [pred,score] = classify(myNet,resized_img);
    diagnosis = sprintf(char(pred));
    img_label = sprintf(char(test_imds.Labels(i)));
    if diagnosis == "COVID-19"
        Covid.count = Covid.count + 1;
        CNN_covid19=[CNN_covid19;test_imds.Files(i),1];
        if img_label == "COVID-19"
            Covid.correct = Covid.correct+1;
        elseif img_label == "NORMAL"
            Covid.true_normal = Covid.true_normal+1;
        elseif img_label == "VIRUS"
            Covid.true_virus_pneumonia = Covid.true_virus_pneumonia + 1;
        else
            fprintf('empty label\n');
            break;
        end
    elseif diagnosis == "NORMAL"
        Normal.count = Normal.count + 1;
        CNN_normal=[CNN_normal;test_imds.Files(i),2];
        if img_label == "COVID-19"
            Normal.true_covid = Normal.true_covid+1;
        elseif img_label == "NORMAL"
            Normal.correct = Normal.correct + 1;
        elseif img_label == "VIRUS"
            Normal.true_virus_pneumonia = Normal.true_virus_pneumonia + 1;
        else
            fprintf('empty label\n');
            break;
        end
    elseif diagnosis == "VIRUS"
        VIRUS_PNEUMONIA.count = VIRUS_PNEUMONIA.count + 1;
        CNN_virus=[CNN_virus;test_imds.Files(i),3];
        if img_label == "COVID-19"
            VIRUS_PNEUMONIA.true_covid = VIRUS_PNEUMONIA.true_covid+1;
        elseif img_label == "NORMAL"
            VIRUS_PNEUMONIA.true_normal = VIRUS_PNEUMONIA.true_normal + 1;
        elseif img_label == "VIRUS"
            VIRUS_PNEUMONIA.correct = VIRUS_PNEUMONIA.correct + 1;
        else
            fprintf('empty label\n');
            break;
        end
    else
        fprintf('empty diagnosis\n');
        break;
    end
        
end
CNN_classify=[CNN_normal; CNN_covid19; CNN_virus];
save mydata CNN_classify;
fprintf('all %d images classified\n',length(test_imds.Files));
%%
CovidTPR = Covid.correct / Covid.count;
NormalTPR = Normal.correct /  Normal.count;
VIRUS_PNEUMONIATPR = VIRUS_PNEUMONIA.correct / VIRUS_PNEUMONIA.count;
fprintf('Covid TPR: %f\n',CovidTPR);
fprintf('Normal TPR: %f\n',NormalTPR);
fprintf('VIRUS_PNEUMONIA TPR: %f\n',VIRUS_PNEUMONIATPR);
%%
Labels = {'NORMAL';'COVID-19';'VIRUS_PNEUMONIA';'SUM';'TrueRate'};
ClassifiedNormal = [Normal.correct;Normal.true_covid;Normal.true_virus_pneumonia;Normal.count;NormalTPR];
ClassifiedCovid = [Covid.true_normal;Covid.correct;Covid.true_virus_pneumonia;Covid.count;CovidTPR];
ClassifiedVIRUS_PNEUMONIA = [VIRUS_PNEUMONIA.true_normal;VIRUS_PNEUMONIA.true_covid;VIRUS_PNEUMONIA.correct;VIRUS_PNEUMONIA.count;VIRUS_PNEUMONIATPR];
%Total = [Normal.count;Covid.count;VIRUS_PNEUMONIA.count];
ActualTotal = [Normal.actual;Covid.actual;VIRUS_PNEUMONIA.actual;Normal.actual+Covid.actual+VIRUS_PNEUMONIA.actual;NaN];
WrongClassification = [];
%TrueRate = [NormalTPR;CovidTPR;VIRUS_PNEUMONIATPR;NaN];
T = table(Labels,ClassifiedNormal,ClassifiedCovid,ClassifiedVIRUS_PNEUMONIA,ActualTotal);
fig = uifigure('Position',[500 500 750 350]);
uit = uitable(fig,'Data',T,'Position',[20 20 710 200]);
color_row = [1;2;3];
color_col = [2;3;4];
s = uistyle('BackgroundColor','yellow');
addStyle(uit,s,'cell',[color_row,color_col]);



