%% clear workspace
% clear all;
% close all;
% clc

%% load features file
load('features.mat');
features = zero_degree_features;
covid=[];
normal=[];
virus=[];
for i=1:size(features,1)
    if features(i,7)==1
        covid = [covid;features(i,:)];
    elseif features(i,7)==2
        normal = [normal;features(i,:)];
    else
        virus = [virus;features(i,:)];
    end
end



%% contrast scatter plot

figure(1);
subplot(2,3,1);
plot(normal(:,1),3*ones(1,size(normal,1)),'o');
hold on 
plot(covid(:,1),2*ones(1,size(covid,1)),'o');
plot(virus(:,1),1*ones(1,size(virus,1)),'o');
hold off
ylim([0,4]);
yticks([1,2,3]);
yticklabels({'Virus','COVID-19','Normal'});
title('Contrast','FontWeight','bold');

%% correlation scatter plot
subplot(2,3,2);
plot(normal(:,2),3*ones(1,size(normal,1)),'o');
hold on 
plot(covid(:,2),2*ones(1,size(covid,1)),'o');
plot(virus(:,2),1*ones(1,size(virus,1)),'o');
hold off
ylim([0,4]);
yticks([1,2,3]);
yticklabels({'Virus','COVID-19','Normal'});
title('Correlation','FontWeight','bold');


%% energy scatter plot
subplot(2,3,3);
plot(normal(:,3),3*ones(1,size(normal,1)),'o');
hold on 
plot(covid(:,3),2*ones(1,size(covid,1)),'o');
plot(virus(:,3),1*ones(1,size(virus,1)),'o');
hold off
ylim([0,4]);
yticks([1,2,3]);
yticklabels({'Virus','COVID-19','Normal'});
title('Energy','FontWeight','bold');


%% homogeneity scatter plot
subplot(2,3,4);
plot(normal(:,4),3*ones(1,size(normal,1)),'o');
hold on 
plot(covid(:,4),2*ones(1,size(covid,1)),'o');
plot(virus(:,4),1*ones(1,size(virus,1)),'o');
hold off
ylim([0,4]);
yticks([1,2,3]);
yticklabels({'Virus','COVID-19','Normal'});
title('Homogeneity','FontWeight','bold');


%% entropy scatter plot
subplot(2,3,5);
plot(normal(:,5),3*ones(1,size(normal,1)),'o');
hold on 
plot(covid(:,5),2*ones(1,size(covid,1)),'o');
plot(virus(:,5),1*ones(1,size(virus,1)),'o');
hold off
ylim([0,4]);
yticks([1,2,3]);
yticklabels({'Virus','COVID-19','Normal'});
title('Entropy','FontWeight','bold');


%% IDE scatter plot
subplot(2,3,6);
plot(normal(:,6),3*ones(1,size(normal,1)),'o');
hold on 
plot(covid(:,6),2*ones(1,size(covid,1)),'o');
plot(virus(:,6),1*ones(1,size(virus,1)),'o');
hold off
ylim([0,4]);
yticks([1,2,3]);
yticklabels({'Virus','COVID-19','Normal'});
title('IDE','FontWeight','bold');


sgtitle('The Scatter Plots of Texture Features','FontWeight','bold');