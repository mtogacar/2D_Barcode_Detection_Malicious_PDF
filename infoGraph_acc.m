% plot info 
close all
load('info.mat')
figure
y = info.TrainingAccuracy;



x=[1 32:32:1878];

lw = 1;
plot(x,y(x),'LineWidth',lw);


hold on
grid on
y2 = info.ValidationAccuracy;
plot(x,y2(x),'LineWidth',lw);

xlim([0 1900])
ylim([0 100]);
xlabel('Iteration');
ylabel('Accuracy (%)');

legend('Training','Test',...
'location','southeast');