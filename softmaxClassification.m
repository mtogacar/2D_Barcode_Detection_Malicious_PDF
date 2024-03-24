% Training
clear all
load('training_birlestir_after1.mat')
load('trainingLabels_after.mat');
training_birlestir_after1 = training_birlestir_after1';
trainTarget = grp2idx(trainingLabels_after);
T = TargetCreation(trainTarget)
net = trainSoftmaxLayer(training_birlestir_after1, T);


%% Testting
load('test_after_birlestir1.mat');
load('testLabels_after.mat');
test_after_birlestir1 = test_after_birlestir1';
testTarget = grp2idx(testLabels_after);
T = TargetCreation(testTarget);

Y = net(test_after_birlestir1);
[~,y] = max(Y);
[~,t] = max(T);
confusion.getMatrix(t,y)


confusionchart(y,t);
