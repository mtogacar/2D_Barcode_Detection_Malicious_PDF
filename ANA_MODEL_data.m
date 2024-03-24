
tic;
clc;
clear;
%imge alma 
% Verisetinin yolunu her bir klasör için "/" kullanýlacak þekilde aþaðýda 
%örnek yoldaki gibi belirtmen gerekir.. Burada 'dataset' klasör ismidir. ve
%veri türlerinin bulunduðu son klasör dür... Bu klasör ismi sizin veri
%kümesinde 'data' ise 'data' olarak deðiþtirin..

outputFolder = fullfile('D:/Analizlerim/pdf417 analizi/','dataset'); 


rootFolder = fullfile(outputFolder);



categories = {'benign','malicious'};


%imds oluþturma, buraya karýþýlmýyor..

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');


tbl = countEachLabel(imds)


net =mobilenetv2(); % Burada hangi CNN ile eðitilecekse model ismini yazýyorsunuz. Ör: alexnet(), vgg16(),googlenet() gibi
%analyzeNetwork(mobilenetv2()) % Bu kod CNN modelin katmanlarý, parametreleri hakkýnda size bir özet ekran çýktýsý verecek...

numel(net.Layers(end).ClassNames)



imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');  %Hocam burada 0.7 dediði oran verisetinizin eðitim verisi oraný%70

featureLayer = 'Logits_softmax';


trainingLabels = trainingSet.Labels;

testLabels = testSet.Labels;


layersTransfer = net.Layers(1:end-3);
imageSize = net.Layers(1).InputSize

trainingFeatures = activations(net, trainingSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
 
 
%%%%%%%%%%%%%%%%%%%% Train A Multiclass SVM Classifier Using CNN Features %%%%%%%%%%%%%%%%%%%%

trainingLabels = trainingSet.Labels;

 
%%%%%%%%%%%%%%%%%%%% Evaluate Classifier %%%%%%%%%%%%%%%%%%%%
 
 
%Bu alt satýr kodlarýnda sadece 16 deðerini deðiþtirebilirsiniz test setinizin mini batch deðeridir.
%16,32,64,128 gibi mini-batch deðerini ayarlayabilirsiniz. Tabi
%bilgisayarýnýzýn ekran kartý GPU destekliyorsa ve minumum 4GB'IN üzerinde
%olmasý gerekir. Yoksa grafik kartý hatalarý alýrsýnýýz..
testFeatures = activations(net, testSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');



numClasses = (length(categories))
if isa(net,'SeriesNetwork') 
  lgraph = layerGraph(net.Layers); 
else
  lgraph = layerGraph(net);
end 
[learnableLayer,classLayer] = findLayersToReplace(lgraph);
numClasses = (length(categories))


% buradaki kod satýrlarýna karýþmayýn...
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
    
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:10) = freezeWeights(layers(1:10));
lgraph = createLgraphUsingConnections(layers,connections);
inputSize = net.Layers(1).InputSize
pixelRange = [-30 30];
%alttaki OPTS kodlarý için;
%mini batch deðerini 8,16,32,64,128 gibi donanýmýnýzýn kapasitesi iyi se
%artýrabilirsiniz. 
%Modelinizin epoch deðerini ayarlayabilirsiniz. Ben 50 seçtim. siz veri
%türüne göre bu deðeri 0-100 arasý deðiþtirebilirsiniz..
%'InitialLearnRate',1e-4 deðeri öðrenme oranýdýr. genelde bu deðer
%kullanýlýr. Eðer modelinizin öðrenme oranýný deðiþtirmek isterseniz 1e-3
%veya 1e-5 KULLANABÝLÝRSÝNÝZ. tabi bu durum modelin baþarýsýný olumlu yada
%olumsuz etkiler... deneyerek görmek gerekir..
%ValidationFrequency 9 seçtim. bu aslýnda train ve test baþarý grafikleri
%çýktýsýnda eðitim verileri her bir iterasyon iþleniyor test baþarý çizgisi
%ise her bir 9 iterasyonda bir iþlenmesini istiyorum diyorum. 
opts = trainingOptions('sgdm', ...
    'MaxEpochs',6, ...
    'MiniBatchSize',32, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',32, ... 
    'Verbose',true, ...
    'ValidationData',testSet);



[netTransfer,info] = trainNetwork(imds,lgraph,opts); %burada CNN modelin eðitime grafiksel olarak geçiyor. baþarý ve kayýp grafikleri..

%Aþaðýdaki kod satýrlarý karmaþýklýk matrisini bize veriyor.. 
[predictedLabels,scores] = classify(netTransfer,testSet);
[cm,cmNames] = confusionmat(testSet.Labels,predictedLabels);

testLabels = testSet.Labels;
confMat = confusionmat(testLabels, predictedLabels); 

sum(diag(confMat))/sum(confMat(:))

confusionchart(testLabels,predictedLabels);

%1000 özellikli dosya WORKSPACE panalinde 'trainingFeatures.mat' ve
%'testFeatures.mat' dosyalarýdýr. Ayrýca bu dosyalarýn etiket bilgileri ise
%'trainingLabels.mat' ve 'testLabels.mat' dosyalarýdýr. 
%EÐER TRAÝNFEATURES ve TESTFEATURES dosyalarýný birleþtirip MAKÝNE ÖÐRENME
%YÖNTEMLERÝNE (SVM, DECÝSÝON TREE, LDA)vermek istiyorsan "append_new.m"
%dosyasýný derlemen gerekir.. Derlendikten sonra "WORKSPACE" panelinde
%"FinalData.mat" dosyasý oluþur. bu dosyayý farklý kaydedip, MAKÝNE ÖÐRENME
%yöntemlerine eðitmek ve sonuç almak için verebilirsiniz. Bunun içide
%MATLAB'ýn APPS menüsündeki "clasification learner" alt menüsünü
%kullanmanýz gerekir..








trainingFeatures_after = activations(netTransfer, trainingSet, 'new_fc', ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
 
trainingLabels_after = trainingSet.Labels;

testFeatures_after = activations(netTransfer, testSet, 'new_fc', ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
 
testLabels_after = testSet.Labels;



