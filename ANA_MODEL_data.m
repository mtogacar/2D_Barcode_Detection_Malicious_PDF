
tic;
clc;
clear;
%imge alma 
% Verisetinin yolunu her bir klas�r i�in "/" kullan�lacak �ekilde a�a��da 
%�rnek yoldaki gibi belirtmen gerekir.. Burada 'dataset' klas�r ismidir. ve
%veri t�rlerinin bulundu�u son klas�r d�r... Bu klas�r ismi sizin veri
%k�mesinde 'data' ise 'data' olarak de�i�tirin..

outputFolder = fullfile('D:/Analizlerim/pdf417 analizi/','dataset'); 


rootFolder = fullfile(outputFolder);



categories = {'benign','malicious'};


%imds olu�turma, buraya kar���lm�yor..

imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');


tbl = countEachLabel(imds)


net =mobilenetv2(); % Burada hangi CNN ile e�itilecekse model ismini yaz�yorsunuz. �r: alexnet(), vgg16(),googlenet() gibi
%analyzeNetwork(mobilenetv2()) % Bu kod CNN modelin katmanlar�, parametreleri hakk�nda size bir �zet ekran ��kt�s� verecek...

numel(net.Layers(end).ClassNames)



imds.ReadFcn = @(filename)readAndPreprocessImage(filename);

[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');  %Hocam burada 0.7 dedi�i oran verisetinizin e�itim verisi oran�%70

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
 
 
%Bu alt sat�r kodlar�nda sadece 16 de�erini de�i�tirebilirsiniz test setinizin mini batch de�eridir.
%16,32,64,128 gibi mini-batch de�erini ayarlayabilirsiniz. Tabi
%bilgisayar�n�z�n ekran kart� GPU destekliyorsa ve minumum 4GB'IN �zerinde
%olmas� gerekir. Yoksa grafik kart� hatalar� al�rs�n��z..
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


% buradaki kod sat�rlar�na kar��may�n...
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
%alttaki OPTS kodlar� i�in;
%mini batch de�erini 8,16,32,64,128 gibi donan�m�n�z�n kapasitesi iyi se
%art�rabilirsiniz. 
%Modelinizin epoch de�erini ayarlayabilirsiniz. Ben 50 se�tim. siz veri
%t�r�ne g�re bu de�eri 0-100 aras� de�i�tirebilirsiniz..
%'InitialLearnRate',1e-4 de�eri ��renme oran�d�r. genelde bu de�er
%kullan�l�r. E�er modelinizin ��renme oran�n� de�i�tirmek isterseniz 1e-3
%veya 1e-5 KULLANAB�L�RS�N�Z. tabi bu durum modelin ba�ar�s�n� olumlu yada
%olumsuz etkiler... deneyerek g�rmek gerekir..
%ValidationFrequency 9 se�tim. bu asl�nda train ve test ba�ar� grafikleri
%��kt�s�nda e�itim verileri her bir iterasyon i�leniyor test ba�ar� �izgisi
%ise her bir 9 iterasyonda bir i�lenmesini istiyorum diyorum. 
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



[netTransfer,info] = trainNetwork(imds,lgraph,opts); %burada CNN modelin e�itime grafiksel olarak ge�iyor. ba�ar� ve kay�p grafikleri..

%A�a��daki kod sat�rlar� karma��kl�k matrisini bize veriyor.. 
[predictedLabels,scores] = classify(netTransfer,testSet);
[cm,cmNames] = confusionmat(testSet.Labels,predictedLabels);

testLabels = testSet.Labels;
confMat = confusionmat(testLabels, predictedLabels); 

sum(diag(confMat))/sum(confMat(:))

confusionchart(testLabels,predictedLabels);

%1000 �zellikli dosya WORKSPACE panalinde 'trainingFeatures.mat' ve
%'testFeatures.mat' dosyalar�d�r. Ayr�ca bu dosyalar�n etiket bilgileri ise
%'trainingLabels.mat' ve 'testLabels.mat' dosyalar�d�r. 
%E�ER TRA�NFEATURES ve TESTFEATURES dosyalar�n� birle�tirip MAK�NE ��RENME
%Y�NTEMLER�NE (SVM, DEC�S�ON TREE, LDA)vermek istiyorsan "append_new.m"
%dosyas�n� derlemen gerekir.. Derlendikten sonra "WORKSPACE" panelinde
%"FinalData.mat" dosyas� olu�ur. bu dosyay� farkl� kaydedip, MAK�NE ��RENME
%y�ntemlerine e�itmek ve sonu� almak i�in verebilirsiniz. Bunun i�ide
%MATLAB'�n APPS men�s�ndeki "clasification learner" alt men�s�n�
%kullanman�z gerekir..








trainingFeatures_after = activations(netTransfer, trainingSet, 'new_fc', ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
 
trainingLabels_after = trainingSet.Labels;

testFeatures_after = activations(netTransfer, testSet, 'new_fc', ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');
 
testLabels_after = testSet.Labels;



