data=[trainingFeatures testFeatures];
Alldata=data';
butun=[trainingLabels' testLabels'];
AllLabel=butun';
FinalLabel = grp2idx(AllLabel);
FinalData=[Alldata FinalLabel];
FinalTarget=grp2idx(AllLabel);

b=max(FinalLabel);
target=zeros(100,b);
for i=1:length(FinalLabel)
    target(i,FinalLabel(i))=1;
end
Target=target';

FinalData2=[Alldata target];