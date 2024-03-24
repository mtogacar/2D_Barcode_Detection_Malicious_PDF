data_after=[trainingFeatures_after testFeatures_after];
Alldata_after=data_after';
butun_after=[trainingLabels_after' testLabels_after'];
AllLabel_after=butun_after';
FinalLabel_after = grp2idx(AllLabel_after);
FinalData_after=[Alldata_after FinalLabel_after];
FinalTarget_after=grp2idx(AllLabel_after);

b_after=max(FinalLabel_after);
target_after=zeros(100,b_after);
for i=1:length(FinalLabel_after)
    target_after(i,FinalLabel_after(i))=1;
end
Target_after=target_after';

FinalData2_after=[Alldata_after target_after];