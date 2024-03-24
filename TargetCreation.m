function Target = TargetCreation(T)
    % This function create a target pattern set. 
    l = length(T);
    n = length(unique(T));
    uniqueSamples = unique(T);

    for i=1:n
        Template(i).T = zeros(n,1);
        Template(i).T(i,1)=1;
    end

    Target=zeros(n,l);

    for i=1:l
        Target(:,i) = Template(find(uniqueSamples==T(i))).T;
    end
end