function img = preProcessing(file)
    % Import image
    img = imread(file); 
    % Hocam sadece aşağıda 224x224 yazılan yeri CNN modelinizin girdi boyutu ne ise onunla değiştirmeniz gerekir. Yoksa model verir. 
    img = imresize(img,[224 224]);
    %img = rgb2gray(img);
end