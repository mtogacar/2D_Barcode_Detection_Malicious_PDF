function Iout = readAndPreprocessImage(filename)
                
        I = imread(filename);
        
        % Some images may be grayscale. Replicate the image 3 times to
        % create an RGB image. 
        if ismatrix(I)
            I = cat(3,I,I,I);
        end
        
        % Hocam sadece aşağıda 224x224 yazılan yeri CNN modelinizin girdi boyutu ne ise onunla değiştirmeniz gerekir. Yoksa model verir. 
        Iout = imresize(I, [224 224]);  
        
      
    end