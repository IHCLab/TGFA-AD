function [Final]=filtering(data_L,mask_x,attention)

   DataMat =  padding(data_L);
    for i = 1:size(DataMat,3)
        Vert = conv2(DataMat(:,:,i), mask_x, 'same');
        Horiz = conv2(DataMat(:,:,i), mask_x', 'same');
        Cpher = (abs(Vert) + abs(Horiz));  
        Final(:,:,i) = (Cpher(2:end-1, 2:end-1).*attention);
        
    end
    
    
   DataMat =padding(Final);
   for i = 1:size(DataMat,3)
        Vert = conv2(DataMat(:,:,i), mask_x, 'same');
        Horiz = conv2(DataMat(:,:,i), mask_x', 'same');
        Cpher = (abs(Vert) + abs(Horiz));  
        Final(:,:,i) =( Cpher(2:end-1, 2:end-1).*attention);

   end




