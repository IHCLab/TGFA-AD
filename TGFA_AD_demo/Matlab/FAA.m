function [Final_result,attention]=FAA(data_L,S_est,model1d)
a = 0.1;
model1d=double(model1d);
mask_x= genMask(a);
[rows, cols, ~]  = size(data_L);
%% Attention Learning

alfa=(model1d*S_est')*inv(S_est*S_est');
attention=((alfa)*S_est).^2;
attention=(1-exp(-1*attention));

%%  Fractional convolution

attention=reshape(attention,rows,cols);
[Fractional_feature]=filtering(data_L,mask_x,attention);
Final_result=sum(Fractional_feature,3);

