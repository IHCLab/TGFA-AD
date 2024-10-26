function [proposed,attention,initial_detection,time_proposed]=TGFA_AD(data,data_name,training,num_endmember,eta)
%% unmixing
t0=tic;
[row,col,band]=size(data); 
data2D=reshape(data,row*col,band)';
[~,abundance]=HyperCSI(data2D,num_endmember,eta); 
abundance3D=reshape((abundance)',row,col,[]); 
%% deep learning
if training==1
    save('Python/HyperCSI_result.mat','abundance3D');
    system('python3 Python/main.py');
    load Python/Cdl.mat
else
    pretrain_pth=strcat('data/ASCR_Former_result/',data_name,'/Cdl.mat');
    load(pretrain_pth)
end
%% initial detection map
initial_detection=nonlinear(abundance3D,Cdl);
%% fractional abundance attention
model1d=reshape(initial_detection,row*col,1)';
a=cat(3,abundance3D,(abundance3D-Cdl));
[final_result,attention]=FAA(a,abundance,model1d);
proposed=(nor(final_result));
time_proposed=toc(t0);

end
