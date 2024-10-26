function [data,map,num_endmember,eta]=load_data(data_name)

if strcmp(data_name,'San_Diego')
    load('data/San_Diego.mat')
    eta=0.9;
    num_endmember=4;
    f_show=show_HSI(data);
    figure('Name',data_name)
    imshow(f_show);
else
    fprintf('=================================/n')
    fprintf('Please select a vaild data name !!!/n')
    fprintf('=================================/n')
end