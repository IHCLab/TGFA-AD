%=====================================================================
% Programmer: Si-Sheng Young
% E-mail: q38121509@gs.ncku.edu.tw
% Date: 08/24/2024
% -------------------------------------------------------
% Reference:
% Unsupervised Abundance Matrix Reconstruction Transformer-Guided Fractional Attention Mechanism for Hyperspectral Anomaly Detection 
% IEEE Transactions on Neural Networks and Learning Systems
%======================================================================
% Unsupervised Hyperspectral Anomaly Detection Algorithm
% [proposed,time_proposed]=TGFA_AD(data,data_name,training,num_endmenber,eta);
%======================================================================
%  Input
%  data is H-by-W-by-C data cube with height H, width W, and channel C.
%  data_name is the selected data in this demo, i.e., San_Diego.
%  training: A flag deciding whether training ASCR-Former or directly using the saved result. 
%  num_endmenber & eta: The model order and HyperCSI parameter,respectively.
%----------------------------------------------------------------------
%  Output
%  proposed is a H-by-W detection result of TGFA-AD.
%  time is the computation time (in secs).
%========================================================================
close all; clear all; clc;
%% dataloading
addpath(genpath('.\Matlab\'));
addpath(genpath('.\data\'));
training=0;
data_name='San_Diego'; 
[data,map,num_endmember,eta]=load_data(data_name);

%% TGFA-AD
[proposed,attention,initial_detection,time_proposed]=TGFA_AD(data,data_name,training,num_endmember,eta);

%% display
figure('Name','Detection Result')

subplot(1,4,1)
imshow(map);
title('GT');xlabel(["AUC:","TIME:"]);

subplot(1,4,2)
imshow(ImGray2Pseudocolor(initial_detection, 'hot', 255)*1);
title('Initial Detection');xlabel([ROC(initial_detection,map,0)]);

subplot(1,4,3)
imshow(ImGray2Pseudocolor(attention, 'hot', 255)*1);
title('Abundance Attention');xlabel([ROC(attention,map,0)]);

subplot(1,4,4)
imshow(ImGray2Pseudocolor(proposed, 'hot', 255)*1);
title('TGFA-AD');xlabel([ROC(proposed,map,0),time_proposed]);