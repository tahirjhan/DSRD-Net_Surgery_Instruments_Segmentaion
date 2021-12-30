% DSRD-Net: Surgical Instruments Segmentation Network

clc;
close all;
clear all;

% Specify the network image size. This is typically the same as the traing image sizes.
imageSize = [512 512 3];
classes = [
    "instrument"
    "background"
    ];

labelIDs   = [255 0];

% Specify the number of classes.
numClasses = numel(classes);

% Load the Pretrained Parameters
params = load("F:\1_Surgery_Instruments_Segmentation\Dataset_1\FinalModel\params_2021_09_13__17_04_52.mat");

% Create Layer Graph
% Create the layer graph variable to contain the network layers.
lgraph = layerGraph();

% Add Layer Branches
% Add the branches of the network to the layer graph. Each branch is a linear array of layers.
tempLayers = imageInputLayer([512 512 3],"Name","data","Normalization","zscore");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([7 7],64,"Name","conv1","BiasLearnRateFactor",0,"Padding",[3 3 3 3],"Stride",[2 2],"Bias",params.conv1.Bias,"Weights",params.conv1.Weights)
    batchNormalizationLayer("Name","bn_conv1","Offset",params.bn_conv1.Offset,"Scale",params.bn_conv1.Scale,"TrainedMean",params.bn_conv1.TrainedMean,"TrainedVariance",params.bn_conv1.TrainedVariance)
    reluLayer("Name","conv1_relu")
    maxPooling2dLayer([3 3],"Name","pool1_1","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","res2a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res2a_branch2a.Bias,"Weights",params.res2a_branch2a.Weights)
    batchNormalizationLayer("Name","bn2a_branch2a","Offset",params.bn2a_branch2a.Offset,"Scale",params.bn2a_branch2a.Scale,"TrainedMean",params.bn2a_branch2a.TrainedMean,"TrainedVariance",params.bn2a_branch2a.TrainedVariance)
    reluLayer("Name","res2a_branch2a_relu")
    convolution2dLayer([3 3],64,"Name","res2a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res2a_branch2b.Bias,"Weights",params.res2a_branch2b.Weights)
    batchNormalizationLayer("Name","bn2a_branch2b","Offset",params.bn2a_branch2b.Offset,"Scale",params.bn2a_branch2b.Scale,"TrainedMean",params.bn2a_branch2b.TrainedMean,"TrainedVariance",params.bn2a_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([7 7],64,"Name","conv1|conv","BiasLearnRateFactor",0,"Padding",[3 3 3 3],"Stride",[2 2],"Bias",params.conv1_conv.Bias,"Weights",params.conv1_conv.Weights)
    batchNormalizationLayer("Name","conv1|bn","Offset",params.conv1_bn.Offset,"Scale",params.conv1_bn.Scale,"TrainedMean",params.conv1_bn.TrainedMean,"TrainedVariance",params.conv1_bn.TrainedVariance)
    reluLayer("Name","conv1|relu")
    maxPooling2dLayer([3 3],"Name","pool1_2","Padding",[1 1 1 1],"Stride",[2 2])];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv2_block1_0_bn","Offset",params.conv2_block1_0_bn.Offset,"Scale",params.conv2_block1_0_bn.Scale,"TrainedMean",params.conv2_block1_0_bn.TrainedMean,"TrainedVariance",params.conv2_block1_0_bn.TrainedVariance)
    reluLayer("Name","conv2_block1_0_relu")
    convolution2dLayer([1 1],128,"Name","conv2_block1_1_conv","BiasLearnRateFactor",0,"Bias",params.conv2_block1_1_conv.Bias,"Weights",params.conv2_block1_1_conv.Weights)
    batchNormalizationLayer("Name","conv2_block1_1_bn","Offset",params.conv2_block1_1_bn.Offset,"Scale",params.conv2_block1_1_bn.Scale,"TrainedMean",params.conv2_block1_1_bn.TrainedMean,"TrainedVariance",params.conv2_block1_1_bn.TrainedVariance)
    reluLayer("Name","conv2_block1_1_relu")
    convolution2dLayer([3 3],32,"Name","conv2_block1_2_conv","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2_block1_2_conv.Bias,"Weights",params.conv2_block1_2_conv.Weights)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = depthConcatenationLayer(2,"Name","conv2_block1_concat");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    batchNormalizationLayer("Name","conv2_block2_0_bn","Offset",params.conv2_block2_0_bn.Offset,"Scale",params.conv2_block2_0_bn.Scale,"TrainedMean",params.conv2_block2_0_bn.TrainedMean,"TrainedVariance",params.conv2_block2_0_bn.TrainedVariance)
    reluLayer("Name","conv2_block2_0_relu")
    convolution2dLayer([1 1],128,"Name","conv2_block2_1_conv","BiasLearnRateFactor",0,"Bias",params.conv2_block2_1_conv.Bias,"Weights",params.conv2_block2_1_conv.Weights)
    batchNormalizationLayer("Name","conv2_block2_1_bn","Offset",params.conv2_block2_1_bn.Offset,"Scale",params.conv2_block2_1_bn.Scale,"TrainedMean",params.conv2_block2_1_bn.TrainedMean,"TrainedVariance",params.conv2_block2_1_bn.TrainedVariance)
    reluLayer("Name","conv2_block2_1_relu")
    convolution2dLayer([3 3],32,"Name","conv2_block2_2_conv","BiasLearnRateFactor",0,"Padding","same","Bias",params.conv2_block2_2_conv.Bias,"Weights",params.conv2_block2_2_conv.Weights)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","conv2_block2_concat")
    maxPooling2dLayer([3 3],"Name","pool1_3","Padding",[1 1 1 1],"Stride",[2 2])
    maxPooling2dLayer([3 3],"Name","pool1_4","Padding",[1 1 1 1],"Stride",[2 2])
    convolution2dLayer([1 1],256,"Name","conv","Padding","same")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res2a")
    reluLayer("Name","res2a_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","res2b_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res2b_branch2a.Bias,"Weights",params.res2b_branch2a.Weights)
    batchNormalizationLayer("Name","bn2b_branch2a","Offset",params.bn2b_branch2a.Offset,"Scale",params.bn2b_branch2a.Scale,"TrainedMean",params.bn2b_branch2a.TrainedMean,"TrainedVariance",params.bn2b_branch2a.TrainedVariance)
    reluLayer("Name","res2b_branch2a_relu")
    convolution2dLayer([3 3],64,"Name","res2b_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res2b_branch2b.Bias,"Weights",params.res2b_branch2b.Weights)
    batchNormalizationLayer("Name","bn2b_branch2b","Offset",params.bn2b_branch2b.Offset,"Scale",params.bn2b_branch2b.Scale,"TrainedMean",params.bn2b_branch2b.TrainedMean,"TrainedVariance",params.bn2b_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res2b")
    reluLayer("Name","res2b_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","res3a_branch2a","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Stride",[2 2],"Bias",params.res3a_branch2a.Bias,"Weights",params.res3a_branch2a.Weights)
    batchNormalizationLayer("Name","bn3a_branch2a","Offset",params.bn3a_branch2a.Offset,"Scale",params.bn3a_branch2a.Scale,"TrainedMean",params.bn3a_branch2a.TrainedMean,"TrainedVariance",params.bn3a_branch2a.TrainedVariance)
    reluLayer("Name","res3a_branch2a_relu")
    convolution2dLayer([3 3],128,"Name","res3a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res3a_branch2b.Bias,"Weights",params.res3a_branch2b.Weights)
    batchNormalizationLayer("Name","bn3a_branch2b","Offset",params.bn3a_branch2b.Offset,"Scale",params.bn3a_branch2b.Scale,"TrainedMean",params.bn3a_branch2b.TrainedMean,"TrainedVariance",params.bn3a_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],48,"Name","dec_c2","BiasLearnRateFactor",0,"WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","dec_bn2")
    reluLayer("Name","dec_relu2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","res3a_branch1","BiasLearnRateFactor",0,"Stride",[2 2],"Bias",params.res3a_branch1.Bias,"Weights",params.res3a_branch1.Weights)
    batchNormalizationLayer("Name","bn3a_branch1","Offset",params.bn3a_branch1.Offset,"Scale",params.bn3a_branch1.Scale,"TrainedMean",params.bn3a_branch1.TrainedMean,"TrainedVariance",params.bn3a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res3a")
    reluLayer("Name","res3b_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","res4a_branch1","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2],"Bias",params.res4a_branch1.Bias,"Weights",params.res4a_branch1.Weights)
    batchNormalizationLayer("Name","bn4a_branch1","Offset",params.bn4a_branch1.Offset,"Scale",params.bn4a_branch1.Scale,"TrainedMean",params.bn4a_branch1.TrainedMean,"TrainedVariance",params.bn4a_branch1.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","res4a_branch2a","BiasLearnRateFactor",0,"Padding","same","Stride",[2 2],"Bias",params.res4a_branch2a.Bias,"Weights",params.res4a_branch2a.Weights)
    batchNormalizationLayer("Name","bn4a_branch2a","Offset",params.bn4a_branch2a.Offset,"Scale",params.bn4a_branch2a.Scale,"TrainedMean",params.bn4a_branch2a.TrainedMean,"TrainedVariance",params.bn4a_branch2a.TrainedVariance)
    reluLayer("Name","res4a_branch2a_relu")
    convolution2dLayer([3 3],256,"Name","res4a_branch2b","BiasLearnRateFactor",0,"Padding",[1 1 1 1],"Bias",params.res4a_branch2b.Bias,"Weights",params.res4a_branch2b.Weights)
    batchNormalizationLayer("Name","bn4a_branch2b","Offset",params.bn4a_branch2b.Offset,"Scale",params.bn4a_branch2b.Scale,"TrainedMean",params.bn4a_branch2b.TrainedMean,"TrainedVariance",params.bn4a_branch2b.TrainedVariance)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    additionLayer(2,"Name","res4a")
    reluLayer("Name","res5b_relu")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = additionLayer(2,"Name","addition");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","aspp_Conv_3","BiasLearnRateFactor",0,"DilationFactor",[12 12],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","aspp_BatchNorm_3")
    reluLayer("Name","aspp_Relu_3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","aspp_Conv_1","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","aspp_BatchNorm_1")
    reluLayer("Name","aspp_Relu_1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","aspp_Conv_4","BiasLearnRateFactor",0,"DilationFactor",[18 18],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","aspp_BatchNorm_4")
    reluLayer("Name","aspp_Relu_4")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","aspp_Conv_2","BiasLearnRateFactor",0,"DilationFactor",[6 6],"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","aspp_BatchNorm_2")
    reluLayer("Name","aspp_Relu_2")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(4,"Name","catAspp")
    convolution2dLayer([1 1],256,"Name","dec_c1","BiasLearnRateFactor",0,"WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","dec_bn1")
    reluLayer("Name","dec_relu1")
    transposedConv2dLayer([8 8],256,"Name","dec_upsample1","BiasLearnRateFactor",0,"Cropping",[2 2 2 2],"Stride",[4 4],"WeightLearnRateFactor",0,"Bias",params.dec_upsample1.Bias,"Weights",params.dec_upsample1.Weights)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = crop2dLayer("centercrop","Name","dec_crop1");
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","dec_cat1")
    convolution2dLayer([3 3],256,"Name","dec_c3","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","dec_bn3")
    reluLayer("Name","dec_relu3")
    convolution2dLayer([3 3],256,"Name","dec_c4","BiasLearnRateFactor",0,"Padding","same","WeightLearnRateFactor",10)
    batchNormalizationLayer("Name","dec_bn4")
    reluLayer("Name","dec_relu4")
    convolution2dLayer([1 1],2,"Name","scorer","BiasLearnRateFactor",0,"WeightLearnRateFactor",10)
    transposedConv2dLayer([8 8],2,"Name","dec_upsample2","BiasLearnRateFactor",0,"Cropping",[2 2 2 2],"Stride",[4 4],"WeightLearnRateFactor",0,"Bias",params.dec_upsample2.Bias,"Weights",params.dec_upsample2.Weights)];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    crop2dLayer("centercrop","Name","dec_crop2")
    softmaxLayer("Name","softmax-out")
    pixelClassificationLayer("Name","classification")];
lgraph = addLayers(lgraph,tempLayers);

% clean up helper variable
clear tempLayers;

% Connect Layer Branches
% Connect all the branches of the network to create the network graph.
lgraph = connectLayers(lgraph,"data","conv1");
lgraph = connectLayers(lgraph,"data","conv1|conv");
lgraph = connectLayers(lgraph,"data","dec_crop2/ref");
lgraph = connectLayers(lgraph,"pool1_1","res2a_branch2a");
lgraph = connectLayers(lgraph,"pool1_1","res2a/in2");
lgraph = connectLayers(lgraph,"pool1_2","conv2_block1_0_bn");
lgraph = connectLayers(lgraph,"pool1_2","conv2_block1_concat/in1");
lgraph = connectLayers(lgraph,"conv2_block1_2_conv","conv2_block1_concat/in2");
lgraph = connectLayers(lgraph,"conv2_block1_concat","conv2_block2_0_bn");
lgraph = connectLayers(lgraph,"conv2_block1_concat","conv2_block2_concat/in1");
lgraph = connectLayers(lgraph,"conv2_block2_2_conv","conv2_block2_concat/in2");
lgraph = connectLayers(lgraph,"conv","addition/in1");
lgraph = connectLayers(lgraph,"bn2a_branch2b","res2a/in1");
lgraph = connectLayers(lgraph,"res2a_relu","res2b_branch2a");
lgraph = connectLayers(lgraph,"res2a_relu","res2b/in2");
lgraph = connectLayers(lgraph,"bn2b_branch2b","res2b/in1");
lgraph = connectLayers(lgraph,"res2b_relu","res3a_branch2a");
lgraph = connectLayers(lgraph,"res2b_relu","dec_c2");
lgraph = connectLayers(lgraph,"res2b_relu","res3a_branch1");
lgraph = connectLayers(lgraph,"bn3a_branch1","res3a/in2");
lgraph = connectLayers(lgraph,"dec_relu2","dec_crop1/ref");
lgraph = connectLayers(lgraph,"dec_relu2","dec_cat1/in1");
lgraph = connectLayers(lgraph,"bn3a_branch2b","res3a/in1");
lgraph = connectLayers(lgraph,"res3b_relu","res4a_branch1");
lgraph = connectLayers(lgraph,"res3b_relu","res4a_branch2a");
lgraph = connectLayers(lgraph,"bn4a_branch2b","res4a/in1");
lgraph = connectLayers(lgraph,"bn4a_branch1","res4a/in2");
lgraph = connectLayers(lgraph,"res5b_relu","addition/in2");
lgraph = connectLayers(lgraph,"addition","aspp_Conv_3");
lgraph = connectLayers(lgraph,"addition","aspp_Conv_1");
lgraph = connectLayers(lgraph,"addition","aspp_Conv_4");
lgraph = connectLayers(lgraph,"addition","aspp_Conv_2");
lgraph = connectLayers(lgraph,"aspp_Relu_1","catAspp/in1");
lgraph = connectLayers(lgraph,"aspp_Relu_3","catAspp/in3");
lgraph = connectLayers(lgraph,"aspp_Relu_4","catAspp/in4");
lgraph = connectLayers(lgraph,"aspp_Relu_2","catAspp/in2");
lgraph = connectLayers(lgraph,"dec_upsample1","dec_crop1/in");
lgraph = connectLayers(lgraph,"dec_crop1","dec_cat1/in2");
lgraph = connectLayers(lgraph,"dec_upsample2","dec_crop2/in");

% Plot Layers
plot(lgraph);

% Set training and validation data
Folder = '/';%directory to all images

%% Training data
train_img_dir = fullfile(Folder,'train/');
imds_train = imageDatastore(train_img_dir); 

train_label_dir = fullfile(Folder,'train_gt/');
pxds_train = pixelLabelDatastore(train_label_dir,classes,labelIDs);

dsTrain = combine(imds_train, pxds_train); 

%%Validation data
val_img_dir = fullfile(Folder,'validation/');
imds_val = imageDatastore(val_img_dir); 

val_label_dir = fullfile(Folder,'validation_gt/');
pxds_val = pixelLabelDatastore(val_label_dir,classes,labelIDs);

dsValidation = combine(imds_val, pxds_val);

% validationFrequency
validationFrequency = 318;


options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',10,...
    'LearnRateDropFactor',0.3,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-2, ...
    'L2Regularization',0.005, ...
    'ValidationData',dsValidation,...
    'MaxEpochs',40, ...  
    'MiniBatchSize',16, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationFrequency',validationFrequency, ...
    'OutputFcn',@(info)stopIfAccuracyNotImproving(info,3));

% Train the network

[net, info] = trainNetwork(dsTrain,lgraph,options);
trained_network_lr2=net;
save trained_network_lr2;

info_trained_network_lr2=info
save info_trained_network_lr2



