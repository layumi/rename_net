clear;
netStruct = load('../imagenet-vgg-verydeep-16.mat') ;
net = dagnn.DagNN.loadobj(netStruct) ;
for i=1:numel(net.layers)
    net.layers(i).inputs = strcat('l',net.layers(i).inputs);
    net.layers(i).name = strcat('l',net.layers(i).name);
    net.layers(i).outputs = strcat('l',net.layers(i).outputs);
    if(~isempty(net.layers(i).params))
        for j=1:numel(net.layers(i).params)
            net.layers(i).params(j) = strcat('l',net.layers(i).params(j));
        end
    end
end

for i=1:numel(net.vars)
    net.vars(i).name = strcat('l',net.vars(i).name);
end
netStruct = net.saveobj() ;
save('vgg16_copy.mat', '-struct', 'netStruct') ;