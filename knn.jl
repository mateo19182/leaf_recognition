using DelimitedFiles
using StatsBase
using LinearAlgebra


include("functions.jl");
# load dataset
dataset = readdlm("example.data",',');
inputs = dataset[:,1:4];
targets = dataset[:,5];
inputs = convert(Array{Float32,2},inputs);
targets = convert(Array{String,1}, targets);
@assert (size(inputs,1)==size(targets,1)) "Error: Diff rows numbers on inputs and targets matrixes " 
posibletargets = unique(targets);
classtoclass = "Alnus";
@assert (isa(classtoclass, String)) "Error : class to classify is not String";
@assert (any(classtoclass .== posibletargets)) "Error: class to clasify is not a possible one";
desiredetargets = Array((posibletargets .== classtoclass)');
inputs = Array(inputs)';

# apply holdout
train_indexes, test_indexes = holdOut(size(inputs, 2), 0.2)

# separate inputs and targets into train and test sets
train_inputs = inputs[:, train_indexes]
train_targets = targets[train_indexes]
test_inputs = inputs[:, test_indexes]
test_targets = targets[test_indexes]

# define kNN function
function knn(train_inputs::Matrix{Float32}, train_targets::Vector{String}, test_input::Vector{Float32}, k::Int)
    # calculate distances between test_input and train_inputs
    distances = [Euclidean(test_input, train_inputs[:, i]) for i in 1:size(train_inputs, 2)]
    
    # get indexes of k nearest neighbors
    nearest_indexes = sortperm(distances)[1:k]
    
    # get corresponding targets of k nearest neighbors
    nearest_targets = train_targets[nearest_indexes]
    
    # return most common target
    return mode(nearest_targets)
end

# train kNN classifier on train set and evaluate on test set
k = 3
num_correct = 0
for i in 1:length(test_indexes)
    pred = knn(train_inputs, train_targets, test_inputs[:, i], k)
    if pred == test_targets[i]
        num_correct += 1
    end
end
accuracy = num_correct / length(test_indexes)

println("kNN accuracy: ", accuracy)