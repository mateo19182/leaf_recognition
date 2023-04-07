using DelimitedFiles
using StatsBase
using LinearAlgebra
include("functions.jl");

# load dataset
dataset = readdlm("samples.data",',');
inputs = dataset[:,1:3];
targets = dataset[:,end];
inputs = convert(Array{Float32,2},inputs);
targets = convert(Array{String,1}, targets);

#variables
pcTest = 0.5;

@assert (size(inputs,1)==size(targets,1)) "Error: Diff rows numbers on inputs and targets matrixes " 
posibletargets = unique(targets);
classtoclass = "Alnus";
@assert (isa(classtoclass, String)) "Error : class to classify is not String";
@assert (any(classtoclass .== posibletargets)) "Error: class to clasify is not a possible one";
desiredetargets = Array((posibletargets .== classtoclass)');
inputs = Array(inputs)';

# apply holdout
train_indexes, test_indexes = holdOut(size(inputs, 2), pcTest)

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

# train kNN  on train set and evaluate on test set
function evaluacion()
    k = 7 # número de vecinos cercanos a considerar
    correct = 0  # contador de clasificaciones correctas
    for i in 1:length(test_indexes)
        
        predicted_target = knn(train_inputs, train_targets, test_inputs[:, i], k)
        x = test_targets[i]
        print("predicted_target = $predicted_target \n")
        print("test_targets = $x \n")
        print("------------------------------------------------\n")
        if predicted_target == test_targets[i]
            #suma mas uno al contador
            correct = correct + 1
        end
    end
    return correct / length(test_targets)
end
accuracy = evaluacion()
println("Accuracy: $accuracy")