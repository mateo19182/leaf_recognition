using ScikitLearn
using DelimitedFiles

include("./functions.jl")

@sk_import neighbors: KNeighborsClassifier

pcTest = 0.5
k =  3

#load dataset
# load dataset
dataset = readdlm("samples.data",',');
inputs = dataset[:,1:3];
targets = dataset[:,end];
inputs = convert(Array{Float32,2},inputs);
targets = convert(Array{String,1}, targets);

#fit KNN KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=k)
fit!(knn,inputs,targets)

#calcular la precision de la predicción
predictions = predict(knn,inputs)
correct = sum(predictions .== targets)
accuracy = correct / length(targets)

println("Prediction accuracy: $accuracy")