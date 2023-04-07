using ScikitLearn
using DelimitedFiles
using Statistics

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
for i in 3:7
    
    knn = KNeighborsClassifier(n_neighbors=i)
    fit!(knn,inputs,targets)

    #calcular la precision de la predicción
    predictions = predict(knn,inputs)
    correct = sum(predictions .== targets)
    accuracy = correct / length(targets)
    # calcular desviacion tipica
    std_dev = std(predictions .== targets)
    print("K neighbors : $i   ")
    print("Prediction accuracy: $accuracy    ")
    println("Standard deviation: $std_dev")

end