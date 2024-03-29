using ScikitLearn
using DelimitedFiles
using Statistics
using Plots
include("../aux/functions.jl");

# Importar módulo neighbors de ScikitLearn
@sk_import neighbors: KNeighborsClassifier
@sk_import metrics: accuracy_score

function knn(modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1},numFold::Int)
    
    # Dividimos los datos en entrenamiento y test
    trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
    testInputs        = inputs[crossValidationIndices.==numFold,:];
    trainingTargets   = targets[crossValidationIndices.!=numFold];
    testTargets       = targets[crossValidationIndices.==numFold];
    
    
    knn = KNeighborsClassifier(modelHyperparameters["numNeighbors"])

    # Entrenamos el modelo con el conjunto de entrenamiento
    model = fit!(knn, trainingInputs, trainingTargets);

    # Pasamos el conjunto de test
    testOutputs = predict(model, testInputs);
    
    # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
    (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, testTargets);
    #printConfusionMatrix(testOutputs, testTargets; weighted=true);

    return acc, F1;
end

function knn(modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1})

    knn = KNeighborsClassifier(modelHyperparameters["numNeighbors"])

    # Entrenamos el modelo con el todo el dataset
    model = fit!(knn, inputs, targets);

    # Pasamos el conjunto de test
    testOutputs = predict(model, inputs);
    
    # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
    (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, targets);
    printConfusionMatrix(testOutputs, targets; weighted=true);

    return acc, F1;
end


#println("Precisión en conjunto de test: ", knn(k))
#png(plot(x, label = "",xlims=(0,7), ylims=(85,100), title = "KNN", xlabel = "K Vecinos", ylabel = "Precisión en el Test (%)", ),"KNN");
