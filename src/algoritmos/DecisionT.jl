using FileIO;
using DelimitedFiles;
using Random;
using Flux;
using Flux.Losses;
using Statistics;
using XLSX:readdata;
using ScikitLearn
using Random:seed!
using LinearAlgebra
@sk_import tree: DecisionTreeClassifier

function DecisionTree(modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1},numFold::Int)
    # Dividimos los datos en entrenamiento y test
    trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
    testInputs        = inputs[crossValidationIndices.==numFold,:];
    trainingTargets   = targets[crossValidationIndices.!=numFold];
    testTargets       = targets[crossValidationIndices.==numFold];
    model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
    # Entrenamos el modelo con el conjunto de entrenamiento
    model = fit!(model, trainingInputs, trainingTargets);
    # Pasamos el conjunto de test
    testOutputs = predict(model, testInputs);
    # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
    (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, testTargets);
    return acc,F1;
end

function DecisionTree(modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1})

    dt = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"])

    # Entrenamos el modelo con el todo el dataset
    model = fit!(dt, inputs, targets);

    # Pasamos el conjunto de test
    testOutputs = predict(model, inputs);
    
    # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
    (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, targets);
    printConfusionMatrix(testOutputs, targets; weighted=true);

    return acc, F1;
end
