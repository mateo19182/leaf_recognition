using ScikitLearn
using DelimitedFiles 
using Statistics
using Plots

include("../aux/functions.jl");

@sk_import svm: SVC 

function SVM(modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1},numFold::Int)
    
    # Dividimos los datos en entrenamiento y test
    trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
    testInputs        = inputs[crossValidationIndices.==numFold,:];
    trainingTargets   = targets[crossValidationIndices.!=numFold];
    testTargets       = targets[crossValidationIndices.==numFold];
    
    svc = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["kernelDegree"], gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"]);
    # Entrenamos el modelo con el conjunto de entrenamiento
    model = fit!(svc, trainingInputs, trainingTargets);

    # Pasamos el conjunto de test
    testOutputs = predict(model, testInputs);

    # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
    (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, testTargets);
    #printConfusionMatrix(testOutputs, testTargets; weighted=true);
    return acc, F1;
end

function SVM(modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1})
    
    # Dividimos los datos en entrenamiento y test
    
    svc = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["kernelDegree"], gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"]);
    # Entrenamos el modelo con el conjunto de entrenamiento
    model = fit!(svc, inputs, targets);

    # Pasamos el conjunto de test
    Outputs = predict(model, inputs);

    # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
    (acc, _, _, _, _, _, F1, _) = confusionMatrix(Outputs, targets);
    printConfusionMatrix(Outputs, targets; weighted=true);
    return acc, F1;
end

