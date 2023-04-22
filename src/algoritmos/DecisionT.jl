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


        # Almacenamos las 2 metricas que usamos en este problema
        testAccuracies[numFold] = acc;
        testF1[numFold]         = F1;

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

    end; # for numFold in 1:numFolds

    println(modelType, ": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println(modelType, ": Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));

    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));

end;

# Fijamos la semilla aleatoria para poder repetir los experimentos
seed!(1);

numFolds = 10;



# Parametros del arbol de decision
maxDepths = [2;3;4;5;7];


tree_values = Dict("Alnus" => 1, "Eucalyptus" => 2, "Cornus" => 3)


dataset = readdlm("samples.data",',');
inputs = dataset[:,1:5];
inputs = convert(Array{Float64,2},inputs);

targets = dataset[:,end]
targets = map(x -> tree_values[x], targets)
# convertir targetsString a matriz de booleanos



#normalizeMinMax!(inputs);

# Entrenamos los arboles de decision

for maxDepth in maxDepths
    println("Profundidad: $maxDepth");
    modelCrossValidation(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, numFolds);
end;


=#
# GONZALO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! 
#REVISA TODO LO QUE HAS HECHO PREVIAMENTE POR SI SE NECESITA ALGO PARA EL USO DE
# ESTE ALGORITMO, PERO HEMOS IMPLEMENTADO TODO CON ESTAS FUNCIONES PARA QUE 
# SEA LO MAS COMODO POSIBLE, EN PRINCIPIO HAY QUE BORRAR EL RESTO DE CLASE MENOS 
# ESTA FUNCIÓN, YA QUE ES CODIGO REPETIDO Y TIENE QUE IR TODO CON LOS MISMOS CONJUNTOS DE DATOS.
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

