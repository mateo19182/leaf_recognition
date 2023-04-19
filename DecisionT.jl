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
@sk_import svm: SVC
@sk_import tree: DecisionTreeClassifier
@sk_import neighbors: KNeighborsClassifier

  

function confusionMatrix(yTrue::Vector{T}, yPred::Vector{T}) where T<:Integer
    
    # Comprobamos que los vectores tienen la misma longitud
    @assert length(yTrue) == length(yPred)
    
    # Obtenemos el número de clases
    nClasses = length(unique(vcat(yTrue, yPred)))
    
    # Inicializamos las variables
    truePositives = zeros(Int, nClasses)
    falsePositives = zeros(Int, nClasses)
    trueNegatives = zeros(Int, nClasses)
    falseNegatives = zeros(Int, nClasses)
    support = zeros(Int, nClasses)
    
    # Calculamos las métricas
    for i in 1:length(yTrue)
        for j in 1:nClasses
            if yTrue[i] == j && yPred[i] == j
                truePositives[j] += 1
            elseif yTrue[i] == j && yPred[i] != j
                falseNegatives[j] += 1
            elseif yTrue[i] != j && yPred[i] == j
                falsePositives[j] += 1
            else
                trueNegatives[j] += 1
            end
        end
    end
    
    # Calculamos el soporte de cada clase
    for i in 1:nClasses
        support[i] = sum(yTrue .== i)
    end
    
    # Calculamos las métricas globales
    accuracy = sum(truePositives) / length(yTrue)
    macroPrecision = sum(truePositives) / (sum(truePositives) + sum(falsePositives))
    macroRecall = sum(truePositives) / (sum(truePositives) + sum(falseNegatives))
    macroF1 = 2 * macroPrecision * macroRecall / (macroPrecision + macroRecall)
    weightedPrecision = dot(support, truePositives) / (dot(support, truePositives) + dot(support, falsePositives))
    weightedRecall = dot(support, truePositives) / (dot(support, truePositives) + dot(support, falseNegatives))
    weightedF1 = 2 * weightedPrecision * weightedRecall / (weightedPrecision + weightedRecall)
    
    # Devolvemos las métricas
    return accuracy, macroPrecision, macroRecall, macroF1, weightedPrecision, weightedRecall, weightedF1, support
    
end
function crossvalidation(N::Int64, k::Int64)
    indices = repeat(1:k, Int64(ceil(N/k)));
    indices = indices[1:N];
    shuffle!(indices);
    return indices;
end;


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::Array{Float64,2}, targets::Vector{Int64}, numFolds::Int64)

    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs,1)==size(targets,1));
    # Que clases de salida tenemos
    # Es importante calcular esto primero porque se va a realizar codificacion one-hot-encoding varias veces, y el orden de las clases deberia ser el mismo siempre
    classes = unique(targets);

    # Primero codificamos las salidas deseadas en caso de entrenar RR.NN.AA.
    

    # Creamos los indices de crossvalidation
    crossValidationIndices = crossvalidation(size(inputs,1), numFolds);

    # Creamos los vectores para las metricas que se vayan a usar
    # En este caso, solo voy a usar precision y F1, en otro problema podrían ser distintas
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Si vamos a usar unos de estos 3 modelos
    

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
        (acc, _, _, F1, _, _, _, _) = confusionMatrix(testOutputs, testTargets);

        

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

