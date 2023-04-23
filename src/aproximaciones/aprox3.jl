using Random
using ScikitLearn
using DelimitedFiles 

include("../aux/leaf.jl");
include("../aux/functions.jl");
include("../algoritmos/knn.jl");
include("../algoritmos/SVM.jl");
include("../algoritmos/DecisionT.jl");


#Fijar la semilla aleatoria para garantizar la repetibilidad de los resultados.
Random.seed!(1);

#Cargar los datos y extraer las características de esa aproximación.

#loadData();
ruta_absoluta = abspath("../data/samples3.data")
bd = readdlm(ruta_absoluta, ',')
entrada = bd[:,1:5];
entrada = convert(Array{Float32}, entrada);
normalmaxmin(entrada);
salida = bd[:,end];
salida = convert(Array{String}, salida);
numPatrones = size(entrada, 1);

println("Tamaño de la matriz de entradas: ", size(entrada,1), "x", size(entrada,2), " de tipo ", typeof(entrada));
println("Longitud del vector de salidas deseadas antes de codificar: ", length(salida), " de tipo ", typeof(salida));

numFolds = 10;

# Parametros principales de la RNA y del proceso de entrenamiento
topology = [4, 3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsANNTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento

# Parametros del SVM
kernel = "rbf";
kernels = ["rbf", "linear", "poly", "sigmoid"];
kernelDegree = 3;
kernelGamma = 2;
C=1;

# Parametros del arbol de decision
maxDepths = [2;3;4;5;7;9];

# Parapetros de kNN
numNeighbors = 3;

# Creamos los indices de validacion cruzada
crossValidationIndices = crossvalidation(numPatrones, numFolds);

# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;
modelHyperparameters["numExecutions"] = numRepetitionsANNTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;
#modelCrossValidation(:ANN, modelHyperparameters, inputs, targets, crossValidationIndices);

# Entrenamos las SVM
modelHyperparameters = Dict();
modelHyperparameters["kernel"] = kernels;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;

#modelCrossValidation(SVM, modelHyperparameters, entrada, salida, crossValidationIndices);

# Entrenamos los arboles de decision
for maxDepth in maxDepths
    println("Profundidad: $maxDepth");
    (meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = modelCrossValidation(DecisionTree, Dict("maxDepth" => maxDepth), entrada, salida, crossValidationIndices);
    println("Accuracy: ", meanTestAccuracies);
    println("desviacionTipica: ", stdTestAccuracies);
    println("AccuracyF1: ", meanTestF1);
    println("desviacionTipicaF1: ", stdTestF1);
end;
# Entrenamos los kNN
#modelCrossValidation(knn, Dict("numNeighbors" => numNeighbors), entrada, salida, crossValidationIndices);



for i in kernels
    println(i);
    modelHyperparameters["kernel"] = i;
    for j in 1:10
        println(" C: ", j*10)  #valor que cambia
        modelHyperparameters["C"] = C*j;

        #modelCrossValidation(:SVM, modelHyperparameters, entrada, salida, crossValidationIndices);
        (meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = modelCrossValidation(SVM, modelHyperparameters, entrada, salida, crossValidationIndices);

        #push!(precisiones, meanTestAccuracies);
        #push!(desviacionTipica, stdTestAccuracies);
        #push!(precisionesF1, meanTestF1);
        #push!(desviacionTipicaF1, stdTestF1);
        
        println("Accuracy: ", meanTestAccuracies);
        println("desviacionTipica: ", stdTestAccuracies);
        println("AccuracyF1: ", meanTestF1);
        println("desviacionTipicaF1: ", stdTestF1);
    end

    #=best=findmax(precisionesF1);
    println("C:")
    print(best)
    println(precisiones[best[2]]);
    println(desviacionTipica[best[2]]);
    println(precisionesF1[best[2]]);
    println(desviacionTipicaF1[best[2]]);=#
end

modelHyperparameters["kernel"] = "rbf";
modelHyperparameters["C"] = 30;
println("best paremeters: kernel=rfb, C=30");
SVM(modelHyperparameters, entrada, salida);

for j in 1:10
    println(" numNeighbors: ", j)  #valor que cambia
    numNeighbors = j;

    (meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = modelCrossValidation(knn, Dict("numNeighbors" => numNeighbors), entrada, salida, crossValidationIndices);
    
    println("Accuracy: ", meanTestAccuracies);
    println("desviacionTipica: ", stdTestAccuracies);
    println("AccuracyF1: ", meanTestF1);
    println("desviacionTipicaF1: ", stdTestF1);
end
numNeighbors = 2;
println("best paremeters:  numNeighbors=1");
knn( Dict("numNeighbors" => numNeighbors), entrada, salida);
