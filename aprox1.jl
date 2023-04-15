using Random
using ScikitLearn
using DelimitedFiles 

include("leaf.jl");
include("functions.jl");
include("knn.jl");
include("SVM.jl");


#Fijar la semilla aleatoria para garantizar la repetibilidad de los resultados.
Random.seed!(1);

#Cargar los datos y extraer las características de esa aproximación.

#loadData();
#bd = readdlm("samples.data",',');
#entrada = bd[:,1:5];
#entrada = convert(Array{Float64}, entrada);
#normalmaxmin(entrada);
#salida = bd[:,end];
#salida = convert(Array{String}, salida);
numPatrones = size(entrada, 1);

inputs , targets = loadData(5);
println("Tamaño de la matriz de entradas: ", size(inputs,1), "x", size(inputs,2), " de tipo ", typeof(inputs));
println("Longitud del vector de salidas deseadas antes de codificar: ", length(targets), " de tipo ", typeof(targets));

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
kernelDegree = 3;
kernelGamma = 2;
C=1;

# Parametros del arbol de decision
maxDepth = 4;

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
modelHyperparameters["kernel"] = kernel;
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
modelHyperparameters["C"] = C;

#modelCrossValidation(:SVM, modelHyperparameters, entrada, salida, crossValidationIndices);
modelCrossValidation(SVM, modelHyperparameters, inputs, targets, crossValidationIndices);

# Entrenamos los arboles de decision
#modelCrossValidation2(:DecisionTree, Dict("maxDepth" => maxDepth), inputs, targets, crossValidationIndices);

# Entrenamos los kNN
modelCrossValidation(knn, Dict("numNeighbors" => numNeighbors), inputs, targets, crossValidationIndices);
#modelCrossValidation(knn(), Dict("numNeighbors" => numNeighbors), inputs, targets, crossValidationIndices);
