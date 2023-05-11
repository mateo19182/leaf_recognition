using Random
using ScikitLearn
using DelimitedFiles 

include("../aux/leaf.jl");
include("../aux/functions.jl");
include("../algoritmos/DeepLearning.jl");

#Fijar la semilla aleatoria para garantizar la repetibilidad de los resultados.
Random.seed!(1);

#Cargar los datos y extraer las características de esa aproximación.

#loadData();
#ruta_absoluta = abspath("../data/samples3.data")
bd = readdlm("src/data/samples4.data",',');
entrada = bd[:,1:7];
entrada = convert(Array{Float32}, entrada);
normalmaxmin(entrada);
salida = bd[:,end];
salida = convert(Array{String}, salida);
numPatrones = size(entrada, 1);

println("Tamaño de la matriz de entradas: ", size(entrada,1), "x", size(entrada,2), " de tipo ", typeof(entrada));
println("Longitud del vector de salidas deseadas antes de codificar: ", length(salida), " de tipo ", typeof(salida));

numFolds = 10;

# Creamos los indices de validacion cruzada
crossValidationIndices = crossvalidation(numPatrones, numFolds);


# Parametros principales de la RNA y del proceso de entrenamiento
# topology = [4,3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
validationRatio = 0.2; # Porcentaje de patrones que se usaran para validacion. (No probar aun)Puede ser 0, para no usar validacion
maxEpochsVal = 6; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsANNTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento

# Entrenamos las RR.NN.AA.
modelHyperparameters = Dict();
# modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;
modelHyperparameters["numExecutions"] = numRepetitionsANNTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;

#Falta declarar la funcion DeepLearning
modelCrossValidation(DeepLearning, modelHyperparameters, entrada, salida, crossValidationIndices);
