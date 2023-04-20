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
bd = readdlm("samples.data",',');
entrada = bd[:,1:5];
entrada = convert(Array{Float64}, entrada);
normalmaxmin(entrada);
salida = bd[:,end];
salida = convert(Array{String}, salida);
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


# Parametros del arbol de decision
maxDepth = 4;

# Parapetros de kNN
numNeighbors = 3;

# Creamos los indices de validacion cruzada
crossValidationIndices = crossvalidation(numPatrones, numFolds);

# Entrenamos las SVM

precisiones = Array{Float64,1}();
desviacionTipica = Array{Float64,1}();
precisionesF1 = Array{Float64,1}();
desviacionTipicaF1 = Array{Float64,1}();


for j in 1:10
    println("numNeighbors = ", j)
    (meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = modelCrossValidation2(knn, Dict("numNeighbors" => j), inputs, targets, crossValidationIndices);
        
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

knn(Dict("numNeighbors" => 2), entrada, salida)
#SVM(modelHyperparameters, entrada, salida);


