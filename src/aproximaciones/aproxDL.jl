using Random
using ScikitLearn
using DelimitedFiles 

include("../aux/leaf.jl");
include("../aux/functions.jl");
include("../algoritmos/DeepLearning.jl");


#Fijar la semilla aleatoria para garantizar la repetibilidad de los resultados.
Random.seed!(1);

#Cargar los datos y extraer las características de esa aproximación.
entrada,salida = loadDataSet("samples4.data",7);
numPatrones = size(entrada, 1);
println("Tamaño de la matriz de entradas: ", size(entrada,1), "x", size(entrada,2), " de tipo ", typeof(entrada));
println("Longitud del vector de salidas deseadas antes de codificar: ", length(salida), " de tipo ", typeof(salida));
numFolds = 10;
crossValidationIndices = crossvalidation(numPatrones, numFolds);
modelHyperparameters = Dict();

resultsDL = Array{Array{Any,1},1}()

for j in 2:10
    modelHyperparameters["salida"] = j;
    (meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = modelCrossValidation(DeepLearning, modelHyperparameters, entrada, salida, crossValidationIndices);
    nuevo =[j, round(meanTestAccuracies, digits=3),round(stdTestAccuracies, digits=3),round(meanTestF1, digits=3),round(stdTestF1, digits=3)]
    push!(resultsDL, nuevo)
end

for h in resultsDL
    (j, meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = h
    println("$j & $meanTestAccuracies & $stdTestAccuracies & $meanTestF1 & $stdTestF1")   
end
