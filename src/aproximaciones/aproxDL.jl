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

#Falta declarar la funcion DeepLearning
modelCrossValidation(DeepLearning, modelHyperparameters, entrada, salida, crossValidationIndices);
