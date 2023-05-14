using Random
using ScikitLearn
using DelimitedFiles 

include("../aux/leaf.jl");
include("../aux/functions.jl");
include("aprox.jl");



#Fijar la semilla aleatoria para garantizar la repetibilidad de los resultados.
Random.seed!(1);

#Cargar los datos y extraer las características de esa aproximación.
entrada,salida = loadDataSet("samples1.data",3);
numPatrones = size(entrada, 1);
println("Tamaño de la matriz de entradas: ", size(entrada,1), "x", size(entrada,2), " de tipo ", typeof(entrada));
println("Longitud del vector de salidas deseadas antes de codificar: ", length(salida), " de tipo ", typeof(salida));
numFolds = 10;
aproximacion(entrada, salida, numFolds,numPatrones);

