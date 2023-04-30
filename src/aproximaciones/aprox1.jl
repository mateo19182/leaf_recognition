using Random
using ScikitLearn
using DelimitedFiles 

include("../aux/leaf.jl");
include("../aux/functions.jl");
include("../algoritmos/knn.jl");
include("../algoritmos/SVM.jl");
include("../algoritmos/RNA.jl");
include("../algoritmos/DecisionT.jl");



#Fijar la semilla aleatoria para garantizar la repetibilidad de los resultados.
Random.seed!(1);

#Cargar los datos y extraer las características de esa aproximación.
#loadData();
bd = readdlm("src/data/samples1.data",',');

#hay 3 caracteristicas: forma de la imagen, numero de pixeles blancos, numero de pixels de borde de la hoja.
entrada = bd[:,1:3];
entrada = convert(Array{Float32}, entrada);
normalmaxmin(entrada);
#hay 2 clases: Alnus y Eucalyptus
salida = bd[:,end];
salida = convert(Array{String}, salida);
numPatrones = size(entrada, 1);
println("Tamaño de la matriz de entradas: ", size(entrada,1), "x", size(entrada,2), " de tipo ", typeof(entrada));
println("Longitud del vector de salidas deseadas antes de codificar: ", length(salida), " de tipo ", typeof(salida));

numFolds = 10;



# Parametros del SVM
kernels = ["rbf", "linear", "poly", "sigmoid"];
kernelDegree = 3;
kernelGamma = "auto";
C=1;

# Parametros del arbol de decision
maxDepths = [2;3;4;5;7;9];

# Parapetros de kNN
numNeighbors = 3;

# Creamos los indices de validacion cruzada
crossValidationIndices = crossvalidation(numPatrones, numFolds);


# Parametros principales de la RNA y del proceso de entrenamiento
# topology = [4,3]; # Dos capas ocultas con 4 neuronas la primera y 3 la segunda
learningRate = 0.01; # Tasa de aprendizaje
numMaxEpochs = 1; # Numero maximo de ciclos de entrenamiento
validationRatio = 0.2; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
maxEpochsVal = 1; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
numRepetitionsANNTraining = 1; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento

# --------------------------------------------------------RR.NN.AA.----------------------------------------------------------------
modelHyperparameters = Dict();
# modelHyperparameters["topology"] = topology;
modelHyperparameters["learningRate"] = learningRate;
modelHyperparameters["validationRatio"] = validationRatio;
modelHyperparameters["numExecutions"] = numRepetitionsANNTraining;
modelHyperparameters["maxEpochs"] = numMaxEpochs;
modelHyperparameters["maxEpochsVal"] = maxEpochsVal;

x= 1:10; # Primera capa
y=0:10; # segunda capa
resultsRNA = Array{Array{Any,1},1}()
for j in y,i in x
    if j==0
        modelHyperparameters["topology"] = [i];
    else
        modelHyperparameters["topology"] = [i,j];  
    end
    (meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = modelCrossValidation(RNA, modelHyperparameters, entrada, salida, crossValidationIndices);
    nuevo =[ modelHyperparameters["topology"] , round(meanTestAccuracies, digits=3),round(stdTestAccuracies, digits=3),round(meanTestF1, digits=3),round(stdTestF1, digits=3)]
    push!(resultsRNA, nuevo)
end;

for h in resultsRNA
            (Tp, meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = h

            println("$Tp & $meanTestAccuracies & $stdTestAccuracies & $meanTestF1 & $stdTestF1")
    
            println("Topology: $Tp");
            println("Accuracy: ", meanTestAccuracies);
            println("desviacionTipica: ", stdTestAccuracies);
            println("AccuracyF1: ", meanTestF1);
            println("desviacionTipicaF1: ", stdTestF1);
            
end

#-------------------------------------------------------------SVM-----------------------------------------------------------------------------------
modelHyperparameters = Dict();
modelHyperparameters["kernelDegree"] = kernelDegree;
modelHyperparameters["kernelGamma"] = kernelGamma;
resultsSVM = Array{Array{Float64,1},1}()

for i in kernels
    println(i);
    modelHyperparameters["kernel"] = i;
    for j in 1:10
        println(" C: ", j*10)  #valor que cambia
        modelHyperparameters["C"] = C*j*10;
        (meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = modelCrossValidation(SVM, modelHyperparameters, entrada, salida, crossValidationIndices);
        nuevo =[ j*10, round(meanTestAccuracies, digits=3),round(stdTestAccuracies, digits=3),round(meanTestF1, digits=3),round(stdTestF1, digits=3)]
        push!(resultsSVM, nuevo)
    end
    for h in resultsSVM
        (C, meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = h
        println("$C & $meanTestAccuracies & $stdTestAccuracies & $meanTestF1 & $stdTestF1")
#         println("Accuracy: ", meanTestAccuracies);
#         println("desviacionTipica: ", stdTestAccuracies);
#         println("AccuracyF1: ", meanTestF1);
#         println("desviacionTipicaF1: ", stdTestF1);       
    end
end

#mejores parametros y matrices de confusion sobre todo el dataset.
modelHyperparameters["kernel"] = "rbf";
modelHyperparameters["C"] = 30;
println("best paremeters: kernel=rfb, C=30");
SVM(modelHyperparameters, entrada, salida);



#-------------------------------------------------------KNN--------------------------------------------------------------------------------------

resultsKNN = Array{Array{Float64,1},1}()

for j in 1:10
    println(" numNeighbors: ", j)  #valor que cambia
    numNeighbors = j;
    (meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = modelCrossValidation(knn, Dict("numNeighbors" => numNeighbors), entrada, salida, crossValidationIndices);
    nuevo =[ numNeighbors,round(meanTestAccuracies, digits=3),round(stdTestAccuracies, digits=3),round(meanTestF1, digits=3),round(stdTestF1, digits=3)]
    push!(resultsKNN, nuevo)
end
for i in resultsKNN
    (numNeighbors, meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = i 
    println("$numNeighbors & $meanTestAccuracies & $stdTestAccuracies & $meanTestF1 & $stdTestF1")
end


sorted_resultsKNN = sort(resultsKNN, by=x->x[4], rev=true)
numNeighbors = convert(Int, sorted_resultsKNN[1][1]);
println("best paremeters:  numNeighbors= $numNeighbors");

knn( Dict("numNeighbors" => numNeighbors), entrada, salida);



#-------------------------------------------------------DecisionTrees--------------------------------------------------------------------------------------
resultsDT = Array{Array{Float64,1},1}()

for maxDepth in maxDepths
    println("Profundidad: $maxDepth");
    (meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = modelCrossValidation(DecisionTree, Dict("maxDepth" => maxDepth), entrada, salida, crossValidationIndices);
    nuevo =[maxDepth,round(meanTestAccuracies, digits=3),round(stdTestAccuracies, digits=3),round(meanTestF1, digits=3),round(stdTestF1, digits=3)]
    push!(resultsDT, nuevo)
end
for i in resultsDT
    (depth, meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = i 
    println("$depth & $meanTestAccuracies & $stdTestAccuracies & $meanTestF1 & $stdTestF1")
end
sorted_resultsDT = sort(resultsDT, by=x->x[4], rev=true)
bestDepth = convert(Int, sorted_resultsDT[1][1]);
println("best paremeters:  depth= $bestDepth");
#DecisionTree(Dict("maxDepth" => bestDepth, entrada, salida));