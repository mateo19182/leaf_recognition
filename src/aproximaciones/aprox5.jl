using Random
using ScikitLearn
using DelimitedFiles 

include("../aux/leaf.jl");
include("../aux/functions.jl");
include("../algoritmos/knn.jl");
include("../algoritmos/SVM.jl");
include("../algoritmos/DecisionT.jl");
include("../algoritmos/RNA.jl");

#Fijar la semilla aleatoria para garantizar la repetibilidad de los resultados.
Random.seed!(1);

#Cargar los datos y extraer las características de esa aproximación.

path_actual = abspath(pwd())
println(path_actual)
 #bd = readdlm("src/data/samples5.data",',');
 #entrada = bd[:,1:7];
 #entrada = convert(Array{Float32}, entrada);
 #normalmaxmin(entrada);
 #salida = bd[:,end];
 #salida = convert(Array{String}, salida);
 entrada,salida = loadDataSet("samples5.data",7);
numPatrones = size(entrada, 1);

println("Tamaño de la matriz de entradas: ", size(entrada,1), "x", size(entrada,2), " de tipo ", typeof(entrada));
println("Longitud del vector de salidas deseadas antes de codificar: ", length(salida), " de tipo ", typeof(salida));

numFolds = 10;


# Parametros del SVM
kernel = "rbf";
kernels = ["rbf", "linear", "poly", "sigmoid"];
kernelDegree = 3;
kernelGamma = 2;
C=10;

# Parametros del arbol de decision
maxDepths = [2;3;4;5;7;9];

# Parapetros de kNN
numNeighbors = 3;

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

resultsSVM = Array{Array{Float64,1},1}()


for i in kernels
    println(i);
    modelHyperparameters["kernel"] = i;
    for j in 1:10
        #println(" C: ", j*10)  #valor que cambia
        modelHyperparameters["C"] = C*j;

        #modelCrossValidation(:SVM, modelHyperparameters, entrada, salida, crossValidationIndices);
        (meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = modelCrossValidation(SVM, modelHyperparameters, entrada, salida, crossValidationIndices);
        nuevo =[ j*10, round(meanTestAccuracies, digits=3),round(stdTestAccuracies, digits=3),round(meanTestF1, digits=3),round(stdTestF1, digits=3)]
        push!(resultsSVM, nuevo)
        #push!(precisiones, meanTestAccuracies);
        #push!(desviacionTipica, stdTestAccuracies);
        #push!(precisionesF1, meanTestF1);
        #push!(desviacionTipicaF1, stdTestF1);
        
        #println("Accuracy: ", meanTestAccuracies);
        #println("desviacionTipica: ", stdTestAccuracies);
        #println("AccuracyF1: ", meanTestF1);
        #println("desviacionTipicaF1: ", stdTestF1);
    end

    for h in resultsSVM
        (C, meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = h
        println("$C & $meanTestAccuracies & $stdTestAccuracies & $meanTestF1 & $stdTestF1")
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
results = Array{Array{Float64,1},1}()

for j in 1:10
    println(" numNeighbors: ", j)  #valor que cambia
    numNeighbors = j;

    (meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = modelCrossValidation(knn, Dict("numNeighbors" => numNeighbors), entrada, salida, crossValidationIndices);
    
    #println("Accuracy: ", meanTestAccuracies);
    #println("desviacionTipica: ", stdTestAccuracies);
    #println("AccuracyF1: ", meanTestF1);
    #println("desviacionTipicaF1: ", stdTestF1);
    nuevo =[ numNeighbors,round(meanTestAccuracies, digits=3),round(stdTestAccuracies, digits=3),round(meanTestF1, digits=3),round(stdTestF1, digits=3)]
    push!(results, nuevo)
end
for i in results
    (numNeighbors, meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = i 
    println("$numNeighbors & $meanTestAccuracies & $stdTestAccuracies & $meanTestF1 & $stdTestF1")
end

sorted_results = sort(results, by=x->x[4], rev=true)
numNeighbors = convert(Int, sorted_results[1][1]);
println("best paremeters:  numNeighbors= $numNeighbors");
knn( Dict("numNeighbors" => numNeighbors), entrada, salida);
