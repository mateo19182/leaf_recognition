using Random
using ScikitLearn
using DelimitedFiles 

include("../aux/leaf.jl");
include("../aux/functions.jl");
include("../algoritmos/knn.jl");
include("../algoritmos/SVM.jl");
include("../algoritmos/RNA.jl");
include("../algoritmos/DecisionT.jl");

function aproximacion(entrada, salida, numFolds,numPatrones)
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
    learningRate = 0.01; # Tasa de aprendizaje
    numMaxEpochs = 1000; # Numero maximo de ciclos de entrenamiento
    validationRatio = 0.2; # Porcentaje de patrones que se usaran para validacion. Puede ser 0, para no usar validacion
    maxEpochsVal = 10; # Numero de ciclos en los que si no se mejora el loss en el conjunto de validacion, se para el entrenamiento
    numRepetitionsANNTraining = 50; # Numero de veces que se va a entrenar la RNA para cada fold por el hecho de ser no determinístico el entrenamiento

    # --------------------------------------------------------RR.NN.AA.----------------------------------------------------------------
    println("")
    println("Ejecución : RNA")
    println("")
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
    println("")
    println("Ejecución : SVM")
    println("")
    modelHyperparameters = Dict();
    modelHyperparameters["kernelDegree"] = kernelDegree;
    modelHyperparameters["kernelGamma"] = kernelGamma;
    resultsSVM = Array{Array{Any,1},1}()

    for i in kernels
        modelHyperparameters["kernel"] = i;
        for j in 1:10
            modelHyperparameters["C"] = C*j*10;
            (meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = modelCrossValidation(SVM, modelHyperparameters, entrada, salida, crossValidationIndices);
            nuevo =[i, j*10, round(meanTestAccuracies, digits=3),round(stdTestAccuracies, digits=3),round(meanTestF1, digits=3),round(stdTestF1, digits=3)]
            push!(resultsSVM, nuevo)
        end
    end

    sorted_resultsSVM = sort(resultsSVM, by=x->x[5], rev=true)
    bestC = convert(Int, sorted_resultsSVM[1][2]);
    bestKernel = convert(String, sorted_resultsSVM[1][1])
    modelHyperparameters["kernel"] = bestKernel;
    modelHyperparameters["C"] = bestC;
    println("best paremeters: kernel= $bestKernel, C= $bestC");
    SVM(modelHyperparameters, entrada, salida);


    #-------------------------------------------------------KNN--------------------------------------------------------------------------------------
    println("")
    println("Ejecución : KNN")
    println("")
    resultsKNN = Array{Array{Float64,1},1}()

    for j in 1:10
        println(" numNeighbors: ", j)  #valor que cambia
        numNeighbors = j;
        (meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = modelCrossValidation(knn, Dict("numNeighbors" => numNeighbors), entrada, salida, crossValidationIndices);
        nuevo =[ numNeighbors,round(meanTestAccuracies, digits=3),round(stdTestAccuracies, digits=3),round(meanTestF1, digits=3),round(stdTestF1, digits=3)]
        push!(resultsKNN, nuevo)
    end


    sorted_resultsKNN = sort(resultsKNN, by=x->x[4], rev=true)
    numNeighbors = convert(Int, sorted_resultsKNN[1][1]);
    println("best paremeters:  numNeighbors= $numNeighbors");
    knn( Dict("numNeighbors" => numNeighbors), entrada, salida);

    #-------------------------------------------------------DecisionTrees--------------------------------------------------------------------------------------
    println("")
    println("Ejecución : Decision Trees")
    println("")
    resultsDT = Array{Array{Float64,1},1}()

    for maxDepth in maxDepths
        println("Profundidad: $maxDepth");
        (meanTestAccuracies, stdTestAccuracies, meanTestF1, stdTestF1) = modelCrossValidation(DecisionTree, Dict("maxDepth" => maxDepth), entrada, salida, crossValidationIndices);
        nuevo =[maxDepth,round(meanTestAccuracies, digits=3),round(stdTestAccuracies, digits=3),round(meanTestF1, digits=3),round(stdTestF1, digits=3)]
        push!(resultsDT, nuevo)
    end
    sorted_resultsDT = sort(resultsDT, by=x->x[4], rev=true)
    bestDepth = convert(Int, sorted_resultsDT[1][1]);
    println("best paremeters:  depth= $bestDepth");
    DecisionTree(Dict("maxDepth" => bestDepth), entrada, salida);

end