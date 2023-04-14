using Random


#solo hace falta normalizar el 3er atributo
function normalmaxmin(patrones::Array{Float64,2})
    minimo = minimum(patrones[:,3], dims=1);
    maximo = maximum(patrones[:,3], dims=1);
    patrones[:,3] .-= minimo;
    patrones[:,3]  ./= (maximo .- minimo);
    #patrones[:,vec(minimo.==maximo)] .= 0;
end;

function holdOut(numPatrones::Int, porcentajeTest::Float64)
    @assert ((porcentajeTest>=0.) & (porcentajeTest<=1.));
    indices = randperm(numPatrones);
    numPatronesEntrenamiento = Int(round(numPatrones*(1-porcentajeTest)));
    return (indices[1:numPatronesEntrenamiento], indices[numPatronesEntrenamiento+1:end]);
end


function modelCrossValidation(modelType::Symbol, modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})
    # Comprobamos que el numero de patrones coincide
    @assert(size(inputs,1)==length(targets));

    # Que clases de salida tenemos
    # Es importante calcular esto primero porque se va a realizar codificacion one-hot-encoding varias veces, y el orden de las clases deberia ser el mismo siempre
    classes = unique(targets);

    # Creamos los vectores para las metricas que se vayan a usar
    # En este caso, solo voy a usar precision y F1, en otro problema podrían ser distintas
    testAccuracies = Array{Float64,1}(undef, numFolds);
    testF1         = Array{Float64,1}(undef, numFolds);

    # Para cada fold, entrenamos
    for numFold in 1:numFolds

        # Si vamos a usar unos de estos 3 modelos
        if (modelType==:SVM) || (modelType==:DecisionTree) || (modelType==:kNN)

            # Dividimos los datos en entrenamiento y test
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold];
            testTargets       = targets[crossValidationIndices.==numFold];

            if modelType==:SVM
                model = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["kernelDegree"], gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"]);
            elseif modelType==:DecisionTree
                model = DecisionTreeClassifier(max_depth=modelHyperparameters["maxDepth"], random_state=1);
            elseif modelType==:kNN
                model = KNeighborsClassifier(modelHyperparameters["numNeighbors"]);
            end;

            # Entrenamos el modelo con el conjunto de entrenamiento
            model = fit!(model, trainingInputs, trainingTargets);

            # Pasamos el conjunto de test
            testOutputs = predict(model, testInputs);

            # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
            (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, testTargets);

        else

            # Vamos a usar RR.NN.AA.
            @assert(modelType==:ANN);

            # Dividimos los datos en entrenamiento y test
            trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
            testInputs        = inputs[crossValidationIndices.==numFold,:];
            trainingTargets   = targets[crossValidationIndices.!=numFold,:];
            testTargets       = targets[crossValidationIndices.==numFold,:];

            # Como el entrenamiento de RR.NN.AA. es no determinístico, hay que entrenar varias veces, y
            #  se crean vectores adicionales para almacenar las metricas para cada entrenamiento
            testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
            testF1EachRepetition         = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);

            # Se entrena las veces que se haya indicado
            for numTraining in 1:modelHyperparameters["numExecutions"]

                if modelHyperparameters["validationRatio"]>0

                    # Para el caso de entrenar una RNA con conjunto de validacion, hacemos una división adicional:
                    #  dividimos el conjunto de entrenamiento en entrenamiento+validacion
                    #  Para ello, hacemos un hold out
                    (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
                    # Con estos indices, se pueden crear los vectores finales que vamos a usar para entrenar una RNA

                    # Entrenamos la RNA, teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"], (trainingInputs[trainingIndices,:],   trainingTargets[trainingIndices,:]),
                        validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),
                        testDataset =       (testInputs,                          testTargets);
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"], maxEpochsVal=modelHyperparameters["maxEpochsVal"]);

                else

                    # Si no se desea usar conjunto de validacion, se entrena unicamente con conjuntos de entrenamiento y test,
                    #  teniendo cuidado de codificar las salidas deseadas correctamente
                    ann, = trainClassANN(modelHyperparameters["topology"], (trainingInputs, trainingTargets),
                        testDataset = (testInputs,     testTargets);
                        maxEpochs=modelHyperparameters["maxEpochs"], learningRate=modelHyperparameters["learningRate"]);

                end;

                # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
                (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], _) = confusionMatrix(collect(ann(testInputs')'), testTargets);

            end;

            # Calculamos el valor promedio de todos los entrenamientos de este fold
            acc = mean(testAccuraciesEachRepetition);
            F1  = mean(testF1EachRepetition);

        end;

        # Almacenamos las 2 metricas que usamos en este problema
        testAccuracies[numFold] = acc;
        testF1[numFold]         = F1;

        println("Results in test in fold ", numFold, "/", numFolds, ": accuracy: ", 100*testAccuracies[numFold], " %, F1: ", 100*testF1[numFold], " %");

    end; # for numFold in 1:numFolds

    println(modelType, ": Average test accuracy on a ", numFolds, "-fold crossvalidation: ", 100*mean(testAccuracies), ", with a standard deviation of ", 100*std(testAccuracies));
    println(modelType, ": Average test F1 on a ", numFolds, "-fold crossvalidation: ", 100*mean(testF1), ", with a standard deviation of ", 100*std(testF1));

    return (mean(testAccuracies), std(testAccuracies), mean(testF1), std(testF1));

end;

function crossValidation(entrada, salida, kFolds::Int)
    println(vc[1:10]);

end

function Euclidean(x::Vector{Float32}, y::Vector{Float32})
    return sqrt(sum((x[i] - y[i])^2 for i in 1:length(x)))
end


# function oneHotEncoding(feature::AbstractArray{<:Any,1},
# classes::AbstractArray{<:Any,1})
# function oneHotEncoding(feature::AbstractArray{<:Any,1})
# function oneHotEncoding(feature::AbstractArray{Bool,1})
# function calculateMinMaxNormalizationParameters(dataset::AbstractArray{<:Real,2})
# function calculateZeroMeanNormalizationParameters(dataset::AbstractArray{<:Real,2})
# function normalizeMinMax!(dataset::AbstractArray{<:Real,2},
# normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
# function normalizeMinMax!(dataset::AbstractArray{<:Real,2})
# function normalizeMinMax( dataset::AbstractArray{<:Real,2},
# normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
# function normalizeMinMax( dataset::AbstractArray{<:Real,2})
# function normalizeZeroMean!(dataset::AbstractArray{<:Real,2},
# normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
# function normalizeZeroMean!(dataset::AbstractArray{<:Real,2})
# function normalizeZeroMean( dataset::AbstractArray{<:Real,2},
# normalizationParameters::NTuple{2, AbstractArray{<:Real,2}})
# function normalizeZeroMean( dataset::AbstractArray{<:Real,2})
# function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
# function accuracy(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
# function accuracy(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2})
# function accuracy(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1};
# threshold::Real=0.5)
# function accuracy(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2};
# threshold::Real=0.5)
# function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int;
# transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
# function trainClassANN(topology::AbstractArray{<:Int,1},
# dataset::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,2}};
# transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
# maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01)
# function trainClassANN(topology::AbstractArray{<:Int,1},
# (inputs, targets)::Tuple{AbstractArray{<:Real,2}, AbstractArray{Bool,1}};
# transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)),
# maxEpochs::Int=1000, minLoss::Real=0.0, learningRate::Real=0.01) 