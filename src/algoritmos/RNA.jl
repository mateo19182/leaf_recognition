
using DelimitedFiles
using Flux
using Flux.Losses
using Random
using Statistics
using Flux: onehotbatch, onecold, crossentropy, binarycrossentropy
using Plots

include("../aux/functions.jl");

# function holdOut(numpatterns::Int, pcvalidation::Float64, pctest::Float64)
#     @assert ((pcvalidation>=0.) & (pcvalidation<=1.));
#     @assert ((pctest>=0.) & (pctest<=1.));
#     @assert ((pcvalidation+pctest)<=1.);
#     indices = randperm(numpatterns);
#     numpatternsValidacion = Int(round(numpatterns*pcvalidation));
#     numpatternsTest = Int(round(numpatterns*pctest));
#     numpatternstraining = numpatterns - numpatternsValidacion - numpatternsTest;
#     return (indices[1:numpatternstraining], indices[numpatternstraining+1:numpatternstraining+numpatternsValidacion], indices[numpatternstraining+numpatternsValidacion+1:numpatternstraining+numpatternsValidacion+numpatternsTest]);
# end
# function holdOutt(numpatterns::Int, pctest::Float64)
#     @assert ((pctest>=0.) & (pctest<=1.));
#     indices = randperm(numpatterns);
#     numpatternstraining = Int(round(numpatterns*(1-pctest)));
#     return (indices[1:numpatternstraining], indices[numpatternstraining+1:end]);
# end
function buildClassANN(numInputs::Int, topology::AbstractArray{<:Int,1}, numOutputs::Int; transferFunctions::AbstractArray{<:Function,1}=fill(σ, length(topology)))
    ann=Chain();
    numInputsLayer = numInputs;
    for numHiddenLayer in 1:length(topology)
        numNeurons = topology[numHiddenLayer];
        ann = Chain(ann..., Dense(numInputsLayer, numNeurons, transferFunctions[numHiddenLayer]));
        numInputsLayer = numNeurons;
    end;
    if (numOutputs == 1)
        ann = Chain(ann..., Dense(numInputsLayer, 1, σ));
    else
        ann = Chain(ann..., Dense(numInputsLayer, numOutputs, identity));
        ann = Chain(ann..., softmax);
    end;
    return ann;
end;

function classifyOutputs(outputs::AbstractArray{<:Real,2}; threshold::Real=0.5)
    numOutputs = size(outputs, 2);
    @assert(numOutputs!=2)
    if numOutputs==1
        return outputs.>=threshold;
    else
        # Miramos donde esta el valor mayor de cada instancia con la funcion findmax
        (_,indicesMaxEachInstance) = findmax(outputs, dims=2);
        # Creamos la matriz de valores booleanos con valores inicialmente a false y asignamos esos indices a true
        outputs = falses(size(outputs));
        outputs[indicesMaxEachInstance] .= true;
        # Comprobamos que efectivamente cada patron solo este clasificado en una clase
        @assert(all(sum(outputs, dims=2).==1));
        return outputs;
    end;
end;

function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
    numInstances = length(targets);
    @assert(length(outputs)==numInstances);
    # Valores de la matriz de confusion
    TN = sum(.!outputs .& .!targets); # Verdaderos negativos
    FN = sum(.!outputs .&   targets); # Falsos negativos
    TP = sum(  outputs .&   targets); # Verdaderos positivos
    FP = sum(  outputs .& .!targets); # Falsos negativos
    # Creamos la matriz de confusión, poniendo en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
    #  Primera fila/columna: negativos
    #  Segunda fila/columna: positivos
    confMatrix = [TN FP; FN TP];
    # Metricas que se derivan de la matriz de confusion:
    acc         = (TN+TP)/(TN+FN+TP+FP);
    errorRate   = 1. - acc;
    # Para sensibilidad, especificidad, VPP y VPN controlamos que algunos casos pueden ser NaN
    #  Para el caso de sensibilidad y especificidad, en un conjunto de entrenamiento estos no pueden ser NaN, porque esto indicaria que se ha intentado entrenar con una unica clase
    #   Sin embargo, sí pueden ser NaN en el caso de aplicar un modelo en un conjunto de test, si este sólo tiene patrones de una clase
    #  Para VPP y VPN, sí pueden ser NaN en caso de que el clasificador lo haya clasificado todo como negativo o positivo respectivamente
    # En estos casos, estas metricas habria que dejarlas a NaN para indicar que no se han podido evaluar
    #  Sin embargo, como es posible que se quiera combinar estos valores al evaluar una clasificacion multiclase, es necesario asignarles un valor. El criterio que se usa aqui es que estos valores seran igual a 0
    # Ademas, hay un caso especial: cuando los VP son el 100% de los patrones, o los VN son el 100% de los patrones
    #  En este caso, el sistema ha actuado correctamente, así que controlamos primero este caso
    if (TN==numInstances) || (TP==numInstances)
        recall = 1.;
        precision = 1.;
        specificity = 1.;
        NPV = 1.;
    else
        recall      = (TP==TP==0.) ? 0. : TP/(TP+FN); # Sensibilidad
        specificity = (TN==FP==0.) ? 0. : TN/(TN+FP); # Especificidad
        precision   = (TP==FP==0.) ? 0. : TP/(TP+FP); # Valor predictivo positivo
        NPV         = (TN==FN==0.) ? 0. : TN/(TN+FN); # Valor predictivo negativo
    end;
    # Calculamos F1, teniendo en cuenta que si sensibilidad o VPP es NaN (pero no ambos), el resultado tiene que ser 0 porque si sensibilidad=NaN entonces VPP=0 y viceversa
    F1          = (recall==precision==0.) ? 0. : 2*(recall*precision)/(recall+precision);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
end;


#
# Aquí teneis otra implementacion de la funcion confusionMatrix:
#
# function confusionMatrix(outputs::AbstractArray{Bool,1}, targets::AbstractArray{Bool,1})
#     @assert(length(outputs)==length(targets));
#     # Para calcular la precision y la tasa de error, se puede llamar a las funciones definidas en la practica 2
#     acc         = accuracy(outputs, targets); # Precision, definida previamente en una practica anterior
#     errorRate   = 1. - acc;
#     recall      = mean(  outputs[  targets]); # Sensibilidad
#     specificity = mean(.!outputs[.!targets]); # Especificidad
#     precision   = mean(  targets[  outputs]); # Valor predictivo positivo
#     NPV         = mean(.!targets[.!outputs]); # Valor predictivo negativo
#     # Controlamos que algunos casos pueden ser NaN
#     #  Para el caso de sensibilidad y especificidad, en un conjunto de entrenamiento estos no pueden ser NaN, porque esto indicaria que se ha intentado entrenar con una unica clase
#     #   Sin embargo, sí pueden ser NaN en el caso de aplicar un modelo en un conjunto de test, si este sólo tiene patrones de una clase
#     #  Para VPP y VPN, sí pueden ser NaN en caso de que el clasificador lo haya clasificado todo como negativo o positivo respectivamente
#     # En estos casos, estas metricas habria que dejarlas a NaN para indicar que no se han podido evaluar
#     #  Sin embargo, como es posible que se quiera combinar estos valores al evaluar una clasificacion multiclase, es necesario asignarles un valor. El criterio que se usa aqui es que estos valores seran igual a 0
#     # Ademas, hay un caso especial: cuando los VP son el 100% de los patrones, o los VN son el 100% de los patrones
#     #  En este caso, el sistema ha actuado correctamente, así que controlamos primero este caso
#     if isnan(recall) && isnan(precision) # Los VN son el 100% de los patrones
#         recall = 1.;
#         precision = 1.;
#     elseif isnan(specificity) && isnan(NPV) # Los VP son el 100% de los patrones
#         specificity = 1.;
#         NPV = 1.;
#     end;
#     # Ahora controlamos los casos en los que no se han podido evaluar las metricas excluyendo los casos anteriores
#     recall      = isnan(recall)      ? 0. : recall;
#     specificity = isnan(specificity) ? 0. : specificity;
#     precision   = isnan(precision)   ? 0. : precision;
#     NPV         = isnan(NPV)         ? 0. : NPV;
#     # Calculamos F1, teniendo en cuenta que si sensibilidad o VPP es NaN (pero no ambos), el resultado tiene que ser 0 porque si sensibilidad=NaN entonces VPP=0 y viceversa
#     F1          = (recall==precision==0.) ? 0. : 2*(recall*precision)/(recall+precision);
#     # Reservamos memoria para la matriz de confusion
#     confMatrix = Array{Int64,2}(undef, 2, 2);
#     # Ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
#     #  Primera fila/columna: negativos
#     #  Segunda fila/columna: positivos
#     # Primera fila: patrones de clase negativo, clasificados como negativos o positivos
#     confMatrix[1,1] = sum(.!targets .& .!outputs); # VN
#     confMatrix[1,2] = sum(.!targets .&   outputs); # FP
#     # Segunda fila: patrones de clase positiva, clasificados como negativos o positivos
#     confMatrix[2,1] = sum(  targets .& .!outputs); # FN
#     confMatrix[2,2] = sum(  targets .&   outputs); # VP
#     return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix)
# end;

confusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = confusionMatrix(outputs.>=threshold, targets);


function confusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    @assert(size(outputs)==size(targets));
    numClasses = size(targets,2);
    # Nos aseguramos de que no hay dos columnas
    @assert(numClasses!=2);
    if (numClasses==1)
        return confusionMatrix(outputs[:,1], targets[:,1]);
    end;

    # Nos aseguramos de que en cada fila haya uno y sólo un valor a true
    @assert(all(sum(outputs, dims=2).==1));
    # Reservamos memoria para las metricas de cada clase, inicializandolas a 0 porque algunas posiblemente no se calculen
    recall      = zeros(numClasses);
    specificity = zeros(numClasses);
    precision   = zeros(numClasses);
    NPV         = zeros(numClasses);
    F1          = zeros(numClasses);
    # Calculamos el numero de patrones de cada clase
    numInstancesFromEachClass = vec(sum(targets, dims=1));
    # Calculamos las metricas para cada clase, esto se haria con un bucle similar a "for numClass in 1:numClasses" que itere por todas las clases
    #  Sin embargo, solo hacemos este calculo para las clases que tengan algun patron
    #  Puede ocurrir que alguna clase no tenga patrones como consecuencia de haber dividido de forma aleatoria el conjunto de patrones entrenamiento/test
    #  En aquellas clases en las que no haya patrones, los valores de las metricas seran 0 (los vectores ya estan asignados), y no se tendran en cuenta a la hora de unir estas metricas
    for numClass in findall(numInstancesFromEachClass.>0)
        # Calculamos las metricas de cada problema binario correspondiente a cada clase y las almacenamos en los vectores correspondientes
        (_, _, recall[numClass], specificity[numClass], precision[numClass], NPV[numClass], F1[numClass], _) = confusionMatrix(outputs[:,numClass], targets[:,numClass]);
    end;

    # Reservamos memoria para la matriz de confusion
    confMatrix = Array{Int64,2}(undef, numClasses, numClasses);
    # Calculamos la matriz de confusión haciendo un bucle doble que itere sobre las clases
    for numClassTarget in 1:numClasses, numClassOutput in 1:numClasses
        # Igual que antes, ponemos en las filas los que pertenecen a cada clase (targets) y en las columnas los clasificados (outputs)
        confMatrix[numClassTarget, numClassOutput] = sum(targets[:,numClassTarget] .& outputs[:,numClassOutput]);
    end;

    # Aplicamos las forma de combinar las metricas macro o weighted
    if weighted
        # Calculamos los valores de ponderacion para hacer el promedio
        weights = numInstancesFromEachClass./sum(numInstancesFromEachClass);
        recall      = sum(weights.*recall);
        specificity = sum(weights.*specificity);
        precision   = sum(weights.*precision);
        NPV         = sum(weights.*NPV);
        F1          = sum(weights.*F1);
    else
        # No realizo la media tal cual con la funcion mean, porque puede haber clases sin instancias
        #  En su lugar, realizo la media solamente de las clases que tengan instancias
        numClassesWithInstances = sum(numInstancesFromEachClass.>0);
        recall      = sum(recall)/numClassesWithInstances;
        specificity = sum(specificity)/numClassesWithInstances;
        precision   = sum(precision)/numClassesWithInstances;
        NPV         = sum(NPV)/numClassesWithInstances;
        F1          = sum(F1)/numClassesWithInstances;
    end;
    # Precision y tasa de error las calculamos con las funciones definidas previamente
    acc = accuracy(outputs, targets);
    errorRate = 1 - acc;

    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;

confusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) = confusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)



function confusionMatrix(outputs::AbstractArray{<:Any,1}, targets::AbstractArray{<:Any,1}; weighted::Bool=true)
    # Comprobamos que todas las clases de salida esten dentro de las clases de las salidas deseadas
    @assert(all([in(output, unique(targets)) for output in outputs]));
    classes = unique([targets; outputs]);
    # Es importante calcular el vector de clases primero y pasarlo como argumento a las 2 llamadas a oneHotEncoding para que el orden de las clases sea el mismo en ambas matrices
    return confusionMatrix(oneHotEncoding(outputs, classes), oneHotEncoding(targets, classes); weighted=weighted);
end;



# Funciones auxiliares para visualizar por pantalla la matriz de confusion y las metricas que se derivan de ella
function printConfusionMatrix(outputs::AbstractArray{Bool,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true)
    (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix) = confusionMatrix(outputs, targets; weighted=weighted);
    numClasses = size(confMatrix,1);
    writeHorizontalLine() = (for i in 1:numClasses+1 print("--------") end; println(""); );
    writeHorizontalLine();
    print("\t| ");
    if (numClasses==2)
        println(" - \t + \t|");
    else
        print.("Cl. ", 1:numClasses, "\t| ");
    end;
    println("");
    writeHorizontalLine();
    for numClassTarget in 1:numClasses
        # print.(confMatrix[numClassTarget,:], "\t");
        if (numClasses==2)
            print(numClassTarget == 1 ? " - \t| " : " + \t| ");
        else
            print("Cl. ", numClassTarget, "\t| ");
        end;
        print.(confMatrix[numClassTarget,:], "\t| ");
        println("");
        writeHorizontalLine();
    end;
    println("Accuracy: ", acc);
    println("Error rate: ", errorRate);
    println("Recall: ", recall);
    println("Specificity: ", specificity);
    println("Precision: ", precision);
    println("Negative predictive value: ", NPV);
    println("F1-score: ", F1);
    return (acc, errorRate, recall, specificity, precision, NPV, F1, confMatrix);
end;
printConfusionMatrix(outputs::AbstractArray{<:Real,2}, targets::AbstractArray{Bool,2}; weighted::Bool=true) =  printConfusionMatrix(classifyOutputs(outputs), targets; weighted=weighted)



printConfusionMatrix(outputs::AbstractArray{Bool,1},   targets::AbstractArray{Bool,1})                      = printConfusionMatrix(reshape(outputs, :, 1), targets);
printConfusionMatrix(outputs::AbstractArray{<:Real,1}, targets::AbstractArray{Bool,1}; threshold::Real=0.5) = printConfusionMatrix(outputs.>=threshold,    targets);




# inputs, targets, numinputs,numtargets, 
# indexvalidation, indextraining, indextest
# numpatterns, pcvalidation, pctest
function RNA(modelHyperparameters, inputs,targets,crossValidationIndices,numFold)
    classtoclass = "Alnus";
    pctest = 0.2;
    showText=false

    possibletargets = unique(targets);
    @assert (isa(classtoclass, String)) "La clase a separar no es un String";
    @assert (any(classtoclass .== possibletargets)) "La clase a separar no es una de las clases"
    classes = unique(targets);
    targets = oneHotEncoding(targets, classes);
   

    testAccuraciesEachRepetition = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);
    testF1EachRepetition         = Array{Float64,1}(undef, modelHyperparameters["numExecutions"]);


    trainingaccuracies = Array{Float64,1}();
    validationaccuracies = Array{Float64,1}();
    testaccuracies = Array{Float64,1}();

    trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
    testInputs        = inputs[crossValidationIndices.==numFold,:];
    trainingTargets   = targets[crossValidationIndices.!=numFold,:];
    testTargets       = targets[crossValidationIndices.==numFold,:];

    # inputs = Array(inputs');
    # targets = Array((targets .== classtoclass)');

    

    numpatterns = size(inputs, 2);
    numinputs = size(inputs, 1);
    numtargets = size(targets, 1);
    architecture = modelHyperparameters["topology"]
    for numTraining in 1:modelHyperparameters["numExecutions"]
        (trainingIndices, validationIndices) = holdOut(size(trainingInputs,1), modelHyperparameters["validationRatio"]*size(trainingInputs,1)/size(inputs,1));
       
        validationInputs = trainingInputs[validationIndices,:]
        validationTargets =  trainingTargets[validationIndices,:]
        trainingInputs= trainingInputs[trainingIndices,:];
        trainingTargets =   trainingTargets[trainingIndices,:]




        # validationDataset = (trainingInputs[validationIndices,:], trainingTargets[validationIndices,:]),

        # testDataset =       (testInputs,                          testTargets);

         # Se supone que tenemos cada patron en cada fila
        # Comprobamos que el numero de filas (numero de patrones) coincide tanto en entrenamiento como en validacion como test
        @assert(size(trainingInputs,   1)==size(trainingTargets,   1));
        @assert(size(testInputs,       1)==size(testTargets,       1));
        @assert(size(validationInputs, 1)==size(validationTargets, 1));
        # Comprobamos que el numero de columnas coincide en los grupos de entrenamiento y validación, si este no está vacío
        !isempty(validationInputs)  && @assert(size(trainingInputs, 2)==size(validationInputs, 2));
        !isempty(validationTargets) && @assert(size(trainingTargets,2)==size(validationTargets,2));
        # Comprobamos que el numero de columnas coincide en los grupos de entrenamiento y test, si este no está vacío
        !isempty(testInputs)  && @assert(size(trainingInputs, 2)==size(testInputs, 2));
        !isempty(testTargets) && @assert(size(trainingTargets,2)==size(testTargets,2));



        # (indextraining, indexvalidation, indextest) = holdOut(numpatterns, modelHyperparameters["validationRatio"], pctest);
        ann = buildClassANN(size(trainingInputs,2), architecture, size(trainingTargets,2));
        trainingLosses   = Float32[];
        validationLosses = Float32[];
        testLosses       = Float32[];
        function calculateLossValues()
            # Calculamos el loss en entrenamiento, validacion y test. Para ello hay que pasar las matrices traspuestas (cada patron en una columna)
            trainingLoss = loss(trainingInputs', trainingTargets');
            showText && print("Epoch ", numEpoch, ": Training loss: ", trainingLoss);
            push!(trainingLosses, trainingLoss);
            if !isempty(validationInputs)
                validationLoss = loss(validationInputs', validationTargets');
                showText && print(" - validation loss: ", validationLoss);
                push!(validationLosses, validationLoss);
            else
                validationLoss = NaN;
            end;
            if !isempty(testInputs)
                testLoss       = loss(testInputs', testTargets');
                showText && print(" - test loss: ", testLoss);
                push!(testLosses, testLoss);
            else
                testLoss = NaN;
            end;
            showText && println("");
            return (trainingLoss, validationLoss, testLoss);
        end;
        # if architecture[2] == 0
        #     ann = Chain( 
        #             Dense(size(trainingInputs,2), architecture[1], σ), 
        #             Dense(architecture[1], size(trainingTargets,2), σ)
        #         );
        # else
        
        # # Más capas ocultas ?
        #     ann = Chain( 
        #             Dense(size(trainingInputs,2), architecture[1], σ), 
        #             Dense(architecture[1], architecture[2], σ),
        #             Dense(architecture[2], size(trainingTargets,2), σ),
        #         );
        # end
        
        loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
        (trainingLoss, validationLoss, _) = calculateLossValues();
        bestann = deepcopy(ann);
        stop = false;
        cycle = 0;  nochangescycles = 0; bestValidationLoss = validationLoss; numEpochsValidation = 0;
        # bestlossvalidation = Inf;
        # bestann = nothing;
        while (!stop)
            Flux.train!(loss, Flux.params(ann), [(trainingInputs', trainingTargets')], ADAM(modelHyperparameters["learningRate"] )); 
            
            cycle += 1;
            # # if (modelHyperparameters["validationRatio"] > 0)
            # if 
            #     lossvalidation = loss(trainingInputs[:,validationIndices], trainingTargets[:,validationIndices]);
            #     if (lossvalidation < bestlossvalidation)
            #         bestlossvalidation = lossvalidation;
            #         bestann = deepcopy(ann);
            #         nochangescycles = 0;
            #     else
            #         nochangescycles += 1;
            #     end;
            # end;

               # Calculamos los valores de loss para este ciclo
            

            # Aplicamos la parada temprana si hay conjunto de validacion
            if (!isempty(validationInputs))
                if (validationLoss<bestValidationLoss)
                    bestValidationLoss = validationLoss;
                    numEpochsValidation = 0;
                    bestann = deepcopy(ann);
                else
                    numEpochsValidation += 1;
                end;
            end;


            if (cycle >= modelHyperparameters["maxEpochs"] )
                stop = true;
            end; 
            if (nochangescycles > modelHyperparameters["maxEpochsVal"])
                stop = true;
            end;
        end;#while
        if (modelHyperparameters["validationRatio"] > 0)
            ann = bestann;
        end;


        # accuracy(x, y) = (size(y, 1) == 1) ? mean((ann(x) .>= 0.5) .== y) : mean(onecold(ann(x)) .== onecold(y));

        # trainingaccuracy = 100 * accuracy(inputs[:,indextraining], targets[:,indextraining]);
        # push!(trainingaccuracies, trainingaccuracy);

        # if (modelHyperparameters["validationRatio"] > 0)
        #     validationaccuracy    = 100 * accuracy(inputs[:,indexvalidation],    targets[:,indexvalidation]);
        #     push!(validationaccuracies,    validationaccuracy);
        # end;

        # testaccuracy          = 100 * accuracy(inputs[:,indextest],          targets[:,indextest]);
        # push!(testaccuracies,          testaccuracy);
        
        (testAccuraciesEachRepetition[numTraining], _, _, _, _, _, testF1EachRepetition[numTraining], _) = confusionMatrix(collect(ann(testInputs')'), testTargets);
    end;#for

    # println("Results:");
    # println("   Training: ", mean(trainingaccuracies), " %, standard deviation: ", std(trainingaccuracies));
    # if (modelHyperparameters["validationRatio"] > 0)
    #     println("   Validation:    ", mean(validationaccuracies),    " %, standard deviation: ", std(validationaccuracies));
    # end;
    # println("   Test:          ", mean(testaccuracies),          " %, standard deviation: ", std(testaccuracies));
    acc = mean(testAccuraciesEachRepetition);
    F1  = mean(testF1EachRepetition);
    return acc, F1;
end#function


##################################################################################################################################################################################
# function normalmaxmin32(patrones::Array{Float32,2})
#     minimo = minimum(patrones[:,3], dims=1);
#     maximo = maximum(patrones[:,3], dims=1);
#     patrones[:,3] .-= minimo;
#     patrones[:,3]  ./= (maximo .- minimo);
#     #patrones[:,vec(minimo.==maximo)] .= 0;
#     return (minimo, maximo)
# end;
# learningRate = 0.01
# pctest = 0.2;
# numRepetitionsANNTraining = 1;
# maxnochangescycles = 1;
# numMaxEpochs = 1; 
# validationRatio = 0.2   ;  
# maxEpochsVal = 6;
# dataset = readdlm("samples2.data",',');
# inputs = dataset[:,1:3];
# targets = dataset[:,end];
# inputs = convert(Array{Float32,2},inputs);
# targets = convert(Array{String,1},targets); 
# inputs = Array(inputs');
# possibletargets = unique(targets);
# @assert (isa(classtoclass, String)) "La clase a separar no es un String";
# @assert (any(classtoclass .== possibletargets)) "La clase a separar no es una de las clases"
# targets = Array((targets .== classtoclass)');
# numpatterns = size(inputs, 2);
# numinputs = size(inputs, 1);
# numtargets = size(targets, 1);
# # @assert (size(inputs,1)==size(targets,1)) "Las matrices de inputs y salidas deseadas no tienen el mismo número de filas"
# normalvalues = normalmaxmin32(inputs);
# trainigaccuracies = Array{Float64,1}();
# validationaccuracies = Array{Float64,1}();
# testaccuracies = Array{Float64,1}();
# x = 1:10;
# y = 0:10;
# z = [prod() for j in y, i in x];
# RNA(modelHyperparameters, inputs, targets)
# png(plot(x, y, z, xlabel = "Neurons 1st layer", ylabel = "Neurons 2nd layer", zlabel = "Test percent", label = "", title = "RNA", st = :surface, fc = :heat), "graphRNA") 
##################################################################################################################################################################################