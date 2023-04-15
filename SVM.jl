using ScikitLearn
using DelimitedFiles 
using Statistics
using Plots
using EvalMetrics

include("functions.jl");

@sk_import svm: SVC 

function SVM(modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1},numFold::Int)
    
    # Dividimos los datos en entrenamiento y test
    trainingInputs    = inputs[crossValidationIndices.!=numFold,:];
    testInputs        = inputs[crossValidationIndices.==numFold,:];
    trainingTargets   = targets[crossValidationIndices.!=numFold];
    testTargets       = targets[crossValidationIndices.==numFold];
    
    svc = SVC(kernel=modelHyperparameters["kernel"], degree=modelHyperparameters["kernelDegree"], gamma=modelHyperparameters["kernelGamma"], C=modelHyperparameters["C"]);
    # Entrenamos el modelo con el conjunto de entrenamiento
    model = fit!(svc, trainingInputs, trainingTargets);

    # Pasamos el conjunto de test
    testOutputs = predict(model, testInputs);

    # Calculamos las metricas correspondientes con la funcion desarrollada en la practica anterior
    (acc, _, _, _, _, _, F1, _) = confusionMatrix(testOutputs, testTargets);
    return acc, F1;
end

function SVM(entrada, salida)
    precisionesEntrenamiento = Array{Float64,1}();
    precisionesTest = Array{Float64,1}();
    x = zeros(0);
    numEjecuciones=50;
    numPatrones = size(entrada, 1);

    for j in 1:1
        println(" C: ", j)  #valor que cambia
        for numEjecucion in 1:numEjecuciones
        println("Ejecucion ", numEjecucion);

            # Radial Basis Function (RBF) seems to be work the best, there is also linear, sigmoid and polynomial
            #C is 1 by default and it’s a reasonable default. If you have a lot of noisy observations, decrease C corresponds to more regularization.
            #gamma defines how much influence a single training example has. The larger gamma is, the closer other examples must be to be affected.
            parameters = Dict("kernel" => "rbf", "gamma" => "auto", "C"=> 20); #j*50
        
            (indicesPatronesEntrenamiento, indicesPatronesTest) = crossValidation(numPatrones, 10);
        
            #The multiclass support is handled according to a one-vs-one scheme.
            model = SVC(kernel=parameters["kernel"], gamma=parameters["gamma"], C=parameters["C"]);
        
            fit!(model, entrada[indicesPatronesEntrenamiento,:], salida[indicesPatronesEntrenamiento]);

            # Calculamos la clasificacion de los patrones de entrenamiento
            clasificacionEntrenamiento = predict(model, entrada[indicesPatronesEntrenamiento,:]);
        
            # Calculamos las distancias al hiperplano de esos mismos patrones
            #distanciasHiperplano = decision_function(model, entrada[indicesPatronesEntrenamiento,:]);
        
            # Calculamos la precision en el conjunto de entrenamiento
            precisionEntrenamiento = 100 * mean(clasificacionEntrenamiento .== salida[indicesPatronesEntrenamiento]);

            # Calculamos la clasificacion de los patrones de test
            clasificacionTest = predict(model, entrada[indicesPatronesTest,:]);
        
            # Calculamos las distancias al hiperplano de esos mismos patrones
            #distanciasHiperplano = decision_function(model, entrada[indicesPatronesTest,:]);
        
            # Calculamos la precision en el conjunto de test
            precisionTest = 100 * mean(clasificacionTest .== salida[indicesPatronesTest]);
            push!(precisionesEntrenamiento, precisionEntrenamiento);
            push!(precisionesTest, precisionTest);

            #@assert(all([in(clasificacionTest, unique(salida[indicesPatronesTest])) for output in clasificacionTest])) 
        
            #matriz d confusion
            #label_map = Dict("Eucalyptus" => 1, "Alnus" => 2, "Cornus" => 3)
            #pred_vec = map(x -> label_map[x], clasificacionTest);
            #true_vec = map(x -> label_map[x], salida[indicesPatronesTest]);
            #cm = ConfusionMatrix(true_vec, pred_vec);      
            #println(cm)

        end;
        #println("   Entrenamiento: ", mean(precisionesEntrenamiento), " %, desviacion tipica: ", std(precisionesEntrenamiento));
        #println("   Test:          ", mean(precisionesTest), " %, desviacion tipica: ", std(precisionesTest));
        #append!(x, mean(precisionesTest))
    end;
    #png(plot(x, label = "", title = "SVM", xlabel = "C*50", ylabel = "Precisión en el Test"),"SVM");
end;


