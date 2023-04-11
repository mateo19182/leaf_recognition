using ScikitLearn
using DelimitedFiles 
using Statistics
using Plots

include("functions.jl");

@sk_import svm: SVC 


porcentajeTest=0.3;     
numEjecuciones = 50;    

bd = readdlm("samples.data",',');
entrada = bd[:,1:5];
entrada = convert(Array{Float64}, entrada);
salida = bd[:,end];
salida = convert(Array{String}, salida);

valoresNormalizados = normalmaxmin(entrada);

numPatrones = size(entrada, 1);

precisionesEntrenamiento = Array{Float64,1}();
precisionesTest = Array{Float64,1}();
x = zeros(0)


for j in 1:50
    println(" C: ", j)
    for numEjecucion in 1:numEjecuciones
        println("Ejecucion ", numEjecucion);

        # Radial Basis Function (RBF) seems to be work the best, there is also linear, sigmoid and polynomial
        #C is 1 by default and it’s a reasonable default. If you have a lot of noisy observations, decrease C corresponds to more regularization.
        #gamma defines how much influence a single training example has. The larger gamma is, the closer other examples must be to be affected.
        parameters = Dict("kernel" => "rbf", "gamma" => "auto", "C"=> j*50); 
        (indicesPatronesEntrenamiento, indicesPatronesTest) = holdOut(numPatrones, porcentajeTest);
        model = SVC(kernel=parameters["kernel"], gamma=parameters["gamma"], C=parameters["C"]);
        
        fit!(model, entrada[indicesPatronesEntrenamiento,:], salida[indicesPatronesEntrenamiento]);

        # Calculamos la clasificacion de los patrones de entrenamiento
        clasificacionEntrenamiento = predict(model, entrada[indicesPatronesEntrenamiento,:]);
        # Calculamos las distancias al hiperplano de esos mismos patrones
        distanciasHiperplano = decision_function(model, entrada[indicesPatronesEntrenamiento,:]);
        # Calculamos la precision en el conjunto de entrenamiento y la mostramos por pantalla
        precisionEntrenamiento = 100 * mean(clasificacionEntrenamiento .== salida[indicesPatronesEntrenamiento]);

        # Calculamos la clasificacion de los patrones de test
        clasificacionTest = predict(model, entrada[indicesPatronesTest,:]);
        # Calculamos las distancias al hiperplano de esos mismos patrones
        distanciasHiperplano = decision_function(model, entrada[indicesPatronesTest,:]);
        # Calculamos la precision en el conjunto de test y la mostramos por pantalla
        precisionTest = 100 * mean(clasificacionTest .== salida[indicesPatronesTest]);
        push!(precisionesEntrenamiento, precisionEntrenamiento);
        push!(precisionesTest, precisionTest);

    end;

    println("   Entrenamiento: ", mean(precisionesEntrenamiento), " %, desviacion tipica: ", std(precisionesEntrenamiento));
    println("   Test:          ", mean(precisionesTest), " %, desviacion tipica: ", std(precisionesTest));
    append!(x, mean(precisionesTest))
end

png(plot(x, label = "", title = "SVM", xlabel = "C*50", ylabel = "Precisión en el Test"),"SVM");