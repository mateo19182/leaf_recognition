using ScikitLearn
using DelimitedFiles 
using Statistics
using Plots

include("functions.jl");

@sk_import svm: SVC 

porcentajeTest=0.5;
numEjecuciones = 50;

bd = readdlm("samples.data",',');
entrada = bd[:,1:3];
entrada = convert(Array{Float64}, entrada);
salida = bd[:,end];
salida = convert(Array{String}, salida);

valoresNormalizados = normalmaxmin(entrada);

numPatrones = size(entrada, 1);

precisionesEntrenamiento = Array{Float64,1}();
precisionesTest = Array{Float64,1}();
x = zeros(0)


for j in 1:100
    println(" C: ", j * 1000)
    for numEjecucion in 1:numEjecuciones
        println("Ejecucion ", numEjecucion);
        parameters = Dict("kernel" => "rbf", "degree" => 3, "gamma" => 2, "C"=> 1); 

        (indicesPatronesEntrenamiento, indicesPatronesTest) = holdOut(numPatrones, porcentajeTest);

        model = SVC(kernel=parameters["kernel"], degree=parameters["degree"], gamma=parameters["gamma"], C=parameters["C"]);
        
        # Ajustamos el model, solo con los patrones de entrenamiento
        fit!(model, entrada[indicesPatronesEntrenamiento,:], salida[indicesPatronesEntrenamiento]);

        # Calculamos la clasificacion de los patrones de entrenamiento
        clasificacionEntrenamiento = predict(model, entrada[indicesPatronesEntrenamiento,:]);
        # Calculamos las distancias al hiperplano de esos mismos patrones
        distanciasHiperplano = decision_function(model, entrada[indicesPatronesEntrenamiento,:]);
        # Calculamos la precision en el conjunto de entrenamiento y la mostramos por pantalla
        precisionEntrenamiento = 100 * mean(clasificacionEntrenamiento .== salida[indicesPatronesEntrenamiento]);
        # println("   Precision en el conjunto de entrenamiento: $precisionEntrenamiento %");

        # Calculamos la clasificacion de los patrones de test
        clasificacionTest = predict(model, entrada[indicesPatronesTest,:]);
        # Calculamos las distancias al hiperplano de esos mismos patrones
        distanciasHiperplano = decision_function(model, entrada[indicesPatronesTest,:]);
        # Calculamos la precision en el conjunto de test y la mostramos por pantalla
        precisionTest = 100 * mean(clasificacionTest .== salida[indicesPatronesTest]);
        # println("   Precision en el conjunto de test: $precisionTest %");

        # Y guardamos esos valores de precision obtenidos en esta ejecucion
        push!(precisionesEntrenamiento, precisionEntrenamiento);
        push!(precisionesTest, precisionTest);

    end;

    println("   Entrenamiento: ", mean(precisionesEntrenamiento), " %, desviacion tipica: ", std(precisionesEntrenamiento));
    println("   Test:          ", mean(precisionesTest), " %, desviacion tipica: ", std(precisionesTest));
    append!(x, mean(precisionesTest))
end

png(plot(x, label = "", title = "SVM", xlabel = "C (x1000)", ylabel = "Precisión tests"),"graphSVM");


println(keys(model));

model.C
model.support_vectors_
model.support_
    
