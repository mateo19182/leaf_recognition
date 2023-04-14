using ScikitLearn
using DelimitedFiles
using Statistics
using Plots
include("functions.jl");

# Importar módulo neighbors de ScikitLearn
@sk_import neighbors: KNeighborsClassifier
@sk_import metrics: accuracy_score

# Cargar datos
dataset = readdlm("samples.data",',')
inputs = dataset[:,1:5]
targets = dataset[:,end]
inputs = convert(Array{Float64,2}, inputs)
targets = convert(Array{String,1}, targets)

# Definir tamaño del conjunto de test
pcTest = 0.3
numEjecuciones = 50

# Dividir datos en conjunto de entrenamiento y test
indicesEntrenamiento, indicesTest = holdOut(size(inputs, 1), pcTest)
inputsEntrenamiento = inputs[indicesEntrenamiento,:]
targetsEntrenamiento = targets[indicesEntrenamiento]
inputsTest = inputs[indicesTest,:]
targetsTest = targets[indicesTest]

# Crear instancia de KNeighborsClassifier con el número de vecinos k
#k = 3

precisionesEntrenamiento = Array{Float64,1}();
precisionesTest = Array{Float64,1}();
x = zeros(0)
function knn(k)
    knn = KNeighborsClassifier(n_neighbors=k)

    # Entrenar el modelo con los datos de entrenamiento
    fit!(knn, inputsEntrenamiento, targetsEntrenamiento)

    # Predecir las etiquetas de los datos de test
    prediccionesTest = predict(knn, inputsTest)
    prediccionesTrain= predict(knn, inputsEntrenamiento)

    precisionTrain = 100*accuracy_score(targetsEntrenamiento, prediccionesTrain)
    # Calcular la precisión del modelo
    precisionTest = 100*accuracy_score(targetsTest, prediccionesTest)
    return precisionTest, precisionTrain
end



for k in 1:7
        #println(" Vecino : ", k)
        for numEjecucion in 1:numEjecuciones
            precisionTest, precisionTrain = knn(k)
            push!(precisionesEntrenamiento, precisionTrain);
            push!(precisionesTest, precisionTest);

        end;
        #println("   Entrenamiento: ", mean(precisionesEntrenamiento), " %, desviacion tipica: ", std(precisionesEntrenamiento));
        #println("   Test:          ", mean(precisionesTest), " %, desviacion tipica: ", std(precisionesTest));
        println("",k," & ",mean(precisionesTest)," &  ", std(precisionesTest))
        append!(x, mean(precisionesTest))
end

#println("Precisión en conjunto de test: ", knn(k))
png(plot(x, label = "",xlims=(0,7), ylims=(85,100), title = "KNN", xlabel = "K Vecinos", ylabel = "Precisión en el Test (%)", ),"KNN");
