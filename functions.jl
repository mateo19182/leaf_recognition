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


function loadData(index : Int64)
    dataset = readdlm("samples.data",',');
    inputs = convert(Array{Float32,2}, dataset[:,1:index]);
    inputs = convert(Array{Float32}, inputs);
    targets = dataset[:,end]
    targets = convert(Array{String}, targets);
    return inputs, targets; 
end


function modelCrossValidation(fun , modelHyperparameters::Dict, inputs::AbstractArray{<:Real,2}, targets::AbstractArray{<:Any,1}, crossValidationIndices::Array{Int64,1})
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

        acc , F1 = fun(modelHyperparameters,inputs,targets,crossValidationIndices,numFold);

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
