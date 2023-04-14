using Random

#solo hace falta normalizar el 3er atributo
function normalmaxmin(patrones::Array{Float64,2})
    minimo = minimum(patrones[:,3], dims=1);
    maximo = maximum(patrones[:,3], dims=1);
    patrones[:,3] .-= minimo;
    patrones[:,3]  ./= (maximo .- minimo);
    #patrones[:,vec(minimo.==maximo)] .= 0;
    return (minimo, maximo)
end;

function holdOut(numPatrones::Int, porcentajeTest::Float64)
    @assert ((porcentajeTest>=0.) & (porcentajeTest<=1.));
    indices = randperm(numPatrones);
    numPatronesEntrenamiento = Int(round(numPatrones*(1-porcentajeTest)));
    return (indices[1:numPatronesEntrenamiento], indices[numPatronesEntrenamiento+1:end]);
end


function Euclidean(x::Vector{Float32}, y::Vector{Float32})
    return sqrt(sum((x[i] - y[i])^2 for i in 1:length(x)))
end
