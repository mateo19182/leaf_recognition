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


    # Preguntar ticher si hace falta

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