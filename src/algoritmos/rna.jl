
using DelimitedFiles
using Flux
using Flux.Losses
using Random

using Statistics
using Flux: onehotbatch, onecold, crossentropy, binarycrossentropy
using Plots



function normalmaxmin32(patrones::Array{Float32,2})
    minimo = minimum(patrones[:,3], dims=1);
    maximo = maximum(patrones[:,3], dims=1);
    patrones[:,3] .-= minimo;
    patrones[:,3]  ./= (maximo .- minimo);
    #patrones[:,vec(minimo.==maximo)] .= 0;
    return (minimo, maximo)
end;

function holdOut(numpatterns::Int, pcvalidation::Float64, pctest::Float64)
    @assert ((pcvalidation>=0.) & (pcvalidation<=1.));
    @assert ((pctest>=0.) & (pctest<=1.));
    @assert ((pcvalidation+pctest)<=1.);
    indices = randperm(numpatterns);
    numpatternsValidacion = Int(round(numpatterns*pcvalidation));
    numpatternsTest = Int(round(numpatterns*pctest));
    numpatternstraining = numpatterns - numpatternsValidacion - numpatternsTest;
    return (indices[1:numpatternstraining], indices[numpatternstraining+1:numpatternstraining+numpatternsValidacion], indices[numpatternstraining+numpatternsValidacion+1:numpatternstraining+numpatternsValidacion+numpatternsTest]);
end
function holdOutt(numpatterns::Int, pctest::Float64)
    @assert ((pctest>=0.) & (pctest<=1.));
    indices = randperm(numpatterns);
    numpatternstraining = Int(round(numpatterns*(1-pctest)));
    return (indices[1:numpatternstraining], indices[numpatternstraining+1:end]);
end

learningRate = 0.01

classtoclass = "Alnus";

executions = 2

pcvalidation = 0.2;

pctest = 0.2;

maxtrainingcycles = 1000;

maxnochangescycles = 100;


#########################################################################################
dataset = readdlm("samples.data",',');

inputs = dataset[:,1:3];
targets = dataset[:,end];

inputs = convert(Array{Float32,2},inputs);
targets = convert(Array{String,1},targets); 

inputs = Array(inputs');

possibletargets = unique(targets);
@assert (isa(classtoclass, String)) "La clase a separar no es un String";
@assert (any(classtoclass .== possibletargets)) "La clase a separar no es una de las clases"

targets = Array((targets .== classtoclass)');

numpatterns = size(inputs, 2);
numinputs = size(inputs, 1);
numtargets = size(targets, 1);
# @assert (size(inputs,1)==size(targets,1)) "Las matrices de inputs y salidas deseadas no tienen el mismo número de filas"

normalvalues = normalmaxmin32(inputs);

trainingaccuracies = Array{Float64,1}();
validationaccuracies = Array{Float64,1}();
testaccuracies = Array{Float64,1}();


function RNA(i, j)
    architecture = [i,j]
    println("\nArchitecture: [", i, ",", j, "]";)
    for exec in 1:executions

        (indextraining, indexvalidation, indextest) = holdOut(numpatterns, pcvalidation, pctest);
        
        if j == 0
            ann = Chain( 
                    Dense(numinputs, architecture[1], σ), 
                    Dense(architecture[1], numtargets, σ)
                );
        else
        # Más capas ocultas ?
            ann = Chain( 
                    Dense(numinputs, architecture[1], σ), 
                    Dense(architecture[1], architecture[2], σ),
                    Dense(architecture[2], numtargets, σ),
                );
        end
        
        loss(x,y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);

        stop = false;
        cycle = 0; bestlossvalidation = Inf; nochangescycles = 0; bestann = nothing;
        while (!stop)
            Flux.train!(loss, Flux.params(ann), [(inputs[:,indextraining], targets[:,indextraining])], ADAM(learningRate)); 
            
            cycle += 1;
            if (pcvalidation > 0)
                lossvalidation = loss(inputs[:,indexvalidation], targets[:,indexvalidation]);
                if (lossvalidation < bestlossvalidation)
                    bestlossvalidation = lossvalidation;
                    bestann = deepcopy(ann);
                    nochangescycles = 0;
                else
                    nochangescycles += 1;
                end;
            end;

            if (cycle >= maxtrainingcycles)
                stop = true;
            end;
            if (nochangescycles > maxnochangescycles)
                stop = true;
            end;
        end;#while
        if (pcvalidation > 0)
            ann = bestann;
        end;


        accuracy(x, y) = (size(y, 1) == 1) ? mean((ann(x) .>= 0.5) .== y) : mean(onecold(ann(x)) .== onecold(y));

        trainingaccuracy = 100 * accuracy(inputs[:,indextraining], targets[:,indextraining]);
        push!(trainingaccuracies, trainingaccuracy);

        if (pcvalidation > 0)
            validationaccuracy    = 100 * accuracy(inputs[:,indexvalidation],    targets[:,indexvalidation]);
            push!(validationaccuracies,    validationaccuracy);
        end;

        testaccuracy          = 100 * accuracy(inputs[:,indextest],          targets[:,indextest]);
        push!(testaccuracies,          testaccuracy);

        
    end;#for

    println("Results:");
    println("   Training: ", mean(trainingaccuracies), " %, standard deviation: ", std(trainingaccuracies));
    if (pcvalidation > 0)
        println("   Validation:    ", mean(validationaccuracies),    " %, standard deviation: ", std(validationaccuracies));
    end;
    println("   Test:          ", mean(testaccuracies),          " %, standard deviation: ", std(testaccuracies));
    return mean(testaccuracies)
end#function

x = 1:10;
y = 0:10;
z = [prod(RNA(i, j)) for j in y, i in x];
png(plot(x, y, z, xlabel = "Neurons 1st layer", ylabel = "Neurons 2nd layer", zlabel = "Test percent", label = "", title = "RNA", st = :surface, fc = :heat), "graphRNA") 