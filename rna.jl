# Requirements
using DelimitedFiles
using Flux
using Flux.Losses

#Global vars
classtoclass = "Alnus"
maxcycles = 1000;
executions = 50;
maxcyclesnovalidation= 100
pcvalidation = 0.2;
pctest= 0.2;

# to store accuracity results
acctrain = Array{Float64,1}();
accvalidation = Array{Float64,1}();
acctest = Array{Float64,1}();




######################################################################################################################################################################################
#load dataset
dataset = readdlm("ejemplo.data",',');

#first 4 colums are inputs, 5th one is output/target
inputs = dataset[:,1:4];
targets = dataset[:,5];

#convert types
inputs = convert(Array{Float32,2},inputs);
targets = convert(Array{String,1}, targets); 

#just a defensive programming moment
@assert (size(inputs,1)==size(targets,1)) "Error: Diff rows numbers on inputs and targets matrixes " 
posibletargets = unique(targets);
@assert (isa(classtoclass, String)) "Error : class to classify is not String";
@assert (any(classtoclass .== posibletargets)) "Error: class to clasify is not a possible one";

# convert classes to true (1) false (0) values and traspose matrixes
desiredetargets = Array((posibletargets .== classtoclass)');
inputs = Array(inputs)';

# 
numpatterns = size(inputs, 2);
numinputs = size(inputs, 1);
numtargets = size(desiredetargets, 1);

@assert (numtargets != 2);

# TO DO: normalize , only inputs ( classify problem )
normal = normalmaxmin(inputs)

###
function rna()
    #TO DO: add rna parameterization
    for numexec in 1:executions
        #TO DO: add holdout function
        indexes = holdout();

        ann = Chain( Dense(numEntradasRNA, 4, σ), Dense(4, 1, σ) ); 

        loss(x, y) = Losses.binarycrossentropy(ann(x), y) 


        cycle = 0;
        stop = false; bestlossvalidation = Inf; cyclesnovalidation = 0; bestmodel = nothing;
        while(!stop)
            #TO DO: add indexes to inputs targetrs from holdout
            Flux.train!(loss, params(ann), [(inputs, targets)], ADAM(0.1)); 
            cycle += 1;
            #validation
            if (pcvalidation > 0)
                #TO DO: add indexes to inputs targetrs from holdout
                lossvalidation = loss(inputs, targets);
                if (lossvalidation < bestlossvalidation)
                    bestlossvalidation = lossvalidation;
                    bestmodel = deepcopy(ann);
                    cyclesnovalidation = 0;
                else
                    cyclesnovalidation += 1;
                end;
            end;


            if (cycle >= maxcycles) || (cyclesnovalidation >= maxcyclesnovalidation )
                stop = true;
            end;

        end; # while
        
        if (pcvalidation > 0)
            model = bestmodel;
        end;

        #TO DO: accuracy shit 

    end; # for
end # rna
###

nn = rna()
#TO DO: print plots shit
######################################################################################################################################################################################