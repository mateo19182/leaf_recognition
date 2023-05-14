using Flux
using Flux.Losses
using Flux: onehotbatch, onecold
using JLD2, FileIO
using Statistics: mean
using StatsBase
using Images

include("../aux/functions.jl");

function DeepLearning(modelHyperparameters, inputs,targets,crossValidationIndices,numFold)
    function extract(path::AbstractString)
        path_parts = split(path, "/")
        filename = last(path_parts)
        filename_without_ext = split(filename, ".")[1]
        filename_parts = split(filename_without_ext, "_")
        species_name = last(filename_parts)
        return species_name
    end




    path_actual = abspath(pwd())
    path_actual = split(path_actual,"src/")[1]
    train_imgs = (load.(path_actual*"/datasets/train_imgs/".*readdir(path_actual*"/datasets/train_imgs/")));

    #hacer todas las imagenes del mismo tamaño con la moda d x e y;
    #image_sizes = size.(train_imgs);

    #teniendo en cuenta que algunas imagenes estan en vertical y otras en horizontal, es mejor ponerlas cuadradas
    train_imgs = [imresize(img, (512, 512)) for img in train_imgs];

    #crear los labels (a partir del nombre d la imagen
    train_labels = [];

    train_labels = convert.(String, extract.((path_actual*"/datasets/train_imgs/".*readdir(path_actual*"/datasets/train_imgs/"))))

    # push!(train_labels, aux);
    test_labels  =  [];

    test_labels      = convert.(String, extract.((path_actual*"/datasets/train_imgs/".*readdir(path_actual*"/datasets/train_imgs/"))))

    # push!(test_labels, aux1);

    labels = ["cornus", "alnus", "eucalyptus"]; # Las etiquetas



    function convertirArrayImagenesHWCN(imagenes)
        numPatrones = length(imagenes);
        nuevoArray = Array{Float32,4}(undef, 512, 512, 1, numPatrones); # Igual cambiar el 28 por el tamaño de las imagenes
        for i in 1:numPatrones
            @assert (size(imagenes[i])==(512,512)) "Las imagenes no tienen tamaño 512x512"; 
            nuevoArray[:,:,1,i] .= imagenes[i][:,:];
        end;
        return nuevoArray;
    end;


    train_imgs = convertirArrayImagenesHWCN(train_imgs);
    test_imgs = convertirArrayImagenesHWCN(test_imgs);

    println("Tamaño de la matriz de entrenamiento: ", size(train_imgs))
    

    # Cuidado: en esta base de datos las imagenes ya estan con valores entre 0 y 1
    # En otro caso, habria que normalizarlas
    println("Valores minimo y maximo de las entradas: (", minimum(train_imgs), ", ", maximum(train_imgs), ")");



    # Cuantos patrones va a tener cada particion
    patrones = size(train_imgs,4)


    numFolds = 10;
    crossValidationIndices= crossvalidation(size(train_imgs,4), numFolds);

    trainingInputs    = train_imgs[:,:,:,crossValidationIndices.!=numFold];
    testInputs        = train_imgs[:,:,:,crossValidationIndices.==numFold];

    trainingTargets   = train_labels[crossValidationIndices.!=numFold,:];
    testTargets       = train_labels[crossValidationIndices.==numFold,:];



    train_set = [(trainingInputs, onehotbatch(trainingTargets, labels))]; 


    test_set = (testInputs, onehotbatch(testTargets, labels));

    # Hago esto simplemente para liberar memoria, las var   iables train_imgs y test_imgs ocupan mucho y ya no las vamos a usar
    train_imgs = nothing;
    test_imgs = nothing;
    trainingInputs    = nothing;
    testInputs        = nothing;
    GC.gc(); # Pasar el recolector de basura




    funcionTransferenciaCapasConvolucionales = relu;

    # Definimos la red con la funcion Chain, que concatena distintas capas
    ann = Chain(


        Conv((3, 3), 1=>modelHyperparameters["salida"], pad=(1,1), funcionTransferenciaCapasConvolucionales),

        MaxPool((2,2)),

        Conv((3, 3), modelHyperparameters["salida"]=>modelHyperparameters["salida"]*2, pad=(1,1), funcionTransferenciaCapasConvolucionales),

        MaxPool((2,2)),

        Conv((3, 3), modelHyperparameters["salida"]*2=>modelHyperparameters["salida"]*2, pad=(1,1), funcionTransferenciaCapasConvolucionales),

        MaxPool((2,2)),

        x -> reshape(x, :, size(x, 4)),

        Dense(modelHyperparameters["salida"] * 4096 * 2  , 3),

        softmax

    )








    # Definimos la funcion de loss de forma similar a las prácticas de la asignatura
    loss(x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    # Para calcular la precisión, hacemos un "one cold encoding" de las salidas del modelo y de las salidas deseadas, y comparamos ambos vectores
    accuracy(batch) = mean(onecold(ann(batch[1])) .== onecold(batch[2]));


    # Mostramos la precision antes de comenzar el entrenamiento:
    println("Ciclo 0: Precision en el conjunto de entrenamiento: ", 100*mean(accuracy.(train_set)), " %");


    # Optimizador que se usa: ADAM, con esta tasa de aprendizaje:
    opt = ADAM(0.001);


    println("Comenzando entrenamiento...")
    mejorPrecision = -Inf;
    criterioFin = false;
    numCiclo = 0;
    numCicloUltimaMejora = 0;
    mejorModelo = nothing;

    while (!criterioFin)


        # Se entrena un ciclo
        Flux.train!(loss, Flux.params(ann), train_set, opt);

        numCiclo += 1;
        # Se calcula la precision en el conjunto de entrenamiento:
        precisionEntrenamiento = mean(accuracy.(train_set));
        println("Ciclo ", numCiclo, ": Precision en el conjunto de entrenamiento: ", 100*precisionEntrenamiento, " %");

        # Si se mejora la precision en el conjunto de entrenamiento, se calcula la de test y se guarda el modelo
        if (precisionEntrenamiento >= mejorPrecision)
            mejorPrecision = precisionEntrenamiento;
            println("test_set ... ", mejorPrecision, precisionEntrenamiento);
            precisionTest = accuracy(test_set);
            println("   Mejora en el conjunto de entrenamiento -> Precision en el conjunto de test: ", 100*precisionTest, " %");
            mejorModelo = deepcopy(ann);
            numCicloUltimaMejora = numCiclo;
        end

        # Si no se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje
        if (numCiclo - numCicloUltimaMejora >= 5) && (opt.eta > 1e-6)
            opt.eta /= 10.0
            println("   No se ha mejorado en 5 ciclos, se baja la tasa de aprendizaje a ", opt.eta);
            numCicloUltimaMejora = numCiclo;
        end

        # Criterios de parada:

        # Si la precision en entrenamiento es lo suficientemente buena, se para el entrenamiento
        if (precisionEntrenamiento >= 0.999)
            println("   Se para el entenamiento por haber llegado a una precision de 99.9%")
            criterioFin = true;
        end

        # Si no se mejora la precision en el conjunto de entrenamiento durante 10 ciclos, se para el entrenamiento
        if (numCiclo - numCicloUltimaMejora >= 10)
            println("   Se para el entrenamiento por no haber mejorado la precision en el conjunto de entrenamiento durante 10 ciclos")
            criterioFin = true;
        end
    end

    return mejorPrecision, mejorPrecision
end
