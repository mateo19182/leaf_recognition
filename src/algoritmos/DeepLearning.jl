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
    test_imgs = (load.(path_actual*"/datasets/train_imgs/".*readdir(path_actual*"/datasets/train_imgs/")));

    #hacer todas las imagenes del mismo tamaño con la moda d x e y;
    #image_sizes = size.(train_imgs);
    #mode_size = mode(image_sizes);
    #train_imgs = [imresize(img, (mode_size[1], mode_size[2])) for img in train_imgs];
    #test_imgs = [imresize(img, (mode_size[1], mode_size[2])) for img in test_imgs];

    #teniendo en cuenta que algunas imagenes estan en vertical y otras en horizontal, es mejor ponerlas cuadradas
    train_imgs = [imresize(img, (512, 512)) for img in train_imgs];
    test_imgs = [imresize(img, (512, 512)) for img in test_imgs];

    #crear los labels (a partir del nombre d la imagen
    train_labels = [];

    train_labels = convert.(String, extract.((path_actual*"/datasets/train_imgs/".*readdir(path_actual*"/datasets/train_imgs/"))))

    # push!(train_labels, aux);
    test_labels  =  [];

    test_labels      = convert.(String, extract.((path_actual*"/datasets/train_imgs/".*readdir(path_actual*"/datasets/train_imgs/"))))

    # push!(test_labels, aux1);

    labels = ["cornus", "alnus", "eucalyptus"]; # Las etiquetas

    # Tanto train_imgs como test_imgs son arrays de arrays bidimensionales (arrays de imagenes), es decir, son del tipo Array{Array{Float32,2},1}
    #  Generalmente en Deep Learning los datos estan en tipo Float32 y no Float64, es decir, tienen menos precision
    #  Esto se hace, entre otras cosas, porque las tarjetas gráficas (excepto las más recientes) suelen operar con este tipo de dato
    #  Si se usa Float64 en lugar de Float32, el sistema irá mucho más lento porque tiene que hacer conversiones de Float64 a Float32

    # Para procesar las imagenes con Deep Learning, hay que pasarlas una matriz en formato HWCN
    #  Es decir, Height x Width x Channels x N
    #  En el caso de esta base de datos
    #   Height = 28
    #   Width = 28
    #   Channels = 1 -> son imagenes en escala de grises
    #     Si fuesen en color, Channels = 3 (rojo, verde, azul)
    # Esta conversion se puede hacer con la siguiente funcion:


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
    println("Tamaño de la matriz de test:          ", size(test_imgs))
    

    # Cuidado: en esta base de datos las imagenes ya estan con valores entre 0 y 1
    # En otro caso, habria que normalizarlas
    println("Valores minimo y maximo de las entradas: (", minimum(train_imgs), ", ", maximum(train_imgs), ")");



    # Cuando se tienen tantos patrones de entrenamiento (en este caso 60000),
    #  generalmente no se entrena pasando todos los patrones y modificando el error
    #  En su lugar, el conjunto de entrenamiento se divide en subconjuntos (batches)
    #  y se van aplicando uno a uno

    # Hacemos los indices para las particiones
    # Cuantos patrones va a tener cada particion
    batch_size = size(train_imgs,4)
    # Creamos los indices: partimos el vector 1:N en grupos de batch_size
    gruposIndicesBatch = Iterators.partition(1:size(train_imgs,4), batch_size);
    println("He creado ", length(gruposIndicesBatch), " grupos de indices para distribuir los patrones en batches");

    numFolds = 10;
    crossValidationIndices= crossvalidation(batch_size, numFolds);

    trainingInputs    = train_imgs[:,:,:,crossValidationIndices.!=numFold];
    testInputs        = train_imgs[:,:,:,crossValidationIndices.==numFold];
    println(size(train_labels))
    println(size(crossValidationIndices))
    trainingTargets   = train_labels[crossValidationIndices.!=numFold,:];
    testTargets       = train_labels[crossValidationIndices.==numFold,:];


    # Creamos el conjunto de entrenamiento: va a ser un vector de tuplas. Cada tupla va a tener
    #  Como primer elemento, las imagenes de ese batch
    #     train_imgs[:,:,:,indicesBatch]
    #  Como segundo elemento, las salidas deseadas (en booleano, codificadas con one-hot-encoding) de esas imagenes
    #     Para conseguir estas salidas deseadas, se hace una llamada a la funcion onehotbatch, que realiza un one-hot-encoding de las etiquetas que se le pasen como parametros
    #     onehotbatch(train_labels[indicesBatch], labels)
    #  Por tanto, cada batch será un par dado por
    #     (train_imgs[:,:,:,indicesBatch], onehotbatch(train_labels[indicesBatch], labels))
    # Sólo resta iterar por cada batch para construir el vector de batches

    train_set = [(trainingInputs, onehotbatch(trainingTargets, labels))]; 

    # Creamos un batch similar, pero con todas las imagenes de test
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




    # Vamos a probar la RNA capa por capa y poner algunos datos de cada capa
    # # Usaremos como entrada varios patrones de un batch
    numBatchCoger = 1; numImagenEnEseBatch = [12, 6];
    # # Para coger esos patrones de ese batch:
    # #  train_set es un array de tuplas (una tupla por batch), donde, en cada tupla, el primer elemento son las entradas y el segundo las salidas deseadas
    # #  Por tanto:
    # #   train_set[numBatchCoger] -> La tupla del batch seleccionado
    # #   train_set[numBatchCoger][1] -> El primer elemento de esa tupla, es decir, las entradas de ese batch
    # #   train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch] -> Los patrones seleccionados de las entradas de ese batch
    entradaCapa = train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch];
    numCapas = length(Flux.params(ann));
    println("La RNA tiene ", numCapas, " capas:");
    for numCapa in 1:numCapas
        println("   Capa ", numCapa, ": ", ann[numCapa]);
        println("   EntradaCapa ", size(entradaCapa));

        # Le pasamos la entrada a esta capa
        #global entradaCapa # Esta linea es necesaria porque la variable entradaCapa es global y se modifica en este bucle
        capa = ann[numCapa];
        salidaCapa = capa(entradaCapa);
        println("      La salida de esta capa tiene dimension ", size(salidaCapa));
        entradaCapa = salidaCapa;
    end



    # # Sin embargo, para aplicar un patron no hace falta hacer todo eso.
    # #  Se puede aplicar patrones a la RNA simplemente haciendo, por ejemplo
    # ann(train_set[numBatchCoger][1][:,:,:,numImagenEnEseBatch]);




    # Definimos la funcion de loss de forma similar a las prácticas de la asignatura
    loss(x, y) = (size(y,1) == 1) ? Losses.binarycrossentropy(ann(x),y) : Losses.crossentropy(ann(x),y);
    # Para calcular la precisión, hacemos un "one cold encoding" de las salidas del modelo y de las salidas deseadas, y comparamos ambos vectores
    accuracy(batch) = mean(onecold(ann(batch[1])) .== onecold(batch[2]));
    # Un batch es una tupla (entradas, salidasDeseadas), asi que batch[1] son las entradas, y batch[2] son las salidas deseadas


    # Mostramos la precision antes de comenzar el entrenamiento:
    #  train_set es un array de batches
    #  accuracy recibe como parametro un batch
    #  accuracy.(train_set) hace un broadcast de la funcion accuracy a todos los elementos del array train_set
    #   y devuelve un array con los resultados
    #  Por tanto, mean(accuracy.(train_set)) calcula la precision promedia
    #   (no es totalmente preciso, porque el ultimo batch tiene menos elementos, pero es una diferencia baja)
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

        # Hay que declarar las variables globales que van a ser modificadas en el interior del bucle
        #global numCicloUltimaMejora, numCiclo, mejorPrecision, mejorModelo, criterioFin;
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
