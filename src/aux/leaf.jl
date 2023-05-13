
using FileIO
using Images

######################################################################################################################
#se encarga de leer las imagenes y extraer las características de ellas.
#las caracteristicas son: forma de la imagen, numero de pixeles blancos, numero de pixels de borde,  la simetría ejeX y ejeY, centro de masa.



function loadData()
    #carga el path de las imagenes y las lee. es necesario correrlo desde el directorio base del repositorio.
    path_actual = abspath(pwd())
    rutaAlnus = path_actual*"/datasets/Alnus/"
    rutaEucalyptus = path_actual*"/datasets/Eucalyptus/"
    rutaCornus = path_actual*"/datasets/Cornus/"
    rutaTilia = path_actual*"/datasets/Tilia/"

    alnusImg = (load.(rutaAlnus.*readdir(rutaAlnus)));
    eucImg = (load.(rutaEucalyptus.*readdir(rutaEucalyptus)));
    corImg = (load.(rutaCornus.*readdir(rutaCornus)));
    tilImg = (load.(rutaTilia.*readdir(rutaTilia)));


    dataTxt = open("samples5.data","w");

    writeData(alnusImg, "Alnus", dataTxt);
    writeData(eucImg, "Eucalyptus", dataTxt);
    writeData(corImg, "Cornus", dataTxt);
    writeData(tilImg, "Tilia", dataTxt);


    close(dataTxt);
end;


function sym(imagenObjetos, formaimg)
    #extrae la simetria de la imagen diviendiendo la imagen en 2 por el eje X y el Y y comparando las mitades.
    img_size = size(imagenObjetos);
    img_croppedx1= @view imagenObjetos[ : , floor(Int, 1/2*img_size[2]) : floor(Int, img_size[2])  ]
    img_croppedx2= @view imagenObjetos[ : , floor(Int, 1) : floor(Int, 1/2*img_size[2]) ]
    img_croppedx2r=imresize(img_croppedx2, size(img_croppedx1));
    img_croppedy1= @view imagenObjetos[ floor(Int, 1/2*img_size[1]+1) : floor(Int, img_size[1]), : ]
    img_croppedy2= @view imagenObjetos[ floor(Int, 1) : floor(Int, 1/2*img_size[1]) , : ]
    img_croppedy2r=imresize(img_croppedy2, size(img_croppedy1));
    img_flipx1 = reverse(img_croppedx1, dims=2)
    img_flipy1 = reverse(img_croppedy1, dims=1)
    sym_y=assess_ssim(img_croppedy2r, img_flipy1);
    sym_x=assess_ssim(img_croppedx2r, img_flipx1);
    #dependiendo de si la imagen esta en vertical o en horizontal cambia el orden.
    if (formaimg>1)
        return (sym_x, sym_y)
    else 
        return (sym_y, sym_x)
    end
end

function writeData(imgArray, type::String, dataTxt)
    for imagen in imgArray
        #se pasa la imagen a blanco y negro
        matrizBN = gray.(imagen);
        
        #se detectan objetos en la imagen (la hoja) y nos quedamos con el mas grande.
        labelArray = ImageMorphology.label_components(matrizBN);
        #println("Se han detectado $(maximum(labelArray)) objetos")
        tamanos = component_lengths(labelArray);
        etiquetasEliminar = findall(tamanos .<= 100) .- 1; 
        matrizBooleana = [!in(etiqueta,etiquetasEliminar) && (etiqueta!=0) for etiqueta in labelArray];
        #display(Gray.(matrizBN));
        labelArray = ImageMorphology.label_components(matrizBooleana);
        #println("Se han detectado $(maximum(labelArray)) objetos grandes")

        #se calcula el bounding box y se dibuja
        imagenObjetos = RGB.(matrizBooleana, matrizBooleana, matrizBooleana);
        boundingBoxes = ImageMorphology.component_boxes(labelArray)[2:end];
        boundingBox=boundingBoxes[1];
        x1 = boundingBox[1][1];
        y1 = boundingBox[1][2];
        x2 = boundingBox[2][1];
        y2 = boundingBox[2][2];
        imagenObjetos[ x1:x2 , y1 ] .= RGB(0,1,0);
        imagenObjetos[ x1:x2 , y2 ] .= RGB(0,1,0);
        imagenObjetos[ x1 , y1:y2 ] .= RGB(0,1,0);
        imagenObjetos[ x2 , y1:y2 ] .= RGB(0,1,0);

        #se calcula la forma de la imagen y el total de pixeles en el bounding box.
        formaimg=(y2-y1)/(x2-x1);
        total_pixels_bb=(y2-y1)*(x2-x1)

        #se calcula el center of mass, en este caso equivalente al centroide.
        m, n = size(labelArray)
        centroides = ImageMorphology.component_centroids(labelArray)[2:end];
        centroide = centroides[1]
        xc = Float32(round(centroide[1])/m);
        yc = Float32(round(centroide[2])/n);
        
        gray_img = Gray.(imagenObjetos);
        pixels = convert(Array{Float64}, gray_img);
        n_pixels = length(pixels)
        # Cuenta el número de píxeles blancos (valor = 1) comparado con los negros
        n_white_pixels = count(x -> x == 1, pixels);
        porcentaje_blancos = n_white_pixels/total_pixels_bb;

        # Cuenta el número de píxeles del borde comparado con el total de blancos
        n_pixels_border = border(gray_img)
        porcentaje_borde = n_pixels_border/n_white_pixels;
        
        #simetria eje X, eje Y
        (sym_x, sym_y) = sym(gray_img, formaimg);

        #escribe los resultados al .data
        write(dataTxt, (string(porcentaje_blancos)*","*string(porcentaje_borde)*","*string(formaimg)*","*string(sym_x)*","*string(sym_y)*","*string(xc)*","*string(yc)*","*type*"\n"));


        # Imprime los resultados
        #numero de pixeles blancos / total pixeles bb
        #println("Porcentaje de borde: $porcentaje");
        #numero de pixeles borde / total pixeles blanco imagen
        #println("Porcentaje de pixeles blancos: $porcentaje");
        #ancho entre alto del bounding box
        #println("bb $formaimg")
    end;
end;

function border(img)
    #calcula el reborde de una imagen
    corners =  fastcorners(img, 11, 0.1)
    return count(corners)
end


#si se quiere correr desde este archivo
loadData();