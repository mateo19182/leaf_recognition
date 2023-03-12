
using FileIO
using Images

#alto entre anho del bounding box
#centro de masa, estara desplazado si es irregular, comparar con total bouding box
#longitud del camino de pixeles de la silueta (tiene que ser relativo, dividido entre el total del bounding box)
#cuanto rellena la hoja el bounding box
# intensidad de los pixeles ?
######################################################################################################################

imagen = load("Alnus/314.jpg"); 
display(imagen);
matrizBN = gray.(imagen);

labelArray = ImageMorphology.label_components(matrizBN);
println("Se han detectado $(maximum(labelArray)) objetos")

tamanos = component_lengths(labelArray);
etiquetasEliminar = findall(tamanos .<= 100) .- 1; 
matrizBooleana = [!in(etiqueta,etiquetasEliminar) && (etiqueta!=0) for etiqueta in labelArray];
display(Gray.(matrizBN));

labelArray = ImageMorphology.label_components(matrizBooleana);
println("Se han detectado $(maximum(labelArray)) objetos grandes")

imagenObjetos = RGB.(matrizBooleana, matrizBooleana, matrizBooleana);

centroides = ImageMorphology.component_centroids(labelArray)[2:end];
for centroide in centroides
    x = Int(round(centroide[1]));
    y = Int(round(centroide[2]));
    imagenObjetos[ x, y ] = RGB(1,0,0);
    imagenObjetos[ x+1, y ] = RGB(1,0,0);
    imagenObjetos[ x, y+1 ] = RGB(1,0,0);
    imagenObjetos[ x+1, y+1 ] = RGB(1,0,0);
    imagenObjetos[ x-1, y ] = RGB(1,0,0);
    imagenObjetos[ x, y-1 ] = RGB(1,0,0);
    imagenObjetos[ x-1, y-1 ] = RGB(1,0,0);


end;

boundingBoxes = ImageMorphology.component_boxes(labelArray)[2:end];
for boundingBox in boundingBoxes
    x1 = boundingBox[1][1];
    y1 = boundingBox[1][2];
    x2 = boundingBox[2][1];
    y2 = boundingBox[2][2];
    imagenObjetos[ x1:x2 , y1 ] .= RGB(0,1,0);
    imagenObjetos[ x1:x2 , y2 ] .= RGB(0,1,0);
    imagenObjetos[ x1 , y1:y2 ] .= RGB(0,1,0);
    imagenObjetos[ x2 , y1:y2 ] .= RGB(0,1,0);
end;
display(imagenObjetos);

#save("imagenProcesada.jpg", imagenObjetos)
gray_img = Gray.(imagenObjetos);
pixels = convert(Array{Float64}, gray_img);

n_pixels = length(pixels)

# Cuenta el número de píxeles blancos (valor = 1) y negros (valor = 0)
n_white_pixels = count(x -> x == 1, pixels);

porcentaje = n_white_pixels/n_pixels;
# Imprime los resultados
println("Número de píxeles blancos: $n_white_pixels");
println("Porcentaje de pixeles blancos: $porcentaje");
