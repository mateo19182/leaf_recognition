
using FileIO
using Images


#alto entre anho del bounding box

#centro de masa, estara desplazado si es irregular, comparar con total bouding box
#longitud del camino de pixeles de la silueta (tiene que ser relativo, dividido entre el total del bounding box)
#cuanto rellena la hoja el bounding box

######################################################################################################################
# Caracteristicas morfologicas de imagenes o partes de imagenes:
# Cargamos la imagen
imagen = load("Eucalyptus/56.jpg"); display(imagen);

# Vamos a detectar los objetos rojos
#  Aquellos cuyo valor de rojo es superior en cierta cantidad al valor de verde y azul
# Definimos en que cantidad queremos que sea mayor
diferenciaRojoVerde = 0.3; diferenciaRojoAzul = 0.3;
canalRojo = red.(imagen); canalVerde = green.(imagen); canalAzul = blue.(imagen);
matrizBooleana = (canalRojo.>(canalVerde.+diferenciaRojoVerde)) .& (canalRojo.>(canalAzul.+diferenciaRojoAzul));
# Mostramos esta matriz booleana para ver que objetos ha encontrado
display(Gray.(matrizBooleana));

# Esto se podria haber hecho, de forma similar, con el siguiente codigo, definiendo primero la funcion a aplicar en todos los pixeles:
esPixelRojo(pixel::RGB) = (pixel.r > pixel.g + diferenciaRojoVerde) && (pixel.r > pixel.b + diferenciaRojoAzul);
# Y despues aplicando esa funcion a toda la imagen haciendo un broadcast:
matrizBooleana = esPixelRojo.(imagen);
display(Gray.(matrizBooleana));

# La siguiente funcion transforma un array booleano (imagen umbralizada) en un array de etiquetas
# Cada grupo de píxeles puesto como "true" en la matriz booleana y conextados se le asigna una etiqueta
# Por ejemplo, la imagen umbralizada
# 0 0 0 0
# 0 1 1 0
# 0 0 1 0
# 0 0 0 0
# 0 1 0 0
# 0 1 0 0
#  contiene 2 objetos, cada pixel se etiqueta como objeto "1", "2", o "0" (ninguno)
labelArray = ImageMorphology.label_components([ 0 0 0 0;
                                                0 1 1 0;
                                                0 0 1 0;
                                                0 0 0 0;
                                                0 1 0 0;
                                                0 1 0 0])
# Resultado:
# 0  0  0  0
# 0  1  1  0
# 0  0  1  0
# 0  0  0  0
# 0  2  0  0
# 0  2  0  0

# Aplicamos esta funcion a la matriz booleana (imagen umbralizada) que construimos antes:
labelArray = ImageMorphology.label_components(matrizBooleana);
# Cuantos objetos se han detectado:
println("Se han detectado $(maximum(labelArray)) objetos")

# A partir de aqui se pueden extraer distintas caracteristicas, como pueden las siguientes:
#  Devuelven una caracteristica por cada etiqueta distinta, incluyendo como primera la etiqueta "0"
boundingBoxes = ImageMorphology.component_boxes(labelArray);
tamanos = ImageMorphology.component_lengths(labelArray);
pixeles = ImageMorphology.component_indices(labelArray);
pixeles = ImageMorphology.component_subscripts(labelArray);
centroides = ImageMorphology.component_centroids(labelArray);

# Sin embargo, suele ser util filtrar los objetos en primer lugar y eliminar los muy grandes o muy pequeños
# Calculamos los tamaños
tamanos = component_lengths(labelArray);
# Que etiquetas son de objetos demasiado pequeños (30 pixeles o menos):
etiquetasEliminar = findall(tamanos .<= 30) .- 1; # Importate el -1, porque la primera etiqueta es la 0
# Se construye otra vez la matriz booleana, a partir de la matriz de etiquetas, pero eliminando las etiquetas indicadas
# Para hacer esto, se hace un bucle sencillo en el que se itera por cada etiqueta
#  Esto se realiza de forma sencilla con la siguiente linea
matrizBooleana = [!in(etiqueta,etiquetasEliminar) && (etiqueta!=0) for etiqueta in labelArray];
display(Gray.(matrizBooleana));


# Con esos objetos rojos "grandes", se toman de nuevo las etiquetas
labelArray = ImageMorphology.label_components(matrizBooleana);
# Cuantos objetos se han detectado:
println("Se han detectado $(maximum(labelArray)) objetos rojos grandes")

# Vamos a situar el centroide de estos objetos en la imagen umbralizada, poniéndolo en color rojo
# Por tanto, hay que construir una imagen en color:
imagenObjetos = RGB.(matrizBooleana, matrizBooleana, matrizBooleana);
# Calculamos los centroides, y nos saltamos el primero (el elemento "0"):
centroides = ImageMorphology.component_centroids(labelArray)[2:end];
# Para cada centroide, ponemos su situacion en color rojo
for centroide in centroides
    x = Int(round(centroide[1]));
    y = Int(round(centroide[2]));
    imagenObjetos[ x, y ] = RGB(1,0,0);
end;

# Vamos a recuadrar el bounding box de estos objetos, en color verde
# Calculamos los bounding boxes, y eliminamos el primero (el objeto "0")
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

# Y hacemos lo mismo con la imagen original:
for boundingBox in boundingBoxes
    x1 = boundingBox[1][1];
    y1 = boundingBox[1][2];
    x2 = boundingBox[2][1];
    y2 = boundingBox[2][2];
    imagen[ x1:x2 , y1 ] .= RGB(0,1,0);
    imagen[ x1:x2 , y2 ] .= RGB(0,1,0);
    imagen[ x1 , y1:y2 ] .= RGB(0,1,0);
    imagen[ x2 , y1:y2 ] .= RGB(0,1,0);
end;
display(imagen);


# Finalmente, guardamos esta imagen
# save("imagenProcesada.jpg", imagen)