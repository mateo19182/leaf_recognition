
using FileIO
using Images
######################################################################################################################
#Funciones#

function border(img)
    corners =  fastcorners(img, 11, 0.1)
    #img_copy = RGB.(img)
    #img_copy[corners] .= RGB(1.0, 0.0, 0.0)
    ###display(img_copy)
    return count(corners)
end

######################################################################################################################

imagen = load("Alnus/314.jpg"); 
##display(imagen);
matrizBN = gray.(imagen);

labelArray = ImageMorphology.label_components(matrizBN);
println("Se han detectado $(maximum(labelArray)) objetos")

tamanos = component_lengths(labelArray);
etiquetasEliminar = findall(tamanos .<= 100) .- 1; 
matrizBooleana = [!in(etiqueta,etiquetasEliminar) && (etiqueta!=0) for etiqueta in labelArray];
##display(Gray.(matrizBN));

labelArray = ImageMorphology.label_components(matrizBooleana);
println("Se han detectado $(maximum(labelArray)) objetos grandes")

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
formaimg=(y2-y1)/(x2-x1);
total_pixels_bb=(y2-y1)*(x2-x1)



#display(imagenObjetos);

#save("imagenProcesada.jpg", imagenObjetos)
gray_img = Gray.(imagenObjetos);
pixels = convert(Array{Float64}, gray_img);

n_pixels = length(pixels)

# Cuenta el número de píxeles blancos (valor = 1) y negros (valor = 0)
n_white_pixels = count(x -> x == 1, pixels);

porcentaje = n_white_pixels/total_pixels_bb;
# Imprime los resultados

n_pixels_border = border(gray_img)
porcentajeborde = n_pixels_border/n_white_pixels;


#simetria eje X
    #img_size = size(imagenObjetos);
    #img_croppedx1= @view imagenObjetos[ : , floor(Int, 1/2*img_size[2]) : floor(Int, img_size[2]) ]
    #img_croppedx2= @view imagenObjetos[ : , floor(Int, 1) : floor(Int, 1/2*img_size[2]) ]
#simetria eje Y
    #img_croppedy1= @view imagenObjetos[ floor(Int, 1/2*img_size[1]) : floor(Int, img_size[1]), : ]
    #img_croppedy2= @view imagenObjetos[ floor(Int, 1) : floor(Int, 1/2*img_size[1]) , : ]
    #img_flipx1 = reverse(img_croppedx1, dims=2)
    #img_flipy1 = reverse(img_croppedy1, dims=1)
    #plain_diffview = @. img - img_r


#centro de masa

#numero de pixeles blancos / total pixeles bb
println("Porcentaje de borde: $porcentaje");

#numero de pixeles borde / total pixeles blanco imagen
println("Porcentaje de pixeles blancos: $porcentaje");

#ancho entre alto del bounding box
println("bb $formaimg")



