import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#PRÁCTICA 1: DISTINCIÓN ENTRE ESPECIES DE PINGÜINOS 


#lectura de archivos 
train_data = pd.read_csv(r'C:\Users\ikerf\Desktop\Upiita\7mo semestre\Patrones\penguins_training.csv')
test_data = pd.read_csv(r'C:\Users\ikerf\Desktop\Upiita\7mo semestre\Patrones\penguins_testing.csv')

#Especies a analizar 
species = {'Adelie Penguin (Pygoscelis adeliae)':'blue',
           'Chinstrap penguin (Pygoscelis antarctica)' : 'green',
           'Gentoo penguin (Pygoscelis papua)' : 'red'}

#----------------------------------------------------------------------
#   FUNCIÓN PARA CÁLCULO DE DISTANCIA EUCLIDIANA
#----------------------------------------------------------------------

def distancia_euclidiana(data,centros,num_clases):
    #Calcula la distancia euclidiana entre cada dato y cada centro
    distancias = np.zeros((data.shape[0],num_clases))
    distancias[:,0] = np.sqrt((data[:,0]-centros[0,0])**2+(data[:,1]-centros[0,1])**2+(data[:,2]-centros[0,2])**2)
    distancias[:,1] = np.sqrt((data[:,0]-centros[1,0])**2+(data[:,1]-centros[1,1])**2+(data[:,2]-centros[1,2])**2)
    distancias[:,2] = np.sqrt((data[:,0]-centros[2,0])**2+(data[:,1]-centros[2,1])**2+(data[:,2]-centros[2,2])**2)
    
    return distancias
#----------------------------------------------------------------------
#   FUNCIÓN PARA CÁLCULO DE DISTANCIA DE MAHALANOBIS 
#----------------------------------------------------------------------
def distancia_mahalanobis(data,centros):
    #la función np.cov() pide las variables(caracaterísticas) en filas
    matriz_cov = np.cov(data.T)# por lo que se obtiene la transpuesta para cumplir con este requisito
    inv_cov_matriz = np.linalg.inv(matriz_cov)
    distancias = np.zeros((data.shape[0],3))
    for k in range(0,3,1):
        for i in range(0,data.shape[0],1):
            #for j in range(0,3,1):
            delta = data[i,:] - centros[k,:]
            distancias[i,k] = np.sqrt(np.dot(np.dot(delta.T,inv_cov_matriz),delta))
    return distancias
        
    
#----------------------------------------------------------------------
#   FUNCIÓN PARA CÁLCULO DE DISTANCIA DE COSENO
#----------------------------------------------------------------------
def distancia_coseno(data, centros):
    distancias = np.zeros((data.shape[0], centros.shape[0]))
    for i in range(data.shape[0]):
        for j in range(centros.shape[0]):
            # Producto punto entre el dato y el centroide
            numerador = np.dot(data[i], centros[j])
            # Normas de los vectores
            norma_data = np.linalg.norm(data[i])
            norma_centro = np.linalg.norm(centros[j])
            # Distancia coseno
            distancias[i, j] = 1 - (numerador / (norma_data * norma_centro))
    return distancias


#----------------------------------------------------------------------
#   FUNCIÓN PARA CÁLCULO DE DISTANCIA MANHATTAN
#----------------------------------------------------------------------

def distancia_manhattan(data,centros,num_clases):
    #Calcula la distancia Manhattan entre cada dato y cada centro
    distancias = np.zeros((data.shape[0],num_clases))
    for i in range(data.shape[0]):
        distancias[i,0] = np.sum(np.abs(data[i,:] - centros[0,:]))
        distancias[i,1] = np.sum(np.abs(data[i,:] - centros[1,:]))
        distancias[i,2] = np.sum(np.abs(data[i,:] - centros[2,:]))

    return distancias

#----------------------------------------------------------------------

# Crear un diccionario para almacenar los promedios por especie
resultados = {
    'Species': [],
    'Culmen Length (mm)': [],
    'Culmen Depth (mm)': [],
    'Flipper Length (mm)': []
}


#diccionario para almacenar los promedios 
promedios_clases = {}

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111,projection='3d')
for specie, color in species.items():
    subset = train_data[train_data['Species']==specie]
    ax.scatter(subset['Culmen Length (mm)'],subset['Culmen Depth (mm)'],subset['Flipper Length (mm)'],color=color,label=specie,alpha=0.6)
    
    #formato de la gráfica 
    ax.set_xlabel('Culmen Length (mm)')
    ax.set_ylabel('Culmen Depth (mm)')
    ax.set_zlabel('Flipper Length (mm)')
    plt.title('TRAINING DATA (TODAS LAS MUESTRAS DE LAS ESPECIES)')
    plt.legend()
     
    # #obtiene los promedios de las características de cada especie  
    promediosCL = np.mean(subset['Culmen Length (mm)'])
    promediosCD = np.mean(subset['Culmen Depth (mm)'])
    promediosFL = np.mean(subset['Flipper Length (mm)'])
    # Agregar los resultados al diccionario
    resultados['Species'].append(specie)
    resultados['Culmen Length (mm)'].append(promediosCL)
    resultados['Culmen Depth (mm)'].append(promediosCD)
    resultados['Flipper Length (mm)'].append(promediosFL)
    # Crear un array con los promedios
    promedio = np.array([promediosCL,promediosCD,promediosFL])
        
    # Guardar el array de promedios en el diccionario con el nombre de la especie
    promedios_clases[specie] = promedio

#----------------------------------------------------------------------
# #obtención de los promedios en un arreglo
centroides_promedio = np.zeros([3,3])
    
centroides_promedio[0,:] = promedios_clases['Adelie Penguin (Pygoscelis adeliae)']
centroides_promedio[1,:] = promedios_clases['Chinstrap penguin (Pygoscelis antarctica)']
centroides_promedio[2,:] = promedios_clases['Gentoo penguin (Pygoscelis papua)']

#Crear diccionario para mapear las clases a nombres de especies
clases_mapeo = {0: 'Adelie Penguin (Pygoscelis adeliae)',
                1: 'Chinstrap penguin (Pygoscelis antarctica)',
                2: 'Gentoo penguin (Pygoscelis papua)'}

#Extrae los datos del archivo test_data
longi = np.array(test_data['Culmen Length (mm)'])
prof = np.array(test_data['Culmen Depth (mm)'])
aleta = np.array(test_data['Flipper Length (mm)'])
#concatena los datos en una sola matriz 
datos  = np.column_stack((longi,prof,aleta))
#----------------------------------------------------------------------    

#Calcula la distancia EUCLIDIANA entre los datos de test y los centroides 
dist_euclidiana = distancia_euclidiana(datos,centroides_promedio,3)
#encontrar la clase con la distancia mínima
asigna_euclidiana = np.argmin(dist_euclidiana,axis = 1)
#----------------------------------------------------------------------
#Calcula la distancia de MAHALANOBIS entre los datos de test y los centroides
dist_mahalanobis = distancia_mahalanobis(datos, centroides_promedio)
#encontrar la clase con la distancia mínima
asigna_mahalanobis = np.argmin(dist_mahalanobis,axis = 1)
# Calcula la distancia de COSENO entre los datos de test y los centroides
dist_coseno = distancia_coseno(datos, centroides_promedio)
# encontrar la clase con la distancia mínima
asigna_coseno = np.argmin(dist_coseno, axis=1)
#Calcula la distancia MANHATTAN entre los datos de test y los centroides 
dist_manhattan = distancia_manhattan(datos,centroides_promedio,3)
#encontrar la clase con la distancia mínima
asigna_manhattan = np.argmin(dist_manhattan,axis = 1)
#----------------------------------------------------------------------
# Convertir las clases numéricas en nombres de especies
test_data['Clase Asignada (Dist euclidiana)'] = [clases_mapeo[clase] for clase in asigna_euclidiana]
test_data['Clase Asignada (Dist Mahalanobis)'] = [clases_mapeo[clase] for clase in asigna_mahalanobis]
test_data['Clase Asignada (Dist Coseno)'] = [clases_mapeo[clase] for clase in asigna_coseno]
test_data['Clase Asignada (Dist Manhattan)'] = [clases_mapeo[clase] for clase in asigna_manhattan]

# Mostrar una tabla con la clase real y la clase asignada
tabla_resultados = test_data[['Species', 'Clase Asignada (Dist euclidiana)', 'Clase Asignada (Dist Mahalanobis)',
                              'Clase Asignada (Dist Coseno)', 'Clase Asignada (Dist Manhattan)']]
print(tabla_resultados)

