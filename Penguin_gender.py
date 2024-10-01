import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Lectura de archivos 
train_data = pd.read_csv(r'C:\Users\ikerf\Desktop\Upiita\7mo semestre\Patrones\penguins_training.csv')
test_data = pd.read_csv(r'C:\Users\ikerf\Desktop\Upiita\7mo semestre\Patrones\penguins_testing.csv')

# Especies a analizar 
species = {'MALE': 'blue',
           'FEMALE': 'pink'}

#---------------------------------------------------------------------- 
# FUNCIÓN PARA CÁLCULO DE DISTANCIA EUCLIDIANA 
#---------------------------------------------------------------------- 
def distancia_euclidiana(data, centros, num_clases):
    # Calcula la distancia euclidiana entre cada dato y cada centro
    distancias = np.zeros((data.shape[0], num_clases))
    for k in range(num_clases):
        distancias[:, k] = np.sqrt(np.sum((data - centros[k, :]) ** 2, axis=1))
    return distancias

#---------------------------------------------------------------------- 
# FUNCIÓN PARA CÁLCULO DE DISTANCIA DE MAHALANOBIS 
#---------------------------------------------------------------------- 
def distancia_mahalanobis(data, centros):
    # La función np.cov() pide las variables (características) en filas
    matriz_cov = np.cov(data.T)  # por lo que se obtiene la transpuesta para cumplir con este requisito
    inv_cov_matriz = np.linalg.inv(matriz_cov)
    distancias = np.zeros((data.shape[0], centros.shape[0]))
    for k in range(centros.shape[0]):
        for i in range(data.shape[0]):
            delta = data[i, :] - centros[k, :]
            distancias[i, k] = np.sqrt(np.dot(np.dot(delta.T, inv_cov_matriz), delta))
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
    #Calcula la distancia manhattan entre cada dato y cada centro
    distancias = np.zeros((data.shape[0],num_clases))
    for i in range(data.shape[0]):
        distancias[i,0] = np.sum(np.abs(data[i,:] - centros[0,:]))
        distancias[i,1] = np.sum(np.abs(data[i,:] - centros[1,:]))

    return distancias

#----------------------------------------------------------------------
# Crear un diccionario para almacenar los promedios por especie
resultados = {
    'Sex': [],
    'Culmen Length (mm)': [],
    'Culmen Depth (mm)': [],
    'Body Mass (g)': []
}

# Diccionario para almacenar los promedios 
promedios_clases = {}

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for specie, color in species.items():
    subset = train_data[train_data['Sex'] == specie]
    ax.scatter(subset['Culmen Length (mm)'], subset['Culmen Depth (mm)'], subset['Body Mass (g)'], color=color, label=specie, alpha=0.6)
    
    # Formato de la gráfica 
    ax.set_xlabel('Culmen Length (mm)')
    ax.set_ylabel('Culmen Depth (mm)')
    ax.set_zlabel('Body Mass (g)')
    plt.title('TRAINING DATA (MACHO Y HEMBRA)')
    plt.legend()
     
    # Obtiene los promedios de las características de cada especie  
    promediosCL = np.mean(subset['Culmen Length (mm)'])
    promediosCD = np.mean(subset['Culmen Depth (mm)'])
    promediosFL = np.mean(subset['Body Mass (g)'])
    
    # Agregar los resultados al diccionario
    resultados['Sex'].append(specie)
    resultados['Culmen Length (mm)'].append(promediosCL)
    resultados['Culmen Depth (mm)'].append(promediosCD)
    resultados['Body Mass (g)'].append(promediosFL)
    
    # Crear un array con los promedios
    promedio = np.array([promediosCL, promediosCD, promediosFL])
        
    # Guardar el array de promedios en el diccionario con el nombre de la especie
    promedios_clases[specie] = promedio

#---------------------------------------------------------------------- 
# Obtención de los promedios en un arreglo 
centroides_promedio = np.zeros([2, 3])  # Dos clases, tres características
    
centroides_promedio[0, :] = promedios_clases['MALE']
centroides_promedio[1, :] = promedios_clases['FEMALE']

# Crear diccionario para mapear las clases a nombres de especies
clases_mapeo = {0: 'MALE',
                1: 'FEMALE'}

# Extrae los datos del archivo test_data 
longi = np.array(test_data['Culmen Length (mm)'])
prof = np.array(test_data['Culmen Depth (mm)'])
masa = np.array(test_data['Body Mass (g)'])
# Concatena los datos en una sola matriz 
datos = np.column_stack((longi, prof, masa))

#----------------------------------------------------------------------    
# Calcula la distancia EUCLIDIANA entre los datos de test y los centroides 
dist_euclidiana = distancia_euclidiana(datos, centroides_promedio, 2)
asigna_euclidiana = np.argmin(dist_euclidiana, axis=1)
# Calcula la distancia de MAHALANOBIS entre los datos de test y los centroides
dist_mahalanobis = distancia_mahalanobis(datos, centroides_promedio)
asigna_mahalanobis = np.argmin(dist_mahalanobis, axis=1)
# Calcula la distancia de COSENO entre los datos de test y los centroides
dist_coseno = distancia_coseno(datos, centroides_promedio)
# encontrar la clase con la distancia mínima
asigna_coseno = np.argmin(dist_coseno, axis=1)
#Calcula la distancia MANHATTAN entre los datos de test y los centroides 
dist_manhattan = distancia_manhattan(datos,centroides_promedio,2)
#encontrar la clase con la distancia mínima
asigna_manhattan = np.argmin(dist_manhattan,axis = 1)

#---------------------------------------------------------------------- 
# Convertir las clases numéricas en nombres de especies 
test_data['Clase Asignada (Dist euclidiana)'] = [clases_mapeo[clase] for clase in asigna_euclidiana]
test_data['Clase Asignada (Dist Mahalanobis)'] = [clases_mapeo[clase] for clase in asigna_mahalanobis]
test_data['Clase Asignada (Dist Coseno)'] = [clases_mapeo[clase] for clase in asigna_coseno]
test_data['Clase Asignada (Dist Manhattan)'] = [clases_mapeo[clase] for clase in asigna_manhattan]

# Mostrar una tabla con la clase real y la clase asignada
tabla_resultados = test_data[['Species', 'Sex','Clase Asignada (Dist euclidiana)',
                              'Clase Asignada (Dist Mahalanobis)', 'Clase Asignada (Dist Coseno)', 'Clase Asignada (Dist Manhattan)']]
print(tabla_resultados)