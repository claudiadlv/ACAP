/**
 * @file ej3.c
 * @author Claudia de la Vieja Lafuente
 * @brief Calculo del coeficiente de Tanimoto Secuencial
 * @version 0.1
 * @date 2024-03-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

// Función para calcular la intersección de dos conjuntos
int interseccionConjuntos(int conjuntoA[], int conjuntoB[], int tamA, int tamB) {
    int interseccion = 0;
    
    for (int i = 0; i < tamA; i++) {
        for (int j = 0; j < tamB; j++) {
            if (conjuntoA[i] == conjuntoB[j]) {
                interseccion++;
                break;
            }
        }
    }
    
    return interseccion;
}

// Función para calcular la unión de dos conjuntos
int unionConjuntos(int conjuntoA[], int conjuntoB[], int tamA, int tamB) {
    int unionAB = 0;
    
    unionAB = tamA + tamB - interseccionConjuntos(conjuntoA, conjuntoB, tamA, tamB);
    
    return unionAB;
}

// Función para calcular el coeficiente de Tanimoto
double coeficienteTanimoto(int interseccion, int unionAB) {
    return (double)interseccion / (double)unionAB;
}

int main(int argc, char *argv[]) {
    
    if (argc != 3) {
        
        printf("Uso: %s <tamaño_set1> <tamaño_set2>\n", argv[0]);
        return 1;
    }

    int tamA = atoi(argv[1]);
    int tamB = atoi(argv[2]);

    // Reservar memoria para los conjuntos
    int *conjuntoA = malloc(tamA * sizeof(int));
    int *conjuntoB = malloc(tamB * sizeof(int));

    for (int i = 0; i < tamA; i++) {
        conjuntoA[i] = i;
    }
    
    for (int i = 0; i < tamB; i++) {
        conjuntoB[i] = 2 * i;
    }

    // Imprimir los conjuntos en cada proceso
    printf("Conjunto A: ");
    for (int i = 0; i < tamA; i++) {
        printf("%d ", conjuntoA[i]);
    }
    printf("\n");

    printf("Conjunto B: ");
    for (int i = 0; i < tamB; i++) {
        printf("%d ", conjuntoB[i]);
    }
    printf("\n");
    
    // Inicio del cronómetro
    clock_t inicio = clock();
    
    // Calculamos la intersección y la unión de los conjuntos
    int interseccion = interseccionConjuntos(conjuntoA, conjuntoB, tamA, tamB);
    int unionAB = unionConjuntos(conjuntoA, conjuntoB, tamA, tamB);
    
    // Calculamos el coeficiente de Tanimoto
    double tanimoto = coeficienteTanimoto(interseccion, unionAB);
    
    // Fin del cronómetro
    clock_t fin = clock();
    
    // Calculamos el tiempo transcurrido
    double tiempo_transcurrido = (double)(fin - inicio) / CLOCKS_PER_SEC;
    
    // Mostramos el resultado
    printf("El coeficiente de Tanimoto entre los conjuntos A y B es: %f\n", tanimoto);
    printf("Tiempo de ejecución: %f segundos\n", tiempo_transcurrido);
    
    return 0;
}