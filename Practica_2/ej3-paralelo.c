/**
 * @file ej3-paralelo.c
 * @author Claudia de la Vieja  
 * @brief Calcula el coeficiente de Tanimoto en paralelo usando MPI.
 * @version 0.2
 * @date 2024-03-27
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

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
    
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 3) {
        if (rank == 0) {
            printf("Uso: %s <tamaño_set1> <tamaño_set2>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int tamA = atoi(argv[1]);
    int tamB = atoi(argv[2]);

    // Reservar memoria para los conjuntos
    int *conjuntoA = malloc(tamA * sizeof(int));
    int *conjuntoB = malloc(tamB * sizeof(int));

    // Inicializar los conjuntos
    for (int i = 0; i < tamA; i++) {
        conjuntoA[i] =  i;
    }
    for (int i = 0; i < tamB; i++) {
        conjuntoB[i] = i * 2 ;
    }

    // Imprimir los conjuntos en cada proceso
    //printf("Proceso %d - Conjunto A: ", rank);
    //for (int i = 0; i < tamA; i++) {
    //    printf("%d ", conjuntoA[i]);
    //}
    //printf("\n");
//
    //printf("Proceso %d - Conjunto B: ", rank);
    //for (int i = 0; i < tamB; i++) {
    //    printf("%d ", conjuntoB[i]);
    //}
    //printf("\n");

    // Inicio del cronómetro
    double tiempo_inicial = MPI_Wtime();

    // Calcular intersección y unión local
    int interseccion_local = interseccionConjuntos(conjuntoA, conjuntoB, tamA, tamB);
    int unionAB_local = unionConjuntos(conjuntoA, conjuntoB, tamA, tamB);

    // Reducción de la intersección y la unión
    int interseccion_global, unionAB_global;
    MPI_Reduce(&interseccion_local, &interseccion_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&unionAB_local, &unionAB_global, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Proceso maestro calcula el coeficiente de Tanimoto total
    if (rank == 0) {
        double tanimoto_total = coeficienteTanimoto(interseccion_global, unionAB_global);
        printf("Coeficiente de Tanimoto entre los conjuntos A y B: %lf\n", tanimoto_total);

        // Fin del cronómetro
        double tiempo_final = MPI_Wtime();
        double tiempo_total = tiempo_final - tiempo_inicial;
        printf("Tiempo de ejecución: %lf segundos\n", tiempo_total);
    }

    free(conjuntoA);
    free(conjuntoB);
    
    MPI_Finalize();

    return 0;
}