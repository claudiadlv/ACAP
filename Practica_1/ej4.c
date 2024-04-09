/**
 * @file ej4.c
 * @author Claudia de la Vieja
 * @brief Aproximacion de pi
 * @version 0.1
 * @date 2024-03-11
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double piRectangles(int intervals, int rank, int numProcs) {
    double ancho = 1.0 / intervals;
    double sum = 0.0, x;

    // Calcular la parte local del sumatorio
    for (int i = rank; i < intervals; i += numProcs) {
        x = (i + 0.5) * ancho;
        sum += 4.0 / (1.0 + x * x);
    }

    // Reducción: sumar los resultados parciales de todos los procesos
    double totalSum;
    MPI_Reduce(&sum, &totalSum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    return totalSum * ancho;
}

int main(int argc, char* argv[]) {
    int rank, numProcs;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    if (argc != 2) {
        if (rank == 0) {
            printf("Uso: %s esfuerzo\n", argv[0]);
        }
    } else {
        int steps = atoi(argv[1]);
        if (steps <= 0) {
            if (rank == 0) { 
                printf("El numero de iteraciones debe ser >= 0\n");
            }
        } else {
            double startTime, endTime;
            startTime = MPI_Wtime(); // Tiempo inicial

            double pi = piRectangles(steps, rank, numProcs);

            endTime = MPI_Wtime(); // Tiempo final            if (rank == 0) {
            
            if (rank == 0) {
                printf("Valor de PI aproximado [%d intervalos] = \t%lf\n", steps, pi);
                printf("Tiempo de ejecución: %lf segundos\n", endTime - startTime);
            }
        }
    }

    MPI_Finalize();
    return 0;
}