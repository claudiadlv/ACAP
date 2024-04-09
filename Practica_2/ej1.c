/**
 * @file ej1.c
 * @author Claudia de la Vieja Lafuente
 * @brief Calculo aproximado de PI mediante la serie de Leibniz
 * @version 0.1
 * @date 2024-03-15
 * mpicc -o ej3 ej3.c
 * mpiexec -n 4 ./ej3
 * @copyright Copyright (c) 2024
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

double piLeibniz(int steps){
    double partpi = 0.0;
    double num = 1.0;
    double denom = 1.0;
    for(int i = 0; i<steps; i++){
        partpi += num/denom;
        num = -1.0*num; // Alternamos el signo
        denom += 2.0;
    }
    return 4.0*partpi;
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

            double pi = piLeibniz(steps);

            endTime = MPI_Wtime(); // Tiempo final

            if (rank == 0) {
                printf("Valor de PI aproximado [%d intervalos] = \t%lf\n", steps, pi);
                printf("Tiempo de ejecución: %lf segundos\n", endTime - startTime);
            }
        }
    }

    MPI_Finalize();
    return 0;
}