/**
 * @file ej2.c
 * @author Claudia de la Vieja Lafuente
 * @brief 
 * @version 0.1
 * @date 2024-03-22
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "pgm.c"
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

//Usar send y recieve nornal sin florituras

#define TAG1 1
#define MASTER 0
#define DESTROY_TAG 666

void toASCII(unsigned char** Original, int rows, int cols, char** Salida) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide las filas entre los procesos
    int filaPorProceso = rows / size;
    int filaInicial = rank * filaPorProceso;
    int filaFinal = (rank + 1) * filaPorProceso;

    // Procesa las filas asignadas a este proceso
    for (int i = filaInicial; i < filaFinal; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Convierte el valor de píxel a carácter ASCII
            Salida[i][j] = pixelToChar(Original[i][j]); // Implementa pixelToChar
        }
    }

    // Combina los resultados de todos los procesos
    MPI_Gather(Salida[filaInicial], filaPorProceso * cols, MPI_CHAR,
               Salida[0], filaPorProceso * cols, MPI_CHAR,
               0, MPI_COMM_WORLD);
}

char intToChar(int pixel) {
    if (pixel >= 0 && pixel <= 25) return '#';
    if (pixel >= 26 && pixel <= 51) return '@';
    if (pixel >= 52 && pixel <= 77) return '%';
    if (pixel >= 78 && pixel <= 103) return '+';
    if (pixel >= 104 && pixel <= 129) return '*';
    if (pixel >= 130 && pixel <= 155) return '=';
    if (pixel >= 156 && pixel <= 181) return ':';
    if (pixel >= 182 && pixel <= 207) return '-';
    if (pixel >= 208 && pixel <= 233) return '.';
    if (pixel >= 234 && pixel <= 255) return ' ';
    return '?'; // Carácter desconocido
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int rows, cols; 
    // Load the complete image in the master process (rank 0)
    unsigned char** Original = pgmread(argv[1],&rows,&cols); // Your implementation
    char** Salida = (char**) GetMem2D(rows,cols,sizeof(char));

    toASCII(Original,rows,cols,Salida);

    toDisk(Salida,rows,cols);
}