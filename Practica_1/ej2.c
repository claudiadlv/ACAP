/**
 * @file ej2.c
 * @author Claudia de la Vieja Lafuente
 * @brief Producto escalar de dos vectores
 * @version 0.1
 * @date 2024-03-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0
#define DESTROY_TAG 666
#define NORMAL_TAG 1

void worker(int rank, int numProcs, int dataSize){
	MPI_Status status;
	MPI_Probe(MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
	
	if(status.MPI_TAG == NORMAL_TAG){
		int numItems = 0;
		MPI_Get_count(&status, MPI_INT, &numItems);//printf("[%d] Expected: %d\n", rank, numItems);
	
		int* myItems = malloc(sizeof(int)*numItems);
		MPI_Recv(myItems, numItems, MPI_INT, MASTER, NORMAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	
		int partialResult = 0;
		for(int i = 0; i<numItems; i++){
			partialResult += myItems[i];
		}
	
		MPI_Send(&partialResult, 1, MPI_INT, MASTER, NORMAL_TAG, MPI_COMM_WORLD);
		free(myItems);
	}
}

void masterTask(int rank, int numProcs, int dataSize){
	int* v = malloc(sizeof(int)*dataSize);
	int* u = malloc (sizeof(int) * dataSize);

    //Inizializar vectores u y v
    for(int i = 0; i < dataSize; i++){
        u[i] = i + 1;
        v[i] = i + 2;
    }

    //Enviar los trabajos a los trabajadores
    for(int i = 1; i < numProcs; i++){
        MPI_Send(u, dataSize, MPI_INT, i, NORMAL_TAG, MPI_COMM_WORLD);
        MPI_Send(v, dataSize, MPI_INT, i, NORMAL_TAG, MPI_COMM_WORLD);
    }

    //Calcular producto escalar local
    int resultado = 0;
    for(int i = 0; i < dataSize; i++){
        resultado += u[i] * v[i];
    }

    //Recibir resultados parcialez de los trabajadores y sumarlos
    int buffer;
    for(int i = 1; i < numProcs; i++){
        MPI_Recv(&buffer, 1, MPI_INT, i, NORMAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        resultado += buffer;
    }

	printf("El resultado es: %d\n", resultado);
	//------------------	
	free(u);
    free(v);
}

void shutDown(int numProcs){
	for(int i = 1; i<numProcs; i++){//Do not count the master!
		MPI_Send(0, 0, MPI_INT, i, DESTROY_TAG, MPI_COMM_WORLD);//https://stackoverflow.com/questions/10403211/mpi-count-of-zero-is-often-valid
	}
}

int main(int argc, char* argv[]){
	int rank, numProcs, dataSize;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	
	if(argc != 2){
		printf("Error: Tamaño de argumentos insuficiente");
		shutDown(numProcs);
	}else{
		dataSize = atoi(argv[1]);
		if(dataSize <= 0){
			printf("Error: Tamaño de datos invalido\n");
			shutDown(numProcs);
		}else{
			if(rank == 0){ //Si el proceso es el master
				masterTask(rank, numProcs, dataSize);
			}else{
				worker(rank, numProcs, dataSize);
			}
		}			
	}

	MPI_Finalize();
	return 0;
}