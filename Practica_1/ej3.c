/**
 * @file ej3.c
 * @author Claudia de la Vieja
 * @brief Haz por tanto un programa que sólo conciba su ejecución con 4 procesos, cada uno con una funcionalidad distinta.
 * @version 0.1
 * @date 2024-03-10
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
#include <ctype.h>


#define TAG1 1
#define TAG2 2
#define TAG3 3
#define MASTER 0
#define DESTROY_TAG 666

char texto[100];

void shutDown(int numProcs){
	for(int i = 1; i<numProcs; i++){//Do not count the master!
		MPI_Send(0, 0, MPI_INT, i, DESTROY_TAG, MPI_COMM_WORLD);//https://stackoverflow.com/questions/10403211/mpi-count-of-zero-is-often-valid
	}
}

void masterTask(int numProcs, int option){
  
    while (option != 0) {
        scanf("%d", &option);
        if (option == 0) {
            printf("Finalizando el programa\n");
            fflush(stdout);
            shutDown(numProcs);
            break;

      } else if (option == 1) {
            printf("Introduzca un texto: ");
            fflush(stdout);
            scanf("%s", texto);
            MPI_Send(&texto, 100, MPI_CHAR, 1, TAG1, MPI_COMM_WORLD);
            MPI_Recv(texto, 100, MPI_CHAR, 1, TAG1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Texto en mayusculas: %s\n", texto);
            printf("Introduzca un comando (0, 1, 2, 3 o 4): ");
            fflush(stdout);

      } else if (option == 2) {

          double result =0;
          double lista[10] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10};
          MPI_Send(&lista, 10, MPI_DOUBLE, 2, TAG2, MPI_COMM_WORLD);

          MPI_Recv(&result, 1, MPI_DOUBLE, 2, TAG2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          printf("La raiz cuadrada de la suma de los numeros es: %f\n", result);
          printf("Introduzca un comando (0, 1, 2, 3 o 4): ");
          fflush(stdout);

      } else if (option == 3) {
        printf("Entrando en funcionalidad 3\n");
        int suma = 0;
        MPI_Send(&texto, 100, MPI_CHAR, 3, TAG3, MPI_COMM_WORLD);

        MPI_Recv(&suma, 1, MPI_INT, 3, TAG3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("La suma de los enteros correspondientes a cada letra es: %d\n", suma);
        printf("Introduzca un comando (0, 1, 2, 3 o 4): ");
        fflush(stdout);

      } else if (option == 4) {
            //Parte del envío de datos
            //Parte1
            printf("Introduzca un texto: ");
            fflush(stdout);
            scanf("%s", texto);
            MPI_Send(&texto, 100, MPI_CHAR, 1, TAG1, MPI_COMM_WORLD);
            //Parte2
            double result =0;
            double lista[10] = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10};
            MPI_Send(&lista, 10, MPI_DOUBLE, 2, TAG2, MPI_COMM_WORLD);
            //Parte3
            printf("Entrando en funcionalidad 3\n");
            int suma = 0;
            MPI_Send(&texto, 100, MPI_CHAR, 3, TAG3, MPI_COMM_WORLD);

            //Recepción de datos
            //Parte1
            MPI_Recv(&texto, 100, MPI_CHAR, 1, TAG1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Texto en mayusculas: %s\n", texto);
            //Parte2
            MPI_Recv(&result, 1, MPI_DOUBLE, 2, TAG2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("La raiz cuadrada de la suma de los numeros es: %f\n", result);
            //Parte3
            MPI_Recv(&suma, 1, MPI_INT, 3, TAG3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("La suma de los enteros correspondientes a cada letra es: %d\n", suma);

            printf("Introduzca un comando (0, 1, 2, 3 o 4): ");
            fflush(stdout);
      } else {
        printf("Comando no valido.\n");
        printf("Introduzca un comando (0, 1, 2, 3 o 4): ");
        fflush(stdout);
      }
    }
}

void worker1(){
  while(1){
    MPI_Status status;
    MPI_Probe(MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      
    if(status.MPI_TAG==TAG1){
        char texto[100];
        MPI_Recv(texto, 100, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            for (int i = 0; texto[i] !='\0'; i++) 
                texto[i] = toupper(texto[i]);
        MPI_Send(&texto, 100, MPI_CHAR, 0, TAG1, MPI_COMM_WORLD);
    }else if (status.MPI_TAG==DESTROY_TAG){
      break;
    }
  }  
}

void worker2(){

  while(1){
    MPI_Status status;
    MPI_Probe(MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

  
    if(status.MPI_TAG==TAG2){

        double lista[10];
        MPI_Recv(lista, 10, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        double sum = 0.0;
        for (int i = 0; i < 10; i++) {
            sum += lista[i];
        }
        double result = sqrt(sum);
        MPI_Send(&result, 1, MPI_DOUBLE, 0, TAG2, MPI_COMM_WORLD);
    }else if (status.MPI_TAG==DESTROY_TAG){
      break;
    } 
  }
    

    
}

void worker3(){
  while(1){
    MPI_Status status;
    MPI_Probe(MASTER, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    
    if(status.MPI_TAG==TAG3){
        
        int sum=0;
        char texto[100];
        MPI_Recv(texto, 100, MPI_CHAR, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            for (int i = 0; texto[i] !='\0'; i++) 
                sum += texto[i];
        MPI_Send(&sum, 1, MPI_INT, 0, TAG3, MPI_COMM_WORLD);
    }else if (status.MPI_TAG==DESTROY_TAG){
      break;
    }

  }
    
}

int main(int argc, char** argv) {
  int numProcs, rank;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  int option = -1;
  double sum;


  if (numProcs != 4) {
    if(rank==0){
      printf("Este programa debe ser ejecutado con 4 procesos MPI.\n");
      fflush(stdout);
      shutDown(numProcs);
    }
    MPI_Finalize();
    return 0;
  }

  if (rank == 0) {
    printf("Introduzca un comando (0, 1, 2, 3 o 4): ");
    fflush(stdout);
    masterTask(numProcs, option);
  }else if (rank == 1) {
    worker1();
      
  }else if (rank == 2) {
    worker2();

    }else if (rank == 3) {
      worker3();
    }
    MPI_Finalize();
    return 0;
}
