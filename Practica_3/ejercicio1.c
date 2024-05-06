/**
 * @file ejercicio1.c
 * @author Claudia de la Vieja Lafuente
 * @brief Escribe un programa que reciba por consola tres parámetros: un tamaño de vector, el
          número de hilos a crear, y un valor de entre {0, 1, 2}, asociado al tipo de reparto de carga.
 * @version 0.1
 * @date 2024-04-11
 * 
 * @copyright Copyright (c) 2024
 * 
 * gcc -o ejercicio1 ejercicio1.c -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define MIN(x, y) (((x) < (y)) ? (x) : (y))

typedef struct {
    int idHilo;
    int numHilos;
    int tipo_reparto;
    int* vector1;
    int* vector2;
    int* resultado;
    int tam_vector;
    int ultimo;
    pthread_mutex_t* mutex;
} tarea;

void* cuerpoHilo(void* arg) {
    tarea misDeberes = *((tarea*)arg);
    int inicio, fin;
    int suma_local = 0;

    if (misDeberes.tipo_reparto == 0) { // Ciclico
        printf("Hilo %d accede a {", misDeberes.idHilo);
        for (int i = misDeberes.idHilo; i < misDeberes.tam_vector; i += misDeberes.numHilos) {
            suma_local += misDeberes.vector1[i] + misDeberes.vector2[i];
            printf("%d", i);
            if (i + misDeberes.numHilos < misDeberes.tam_vector)
                printf(", ");
        }
        printf("}\n");
    } else if (misDeberes.tipo_reparto == 1) { // Por Bloques
        printf("Hilo %d accede a {", misDeberes.idHilo);
        int bloqueSize = misDeberes.tam_vector / misDeberes.numHilos;
        int excedente = misDeberes.tam_vector % misDeberes.numHilos;
        // Cálculo del inicio y fin del bloque para este hilo
        inicio = misDeberes.idHilo * bloqueSize;
        fin = inicio + bloqueSize;
        if(misDeberes.idHilo==misDeberes.ultimo-1){
            fin=fin+excedente;
            //printf("Hilo %d con excedente %d\n", misDeberes.idHilo,excedente);
        }
        for (int i = inicio; i < fin; i++) {
            suma_local += misDeberes.vector1[i] + misDeberes.vector2[i];
            printf("%d", i);
            if (i != fin - 1)
                printf(", ");
        }
        printf("}\n");
    } else if (misDeberes.tipo_reparto == 2) { // Bloques balanceados
        printf("Hilo %d accede a {", misDeberes.idHilo);
        int bloque_size = misDeberes.tam_vector / misDeberes.numHilos;
        int excedente = misDeberes.tam_vector % misDeberes.numHilos;
        // Cálculo del inicio y fin del bloque para este hilo
        inicio = misDeberes.idHilo * bloque_size + MIN(misDeberes.idHilo, excedente);
        fin = inicio + bloque_size + (misDeberes.idHilo < excedente ? 1 : 0);
        for (int i = inicio; i < fin; i++) {
            suma_local += misDeberes.vector1[i] + misDeberes.vector2[i];
            printf("%d", i);
            if (i != fin - 1)
                printf(", ");
            }
        printf("}\n");
    }

    pthread_mutex_lock(misDeberes.mutex);
    *(misDeberes.resultado) += suma_local;
    pthread_mutex_unlock(misDeberes.mutex);

    return NULL;
}


void inicializarVector(int** pointToVec, int tam) {
    *pointToVec = malloc(sizeof(int) * tam);
    int* vec = *pointToVec;

    srand(time(NULL));
    for (int i = 0; i < tam; i++) {
        vec[i] = rand() % 100;
    }
}

void mostrarVector(int* vector, int tam) {
    printf("[");
    for (int i = 0; i < tam; i++) {
        printf("%d", vector[i]);
        if (i != tam - 1) printf(", ");
    }
    printf("]\n");
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Uso: ./programa <tamVector> <numHilos> <tipoReparto>\n");
    } else {
        int tam_vector = atoi(argv[1]);
        int nHilos = atoi(argv[2]);
        int tipo_reparto = atoi(argv[3]);

        int* vector1 = NULL;
        int* vector2 = NULL;
        int* resultado = malloc(sizeof(int));
        *resultado = 0;

        inicializarVector(&vector1, tam_vector);
        inicializarVector(&vector2, tam_vector);

        pthread_t* hilos = malloc(sizeof(pthread_t) * nHilos);
        pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

        tarea* deberes = malloc(sizeof(tarea) * nHilos);
        for (int i = 0; i < nHilos; i++) {
            deberes[i].idHilo = i;
            deberes[i].numHilos = nHilos;
            deberes[i].tipo_reparto = tipo_reparto;
            deberes[i].vector1 = vector1;
            deberes[i].vector2 = vector2;
            deberes[i].resultado = resultado;
            deberes[i].tam_vector = tam_vector;
            deberes[i].mutex = &mutex;
            deberes[i].ultimo=nHilos;
            //printf("nHilos:%d",nHilos);
            pthread_create(&hilos[i], NULL, cuerpoHilo, &deberes[i]);
        }

        for (int i = 0; i < nHilos; i++) {
            pthread_join(hilos[i], NULL);
        }

        if (tam_vector <= 10) {
            printf("Vector 1: ");
            mostrarVector(vector1, tam_vector);
            printf("Vector 2: ");
            mostrarVector(vector2, tam_vector);
        }

        printf("Resultado: %d\n", *resultado);

        free(vector1);
        free(vector2);
        free(resultado);
        free(hilos);
        free(deberes);
    }
    return 0;
}