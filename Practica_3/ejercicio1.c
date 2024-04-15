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
 * gcc -o ejercicio1 ejercici1.c -lpthread
 */

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

typedef struct {
    int idHilo;
    int numHilos;
    int* vector;
    int tamVector;
    pthread_mutex_t* mutex;
    int* resultadoGlobal;
    int tipo_reparto;
} tarea;

void* cuerpoHilo(void* arg) {
    tarea misDeberes = *((tarea*) arg); // Copiamos nuestra asignacion
    int sumaLocal = 0;

    misDeberes.tipo_reparto;
    char* corchetes;

    char corcheteInicio, corcheteFin;
    switch (misDeberes.tipo_reparto) {
        case 0:
            corcheteInicio = '{';
            corcheteFin = '}';
            break;
        case 1:
        case 2:
            corcheteInicio = '[';
            corcheteFin = ')';
            break;
        default:
            corcheteInicio = '[';
            corcheteFin = ')';
            break;
    }

    printf("Hilo %d: %c", misDeberes.idHilo, corcheteInicio); // Imprimir el identificador del hilo

    if (misDeberes.idHilo == misDeberes.numHilos - 1) {
        for (int i = misDeberes.idHilo * (misDeberes.tamVector / misDeberes.numHilos); i < misDeberes.tamVector; i++) {
            sumaLocal += misDeberes.vector[i];
            printf("%d", misDeberes.vector[i]); // Imprimir el valor del vector
            if (i != misDeberes.tamVector - 1) {
                printf(", ");
            }
        }
    } else {
        for (int i = misDeberes.idHilo * (misDeberes.tamVector / misDeberes.numHilos); i < (misDeberes.idHilo + 1) * (misDeberes.tamVector / misDeberes.numHilos); i++) {
            sumaLocal += misDeberes.vector[i];
            printf("%d", misDeberes.vector[i]); // Imprimir el valor del vector
            if (i != (misDeberes.idHilo + 1) * (misDeberes.tamVector / misDeberes.numHilos) - 1) {
                printf(", ");
            }
        }
    }

    printf("%c\n", corcheteFin); // Cerrar la impresión del vector

    pthread_mutex_lock(misDeberes.mutex);

    *(misDeberes.resultadoGlobal) += sumaLocal;

    pthread_mutex_unlock(misDeberes.mutex);
    return 0;
}


void inicializarVector(int** pointToVec, int tam, int tipo_reparto, int numHilos) {
    *pointToVec = malloc(sizeof(int) * tam);
    int* vec = *pointToVec; // Esto es como hacer un "alias", por comodidad

    srand(time(0));

    // Reparto cíclico
    if (tipo_reparto == 0) { 
/*         for (int i = 0; i < tam; i++) {
            vec[i] = i % tam;
        } */

        int currentIdx = 0; // Índice actual del vector

        // Calcular el salto entre elementos para cada hilo
        int jump = (tam + numHilos - 1) / numHilos;

        // Asignar elementos alternados a cada hilo
        for (int j = 0; j < jump; j++) {
            for (int i = j; i < tam; i += jump) {
                vec[currentIdx++] = i; // Asignar el número al vector
            }
        }    
        // Reparto por bloques
    } else if (tipo_reparto == 1) { 
        int blockSize = tam / numHilos;
        int remainder = tam % numHilos;
        int currentIdx = 0;
        
        for (int i = 0; i < numHilos; i++) {
            int blockEnd = currentIdx + blockSize + (i < remainder ? 1 : 0);
        
            for (int j = currentIdx; j < blockEnd; j++) {
                vec[j] = i;
            }
        
            currentIdx = blockEnd;
        }
        // Reparto por bloques balanceados
    } else if (tipo_reparto == 2) { 
        int *used = calloc(tam, sizeof(int)); // Array para rastrear los números usados
        int perThread = tam / numHilos; // Números por hilo
        int remainder = tam % numHilos; // Resto de la división
        int currentIdx = 0; // Índice actual del vector

        for (int i = 0; i < numHilos; i++) {
            int count = perThread + (i < remainder ? 1 : 0); // Números que este hilo debe tomar
            for (int j = 0; j < count; j++) {
                int num;
                do {
                    num = rand() % tam; // Escoger un número al azar
                } while (used[num]); // Comprobar si ya fue usado
                vec[currentIdx++] = num; // Asignar el número al vector
                used[num] = 1; // Marcar el número como usado
            }
        }

        free(used); // Liberar la memoria usada para el rastreo
    }

    // Imprimir el vector
    printf("Vector inicializado: [");
    for (int i = 0; i < tam; i++) {
        printf("%d", vec[i]);
        if (i != tam - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}


int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Uso: %s tamVector numHilos tipoReparto\n", argv[0]);
        return 1;
    }

    int tamVector = atoi(argv[1]);
    int nHilos = atoi(argv[2]);
    int tipo_reparto = atoi(argv[3]);

    int* vector = NULL;
    inicializarVector(&vector, tamVector, tipo_reparto, nHilos);
    int resultado = 0;

    pthread_t* hilos = malloc(sizeof(pthread_t) * nHilos);
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER; // Un mutex para "gobernarlos" a todos
    tarea* deberes = malloc(sizeof(tarea) * nHilos);

    for (int i = 0; i < nHilos; i++) {
        deberes[i].idHilo = i;
        deberes[i].numHilos = nHilos;
        deberes[i].vector = vector;
        deberes[i].tamVector = tamVector;
        deberes[i].mutex = &mutex;
        deberes[i].resultadoGlobal = &resultado;
        deberes[i].tipo_reparto = tipo_reparto;
        pthread_create(&(hilos[i]), NULL, cuerpoHilo, &(deberes[i]));
    }

    for (int i = 0; i < nHilos; i++) {
        pthread_join(hilos[i], NULL);
    }

    printf("Resultado: %d\n", resultado);

    free(hilos);
    free(deberes);
    free(vector);
    
    return 0;
}
