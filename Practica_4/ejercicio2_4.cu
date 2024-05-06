#include <cstdio>
#include <cstdlib>     
#include <ctime>       
#include <sys/time.h>  

#define IMDEP 256
#define SIZE (100*1024*1024) // 100 MB
#define THREADS_PER_BLOCK 32 // Mejor valor encontrado anteriormente
#define NBLOCKS 512

const int numRuns = 10;

double get_wall_time(){
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        printf("Error en la medicion de tiempo CPU!!\n");
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void* inicializarImagen(unsigned long nBytes){
    unsigned char * img = (unsigned char*) malloc( nBytes );
    
    for(unsigned long i = 0; i<nBytes; i++){
            img[i] = rand() % IMDEP;
    }
    return img;     
}

void histogramaCPU(unsigned char* img, unsigned long nBytes, unsigned int* histo){
    double aux0 = get_wall_time();
    
    for(int i = 0; i<IMDEP; i++){
        histo[i] = 0;//Inicializacion
    }
    
    for(unsigned long i = 0; i<nBytes; i++){
        histo[img[i]]++;
    }
    
    double aux1 = get_wall_time();
    
    printf("Tiempo de CPU (s): %.4lf\n", aux1-aux0);
}

long calcularCheckSum(unsigned int* histo){
    long checkSum = 0;
    
    for(int i = 0; i<IMDEP; i++){
            checkSum += histo[i];
    }
    
    return checkSum;
}

int compararHistogramas(unsigned int* histA, unsigned int* histB){
    int valido = 1; 
    
    for(int i = 0; i<IMDEP; i++){
        if(histA[i] != histB[i]){
            printf("Error en [%d]: %u != %u\n", i, histA[i], histB[i]);
            valido = 0;
        }
    }
    return valido;
}

__global__ void kernelHistograma(unsigned char* imagen, unsigned long size, unsigned int* histo){

    __shared__ unsigned int temp[IMDEP];
    int focus = threadIdx.x;

    __syncthreads();

    while (focus < IMDEP){
        temp[focus] = 0;
        focus += blockDim.x;
    }
    
    unsigned long i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
    
    while (i < size) {
        atomicAdd( &temp[imagen[i]], 1);
        i += offset;
    }
    
    __syncthreads();
    focus = threadIdx.x;
    
    while ( focus <IMDEP){
        atomicAdd( &(histo[focus]), temp[focus] );
        focus += blockDim.x;
    }
}

int main(void){

    unsigned char* imagen = (unsigned char*) inicializarImagen(SIZE);
    unsigned int histoCPU[IMDEP];
    histogramaCPU(imagen, SIZE, histoCPU);
    long chk = calcularCheckSum(histoCPU);
    printf("Check-sum CPU: %ld\n", chk);

    unsigned char *dev_imagen = 0;
    unsigned int *dev_histo = 0;

    cudaMalloc( (void**) &dev_imagen, SIZE );
    cudaMemcpy( dev_imagen, imagen, SIZE, cudaMemcpyHostToDevice );

    float aveGPUMS = 0.0;

    for(int iter = -1; iter<numRuns; iter++){
        cudaMalloc( (void**) &dev_histo, IMDEP * sizeof( unsigned int) );
        cudaMemset( dev_histo, 0, IMDEP * sizeof( unsigned int ) );
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        if(iter<0){
            kernelHistograma<<<NBLOCKS, THREADS_PER_BLOCK>>>(dev_imagen, SIZE, dev_histo);
        } else {
            cudaDeviceSynchronize();
            cudaEventRecord(start);
            kernelHistograma<<<NBLOCKS, THREADS_PER_BLOCK>>>(dev_imagen, SIZE, dev_histo);
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float milliSeconds = 0.0;
            cudaEventElapsedTime(&milliSeconds, start, stop);
            aveGPUMS += milliSeconds;
        }
        
        cudaFree(dev_histo);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    printf("Tiempo medio de ejecucion del kernel en GPU [s]: %.4f\n", aveGPUMS / (numRuns*1000.0));

    // Calculando tiempo de cálculo en CPU
    double start_cpu = get_wall_time();
    histogramaCPU(imagen, SIZE, histoCPU);
    double end_cpu = get_wall_time();
    double cpu_time = end_cpu - start_cpu;
    printf("Tiempo de ejecucion del cálculo en CPU [s]: %.4lf\n", cpu_time);

    // Calculando aceleración basada en los tiempos de cálculo
    double aceleracion_calculo = cpu_time / (aveGPUMS / (numRuns*1000.0));
    printf("Aceleracion basada en los tiempos de cálculo: %.2lf\n", aceleracion_calculo);

    // Liberando memoria
    free(imagen);
    cudaFree(dev_imagen);
    
    return 0;
}