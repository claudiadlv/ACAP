// Calculo aproximado de PI mediante la serie de Leibniz
// https://es.wikipedia.org/wiki/Serie_de_Leibniz
// N.C. Cruz, Universidad de Granada, 2024

#include <stdio.h>
#include <stdlib.h>

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

int main(int argc, char* argv[]){
	if(argc!=2){	//El primer argumento siempre es el nombre del programa
		printf("Uso: ./prog esfuerzo\n");
	}else{
		int steps = atoi(argv[1]);
		if(steps<=0){
			printf("El número de iteraciones debe ser un entero positivo!\n");
		}else{
			double pi = piLeibniz(steps);
			printf("PI por la serie de G. Leibniz [%d iteraciones] =\t%lf\n", steps, pi);
		}
	}
	return 0;
}
