#include <stdio.h>
#include <stdlib.h>

int main() {
    FILE *fp;
    char *filename = "valores.txt";

    // Abrir el archivo en modo lectura
    fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("No se pudo abrir el archivo %s\n", filename);
        return 1;
    }

    // Crear el archivo de script para Gnuplot
    FILE *gnuplot = popen("gnuplot -persistent", "w");
    if (gnuplot == NULL) {
        printf("Error al iniciar Gnuplot.\n");
        return 1;
    }

    // Escribir los comandos Gnuplot en el archivo de script
    fprintf(gnuplot, "set title 'Gráfica de Columnas'\n");
    fprintf(gnuplot, "set xlabel 'Interacciones'\n");
    fprintf(gnuplot, "set ylabel 'Tiempo'\n");
    fprintf(gnuplot, "plot '%s' using 1:2 with boxes\n", filename);

    // Cerrar el archivo de script
    fclose(gnuplot);

    // Leer y descartar la primera línea (encabezado) del archivo
    char buffer[256];
    fgets(buffer, sizeof(buffer), fp);

    // Leer los datos del archivo y escribirlos en un archivo temporal
    FILE *tmpFile = tmpfile();
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
        double x, y;
        sscanf(buffer, "%lf %lf", &x, &y);
        fprintf(tmpFile, "%.2lf %.2lf\n", x, y);
    }

    // Cerrar el archivo de datos original
    fclose(fp);

    // Volver al inicio del archivo temporal
    rewind(tmpFile);

    // Leer el archivo temporal y escribir los datos en el archivo de script
    while (fgets(buffer, sizeof(buffer), tmpFile) != NULL) {
        fprintf(gnuplot, "%s", buffer);
    }

    // Cerrar el archivo temporal
    fclose(tmpFile);

    return 0;
}