#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#define ITERATIONS 100 //num of iterations

// ==== Граф у вигляді списку суміжності ====
int **graph;   // Динамічний масив для графа
int *degrees;  // Динамічний масив для кількості сусідів кожного вузла
int *capacities; // Ємність для кожного вузла
int *part; //частини
int numNodes; 
int numParts; 
int numEdges;
const char* filename = "my_graph.txt";
double minConsTime = INFINITY;
double maxConsTime = 0.0;
double avgConsTime = 0.0;
double minParTime = INFINITY;
double maxParTime = 0.0;
double avgParTime = 0.0;

void allocateMemory() {
    graph = (int **)malloc(numNodes * sizeof(int *));
    degrees = (int *)malloc(numNodes * sizeof(int));
    capacities = (int *)malloc(numNodes * sizeof(int));
    part = (int *)malloc(numNodes * sizeof(int));

    if (graph == NULL || degrees == NULL || capacities == NULL || part == NULL) {
        perror("Memory allocation failed!");
        exit(1);
    }

    for (int i = 0; i < numNodes; i++) {
        graph[i] = (int *)malloc(numEdges * sizeof(int));
        degrees[i] = 0;
        capacities[i] = numEdges;
        part[i] = -1;
    }
}

void expandNodeCapacity(int node) {
    capacities[node] *= 2;
    graph[node] = (int *)realloc(graph[node], capacities[node] * sizeof(int));
    if (graph[node] == NULL) {
        perror("Memory reallocation failed!");
        exit(1);
    }
}

// generating and saving graph to file(files needed for repeated usage of 1 graph)
void generateRandomGraph(const char* filename) {
    srand(time(NULL));
    // Очищаємо граф
    for (int i = 0; i < numNodes; i++) {
        degrees[i] = 0;
    }

    // Генерація випадкових з'єднань для кожного вузла
    for (int i = 0; i < numNodes; i++) {
        int desired_edges = rand() % (numEdges + 1);
        
        if (desired_edges == 0) desired_edges = 1;
        while (degrees[i] < desired_edges) {
            int target = rand() % numNodes;
            if (target != i) {
                bool already_connected = false;
                
                // Перевіряємо, чи вже є з'єднання між вузлами
                for (int k = 0; k < degrees[i]; k++) {
                    if (graph[i][k] == target) {
                        already_connected = true;
                        break;
                    }
                }
                for (int k = 0; k < degrees[target]; k++) {
                    if (graph[target][k] == i) {
                        already_connected = true;
                        break;
                    }
                }

                if (!already_connected) {
                    // Якщо потрібно, розширюємо масив для нового з'єднання
                    if (degrees[i] >= capacities[i]) {
                        expandNodeCapacity(i);
                    }
                    if (degrees[target] >= capacities[target]) {
                        expandNodeCapacity(target);
                    }

                    // Додаємо нове з'єднання
                    graph[i][degrees[i]++] = target;
                    graph[target][degrees[target]++] = i;
                }
            }
        }
    }

    // Записуємо граф у файл
    FILE *f = fopen(filename, "w");
    if (!f) {
        perror("Error opening file for writing");
        return;
    }
   
    fprintf(f, "%d\n", numNodes);  // Кількість вузлів
    for (int i = 0; i < numNodes; i++) {
        fprintf(f, "%d", degrees[i]);
        for (int j = 0; j < degrees[i]; j++) {
            fprintf(f, " %d", graph[i][j]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    printf("Graph saved to: %s\n", filename);
}




void loadGraphFromFile(const char *filename) {
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror("Error opening file for reading");
        exit(1);
    }

    fscanf(f, "%d", &numNodes);  // Зчитуємо кількість вузлів

    // --- ВАЖЛИВО: перед читанням очистити старі дані ---
    if (graph != NULL) {
        for (int i = 0; i < numNodes; i++) {
            free(graph[i]);
        }
        free(graph);
        free(degrees);
        free(capacities);
        free(part);
    }

    // Тепер виділяємо нову пам'ять
    allocateMemory();

    for (int i = 0; i < numNodes; i++) {
        fscanf(f, "%d", &degrees[i]);
        
        // Якщо кількість сусідів більша за поточну ємність - розширяємо
        if (degrees[i] > capacities[i]) {
            graph[i] = (int *)realloc(graph[i], degrees[i] * sizeof(int));
            if (graph[i] == NULL) {
                perror("Memory reallocation failed during graph loading!");
                exit(1);
            }
            capacities[i] = degrees[i];
        }

        for (int j = 0; j < degrees[i]; j++) {
            fscanf(f, "%d", &graph[i][j]);
        }
    }

    fclose(f);
}




void printGraph() {
    printf("Graph Representation:\n");
    for (int i = 0; i < numNodes; i++) {
        printf("Node %d: ", i);
        if (degrees[i] == 0) {
            printf("(No edges)\n");
        } else {
            for (int j = 0; j < degrees[i]; j++) {
                printf("%d ", graph[i][j]);
            }
            printf("\n");
        }
    }
}

void print_partition() {
    printf("\nPartitioning result:\n");
    for (int i = 0; i < numNodes; i++) {
        printf("Node %d -> Part %d\n", i, part[i]);
    }
}

// --- Жадібний алгоритм ---
void greedyPartition() {
    for (int i = 0; i < numNodes; i++) {
        int connections[100] = {0}; // максимум 100 частин

        for (int j = 0; j < degrees[i]; j++) {
            int neighbor = graph[i][j];
            if (part[neighbor] != -1) {
                connections[part[neighbor]]++;
            }
        }

        int best_part = 0;
        for (int p = 1; p < numParts; p++) {
            if (connections[p] < connections[best_part]) {
                best_part = p;
            }
        }

        part[i] = best_part;
    }
}

void greedyPartitionParallel(int rank, int size) {
    int chunk = numNodes / size;
    int start = rank * chunk;
    int end = (rank == size - 1) ? numNodes : start + chunk; // останній процес бере все що залишилось

    // Локальний розрахунок для свого підмножини вершин
    for (int i = start; i < end; i++) {
        int connections[100] = {0}; 

        for (int j = 0; j < degrees[i]; j++) {
            int neighbor = graph[i][j];
            if (part[neighbor] != -1) {
                connections[part[neighbor]]++;
            }
        }

        int best_part = 0;
        for (int p = 1; p < numParts; p++) {
            if (connections[p] < connections[best_part]) {
                best_part = p;
            }
        }

        part[i] = best_part;
    }

    // Збираємо всі частини на процесі 0
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                  part, chunk, MPI_INT,
                  MPI_COMM_WORLD);
}

void resetParts(){
    for (int i = 0; i < numNodes; i++) part[i] = -1;
}


void runSequentialPartitioning() {
    double total_time = 0.0;
    double iterationTime = 0.0;
    for (int i = 0; i < ITERATIONS; ++i) {
        resetParts();
        double start_time = MPI_Wtime();
        greedyPartition();
        double end_time = MPI_Wtime();
        iterationTime =end_time - start_time;
        total_time += iterationTime;
        if(minConsTime > iterationTime){
            minConsTime = iterationTime;
        }
        if(maxConsTime < iterationTime){
            maxConsTime = iterationTime;
        }
    }
    avgConsTime = total_time / ITERATIONS;
    printf("Average sequential execution time over %d runs: %f seconds\n", ITERATIONS, avgConsTime);
}

void runParallelPartitioning(int rank, int size) {
    double total_time = 0.0;
    double iterationTime = 0.0;
    for (int i = 0; i < ITERATIONS; ++i) {
        resetParts();
        MPI_Barrier(MPI_COMM_WORLD); // Синхронізація перед початком
        double start_time = MPI_Wtime();
        greedyPartitionParallel(rank, size);
        MPI_Barrier(MPI_COMM_WORLD); // Синхронізація після завершення
        double end_time = MPI_Wtime();
        iterationTime = end_time - start_time;
        total_time += iterationTime;
        if(minParTime > iterationTime){
            minParTime = iterationTime;
        }
        if(maxParTime < iterationTime){
            maxParTime = iterationTime;
        }
    }
    if (rank == 0) {
        avgParTime = total_time / ITERATIONS;
        printf("Average parallel execution time over %d runs: %f seconds\n", ITERATIONS, avgParTime);
    }
}

void freeMemory() {
    if (graph != NULL) {
        for (int i = 0; i < numNodes; i++) {
            if (graph[i] != NULL) {
                free(graph[i]);
            }
        }
        free(graph);
    }
    if (degrees != NULL) free(degrees);
    if (capacities != NULL) free(capacities);
    if (part != NULL) free(part);
}

//**************      **************
//************** MAIN **************
//**************      **************
int main(int argc, char** argv) {


    MPI_Init(&argc, &argv);  // Ініціалізація MPI

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    numNodes = 1000;
    numEdges = 4;
    numParts = 100;
    
    //print_graph();
    if (rank == 0) {
        printf("Number of nodes: %d\n", numNodes);
        printf("Number of edges per node: %d\n", numEdges);
        printf("Number of parts: %d\n", numParts);
        allocateMemory();
        generateRandomGraph(filename);
        runSequentialPartitioning();
    }
    loadGraphFromFile(filename);

    
    runParallelPartitioning(rank, size);

    if(rank == 0){
        printf("Sequential execution times:\n");
        printf("  Min: %f\n", minConsTime);
        printf("  Max: %f\n", maxConsTime);
        printf("  Avg: %f\n", avgConsTime);

        printf("\nParallel execution times:\n");
        printf("  Min: %f\n", minParTime);
        printf("  Max: %f\n", maxParTime);
        printf("  Avg: %f\n", avgParTime);
   


        
    }  // Завершення MPI
    freeMemory();
    MPI_Finalize();
    return 0;
}


