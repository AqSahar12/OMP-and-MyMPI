#include <stdio.h>
#include <mpi.h>
#include "dataset.h" // Include the dataset header

void calculate_max_min_avg_double(double *dataset, int size, double *max, double *min, double *avg)
{
    *max = dataset[0]; 
    *min = dataset[0]; 
    double sum = 0.0;

    for (int i = 0; i < size; i++)
    {
        if (dataset[i] > *max)
            *max = dataset[i]; // Update slowest time
        if (dataset[i] < *min)
            *min = dataset[i]; // Update fastest time
        sum += dataset[i];
    }
    *avg = sum / (double)size;
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv); // Initialize MPI

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs); 

    if (SIZE % num_procs != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: DATASET SIZE (%d) must be divisible by the number of processes (%d) for this scatter implementation.\n", SIZE, num_procs);
        }
        MPI_Finalize();
        return 1;
    }

    int local_data_size = SIZE / num_procs;

    // Start the timer
    double start_time = MPI_Wtime();
    double local_football_sprint_times[local_data_size];
    double local_rugby_sprint_times[local_data_size];
    double local_badminton_sprint_times[local_data_size];
    double local_frisbee_sprint_times[local_data_size];

    MPI_Scatter(football_sprint_times, local_data_size, MPI_DOUBLE, local_football_sprint_times, local_data_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(rugby_sprint_times, local_data_size, MPI_DOUBLE, local_rugby_sprint_times, local_data_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(badminton_sprint_times, local_data_size, MPI_DOUBLE, local_badminton_sprint_times, local_data_size, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    MPI_Scatter(frisbee_sprint_times, local_data_size, MPI_DOUBLE, local_frisbee_sprint_times, local_data_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);   

    double local_slowest_football, local_fastest_football, local_avg_football;
    double local_slowest_rugby, local_fastest_rugby, local_avg_rugby;
    double local_slowest_badminton, local_fastest_badminton, local_avg_badminton;
    double local_slowest_frisbee, local_fastest_frisbee, local_avg_frisbee;

    calculate_max_min_avg_double(local_football_sprint_times, local_data_size, &local_slowest_football, &local_fastest_football, &local_avg_football);
    calculate_max_min_avg_double(local_rugby_sprint_times, local_data_size, &local_slowest_rugby, &local_fastest_rugby, &local_avg_rugby);
    calculate_max_min_avg_double(local_badminton_sprint_times, local_data_size, &local_slowest_badminton, &local_fastest_badminton, &local_avg_badminton);
    calculate_max_min_avg_double(local_frisbee_sprint_times, local_data_size, &local_slowest_frisbee, &local_fastest_frisbee, &local_avg_frisbee);

    double global_slowest_football, global_fastest_football, global_avg_football;
    double global_slowest_rugby, global_fastest_rugby, global_avg_rugby;
    double global_slowest_badminton, global_fastest_badminton, global_avg_badminton;
    double global_slowest_frisbee, global_fastest_frisbee, global_avg_frisbee;

    MPI_Reduce(&local_slowest_football, &global_slowest_football, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_fastest_football, &global_fastest_football, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_avg_football, &global_avg_football, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local_slowest_rugby, &global_slowest_rugby, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_fastest_rugby, &global_fastest_rugby, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_avg_rugby, &global_avg_rugby, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local_slowest_badminton, &global_slowest_badminton, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_fastest_badminton, &global_fastest_badminton, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_avg_badminton, &global_avg_badminton, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local_slowest_frisbee, &global_slowest_frisbee, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_fastest_frisbee, &global_fastest_frisbee, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_avg_frisbee, &global_avg_frisbee, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // End the timer
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;

    if (rank == 0)
    {
        global_avg_football /= num_procs;
        global_avg_rugby /= num_procs;
        global_avg_badminton /= num_procs;
        global_avg_frisbee /= num_procs;

        printf("--- Sports Performance Metrics (Sprint Times) ---\n");
        printf("Football Player - Fastest Sprint: %.2f s, Slowest Sprint: %.2f s, Average Sprint: %.2f s\n", global_fastest_football, global_slowest_football, global_avg_football);
        printf("Rugby Player    - Fastest Sprint: %.2f s, Slowest Sprint: %.2f s, Average Sprint: %.2f s\n", global_fastest_rugby, global_slowest_rugby, global_avg_rugby);
        printf("Badminton Player- Fastest Sprint: %.2f s, Slowest Sprint: %.2f s, Average Sprint: %.2f s\n", global_fastest_badminton, global_slowest_badminton, global_avg_badminton);
        printf("Frisbee Player  - Fastest Sprint: %.2f s, Slowest Sprint: %.2f s, Average Sprint: %.2f s\n", global_fastest_frisbee, global_slowest_frisbee, global_avg_frisbee);
        printf("MPI Execution Time: %.4f seconds\n", elapsed_time);
    }

    MPI_Finalize(); // Finalize MPI
    return 0;
}