#include <stdio.h>
#include <omp.h>    
#include <float.h>  
#include "dataset.h" 


void calculate_max_min_avg_omp(double *dataset, int size, double *max, double *min, double *avg)
{
    double current_sum = 0.0;
    double current_max = -DBL_MAX;
    double current_min = DBL_MAX;


    #pragma omp parallel for reduction(+:current_sum) reduction(max:current_max) reduction(min:current_min)
    for (int i = 0; i < size; ++i)
    {
        current_sum += dataset[i];
        if (dataset[i] > current_max)
            current_max = dataset[i]; 
        if (dataset[i] < current_min)
            current_min = dataset[i]; 
    }

    *max = current_max;
    *min = current_min;
    *avg = current_sum / (double)size;
}

int main(int argc, char *argv[])
{

    int nthreads = 4; 
    omp_set_num_threads(nthreads);
    double start_time = omp_get_wtime();

    double fastest_football, slowest_football, avg_football;
    double fastest_rugby, slowest_rugby, avg_rugby;
    double fastest_badminton, slowest_badminton, avg_badminton;
    double fastest_frisbee, slowest_frisbee, avg_frisbee;

    calculate_max_min_avg_omp(football_sprint_times, SIZE, &slowest_football, &fastest_football, &avg_football);
    calculate_max_min_avg_omp(rugby_sprint_times, SIZE, &slowest_rugby, &fastest_rugby, &avg_rugby);
    calculate_max_min_avg_omp(badminton_sprint_times, SIZE, &slowest_badminton, &fastest_badminton, &avg_badminton);
    calculate_max_min_avg_omp(frisbee_sprint_times, SIZE, &slowest_frisbee, &fastest_frisbee, &avg_frisbee);

    // End the timer
    double end_time = omp_get_wtime();
    double elapsed_time = end_time - start_time;

    printf("--- Sports Performance Metrics (Sprint Times) using OpenMP ---\n");
    printf("Number of threads used: %d\n", nthreads);
    printf("Football Player - Fastest Sprint: %.2f s, Slowest Sprint: %.2f s, Average Sprint: %.2f s\n", fastest_football, slowest_football, avg_football);
    printf("Rugby Player    - Fastest Sprint: %.2f s, Slowest Sprint: %.2f s, Average Sprint: %.2f s\n", fastest_rugby, slowest_rugby, avg_rugby);
    printf("Badminton Player- Fastest Sprint: %.2f s, Slowest Sprint: %.2f s, Average Sprint: %.2f s\n", fastest_badminton, slowest_badminton, avg_badminton);
    printf("Frisbee Player  - Fastest Sprint: %.2f s, Slowest Sprint: %.2f s, Average Sprint: %.2f s\n", fastest_frisbee, slowest_frisbee, avg_frisbee);
    printf("OpenMP Execution Time: %.4f seconds\n", elapsed_time);

    return 0;
}