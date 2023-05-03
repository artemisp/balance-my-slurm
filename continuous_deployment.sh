#!/bin/bash










# Maximum number of continuous deployments
MAX_ITERATIONS=$1
SBATCH_OPTIONS=$2
# get user
USER = $(whoami)

# Name of the Slurm job script with arguments
SLURM_JOB_SCRIPT="train_with_last_checkpoint.sh"

# Remove the first two arguments (max_iterations, sbatch options) to pass arguments in SLURM_JOB_SCRIPT
shift
shift

echo  $SBATCH_OPTIONS $SLURM_JOB_SCRIPT $@

iteration=0
while [ $iteration -lt $MAX_ITERATIONS ]; do
    # Submit the job and store the job ID
    job_id=$(sbatch $SBATCH_OPTIONS $SLURM_JOB_SCRIPT $@ | awk '{print $4}')

    # Wait for the job to start
    while [ -z "$(squeue -u $USER --format=%i | grep $job_id)" ]; do
        sleep 10
    done

    # Wait for the job to complete
    while [ -n "$(squeue -u $USER --format=%i | grep $job_id)" ]; do
        sleep 10
    done

    echo "Job $job_id completed, checking if the next job can be started."

    iteration=$((iteration + 1))
done

echo "All jobs completed."
