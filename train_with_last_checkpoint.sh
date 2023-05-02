#!/bin/bash
#

# Set the path to the checkpoints folder
PYTHON_BINARY=$1
CHECKPOINTS_DIR=$2
echo checkpoints_dir: $CHECKPOINTS_DIR
export PWD=$(pwd)

# Remove the first two arguments (python binary, checkpoints_dir) to get python script
shift
shift

# Find the last checkpoint
LAST_CHECKPOINT=$(ls -v $CHECKPOINTS_DIR | tail -n 1)
echo last_checkpoint: $LAST_CHECKPOINT
# If the last checkpoint exists, pass it as a parameter
if [ -n "$LAST_CHECKPOINT" ]; then
    echo "Continuing from checkpoint $LAST_CHECKPOINT"
    srun $PYTHON_BINARY $PWD/$@ --resume_from_checkpoint $CHECKPOINTS_DIR/$LAST_CHECKPOINT --output_dir $CHECKPOINTS_DIR
else
    echo "Starting training from scratch"
    srun $PYTHON_BINARY $PWD/$@ --output_dir $CHECKPOINTS_DIR --cache_dir $CACHE_DIR
fi
