# Load balancer for SLURM in the PennNLP Cluster

<a name="files-section"></a>
## Files

* `train_with_last_checkpoint.sh`: Runs a slurm job that automatically detects the last checkpoint of a model and passes it as a parameter `--resume_from_checkpoint` to your python script (See Section [Python Script Requirements](#script-reqs-section")). This has been tested with models using the [`transformers.Trainer` API](https://huggingface.co/docs/transformers/main_classes/trainer) but it should be applicable with minor modifications to other training libraries (See Section [Modifications for Other Training Libraries](#modifications-section)).<br>
You can run the script as follows:
```
sbatch <SBATCH_OPTIONS> train_with_last_checkpoint.sh $1 $2 $3
```
For the `<SBATCH_OPTIONS>` (run `sbatch --help` for documentation). The arguments to `train_with_last_checkpoint.sh` are specified as follows:
    - `$1`: python path. You can find which python is used by your envrironment by running `which python` on the terminal. 
    - `$2`: your checkpoint directory. 
    - `$3`: your python script with all arguments other than the `--resume_from_checkpoint` and `--output_dir` arguments since they are handled separately. `--output_dir` is set by argument `$4` and checkpoint is automatically found to be the latest checkpoint in `--output_dir`.

Here is a concrete example on how to run this to train a small T5 model on SQuAD. You can use an existing conda environment if you already have `pytorch`, `transformers`, and `datasets` installed. Otherwise first run the installation:
```
>> conda create -n test_me python=3.7
>> conda activate test_me
>> conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
>> conda install -c huggingface transformers
>> python -m pip install -r simple_train_example/requirements.txt
```

Then you can run the following to get the dummy example to run. 
```
>> export PYTHON_DIR=$(which python) && echo $PYTHON_DIR
/your/python/path/bin/python
>> export PWD=$(pwd) && echo $PWD
/path/to/auto_last_ckpt
>> mkdir $PWD/test_slurm_logs/
>> sbatch -J test_auto_ckpt -c 1 --mem 2400 -o $PWD/test_slurm_logs/out.txt -e $PWD/test_slurm_logs/err.txt -G 1 -t 10 train_with_last_checkpoint.sh $PYTHON_DIR simple_train_example/checkpoints simple_train_example/train.py --tiny --cache_dir simple_train_example/cache 
```
* `continuous_deployment.sh`: wrapper script that creates a new job every time a job ends. Uses `squeue` to monitor the status of submitted jobs. 
You can run the script as follows
```
>> chmod +x continuous_deployment.sh
>> ./continuous_deployment.sh <MAX_ITERATIONS> "<BATCH_OPTIONS>" $PYTHON_DIR <checkpoint_dir>  <python_file> <python_args>> <log_file> 2>&1 & 
```
For example
```
>> conda activate test_me
>> export PYTHON_DIR=$(which python) && echo $PYTHON_DIR
>> export PWD=$(pwd) && echo $PWD
>> mkdir $PWD/test_slurm_logs/
>> chmod +x continuous_deployment.sh
>> ./continuous_deployment.sh 10 "-J test_auto_ckpt -c 1 --mem 64 -o $PWD/test_slurm_logs/out.txt -e $PWD/test_slurm_logs/err.txt -G 1 -t 10" $PYTHON_DIR simple_train_example/checkpoints simple_train_example/train.py --tiny --cache_dir simple_train_example/cache   > continuous_deployment_logs.txt 2>&1 & 
``` 

### Cleanup test
To clean up after testing make sure to run:
```
>> conda remove -n test_me --all
>> rm -r /nlp/data/$USER_NAME/test_slurm_logs
>> rm -r simple_train_example/checkpoints
>> rm -r simple_train_example/cache

```
You could also consider 
<a name="script-reqs-section"></a>
## Python Script Requirements

The python script to run should contain an argument `--resume_from_checkpoint` that stores the path to the checkpoint. This can be achieved using the following code in the argument parser
```
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        const=None,
        default=None,
        help="Resume training from a given checkpoint.",
    )
    return parser.parse_args()
args = parse_args()
```
Then the trainer should employ this checkpoint by passing it to its `TrainingArguments`
```
 training_args = TrainingArguments(
        ...
        output_dir='./checkpoints',
        resume_from_checkpoint=args.resume_from_checkpoint,
    )

 trainer = Trainer(
        ....
        args=training_args,
    )

    trainer.train()
```


<a name="tips-section"></a>
## Tips and Tricks

* Cache your dataloaders: preprocessing can take time, and we do not want to waste this time on the clusters. A way to do that is using the `datasets.load_from_disk` and `Dataset.save_to_disk` [methods](https://huggingface.co/docs/datasets/loading) in transformers. The `simple_train_example/train.py` includes an example of how you can do that. 
* To increase fairness, try to checkpoint often and set low times for sbatch jobs (~2-5hrs seems reasonable depending on model size). HOWEVER, try not to store too many checkpoints - `Trainer` handles deletion of old checkpoints if you set `save_total_limit` in `TrainingArgs`. Checkpoints fill up storage very fast. 

<a name="modifications-section"></a>
## Modifications for Other Training Libraries

As long as:
* you specify the correct checkpoint directory,
* checkpoints are saved in increasing alphabetical order based on step number, and 
* have the requirements specified in Section [Python Script Requirements](#script-reqs-section)
I do not see a reason why this would not work of the shelf. Please post issues on github and I will get to them as soon as I can :) The README will be updated as issues come through with the corresponding solutions. 

