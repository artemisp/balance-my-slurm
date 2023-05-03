import argparse
import os
from datasets import load_dataset, load_from_disk
from transformers import T5ForConditionalGeneration, T5TokenizerFast, Seq2SeqTrainingArguments, Seq2SeqTrainer

os.environ["WANDB_RESUME"] = 'allow'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        const=None,
        default=None,
        help="Resume training from a given checkpoint.",
        nargs='?'
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default= os.path.join(os.getcwd(), "checkpoints"),
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use a tiny subset of the dataset (200 examples) for training and validation.",
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default= os.path.join(os.getcwd(), "cache"),
        help="Transformers cache dir",
    )
    
    return parser.parse_args()


def load_squad_data(tokenizer, tiny=False, cache_dir=None):
    dataset = load_dataset("squad", cache_dir=cache_dir)
    if tiny:
        dataset["train"] = dataset["train"].select(range(200))
        dataset["validation"] =  dataset["train"] # overfit on train
    
    
    def tokenize_function(examples):
        model_inputs = tokenizer([f"context: {c} question: {q} answer: " for c,q in zip(examples['context'],examples['question'])], max_length=512, padding="max_length", truncation=True)

        # just keeping the first answer for each
        answers = [answer['text'][0] for answer in examples['answers']]
        labels = tokenizer(answers, max_length=32, padding="max_length", truncation=True).input_ids
        
        ## don't forget to ignore padding! ;) 
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)
  
        model_inputs["labels"] = labels_with_ignore_index
        return model_inputs
    
    if os.path.exists(os.path.join(cache_dir, "dataset.pt")):
        dataset = load_from_disk(os.path.join(cache_dir, "dataset.pt"))
    else:
        dataset = dataset.map(
            tokenize_function,
            batched=True,
        )
        dataset.save_to_disk(os.path.join(cache_dir, "dataset.pt"))
    return dataset

def main():
    args = parse_args()

    model = T5ForConditionalGeneration.from_pretrained("t5-small",cache_dir=args.cache_dir)
    tokenizer = T5TokenizerFast.from_pretrained("t5-small", use_fast=True, cache_dir=args.cache_dir)

    dataset = load_squad_data(tokenizer, args.tiny, cache_dir=args.cache_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        run_name='test-auto-ckpt',
        overwrite_output_dir=False,
        num_train_epochs=20,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=5,
        save_total_limit=1,
        save_strategy="steps",
        save_steps=5,
        learning_rate=3e-4,
        # resume_from_checkpoint=args.resume_from_checkpoint,
        report_to="wandb"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
    )
    
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        
    
if __name__ == "__main__":
    main()
