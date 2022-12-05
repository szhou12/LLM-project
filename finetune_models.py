import numpy as np
import datasets
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
import json
import sys

# key = model_name
# value = (tokenizer_name, list of saved models)
CHECKPOINTS = {
    'gpt2': ('gpt2', ["gpt2", "./models/gpt2_0.5/", "./models/gpt2_0.9/"]),
    'distilgpt2': ('distilgpt2', ["distilgpt2", "./models/distilgpt2_0.5/", "./models/distilgpt2_0.9/"]),
    'gpt-neo': ('EleutherAI/gpt-neo-1.3B', ["EleutherAI/gpt-neo-1.3B", "./models/gpt-neo-1.3B_0.5/", "./models/gpt-neo-1.3B_0.9/"]),
}

SPARSE_PERCENT = [0, 50, 90]
context_length = 128


def fine_tune(model_name):
    tokenizer_checkpoint, model_checkpoint_list = CHECKPOINTS[model_name]

    raw_datasets = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    ## Tokenization
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_checkpoint)
    def tokenize_function(samples):
        outputs = tokenizer(
            samples["text"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        tokenize_function, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    ## Feed into Trainer
    for idx, model_checkpoint in enumerate(model_checkpoint_list):
        config = AutoConfig.from_pretrained(
            model_checkpoint,
            vocab_size=len(tokenizer),
            n_ctx=context_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = AutoModelForCausalLM.from_config(config)

        training_args = TrainingArguments(
            output_dir=f'finetune/{model_name}_{SPARSE_PERCENT[idx]}/',
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            # num_train_epochs = 1,
        )

        trainer = Trainer(
            model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator
        )

        trainer.train()
        # trainer.save(f'finetune/{model_name}_{SPARSE_PERCENT[idx]}')



if __name__ == "__main__":
    model_name = sys.argv[1].strip()

    # tokenizer_checkpoint = sys.argv[2].strip()
    # model_checkpoint = sys.argv[3].strip()
    # SPARSE_PERCENT = sys.argv[4].strip()

    # run_benchmark_single(
    #     model_name, 
    #     tokenizer_checkpoint, 
    #     model_checkpoint, 
    #     SPARSE_PERCENT
    # )


    fine_tune(model_name)