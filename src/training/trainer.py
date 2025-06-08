from transformers import TrainingArguments, Trainer

def get_training_args(
    output_dir="./llama_finetuned",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    eval_steps=50,
    save_steps=50
):
    """Configure training arguments."""
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=logging_dir,
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        load_best_model_at_end=True,
        fp16=True,
    )

def setup_trainer(model, training_args, train_dataset, eval_dataset):
    """Initialize the Trainer with the model and datasets."""
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

def train_and_save(trainer, model, tokenizer, output_dir="./llama_finetuned/final"):
    """Run training and save the model."""
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir) 