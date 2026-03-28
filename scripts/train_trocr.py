from pathlib import Path
from omegaconf import OmegaConf
from datasets import load_dataset
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)
from src.ocr.trocr_dataset import make_preprocess_fn


def main():
    cfg = OmegaConf.load("config/training/trocr.yaml")

    data_path = Path(cfg.data.processed_output_path) / "train"
    dataset = load_dataset("imagefolder", data_dir=str(data_path), split="train")
    dataset = dataset.train_test_split(test_size=1 - cfg.data.train_split)

    processor = TrOCRProcessor.from_pretrained(cfg.model.name)
    model = VisionEncoderDecoderModel.from_pretrained(cfg.model.name)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size

    preprocess_train = make_preprocess_fn(processor, cfg.model.max_length, augment=True)
    preprocess_eval  = make_preprocess_fn(processor, cfg.model.max_length, augment=False)

    train_ds = dataset["train"].with_transform(preprocess_train)
    eval_ds  = dataset["test"].with_transform(preprocess_eval)

    args = Seq2SeqTrainingArguments(
        output_dir=cfg.training.output_dir,
        predict_with_generate=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.lr,
        num_train_epochs=cfg.training.epochs,
        logging_steps=10,
        load_best_model_at_end=True,
        remove_unused_columns=False,   # ← добавить
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        processing_class=processor,   # ← исправленный параметр
    )

    trainer.train()

    model.save_pretrained(cfg.training.output_dir + "/final")
    processor.save_pretrained(cfg.training.output_dir + "/final")
    print(f"Модель сохранена в {cfg.training.output_dir}/final")


if __name__ == "__main__":
    main()