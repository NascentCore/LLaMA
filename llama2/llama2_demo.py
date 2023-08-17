import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer,  LlamaConfig

model_path="./Llama2-Chinese-7b-Chat/"
config = LlamaConfig.from_pretrained("./Llama2-Chinese-7b-Chat/config.json")
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM(config)
print(model)

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
train_file = "wikitext-103-raw/wiki.train.raw"
eval_file = "wikitext-103-raw/wiki.valid.raw"
max_seq_length = 512
out_model_path = "llama2_output"
train_epoches = 10
batch_size = 1

tokenizer.pad_token = tokenizer.eos_token

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=max_seq_length,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) #这里bert不一样
eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=eval_file,
    block_size=max_seq_length,
)


deepspeed_config="./config/dp_zero3_config.json"
training_args = TrainingArguments(
        output_dir=out_model_path,
        overwrite_output_dir=True,
        num_train_epochs=train_epoches,
        per_device_train_batch_size=batch_size,
        save_steps=2000,
        save_total_limit=2,
        prediction_loss_only=True,
        report_to="none",
        deepspeed =deepspeed_config
    )

#model = nn.DataParallel(model)
#model = model.cuda()
#print(model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()
