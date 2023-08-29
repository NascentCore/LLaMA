import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaTokenizer,  LlamaConfig
import argparse


parse=argparse.ArgumentParser()

parse.add_argument("--model_path",type=str, default="./Llama2-Chinese-7b-Chat/", help="model path")
parse.add_argument("--config_path",type=str, default="./Llama2-Chinese-7b-Chat/config.json", help="model config path")
parse.add_argument("--ds_config",type=str, default="./config/dp_zero3_config.json", help="deepspeed config json")

args = parser.parse_args()


config = LlamaConfig.from_pretrained(args.config_path)
tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
model = LlamaForCausalLM(config)
print(model)

from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import LineByLineTextDataset
max_seq_length = 512
out_model_path = "llama2_output"
train_epoches = 10
batch_size = 1

tokenizer.pad_token = tokenizer.eos_token


#train_file = "wikitext-103-raw/wiki.train.raw"
#eval_file = "wikitext-103-raw/wiki.valid.raw"
#train_dataset = LineByLineTextDataset(tokenizer=tokenizer,file_path=train_file,block_size=max_seq_length)
#eval_dataset = LineByLineTextDataset(tokenizer=tokenizer,file_path=eval_file,block_size=max_seq_length)

train_dataset = torch.load('data/train_dataset.pt')
eval_dataset = torch.load('data/eval_dataset.pt')

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) #这里bert不一样

training_args = TrainingArguments(
        output_dir=out_model_path,
        overwrite_output_dir=True,
        num_train_epochs=train_epoches,
        per_device_train_batch_size=batch_size,
        save_steps=2000,
        save_total_limit=2,
        prediction_loss_only=True,
        report_to="none",
        deepspeed =args.ds_config
    )

#model = nn.DataParallel(model)
#model = model.cuda()
#print(model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()
