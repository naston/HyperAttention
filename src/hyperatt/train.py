from model import LlamaModel, LlamaConfig
from utils import parse_args
from transformers import Trainer, AutoTokenizer
from datasets import load_dataset
import tensorboardX
import argparse


def train_lm(config_file):
    config_file = f'./configs/{config_file}_config.json'
    print('Using Config:',config_file)
    training_args, args = parse_args(file_path=config_file)#/work/09276/naston/ls6/experts/MoE-Stability/configs

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    args.vocab_size = len(tokenizer)

    config = LlamaConfig()
    config.update_config(args)

    model = LlamaModel(config) 
    #model = MoE.from_pretrained('./MoE-Stability/models',config=config)
    print(model)

    dataset = load_dataset("<dataset>", split=f"train[:{config.num_samples}]", cache_dir= './data')

    def tokenize_function(example):
        outputs = tokenizer(example['text'], truncation=True, max_length=(args.context_length+1),
            return_overflowing_tokens=True, return_length=True,)
        input_batch = []
        labels_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == (args.context_length+1):
                input_batch.append(input_ids[:-1])
                labels_batch.append(input_ids[1:])
        return {'input_ids':input_batch, 'labels':labels_batch}
    
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    print(len(dataset))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',type=str,default='base')
    args = parser.parse_args()
    train_lm(args.cfg)