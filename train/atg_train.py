
import os
#os.environ["TOKENIZERS_PARALLELISM"] = "False"

from aitextgen.tokenizers import train_tokenizer
from aitextgen import aitextgen
from aitextgen.utils import build_gpt2_config

def train_atg_tokenizer():
    train_tokenizer("svg_flat.txt", vocab_size=1000)

def prepare_model():
    config = build_gpt2_config(vocab_size=1000, max_length=4096, dropout=0.1, n_embd=768, n_layer=8, n_head=12)
    ai = aitextgen(tokenizer_file="aitextgen.tokenizer.json", config=config)
    ai.save_for_upload("./trained_model")

def do_train():
    ai = aitextgen(model_folder="./trained_model", tokenizer_file="aitextgen.tokenizer.json")
    ai.train("svg_flat.txt", batch_size=1, num_steps=60000, save_every= 2500, fp16=False, generate_every=1000, learning_rate=0.001)
    ai.train("svg_flat.txt", batch_size=1, num_steps=40000, save_every= 2500, fp16=False, generate_every=1000, learning_rate=0.0001)

def do_sample():
    ai = aitextgen(model_folder="./trained_model", tokenizer_file="./trained_model/tokenizer.json", to_gpu=True)
    ai.generate(prompt="\n", max_length=4000,seed=42,do_sample=True)

    # svg_file_header = "<svg width=\"32\" height=\"32\" viewBox=\"0 0 32 32\" fill=\"none\" xmlns=\"http://www.w3.org/2000/svg\">"
    # TODO: Extract from generated output and into a seperate .svg file all sequences which starts with svg_file_header and ends with:
    #  A. </svg>
    #  B. If the sequence does not end with </svg> then find the last > in the sequence and append </svg> to it

def main():
    #train_atg_tokenizer()
    #prepare_model()
    #do_train()
    do_sample()

if __name__ == "__main__":
    main()
