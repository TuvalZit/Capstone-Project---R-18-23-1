###############################################################################
# Imports #####################################################################
###############################################################################
import pandas as pd
import numpy as np
import wandb

from datetime import datetime
from transformers import AutoTokenizer, AutoModel, T5EncoderModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from ast import literal_eval
from tqdm import tqdm

from data import build_dataset
from model import (
                   ContrastiveLSTMHead,
                   )


###############################################################################
# Runtime parameters ##########################################################
###############################################################################
arg_parser = ArgumentParser(description='Run an experiment.')
arg_parser.add_argument('--books', action='store_true', help='Use books dataset')
arg_parser.add_argument('--shake', action='store_true', help='Use shakespeare dataset')
arg_parser.add_argument('--imposters', action='store_true', help='Use imposters dataset')

arg_parser.add_argument('--model', type=str, required=True, help='Model type',
                        choices=['max', 'mean', 'lstm'],
                        )
arg_parser.add_argument('--scheduler', type=str, default='none', help='Model type',
                        choices=['enable', 'none'],
                        )
arg_parser.add_argument('--transformer', type=str, default='roberta-large', help='Model type',
                        choices=['roberta-large', 'roberta-base', 'distilroberta-base', 'google/t5-v1_1-base'],
                        )
arg_parser.add_argument('--batch_size', type=int, default=0, help='Batch size')
arg_parser.add_argument('--vbatch_size', type=int, default=0, help='Validation batch size')
arg_parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate')

args = arg_parser.parse_args()

BATCH_SIZE = args.batch_size
VALID_BATCH_SIZE = args.vbatch_size
ENABLE_SCHEDULER = args.scheduler == 'enable'
DEVICES = 0
MODEL_TYPE = args.model
BASE_CODE = args.transformer
if MODEL_TYPE ==  'lstm':
    MODEL = ContrastiveLSTMHead



TRAIN_FILES = {'books': '/content/drive/MyDrive/Databases/book_train.csv',
                'shake':'/content/drive/MyDrive/Databases/shakespeare_train.csv',
                'imposters':'/content/drive/MyDrive/Databases/imposters_train.csv'
                }
TEST_FILES = {'books': '/content/drive/MyDrive/Databases/book_test.csv',
              'shake':'/content/drive/MyDrive/Databases/shakespeare_test.csv',
              'imposters':'/content/drive/MyDrive/Databases/imposters_test.csv'
                }
USED_FILES = []
if args.books:
    USED_FILES.append('books')
if args.shake:
    USED_FILES.append('shake')
if args.imposters:
    USED_FILES.append('imposters')

MINIBATCH_SIZE = 64
VALID_STEPS = 50
CHUNK_SIZE = 512
LEARNING_RATE = 5e-3
DROPOUT = .1
WEIGHT_DECAY = .01
LABEL_SMOOTHING = .0
TRAINING_STEPS = 3000
WARMUP_STEPS = 0 #1000 #int(TRAINING_STEPS*.1)

###############################################################################
# Main method #################################################################
###############################################################################

def main():
    # Load preferred datasets
    train_datasets, test_datasets = [], []
    tqdm.pandas()
    for file_code in USED_FILES:
        print(f'Loading {file_code} dataset...')
        train_file = pd.read_csv(TRAIN_FILES[file_code])
        test_file = pd.read_csv(TEST_FILES[file_code])

        train_file['unique_id'] = train_file.index.astype(str) + f'_{file_code}'
        test_file['unique_id'] = test_file.index.astype(str) + f'_{file_code}'

        train_datasets.append(train_file[['unique_id', 'id', 'pretokenized_text', 'decoded_text']])
        test_datasets.append(test_file[['unique_id', 'id', 'pretokenized_text', 'decoded_text']])
    
    train = pd.concat(train_datasets).sample(frac=1.)
    test = pd.concat(test_datasets)

    del train_datasets
    del test_datasets

    # Build dataset
    n_auth = len(train.id.unique()) if BATCH_SIZE == 0 else BATCH_SIZE
    n_auth_v = len(test.id.unique()) if VALID_BATCH_SIZE == 0 else VALID_BATCH_SIZE

    # get closest power of 2 to n_auth
    n_auth = int(2 ** np.floor(np.log(n_auth)/np.log(2)))
    n_auth_v = int(2 ** np.floor(np.log(n_auth_v)/np.log(2)))

    print(f'Batch size equals: {n_auth}')
    train_data = build_dataset(train,
                               steps=TRAINING_STEPS*n_auth,
                               batch_size=n_auth,
                               num_workers=8, 
                               prefetch_factor=8,
                               max_len=CHUNK_SIZE,
                               tokenizer = AutoTokenizer.from_pretrained(BASE_CODE),
                               mode='text')
    test_data = build_dataset(test, 
                              steps=VALID_STEPS*n_auth_v, 
                              batch_size=n_auth_v, 
                              num_workers=8, 
                              prefetch_factor=8, 
                              max_len=CHUNK_SIZE,
                              tokenizer = AutoTokenizer.from_pretrained(BASE_CODE),
                              mode='text')

    # Name model
    model_datasets = '+'.join(USED_FILES)
    date_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_name = f'final_{date_time}_{MODEL_TYPE}_{model_datasets}'
    print(f'Saving model to {save_name}')

    # Callbacks
    wandb_key=""#change the key, signup for wandb and get the key and place it here
    wandb.login(key = wandb_key)
    wandb_logger = WandbLogger(name=save_name, project="author_profiling_extension")
    checkpoint_callback = ModelCheckpoint('model',
                                          filename=save_name,
                                          monitor=None,
                                          every_n_epochs=1,
                                          )
    lr_monitor = LearningRateMonitor('step')

    # Define training arguments
    trainer = Trainer(
                    max_steps=TRAINING_STEPS,
                    accelerator='gpu',
                    log_every_n_steps=500,
                    logger=wandb_logger,
                    #strategy='dp',
                    precision=16,
                    val_check_interval=100,
                    callbacks=[checkpoint_callback, lr_monitor],
                    )

    # Define model
    if ('T0' in BASE_CODE) or ('t5-v1_1' in BASE_CODE):
        base_transformer = T5EncoderModel.from_pretrained(BASE_CODE)

    else:
        base_transformer = AutoModel.from_pretrained(BASE_CODE, 
                                                     hidden_dropout_prob = DROPOUT, 
                                                     attention_probs_dropout_prob = DROPOUT)
    train_model = MODEL(base_transformer,
                        learning_rate=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        num_warmup_steps=WARMUP_STEPS,
                        num_training_steps=TRAINING_STEPS,
                        enable_scheduler=ENABLE_SCHEDULER,
                        minibatch_size=MINIBATCH_SIZE,)

    # Fit and log
    trainer.fit(train_model, train_data, test_data)
    wandb.finish()

if __name__ == '__main__':
    main()
