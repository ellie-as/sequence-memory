import logging
from random import shuffle
from matplotlib import pyplot as plt
from simpletransformers.language_modeling import (
    LanguageModelingModel,
    LanguageModelingArgs,
)
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import math

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

class GPT:

    def __init__(self, base_model=None, base_model_name='gpt2', vocab_size=500):
        self.base_model = base_model
        self.base_model_name = base_model_name
        self.vocab_size = vocab_size

        if self.base_model is not None:
            self.tokenizer = GPT2Tokenizer.from_pretrained(base_model)
            self.model = GPT2LMHeadModel.from_pretrained(base_model)

    def train(self, segmented_sequence_list, best_model_dir, eps=1, lr=1e-5):
        text_file = open("train.txt", "w")
        shuffle(segmented_sequence_list)
        text_file.write('\n'.join(segmented_sequence_list))
        text_file.close()

        model_args = LanguageModelingArgs()
        model_args.reprocess_input_data = True
        model_args.overwrite_output_dir = True
        model_args.dataset_type = "simple"
        model_args.save_model_every_epoch = False
        model_args.evaluate_during_training = False
        model_args.mlm = False
        model_args.use_early_stopping = True
        model_args.manual_seed = 123
        model_args.learning_rate = lr
        model_args.num_train_epochs = eps
        model_args.train_batch_size = 2
        model_args.best_model_dir = best_model_dir
        # only set vocab size if training from scratch:
        if self.base_model is None:
            model_args.vocab_size = self.vocab_size

        train_file = "train.txt"
        model = LanguageModelingModel(self.base_model_name,
                                      self.base_model,
                                      # provide file to train tokenizer - not used unless self.base_model is None:
                                      train_files='train.txt',
                                      args=model_args,
                                      use_cuda=False)
        model.device = 'mps'
        # Train the model
        model.train_model(train_file)
        # Load trained model with transformers library
        self.tokenizer = GPT2Tokenizer.from_pretrained("outputs")
        self.model = GPT2LMHeadModel.from_pretrained("outputs")

    def segment(self, sequence_to_segment, rolling_mean=True, window_size=3, threshold=10,
                surprise_offset=5, plot=True, yrange=None):

        input_ids = self.tokenizer.encode(sequence_to_segment, return_tensors='pt')

        segments = []
        start_index = 0
        surprises = []
        mean_surprises = []
        subword_units = []

        def get_surprise(input_tokens, output_token):
            outputs = self.model(input_ids=input_tokens)
            predictions = outputs.logits

            softmax = torch.nn.Softmax(dim=-1)
            probabilities = softmax(predictions)

            # Get the conditional probability of the output token
            token_probability = probabilities[0, -1, output_token].item()
            token_surprise = -math.log(token_probability)
            return token_surprise

        # Calculate surprise for each subword unit
        for i in range(len(input_ids[0])):
            if i - surprise_offset < 0:
                surprise = 0
            else:
                surprise = get_surprise(input_ids[0, max(0, i-surprise_offset):i].unsqueeze(0), input_ids[0, i])
            surprises.append(surprise)
            subword_units.append(self.tokenizer.decode([input_ids[0][i].item()]))

            if rolling_mean and len(surprises) >= window_size:
                mean_surprise = np.mean(surprises[-window_size:])
                mean_surprises.append(mean_surprise)
            elif rolling_mean and len(surprises) < window_size:
                mean_surprise = surprise
                mean_surprises.append(mean_surprise)
            else:
                mean_surprise = surprise

            if mean_surprise > threshold:
                segment = self.tokenizer.decode(input_ids[0][start_index:i].tolist())
                segments.append(segment.strip())
                start_index = i

        # Add the last segment
        segment = self.tokenizer.decode(input_ids[0][start_index:].tolist())
        segments.append(segment.strip())

        if plot:
            subword_units = ['unit{}:{}'.format(ind, i) for ind, i in enumerate(subword_units)]
            plt.figure(figsize=(10, 5))
            plt.plot(subword_units[1:], surprises[1:], label='Surprise per Item')
            if rolling_mean:
                plt.plot(subword_units[1:], mean_surprises[1:], label='Rolling mean')
            plt.xticks(rotation=90)
            plt.xlabel('Items')
            plt.ylabel('Surprise')
            plt.title('Surprise per Item in the Sequence')
            if yrange is not None:
                plt.ylim(yrange[0], yrange[1])
            plt.legend()
            plt.tight_layout()
            plt.show()

        return segments

    def continue_input(self, input_sequence, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2,
                       do_sample=True, temperature=0.7):
        input_ids = self.tokenizer.encode(input_sequence, return_tensors='pt')

        # Generate text
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
            do_sample=do_sample,
            temperature=temperature,
        )

        # Decode the output
        for i in range(1):  # because num_return_sequences=1
            sequence = output[i].tolist()
            text = self.tokenizer.decode(sequence)
            print(text)