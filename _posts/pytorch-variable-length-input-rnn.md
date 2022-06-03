---
layout: post
title: Variable length input for RNN models in PyTorch
tags: [personal]
comments: true
---




<!-- **Here is some bold text** -->
It's common to deal with sequences of variable lengths in NLP. This article provides a simple yet effective way of dealing with variable length sequences when training an LSTM network on mini-batches. For transformers the procedure will be slightly different yet the principle is the same. If you are to treat one invidual sequence at each time then you will not need to go though all the trouble, simply pass one sequence at a time(and perhaps do gradient accumulation if you want to 'mock' mini-batch training to make training more stable). But remember training with mini-batches can significantly improve training speed and this is especially true for current popular benchmark datasets with more than tens of thousands sequences.

## Create a custom `Dataset` class

First define a Dataset class that hosts custom data. It needs to have `__len__` and `__getitem__` functions. In particular, `__getitem__` returns the indexed element in the dataset. For this particular example, the training data consists of a list of verb and nouns, and the target labels are also a list of verbs and nouns. This poses the problem as a sequence to sequence (seq2seq) problem with varibale length inputs and labels.

<!-- | data | input | label |
| :------ |:--- | :--- |
| 1 | cut tomoto, cut tomato, add tomato, ... | cut tomoto, cut tomato, cut tomato,... |
| 2 | cut cucumber, cut onion, ... | cut tomato, cut onion, ... |
| 3 | grate cheese, grate cheese, peel onion,... | cut cheese, cut cheese, peel onion,... |
| ... | ... | ... |

<!-- ```
data 1: cut tomoto - > cut cucumber
data 2: peel onion - > 
``` -->
```python
# training_data is a list of tuples in the form of ((verb_input, verb_target),(noun_input, noun_target))
class CustomTextDataset(Dataset):
    def __init__(self, training_data):
        self.verb_input = [x[0][0] for x in training_data]
        self.verb_target = [x[0][1] for x in training_data]
        self.noun_input = [x[1][0] for x in training_data]
        self.noun_target = [x[1][1] for x in training_data]
    def __len__(self):
            return len(self.verb_input)
    def __getitem__(self, idx):
        return (self.verb_input[idx], self.noun_input[idx], self.verb_target[idx], self.noun_target[idx])

custom_train_dataset = CustomTextDataset(training_data)
```

## Initialize `DataLoader`
DataLoader takes the custom dataset as input, and for variable length input, a collate function is also needed for the `collate_fn` argument when initializing. For variable length input sequences, we need to pad them so that all the sequences have the same length as the longest sequence. This can be done with [`pad_sequence`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html) function. By default the padded locations are zeros. If the target labels also have variable length (e.g. a seq2seq model), we also need to pad the targets with zeros. And during training, we need to define the ignore_index argument of the model to ignore zeros in the labels when backpropagating. A different but also valid way to do this is to use [`pack_padded_sequence`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html) function to both `pad` and `pack` the inputs at the same time. The result will be a [`PackedSequence`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.PackedSequence.html#torch.nn.utils.rnn.PackedSequence) object which can be directly feed into RNN types of networks in PyTorch. However for my case I am using [`Embedding`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) layers before LSTM, which does not take `PackedSequence` objects as input.

```python
def my_collate(batch):
    v_in = [item[0] for item in batch]
    n_in = [item[1] for item in batch]
    v_tar = [item[2] for item in batch]
    n_tar = [item[3] for item in batch]

    v_in = pad_sequence([torch.tensor(x, dtype=torch.long, device=device) for x in v_in],batch_first=True)
    n_in = pad_sequence([torch.tensor(x, dtype=torch.long, device=device) for x in n_in],batch_first=True)
    v_tar = pad_sequence([torch.tensor(x, dtype=torch.long, device=device) for x in v_tar],batch_first=True)
    n_tar = pad_sequence([torch.tensor(x, dtype=torch.long, device=device) for x in n_tar],batch_first=True)

    return [v_in, n_in, v_tar, n_tar]

data_loader_train = DataLoader(custom_train_dataset, batch_size=32, shuffle=False, collate_fn=my_collate)
```
## Training
For this task I used a LSTM with two separate `Embedding` layers to encode verbs and nouns, and two linear layers for classification. Both embedding layers used pretrained FastText embeddings to initialize with `from_pretrained` function.

```python
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, tagset_verb_size, tagset_noun_size):#, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings_verb = nn.Embedding.from_pretrained(torch.tensor(embedding_verb_matrix), freeze=False)
        self.word_embeddings_noun = nn.Embedding.from_pretrained(torch.tensor(embedding_noun_matrix), freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=0.4, batch_first=True, num_layers=1)
        self.hidden2tag_verb = nn.Linear(hidden_dim, tagset_verb_size)
        self.hidden2tag_noun = nn.Linear(hidden_dim, tagset_noun_size)

    def forward(self, sentence_verb, sentence_noun):
        embeds_verb = self.word_embeddings_verb(sentence_verb)
        embeds_noun = self.word_embeddings_noun(sentence_noun)
        embeds_verb = embeds_verb.float()
        embeds_noun = embeds_noun.float()
        embeds = torch.cat((embeds_verb, embeds_noun),2)
        lstm_out, _ = self.lstm(embeds)
        tag_space_verb = self.hidden2tag_verb(lstm_out)
        tag_space_noun = self.hidden2tag_noun(lstm_out)
        tag_scores_verb = F.log_softmax(tag_space_verb, dim=2)
        tag_scores_noun = F.log_softmax(tag_space_noun, dim=2)
        tag_scores_verb = torch.transpose(tag_scores_verb,1,2)
        tag_scores_noun = torch.transpose(tag_scores_noun,1,2)
        return tag_scores_verb, tag_scores_noun

EMBEDDING_DIM = 100
HIDDEN_DIM = 500
BATCH_SIZE = 32
N_LAYERS = 1

model = LSTMTagger(2*EMBEDDING_DIM, HIDDEN_DIM, N_VERB_CLASSES, N_NOUN_CLASSES)
model.to(device)
loss_function = nn.NLLLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int((len(training_data)/BATCH_SIZE)*N_EPOCH))

for (idx, batch) in enumerate(data_loader_train):
      sentence_verb, sentence_noun, target_verb, target_noun = batch
      tag_scores_verb, tag_scores_noun = model(sentence_verb, sentence_noun)

      loss_verb = loss_function(tag_scores_verb, target_verb)
      loss_noun = loss_function(tag_scores_noun, target_noun)
      loss = loss_verb + loss_noun
      loss.backward()

      optimizer.step()
      scheduler.step()
      optimizer.zero_grad()
``` -->

