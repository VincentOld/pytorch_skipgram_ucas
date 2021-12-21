import logging
import numpy as np
import torch
from torch import optim
from loss import negative_sampling_loss, nce_loss
from model import SkipGram
from utils.negative_sampler import NegativeSampler
from utils.vocab import Corpus

NEGATIVE_TABLE_SIZE = int(1e8)
ws = 2
num_negatives = 5
epochs = 20
num_minibatches = 512
starting_lr = 0.001 * num_minibatches
lr_update_rate = 1000
embedding_dim = 100
seed = 7
#固定初始随机赋值，确保每次运行.py文件时，生成的随机数都是固定的
rnd = np.random.RandomState(seed)
torch.manual_seed(seed)

#更新学习率
def update_lr(starting_lr, num_processed_words, epochs, num_words):
    new_lr = starting_lr * (1. - num_processed_words / (epochs * num_words + 1))
    lower_lr = starting_lr * 0.0001
    return max(new_lr, lower_lr)

#word_id
def generate_words_from_doc(doc, num_processed_words, corpus, rnd):
    new_doc = []
    for word_id in doc:
        num_processed_words += 1
        if corpus.discard(word_id=word_id, rnd=rnd):
            continue
        new_doc.append(word_id)
        if len(new_doc) >= 1000:
            yield np.array(new_doc), num_processed_words
            new_doc = []
    yield np.array(new_doc), num_processed_words

def train_on_minibatches(model, optimizer, inputs, contexts, negatives, is_neg_loss, log_k_prob=None):
    num_minibatches = len(contexts)
    #inputs填充成一列数据
    inputs = torch.LongTensor(inputs).view(num_minibatches, 1)
    optimizer.zero_grad()

    if is_neg_loss:
        contexts = torch.LongTensor(contexts).view(num_minibatches, 1)
        negatives = torch.LongTensor(negatives)
        pos, neg = model.forward(inputs, contexts, negatives)
        loss = negative_sampling_loss(pos, neg)
    else:
        pos_log_k_negative_prob = torch.FloatTensor(log_k_prob[contexts]).view(num_minibatches, 1)
        neg_log_k_negative_prob = torch.FloatTensor(log_k_prob[negatives])
        contexts = torch.LongTensor(contexts).view(num_minibatches, 1)
        negatives = torch.LongTensor(negatives)
        pos, neg = model.forward(inputs, contexts, negatives)
        loss = nce_loss(pos, neg, pos_log_k_negative_prob, neg_log_k_negative_prob)
    loss.backward()
    optimizer.step()
    return loss.item()

def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.terminator = ''
    logger.addHandler(stream_handler)
    logger.info('Loading training corpus...\n')
    corpus = Corpus(min_count=5)
    docs = corpus.tokenize_from_file("../data/zh.txt")
    corpus.build_discard_table(t=1e-3)
    logger.info('V:{}, #words:{}\n'.format(corpus.num_vocab, corpus.num_words))
    is_neg_loss = True
    negative_sampler = NegativeSampler(
        frequency=corpus.dictionary.id2freq,
        negative_alpha=0.75,
        is_neg_loss=is_neg_loss,
        table_length=NEGATIVE_TABLE_SIZE
    )
    if is_neg_loss:
        log_k_prob = None
        logger.info('loss function: Negative Sampling\n')
    else:
        log_k_prob = np.log(num_negatives * negative_sampler.noise_dist)
        logger.info('loss function: NCE\n')

    model = SkipGram(V=corpus.num_vocab, embedding_dim=embedding_dim)
    optimizer = optim.SGD(model.parameters(), lr=starting_lr)
    model.train()
    num_processed_words = last_check = 0
    num_words = corpus.num_words
    loss_value = 0
    num_add_loss_value = 0
    for epoch in range(epochs):
        inputs = []
        contexts = []
        for sentence in docs:
            for doc, num_processed_words in generate_words_from_doc(
                    doc=sentence, num_processed_words=num_processed_words, corpus=corpus, rnd=rnd
            ):
                doclen = len(doc)
                dynamic_window_sizes = rnd.randint(low=1, high=ws + 1, size=doclen)
                for (position, (word_id, dynamic_window_size)) in enumerate(zip(doc, dynamic_window_sizes)):
                    begin_pos = max(0, position - dynamic_window_size)
                    end_pos = min(position + dynamic_window_size, doclen - 1) + 1
                    for context_position in range(begin_pos, end_pos):
                        if context_position == position:
                            continue
                        contexts.append(doc[context_position])
                        inputs.append(word_id)
                        if len(inputs) >= num_minibatches:
                            negatives = negative_sampler.sample(k=num_negatives, rnd=rnd, exclude_words=contexts)
                            loss_value += train_on_minibatches(
                                model=model,
                                optimizer=optimizer,
                                inputs=inputs,
                                contexts=contexts,
                                negatives=negatives,
                                is_neg_loss=is_neg_loss,
                                log_k_prob=log_k_prob
                            )
                            num_add_loss_value += 1
                            inputs.clear()
                            contexts.clear()
                if len(inputs) > 0:
                    negatives = negative_sampler.sample(k=num_negatives, rnd=rnd, exclude_words=contexts)
                    loss_value += train_on_minibatches(
                        model=model,
                        optimizer=optimizer,
                        inputs=inputs,
                        contexts=contexts,
                        negatives=negatives,
                        is_neg_loss=is_neg_loss,
                        log_k_prob=log_k_prob
                    )
                    num_add_loss_value += 1
                    inputs.clear()
                    contexts.clear()

                # update lr and print progress
                if num_processed_words - last_check > lr_update_rate:
                    optimizer.param_groups[0]['lr'] = lr = update_lr(starting_lr,
                                                                     num_processed_words,
                                                                     epochs,
                                                                     num_words)

                    logger.info('\rprogress: {0:.7f}, lr={1:.7f}, loss={2:.7f}'.format(
                        num_processed_words / (num_words * epochs),
                        lr, loss_value / num_add_loss_value),
                    )
                    last_check = num_processed_words

    with open("zh.vec", 'w') as f:
        f.write('{} {}\n'.format(corpus.num_vocab, embedding_dim))
        embeddings = model.in_embeddings.weight.data.numpy()
        for word_id, vec in enumerate(embeddings):
            word = corpus.dictionary.id2word[word_id]
            vec = ' '.join(list(map(str, vec)))
            f.write('{} {}\n'.format(word, vec))

if __name__ == '__main__':
    main()
