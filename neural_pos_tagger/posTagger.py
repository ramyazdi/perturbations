from __future__ import print_function

__author__ = 'max'
"""
Implementation of Bi-directional LSTM-CNNs model for POS tagging.
"""
import sys

sys.path.append(".")
sys.path.append("..")
import csv
import time
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, Adamax
from neuro_nlp.io_ import get_logger, conllx_data
from neuro_nlp.models.sequence_labeling import BiRecurrentConv, BiVarRecurrentConv
from neuro_nlp import utils
from neuro_nlp.nn import Embedding
from neuro_nlp.models.perturbated_pos_tagger import PerturbatedPosTagger

K_PERTURBATE = 100


def main():


    def generate_optimizer(opt, lr, params):
        params = filter(lambda param: param.requires_grad, params)
        if opt == 'adam':
            return Adam(params, lr=lr, betas=betas, weight_decay=gamma)
        elif opt == 'sgd':
            return SGD(params, lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
        elif opt == 'adamax':
            return Adamax(params, lr=lr, betas=betas, weight_decay=gamma)
        else:
            raise ValueError('Unknown optimization algorithm: %s' % opt)

    def save_checkpoint(network, optimizer, opt, model_name):
        print('Saving model to: %s' % model_name)
        state = {'model_state_dict': network.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'opt': opt}
        torch.save(state, model_name)

    def load_checkpoint(network, optimizer, load_path):
        print('Loading saved model from: %s' % load_path)
        checkpoint = torch.load(load_path)
        if checkpoint['opt'] != opt:
            raise ValueError('loaded optimizer type is: %s instead of: %s' % (checkpoint['opt'], opt))
        else:
            network.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return network, optimizer

    def build_network_and_optimizer():
        network = BiRecurrentConv(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), num_filters, window,
                                  mode, hidden_size, num_layers, num_labels,
                                  tag_space=tag_space, embedd_word=word_table, p_in=p_in, p_out=p_out, p_rnn=p_rnn,
                                  initializer=initializer, use_char=use_char)

        optim = generate_optimizer(opt, learning_rate, network.parameters())

        if load_path:
            english_embed_table_size = 20882
            english_num_labels = 20

            network = BiRecurrentConv(embedd_dim, english_embed_table_size, char_dim, char_alphabet.size(), num_filters,
                                      window, mode, hidden_size, num_layers, num_labels=english_num_labels,
                                      tag_space=tag_space, p_in=p_in, p_out=p_out, p_rnn=p_rnn,
                                      initializer=initializer, use_char=use_char)

            network, optim = load_checkpoint(network, optim, load_path)
            english_oov_vec = network.word_embedd.weight[0]
            word_table[0,:].data.copy_(english_oov_vec)

            network.word_embedd = Embedding(word_table.size()[0], embedd_dim, init_embedding=word_table,freeze=True)

        device = torch.device('cuda' if use_gpu else 'cpu')
        network.to(device)
        return network, optim

    parser = argparse.ArgumentParser(description='Tuning with bi-directional RNN-CNN')
    parser.add_argument('--mode', choices=['RNN', 'LSTM', 'GRU'], help='architecture of rnn', required=True)
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden units in RNN')
    parser.add_argument('--opt', choices=['adam', 'sgd', 'adamax'], help='optimization algorithm',default='sgd')
    parser.add_argument('--tag_space', type=int, default=0, help='Dimension of tag space')
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
    parser.add_argument('--num_filters', type=int, default=30, help='Number of filters in CNN')
    parser.add_argument('--char_dim', type=int, default=30, help='Dimension of Character embeddings')
    parser.add_argument('--use_char', action='store_true', help='whether to use Character embeddings or not')
    parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='Decay rate of learning rate')
    parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
    parser.add_argument('--dropout', choices=['std', 'variational'], help='type of dropout', required=True)
    parser.add_argument('--p_rnn', nargs=2, type=float, required=True, help='dropout rate for RNN')
    parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
    parser.add_argument('--p_out', type=float, default=0.33, help='dropout rate for output layer')
    parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
    parser.add_argument('--unk_replace', type=float, default=0., help='The rate to replace a singleton word with UNK')
    parser.add_argument('--embedding', choices=['glove', 'senna', 'sskip', 'polyglot','fasttext'], help='Embedding for words', required=True)
    parser.add_argument('--embedding_dict', help='path for embedding dict')
    parser.add_argument('--train')  # "data/POS-penn/wsj/split1/wsj1.train.original"
    parser.add_argument('--dev')  # "data/POS-penn/wsj/split1/wsj1.dev.original"
    parser.add_argument('--test')  # "data/POS-penn/wsj/split1/wsj1.test.original"
    parser.add_argument('--model_path', help='path for saving model file.')
    parser.add_argument('--load_path', help='path for loading saved model file, indicates that only inference is necessary', default=None)
    parser.add_argument('--alignment_file_name',help='name of alignment matrix file (inside alignment_matrices folder)',default=None)
    parser.add_argument('--model_name', help='name of the trained model to be saved', default='')
    parser.add_argument('--inference_mode', choices=['perturbated', 'regular','both','baseline_k_list','regular_predict_file'])

    args = parser.parse_args()

    logger = get_logger("POSTagger")

    mode = args.mode
    train_path = args.train
    dev_path = args.dev
    inference_mode = args.inference_mode
    test_path_list = [t_path for t_path in args.test.split(",")]
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    hidden_size = args.hidden_size
    num_filters = args.num_filters
    learning_rate = args.learning_rate
    momentum = 0.9
    decay_rate = args.decay_rate
    gamma = args.gamma
    schedule = args.schedule
    p_rnn = tuple(args.p_rnn)
    p_in = args.p_in
    p_out = args.p_out
    unk_replace = args.unk_replace
    use_char = args.use_char
    model_path = args.model_path
    model_name = args.model_name
    load_path = args.load_path
    alignment_file_name = args.alignment_file_name
    embedding = args.embedding
    embedding_path = args.embedding_dict
    opt = args.opt
    embedd_dict, embedd_dim = utils.load_embedding_dict(embedding, embedding_path)
    momentum = 0.9
    betas = (0.9, 0.9)

    if (load_path): # only inference should be done
        embedd_dict.apply_transform(alignment_file_name)

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(model_path, 'alphabets','pos/')

    word_alphabet, char_alphabet, pos_alphabet,_,\
    arc_alphabet = conllx_data.create_alphabets(alphabet_path, train_path, data_paths=[dev_path].extend(test_path_list),
                                                 max_vocabulary_size=70000, embedd_dict=embedd_dict)

    logger.info("Word Alphabet Size: %d" % word_alphabet.size())
    logger.info("Character Alphabet Size: %d" % char_alphabet.size())
    logger.info("POS Alphabet Size: %d" % pos_alphabet.size())

    if (load_path):
        print("loading English POS mapping")
        pos_alphabet.load(input_directory=os.path.join(os.path.dirname(load_path),'alphabets','pos'),name ='pos')
        logger.info("POS Alphabet after English update Size: %d" % pos_alphabet.size())
        pos_alphabet.save(output_directory=alphabet_path,name='pos')


    logger.info("Reading Data")
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')

    data_train = conllx_data.read_data_to_variable(train_path, word_alphabet, char_alphabet, pos_alphabet,pos_alphabet,
                                                   arc_alphabet, use_gpu=use_gpu)
    # data_train = conllx_data.read_data(train_path, word_alphabet, char_alphabet, pos_alphabet, arc_alphabet)
    # num_data = sum([len(bucket) for bucket in data_train])
    num_data = sum(data_train[1])
    num_labels = pos_alphabet.size()

    data_dev = conllx_data.read_data_to_variable(dev_path, word_alphabet, char_alphabet, pos_alphabet,pos_alphabet, arc_alphabet,
                                                 use_gpu=use_gpu)
    data_test = conllx_data.read_data_to_variable(test_path_list[0], word_alphabet, char_alphabet, pos_alphabet,pos_alphabet, arc_alphabet,
                                                  use_gpu=use_gpu)

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / embedd_dim)
        table = np.empty([word_alphabet.size(), embedd_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in embedd_dict:
                embedding = embedd_dict[word]
            elif word.lower() in embedd_dict:
                embedding = embedd_dict[word.lower()]
            elif word.replace("_","").strip() in embedd_dict:
                embedding = embedd_dict[word.replace("_","").strip()]
            else:
                embedding = np.random.uniform(-scale, scale, [1, embedd_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('oov: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()
    logger.info("constructing network...")

    char_dim = args.char_dim
    window = 3
    num_layers = args.num_layers
    tag_space = args.tag_space
    initializer = nn.init.xavier_uniform_
    if args.dropout == 'std':
        network = BiRecurrentConv(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), num_filters, window, mode, hidden_size, num_layers, num_labels,
                                  tag_space=tag_space, embedd_word=word_table,  p_in=p_in, p_out=p_out, p_rnn=p_rnn, initializer=initializer,use_char=use_char)
    else:
        network = BiVarRecurrentConv(embedd_dim, word_alphabet.size(), char_dim, char_alphabet.size(), num_filters, window, mode, hidden_size, num_layers, num_labels,
                                     tag_space=tag_space, embedd_word=word_table, p_in=p_in, p_out=p_out, p_rnn=p_rnn, initializer=initializer,use_char=use_char)
    if use_gpu:
        network.cuda()

    lr = learning_rate
    # optim = Adam(network.parameters(), lr=lr, betas=(0.9, 0.9), weight_decay=gamma)
    optim = generate_optimizer(opt,lr,network.parameters())
    logger.info("Network: %s, num_layer=%d, hidden=%d, filter=%d, tag_space=%d" % (mode, num_layers, hidden_size, num_filters, tag_space))
    logger.info("training: l2: %f, (#training data: %d, batch: %d, unk replace: %.2f)" % (gamma, num_data, batch_size, unk_replace))
    logger.info("dropout(in, out, rnn): (%.2f, %.2f, %s)" % (p_in, p_out, p_rnn))

    num_batches = num_data / batch_size + 1
    dev_correct = 0.0
    best_epoch = 0
    test_correct = 0.0
    test_total = 0
    if (not load_path):
        for epoch in range(1, num_epochs + 1):
            print('Epoch %d (%s(%s), learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (epoch, mode, args.dropout, lr, decay_rate, schedule))
            train_err = 0.
            train_corr = 0.
            train_total = 0.

            start_time = time.time()
            num_back = 0
            network.train()
            for batch in range(1, num_batches + 1):
                word, char, labels, _, _,_, masks, lengths = conllx_data.get_batch_variable(data_train, batch_size, unk_replace=unk_replace)

                optim.zero_grad()
                loss, corr, _ = network.loss(word, char, labels, mask=masks, length=lengths, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                loss.backward()
                optim.step()

                num_tokens = masks.data.sum()

                train_err += loss.item() * num_tokens #loss.data[0] * num_tokens
                train_corr += corr.sum().item() #corr.data[0]
                train_total += num_tokens

                time_ave = (time.time() - start_time) / batch
                time_left = (num_batches - batch) * time_ave

                # update log
                if batch % 100 == 0:
                    # sys.stdout.write("\b" * num_back)
                    # sys.stdout.write(" " * num_back)
                    # sys.stdout.write("\b" * num_back)
                    log_info = 'train: %d/%d loss: %.4f, acc: %.2f%%, time left (estimated): %.2fs' % (batch, num_batches, train_err / train_total, train_corr * 100 / train_total, time_left)
                    print("\n")
                    sys.stdout.write(log_info)
                    sys.stdout.flush()
                    num_back = len(log_info)

            # sys.stdout.write("\b" * num_back)
            # sys.stdout.write(" " * num_back)
            # sys.stdout.write("\b" * num_back)
            print('train: %d loss: %.4f, acc: %.2f%%, time: %.2fs' % (num_batches, train_err / train_total, train_corr * 100 / train_total, time.time() - start_time))

            # evaluate performance on dev data
            network.eval()
            dev_corr = 0.0
            dev_total = 0
            for batch in conllx_data.iterate_batch_variable(data_dev, batch_size):
                word, char, labels, _, _,_, masks, lengths = batch
                _, corr, preds = network.loss(word, char,labels, mask=masks, length=lengths, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                num_tokens = masks.data.sum()
                dev_corr += corr.sum().item()
                dev_total += num_tokens
            print('dev corr: %d, total: %d, acc: %.2f%%' % (dev_corr, dev_total, dev_corr * 100 / dev_total))

            if dev_correct < dev_corr:
                dev_correct = dev_corr
                best_epoch = epoch
                save_checkpoint(network,optim,opt,os.path.join(model_path,model_name))
                # evaluate on test data when better performance detected
                test_corr = 0.0
                test_total = 0
                for batch in conllx_data.iterate_batch_variable(data_test, batch_size):
                    word, char, labels, _, _, _, masks, lengths = batch
                    _, corr, preds = network.loss(word, char, labels, mask=masks, length=lengths, leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
                    num_tokens = masks.data.sum()
                    test_corr += corr.sum().item()
                    test_total += num_tokens
                test_correct = test_corr
            print("best dev  corr: %d, total: %d, acc: %.2f%% (epoch: %d)" % (dev_correct, dev_total, dev_correct * 100 / dev_total, best_epoch))
            print("best test corr: %d, total: %d, acc: %.2f%% (epoch: %d)" % (test_correct, test_total, test_correct * 100 / test_total, best_epoch))

            if epoch % schedule == 0:
                lr = learning_rate / (1.0 + epoch * decay_rate)
                optim = generate_optimizer(opt,lr,network.parameters())

    else: # only inference
        network, optimizer = build_network_and_optimizer()
        network.eval()
        torch.manual_seed(999)
        perturbated_network = PerturbatedPosTagger(network, alpha=0)

        for alpha in [0.002]:
            print("Current alpha: ",alpha)
            if (inference_mode in ['perturbated', 'both']):
                perturbated_network.alpha = nn.Parameter(torch.Tensor([alpha]))
                perturbated_network.to(device)

            for test_path_t in test_path_list:
                data_test = conllx_data.read_data_to_variable(test_path_t, word_alphabet, char_alphabet, pos_alphabet,
                                                              pos_alphabet, arc_alphabet,
                                                              use_gpu=use_gpu)

                if (inference_mode in ['regular','both','regular_predict_file']):

                    test_corr = 0.0
                    test_total = 0

                    if (inference_mode =='regular_predict_file'):
                        batch_size = 1
                        outfile_predicted_pos = open(test_path_t+'_predicted_pos', 'w')

                    for batch in conllx_data.iterate_batch_variable(data_test, batch_size):
                        word, char, labels, _, heads,_, masks, lengths = batch
                        _, corr, preds = network.loss(word, char, labels, mask=masks, length=lengths,
                                                      leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)

                        # print("word: ",word)
                        # print("preds: ",preds)
                        # print("labels: ", labels)
                        num_tokens = masks.data.sum()
                        test_corr += corr.sum().item()
                        test_total += num_tokens

                        if (inference_mode == 'regular_predict_file'):
                            sent_len = lengths.max()
                            sent_heads = heads[0,:sent_len].cpu().numpy()
                            sent_predicted_pos = [pos_alphabet.get_instance(token_pos_pred) for token_pos_pred in preds[0, :sent_len].cpu().numpy()]
                            sentence = [[index,'_','_',head_pos_zipped[1],head_pos_zipped[1],'_',head_pos_zipped[0],'_']
                                        for index,head_pos_zipped in  enumerate(zip(sent_heads,sent_predicted_pos),1) ]


                            for line in sentence:
                                outfile_predicted_pos.write('\t'.join(str(i) for i in line))
                                outfile_predicted_pos.write('\n')
                            outfile_predicted_pos.write('\n')

                    test_correct = test_corr

                    with open(test_path_t + '_pos_tagging_without_perturbating.csv', 'wb') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')
                        writer.writerow(['POS acc. not perturbated',(test_correct * 100.0 / test_total).data.item()])


                if (inference_mode in ['perturbated', 'both']):

                    test_corr = 0.0
                    test_total = 0
                    corr_worst_total, corr_25_total, corr_50_total, corr_75_total, corr_best_total = 0, 0, 0, 0, 0
                    corr_majority_total = 0

                    for batch in conllx_data.iterate_batch_variable(data_test, batch_size):

                        word, char, labels, _, _,_, masks, lengths = batch

                        correct_per_batch = []
                        preds_per_batch = []
                        for k in range(K_PERTURBATE):
                            ## corr is a tensor [batch]
                            loss, corr, preds = perturbated_network.loss(word, char, labels, mask=masks, length=lengths,
                                                          leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)

                            # if (k%50 ==0):
                            #     print('$%$'*20,k,'$%$'*20)
                            #     print("preds: ", preds)
                            #     print("labels: ", labels)
                            #     print ('correct: ',corr)

                            correct_per_batch.append(corr)
                            preds_per_batch.append(preds)
                            num_tokens = masks.data.sum(dim=1)
                            test_corr += corr.sum().item()



                        test_total += num_tokens.sum()

                        max_len = lengths.max()
                        labels = labels[:, :max_len]

                        preds_per_batch = torch.stack(preds_per_batch)

                        preds_per_batch = preds_per_batch.permute(1,0,2)

                        preds_majority = BiRecurrentConv.calc_majority_vote_per_batch(preds_per_batch)

                        corr_majority_total += (torch.eq(preds_majority.cpu(), labels.cpu()).type_as(masks[:, :max_len].cpu()) * masks[:, :max_len].cpu()).sum()


                        # correct_per_batch: [k_repetition,batch_size,seq_length]
                        correct_per_batch_numpy = torch.stack(correct_per_batch).cpu().numpy()
                        #print ("correct_per_batch_numpy: ",correct_per_batch_numpy.shape)
                        corr_worst,corr_25,corr_50,corr_75,corr_best = np.percentile(a=correct_per_batch_numpy, q=[0,25,50,75,100], axis=0, interpolation='higher')

                        corr_worst_total+= corr_worst.sum()
                        corr_25_total+= corr_25.sum()
                        corr_50_total+= corr_50.sum()
                        corr_75_total+= corr_75.sum()
                        corr_best_total+= corr_best.sum()

                    print("test file: ",test_path_t)
                    print('POS acc. perturbated worst',(corr_worst_total * 100.0 / test_total).data.item())
                    print('POS acc. perturbated best', (corr_best_total * 100.0 / test_total).data.item())
                    print('POS acc. perturbated Majority',(corr_majority_total.cpu() * 100.0 / test_total.cpu()).data.item())

                    with open(test_path_t + '_pos_tagging_perturbating_percentiles.csv', 'wb') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')
                        test_total = test_total.cpu()
                        writer.writerow(['POS acc. perturbated worst',(corr_worst_total * 100.0 / test_total).data.item()])
                        writer.writerow(['POS acc. perturbated percentile 25', (corr_25_total * 100.0 / test_total).data.item()])
                        writer.writerow(['POS acc. perturbated percentile 50', (corr_50_total * 100.0 / test_total).data.item()])
                        writer.writerow(['POS acc. perturbated percentile 75', (corr_75_total * 100.0 / test_total).data.item()])
                        writer.writerow(['POS acc. perturbated best', (corr_best_total * 100.0 / test_total).data.item()])
                        writer.writerow(['POS acc. perturbated Majority vote', (corr_majority_total.cpu() * 100.0 / test_total).data.item()])


                if (inference_mode =='baseline_k_list'):

                    test_total = 0
                    corr_worst_total, corr_25_total, corr_50_total, corr_75_total, corr_best_total = 0, 0, 0, 0, 0
                    corr_majority_total = 0

                    for batch in conllx_data.iterate_batch_variable(data_test, batch_size):

                        word, char, labels, _, _, _, masks, lengths = batch

                        correct_per_batch,preds_per_batch = network.loss(word, char, labels, mask=masks, length=lengths,
                                                                         leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS,find_baseline_list=True)
                        num_tokens = masks.data.sum()

                        test_total += num_tokens
                        correct_per_batch_numpy = correct_per_batch.cpu().numpy()

                        # print ("correct_per_batch_numpy: ",correct_per_batch_numpy.shape)
                        corr_worst, corr_25, corr_50, corr_75, corr_best = np.percentile(a=correct_per_batch_numpy,
                                                                                         q=[0, 25, 50, 75, 100], axis=1,
                                                                                         interpolation='higher')
                        corr_worst_total += corr_worst.sum()
                        corr_25_total += corr_25.sum()
                        corr_50_total += corr_50.sum()
                        corr_75_total += corr_75.sum()
                        corr_best_total += corr_best.sum()

                        max_len = lengths.max()
                        labels = labels[:, :max_len]

                        preds_majority = BiRecurrentConv.calc_majority_vote_per_batch(preds_per_batch)

                        corr_majority_total += (torch.eq(preds_majority.cpu(), labels.cpu()).type_as(masks[:, :max_len].cpu()) * masks[:, :max_len].cpu()).sum()

                    print("test file: ", test_path_t)
                    print ("test_total: ",test_total)

                    print('POS acc. Kbest baseline worst', (corr_worst_total * 100.0 / test_total).data.item())
                    print('POS acc. Kbest baseline best', (corr_best_total * 100.0 / test_total).data.item())
                    print('POS acc. Kbest baseline Majority', (corr_majority_total.cpu() * 100.0 / test_total.cpu()).data.item())

                    with open(test_path_t + '_pos_tagging_baseline_k_best_percentiles.csv', 'wb') as csvfile:
                        writer = csv.writer(csvfile, delimiter=',')
                        test_total = test_total.cpu()
                        writer.writerow(['POS acc. kbest baseline worst', (corr_worst_total * 100.0 / test_total).data.item()])
                        writer.writerow(
                            ['POS acc. kbest baseline 25', (corr_25_total * 100.0 / test_total).data.item()])
                        writer.writerow(
                            ['POS acc. kbest baseline 50', (corr_50_total * 100.0 / test_total).data.item()])
                        writer.writerow(
                            ['POS acc. kbest percentile 75', (corr_75_total * 100.0 / test_total).data.item()])
                        writer.writerow(['POS acc. kbest baseline best', (corr_best_total * 100.0 / test_total).data.item()])

                        writer.writerow(['POS acc. kbest baseline Majority vote', (corr_majority_total.cpu() * 100.0 / test_total).data.item()])


if __name__ == '__main__':
    main()
