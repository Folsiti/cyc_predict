import os
import csv
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

sys.path.append("../")
from bert_sklearn import BertTokenClassifier

def read_tsv(filename, quotechar=None):
    with open(filename, "r", encoding='utf-8') as f:
        return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

def flatten(l):
    return [item for sublist in l for item in sublist]

def read_txt(filename, idx=1):
    # read file
    lines =  open(filename).read().strip()

    # find sentence-like boundaries
    lines = lines.split("\n\n")

     # split on newlines
    lines = [line.split("\n") for line in lines]

    tokens = [[l.split()[0] for l in line] for line in lines]

    # get labels/tags
    labels = [[l.split()[idx] for l in line] for line in lines]

    #convert to df
    data= {'tokens': tokens, 'labels': labels}
    df=pd.DataFrame(data=data)
    return df

def get_data(opt):
    '''
    trainfile = DATADIR + "train.txt",
    devfile = DATADIR + "dev.txt",
    testfile = DATADIR + "test.txt"
    '''
    trainfile = opt.train_data
    id = opt.labels_idx
    train = read_txt(trainfile, idx=id)
    print("Train data: %d sentences, %d tokens" % (len(train), len(flatten(train.tokens))))
    #print(train)

    testfile = opt.test_data
    test = read_txt(testfile, idx=id)
    print("Test data: %d sentences, %d tokens"%(len(test),len(flatten(test.tokens))))
    #train = pd.concat([train, test])
    #print(test)

    #return train, test
    return train,test

def main(opt):
    savepath = opt.save_path
    train,test = get_data(opt)
    #train = read_cvs()

    X_train, y_train = train.tokens, train.labels
    X_test, y_test = test.tokens, test.labels

    label_list = np.unique(flatten(y_train))
    label_list = list(label_list)
    #print("\nNER tags:",label_list)

    train.head()

    # bert-base-uncased
    model = BertTokenClassifier(#'bert-base-cased',
                            max_seq_length=178,
                            epochs=16,
                            gradient_accumulation_steps=1,
                            learning_rate=5e-5,
                            train_batch_size=16,
                            eval_batch_size=16,
                            validation_fraction=0.15,
                            label_list=label_list,
                            ignore_label=['non_Circle'],
                            use_cuda=True)

    print(model)
    # finetune model on train data
    #print(savepath + str(3) + '_cyc_out.bin')
    model.fit(X_train, y_train)

    # score model on test data

    # f1_test = model.score(X_test, y_test,'macro')
    # print("Test f1: %0.02f"%(f1_test))

    # get predictions on test data
    y_preds = model.predict(X_test)

    #save test result
    out_tokens = list()
    out_pred_labels = list()
    out_test_labels = list()
    for i in X_test:
        p = ""
        for s in i:
            p += s
        out_tokens.append(p)
    for i in y_preds:
        out_pred_labels.append(max(i, key=i.count))
    for i in y_test:
        out_test_labels.append(max(i, key=i.count))
    save_data = {'tokens': out_tokens, 'labels': out_test_labels,'preds': out_pred_labels}
    dfs = pd.DataFrame(data=save_data)
    pred_save = opt.save_path + 'tset_out.csv'
    dfs.to_csv(pred_save)

    # print report on classifier stats
    print(classification_report(flatten(y_test), flatten(y_preds)))
    path = savepath+'non_pred_out.bin'
    model.save(path)
    print("Successfully saved in "+path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='DATADIR/train.txt', help='initial train database path')
    parser.add_argument('--test_data', type=str, default='DATADIR/test.txt', help='initial test database path')
    parser.add_argument('--labels_idx', type=str, default=1, help='initial labels idx')
    parser.add_argument('--save_path', type=str, default='save/', help='save model path')
    opt = parser.parse_args()
    if os.path.exists(opt.save_path) == False:
        os.makedirs(opt.outputpath)
    main(opt)