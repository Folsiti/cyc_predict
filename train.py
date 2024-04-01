import os
import csv
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
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

    # get tokens
    tokens = [[l.split()[0] for l in line] for line in lines]

    # get labels/tags
    labels = [[l.split()[idx] for l in line] for line in lines]

    #convert to df
    data= {'tokens': tokens, 'labels': labels}
    df=pd.DataFrame(data=data)
    return df

def get_data(opt):
    trainfile = opt.train_data
    id = opt.labels_idx
    train = read_txt(trainfile, idx=id)
    print("Train and dev data: %d sentences, %d tokens" % (len(train), len(flatten(train.tokens))))
    #print(train)

    testfile = opt.test_data
    test = read_txt(testfile, idx=id)
    print("Test data: %d sentences, %d tokens"%(len(test),len(flatten(test.tokens))))
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


    model = BertTokenClassifier('bert-base-cased',
                            max_seq_length=178,
                            epochs=4,
                            gradient_accumulation_steps=4,
                            learning_rate=3e-5,
                            train_batch_size=20,
                            eval_batch_size=20,
                            validation_fraction=0,
                            label_list=label_list,
                            use_cuda=True)

    print(model)
    # finetune model on train data

    model.fit(X_train, y_train)


    # score model on test data
    # f1_test = model.score(X_test, y_test,'macro')
    # print("Test f1: %0.02f"%(f1_test))

    # get predictions on test data
    y_preds = model.predict(X_test)

    save_data = {'tokens': X_test, 'labels': y_test, 'preds': y_preds}
    dfs = pd.DataFrame(data=save_data)
    pred_save = opt.save_path + 'out.csv'
    dfs.to_csv(pred_save)

    # print report on classifier stats
    print(classification_report(flatten(y_test), flatten(y_preds)))
    model.save(savepath+'out.bin')
    print("Successfully saved in "+savepath)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='DATADIR/x_train.txt', help='initial train database path')
    parser.add_argument('--test_data', type=str, default='DATADIR/x_test.txt', help='initial test database path')
    parser.add_argument('--labels_idx', type=str, default=1, help='initial labels idx')
    parser.add_argument('--save_path', type=str, default='save/', help='save model path')
    opt = parser.parse_args()
    if os.path.exists(opt.save_path) == False:
        os.makedirs(opt.save_path)
    main(opt)