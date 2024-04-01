import argparse
import os
import pandas as pd
from bert_sklearn import load_model

def flatten(l):
    return [item for sublist in l for item in sublist]

def read_data(filename, idx=0):
    """Read file in shared task format"""

    # read file
    lines = open(filename).read().strip()

    # find sentence-like boundaries
    lines = lines.split("\n\n")

     # split on newlines
    lines = [line.split("\n") for line in lines]

    # get tokens
    tokens = [[l.split()[0] for l in line] for line in lines]

    # get labels/tags
    labels = [[l.split()[idx] for l in line] for line in lines]

    #convert to df
    data = {'tokens': tokens, 'labels': labels}
    df = pd.DataFrame(data=data)
    return df

def get_data(load_data):
    data = read_data(load_data, idx=0)
    print("Train and dev data: %d sentences, %d tokens" % (len(data), len(flatten(data.tokens))))
    print(data)
    return data.tokens, data.labels

def save(path, tokens, labels):
    save_data = {'tokens': tokens, 'labels': labels}
    df = pd.DataFrame(data=save_data)
    save_csv = path + 'out.csv'
    df.to_csv(save_csv)

def main(opt):
    tokens, labels = get_data(opt.data)
    model = load_model(opt.model)
    pred_label = model.predict(tokens)
    save(opt.outputpath, tokens, pred_label)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='DATADIR/c_test.txt', help='initial datapath path')
    parser.add_argument('--model', type=str, default='save/out.bin', help='initial model path')
    parser.add_argument('--outputpath', type=str, default='DATADIR/', help='output path')
    opt = parser.parse_args()
    if os.path.exists(opt.outputpath) == False:
        os.makedirs(opt.outputpath)
    main(opt)