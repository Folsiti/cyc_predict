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
    tokens = [[l[0] for l in line] for line in lines]

    # get labels/tags
    labels = [[l.split()[idx] for l in line] for line in lines]

    #convert to df
    data = {'tokens': tokens, 'labels': labels}
    #data = {'tokens': tokens}
    df = pd.DataFrame(data=data)
    return df

def get_data(load_data):
    data = read_data(load_data, idx=1)
    print("Prediction data: %d sentences, %d tokens" % (len(data), len(flatten(data.tokens))))
    print(data)
    return data

def save(path, tokens, labels):
    out_tokens = list()
    out_labels = list()
    for i in tokens:
        p = ""
        for s in i:
            p += s
        out_tokens.append(p)
    for i in labels:
        out_labels.append(max(i, key=i.count))
    save_data = {'tokens': out_tokens, 'labels': out_labels}
    df = pd.DataFrame(data=save_data)
    out_csv = path + 'new_out.csv'
    df.to_csv(out_csv)

def main(opt):
    data = get_data(opt.data)
    tokens = data.tokens
    model = load_model(opt.model)
    pred_label = model.predict(tokens)
    # pred = sklearn.preprocessing.MultiLabelBinarizer().fit_transform(pred_label)
    # y_test = sklearn.preprocessing.MultiLabelBinarizer().fit_transform(data.labels)

    tp=tn=fn=fp=0

    for i,j in zip(pred_label,data.labels):
        pr=max(i, key=i.count)
        tr=max(j, key=j.count)
        if pr==tr:
            if tr=='Circle':
                tp+=1
            else:
                tn+=1
        else:
            if tr=='Circle':
                fp+=1
            else:
                fn+=1

    #print(classification_report(flatten(labels), flatten(pred_label)))
    save(opt.outputpath, tokens,pred_label)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='DATADIR/test.txt', help='initial datapath path')
    parser.add_argument('--model', type=str, default='save/out.bin', help='initial model path') #cyc_out.bin
    parser.add_argument('--outputpath', type=str, default='DATADIR/', help='output path')
    opt = parser.parse_args()
    if os.path.exists(opt.outputpath) == False:
        os.makedirs(opt.outputpath)
    main(opt)