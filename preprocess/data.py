import numpy as np
import csv
import glob
import re
import pickle
import MeCab
m = MeCab.Tagger('-Ochasen')

def make_data(fname, data2):
    f = open(fname, 'r', encoding='utf-8')
    df1 = csv.reader(f)
    data1 = [v for v in df1]

    print(len(data1))
    # ファイル読み込み
    text = ''
    for i in range(0, len(data1)):
        if len(data1[i]) == 0:
            print('null')
            continue

        s = data1[i][0]
        if s[0:5] == "％ｃｏｍ：":
            continue
        if s[0] != '＠':
            # 不明文字をUNKに置き換え
            s = s.replace('＊＊＊', 'UNK')
            # 会話文セパレータ
            if s[0] == 'F' or s[0] == 'M':
                s = 'SSSS' + s[5:]
            if s[0:2] == 'Ｘ：':
                s = 'SSSS' + s[2:]

            s = re.sub('F[0-9]{3}', "UNK", s)
            s = re.sub('M[0-9]{3}', "UNK", s)
            s = s.replace("＊", "")
        else:
            continue

        while s.find("（") != -1:
            start_1 = s.find("（")
            if s.find("）") != -1:
                end_1 = s.find("）")
                if start_1 >= end_1:
                    s = s.replace(s[end_1], "")
                else:
                    s = s.replace(s[start_1:end_1 + 1], "")
                if len(s) == 0:
                    continue
            else:
                s = s[0:start_1]

        while s.find("［") != -1:
            start_2 = s.find("［")
            if s.find("］") != -1:
                end_2 = s.find("］")
                s = s.replace(s[start_2:end_2 + 1], "")
            else:
                s = s[0:start_2]

        while s.find("＜") != -1:
            start_3 = s.find("＜")
            if s.find("＞") != -1:
                end_3 = s.find("＞")
                s = s.replace(s[start_3:end_3 + 1], "")
            else:
                s = s[0:start_3]

        while s.find("【") != -1:
            start_4 = s.find("【")
            if s.find("】") != -1:
                end_4 = s.find("】")
                s = s.replace(s[start_4:end_4 + 1], "")
            else:
                s = s[0:start_4]

                # いろいろ削除したあとに文字が残っていたら出力文字列に追加
        if s != "\n" and s != "SSSS":
            text += s
    # セパレータごとにファイル書き込み
    text = text[4:]
    while text.find("SSSS") != -1:
        end_s = text.find("SSSS")
        t = text[0:end_s]
        # 長い会話文を分割
        if end_s > 100:
            while len(t) > 100:
                if t.find("。") != -1:
                    n_period = t.find("。")
                    data2.append("SSSS" + t)
                    t = t[n_period + 1:]
                else:
                    break
        data2.append("SSSS" + t)
        text = text[end_s + 4:]
    f.close()
    return


def genarate_npy(source_csv,genarated_npy) :

    df2 = csv.reader(open(source_csv, 'r', encoding='utf-8'),delimiter='\t')

    data2 = [ v for v in df2]

    mat=np.array(data2)
    print(mat)
    print(mat.shape)
    mat_corpus=[]

    #補正
    for i in range(0,mat.shape[0]):
        if mat[i][0] != '@' and mat[i][0] != 'EOS' and mat[i][0] != '┐' and mat[i][0] != '┘':
            if mat[i][0]=='SSSSUNK：' :
                mat_corpus.append('SSSS')
            elif mat[i][0]=='SSSSUNKUNK' :
                mat_corpus.append('SSSS')
            elif len(mat[i][0]) > 4 and mat[i][0][0:4]=='SSSS' :
                mat_corpus.append('SSSS')
                mat_corpus.append(mat[i][0][4:])
            elif mat[i][0]=='UNK：'or mat[i][0]=='X：':
                mat_corpus.append('SSSS')
            elif mat[i][0]=='UNKUNK' :
                mat_corpus.append('UNK')
            else :
                mat_corpus.append(mat[i][0])

    #デリミタ連続対策
    print(np.array(mat_corpus))
    mat_corpus1=[]

    for i in range(1,len(mat_corpus)) :
        if mat_corpus[i] == 'SSSS' and mat_corpus[i-1] == 'SSSS' :
            continue
        else :
            mat_corpus1.append((mat_corpus[i]))

    mat_corpus1.append('SSSS')                     #最終行にセパレータを入れる
    mat_corpus1=np.array(mat_corpus1).reshape(len(mat_corpus1),1)
    #コーパス行列セーブ
    np.save(genarated_npy, mat_corpus1)
    print('in gene :', mat_corpus1.shape)

    return


file_list = glob.glob('data/nagoya_corpus/*')
print(len(file_list))

data2=[]
for j in range(0,len(file_list)) :
    print(file_list[j])
    make_data(file_list[j],data2)

#ファイルセーブ
f = open("data/nagoya_corpus/corpus.txt", 'w', encoding='utf-8')
for i in range(0,len(data2)):
    f.write(str(m.parse(data2[i])))
f.close()
print(len(data2))

genarate_npy('data/nagoya_corpus/corpus.txt', 'data/nagoya_corpus/corpus.npy')


#１次元配列ロード
mat_corpus = np.load('data/nagoya_corpus/corpus.npy')
print(mat_corpus.shape)
data1 = [v[0] for v in mat_corpus]

mat1 = np.array(data1).reshape((len(data1),1))
mat0 = ['SSSS']                                               #先頭のデリミタ
mat0 = np.array(mat0).reshape((1,1))

mat = np.r_[mat0[:,0],mat1[:,0]]                            #各配列の先頭にデリミタがないので、

print('mat :', mat.shape)
                                                             #マージ後に改めて付与する
words = sorted(list(set(mat)))
cnt = np.zeros(len(words))

print('total words:', len(words))
word_indices = dict((w, i) for i, w in enumerate(words))    #単語をキーにインデックス検索
indices_word = dict((i, w) for i, w in enumerate(words))    #インデックスをキーに単語を検索

#単語の出現数をカウント
for j in range (0,len(mat)):
    cnt[word_indices[mat[j]]] += 1

#出現頻度の少ない単語を「UNK」で置き換え
words_unk = []                                #未知語一覧

for k in range(0,len(words)):
    if cnt[k] <= 3 :
        words_unk.append(words[k])
        words[k] = 'UNK'

print('words_unk:',len(words_unk))                   # words_unkはunkに変換された単語のリスト

#低頻度単語をUNKに置き換えたので、辞書作り直し
words = list(set(words))
words.append('\t')                                   #０パディング対策。インデックス０用キャラクタを追加
words = sorted(words)
print('new total words:', len(words))
word_indices = dict((w, i) for i, w in enumerate(words))    #単語をキーにインデックス検索
indices_word = dict((i, w) for i, w in enumerate(words))    #インデックスをキーに単語を検索

#単語インデックス配列作成
mat_urtext = np.zeros((len(mat),1),dtype=int)
for i in range(0,len(mat)):
    if mat[i] in word_indices :           #出現頻度の低い単語のインデックスをunkのそれに置き換え
        mat_urtext[i,0] = word_indices[mat[i]]
    else:
        mat_urtext[i,0] = word_indices['UNK']

print(mat_urtext.shape)

#作成した辞書をセーブ
with open('data/nagoya_corpus/word_indices.pickle', 'wb') as f :
    pickle.dump(word_indices , f)

with open('data/nagoya_corpus/indices_word.pickle', 'wb') as g :
    pickle.dump(indices_word , g)

#単語ファイルセーブ
with open('data/nagoya_corpus/words.pickle', 'wb') as h :
    pickle.dump(words , h)

#コーパスセーブ
with open('data/nagoya_corpus/mat_urtext.pickle', 'wb') as ff :
    pickle.dump(mat_urtext , ff)


import numpy.random as nr

maxlen_e = 50                             #入力語数
maxlen_d = 50                             #出力語数

#
#コーパスを会話文のリストに変換
#
separater = word_indices['SSSS']
data=[]

for i in range(0,mat_urtext.shape[0]-1) :
    if mat_urtext[i,0] == separater :
        dialog = []
    else :
        dialog.append(mat_urtext[i,0])
    if mat_urtext[i+1,0] == separater :
        data.append(dialog)

print(len(data))

#encode_input_data
enc_input = data[:-1]

#decode_input_data
dec_input = []
for i in range(1,len(data)):
    enc_dialog = data[i][:]
    enc_dialog.insert(0, separater)
    dec_input.append(enc_dialog)

#target
target = []
for i in range(1,len(data)):
    dec_dialog = data[i][:]
    dec_dialog.append(separater)
    target.append(dec_dialog)

e_input = []
d_input = []
t_l=[]
for i in range(len(enc_input)) :
    if len(enc_input[i]) <= maxlen_e and len(dec_input[i]) <= maxlen_d :
        e_input.append(enc_input[i][:])
        d_input.append(dec_input[i][:])
        t_l.append(target[i][:])
#
#0パディング
#
for i in range (0,len(e_input)):
    #リストの後ろに0追加
    e_input[i].extend([0]*maxlen_e)
    d_input[i].extend([0]*maxlen_d)
    t_l[i].extend([0]*maxlen_d)
    #系列長で切り取り
    e_input[i] = e_input[i][0:maxlen_e]
    d_input[i] = d_input[i][0:maxlen_d]
    t_l[i] = t_l[i][0:maxlen_d]

#リストから配列に変換
e = np.array(e_input).reshape(len(e_input),maxlen_e,1)
d = np.array(d_input).reshape(len(d_input),maxlen_d,1)
t = np.array(t_l).reshape(len(t_l),maxlen_d,1)

#
#シャッフル
#
z = list(zip(e, d, t))
nr.seed(12345)
nr.shuffle(z)                               #シャッフル
e,d,t=zip(*z)
nr.seed()

e = np.array(e).reshape(len(e_input), maxlen_e, 1)
d = np.array(d).reshape(len(d_input), maxlen_d, 1)
t = np.array(t).reshape(len(t_l), maxlen_d, 1)

print(e.shape,d.shape,t.shape)

#Encoder Inputデータをセーブ
with open('data/nagoya_corpus/e.pickle', 'wb') as f :
    pickle.dump(e , f)

#Decoder Inputデータをセーブ
with open('data/nagoya_corpus/d.pickle', 'wb') as g :
    pickle.dump(d , g)

#ラベルデータをセーブ
with open('data/nagoya_corpus/t.pickle', 'wb') as h :
    pickle.dump(t , h)

#maxlenセーブ
with open('data/nagoya_corpus/maxlen.pickle', 'wb') as maxlen :
    pickle.dump([maxlen_e, maxlen_d] , maxlen)