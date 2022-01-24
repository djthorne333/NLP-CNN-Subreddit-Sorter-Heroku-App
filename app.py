from flask import Flask, request, jsonify, render_template
#installed no cuda torch into venv
#REMOVED SPACE BETWEEN DEF FUNCTIONS BUT I THINK THEY ARE NORMALLY SUPPOSED TO BE THERE
#maybe this stuff in other file
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
import spacy
nlp = spacy.load("en_core_web_sm")
all_stopwords = nlp.Defaults.stop_words
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

embed_dim = 40
embedding = nn.Embedding(804, embed_dim)
#or 300
W2V_WINDOW = 5 #NOT BIG DIF #4 good
W2V_EPOCH = 300
W2V_MIN_COUNT = 3 #BIG DIF 3 AND 4 BEST
#venv\Scripts\activate after cd


#no good way to save and load vocab
#instead, define vocab and uembedding again, load word2vec so dont have to train!
#comsdata = pd.read_csv(r'C:\Users\Dave\Desktop\datadata\Practice\csgooo\reddit_proj\title stuff\titles_train.csv', usecols=['text', 'label'])

# train_data = comsdata.copy() #problamatic?
# print(train_data)
# train_data['text'] = comsdata['text'].str.lower().apply(word_tokenize)
#
# # now split df here cus splitting excel first resulted in nans when convert to csv
# train_data = comsdata.copy()  # problamatic?
# print(train_data)
# train_data['text'] = comsdata['text'].str.lower().apply(word_tokenize)
#
# coms = train_data['text']
# coms = coms.apply(lambda x: [item for item in x if item not in stopwords])
# sentences = coms

TEXT = data.Field(tokenize=word_tokenize, lower = True, batch_first=True, include_lengths=True, stop_words=stopwords)
LABEL = data.LabelField(dtype = torch.float, batch_first=True) #float or long i guess, which is it?
fields = [(None, None), ('text',TEXT), ('label', LABEL)] #none,none so it ignores index column of csv
#DIDNT GET TO USE LOWERCASE FOR BULD VOCAB /WPRD2VWC YET AND STOPWORDS ARE DIFFERENT


#loading custom dataset
train_data2 = data.TabularDataset(path = r'titles_train.csv', format = 'csv', fields = fields, skip_header = True)
valid_data2 = data.TabularDataset(path = r'titles_valid.csv', format = 'csv', fields = fields, skip_header = True)
test_data2 = data.TabularDataset(path = r'titles_test.csv', format = 'csv', fields = fields, skip_header = True)

TEXT.build_vocab(train_data2, min_freq=W2V_MIN_COUNT) #since word2vec tutorial built with tabdata i changed
#from sentences and labs to tabdata
LABEL.build_vocab(train_data2) #THIER LABEL AND TEXT VOCAB ARE BOTH BUILT FROM TRAIN_DATA
print(LABEL.vocab.stoi) #check the labels
#WHICH IS BULIT FROM FIELDS
w2v_model = gensim.models.word2vec.Word2Vec.load('word2vec_sentiment.model')
word2vec_vectors = []
for token, idx in np.array(list(TEXT.vocab.stoi.items())): #he used some wierd thing,
    # i turned dictionary into array
    if token in w2v_model.wv.index_to_key:
        word2vec_vectors.append(torch.FloatTensor(w2v_model.wv[token]))#cant say model[] anymore
        #"model object not subscriptbalbe", so I added .wv like when we found a vec for word
    else:
        word2vec_vectors.append(torch.zeros(embed_dim))


TEXT.vocab.set_vectors(TEXT.vocab.stoi, word2vec_vectors, embed_dim)
custom_word2vec_emb = torch.FloatTensor(TEXT.vocab.vectors)
embedding = nn.Embedding.from_pretrained(custom_word2vec_emb)

###################################################################################
class Convo(nn.Module):
    def __init__(self):
        super(Convo, self).__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), embed_dim) #need this in order to update embedding
        #can use an nn module_list so you dont have to define convolutions seperatley
        self.conv1 = nn.Conv2d(in_channels=1, kernel_size=(1, embed_dim), out_channels=260)
        self.conv2 = nn.Conv2d(in_channels=1, kernel_size=(2, embed_dim), out_channels=60)
        self.conv3 = nn.Conv2d(in_channels=1, kernel_size=(3, embed_dim), out_channels=15)
        self.conv4 = nn.Conv2d(in_channels=1, kernel_size=(4, embed_dim), out_channels=2)
        self.dropout = nn.Dropout(0.35)
        #bigram, embed dim, 25 of em
        #in_channels=1 cus we are filtereing over data that is all together (vs r,g,b matrices
        #for some image data)
        #self.pool = nn.MaxPool1d(1, 1) #looks at each feature "activation" (size1, slides by 1)
        #maxpool2d for images to reduce to important pixels
        #pool starts the first real layer, "activation becuase not associated with a nueron,
        #just gets max value of filter.
        ##########################################################################################
        self.fc1 = nn.Linear(337, len(LABEL.vocab))
        # self.fc1 = nn.Linear(50, 25)
        # self.fc2 = nn.Linear(25, len(LABEL.vocab))#number of filters, class choices,
        #culdve just wrote 2 for linear output but want to try other dataset without having to change
        # it back all the time
        ###############################################################################################
        # #also batch size is 5 (why didnt hw account for this?)

    def forward(self, x):
        print(embedding(x).size())
        x = embedding(x).unsqueeze(1) #our word2vec embedding, cant use strings tensor in convd2d
        x = F.pad(x, (0, 0, 2, 2)) #had to pad cus some sentences are under 4 words
        #so that conv filter dont fit. (2 layers above and below), 3 below also works
        print(self.conv1(x).size()) # torch.Size([5, 25, 30, 1]), so squeeze 3
        print(self.conv1(x).squeeze(3).size())
        x1 = F.relu(self.conv1(x).squeeze(3))
        x1 = F.max_pool1d(x1, x1.shape[2]).squeeze(2)#pool on x1, why does pool take
        #.shape[2] ??
        # should be 5 25?, so i thought to squeze on 2 and hw and git did it too.
        #why whould it do this? why not just have tensors maintian expected shape?
        # parameters hence the further squezing in hw3 and github reference
        print(x1.size(), "after pool, one fiolter")
        # x2 = self.pool(F.relu(self.conv2(x)))
        # x3 = self.pool(F.relu(self.conv3(x)))
        # x4 = self.pool(F.relu(self.conv4(x)))
        x2 = F.relu(self.conv2(x).squeeze(3))
        x2 = F.max_pool1d(x2, x2.shape[2]).squeeze(2)
        x3 = F.relu(self.conv3(x).squeeze(3))
        x3 = F.max_pool1d(x3, x3.shape[2]).squeeze(2)
        x4 = F.relu(self.conv4(x).squeeze(3))
        x4 = F.max_pool1d(x4, x4.shape[2]).squeeze(2)
        first_layer = self.dropout(torch.cat((x1, x2, x3, x4), dim=1) )#no dim or 0 yields [20, 25]
        # f0r 1st layer
        print(first_layer.size(), "first layer")
        # x = x.view(-1, 16 * 5 * 5)  # -> n, 400
        ######################################################################################
        xf = F.relu(self.fc1(first_layer))
        # x = F.relu(self.fc1(first_layer))
        # xf = F.relu(self.fc2(x))
        #######################################################################################
        print(xf.size())
        return xf


conv_model = Convo()
####################################################################################

###############################################################################
pretrained_embeddings = TEXT.vocab.vectors
conv_model.embedding.weight.data.copy_(pretrained_embeddings)

#zero the initial weights of the unknown and padding tokens:
UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]
conv_model.embedding.weight.data[UNK_IDX] = torch.zeros(embed_dim)

PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
conv_model.embedding.weight.data[PAD_IDX] = torch.zeros(embed_dim)
######################################################################################


app = Flask(__name__, template_folder='Templates', static_folder='static') #it was looking for folder of this name when running
# model = pickle.load(open('conv_model.pkl', 'rb'))
# model = torch.load('conv_model.pkl')
# conv_model = Convo().load_state_dict(torch.load(r'C:\Users\Dave\Desktop\datadata\Practice\csgooo\reddit_proj\title stuff\flask stuff\conv_model.pkl'))
conv_model.load_state_dict(torch.load(r'C:\Users\Dave\Desktop\datadata\Practice\csgooo\reddit_proj\title stuff\flask stuff\conv_model.pt'))
#DONT SAY CONVMOD=CONVMOD.LOAD, JUST APPLY THE LOAD METHOD


#home is bound to slash and should return the read template
@app.route('/')
def home():
    return render_template('first_page.html')  # text wasnt centered in the file caused unicode cant find error
    # return "hello"


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        inputs = request.form['message']
        # inputs = [inputs]
        # inputs = pd.Series(inputs) #made it a series cus next part
        # inputs = inputs.str.lower().apply(word_tokenize) #input is now a list not series which str works for
        # print(inputs)
        # inputs = inputs.apply(lambda x: [item for item in x if item not in stopwords])
        # inputs = inputs.astype(str)
        # # inputs = data.BucketIterator(inputs, batch_size=1)  #10 for now #batches were tensors, so we skipped that step here and had to say long()
        # # for batch_no, batch in enumerate(inputs):
        # #     inputs, batch_len = batch.text
        # inputs = (inputs, 0) #our field had set length=true so we must pass a tuple to numericalize
        # inputs = TEXT.numericalize(inputs) #will return like with batch now, have to grab with .text
        # inputs, batch_len = inputs.text
        # output = conv_model(inputs)
        # predictedLabel = inputs
        inputs = inputs.split()
        # print(inputs)
        inputs = pd.Series(inputs)
        # print(inputs, "after series")
        inputs = inputs.str.lower().apply(word_tokenize)  # input is now a list not series which str works for
        # print(inputs, "after lower")
        inputs = inputs.apply(lambda x: [item for item in x if item not in stopwords])
        # print(inputs, "after labda")
        inputs = inputs.astype(str)
        # print(inputs, "after str")

        # lower and word tokenizer converts entences into lists, soo our words looked like [word]
        # .astype I thought was universal but didnt change individuals to string, or if it did left [] with it
        # so I had to remove those characters manually.

        inputs2 = []
        for s in inputs:
            s = str.replace(s, "[", "")
            inputs2.append(s)

        # print(inputs2, "after [")

        inputs3 = []
        for s in inputs2:
            s = str.replace(s, "]", "")
            inputs3.append(s)

        # print(inputs3, "after ]")

        inputs4 = []
        for s in inputs3:
            s = str.replace(s, "'", "")
            inputs4.append(s)

        # print(inputs4, "after '")

        while ("" in inputs4):
            inputs4.remove("")

        # print(inputs4, "after remove")

        # print(inputs)
        # for t in inputs4:
        #     print(t)

        indexed = [TEXT.vocab.stoi[t] for t in inputs4]
        # print(indexed, "indexed")

        inputs = torch.LongTensor(indexed)
        # add a batch dim
        inputs = inputs.unsqueeze(0)
        # print(inputs)

        output = conv_model(inputs)
        print(output)
        predictedlabel = torch.argmax(output, dim=1)
        print(predictedlabel)
    return render_template('second_page2.html', prediction=predictedlabel)
    #needed {{}} to define variable whos value in here that we wanted to display


if __name__ == "__main__":
    app.run(debug=True)
