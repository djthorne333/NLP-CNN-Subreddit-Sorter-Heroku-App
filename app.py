from flask import Flask, request, jsonify, render_template
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

embed_dim = 500
embedding = nn.Embedding(804, embed_dim)
#or 300
W2V_WINDOW = 5 #NOT BIG DIF #4 good
W2V_EPOCH = 300
W2V_MIN_COUNT = 3 #BIG DIF 3 AND 4 BEST

#creating the vocab within the app from the saved word2vec model, since saving and uploading the vocab and embedding() eluded me


TEXT = data.Field(tokenize=word_tokenize, lower = True, batch_first=True, include_lengths=True, stop_words=stopwords)
LABEL = data.LabelField(dtype = torch.float, batch_first=True) 
fields = [(None, None), ('text',TEXT), ('label', LABEL)] #none,none so it ignores index column of csv



#loading custom dataset
train_data2 = data.TabularDataset(path = r'titles_train.csv', format = 'csv', fields = fields, skip_header = True)
valid_data2 = data.TabularDataset(path = r'titles_valid.csv', format = 'csv', fields = fields, skip_header = True)
test_data2 = data.TabularDataset(path = r'titles_test.csv', format = 'csv', fields = fields, skip_header = True)

TEXT.build_vocab(train_data2, min_freq=W2V_MIN_COUNT) 
LABEL.build_vocab(train_data2) 
w2v_model = gensim.models.word2vec.Word2Vec.load('word2vec_reddit.model')
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
#single fully connected layer working best 
class Convo(nn.Module):
    def __init__(self):
        super(Convo, self).__init__()
        self.embedding = nn.Embedding(len(TEXT.vocab), embed_dim) #embed_dim is straight from word2vec
        self.conv1 = nn.Conv2d(in_channels=1, kernel_size=(1, embed_dim), out_channels=300) #here we input # of filters we found
        self.conv2 = nn.Conv2d(in_channels=1, kernel_size=(2, embed_dim), out_channels=187)
        self.conv3 = nn.Conv2d(in_channels=1, kernel_size=(3, embed_dim), out_channels=25)
        self.conv4 = nn.Conv2d(in_channels=1, kernel_size=(4, embed_dim), out_channels=1)
        self.dropout = nn.Dropout(0.35)
        #in_channels=1 cus we are filtereing over data that is all together (vs r,g,b matrices for some image data)
        #self.pool = nn.MaxPool1d(1, 1) #looks at each feature "activation" (size1, slides by 1)
        #maxpool2d for images to reduce to important pixels
        #pool starts the first real layer, just gets max value of filter.
        ##########################################################################################
        self.fc1 = nn.Linear(513, len(LABEL.vocab))
        # self.fc1 = nn.Linear(50, 25) #(for trying out 2 fully connected layers)
        # self.fc2 = nn.Linear(25, len(LABEL.vocab))#number of filters, class choices,
        ###############################################################################################
    def forward(self, x):
        #print(embedding(x).size())
        x = embedding(x).unsqueeze(1) #our word2vec embedding
        x = F.pad(x, (0, 0, 2, 2)) #had to pad cus some sentences are under 4 words so that conv filter doesnt fit. 
        #print(self.conv1(x).size()) # torch.Size([5, 25, 30, 1]), so squeeze 3
        #print(self.conv1(x).squeeze(3).size())
        x1 = F.relu(self.conv1(x).squeeze(3))
        x1 = F.max_pool1d(x1, x1.shape[2]).squeeze(2)#pool on x1
        #print(x1.size(), "after pool, one filter")
        x2 = F.relu(self.conv2(x).squeeze(3))
        x2 = F.max_pool1d(x2, x2.shape[2]).squeeze(2)
        x3 = F.relu(self.conv3(x).squeeze(3))
        x3 = F.max_pool1d(x3, x3.shape[2]).squeeze(2)
        x4 = F.relu(self.conv4(x).squeeze(3))
        x4 = F.max_pool1d(x4, x4.shape[2]).squeeze(2)
        first_layer = self.dropout(torch.cat((x1, x2, x3, x4), dim=1))
        #print(first_layer.size(), "first layer")
        ######################################################################################
        xf = F.relu(self.fc1(first_layer))
        # x = F.relu(self.fc1(first_layer)) #(for trying out 2 fully connected layers)
        # xf = F.relu(self.fc2(x))
        #######################################################################################
        #print(xf.size())
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


template_dir = r'templates'
app = Flask(__name__, template_folder=template_dir)
conv_model.load_state_dict(torch.load(r'conv_model.pt'))

            
#home bound
@app.route('/')
def home():
    return render_template('first_page.html')  
  


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        inputs = request.form['message']
        inputs = inputs.split()
        inputs = pd.Series(inputs)
        inputs = inputs.str.lower().apply(word_tokenize)  # input is now a list not series which str works for
        inputs = inputs.apply(lambda x: [item for item in x if item not in stopwords])
        inputs = inputs.astype(str)
        
        # lower and word tokenizer converts sentences into lists of tokens, so our words looked like [word]
        # so I chose to remove those characters manually for some reason which is uh, well it works

        inputs2 = []
        for s in inputs:
            s = str.replace(s, "[", "")
            inputs2.append(s)

        inputs3 = []
        for s in inputs2:
            s = str.replace(s, "]", "")
            inputs3.append(s)

        inputs4 = []
        for s in inputs3:
            s = str.replace(s, "'", "")
            inputs4.append(s)

        while ("" in inputs4):
            inputs4.remove("")

            
        indexed = [TEXT.vocab.stoi[t] for t in inputs4]
        

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
