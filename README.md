# NLP-CNN-Subreddit-Sorter-Application
## Project Goals: 
    
###  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;   *Internet subforums tend to have a  lot of traffic consisting of posts that would have been better posted somewhere else. Sometimes users don’t know where their post best belongs among these similar technical subforums because of their similar nature.  I took on this project with the goal of creating an application that could sort reddit posts to their ideal subreddits, when those subreddits are similar in terms of technical content. Specifically, I used the subreddits r/Python, r/compsci, r/datascience, and r/learnmachinelearning. I wanted to attempt this both because it would be a useful application and because it would be challenge. It would be a challenge because the similarity of those subreddits would make classification difficult, and the technical nature of those subreddits would mean they have a distinct vocabulary of words likely not trained within traditional embeddings like glove. The application would be useful to both users and to subreddit moderators. It could flag to subreddit moderators that a new post doesn’t belong and is taking up traffic/page space and is off topic, without the moderators having to read so many posts. Or it could suggest to a user before actually posting where their post may be better suited to, or somewhere they may like to cross post too.  Custom word2vec embeddings and feature creation techniques (discovering frequent bigram/trigram/quadgram vector meanings before telling the model how many and what size of filter to use, rather than plugging in numbers and hoping for the best) were employed to solve the problem using a Convolutional neural  network. The application using this model was deployed to Heroku.*



    
    
## Data web scraping:
* Data was scraped at the beginning of January using praw, a reddit API scraper package. Posts from these four subreddits: (Python, datascience, learnmachinelearning, compsci) were collected over every available timeframe that reddit allows users to sort by, and by top and by controversial posts, then duplicates were removed. Posts with URLs and other characters (see scraper file) were coded not to be scraped. 100 posts from each subreddit for each timeframe and category were collected. Posts with over 200 characters were not scraped.

## Cleaning:
* Data was cleaned using spacy stopwords and tokenized using nltk word_tokenize. Common in consequential words (stopwords) were removed from sentences, and the remaining words were left as comma separated tokens.

## Feature Creation:

#### Custom word2vec embedding:
* Since these subreddits contain a vocabulary of technical nature, pretrained embeddings may not have been trained on many of the unique words found in vocab of these subreddits and may not help with text classification . I trained a custom Word2Vec embedding with reddit titles contained within the train data only.

#### "Finding" the filters:
* First I identified all frequently occurring unigrams/bigrams/trigrams within the data. Then I converted them into tensors according to the custom word2vec embedding. Once the tensor representations were obtained (the "meaning" of the grams), I found which of those tensors had a high frequency of significant cosine similarity among all other grams of the same size in the data. By identifying the number of frequently occurring gram "meanings" within the dataset, I was able to determine the optimal number of convolutional filters for the model to use.




## Model:
* A convolutional neural network with filters of size (1,2,3,4) words. Embedding dimensions were experimented with to obtain optimal results . A Single fully connected layer also gave the best results, as well as a learning rate = .001, Batch size = 5, and dropout = 0.3.

## Training:
* I evaluated the loss/accuracy of a validation set against the train set and trained past the plateau of model accuracy to obtain a model with that has the highest accuracy while still having a lower validation loss than train loss to avoid overfitting, and carry performance to the real world. I used enough epochs so that the plateau of validation loss/accuracy is clearly visible on the plots of loss/accuracy vs training epoch.


## Issues with data:
* We only have about a thousand data points to train with, This is certainly not ideal for a nueral network. Also, posts that include links without a lot of context will be difficult for this model to classify.

## Results:
* Model accuracy using custom word2vec :62%
* Model accuracy using glove : 53%



## Deployment:
* Application was created using flask and Heroku. It takes about 30 seconds to load, please be patient. The app can be found here: https://datascience-reddit-post-sorter.herokuapp.com . Please see the app.py file in the repo for the application code. 

## Instructions to run:
* Again, the app url can be found here: https://datascience-reddit-post-sorter.herokuapp.com . It takes about 30 seconds to load, please be patient.  Type the title of the reddit post you wish to make and it will suggest where it belongs among those four subreddits. All csv files and saved model outputs are included in the repo. If you wish to run this project, then after downloading simply link to the necessary paths within the project to where you keep those files. Cells should be run in order of course.


## Discussion:
* The custom embeddings made a clear difference in classification performance. Also, after playing with the number of filters in the model and viewing the resulting performance, it seems the original number of filters identified during feature creation did in fact optimize the model! With an accuracy of 62%, vs an accuracy of 25%  by guessing at random what subreddit a post should belong in, this model deployed as the application would in fact be helpful in automatically flagging posts that don’t belong and take up subreddit space (for frustrated reddit moderators), or for quickly suggesting to users where their post may be better suited or where it should be cross-posted to. Since there are only four subreddits trained here, the app is sort of a proof-of-concept. This application could be easily expanded to using multiple subreddits of similar or technical nature where the user may not be certain where best to make their reddit post.
