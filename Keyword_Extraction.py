#runs on Satish's port on workbook. port :8011

import pandas as pd
import numpy as np
import platform_toolkit as ptk
import cyavro
import avro.schema
from avro.datafile import DataFileWriter
from avro.io import DatumWriter
import os as os
import mpy.utils.avrotools as av
import itertools
from collections import Counter
import re
import platform_toolkit as ptk




import nltk
from bs4 import BeautifulSoup
import gensim
import networkx


##need to execute the nltk.download() commands once for each time setting up the environment
##nltk.download('punkt')
##nltk.download('wordnet')
##nltk.download('maxent_treebank_pos_tagger')
##nltk.download("stopwords")
##nltk.download('averaged_perceptron_tagger')


import string
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
punct = set(string.punctuation)
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

platform = ptk.init_platform("ls_get_tfidf_data")
mp = platform.iSparkPlatform




class WORD:
    def __init__(self, text, tag):
        self.text = text
        self.tag = tag
    
    def lemmatization(self):
        if self.tag[0] == 'N':
            self.text = wordnet_lemmatizer.lemmatize(self.text)
        elif self.tag[0] == 'V':
            self.text = wordnet_lemmatizer.lemmatize(self.text, pos=wordnet.VERB)
        elif self.tag[0] == 'J':
            self.text = wordnet_lemmatizer.lemmatize(self.text, pos=wordnet.ADJ)
        elif self.tag[0] == 'R':
            self.text = wordnet_lemmatizer.lemmatize(self.text, pos=wordnet.ADV)
        return self




class sentence:
    def __init__(self, word_list):
        self.wordlist = [WORD(i[0], i[1]) for i in word_list]
        
    def filter_punct(self):
        self.wordlist = [word for word in self.wordlist if not word.text in punct]
        return self
    
    ##Filter by the stopword list
    def filter_stop(self):
        self.wordlist = [word for word in self.wordlist if not word.text in stop_words]
        return self
    
    ## Filter by part-of-speech
    def pos_filter(self):
        self.wordlist = [word for word in self.wordlist if word.tag[0] in set(['N','V','J'])]
        return self
        
    ##Lemmatization
    def lemmatize_word(self):
        self.wordlist = [word.lemmatization() for word in self.wordlist]
        return self
        
    ##generate a list of filtered and cleaned up words from the object
    def get_bow(self):
        return [word.text for word in self.wordlist if len(word.text)>1]     




def get_webText(raw):    
    text = BeautifulSoup(raw, 'lxml')
    title = [i.getText() for i in text.findAll('title')]
    body = [j.getText(separator=u' ') for j in text.findAll(['p','h1','h2','h3','h4','h5','h6']) ]
    return title+body




def get_phrase(tagged_sent, grammar):
    chunker = nltk.chunk.regexp.RegexpParser(grammar)
    all_chunks = nltk.chunk.tree2conlltags(chunker.parse(tagged_sent))
    candidates = [[word for word, pos, chunk in group]
                  for key, group in itertools.groupby(all_chunks, lambda (word,pos,chunk): chunk != 'O') if key]
    candidates = [set(cand) for cand in candidates if all(word not in punct or word=='-' for word in cand)]
    return candidates




def preprocess(filename):       ## Both gensim tfidf and TextRank require inclusion of phrases
        ## TextRank calculate the score of phrase as average of scores of each word
    content = av.hadoop_file_to_dataframe(filename)
    raw = content.body
    text = list(itertools.chain.from_iterable(i for i in map(get_webText, raw)))
    ##remove digits, punctuations and weblinks
    text = [filter(lambda x: not x.isdigit() and not x in punct, re.sub(r'[^\x00-\x7F]+|(^https?:\/\/.*[\r\n]*)|(.*\.com$)','', i)) for i in text]

    sentences = list(itertools.chain.from_iterable(nltk.sent_tokenize(paragraph) for paragraph in text))  #split sentence
    tokens = [nltk.word_tokenize(sent.lower()) for sent in sentences]   #split words in sentence
    tags = nltk.pos_tag_sents(tokens)   ##pos_tag_sents() works much faster than pos_tag() in nltk because of certain i/o settings
    
    grammar = r'KT: {(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+}' #grammar patterns for extracting phrases in next line.
    phrases = list(itertools.chain.from_iterable(get_phrase(tag, grammar) for tag in tags))
        
    BagofWord = [sentence(i).pos_filter().filter_punct().filter_stop().lemmatize_word() for i in tags]
    document = [sent.get_bow() for sent in BagofWord if len(sent.wordlist)>1]
    return document,phrases




def TFIDF(doc, phrases):
    '''doc is the list of the words that is already preprocessed'''
    
    idf_df = mp.sqlContext.read.load('/user/p-lsnoddy/intern_idf_score')
    idf_df.registerTempTable('itf_rollup')
    
    wordlist = list(itertools.chain.from_iterable(i for i in doc))
    count = Counter(wordlist)
    tfscore = pd.DataFrame(count.items(), columns=['words', 'tfscore']) 
    
    sdf = mp.sqlContext.createDataFrame(tfscore)
    joined = sdf.join(idf_df, how='left', on=sdf.words==idf_df.token)
    idf = joined.toPandas()
    
    idf['tfidf'] = idf.tfscore*idf.normed_avg_itf
    result = idf.sort('tfidf', ascending=False)
    
    #phraseScore is the avg tfidf score of each word in the phrase.
    phraseScore = {' '.join(phrase).lower():sum(idf.tfidf[idf.words.isin(phrase)])/len(phrase) for phrase in phrases if len(phrase)>1}
    phraseScore= sorted(phraseScore.iteritems(), key=lambda x: x[1], reverse=True)
    phrase = [i[0] for i in phraseScore[:10]]
    return list(result.words[:10])+ phrase



##In each sentence, zip through every pair of consecutively appearing words 
def pairwise(sentence):
    a,b = itertools.tee(sentence)
    next(b, None)
    return itertools.izip(a,b)



##Adding node and edges from each sentence
def sentence_to_graph(graph, sentence):
    graph.add_nodes_from(sentence)

    ##w1, w2 iterate over every consecutive pair of words
    for w1, w2 in pairwise(sentence):
        if w2:
            graph.add_edge(w1, w2)  




def construct_graph(list_of_sentences):
    ##initiate the graph object
    graph = networkx.Graph()
    
    ##adding nodes and edges to the graph from all sentences
    for i in list_of_sentences:
        sentence_to_graph(graph, i)
    return networkx.pagerank(graph)




def Score_Phrase(phrase, wordranks):
    avgScore = sum(wordranks[i] for i in phrase if i in wordranks)/float(len(phrase))
    return avgScore




def TextRank(doc, phrases):
    wordranks = construct_graph(doc)
    #wordranks is a dictionary. Each word has a score.
    #phraseScore is calculated the same way as tfidf, but with textrank score.
    phraseScore = {' '.join(phrase).lower():Score_Phrase(phrase, wordranks) for phrase in phrases if len(phrase)>1}
    
    wordranks = sorted(wordranks.iteritems(), key=lambda x: x[1], reverse=True)
    phraseranks = sorted(phraseScore.iteritems(), key=lambda x: x[1], reverse=True)
    return [i[0] for i in wordranks[:20]] + [i[0] for i in phraseranks[:10]]




def find_outdoc_vocab(word, word2vecmodel, dictionary):
    outdoc_word = [i for i in word2vecmodel.most_similar(word, topn=10) if i[1]>0.5]
    for i in outdoc_word:
        oldscore = dictionary[i[0]] if i[0] in dictionary else 0
        newscore = dictionary[word]*i[1]
        if newscore > oldscore:
            dictionary[i[0]] = newscore




def find_outdoc_bigrams(phrase, word2vecmodel, textScore, phraseScore):
    bigram = ' '.join(sorted(phrase))
    oldscore = phraseScore[bigram] if bigram in phraseScore else 0
    newscore = Score_Phrase(phrase, textScore)
    if  word2vecmodel.similarity(phrase[0], phrase[1]) > 0.8 and newscore > oldscore:
        phraseScore[bigram] = newscore    




def TextRank_Word2vec(doc, phrases):
    wordranks = construct_graph(doc)
    textRankdic = {i[0]:i[1] for i in wordranks.iteritems()} 
    #create new dictionary of every item in wordranks to add more scores w/o changing the original scores.
    word2vecmodel =  gensim.models.Word2Vec.load('/mnt/datavault/lsnoddy/word2vec/Main_mincount5_stopremoved_new_100dim.model')
    #used 100 dim for this word2vec model, but 200 and 500 dim should perform similarly, though more accurately, but harder to train.

    ##adding out-of-document words and calculate its new scores
    for i in wordranks.keys():
        if i in word2vecmodel:
            find_outdoc_vocab(i, word2vecmodel, textRankdic)
    
    ##calculate scores for the existing phrases
    phraseScore = {' '.join(sorted(phrase)).lower():Score_Phrase(phrase, wordranks) for phrase in phrases if len(phrase)>1}
    
    ##calculate scores for bi_grams that do not actually in the document but are semantically similar and may form a phrase
    for i in itertools.combinations(textRankdic.keys(),2):
        if i[0] in word2vecmodel and i[1] in word2vecmodel:
            find_outdoc_bigrams(i, word2vecmodel, textRankdic, phraseScore)
    
    textRankdic = sorted(textRankdic.iteritems(), key=lambda x: x[1], reverse=True)
    phraseScore = sorted(phraseScore.iteritems(), key=lambda x: x[1], reverse=True)

    return [i[0] for i in textRankdic[:20]] +[i[0] for i in phraseScore[:10]]




def Word2vec_TextRank(doc, phrases):

    ##load trained word2vec model
    word2vecmodel =  gensim.models.Word2Vec.load('/mnt/datavault/lsnoddy/word2vec/Main_mincount5_stopremoved_new_100dim.model')
    wordlist = [word for word in set(itertools.chain.from_iterable(i for i in doc)) if word in word2vecmodel]

    ##expand to out-of-doc vocabulary
    all_vocab = []
    for word in wordlist:
        similar_words = [i[0] for i in word2vecmodel.most_similar(word) if i[1]>0.9]
        all_vocab = all_vocab + similar_words
    all_vocab = list(set(wordlist + all_vocab))
    
    threshold = 0.6

    ##perform TextRank on all words
    graph = networkx.Graph()
    graph.add_nodes_from(all_vocab)
    for i in itertools.combinations(all_vocab,2):
        if word2vecmodel.similarity(i[0],i[1]) > threshold:
            graph.add_edge(i[0], i[1])
    wordranks = networkx.pagerank(graph)
    
    phraseScore = {' '.join(sorted(phrase)).lower():Score_Phrase(phrase, wordranks) for phrase in phrases}

    ##find bigrams that might form a phrase
    for i in itertools.combinations(wordranks.keys(),2):
        find_outdoc_bigrams(i, word2vecmodel, wordranks, phraseScore)
    
    wordranks = sorted(wordranks.iteritems(), key=lambda x: x[1], reverse=True)
    phraseScore = sorted(phraseScore.iteritems(), key=lambda x: x[1], reverse=True)

    #return top 10 keyword and top 10 key phrases
    return [i[0] for i in wordranks[:10]] +[i[0] for i in phraseScore[:10]]




def Main_Solution(campaignId):
    path = 'hdfs://coral-m01/user/streams/r0/published/ad-categorization/content/'+str(campaignId)
    files = get_ipython().getoutput(u'hadoop fs -ls $path')
    pathnames = files[-1].split(' ')[-1]
    files = get_ipython().getoutput(u'hadoop fs -ls $pathnames')
    if len(files)>1:
        filepath = files[1].split(' ')[-1]
    else:
        print 'NO FILE FOUND'
        return None
    
    class Solution():
        def __init__(self, doc, phrase):
            self.TFIDF = TFIDF(doc, phrases)
            self.TextRank = TextRank(doc, phrases)
            self.TextRank_Word2vec = TextRank_Word2vec(doc, phrases)
            self.Word2vec_TextRank = Word2vec_TextRank(doc, phrases)
        
    doc, phrases= preprocess(filepath)
    return Solution(doc, phrases)


##Initiate the training of word2vec (as of the latest version): skip-gram model in the latest version gives better performance, but CBOW model in the old version (0.9.0) gives better performance
def continue_training_word2vec_model(doc):
    word2vecmodel = gensim.models.Word2Vec(doc, min_count=5, workers =4, size = 100, sg=1, negative = 20)
    word2vecmodel.save('/mnt/datavault/lsnoddy/word2vec/Main_mincount5_stopremoved_new_100dim.model')



##Latest version of gensim support on-line traning of word2vec
def continue_training_word2vec_model(doc):
    model =  gensim.models.Word2Vec.load('/mnt/datavault/lsnoddy/word2vec/Main_mincount5_stopremoved_new_100dim.model')
    model.train(doc)
    model.save('/mnt/datavault/lsnoddy/word2vec/Main_mincount5_stopremoved_new_100dim.model')







##This list contain all the campaign IDs that will kill the kernel

stopid = set([15225, 15013, 15009, 14985, 14973, 14959, 14943, 14942, 14775, 14749, 14720, 14717, 14611, 14378, 14302, 14242, 14229, 14226, 
              14223, 14202, 14174, 14089, 14014, 14001, 13939, 13890, 13665, 13617, 13613, 13588, 13544, 13496, 13418, 13261, 
              13220, 13094, 13015, 13000, 12893, 12805, 12769, 10493, 7475] )


##contains all the campaign ids we will test
df = pd.read_pickle('/mnt/datavault/lsnoddy/campaign_ids_for_testing.pkl')
campaignIds = df.campaign_id
result_dict = {i:Main_Solution(i) for i in campaignIds if i not in stopid}



##Save as pickle for later use
newresult = {}
for i in result_dict:
    newresult[i] = {'TFIDF': result_dict[i].TFIDF, 'TextRank':result_dict[i].TextRank, 'TextRank_Word2vec':result_dict[i].TextRank_Word2vec, 'Word2vec_TextRank':result_dict[i].Word2vec_TextRank}
print newresult[13553]



with open('/mnt/datavault/lsnoddy/result_dict.pkl','wb') as f:
    pickle.dump(newresult, f, pickle.HIGHEST_PROTOCOL)


#with open('/mnt/datavault/lsnoddy/result_dict.pkl','r') as f:
#    newresult = pickle.load(f)






