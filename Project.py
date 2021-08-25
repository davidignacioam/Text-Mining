

#%% Importing

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from summarizer import Summarizer
from wordcloud import WordCloud
import nltk
from nltk import word_tokenize     
from nltk.stem import WordNetLemmatizer
import heapq
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from gensim.test.utils import common_corpus, common_dictionary
from gensim.models import LdaModel
from gensim.matutils import Sparse2Corpus
from sklearn.decomposition import TruncatedSVD
from collections import Counter
from matplotlib.ticker import FuncFormatter

# Config
%matplotlib inline
plt.style.use('seaborn-whitegrid')
#nltk.download('punkt')
#nltk.download('wordnet')
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore")

# Loading data
DF = pd.read_excel(
    'Data'
    ) \
    .drop(columns='Q.No.') \
    .rename(columns={'Questionnaire' : 'Questionaire'})

# %%

for i in range(len(list(DF.Questionaire))):
    if DF.Questionaire.isnull()[i]:
        DF.Questionaire[i]=DF.Questionaire[i-1]
        DF.replace('n’t','nt',regex=True, inplace=True)

# Defining useful functions
class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(self.wnl.lemmatize(t,pos='v')) for t in word_tokenize(articles)]

def removeUndesiredChars(df, col, 
                         remDigits=True, 
                         remShortWords=True, 
                         remPuntuation=True):
    if remDigits==True:
        df[col] = df[col].str.replace('\d+', '') # removing digits
    if remShortWords==True:
        df[col] = df[col].str.replace(r'(\b\w{1,2}\b)', '') # removing words with 2 or less characters
    if remPuntuation==True:
        df[col] = df[col].str.replace('[^\w\s]', '') # removing punctuation 
    return df[col]

def entryBagDisplay(i,vectorizer, fitted):
  L=len(fitted.todense().tolist())
  if i>L-1:
    print("i out of boundaries, can't to be greater than "+str(L-1))
  else:
    countDisplay_i = {
        vectorizer.get_feature_names()[k]: fitted.todense()[i] \
           .tolist()[0][k] for k in range(len(vectorizer.get_feature_names()))
        }
    return countDisplay_i

def totalBagDisplay(vectorizer, fitted,mode='absolute'):
  if mode not in ['absolute','relative']:
    print("Error: The availables modes are 'absolute' or 'relative'")
  else:
    countSum = [0] * len(fitted.todense()[0].tolist()[0])
    for i in range(0,len(fitted.todense())):
      countSum=[x + y for (x, y) in zip(countSum, fitted.todense()[i].tolist()[0])]
    if mode=='absolute':
      countTotal={
         vectorizer.get_feature_names()[k]: countSum[k] for k in range(len(vectorizer.get_feature_names()))
         }
      return countTotal
    else:
      countTotalRelative={
          vectorizer \
             .get_feature_names()[k]: countSum[k]/sum(countSum) for k in range(len(vectorizer.get_feature_names()))
          }
      return countTotalRelative

def get_keys(topic_matrix):
    '''returns an integer list of predicted topic categories 
    for a given topic matrix'''
    keys = []
    for i in range(topic_matrix.shape[0]):
        keys.append(topic_matrix[i].argmax())
    return keys

def keys_to_counts(keys):
    '''returns a tuple of topic categories and their 
    accompanying magnitudes for a given list of keys'''
    count_pairs = Counter(keys).items()
    categories = [pair[0] for pair in count_pairs]
    counts = [pair[1] for pair in count_pairs]
    return (categories, counts)

def get_top_n_words(n, keys, document_term_matrix, count_vectorizer):
    '''returns a list of n_topic strings, where each string contains 
    the n most common words in a predicted category, in order'''
    top_word_indices = []
    for topic in range(n_topics):
        temp_vector_sum = 0
        for i in range(len(keys)):
            if keys[i] == topic:
                temp_vector_sum += document_term_matrix[i]
        temp_vector_sum = temp_vector_sum.toarray()
        top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:],0)
        top_word_indices.append(top_n_word_indices)   
    top_words = []
    for topic in top_word_indices:
        topic_words = []
        for index in topic:
            temp_word_vector = np.zeros((1,document_term_matrix.shape[1]))
            temp_word_vector[:,index] = 1
            the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]
            topic_words.append(the_word.encode('ascii').decode('utf-8'))
        top_words.append(" ".join(topic_words))         
    return top_words

def get_mean_topic_vectors(keys, two_dim_vectors):
    '''returns a list of centroid vectors from each predicted topic category'''
    mean_topic_vectors = []
    for t in range(n_topics):
        articles_in_that_topic = []
        for i in range(len(keys)):
            if keys[i] == t:
                articles_in_that_topic.append(two_dim_vectors[i])    
        
        articles_in_that_topic = np.vstack(articles_in_that_topic)
        mean_article_in_that_topic = np.mean(articles_in_that_topic, axis=0)
        mean_topic_vectors.append(mean_article_in_that_topic)
    return mean_topic_vectors

def topics_per_document(model, corpus):
    #corpus_sel = corpus[start:end]
    dominant_topics = []
    topic_percentages = []
    for i, corp in enumerate(corpus):
        topic_percs, wordid_topics, wordid_phivalues = model[corp]
        dominant_topic = sorted(topic_percs, key = lambda x: x[1], reverse=True)[0][0]
        dominant_topics.append((i, dominant_topic))
        topic_percentages.append(topic_percs)
    return(dominant_topics, topic_percentages)


# %%

for k in range(len(list(DF.Questionaire.unique()))):
    print('---'*20)
    df=DF[DF.Questionaire==list(DF.Questionaire.unique())[k]] \
      .drop('Questionaire',axis=1) \
      .set_index('Question') \
      .transpose()
    print('Questionaire : ' +str(DF.Questionaire.unique()[k]))
    n0=df.shape[0]
    print('There are ' +str(n0)+ ' entries.')
    df=df.dropna(how='all')
    print('Of which ' +str(n0-df.shape[0])+ ' are fully empty entries.')
    print('=> There are ' +str(df.shape[0])+ ' valid entries.')
    listq=[]
    listnas=[]
    for j in range(len(list(df.columns))):
        listq.append('q'+str(j+1))
        listnas.append(df.iloc[:,j].isna().sum())
    NAsDF=pd.DataFrame({'#NAs responses':listnas})
    NAsDF.index=listq
    print(NAsDF.transpose())
    print('---'*20)
    for j in range(len(list(df.columns))):
        print(str('---'*20))
        print('Analyzing Question : "' +str(df.columns[j])+'"')
        print('Number of NaNs responses : ' +str(df.iloc[:,j].isna().sum()))
        #countvectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),
        #                                  ngram_range = (1,3),
        #                                  analyzer= 'word',
        #                                  strip_accents = 'unicode',
        #                                  stop_words = 'english', 
        #                                  lowercase = True)
        tfidfvectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(),
                                          ngram_range = (1,3),
                                          analyzer= 'word',
                                          strip_accents = 'unicode',
                                          stop_words = 'english', 
                                          lowercase = True)

        texts=removeUndesiredChars(df,df.columns[j]).dropna()
        #count_fitted = countvectorizer.fit_transform(texts)
        tfidf_fitted = tfidfvectorizer.fit_transform(texts)

        # Showing most important words
        topN=10
        quantile=0.5
        Dict=totalBagDisplay(
           mode='absolute',vectorizer=tfidfvectorizer, fitted=tfidf_fitted
           )
        while len(Dict)>topN:
            Qvalue=np.quantile(np.array(list(totalBagDisplay(
                mode='absolute',vectorizer=tfidfvectorizer, fitted=tfidf_fitted
                ).values())),quantile)
            auxDict={k: v for k, v in totalBagDisplay(
                mode='absolute',vectorizer=tfidfvectorizer, fitted=tfidf_fitted
                ).items() if v >= Qvalue}
            if len(auxDict)>topN:
                Dict=auxDict
            if quantile>=0.95:
                break
            else:
                quantile=quantile+0.05

        topwords=heapq.nlargest(topN, Dict.keys(), key=lambda k: Dict[k])
        print('\n [TOP'+str(topN)+'] The most relevant concepts founded are : ')
        count=1
        for word in topwords:
            print(str(count)+'. '+word)
            count=count+1
        print('\n')
        
        
        ## CSV DOWNLOADS
        csvN=100
        topwords_csv=heapq \
           .nlargest(csvN, 
                     totalBagDisplay(
                        mode='absolute',vectorizer=tfidfvectorizer, fitted=tfidf_fitted
                        ).keys(), 
                     key=lambda k: totalBagDisplay(
                        mode='absolute',vectorizer=tfidfvectorizer, fitted=tfidf_fitted
                        )[k])
        csv_df=pd.DataFrame(
           data={'Word': topwords_csv,
                 'TF-IDF Score': [totalBagDisplay(mode='absolute',
                                                  vectorizer=tfidfvectorizer, 
                                                  fitted=tfidf_fitted)[key] for key in topwords_csv]}
           )
        csv_df.to_csv('Q'+str(k+1)+'_q'+str(j+1)+'_tfidf.csv')
        
        
        #plot topwords 
        topwords_df=pd.DataFrame(data={'Word': topwords,
                           'TF-IDF Score': [Dict[key] for key in topwords]})
        topwords_df.index=topwords_df['Word']
        plt.rcParams["figure.figsize"] = [5, 3]
        topwords_df['TF-IDF Score'].plot(kind="bar", title="test")
        # Rotate the x-labels by 30 degrees, and keep the text aligned horizontally
        plt.xticks(rotation=80, horizontalalignment="center")
        plt.title("TF-IDF Score for the Top"+str(topN)+" concepts")
        plt.xlabel("Word")
        plt.ylabel("TF-IDF Score")
        plt.show()

        # Wordcloud (TF-IDF)
        weights = totalBagDisplay(mode='relative',vectorizer=tfidfvectorizer, fitted=tfidf_fitted)
        N_meaningful=len([val for val in list(weights.values()) if val > 0]) 
        thresholdWords=100
        wordcloud = WordCloud(
           background_color='white',max_words=min(thresholdWords,N_meaningful)
           ).fit_words(weights)
        print('TF-IDF WordCloud : ')
        import matplotlib.pyplot as plt
        %matplotlib inline
        fig, ax = plt.subplots(figsize=(15,15))
        _ = ax.imshow(wordcloud, interpolation='bilinear')
        _ = ax.axis("off")
        plt.show()


# %%

for k in range(len(list(DF.Questionaire.unique()))):
    print('---'*20)
    df=DF[DF.Questionaire==list(DF.Questionaire.unique())[k]] \
       .drop('Questionaire',axis=1) \
       .set_index('Question') \
       .transpose()
    print('Questionaire : ' +str(DF.Questionaire.unique()[k]))
    n0=df.shape[0]
    print('There are ' +str(n0)+ ' entries.')
    df=df.dropna(how='all')
    print('Of which ' +str(n0-df.shape[0])+ ' are fully empty entries.')
    print('=> There are ' +str(df.shape[0])+ ' valid entries.')
    df_Q=pd.DataFrame(df.stack()).reset_index()
    df_Q.columns=['N_response', 'question', 'answer']
    
    countvectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),
                                      ngram_range = (1,3),
                                      analyzer= 'word',
                                      strip_accents = 'unicode',
                                      stop_words = 'english', 
                                      lowercase = True)
    tfidfvectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(),
                                      ngram_range = (1,3),
                                      analyzer= 'word',
                                      strip_accents = 'unicode',
                                      stop_words = 'english', 
                                      lowercase = True)

    texts=removeUndesiredChars(df_Q,'answer').dropna()
    count_fitted = countvectorizer.fit_transform(texts)
    tfidf_fitted = tfidfvectorizer.fit_transform(texts)

    # Showing most important words
    topN=10
    quantile=0.5
    Dict=totalBagDisplay(mode='absolute',vectorizer=tfidfvectorizer, fitted=tfidf_fitted)
    while len(Dict)>topN:
        Qvalue=np.quantile(np.array(list(totalBagDisplay(
            mode='absolute',vectorizer=tfidfvectorizer, fitted=tfidf_fitted
            ).values())),quantile)
        auxDict={k: v for k, v in totalBagDisplay(
            mode='absolute',vectorizer=tfidfvectorizer, fitted=tfidf_fitted
            ).items() if v >= Qvalue}
        if len(auxDict)>topN:
            Dict=auxDict
            if quantile>=0.95:
                break
            else:
                quantile=quantile+0.05

    topwords=heapq.nlargest(topN, Dict.keys(), key=lambda k: Dict[k])
    print('\n [TOP'+str(topN)+'] The most relevant concepts founded are : ')
    count=1
    for word in topwords:
        print(str(count)+'. '+word)
        count=count+1
    print('\n')

    topwords_df=pd.DataFrame(data={'Word': topwords,
                           'TF-IDF Score': [Dict[key] for key in topwords]})
    topwords_df.index=topwords_df['Word']
    plt.rcParams["figure.figsize"] = [15, 10]
    topwords_df['TF-IDF Score'].plot(kind="bar", title="test")
    
    
    ## CSV DOWNLOADS
    topwords_df.to_csv('Q'+str(k+1)+'_tfidf.csv', index=False)
    
    
    # Rotate the x-labels by 30 degrees, and keep the text aligned horizontally
    plt.xticks(rotation=80, horizontalalignment="center")
    plt.title("TF-IDF Score for the Top"+str(topN)+" concepts")
    plt.xlabel("Word")
    plt.ylabel("TF-IDF Score")
    plt.show()

    # Wordcloud (TF-IDF)
    weights = totalBagDisplay(mode='relative',vectorizer=tfidfvectorizer, fitted=tfidf_fitted)
    N_meaningful=len([val for val in list(weights.values()) if val > 0]) 
    thresholdWords=100
    wordcloud = WordCloud(
       background_color='white',max_words=min(thresholdWords,N_meaningful)
       ).fit_words(weights)

    print('TF-IDF WordCloud : ')
    import matplotlib.pyplot as plt
    %matplotlib inline
    fig, ax = plt.subplots(figsize=(15,15))
    _ = ax.imshow(wordcloud, interpolation='bilinear')
    _ = ax.axis("off")
    plt.show()

    print('LSA Topic Modelling')
    max_topics=20
    var_expl=[]
    for n_topics in range(1,max_topics):
        lsa_model = TruncatedSVD(n_components=n_topics,algorithm='arpack')
        lsa_model.fit_transform(tfidf_fitted)
        var_expl.append(sum(lsa_model.explained_variance_))

    from kneed import KneeLocator
    kn = KneeLocator(list(range(1,max_topics)),var_expl)
    n_topics=kn.knee
    lsa_model = TruncatedSVD(n_components=n_topics,algorithm='arpack') # Instantiates the LSA model
    lsa_topic_matrix = lsa_model.fit_transform(tfidf_fitted) # Runs the truncated SVD
    lsa_keys = get_keys(lsa_topic_matrix)
    lsa_categories, lsa_counts = keys_to_counts(lsa_keys)
    top_n_words_lsa = get_top_n_words(10, 
                                      lsa_keys, 
                                      tfidf_fitted,
                                      tfidfvectorizer)

    for i in range(len(top_n_words_lsa)):
        print('Topic {}: {}'.format(i, top_n_words_lsa[i]))


    top_3_words = get_top_n_words(3, 
                                  lsa_keys, 
                                  tfidf_fitted, 
                                  tfidfvectorizer)
    labels = ['Topic {}: \n'.format(i) + top_3_words[i] for i in lsa_categories]
    
    
    ## CSV DOWNLOADS
    lsa_csv_df=pd.DataFrame(data={'LSA Categories': lsa_categories,
                                  'LSA Words': top_3_words,
                                  'LSA Topic Dominance': [100*x / sum(lsa_counts) for x in lsa_counts]})
    lsa_csv_df.to_csv('Q'+str(k+1)+'_LSA.csv')
    
    
    # magnitudes of categories generated
    #fig, ax = plt.subplots(figsize=(12,5))
    #ax.bar(lsa_categories, lsa_counts)
    #ax.set_xticks(lsa_categories);
    #ax.set_xticklabels(labels, rotation='vertical');
    #ax.set_title('LSA Topic Category Counts');
    #plt.savefig('lsadist.png', dpi=600, bbox_inches='tight')
    #plt.show()
    
    # magnitudes of categories generated
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(lsa_categories,  [100*x / sum(lsa_counts) for x in lsa_counts])
    ax.set_xticks(lsa_categories);
    ax.set_xticklabels(labels, rotation='vertical');
    ax.set_title('LSA Topic Dominance');
    ax.set_xlabel('Topic')
    ax.set_ylabel('% of dominance')
    ax.set_ylim(0, 100)
    #plt.savefig('lsadist.png', dpi=600, bbox_inches='tight')
    plt.show()
    
    print('\n')
    print('LDA Topic Modelling')
    max_topics=20
    perplexity=[]
    for n_topics in range(2,max_topics):
        lda = LdaModel(Sparse2Corpus(count_fitted),
                       num_topics=n_topics,
                       alpha='auto',
                       eta='auto',
                       iterations=10000)
        perplexity.append(lda.log_perplexity(common_corpus))

    kn = KneeLocator(list(range(2,max_topics)),perplexity)
    n_topics=kn.knee
    #print(n_topics)
    lda = LdaModel(Sparse2Corpus(count_fitted, documents_columns=False),
                   id2word=dict((v, k) for k, v in countvectorizer.vocabulary_.items()),
                   num_topics=n_topics,
                   alpha='auto',
                   eta='auto',
                   iterations=10**(100),
                   per_word_topics=True)
    #lda.print_topics()
    for topic_id in range(lda.num_topics):
        topk = lda.show_topic(topic_id, 10)
        topk_words = [ w for w, _ in topk ]

        print('{}: {}'.format(topic_id, ' '.join(topk_words)))
    
    dominant_topics, topic_percentages = topics_per_document(model=lda, 
                                                             corpus=Sparse2Corpus(count_fitted))            
    # Distribution of Dominant Topics in Each Document
    df = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic'])
    dominant_topic_in_each_doc = df.groupby('Dominant_Topic').size()
    df_dominant_topic = dominant_topic_in_each_doc.to_frame(name='count').reset_index()
    df_dominant_topic['count']=100*df_dominant_topic['count']/sum(df_dominant_topic['count'])
    # Total Topic Distribution by actual weight
    topic_weightage_by_doc = pd.DataFrame([dict(t) for t in topic_percentages])
    df_topic_weightage_by_doc = topic_weightage_by_doc.sum().to_frame(name='count').reset_index()

    # Top 3 Keywords for each Topic
    topic_top3words = [(i, topic) for i, topics in lda.show_topics(formatted=False) 
                                     for j, (topic, wt) in enumerate(topics) if j < 3]

    df_top3words_stacked = pd.DataFrame(topic_top3words, columns=['topic_id', 'words'])
    df_top3words = df_top3words_stacked.groupby('topic_id').agg(', \n'.join)
    df_top3words.reset_index(level=0,inplace=True)
    
    
    ## CSV DOWNLOADS
    lda_csv_df=pd.DataFrame(data={'LDA Topic': df_dominant_topic['Dominant_Topic'],
                                  'LDA Words': df_top3words['words'],
                                  'LDA Topic Dominance': df_dominant_topic['count']})
    lda_csv_df.to_csv('Q'+str(k+1)+'_LDA.csv')
   
   
    # Plot
    fig, ax1 = plt.subplots(figsize=(5,3))
    # Topic Distribution by Dominant Topics
    ax1.bar(x='Dominant_Topic', height='count', data=df_dominant_topic, width=.5, color='firebrick')
    ax1.set_xticks(range(df_dominant_topic.Dominant_Topic.unique().__len__()))
    tick_formatter = FuncFormatter(
        lambda x, pos: 'Topic ' + str(x)+ '\n' + df_top3words.loc[df_top3words.topic_id==x, 'words'].values[0]
        )
    ax1.xaxis.set_major_formatter(tick_formatter)
    ax1.set_title('LDA Topic Dominance');
    ax1.set_xlabel('Topic')
    ax1.set_ylabel('% of dominance')
    ax1.set_ylim(0, 100)
    plt.show()
    
    df=DF[DF.Questionaire==list(DF.Questionaire.unique())[k]] \
        .drop('Questionaire',axis=1) \
        .set_index('Question') \
        .transpose()
    df_Q=pd.DataFrame(df.stack()).reset_index()
    df_Q.columns=['N_response', 'question', 'answer']
    body = ' '.join(map(str,list(df_Q.answer.dropna()))).lower()

    model = Summarizer()
    kMax=10
    withinSumSquares = model.calculate_elbow(body, k_max=kMax)
    kOptimalElbow = model.calculate_optimal_k(body, k_max=kMax)
    summary=model(body, num_sentences=kOptimalElbow)
    #print(kOptimalElbow)
    print(summary)
    
    ## Exporting Summary
    pd.DataFrame({'Q'+str(k+1)+' Summary': summary}, index=['Q'+str(k+1)]) \
        .to_csv(r'Q'+str(k+1)+'_summary.csv')


# %%

# %%
