

####  Setup  #### 

## Packages to have installed
# install.packages(
#   "tidytext","dplyr","tidyr","ggplot2","plotly","xlsx",
#   "RWeka","RWeka","NLP","tm","topicmodels","ldatuning",
#   "wordcloud2","qdap","ngram","textstem","lexRank"
#   )

## Importing Libraries
# DF Manipulation
library(tidytext) # Text Mining Manipulation
library(dplyr) # DF Manipulation
library(tidyr) # Tidy Manipulation
# Visualization
library(ggplot2) # General Graphics
library(plotly) # Interactive plots
library(wordcloud) # Normal wordclouds
library(wordcloud2) # Interactive wordclouds
# Text Mining
library(RWeka) # For NGramTokenizer()
library(NLP) # Natural Language Processing Infrastructure
library(tm) # Text Mining Package
library(textstem) # For Lemmatisation
library(topicmodels) # For LDA()
library(ngram) # For concatenate()
library(ldatuning) # For K selection in LDA()
# Others
library(xlsx) # For Writing .xlsx

## Importing Document
df.xlsx <- read.xlsx("DATA"/,1) %>% 
  as.data.frame() %>%
  drop_na()


####  Summarization  #### 

# Creating an Empty DF
df_summary <- 
  data.frame(
    Category = factor(), 
    Id = factor(), 
    Sentence = character(), 
    Value = double()
  )
# Loop for filling the Empty DF
N <- 3
for (i in seq(2, 8, by=1)) {
  df_summary <- 
    rbind(
      df_summary,
      df.xlsx[,i] %>%
        # Creating Summaries
        lexRankr::lexRank(
          n = N, 
          continuous = FALSE,
          docId = rep(1, length(df.xlsx[,i]))
        ) %>% 
        # Data Wrangling
        as.data.frame() %>%
        select(
          !docId,
          Id = sentenceId,
          Sentence = sentence,
          Value = value
        ) %>%
        mutate(
          Id = Id %>% stringr::str_remove("1_") %>% as.factor(),
          Sentence = Sentence %>% as.character(),
          Value = Value %>% as.numeric() %>% round(3)
        ) %>%
        mutate(
          Category = colnames(df.xlsx)[i] %>% as.factor(), .before = "Id"
        ) %>%
        arrange(desc(Value)) 
    )
}
# Visualizing Results
View(df_summary)


####  Cleaning Data  #### 

# Creating new Object for the Loop
df.raw <- df.xlsx
for (i in seq(2, 8, by=1)) {
  # Removing Control Characters
  df.raw[,i] <- gsub("[[:cntrl:]]", " ", df.raw[,i])
  # General
  df.raw[,i] <- 
    df.raw[,i] %>%
    # Lowercassing
    tolower() %>%
    # Removing Sotpwords
    removeWords(c(
      # Selecting words to delete
      "get","can","like","also","even","toward","etc",
      "another","come","enough","common","one","just",
      # General package
      words = stopwords("english"))) %>%
    # Removing Punctuation
    removePunctuation() %>%
    # Replacing numbers with words
    # qdap::replace_number() %>%
    # Removing numbers
    removeNumbers() %>%
    # Removing extra white spaces
    stripWhitespace() %>%
    # Removing text within brackets
    qdap::bracketX() %>%
    # Replacing abbreviations
    qdap::replace_abbreviation() %>%
    # Replacing symbols
    qdap::replace_symbol() 
}
# Final Object
df <- df.raw


####  Frequency Description  #### 

## General Option with all Columns
# Creating an Empty DF
df_freq <- 
  data.frame(
    Category = factor(), 
    Number.Ngram = factor(), 
    Ngram = factor(), 
    Frequency = double()
  )
# Defining number of Rows for each Ngram
N <- 20
# Loop for filling in the Empty DF
for (num in seq(1, 3, by=1)) {
  for (i in seq(2, 8, by=1)) {
    # Tokenizing
    df_token <- 
      df %>% 
      unnest_tokens(n = num,
                    input = colnames(df)[i],
                    output = "Ngram", 
                    token = "ngrams")
    # Condition for Lemmatisation: words / strings
    if (num == 1) {
      df_token <- 
        df_token %>%
        mutate(Ngram = Ngram %>% 
                 lemmatize_words())
    } else {
      df_token <- 
        df_token %>%
        mutate(Ngram = Ngram %>% 
                 lemmatize_strings())
    }
    # Final DF
    df_freq <- 
      rbind(
        df_freq,
        df_token %>%
          # Count
          count(Ngram) %>% 
          rename(Frequency = n) %>%
          arrange(desc(Frequency)) %>%
          # Selecting main rows
          slice(1:N) %>%
          # Adding Initial Columns
          mutate(
            Category = colnames(df)[i] %>% as.factor(),
            Number.Ngram = paste(num),
            .before = "Ngram"
          ) %>% 
          # Deleting NA
          drop_na()
      ) 
  }
}

## Selecting Category (Column) and Ngram(s) for Visualizations
# Columns
levels(df_tf_idf$Category)
# Selection
df_freq_filtered <- 
  df_freq %>%
  filter(
    Category %in% c("Improvements"),
    Number.Ngram %in% c("1","2","3")
  )

## Worldcloud
# Interactive Dark
wordcloud2(data = df_freq_filtered %>% select(Ngram,Frequency), 
           #minRotation = -pi/6, maxRotation = -pi/6, rotateRatio = 1,
           size = 1.1, color='random-light', backgroundColor="black")
# Interactive White
wordcloud2(data = df_freq_filtered %>% select(Ngram,Frequency), 
           size = 1.1, color='random-dark', backgroundColor="white")
## Horizontal Barplot with Mean line
ggplotly(
  ggplot(df_freq_filtered[1:30,], aes(x=Ngram, y=Frequency, fill=Ngram)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    geom_hline(yintercept=mean(df_freq_filtered$Frequency[1:30]),
               linetype="dashed", color="#E31414", size=.6, alpha=.8) +
    labs(x=NULL, y=NULL, fill=NULL) +
    theme(panel.grid.major.x = element_blank(),
          panel.grid.major=element_line(colour="#00000018"),
          panel.grid.minor=element_line(colour="#00000018"),
          panel.background=element_rect(fill="transparent",colour=NA))
) %>% 
  layout(showlegend = FALSE)


####  TF-IDF Models  #### 

## If a word appears frequently in a document, then it should be important ##
## and we should give that word a high score.                              ##
## But if a word appears in too many other documents,                      ##
## it’s probably not a unique identifier,                                  ##
## therefore we should assign a lower score to that word.                  ##
##                                                                         ##
## tf = n / total : word's frequency per ID, document, etc.                ##
## idf = log(documents per corpus) / (documents where term appears + 1)    ##
##                                                                         ##
## The inverse document frequency (tf-idf) is very low (near zero)         ## 
## for words that occur in many of the documents in a collection;          ##
## this is how this approach decreases the weight for common words.        ##

## General Option with all Columns
# Creating an Empty DF
df_tf_idf <- 
  data.frame(
    Category = factor(), 
    Number.Ngram = factor(), 
    tf = double(),
    idf = double(),
    tf_idf = double()
  )
# Defining number of Rows for each Ngram
N <- 10
# Loop for filling in the Empty DF
for (num in seq(1, 3, by=1)) {
  for (i in seq(2, 8, by=1)) {
    # Tokenizing
    df_token <- 
      df %>% 
      unnest_tokens(n = num,
                    input = colnames(df)[i],
                    output = "Ngram", 
                    token = "ngrams")
    # Condition for Lemmatisation: words / strings
    if (num == 1) {
      df_token <- 
        df_token %>%
        mutate(Ngram = Ngram %>% 
                 lemmatize_words())
    } else {
      df_token <- 
        df_token %>%
        mutate(Ngram = Ngram %>% 
                 lemmatize_strings())
    }
    # Final DF
    df_tf_idf <- 
      rbind(
        df_tf_idf,
        df_token %>%
          # Tf Idf
          count(ID, Ngram) %>% 
          bind_tf_idf(Ngram, ID, n) %>% 
          # Grouping by sum
          group_by(Ngram) %>% 
          summarise(
            tf = sum(tf) %>% round(1),
            idf = sum(idf) %>% round(1),
            tf_idf = sum(tf_idf) %>% round(1)
          ) %>% 
          # Creating DF and ordering
          as.data.frame() %>% 
          arrange(desc(tf_idf)) %>%
          # Filtering
          filter(!Ngram %in% c("n","na")) %>%
          # Selecting main rows
          slice(1:N) %>%
          # Adding Initial Columns
          mutate(
            Category = colnames(df)[i] %>% as.factor(),
            Number.Ngram = paste(num),
            .before = "Ngram"
          ) %>% 
          # Deleting NA
          drop_na()
      ) 
  }
}

## Selecting Category (Column) and Ngram(s) for Visualizations
# Columns
levels(df_tf_idf$Category)
# Selection
df_tf_idf_filtered <- 
  df_tf_idf %>%
  filter(
    Category %in% c("Empowerment"),
    Number.Ngram %in% c("1","2","3")
  )

## Worldcloud
wordcloud2(data = df_tf_idf_filtered %>% select(Ngram,tf_idf), 
           size = .4, color='random-dark', backgroundColor="white")
## Horizontal Barplot
ggplotly(
  ggplot(df_tf_idf_filtered, aes(x=Ngram, y=tf_idf, fill=Ngram)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    geom_hline(yintercept=mean(df_tf_idf_filtered$tf_idf),
               linetype="dashed", color="#E31414", size=.6, alpha=.8) +
    labs(x=NULL, y=NULL, fill=NULL) +
    theme(panel.grid.major.x = element_blank(),
          panel.grid.major=element_line(colour="#00000018"),
          panel.grid.minor=element_line(colour="#00000018"),
          panel.background=element_rect(fill="transparent",colour=NA))
) %>% 
  layout(showlegend = FALSE)



####  Topic Modeling  #### 

### CREATING VOCABULARY AND MODEL

## Creating Term Frequency Matrix
myCorpus <- 
  df %>%
  # Selecting Column
  pull(Empowerment) %>%
  # Steming
  # as.character() %>% 
  # stemDocument(language="english") %>%
  # Creating Corpus with Vectors
  VectorSource() %>% VCorpus() %>%
  # Creating the Matrix
  DocumentTermMatrix() %>%
  # Adjusting Sparse Terms
  removeSparseTerms(sparse = 0.99) %>%
  as.matrix()
## Selecting appropriate number of K
# Creating the Corpus Matrix
k_topics<- 
  myCorpus %>%
  FindTopicsNumber(
    topics = seq(2, 30, by=1),
    metrics = c("Griffiths2004", 
                "CaoJuan2009", 
                "Arun2010", 
                "Deveaud2014"),
    method = "Gibbs",
    control = list(seed = 12345),
    mc.cores = 4L, # It depends on the computer performance
    verbose = TRUE
  )
# Selecting K 
FindTopicsNumber_plot(k_topics)
## Creating the Latent Dirichlet Allocation Model
LDA_model <- 
  myCorpus %>%
  LDA(method = "Gibbs", k = 4, 
      control = list(seed = 12345))
#LDA_model@loglikelihood

### WORD-TOPIC PROBABILITIES ANALYSIS 

## Topics & Terms
# Topics 
LDA_topics <- 
  LDA_model %>% 
  topics() %>% 
  as.data.frame() %>%
  count(topics(LDA_model), 
        sort = TRUE) %>%
  rename(Topics = 'topics(LDA_model)',
         Frequency = n) %>%
  mutate(Topics = paste("Topic ",Topics,sep=""))
# Terms
LDA_terms <- 
  LDA_model %>% 
  terms(10) %>%
  t() %>%
  as.data.frame() 
# Loop for creating the General DF
LDA_top.term <- LDA_topics %>% mutate(Terms = "")
for (i in seq(1:nrow(LDA_terms))) {
  LDA_top.term[i,3] <- concatenate(LDA_terms[i,],collapse=" - ")
}

## Weight
LDA_beta <- 
  LDA_model %>% 
  # Extracting LDA_model@gamma
  tidy(matrix = "beta") %>% # For documents, ID or subjects: "gamma"
  # Building DF
  group_by(topic) %>%
  top_n(10, beta) %>%
  ungroup() %>%
  arrange(topic,-beta) %>%
  as.data.frame() %>%
  mutate(Topic = paste0("Topic ", topic) %>% as.factor(),
         Term = term %>% as.factor(),
         Beta = beta %>% round(3)) %>%
  select(!c(beta,term,topic))
## Visualization
ggplotly(
  ggplot(LDA_beta, aes(x=Term, y=Beta, fill=Topic)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    labs(x=NULL, y=NULL, fill=NULL) +
    theme(panel.grid.major.x = element_blank(),
          panel.grid.major=element_line(colour="#00000018"),
          panel.grid.minor=element_line(colour="#00000018"),
          panel.background=element_rect(fill="transparent",colour=NA))
)









#





