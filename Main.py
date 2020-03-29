import pandas as pd
import re, csv, time, math
import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, calinski_harabasz_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from wordcloud import WordCloud
from matplotlib.ticker import PercentFormatter
from sklearn.decomposition import PCA
import collections
import datetime
from collections import OrderedDict
import random
from scipy.sparse.linalg import svds

  
# First run things...
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
set(stopwords.words('english'))

# Timer
def TicTocGenerator():
    ti = 0
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti
    
TicToc = TicTocGenerator()

def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    toc(False)


# Stemming function
def stemSentence(sentence,stemmer):
    stem_sentence=[]

    for word in word_tokenize(sentence):
        stem_sentence.append(stemmer.stem(word))
        stem_sentence.append(" ")

    return "".join(stem_sentence)


# Lemmatizer
def lemmaSentence(sentence,lemmat):
    lemma_sentence=[]

    for word in word_tokenize(sentence):
        lemma_sentence.append(lemmat.lemmatize(word, get_wordnet_pos(word)))
        lemma_sentence.append(" ")

    return "".join(lemma_sentence)


# Word classification - needed by Lemmatizer
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


# Write list with words (non repeating) and a second list with
# the amount of ocurrences of the words in the initial list
def countWords(stringList):
    wordList = []
    wordFreq = []
    for line in stringList:
        for word in line.split():
            if word not in wordList:
                wordList.append(word)
                wordFreq.append(1)
            else:
                wordFreq[wordList.index(word)] += 1
    return wordList, wordFreq


class Tweets:
    '''Class performing the tweet preprocessing. Computation is done when
        an object is instanciated e.g.: myInstance = Tweets()
        
        Receives three optional parameters:
            
            lemma: True, False - Lemmatize content?
            
            stem: None, 'lanc', 'porter' - Stemming algorithm
            
            minOccurrances: 1,2,3... - Remove all words that occurr less than x times
    '''

    # Stemming and Lemmatizing Objects
    global lemmatizer,porter,lanc
    lemmatizer = WordNetLemmatizer()
    porter = PorterStemmer()
    lanc = LancasterStemmer()

    def __init__(self, path = "RawData/", lemma = True, stem = None, minOccurrances = 3, tf_idf = False):
        
        print('Preprocessing tweets and extracting features')
        # Define containers
        self.tweetList = []
        self.tweetTime = []
        self.tagList   = []
        self.wordList  = []
        self.wordFreq  = []
        self.tweetIndices = []

        # Read csv file in
        try:
            data = pd.read_csv(path + "trump.csv", sep = ';',\
                encoding= 'unicode_escape', error_bad_lines = False)
        except:
            print('Error: Could not open file: {}trump.csv\n'.format(path))
            return

        # Load stopwords in
        stopWords   = stopwords.words('english') + open(path + 'rm_words.txt').read().splitlines()
        tags        = open(path + 'rm_tags.txt').read().splitlines()
        symbols     = open(path + 'rm_symbols.txt').read().splitlines()

        idx_tweet = 0
        for line, date  in zip(data['text'],data['created_at']):
            # print("Text: "+line)
            # Take lower case data
            line = line.lower()

            # Erase all symbols listed in RawData/rm_symbols.txt
            for w in symbols:
                line = line.replace(w,'')

            # Cut out all the hashtags and write them as list elements
            # without @ into tagList
            for htag in re.findall("#\w+", line):
                tagWord = re.sub("#","",htag)
                if len(tagWord) > 2:
                    self.tagList.append(tagWord)

            # Erase all other taglike things (htto, @, ...)
            for k in tags:
                line = " ".join(x for x in line.split() if not x.startswith(k))

            # Remove all non-letter digits
            line = re.compile('[^a-zA-Z\s]').sub('',line)

            # Remove all ly as advebs
            line = re.compile(r'ly ').sub(" ", line)

            # Remove all small words with lenght < 3
            line = re.sub(re.compile(r'\W*\b\w{1,2}\b'),"",line)

            # Remove stopwords and custom stopwords listed in RawData/rm_words.txt
            for w in stopWords:
                line = line.replace(' '+w+' ',' ')
                try:
                    if w == line.split()[0]:
                        line = line.replace(w+' ','',1)
                    if w == line.split()[-1]:
                        line = line.replace(' '+w,'',1)
                except:
                    continue

            # Lemmatize
            if lemma: 
                line = lemmaSentence(line,lemmatizer)

            # Remove small words again
            line = re.sub(re.compile(r'\W*\b\w{1,2}\b'),"",line)

            # Stemming: porter, lanc
            if stem == 'porter' or stem == 'Porter':
                line = stemSentence(line, porter)
            elif stem == 'lanc' or stem == 'Lanc' or stem == 'lancaster':
                line = stemSentence(line, lanc)

            # Remove stopwords and custom stopwords listed in RawData/rm_words.txt
            for w in stopWords:
                line = line.replace(' '+w+' ',' ')
                try:
                    if w == line.split()[0]:
                        line = line.replace(w+' ','',1)
                    if w == line.split()[-1]:
                        line = line.replace(' '+w,'',1)
                except:
                    continue

            # print("\nText: "+line+"\n")
            # Add lines to Tweet list
            if line:
                self.tweetList.append(line)
                self.tweetTime.append(time.strptime(date.split()[0], "%m-%d-%Y").tm_yday)
                self.tweetIndices.append(idx_tweet)
            idx_tweet = idx_tweet + 1
        
        # Free memory
        data = []
        stopWords = []
        tags = []
        symbols = []

        # Write a list with all words wordList (no repetitions)
        # And notify how often they occur in wordFreq
        [self.wordList, self.wordFreq] = countWords(self.tweetList)

        # Remove words that appear less often than minOccurrances
        while min(self.wordFreq) < minOccurrances:
            for i in range(1,minOccurrances):
                try:
                    nIdx = self.wordFreq.index(i)
                    self.wordFreq.pop(nIdx)
                    self.wordList.pop(nIdx)
                except:
                    continue

        self.matrix = createMatrix(self.tweetList,self.wordList, tf_idf)

        print('Done')
        # EO init()


    def saveData(self, files = 'all'):
        if files == 'all' or files == 'matrix':
            try:
                sparse.save_npz('RawData/matrix.npz', self.matrix)

                with open('RawData/tweetList.csv','w', newline = '') as file:
                    csv.writer(file).writerow(self.tweetList)

                with open('RawData/tweetTime.csv','w', newline = '') as file:
                    csv.writer(file).writerow(self.tweetTime)
                    
                with open('RawData/tweetIndices.csv','w', newline = '') as file:
                    csv.writer(file).writerow(self.tweetIndices)
                    
                with open('RawData/wordList.csv','w', newline = '') as file:
                    csv.writer(file).writerow(self.wordList)
                    
                with open('RawData/wordFreq.csv','w', newline = '') as file:
                    csv.writer(file).writerow(self.wordFreq)

            except:
                print('Failed saving the matrix and/or the tweetList.')

        if files == 'all' or files == 'wordList':
            with open('RawData/wordList.csv','w', newline = '') as file:
                csv.writer(file).writerow(self.wordList)

        if files == 'all' or files == 'hashTags':
            with open('RawData/hashTags.csv','w', newline = '') as file:
                csv.writer(file).writerow(self.tagList)


def createMatrix(tweetList, wordList, tf_idf = False):
    matrix = np.zeros((len(wordList),len(tweetList)),\
        dtype = np.uint8, order = 'C')
    
    # Term frequency
    for j in range(len(tweetList)):
        tmp = tweetList[j].split()
        for i in range(len(wordList)):
            matrix[i,j] = np.uint8(tmp.count(wordList[i]))
    
    if tf_idf:
        # Inverse document frequency
        numTweets = len(tweetList)
        idf = []
        for i in range(len(wordList)):
            word = wordList[i]
            count = 0
            for j in range(len(tweetList)):
                if word in tweetList[j]:
                    count += 1
            idf = np.append(idf,np.log10(numTweets/(1+count)))
        matrix = matrix * idf[:, np.newaxis]
    
    return sparse.csr_matrix(matrix)

def LSI(matrix, k=10):
    # Do SVD with k-rank reduction 
    try:
        _, s, vt = svds(matrix, k)
    except:
        _, s, vt = svds(matrix.asfptype(), k)
    
    # Reconstruct tweets (S*vt)
    return np.matmul(np.diag(s),vt)

def clusterData(data, alg = 'kmeans', n_clusters = 10, eps = 0.1, min_samples = 4):
    '''Returns the calculated cluster with the attributes:
        
    labels_  : Vector with the correspondence of the tweets to the clusters.
    inertia_ : Sum of squared distances of samples to their closest cluster center.
    n_iter_  : Number of iterations run.
    n_clusters is needed for KMeans and Hierarcical clustering. eps and min_samples
    are exclusively needed by DBSCAN'''

    # Kmeans clustering: models with different number of clusters are created
    # (specify the numbers in "clusters" variable). "model_kmeans" is the dictionary
    # where those models are saved. In that dictionary, each key (an integer with format
    # "clusters") has the model with respective number of clusters. n_jobs distributes
    # the clustering across multiple processors e.g. 2, for quicker computation. n_init
    # defines the amount of computations among the algorithm can pick the best (Big influence!)
    if alg == 'kmeans' or alg == 'kMeans' or alg == 'KMeans' or alg == 'Kmeans':
        seed = 100
        print('Kmeans clustering started with {} clusters'.format(n_clusters))
        return KMeans(n_clusters=n_clusters, precompute_distances = True,\
           n_jobs = 4, n_init = 50, random_state = seed).fit(data)


    # Hierarchical clustering: models with different number of clusters are created
    # (specify the numbers in "clusters" variable). "model_hierarchical" is the dictionary
    # where those models are saved. In that dictionary, each key (an integer with format "clusters")
    # has the model with respective number of clusters
    elif alg == 'hierarchical' or alg == 'hier':
        print('Hierarchical clustering started with {} clusters'.format(n_clusters))
        return AgglomerativeClustering(n_clusters = n_clusters,\
             affinity='euclidean', memory=None, connectivity=None,\
             compute_full_tree='auto', linkage='ward', distance_threshold=None).fit(data)


    # DBSCAN clustering: models with different epsilon and min_samples are
    # created (specify the numbers in "epsilon" and "min_points" variables).
    # "model_dbscan" is the dictionary where those models are saved. In that dictionary,
    # each key (a string with format "epsilon-min_points") has the model with respective
    # epsilon and min_points
    elif alg == 'DBSCAN' or alg == 'dbscan' or alg == 'dbScan':
        print('DBSCAN clustering started with epsilon {} and {} min_points'.format(eps, min_samples))
        return DBSCAN(eps=eps, min_samples=min_samples, metric = 'cosine').fit(data)
    else:
        print('Wrong function call: Please define alg as kmeans, hierarchical or DBSCAN\n.')
        return


def silhouette_metric(model, data, name_model, plotGraph = False):
    '''Returns the silhouette metric and the respective graph (if required):
    Receives three parameters:
        - model: clustering model
        - data: data where the model is applied
        - plotGraph (default False)
        - name_model = Name of the evaluated model
        
    s_avg  : Average silhouette metric for the clustering model
    '''
    
    labels = model.labels_
    n_clusters = len(set(labels))
    if n_clusters == 1:
        return 1
    
    s_samples = silhouette_samples(X = data, labels = labels)
    s_avg = silhouette_score(X = data, labels = labels)

    if plotGraph:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 8)
        ax.set_xlim([-1, 1])
        ax.set_ylim([0, len(data) + (n_clusters + 1) * 10])
        #ax.set_ylim([0, data.getnnz() + (n_clusters + 1)])


        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = s_samples[labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]

            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples
        
        # Adjust the plot
        ax.set_title("Silhouette plot for the various clusters for the model " + name_model + ".\nAverage value: {}".format(s_avg))
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel("Cluster label")
        ax.axvline(x=s_avg, color="red", linestyle="--")
        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.show()
    
    return s_avg
    

def calinski(model, data):
    '''Returns the Calinsky-Harabasz metric
        Receives three parameters:
        - model: clustering model
        - data: data where the model is applied
    '''
    labels = model.labels_
    return calinski_harabasz_score(data, labels)


def trim_axs(axs, N):
    """little helper to massage the axs list to have correct length..."""
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def plot_wordCloud(model, tweetList):
    cluster = model.labels_
    n_clusters = len(set(cluster))
    n_subplotcols = math.ceil(n_clusters/4)
    n_subplotrows = math.ceil(n_clusters/n_subplotcols)

    n_subplotcols = max([n_subplotcols,n_subplotrows])
    n_subplotrows = min([n_subplotcols,n_subplotrows])
    
    # Create 3 plots for 3 different images
    fig, ax = plt.subplots(n_subplotrows, n_subplotcols)
    
    ax = trim_axs(ax, n_clusters)
    for clusterNumber, axis in enumerate(ax,0):
    
        # Gather all words of the tweets in one cluster
        tweetWords = []
        for idx in np.argwhere(cluster == clusterNumber):
            for word in tweetList[idx[0]].split():
                tweetWords.append(word)
    
        # Count frequency and order them in descending frequency
        [wordList, wordFreq] = countWords(tweetWords)
        [wordFreq,wordList] = (list(x) for x in zip(*sorted(zip(wordFreq,wordList), reverse=True)))
    
        # Generate and plot WordCloud
        wordcloud = WordCloud(width = 1080, height = 1080,\
                        background_color ='white',\
                        min_font_size = 10,\
                        colormap = 'cividis').generate(' '.join(wordList[0:20])) 
    
        axis.imshow(wordcloud)
        axis.set_title('Cluster: '+str(clusterNumber))
        axis.set_axis_off()
    
    plt.tight_layout()
    plt.show()

def plotWordClusterVsTotal(model, tweetList, limWords = 100, minOccurrances=3):  
    labels = model.labels_
    n_clusters = len(set(labels))
    [wordList, wordFreq] = countWords(tweetList)
    
    # Remove words that appear less often than minOccurrances
    while min(wordFreq) < minOccurrances:
        for i in range(1,minOccurrances):
            try:
                nIdx = wordFreq.index(i)
                wordFreq.pop(nIdx)
                wordList.pop(nIdx)
            except:
                continue
            
    ## For all the dataset        
    # Descending order
    [wordFreq,wordList] = (list(x) for x in zip(*sorted(zip(wordFreq,wordList), reverse=True)))
    # In percentage
    num_words = len(wordList)
    wordFreq[:] = [x /num_words for x in wordFreq]
    
    x = np.arange(num_words)

    
    ## For each cluster
    #clusterNumber = 1
    for i in range(n_clusters):
        tweetWords = []
        for idx in np.argwhere(labels == i):
            for word in tweetList[idx[0]].split():
                tweetWords.append(word)
        num_words_cluster = len(set(tweetWords))
        
        wordFreqCluster = [tweetWords.count(word) for word in wordList]
        # In percentage
        wordFreqCluster[:] = [x /num_words_cluster for x in wordFreqCluster]
        
        #plt.figure()
        fig, ax = plt.subplots()
        ax.plot(x, wordFreq, 'r', label='Total')
        ax.plot(x, wordFreqCluster, 'b', label='Cluster {}'.format(i))
        #plt.plot(x, wordFreq, 'r', x, wordFreqCluster, 'b')
        if limWords > 0 and limWords <= num_words:
            plt.xlim((0, limWords))
        plt.xticks([])
        plt.xlabel('Words')
        plt.ylabel('Number of occurences')
        plt.legend()
        plt.show()
        

def tweetsPerCluster(model):
    # Count tweets per cluster
    counter = []
    labels = model.labels_
    num_tweets = len(labels)

    # Removed counter class to conserve the order of the cluster size!
    for i in range(max(labels)+1):
        counter.append(np.count_nonzero(model.labels_ == i)/num_tweets)
    
    # Plot
    plt.figure()
    plt.bar([i for i in range(len(counter))], counter)
    plt.ylim((0, max(counter)+0.05))
    plt.xlabel('Cluster')
    plt.ylabel('Percentage of tweets')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.grid('on')
    plt.show()

#Plots histogram (distribution) of the number of tweets over the year in cluster n, tweetTime is the list with days corresponding to tweets
def PlotHistogramYear(model, n, tweetTime):
    labels = model.labels_
    n_clusters = len(set(labels))
    
    #Create overview of tweets in which cluster e.g. cluster 1: [1,3,5,9,...], cluster 2: [2,..]
    ClusterDict = {}
    for i in range(n_clusters):
        ClusterDict["Cluster" + str(i)] = []
    for j in range(len(model.labels_)):
        for i in range(n_clusters):
            if model.labels_[j] == i:
                ClusterDict["Cluster" + str(i)].append(j)
    
    ClusterDict_days = ClusterDict
    for key in ClusterDict_days:
        for i in range(len(ClusterDict_days[key])):
            ClusterDict_days[key][i] = tweetTime[ClusterDict_days[key][i]]
    
    plt.hist(ClusterDict_days["Cluster" + str(n)], bins = list(range(1,366)))
    plt.title('Distribution of Tweets of Cluster ' + str(n))
    plt.xlabel('Day of the year')
    plt.ylabel('Number of tweets on this day')
    plt.show()
    
#Plots histogram (distribution) of occurences of words in the whole dataset
def plotOccurencesOfWords(tweets):
    word_count = tweets.wordFreq
    word_count = sorted(tweets.wordFreq)
    counter = collections.Counter(word_count)
    frequency = list(counter.values())
    times_of_occurence = list(counter.keys())
    num_words = len(word_count)
    frequency[:] = [x /num_words for x in frequency]
    
    plt.figure(figsize=(20,10))
    plt.bar(times_of_occurence, frequency)
    plt.ylim((0, max(frequency)+0.05))
    plt.xlim((0, 50))
    plt.xlabel('Number of occurences of word in tweets')
    plt.ylabel('Number of words')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()

def pcaPlot(pcaDim,data,clusterAlg,alg,tweetList):

    try:
        data = data.todense()
    except:
        pass

    # Calculate PCA-reduced data
    data = PCA(n_components = pcaDim).fit_transform(data)

    # Cluster reduced data
    model = clusterAlg(data)

    fig = plt.figure()

    if pcaDim == 2:

        plt.figure(1)
        plt.clf()

        x_min, x_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
        y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1

        if alg == 'kmeans' or alg == 'kMeans' or alg == 'KMeans' or alg == 'Kmeans':
            # Step size of the mesh. Decrease to increase the quality of the VQ.
            h = 0.01

            # Plot the decision boundary. For that, we will assign a color to each
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

            # Obtain labels for each point in mesh. Use last trained model.
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)

            plt.imshow(Z, interpolation='nearest',
                       extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                       aspect='auto', origin='lower')

            plt.plot(data[:, 0], data[:, 1], 'k.', markersize=2)

            # Plot the centroids as a white number
            centroids = model.cluster_centers_

            for i in range(n_clusters):
                plt.text(centroids[i, 0], centroids[i, 1],\
                    str(i), color='tab:orange')

            plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
                      'Centroids are marked with white cross')


        else:
            plt.scatter(data[:, 0], data[:, 1], c = model.labels_, cmap = cm.nipy_spectral)


        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.xticks(())
        plt.yticks(())


    elif pcaDim == 3:

        # Axis limits
        x_min, x_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
        y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
        z_min, z_max = data[:, 2].min() - 0.1, data[:, 2].max() + 0.1

        # Create 3D axis
        ax = fig.add_subplot(111, projection='3d')

        # Plot points
        ax.scatter(data[:, 0], data[:, 1], data[:, 2],\
           c=model.labels_, marker='^', cmap = cm.nipy_spectral)

        # Axis properties
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_title('PCA-reduced Kmeans Clustering')

    plt.show()
    

# Plot tweets per day of the week
def plotTweetsPerDayOfWeek(tweetTimes):
    # Convert day of year to day of week
    tmp = map(lambda x: datetime.datetime(2017, 1, 1) + datetime.timedelta(int(x) - 1), tweetTimes)
    dayOfWeek = list(map(lambda x: x.weekday(), tmp))
    count_dow = dict(collections.Counter(t for t in dayOfWeek))
    
    # Rename keys
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    i = 0
    for day in days:
        try:
            count_dow[day] = count_dow.pop(i)
        except:
            count_dow[day] = 0
        i = i+1
    
    # Convert to percentage
    numTweets = sum(count_dow.values())
    count_percentage = {}
    for key in count_dow:
        count_percentage[key] = count_dow[key]/numTweets
    
    list_of_tuples = [(key, round(count_percentage[key],2)) for key in days]
    count_percentage = OrderedDict(list_of_tuples)
    
    # Plot    
    plt.figure()  
    plt.bar(range(len(count_percentage)), list(count_percentage.values()), align='center')
    plt.xticks(range(len(count_percentage)), list(count_percentage.keys()))
    plt.ylabel('Percentage of tweets')
    plt.xlabel('Day of the week')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()
  
# Check cluster assignment to random number of tweets
def checkRandomTweets(model, numRandomTweets, tweetIndices, seed = None):
    
    try:
        originalTweets = pd.read_csv("RawData/trump.csv", sep = ';',\
        encoding= 'unicode_escape', error_bad_lines = False)
    except:
        print('Error: Could not open file: RawData/trump.csv\n')
        return 
    
    clusters = model.labels_
    random.seed(seed, version=2) # For reproducibility
    
    randomTweets = random.sample(range(len(clusters)), numRandomTweets)
    
    assignedClusters = {}
    for i in randomTweets:
        tweetIndexOriginal = tweetIndices[i]
        originalTweet = originalTweets.iloc[int(tweetIndexOriginal), 1]
        assignedClusters[originalTweet] = clusters[i]
        
        print(originalTweet + ' - Cluster number ' + str(clusters[i]) + '\n')
    
    return assignedClusters
        
# Compare original tweets and preprocessed ones
def compareOriginalPreprocessed(numRandomTweets, tweetIndices, tweetList, seed = None):
    
    try:
        originalTweets = pd.read_csv("RawData/trump.csv", sep = ';',\
        encoding= 'unicode_escape', error_bad_lines = False)
    except:
        print('Error: Could not open file: RawData/trump.csv\n')
        return 
    
    random.seed(seed, version=2) # For reproducibility

    randomTweets = random.sample(range(len(tweetIndices)), numRandomTweets)
    
    for i in randomTweets:
        tweetIndexOriginal = tweetIndices[i]
        print(originalTweets.iloc[int(tweetIndexOriginal), 1] + ' -> ' + tweetList[i] + '\n')

# Plot histogram of n most common words
def commonWords(wordList, wordFreq, n=50):
    totalCount = sum(wordFreq)
    tmp = sorted(zip(wordList,wordFreq), key=lambda x: x[1], reverse = True)
    tmp = list(map(lambda x: (x[0],x[1]/totalCount), tmp))
    
    try:
        top_n = tmp[0:n]
    except:
        "n has to be a positive number and smaller or equal than the number of words"
        return
    
    plt.figure()
    plt.bar(range(len(top_n)), [val[1] for val in top_n], align='center')
    plt.xticks(range(len(top_n)), [val[0] for val in top_n])
    plt.xticks(rotation=70)
    plt.ylabel('Occurence of word in whole dataset')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()
    