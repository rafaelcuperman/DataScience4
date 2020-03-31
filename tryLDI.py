from Main import *
from scipy.sparse import load_npz
from sklearn.cluster import FeatureAgglomeration


plt.close('all')

#------- Set parameters -------#

## Preprocessing parameters
recomputeData = False
lemma = True
stem = None
minOccurrances = 3
tf_idf = False
path = "RawData/"

## Rank reductions for LDI
k_LDI = [5, 10, 50, 100, 200, 500, 1000]

## Clustering parameter for hierarchical
n_clusters = 5

# Plot silhouettes
plotSilhouettes = True

# Plot word clouds
plotWordClouds = False

# Plot distribution
plotClusterDist = True


#-----------------------------#

if not recomputeData:
    try:
        print("Reading files...\n")
        data = sparse.load_npz(path +'matrix.npz')
        with open(path +'tweetList.csv', newline='') as csvfile:
            tweetList = list(csv.reader(csvfile, delimiter = ','))[0]
        with open(path +'tweetTime.csv', newline='') as csvfile:
            tweetTime = list(csv.reader(csvfile, delimiter = ','))[0]
        with open(path +'tweetIndices.csv', newline='') as csvfile:
            tweetIndices = list(csv.reader(csvfile, delimiter = ','))[0]
        with open(path +'wordList.csv', newline='') as csvfile:
            wordList = list(csv.reader(csvfile, delimiter = ','))[0]
        with open(path +'wordFreq.csv', newline='') as csvfile:
            wordFreq = list(csv.reader(csvfile, delimiter = ','))[0]

    except:
        print("No precalculated data in RawData - Proceeding to calculate new data set...\n")
        tmp = Tweets(path = path, lemma = lemma, stem = stem, minOccurrances = minOccurrances, tf_idf = tf_idf)
        tmp.saveData(path = path, files = 'all')
        data = tmp.matrix
        tweetList = tmp.tweetList
        tweetTime = tmp.tweetTime
        tweetIndices = tmp.tweetIndices
        wordList = tmp.wordList
        wordFreq = tmp.wordFreq
        tmp = []

    finally:
        print("Starting cluster analysis\n")
else:
    print("Proceeding to calculate new data set...\n")
    tmp = Tweets(path = path, lemma = lemma, stem = stem, minOccurrances = minOccurrances, tf_idf = tf_idf)
    tmp.saveData(path = path, files = 'all')
    data = tmp.matrix
    tweetList = tmp.tweetList
    tweetTime = tmp.tweetTime
    tweetIndices = tmp.tweetIndices
    wordList = tmp.wordList
    wordFreq = tmp.wordFreq
    tmp = []
    print("Starting cluster analysis\n")


n_samples, n_features = data.shape
print("n_clusters: %d, \t n_samples: %d, \t n_features: %d"
          % (n_clusters, n_samples, n_features))


# Execute LDI
model = {}
silhouette = {}
calinsky = {}
for k in k_LDI:
    print("Executing model with LDI with k = {}...\n".format(k))
    dataLSI = LSI(data, k)
    dataLSI = np.transpose(dataLSI)
    
    # Execute Hierarchical clustering and compute silhouette metric
    name = 'Hierarchical{}'.format(k)
    model[k] = clusterData(dataLSI, alg = 'hierarchical', n_clusters = n_clusters)
    calinsky[k] = calinski(model[k], dataLSI)
    silhouette[k] = silhouette_metric(model[k], dataLSI,\
       name_model = name, plotGraph = plotSilhouettes)
    print("Done\n")
    
    ## Plot Word Cloud
    if plotWordClouds:
        print("Building word cloud...\n")
        plot_wordCloud(model[k], tweetList)
        print("Done\n")
        
    ## Plot tweets per cluster
    if plotClusterDist:
        print("Building bar plot (tweets per cluster)...\n")
        tweetsPerCluster(model[k])
        print("Done\n")