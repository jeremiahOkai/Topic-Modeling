
m memory_profiler import profile
import os
import psutil
process = psutil.Process(os.getpid())
print(process.memory_info().rss)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans
start_time = time.time()
tfidf_vectorizer = TfidfVectorizer()
countVect = CountVectorizer()
%time tfidf_matrix = tfidf_vectorizer.fit_transform(d) #fit the vectorizer to titles
# %time countVect_matrix = countVect.fit_transform(d) #fit the vectorizer to titles

#terms is just a list of the features used in the tf-idf matrix.
terms_tfidf = tfidf_vectorizer.get_feature_names()
# terms_count = tfidf_vectorizer.get_feature_names()

#performing clustering 
num_clusters = 5

km = KMeans(n_clusters=num_clusters)

%time km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

# dist is defined as 1 - the cosine similarity of each document. 
#Cosine similarity is measured against the tf-idf matrix and can be used to generate 
#a measure of similarity between each document and the other documents in the corpus 
dist = 1 - cosine_similarity(tfidf_matrix)

#uncomment the below to save your model 
#since I've already run my model I am loading from the pickle

# joblib.dump(km,  '/data/s1931628/doc_cluster.pkl')
km = joblib.load('/data/s1931628/doc_cluster.pkl')
clusters = km.labels_.tolist()

print("Top terms per cluster:")
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d:" % i),
    for ind in order_centroids[i, :4]:
        print(' %s' % terms_tfidf[ind])
print("--- %s seconds ---" % (time.time() - start_time))
# MDS()

# # convert two components as we're plotting points in a two-dimensional plane
# # "precomputed" because we provide a distance matrix
# # we will also specify `random_state` so the plot is reproducible.
# mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

# pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

# xs, ys = pos[:, 0], pos[:, 1]

# #set up colors per clusters using a dict
# cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3', 3: '#e7298a', 4: '#66a61e'}
# # cluster_colors = {0: '#1b9e77', 1: '#d95f02', 2: '#7570b3'}
# # set up cluster names using a dict
# cluster_names = {0: 'movies, holidays, history', 
#                  1: 'government, yard, earth_sciences', 
#                  2: 'television, yard, family', 
#                  3: 'music, ethnicity, family', 
#                  4: 'work, drugs, family'}

# # cluster_names = {0: 'sports, illustration',
# #                  1: 'personal_finance,yard'}
# #create data frame that has the result of the MDS plus the cluster numbers and titles
# df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, titles=d)) 

# #group by cluster
# groups = df.groupby('label')

# # set up plot
# fig, ax = plt.subplots(figsize=(17, 9)) # set size
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

# for name, group in groups:
#     ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, 
#             label=cluster_names[name], color=cluster_colors[name], 
#             mec='none')
#     ax.set_aspect('auto')
#     ax.tick_params(\
#         axis= 'x',          # changes apply to the x-axis
#         which='both',      # both major and minor ticks are affected
#         bottom='off',      # ticks along the bottom edge are off
#         top='off',         # ticks along the top edge are off
#         labelbottom='off')
#     ax.tick_params(\
#         axis= 'y',         # changes apply to the y-axis
#         which='both',      # both major and minor ticks are affected
#         left='off',      # ticks along the bottom edge are off
#         top='off',         # ticks along the top edge are off
#         labelleft='off')
    
# ax.legend(numpoints=1)  #show legend with only 1 point

# #add label in x,y position with the label as the film title
# for i in range(len(df)):
#     ax.text(df.iloc[i]['x'], df.iloc[i]['y'], df.iloc[i]['titles'], size=8)  

# plt.show() #show the plot

# #uncomment the below to save the plot if need be
# plt.savefig('clusters_tweets_title1.png', dpi=200)
