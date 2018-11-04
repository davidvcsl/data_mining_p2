import nltk
import re
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
import numpy
from nltk.stem import SnowballStemmer
from sklearn.decomposition import TruncatedSVD



text = open("lavoztextodump.txt", "r").read()

dictionary = dict()
stemmer = SnowballStemmer('spanish')

print("Tokenizando, taggeando y creando diccionario...")
for line in text.split("\n"):
    tokens = [word for sent in nltk.sent_tokenize(line) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z0-9_]', token):
            filtered_tokens.append(re.sub(r'[^\w]', '',token))

    tagged = nltk.pos_tag(filtered_tokens,lang='es')

    for index, word in enumerate(tagged):
        tag = word[1]
        palabra = word[0].lower() if tag != "CD" else "NUMERO" #Hacer mas eficiente esto?
        if palabra not in nltk.corpus.stopwords.words('spanish') and len(palabra)>1:
            if palabra not in dictionary:
                dictionary[palabra] = defaultdict(int)
            dictionary[palabra]['is_number'] += (tag == "CD")
            dictionary[palabra]['is_upper'] += word[0].isupper()
            dictionary[palabra][tag] += 1 #POSTAG
            dictionary[palabra]['frequency'] += 1
            dictionary[palabra][stemmer.stem(palabra)+'_stem'] += 1
            if index!=0:
                prevword = tagged[index - 1][0] 
                prevtag = tagged[index - 1][1]
                dictionary[palabra][prevword+'-'] += 1
                dictionary[palabra][prevtag+'-'] += 1 
                dictionary[palabra]['prevword_upper'] += prevword.isupper()
            else:
                dictionary[palabra]['START'] += 1
            if index != len(tagged) -1:
                nextword = tagged[index+1][0]
                nexttag = tagged[index+1][1]
                dictionary[palabra][nextword+'+'] += 1
                dictionary[palabra][nexttag+'+'] += 1
                dictionary[palabra]['nextword_upper'] += nextword.isupper()
            else:
                dictionary[palabra]['END'] += 1

print("Sacando palabras poco frecuentes...")
elim = list()
for elem in dictionary:
    if dictionary[elem]['frequency']<150:
        elim.append(elem)
for elem in elim:
    del dictionary[elem]

#vectorizamos el diccionario
print("Vectorizando...")
wordindex = list([word for word in dictionary]) # guardamos los indices
feat_dict = [dictionary[word] for word in dictionary.keys()]
dv = DictVectorizer(sparse=False)
word_vectors = dv.fit_transform(feat_dict)

print("Normalizando...")
vec_sums = word_vectors.sum(axis=1)
word_vectors = word_vectors / vec_sums[:,numpy.newaxis] 

print("Reduciendo dimensionalidad...") 
pca = TruncatedSVD(n_components=300)
pca.fit(word_vectors)
word_vecs_new = pca.transform(word_vectors)

#Kmeans
print("Kmeans...")
kmwv = numpy.array(word_vecs_new)
clusters_size = 30 #CANTIDAD DE CLUSTERS A USAR
kmeans = KMeans(clusters_size, max_iter=500, random_state=0).fit(kmwv)
labels = kmeans.labels_

#JUNTAMOS LAS PALABRAS POR CLUSTERS
print("Juntando clusters...")
diclong = len(kmeans.labels_)
clusters = [[] for _ in range(clusters_size)]
i = 0
for i in range(diclong):
    clusters[kmeans.labels_[i]].append(wordindex[i])

for cluster in clusters:
    print(cluster)
    print(len(cluster))
