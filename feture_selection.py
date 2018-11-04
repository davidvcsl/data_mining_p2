import nltk
import re
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.cluster import KMeans
import numpy
from nltk.stem import SnowballStemmer
from sklearn.feature_selection import SelectKBest, f_regression,chi2, VarianceThreshold,SelectPercentile

print("Seleccione el modo de feature selection supervisado:")
inp = input('Usar SINSET -> 0 \nUsar POS -> 1\n-> ')
try:
    mode = int(inp)
except ValueError:
    print("Invalid number")

print("Parseando corpus...")
words = []
with open("wikicorpus_48") as f:
    for line in f:
        split = line.split(" ")
        word = split[0]
        if re.search('[a-zA-Z0-9_]', word):
            if len(split) == 4 and word!="ENDOFARTICLE": #Vemos si la linea tiene una palabra valida.
                tag = nltk.pos_tag([word],lang='es')[0][1]
                #GUARDAMOS PALABRA POS SINSET LEMA Y TAG
                words.append([re.sub('ENDOFARTICLE', '', word),split[2],re.sub('\n', '', split[3]),re.sub('ENDOFARTICLE', '', split[1]),tag]) #A word y el lema le hacemos un sub ya que hay lineas bugueadas al estilo San_Vadim_ENDOFARTICLE san_vadim_endofarticle NP00000 0
            elif (word == "ENDOFARTICLE"):
                words.append(word)

print("Creando diccionario...")
dictionary = dict()
stemmer = SnowballStemmer('spanish')
tam = len(words)
for i in range(tam):
    vect = words[i]
    word = vect[0]
    if word!= "ENDOFARTICLE":
        tag = vect[4]
        POS = vect[1]
        SIN = vect[2]
        lemma = vect[3]
        palabra = word.lower() if tag != "CD" else "NUMERO"
        if palabra not in nltk.corpus.stopwords.words('spanish') and len(palabra)>1:
            if palabra not in dictionary:
                dictionary[palabra] = defaultdict(int)
                if mode: #Entra si mode == 1
                    dictionary[palabra]['POS'] = POS
                else:
                    dictionary[palabra]['SIN'] = SIN
            dictionary[palabra][lemma] += 1
            dictionary[palabra]['is_number'] += (tag == "CD")
            dictionary[palabra]['is_upper'] += word.isupper()
            dictionary[palabra][tag] += 1 #POSTAG
            dictionary[palabra]['frequency'] += 1
            dictionary[palabra][stemmer.stem(palabra)+'_stem'] += 1
            if i!=0:
                prevword = words[i - 1][0]
                if prevword != "ENDOFARTICLE":
                    prevtag = words[i - 1][4]
                    prevword_lema = words[i-1][3]
                    dictionary[palabra][prevword+'-'] += 1
                    dictionary[palabra][prevtag+'-'] += 1 
                    dictionary[palabra]['prevword_upper'] += prevword.isupper()
                    dictionary[palabra]['prev_lema'+prevword_lema] += 1
                else:
                    dictionary[palabra]['INITOFARTICLE'] += 1
            else:
                dictionary[palabra]['INITOFARTICLE'] += 1
            if i != tam -1:
                nextword = words[i+1][0]
                if nextword != "ENDOFARTICLE":
                    nexttag = words[i+1][4]
                    nextword_lema = words[i+1][3]
                    dictionary[palabra][nextword+'+'] += 1
                    dictionary[palabra][nexttag+'+'] += 1
                    dictionary[palabra]['nextword_upper'] += nextword.isupper()
                    dictionary[palabra]['next_lema'+nextword_lema] +=1
                else:
                    dictionary[palabra]['ENDOFARTICLE'] += 1
            else:
                dictionary[palabra]['ENDOFARTICLE'] += 1

print("Sacando palabras poco frecuentes...")
elim = list()
if mode:
    for elem in dictionary:
        if dictionary[elem]['frequency']<50:
            elim.append(elem)
else:
    for elem in dictionary:
        if dictionary[elem]['frequency']<35 or dictionary[elem]["SIN"] == "0": #Eliminamos las que tienen SINSET = 0
            elim.append(elem)
for elem in elim:
    del dictionary[elem]

target_vec= [] #Armamos el vector con el target recuperando el pos/sin del diccionario y luego eliminamos el feature
if mode:
    for elem in dictionary:
        target_vec.append(dictionary[elem]["POS"])
        del dictionary[elem]["POS"]
else:
    for elem in dictionary:
        target_vec.append(dictionary[elem]["SIN"])
        del dictionary[elem]["SIN"]

print("Vectorizando...")
wordindex = list([word for word in dictionary]) # guardamos los indices
feat_dict = [dictionary[word] for word in dictionary.keys()]
dv = DictVectorizer(sparse=False)
word_vectors = dv.fit_transform(feat_dict)

print("Normalizando...")
vec_sums = word_vectors.sum(axis=1)
word_vectors = word_vectors / vec_sums[:,numpy.newaxis] 

print("Reduciendo dimensionalidad...") 
selector = VarianceThreshold(threshold = 0.000000001)
new_word_vecs = selector.fit_transform(word_vectors)

#selected = SelectPercentile(chi2, percentile = 10)
#word_vecs_new=selected.fit_transform(new_word_vecs,target_vec)

if mode:
    selected = SelectKBest(chi2, k=800)
else:
    selected = SelectKBest(chi2, k=5000)
word_vecs_new = selected.fit_transform(new_word_vecs, target_vec)

print("Kmeans...")
kmwv = numpy.array(word_vecs_new)
clusters_size = 45 #CANTIDAD DE CLUSTERS A USAR
kmeans = KMeans(clusters_size, max_iter=500, random_state=0).fit(kmwv)
labels = kmeans.labels_

print("Juntando clusters...")
diclong = len(kmeans.labels_)
clusters = [[] for _ in range(clusters_size)]
i = 0
for i in range(diclong):
    clusters[kmeans.labels_[i]].append(wordindex[i])

for cluster in clusters:
    print(cluster)
    print(len(cluster))
