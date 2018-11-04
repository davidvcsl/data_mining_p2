# Práctico de Feature Selection - David González  
- - -

Preproceso:
-- 
##### **Corpus anotado:**
- El corpus utilizado es el wikicorpus de Cristian, el cuál ya viene en utf-8, por lo tanto "ahorra" tener que lidiar con el encoding de ISO-8859-1.
- Elegí un archivo de 1 millón de palabras aproximadamente, ya que tomando más palabras el algoritmo demora demasiado y en mayor medida tira errores de memoria con más facilidad. 
- El archivo contiene temas muy variados por lo que no hubo necesidad de samplear varios archivos para obtener buenos clusters.
- El corpus contiene algunas líneas con bugs, como por ejemplo "San_Vadim_ENDOFARTICLE san_vadim_endofarticle NP00000 0", las cuáles necesitan ser "limpiadas". También trae líneas donde se taguean signos de puntuación, las cuáles deben ser ignoradas.
##### **Corpus no anotado:**
 - El corpus elegido fué, al igual que para el práctico anterior, el de **La Voz**. 
 - Tiene 15700 líneas aprox., y alrededor de 5 millones de palabras.
 - Particularmente tiene pocas palabras "basura", por lo que no es mucho el esfuerzo para "limpiarlo" (en general nltk se encarga de la mayoría del trabajo).

##### **Librerías utilizadas:**
- **re**:
	 - search: Busca si un string matchea de alguna forma otro string.
	 - sub: Devuelve el string reemplazando un sub-string por otro.
- **nltk**:
	 - SnowballStemmer: Stemmer.
	 - sent_tokenize: Tokenizador de oraciones.
	 - word_tokenize: Tokenizador de palabras.
	 - pos_tag: Tagger.
	 - stopwords: Para eliminar stopwords.
- **sklearn**: 
	 - DictVectorizer: Transforma listas con mapeos valor-features en vectores. 
	 - VarianceThreshold: Método de selección de features basado en eliminar las features que no superan la varianza establecida.
	 - TruncatedSVD: Reducción de dimensionalidad usando LSA.
	 - SelectKBest: Método de selección de features supervisado, basado en los K features más significativos.
	 - SelectPercentile: Método de selección de features basado en elegir un porcentaje de los features más significativos.
	 - chi2: Funcion chi2 utilizada en SelectKBest y SelectPercentile para obtener los features más significativos.
	 - KMeans: Algoritmo de clustering K-means.
	 
##### **Feature selection supervisado:**
- Utilicé SelectKBest y SelectPercentile (por separado), ambas utilizando como función chi2. 
- Usando los **sentidos** como target: 
	- **SelectKBest**: Se encontraron mejores resultados eligiendo 5000 features, ya que con números inferiores (300 - 800 - 1200) se obtenía un cluster con el 90% de las palabras y el resto singletones.
	- **SelectPercentile**: Se encontraron mejores resultados usando el percentil por defecto (10), ya que era el porcentaje adecuado para que la cantidad resultante sea de entre 4000 y 5000 features. La razón de esta cantidad de features es análoga a la de SelectKBest.
- Usando las **PoS** como target: 
	- **SelectKBest**: Se encontraron mejores resultados eligiendo 800 features. Con más features la cantidad de singletons aumentaba considerablemente.
	- **SelectPercentile**: Se encontraron mejores resultados usando percentil igual a 2, debido a que es el porcentaje adecuado para que la cantidad resultante sea de entre 700 y 800 features. La razón de esta cantidad de features es análoga a la de SelectKBest.
##### **Feature selection no supervisado:**
- Utilicé VarianceThresHold y TruncatedSVD.
- **VarianceThreshold**: Este método fué aplicado en el **Wikicorpus**: El threshold más adecuado fue de 0.000000001, ya que disminuyó tres veces la cantidad de features aproximadamente (de 150000 a 40000). Con 0.000001 se obtuvieron disminuciones de features de hasta cuatro veces aproximadamente, aunque me pareció demasiado, por lo que al final lo dejé en el primer valor mencionado.
- **TruncatedSVD**:  Este método fué aplicado en el corpus de **La Voz**: la cantidad de features que mejor resultado demostró fué de 300, ya que a mayor cantidad se obtuvieron clusters con mucho porcentaje de palabras y varios singletons.

##### **Clusters obtenidos:**
-  En los métodos **supervisados**, en general la calidad de los clusters, en cuanto a las palabras contenidas en ellos, no cambió mucho respecto a si se usó **SelectKBest** o **SelectPercentile**, así como tampoco afectó demasiado el uso de **VarianceThreshold**. 
Lo que definitivamente demostró resultados totalmente distintos fue el target elegido. 
Por un lado, como puede observarse en la sección anterior, los parámetros utilizados en las funciones  tuvieron que setearse de forma muy diferente para poder obtener clusters aceptables. 
Por otro lado, los contenidos de los clusters variaron bastante, siendo los obtenidos usando **PoS** los de peor resultado:
	- Clusters basados en el **PoS**: Se obtuvieron clusters sin nada en común, como (fútbol, oro, áfrica, madera, color, ...). Otros con palabras relacionadas, mezcladas con otras absolutamente distintas, como (jugar, francia, roma, paris, ser, como, ello, dicha, obama, ...), (página, lima, argentina, barcelona, america, venezuela, and, the, was, quiere, jaguar, ...), (diez, ocho, siete, dos, seis, rios, sitio, hojas, campos, huecos, provincias, discografía, ...), que como podrá verse, podrían estar separados en distintos clusters.
	Por otro lado tiene buenos clusters como (condesa, jefe, marqués, conde, duque, ... ), y otros con conjugaciones de verbos.
	- Clusters basados en los **sentidos**: Los clusters obtenidos fueron mejores. Se obtuvieron clusters con sustantivos como (meses, minutos, notas, cuerdas, libros, clubes, locales, actores, militares, tierras, habitantes, mujeres, hombres, hogares, cristianos, prisioneros, jovenes, aviones, peliculas, jugadores, artistas, novelas, amigos, humanos, ...), otros como (ingles, inglesa, portugues), (situada, situado, ubicada, ubicado), (condesa, barón, marquesa, marqués, conde, duque, rey, ... ), (hermana, hermano, hermanos), (nacional, social, popular, internacional, cultural, municipal, global, rural, natural, tropical, industrial, civil, espacial, ...) y otros clusters obteniendo verbos con sus conjugaciones.
-  En los métodos **no supervisados**, como se mencionó en el práctico anterior, se obtuvieron clusters con nombres propios, con verbos, con países y provincias, con palabras que se escriben típicamente en mayúscula, con establecimientos, con números, etc. 
Si bien estos clusters son mejores que en el método supervisado, probablemente sea debido a que el texto de La Voz abarca muchas más temáticas que al mismo tiempo están en general más relacionadas que los textos del wikicorpus.