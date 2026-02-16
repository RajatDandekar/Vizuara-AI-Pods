Embeddings in Natural Language Processing (NLP)
Token embeddings are the foundational building block in understanding large language models
Danyal Alam
Jan 07, 2025




In the fascinating world of artificial intelligence and natural language processing (NLP), token embeddings serve as the foundation upon which large language models (LLMs) like GPT, BERT, and others are built. These embeddings convert the discrete symbols of language into a continuous vector space that computers can understand and manipulate. In this blog, we’ll explore the evolution of token embeddings, their role in LLMs, and use fun examples and visualizations to simplify complex concepts.

The Journey to Token Embeddings
Before diving into token embeddings, let's take a brief journey through the history of representing words in NLP:

Bag of Words
The most basic approach to converting texts into vectors is a bag of words. Let’s look at one of the famous quotes of Richard P. Feynman“We are lucky to live in an age in which we are still making discoveries”.

We will use it to illustrate a bag of words approach.

The first step to get a bag of words vector is to split the text into words (tokens) and then reduce words to their base forms. For example, “running” will transform into “run”. This process is called stemming. We can use the NLTK Python package for it.

from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize

text = 'We are lucky to live in an age in which we are still making discoveries'

# tokenization - splitting text into words
words = word_tokenize(text)
print(words)
# ['We', 'are', 'lucky', 'to', 'live', 'in', 'an', 'age', 'in', 'which',
#  'we', 'are', 'still', 'making', 'discoveries']

stemmer = SnowballStemmer(language = "english")
stemmed_words = list(map(lambda x: stemmer.stem(x), words))
print(stemmed_words)
# ['we', 'are', 'lucki', 'to', 'live', 'in', 'an', 'age', 'in', 'which', 
#  'we', 'are', 'still', 'make', 'discoveri']
Now, we have a list of base forms of all our words. The next step is to calculate their frequencies to create a vector.

import collections
bag_of_words = collections.Counter(stemmed_words)
print(bag_of_words)
# {'we': 2, 'are': 2, 'in': 2, 'lucki': 1, 'to': 1, 'live': 1, 
# 'an': 1, 'age': 1, 'which': 1, 'still': 1, 'make': 1, 'discoveri': 1}
Actually, if we wanted to convert our text into a vector, we would have to take into account not only the words we have in the text but the whole vocabulary. Let’s assume we also have “i”, “you” and ”study” in our vocabulary and let’s create a vector from Feynman’s quote.


This approach is quite basic, and it doesn’t take into account the semantic meaning of the words, so the sentences “the girl is studying data science” and “the young woman is learning AI and ML” won’t be close to each other.

TF-IDF
TF-IDF (Term Frequency - Inverse Document Frequency) is an enhancement over the traditional bag-of-words model. It combines two key metrics through multiplication.


Term Frequency (TF) measures how often a term appears within a document. A common way to compute this is by dividing the raw count of the term (similar to the bag-of-words approach) by the total number of terms in the document. That said, there are alternative methods to calculate TF, such as using the raw count, boolean values to indicate presence, or applying various normalization techniques.


Inverse Document Frequency (IDF) gauges the uniqueness of a word across a collection of documents. Words like “a” or “the” don’t contribute much to understanding the topic of a document, while terms such as “ChatGPT” or “bioinformatics” can provide valuable clues about its subject. IDF is typically computed as the logarithm of the ratio between the total number of documents and the number of documents containing the term. A lower IDF value indicates the word is very common and carries little meaningful information.


By combining TF and IDF, we generate vectors where frequently occurring yet common words (e.g., "I" or "you") are assigned lower weights, while rare but significant terms receive higher weights. This approach offers improved results compared to the bag-of-words method but still falls short of capturing semantic meaning.

One limitation of TF-IDF is that it results in sparse vectors, as their length corresponds to the size of the vocabulary. For example, the English language contains approximately 470,000 unique words (source), leading to extremely large vectors. A typical sentence might use only 50 unique words, leaving 99.99% of the vector filled with zeros and contributing no useful information. Recognizing this inefficiency, researchers have turned their focus toward dense vector representations to better encode meaning.

However, TF-IDF still does not capture the semantic meaning of words.

Word2Vec
Word2Vec, one of the most renowned methods for dense vector representation, was introduced by Google in 2013 in the groundbreaking paper “Efficient Estimation of Word Representations in Vector Space” by Mikolov et al.

Here is the main idea of Word2Vec:

Words are represented as dense, low-dimensional vectors instead of sparse one-hot encodings or frequency-based representations.

These vectors encode semantic meaning, so words with similar meanings are closer in the embedding space.

The paper describes two primary approaches for Word2Vec:

Continuous Bag of Words (CBOW): Predicts a target word based on its surrounding words.

Skip-Gram: Performs the reverse by predicting the surrounding context given a target word.




The central idea behind Word2Vec lies in training two models: an encoder and a decoder. For instance, in the Skip-Gram approach, we might input the word “Christmas” into the encoder, which transforms it into a vector representation. This vector is then passed to the decoder, which predicts related words like “merry,” “to,” and “you.”


This method represents a significant step forward, as it begins to capture the semantic meaning of words by learning from their context.

Word2Vec embeddings can also perform vector arithmetic. For example:

King−Man+Woman≈Queen

However, Word2Vec has certain limitations. For example, Word2Vec doesn’t account for morphology, meaning it overlooks the valuable information contained in word parts (e.g., the suffix “-less” indicating absence). Later models, such as GloVe, addressed this issue by introducing subword skip-grams.

Another limitation of Word2Vec is its focus solely on individual words, making it unsuitable for encoding entire sentences. This limitation paved the way for the next major leap in natural language processing — transformers, which excel at capturing contextual relationships across words and sentences.

A Deep Dive into NLP Tokenization and Encoding with Word and Sentence  Embeddings – Data Jenius


Transformers and Sentence Embeddings
A Deep Dive into Transformers with TensorFlow and Keras: Part 1 -  PyImageSearch


The introduction of transformers in the seminal 2017 paper “Attention Is All You Need” by Vaswani et al. marked a paradigm shift in NLP. Transformers, built on a mechanism called attention, enabled models to process text contextually, considering relationships between all words in a sentence, regardless of their position.

What is Attention?
At its core, attention is about focus. When processing a word in a sentence, the model doesn’t just look at the word itself—it assigns varying levels of importance to other words in the sequence, based on how relevant they are to the current word.

For instance, in the sentence:
“The cat sat on the mat because it was warm,”
the word “it” can be linked to “the mat” due to attention mechanisms. This dynamic approach allows transformers to understand context better than earlier models.

Transformers: Key Advantages
Contextual Understanding: Unlike Word2Vec, transformers generate word representations that vary based on the surrounding context, capturing more nuanced meanings.

Flexibility: A single transformer model can be fine-tuned for diverse tasks—such as translation, summarization, and question answering—without retraining the model entirely, saving time and computational resources.

Scalability: Transformers excel in parallel computation, making them highly efficient for large datasets.

Google AI’s BERT (Bidirectional Encoder Representations from Transformers) was among the first widely adopted pre-trained transformer models. BERT processes text bidirectionally, meaning it considers the context from both preceding and following words in a sentence. This makes it more effective for tasks like named entity recognition or semantic similarity.

However, BERT operates at the token level, and generating meaningful sentence embeddings (representations of entire sentences) was initially challenging. Averaging the vectors of all tokens in a sentence was one basic approach, but it often produced subpar results.

This challenge was effectively addressed in 2019 with the release of Sentence-BERT. This model significantly outperformed its predecessors on semantic similarity tasks and provided a robust way to calculate embeddings for entire sentences.

Since the topic of sentence embeddings is vast, it’s impossible to cover all its aspects here. If you're eager to dive deeper, there are comprehensive resources available on the subject.

In this overview, we’ve traced the evolution of embeddings, gaining a high-level understanding of how they’ve progressed.

Distance Between Vectors
Embeddings are essentially vectors, and to determine how similar two sentences are, we calculate the distance between their corresponding vectors. A smaller distance indicates a closer semantic relationship.

Several metrics are commonly used to measure the distance between vectors, including:

Euclidean distance (L2)

Manhattan distance (L1)

Dot product

Cosine distance

Let’s explore these metrics using two simple 2D vectors as examples.

vector1 = [1, 4]
vector2 = [2, 2]
Euclidean Distance (L2)
Euclidean distance, also known as the L2 norm, is the most common way to measure the straight-line distance between two points or vectors. It’s widely used in everyday contexts, such as calculating the distance between two towns.

Here’s a formula and visual representation of L2 distance.




You can compute this metric either by writing the formula manually in Python or by using the built-in functionality in libraries like NumPy.

import numpy as np

sum(list(map(lambda x, y: (x - y) ** 2, vector1, vector2))) ** 0.5
# 2.2361

np.linalg.norm((np.array(vector1) - np.array(vector2)), ord = 2)
# 2.2361
Here’s a rephrased version of the text with a fresh tone and structure:

Manhattan Distance (L1)
The L1 norm, or Manhattan distance, derives its name from the grid-like street layout of Manhattan, New York. The shortest path between two points in such a grid follows the streets, resembling the L1 distance calculation.




This distance can also be implemented manually or computed using NumPy functions.

sum(list(map(lambda x, y: abs(x - y), vector1, vector2)))
# 3

np.linalg.norm((np.array(vector1) - np.array(vector2)), ord = 1)
# 3.0
Dot Product
The dot product (or scalar product) offers another way to understand the relationship between two vectors. This metric, however, is a bit more nuanced:


sum(list(map(lambda x, y: x*y, vector1, vector2)))
# 11

np.dot(vector1, vector2)
# 11
It indicates whether the vectors point in the same direction.

The result is influenced by the magnitudes of the vectors.

For instance, consider the dot product of two pairs of vectors:

In both cases, the vectors are collinear, but the result differs significantly based on magnitude (e.g., 2 vs. 20).

Cosine Similarity

Cosine similarity is widely used to compare vectors by measuring the cosine of the angle between them. This is calculated as the dot product of two vectors divided by the product of their magnitudes (or norms). It essentially reflects how closely aligned two vectors are in their direction.




To compute cosine similarity, you can either implement it manually or use libraries like sklearn. Note that sklearn's cosine_similarity function requires the input to be in 2D arrays, so reshaping NumPy arrays may be necessary.

dot_product = sum(list(map(lambda x, y: x*y, vector1, vector2)))
norm_vector1 = sum(list(map(lambda x: x ** 2, vector1))) ** 0.5
norm_vector2 = sum(list(map(lambda x: x ** 2, vector2))) ** 0.5

dot_product/norm_vector1/norm_vector2

# 0.8575

from sklearn.metrics.pairwise import cosine_similarity

cosine_similarity(
  np.array(vector1).reshape(1, -1), 
  np.array(vector2).reshape(1, -1))[0][0]

# 0.8575
The closer two vectors are in direction, the higher the cosine similarity. For example, if two vectors form an angle of approximately 30 degrees, they are considered quite similar. It’s also possible to determine the exact angle between vectors in degrees.


import math
math.degrees(math.acos(0.8575))

# 30.96
Choosing the Right Metric
When comparing embeddings, selecting the appropriate distance metric depends on the task. Here's a breakdown:

Interpreting Metrics:
Both Euclidean distance (L2) and cosine similarity can identify relationships, such as how objects within the same cluster are closer to one another than objects in different clusters. However, they are interpreted differently:

L2 distance: Smaller values signify closer objects.

Cosine similarity: Higher values indicate closer objects.







Why Use Cosine Similarity for NLP?
For NLP tasks, cosine similarity is often the preferred choice due to several advantages:

Its values are bounded between -1 and 1, making it easier to interpret than unbounded metrics like L1 or L2.

Calculating cosine similarity is computationally efficient, as it relies on dot products rather than square roots.

It performs better in high-dimensional spaces and is less impacted by the “curse of dimensionality.”

Understanding the Curse of Dimensionality
As the dimensionality of vectors increases, the distances between them tend to converge, reducing the ability to distinguish meaningful patterns. This phenomenon, known as the "curse of dimensionality," can complicate analysis.

For example, OpenAI embeddings demonstrate that higher dimensions lead to narrower distance distributions. Cosine similarity helps mitigate this issue, as it is less sensitive to dimensionality compared to other metrics.

Interestingly, OpenAI embeddings are pre-normalized, meaning that their dot product and cosine similarity produce identical results.




The Value of Visualization
While understanding similarities theoretically is essential, visualizing embeddings can provide additional clarity. Visual exploration often helps uncover structures and relationships in the data, making it a great first step before diving into practical applications.

Visualizing Embeddings
To better understand embeddings, visualizing them is a great approach. However, since embeddings often have a high number of dimensions (e.g., 1536), directly visualizing them is impractical. Dimensionality reduction techniques can help by projecting these vectors into a 2D space, making visualization possible.

PCA (Principal Component Analysis)
One of the simplest techniques for dimensionality reduction is PCA. Here’s how it works:

Convert the embeddings into a 2D NumPy array to make it compatible with sklearn.

import numpy as np
embeddings_array = np.array(df.embedding.values.tolist())
print(embeddings_array.shape)
# (1400, 1536)
Initialize a PCA model with n_components = 2 (since we want a 2D visualization).

from sklearn.decomposition import PCA

pca_model = PCA(n_components = 2)
pca_model.fit(embeddings_array)

pca_embeddings_values = pca_model.transform(embeddings_array)
print(pca_embeddings_values.shape)
# (1400, 2)
Train the PCA model on the data and transform the embeddings to obtain their 2D representation.

Once transformed, the embeddings can be plotted on a scatter plot.




The result shows that questions from the same topic are closely grouped, which is promising. However, the clusters overlap significantly, leaving room for improvement.

t-SNE (t-Distributed Stochastic Neighbor Embedding)
PCA is a linear method, and it may struggle to separate clusters effectively if the relationships between embeddings are non-linear. To address this, t-SNE, a non-linear dimensionality reduction algorithm, can be used for better visualization.

The process for applying t-SNE is similar to PCA, except we replace the PCA model with a t-SNE model.

from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2, random_state=42)
tsne_embeddings_values = tsne_model.fit_transform(embeddings_array)

fig = px.scatter(
    x = tsne_embeddings_values[:,0], 
    y = tsne_embeddings_values[:,1],
    color = df.topic.values,
    hover_name = df.full_text.values,
    title = 't-SNE embeddings', width = 800, height = 600,
    color_discrete_sequence = plotly.colors.qualitative.Alphabet_r
)

fig.update_layout(
    xaxis_title = 'first component', 
    yaxis_title = 'second component')
fig.show()



The results from t-SNE are much clearer. Most clusters are well-separated, though some overlap remains — particularly between topics like "genai," "datascience," and "ai." This overlap is understandable, as these topics are semantically related.

Exploring 3D Projections
For additional insights, embeddings can also be projected into three-dimensional space. While this may not always be practical, it can provide a more engaging and interactive way to explore the data’s structure.

tsne_model_3d = TSNE(n_components=3, random_state=42)
tsne_3d_embeddings_values = tsne_model_3d.fit_transform(embeddings_array)

fig = px.scatter_3d(
    x = tsne_3d_embeddings_values[:,0], 
    y = tsne_3d_embeddings_values[:,1],
    z = tsne_3d_embeddings_values[:,2],
    color = df.topic.values,
    hover_name = df.full_text.values,
    title = 't-SNE embeddings', width = 800, height = 600,
    color_discrete_sequence = plotly.colors.qualitative.Alphabet_r,
    opacity = 0.7
)
fig.update_layout(xaxis_title = 'first component', yaxis_title = 'second component')
fig.show()