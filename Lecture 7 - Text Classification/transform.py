"""
(C) 2017 Nikolay Manchev
[London Machine Learning Study Group](http://www.meetup.com/London-Machine-Learning-Study-Group/members/)

This work is licensed under the Creative Commons Attribution 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by/4.0/.
"""

import math
import os
import pandas as pd
import numpy as np

import scipy.sparse
import scipy.io

import nltk.data
import nltk.tokenize
import nltk.stem

from nltk.corpus import stopwords

from collections import Counter

import numpy as np

def extract_words(text, stemmer = None, remove_stopwords = False):
    """
    Extracts all words from a document. The document is first tokenized,
    morphological affixes from words are removed, and stop words
    are excluded from the resulting list of words.
    
    Parameters
    ----------
    text             : input document (String)
    stemmer          : NLTK stemmer for the stemming process. Must be an NLTK 
                       stem package class. E.g:
                    
                       nltk.stem.porter.PorterStemmer()
                       nltk.stem.lancaster.LancasterStemmer()
                       nltk.stem.snowball.EnglishStemmer()
                      
                       If set to None, no stemming is performed on the input text
    remove_stopwords : If set to True, removes any stop words from the output,
                       using the nltk.corpus.stopwords corpus (English)
    
    Returns
    -------
    A list of words extracted from the input text.
    
    """

    # Get the stopwords corpus
    if "stopwords" not in os.listdir(nltk.data.find("corpora")):
        nltk.download("stopwords")

    # Tokenize the document
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    
    if stemmer is None:
        # No stemmer? Just convert to lower case.
        words = [token.lower() for token in tokens]
    else:
        # Apply stemming    
        words = [stemmer.stem(word.lower()) for word in tokens]
        
    
    # Remove stop words
    if remove_stopwords:
        words = [word for word in words if word not in stopwords.words('english')]
    
    return words
    
def build_vocabulary(documents):
    """
    Builds a vocabulary based on all documents in the corpus.
    
    Parameters
    ----------
    documents : document corpus
    
    Returns
    -------
    A list, containing all unique words from the corpus
    
    """    
    vocabulary = set()
    
    # Iterate over each document in the corpus
    for doc in documents:
        # Iterate over all words in the current document and
        # add each word to the vocabulary set
        vocabulary.update([word for word in doc])
        
    # Convert the vocabulary to list
    vocabulary = list(vocabulary)
    
    return vocabulary


def get_idfs_dict(vocabulary, documents):
    """
    Gets a dictionary containing the vocabulary and their respective IDFs.
    This method is used for debug purposes only.
    
    Parameters
    ----------
    vocabulary : vocabulary of the corpus
    documents  : all documents in the corpus
    
    Returns
    -------
    A dictionary in the form of {(word1, word1_IDF), (word2, word2_IDF), ... }
    
    """

    # Get number of documents where each word from the vocabulary appears
    counts = Counter()

    # Iterate over the vocabulary and count the occurrence of each word
    for word in vocabulary:
        for doc in documents:
            if word in doc:
                counts[word] += 1
    
    # Get the number of documents in the corpus
    number_of_docs = len(documents)
    
    # Create an empty dictionary
    idfs = dict()

    # Iterate over the counts
    for term in list(counts.items()):
        
        # Normalise the count by the number of documents, and take the log
        # Add the (word, IDF) pair to the dictionary
        idfs[term[0]] = math.log(number_of_docs / term[1], 2)
    
    return idfs


def get_idfs(vocabulary, documents):
    """
    Gets a sparse diagonal matrix containing the IDFs for all words in the
    vocabulary. The IDF are computed as the logarithmically scaled inverse 
    fraction of the documents that contain the word, obtained by dividing 
    the total number of documents by the number of documents containing the 
    term, and then taking the logarithm of that quotient.
    
    Parameters
    ----------
    vocabulary : vocabulary of the corpus
    documents  : all documents in the corpus
    
    Returns
    -------
    A diagonal matrix of size len(vocabulary) x len(vocabulary), where the
    word's IDFs are located on the main diagonal, and all other elements in
    the matrix are 0. Eg:
    
    index in the vocabulary    0      1      2      3     ...     N
                               1 idf(word1)  0      0     ...     0  
                               2      0  idf(word2) 0     ...     0  
                               3      0      0 idf(word3) ...     0
                             ...     ...    ...    ...    ...    ...
                               N      0       0      0     ... idf(wordN)
                  
    where N = len(vocabulary)
    
    """

    # Get number of documents where each word from the vocabulary appears
    counts = dict()

    for word in vocabulary:
        for doc in documents:
            if word in doc:
                if word in counts:
                  counts[word] += 1
                else:
                  counts[word] = 1
    
    # Compute inverse document frequency
    number_of_docs = len(documents)
    
    # Create a list to hold all the IDFs
    idfs = []
    
    # Iterate over the counts
    for word in vocabulary:
        
        # Normalise the count by the number of documents, and take the log
        # Add the value to the list of IDFs
        idfs.append(math.log(number_of_docs / counts[word], 2))

    # Create a sparse diagonal matrix with the values from IDFs list located
    # on the main diagonal
    idf_matrix = scipy.sparse.diags(np.squeeze(np.asarray(idfs)))

    return idf_matrix    

def get_tf_vectors(vocabulary, documents):    
    """
    Computes the term frequency vectors for all documents. This method uses
    raw count of a term in a document, i.e. the number of times that term 
    t occurs in document d.
    
    Parameters
    ----------
    vocabulary : vocabulary of the corpus
    documents  : all documents in the corpus
    
    Returns
    -------
    A sparse matrix of size len(documents) x len(vocabulary), containing the
    raw counts for each term. Entries in the matrix can be viewed using
    the print_sparse_row(matrix, row_index) method. Ex:

    tf_matrix.shape
    (6918, 1869)

    print_sparse_row(tf_matrix,0)
    col[106] 1
    col[289] 1
    col[482] 1
    col[815] 1
    col[1074] 1
    col[1145] 1
    col[1232] 1
    col[1565] 1    
    """

    # Document / sparse matrix row index
    row_index = 0
    
    # Values and indices for the sparse matrix
    rows = []
    cols = []
    values = []

    # Iterate over all documents in the corpus
    for doc in documents:        
        col_index = 0
        
        # Iterate over all words in the vocabulary
        for word in vocabulary:
            
            # Current word in current document?
            if word in doc:
                # Increase the term frequency for this word
                rows.append(row_index)
                cols.append(col_index)
                values.append(doc.count(word))
                col_index += 1
            else:
                # Move to the next word in the vocabulary
                col_index += 1
                
        # Move to the next document
        row_index += 1
    
    # Compose a sparse matrix of size len(documents) x len(vocabulary) with
    # all term frequencies
    tf_matrix = scipy.sparse.csr_matrix((values, (rows, cols)), shape=(row_index, len(vocabulary)))
    
    return tf_matrix

def print_sparse_row(matrix, row_index):
    """
    Prints the indices and their respective values for a sparse matrix row.
    This method is used for debugging purposes.    
    
    Ex:

    print_sparse_row(tf_matrix,0)
    col[106] 1
    col[289] 1
    col[482] 1
    col[815] 1
    col[1074] 1
    col[1145] 1
    col[1232] 1
    col[1565] 1    
    
    Parameters
    ----------
    matrix     : a sparse matrix
    row_index  : index of a row from the sparse matrix
    
    Returns
    -------
    
    """
    # Convert the row of interest to a Numpy array
    row = np.asarray(matrix[row_index].todense()).flatten()
    
    # Iterate over all columns of the row
    col = 0
    for el in row:
        if el != 0:
            # Print the column index and the respective value
            print("col[%i] %s"%(col, el))
        col += 1


def print_tfidf(matrix, row_index, idfs):
    """
    For a given row from a TF matrix, this method prints a table containing
    all words, their term frequency, IDF, and TFxIDF values. Ex:
    
    >>> idfs = get_idfs_dict(vocabulary, dataDF["Words"])
    >>> print_tfidf(tf_matrix, 0, idfs)

    Column   Word      TF      IDF                 TFxIDF              
    ------   ----      --      ---                 ------              
    106      is        1       2.2355206178166482  2.23552061782       
    289      just      1       4.616587945974135   4.61658794597       
    482      the       1       1.448369266482225   1.44836926648       
    815      vinc      1       1.8033980511864398  1.80339805119       
    1074     da        1       1.8033980511864398  1.80339805119       
    1145     book      1       5.434211203485566   5.43421120349       
    1232     code      1       1.801942987986053   1.80194298799       
    1565     awesom    1       2.6320179865437403  2.63201798654  

    Parameters
    ----------
    matrix     : matrix containg document term frequencies (see the 
                 get_tf_vectors method)
    row_index  : index of a row from the TF matrix
    
    Returns
    -------
    """    
    
    # Get the row of interest as a Numpy array
    row = np.asarray(matrix[row_index].todense()).flatten()
    col = 0
    
    # Set the output header
    output = [["Column", "Word", "TF", "IDF", "TFxIDF"],
              ["------", "----", "--", "---", "------"]]
    
    # Go over each element of the row (i.e. word from the document)
    for el in row:
        if el != 0:
          # Append the column index, the word, and the TF, IDF, and TFxIDF
          # values to the output
          output.append([str(col), vocabulary[col], str(el), str(idfs[vocabulary[col]]), 
                         str(idfs[vocabulary[col]]*el)])
        col += 1
                
    # Print the output as a table
    col_width = max(len(word) for row in output for word in row) + 2  # padding
    for row in output:
      print("".join(word.ljust(col_width) for word in row))

def l2_normalized_matrix(matrix):
    """
    Normalises a sparse matrix by scaling its rows individually to L2 unit norm

    The new row values are computed as
    
        ||x|| = sqrt(sum(x^2))
        
    For efficiency, the resulting new matrix is formed by computing
    
    normalized_matrix = 
        transpose(transpose transpose(matrix) * l2_norm)
        
    where matrix is the original sparse matrix and l2_norm is diagonal 
    matrix of the reciprocals of sqrt(sum(x^2))
    
    Parameters
    ----------
    matrix     : a sparse matrix to be normalized
    
    Returns
    -------
    An L2 normalised sparse matrix based on the input matrix
    
    """     
    # Compute the L2 norms
    l2_norm = np.sqrt(matrix.power(2).sum(axis=1))
    
    # Get the reciprocals
    with np.errstate(divide="ignore", invalid="ignore"):
        l2_norm = np.reciprocal(l2_norm)
        # Treat infinity and NaN as 0
        l2_norm[~np.isfinite(l2_norm)] = 0  # -inf inf NaN   
    
    # Form a diagonal matrix of the reciprocals
    l2_norm = scipy.sparse.diags(np.squeeze(np.asarray(l2_norm)))           
        
    # Compute the normalised matrix
    normalized_matrix = (matrix.T * l2_norm).T
    
    return normalized_matrix
       
def mtx_save(file_name, matrix):
    """
    Writes a sparse matrix a to Matrix Market file-like target.
    
    Parameters
    ----------
    file_name : target file name
    matrix    : a sparse matrix
    
    Returns
    -------
    
    """
    scipy.io.mmwrite(file_name, matrix)

def encode_labels(labelsDF):
    """
    Encodes a string set of target classes to a Numpy array of label indices
    
    Parameters
    ----------
    labelsDF : a Pandas DataFrame or Numpy array containing the labels
    
    Returns
    -------
    An encoded Numpy array
    
    Ex:
    
    >>> A = np.array(["a", "a", "b", "a"])
    >>> encode_labels(A)
        array([0, 0, 1, 0], dtype=int8)
        
    """    
    # Factorize the labels
    labelsDF = pd.Categorical(labelsDF)
    catLabelsDF = labelsDF.codes

    return catLabelsDF

def labels_save(file_name, labels):
    """
    Saves the target class labels to an external file.
    
    Parameters
    ----------
    file_name : target file name
    labels    : a Numpy array containing the labels
    
    Returns
    -------
        
    """       
    labels.tofile(file_name, sep='\n')

def hash_vectors(tf_idf_matrix, vocabulary, N=8000):
    """
    Applies feature hashing / hashing trick to a sparse matrix. This method
    turns features into indices in a vector or matrix. It works by applying a 
    hash function to the features and using their hash values as indices 
    directly, rather than looking the indices up in an associative array.
    
    Parameters
    ----------
    tf_idf_matrix : a sparse matrix of TFxIDF values
    vocabulary    : vocabulary of the corpus
    N             : size of the hased vector
    
    Returns
    -------
    A sparse matrix of size tf_idf_matrix.shape[0] x N, containing the hashed
    features
    
    >>> tf_idf_matrix.shape
        (6918, 1869)

    >>> hash_vectors(tf_idf_matrix, vocabulary, 100).shape
        (6918, 100)
            
    """       
    
    # Make sure the input is a csr_matrix (wee need to access the sparse
    # matrix elements directly )
    if not isinstance(tf_idf_matrix, scipy.sparse.csr.csr_matrix):
      print("WARN: Input %s is not a Compressed Sparse Row matrix. Converting...")      
      tf_idf_matrix = tf_idf_matrix.tocsr()


    row_count = tf_idf_matrix.shape[0]
  
    hashed_rows = []
    hashed_cols = []
    hashed_data = []
  
    # Iterate over the matrix rows
    for row_index in range(row_count):
      
      # Get the current row indices
      row = tf_idf_matrix.getrow(row_index)
      col_indices = row.indices
      
      # Iterate over the columns
      for col_index in range(len(col_indices)):
          # Get the word and its corresponding TFxIDF value
          tf_idf_value = tf_idf_matrix[row_index, col_indices[col_index]]
          word = vocabulary[col_indices[col_index]]
          
          # Apply a hash function h to the features (e.g., words), then use 
          # the hash values directly as feature indices and update the
          # resulting vector at those indices

          h = hash(word)
          hashed_rows.append(row_index)
          hashed_cols.append(h % N)
          hashed_data.append(tf_idf_value)
      
    # Create a new sparse matrix with the hashed features
    hashed_features_matrix = scipy.sparse.csr_matrix((hashed_data, 
                                                     (hashed_rows, hashed_cols)), 
                                                   shape=(row_count, N))
                                       
    return hashed_features_matrix                                


# Read a data set
dataDF = pd.read_csv("data/SMSSpamCollection", 
                     sep='\t', lineterminator='\n', names = ["Label", "Text"])

# Initialise a stemmer

#porter = nltk.stem.porter.PorterStemmer()
#lancaster = nltk.stem.lancaster.LancasterStemmer()
snowball = nltk.stem.snowball.EnglishStemmer()

# Apply stemming
print("Stemming...")
dataDF["Words"] = dataDF.apply(lambda row: extract_words(row['Text'], snowball), axis=1)

# Remove empty rows. Messages like ":)" which will get removed by the stemmer
dataDF = dataDF[dataDF.astype(str)["Words"] != '[]']
dataDF = dataDF.reset_index(drop=True)

# Build a vocabulary
print("Building vocabulary...")
vocabulary = build_vocabulary(dataDF["Words"])

# Get the TF vectors
print("Forming the TF matrix...")
tf_matrix = get_tf_vectors(vocabulary, dataDF["Words"])

# Get the IDF matrix
print("Forming the IDF matrix...")
idf_matrix = get_idfs(vocabulary, dataDF["Words"])

# Compute the TFxIDF values
print("Computing the TFxIDF matrix...")
tf_idf_matrix = (tf_matrix * idf_matrix)
tf_idf_matrix = l2_normalized_matrix(tf_idf_matrix)

#tf_idf_matrix = hash_vectors(tf_idf_matrix, vocabulary, 125)

# Encode the labels
print("Encoding labels...")
labels = encode_labels(dataDF["Label"])

# Save the TFxIDF matrix and the corresponding values
print("Saving features and labels...")
mtx_save("data/training.mtx", tf_idf_matrix)
labels_save("data/labels.csv", labels)

print("all done!")
