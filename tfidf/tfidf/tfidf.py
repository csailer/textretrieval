import math
from corpus import corpus
import operator
from textblob.utils import lowerstrip
from collections import defaultdict
class TFIDF():

    def __init__(self):
        pass
    
    def raw_frequency_tf(self, word, doc):
        '''Count how many times the word parameter appears in the doc parameter to come up with the "Raw Frequency"'''
        return doc.words.count(word)

    def log_scaled_tf(self, word, doc):
        '''logarithmically scaled frequency'''    
        raw_frequency = self.raw_frequency_tf(word,doc)
        if(raw_frequency == 0):
            return 0
        else: 
            return 1 + math.log(raw_frequency)


    def augmented_frequency_tf(self, word, doc):
        '''augmented frequency, to prevent a bias towards longer documents, e.g. 
            raw frequency divided by the maximum raw frequency of any term in the document'''
        all_tfs = self.word_counts(doc)
        max_t = max(all_tfs.iteritems(), key=operator.itemgetter(1))[0]
        max_tf = doc.words.count(max_t)
        bias = 0.5 
        atf = bias + (bias * self.raw_frequency_tf(word, doc))/max_tf
        return atf
    
    def docs_containing(self, word, corpus):
        '''Count the number of documents, within the corpus, which contain the word parameter'''
        return sum(1 for doc in corpus if word in doc)
    
    
    def inverse_frequency_idf(self, word, corpus):
        '''Computes the Inverse Document Frequency'''
        #Number of documents in the corpus
        document_count = len(corpus) 
        #Number of documents in corpus containing the term (Wiki states that it is common to adjust this value by +1 to
        #(avoid a division by zero - doesn't that skew the results? I don't know, so if its zero I return zero)
        word_count = self.docs_containing(word, corpus)
        if word_count > 0:
            return  math.log(document_count / word_count)
        else:
            return 0

    def inverse_frequency_smooth_idf(self, word, corpus):
        '''This function applies a smoothing component to the IDF: log(1 + N/t)'''
        N = len(corpus)
        t = self.docs_containing(word, corpus)
        idf = math.log(1 + N/t)
        return idf

   

    #Computes TF-IDF
    def tfidf(self, word, doc, corpus, tf_type='aug_freq', idf_type='inv_idf'):
        tf = 0
        if tf_type=='aug_freq':
            tf = self.augmented_frequency_tf(word, doc)
        elif tf_type=='raw_freq':
            tf = self.raw_frequency_tf(word, doc)
        elif tf_type=='log_freq':
            tf = self.log_scaled_tf(word, doc)
         
        idf = 0
        if idf_type == 'inv_idf':
            idf = self.inverse_frequency_idf(word, corpus)
        elif idf_type == 'inv_smooth_idf':
            idf = self.inverse_frequency_smooth_idf(word, corpus)
       
         
        return tf * idf

    def word_counts(self, doc):
        """Dictionary of word frequencies in this text.
        """
        counts = defaultdict(int)
        stripped_words = [lowerstrip(word) for word in doc.words]
        for word in stripped_words:
            counts[word] += 1
        return counts



