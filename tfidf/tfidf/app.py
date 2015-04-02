from corpus import corpus
from tfidf import TFIDF
import sys
def main():
    
    #try:
    c = corpus();
    tfidf = TFIDF()
    tf_type='aug_freq'
    idf_type='inv_smooth_idf'    
    for i, doc in enumerate(c.documents):
        cnt=0
        print("Top words in document {}".format(i + 1))
        scores = {word: tfidf.tfidf(word, doc, c.documents,tf_type, idf_type) for word in doc.words}
        sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for word, score in sorted_words[:10]:
            cnt+=1
            if(score>0):
                print("\tWord {}: {}, TF-IDF: {}".format(cnt, word, round(score, 5)))
    #except:
        #sys.exit(2);
    #    pass
    
if __name__ == "__main__":
    main()