from konlpy.tag import Mecab
from tqdm import tqdm
import re
from gensim.models.ldamodel import LdaModel
from gensim.models.callbacks import CoherenceMetric
from gensim import corpora
from gensim.models.callbacks import PerplexityMetric
import logging
import pickle
import pyLDAvis.gensim
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt


if __name__ == '__main__':
    processed_data = [sent.strip().split(",") for sent in tqdm(open('tosel_no_ad.csv', 'r', encoding='utf-8').readlines())]

    dictionary = corpora.Dictionary(processed_data)

    dictionary.filter_extremes(no_below=10, no_above=0.05)
    corpus = [dictionary.doc2bow(text) for text in processed_data]
    print('Number of unique tokens: %d' % len(dictionary))
    print('Number of documents: %d' % len(corpus))

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    perplexity_logger = PerplexityMetric(corpus=corpus, logger='shell')
    coherence_logger = CoherenceMetric(corpus=corpus, coherence="u_mass", logger='shell')
    lda_model = LdaModel(corpus, id2word=dictionary, num_topics=25, passes=30,
                         callbacks=[coherence_logger, perplexity_logger])
    topics = lda_model.print_topics(num_words=5)
    for topic in topics:
        print(topic)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_data, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score (c_v): ', coherence_lda)

    coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_data, dictionary=dictionary,
                                         coherence="u_mass")
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score (u_mass): ', coherence_lda)

    pickle.dump(corpus, open('tosel_no_ad.csv.pkl', 'wb'))
    dictionary.save('tosel_no_ad.csv.gensim')
    lda_model.save('tosel_no_ad.csv.gensim')

    lda_visualization = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary, sort_topics=False)
    pyLDAvis.save_html(lda_visualization, 'tosel_no_ad.csv.html')
    pyLDAvis.show(lda_visualization)
