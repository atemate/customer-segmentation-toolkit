# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/03_data_analyse_product_descritions.ipynb (unless otherwise specified).

__all__ = ['compute_product_list', 'display_list_products']

# Cell
import logging
import pandas as pd
import nltk

# Cell
def _keywords_inventory(dataframe, colonne = 'Description'):
    def is_noun(pos):
        return pos[:2] == 'NN'

    stemmer = nltk.stem.SnowballStemmer("english")
    keywords_roots  = dict()  # collect the words / root
    keywords_select = dict()  # association: root <-> keyword
    category_keys   = []
    count_keywords  = dict()
    icount = 0
    for s in dataframe[colonne]:
        if pd.isnull(s): continue
        lines = s.lower()
        tokenized = nltk.word_tokenize(lines)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]

        for t in nouns:
            t = t.lower() ; racine = stemmer.stem(t)
            if racine in keywords_roots:
                keywords_roots[racine].add(t)
                count_keywords[racine] += 1
            else:
                keywords_roots[racine] = {t}
                count_keywords[racine] = 1

    for s in keywords_roots.keys():
        if len(keywords_roots[s]) > 1:
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k ; min_length = len(k)
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]

    #print("Nb of keywords in variable '{}': {}".format(colonne,len(category_keys)))
    return category_keys, keywords_roots, keywords_select, count_keywords

# Cell

def compute_product_list(df: pd.DataFrame) -> pd.DataFrame:
    df_initial = df
    df_produits = pd.DataFrame(df_initial['Description'].unique()).rename(columns = {0:'Description'})

    keywords, keywords_roots, keywords_select, count_keywords = _keywords_inventory(df_produits)

    list_products = []
    for k,v in count_keywords.items():
        #list_products.append([keywords_select[k],v])
        word = keywords_select[k]
        if word in ['pink', 'blue', 'tag', 'green', 'orange']: continue
        if len(word) < 3 or v < 13: continue
        if ('+' in word) or ('/' in word): continue
        list_products.append([word, v])
    list_products.sort(key = lambda x:x[1], reverse = True)

    liste = sorted(list_products, key = lambda x:x[1], reverse = True)
    return liste

# Cell
import matplotlib.pyplot as plt

def display_list_products(list_products):
    liste = sorted(list_products, key = lambda x:x[1], reverse = True)
    #_______________________________
    plt.rc('font', weight='normal')
    fig, ax = plt.subplots(figsize=(7, 25))
    y_axis = [i[1] for i in liste[:125]]
    x_axis = [k for k,i in enumerate(liste[:125])]
    x_label = [i[0] for i in liste[:125]]
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 13)
    plt.yticks(x_axis, x_label)
    plt.xlabel("Nb. of occurences", fontsize = 18, labelpad = 10)
    ax.barh(x_axis, y_axis, align = 'center')
    ax = plt.gca()
    ax.invert_yaxis()
    #_______________________________________________________________________________________
    plt.title("Words occurence",bbox={'facecolor':'k', 'pad':5}, color='w',fontsize = 25)
    plt.show()