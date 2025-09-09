import re
import nltk 
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer
from nltk.tokenize import word_tokenize
from collections import Counter 
import spacy
nlp = spacy.load("pt_core_news_sm")
STEMMER = RSLPStemmer()
PT_STOPWORDS = set(stopwords.words('portuguese'))
from itertools import chain
import pandas as pd


print('EXPRESSÕES REGULARES')
#REGEX 

#CPF
cpf = 'Fulana de tal, com cpf 232.867.920-04, relatou que ...'
cpf_clean = re.sub(r"\d{3}\.\d{3}\.\d{3}\W*\d{2}", ' ', cpf)
print(cpf_clean)

#email
email = 'Fulana de tal, com email mluizars9@gmail.com, relatou que...'
email_clean = re.sub(r"\b[\w._]+@[\w.]+\.\w{2,}\b", ' ', email)
print(email_clean)

#rg de todos os tipos
rg = 'Fulana de tal, portadora do rg 12.345.678-9, relatou que...'
rg2 = 'Ciclana de tal com o rg 1.234.567'
rg1_clean = re.sub(r"\d\.?\d\.?-?\d", ' ', rg )
rg2_clean = re.sub(r"\d\.?\d\.?-?\d", ' ', rg )
print(rg1_clean, '\n', rg2_clean)

print(50*'-')

print('TOKENIZAÇÃO, STOPWORDS, STEMMING E LEMMATIZAÇÃO')

#TOKENIZAÇÃO, STOPWORDS, STEMMING, LEMMATIZAÇÃO
texto = 'Com a evolução contínua da tecnologia, o processamento de linguagem natural (PLN) tornou-se uma ferramenta essencial na análise de grandes volumes de dados textuais.' \
' O PLN permite que máquinas compreendam e interpretem a linguagem humana, facilitando a extração de informações valiosas e a automação de tarefas que antes exigiam intervenção humana. ' \
'Técnicas avançadas de PLN são utilizadas em diversas aplicações, desde assistentes virtuais até sistemas de recomendação, contribuindo significativamente para a eficiência e a personalização dos serviços oferecidos. ' \
'À medida que o campo avança, novas metodologias e algoritmos são desenvolvidos para aprimorar a precisão e a eficácia do PLN, tornando-o uma área de estudo fascinante e em constante evolução.'


#tokenizar 
def tokenize(txt):
    return txt.split()


tokens = tokenize(texto)
print(tokens)

print(50*'-')

#remover stopwords
def stop_words(tk):
    no_stpwrd = []
    for tok in tk:
        if tok not in PT_STOPWORDS:
            no_stpwrd.append(tok)
    return no_stpwrd

tk_no_stp = stop_words(tokens)
print(tk_no_stp)

print(50*'-')

#lemmatização
def lemmanize(txt):
    lemmas = []
    pt = nlp(txt)
    for l in pt:
        lemmas.append(l.lemma_)
    return lemmas

print(lemmanize(texto))
print(50*'-')

#stemming
def stemming(tk):
    stemming_tok = []
    for s in tk:
        stemming_tok.append(STEMMER.stem(s))
    return stemming_tok

print(stemming(tokens))

print('FREQUÊNCIA DE PALAVRAS')

def top_freq(tokens_list, k=10):
    qnt = Counter(chain.from_iterable(tokens_list))
    return pd.DataFrame(qnt.most_common(k), columns = ['Token', 'Frequência'])

print(top_freq([tk_no_stp]))
