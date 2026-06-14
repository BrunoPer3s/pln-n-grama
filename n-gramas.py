import re
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline

nltk.download('punkt')
nltk.download('punkt_tab')

generator = pipeline("text-generation", model="TucanoBR/Tucano-160m")

def leitura(nome):
    with open(nome, 'r', encoding='utf-8') as arq:
        texto = arq.read()
    return texto

def limpar(lista):
    lixo = '.,:;?!"\'()[]{}\/|#$%^&*-'
    quase_limpo = [x.strip(lixo).lower() for x in lista]
    return [x for x in quase_limpo if x.isalpha() or '-' in x]

def preprocessar_texto(texto):
    sentencas = sent_tokenize(texto)
    sentencas_limpas = []
    for sentenca in sentencas:
        palavras = word_tokenize(sentenca)
        palavras_limpas = limpar(palavras)
        sentencas_limpas.append(palavras_limpas)
    
    return sentencas_limpas

corpus_base = leitura('/corpus_bruto.txt')
sentencas_limpas = preprocessar_texto(corpus_base)

sentencas = [['<s>'] + s + ['</s>'] for s in sentencas_limpas]

with open('corpus_preparado.txt', 'w', encoding='utf-8') as arq:
    for s in sentencas:
        arq.write(' '.join(s) + '\n')

with open('corpus_preparado.txt', 'r', encoding='utf-8') as arq:
    sentencas = arq.readlines()

vocab = set()
contagens = defaultdict(int)
for linha in sentencas:
    sent = linha.split()
    for palavra in sent:
        vocab |= {palavra}
        contagens[palavra] += 1

def ngramas(n, sent):
    return [tuple(sent[i:i+n]) for i in range(len(sent) - n + 1)]

unigramas = defaultdict(int)
bigramas = defaultdict(int)
trigramas = defaultdict(int)

for linha in sentencas:
    sent = linha.split()
    uni = ngramas(1, sent)
    bi = ngramas(2, sent)
    tri = ngramas(3, sent)
    for x in uni:
        unigramas[x] += 1
    for x in bi:
        bigramas[x] += 1
    for x in tri:
        trigramas[x] += 1

def prob_uni(x):
    V = len(vocab)
    N = sum(unigramas.values())
    return (unigramas[x] + 1) / (N + V)

def prob_bi(x):
    V = len(vocab)
    return (bigramas[x] + 1) / (unigramas[(x[0],)] + V)

def prob_tri(x):
    V = len(vocab)
    return (trigramas[x] + 1) / (bigramas[(x[0], x[1])] + V)

def prever_ngramas(palavra1, palavra2):
    lista_tri = [ch for ch in trigramas.keys() if ch[0] == palavra1 and ch[1] == palavra2]

    if lista_tri:
        ordem_tri = sorted(lista_tri, key=lambda x: prob_tri(x), reverse=True)
        previsoes = [ordem_tri[i][2] for i in range(min(3, len(ordem_tri)))]
        return previsoes

    lista_bi = [ch for ch in bigramas.keys() if ch[0] == palavra2]

    if lista_bi:
        ordem_bi = sorted(lista_bi, key=lambda x: prob_bi(x), reverse=True)
        previsoes = [ordem_bi[i][1] for i in range(min(3, len(ordem_bi)))]
        return previsoes

    return []

def gerar_proximas_palavras_tucano(contexto):
    """
    Gera as próximas 3 palavras mais prováveis com base no contexto fornecido usando o Tucano-160m.
    """
    completions = generator(contexto, num_return_sequences=1, max_new_tokens=10)
    
    texto_gerado = completions[0]['generated_text']
    
    palavras_geradas = texto_gerado.split()
    
    palavras_novas = palavras_geradas[len(contexto.split()):]
    
    return palavras_novas[:3]

def menu_interativo():
    print("Bem-vindo ao Menu Interativo de Previsão de Palavras!")
    print("Escolha o método de previsão:")
    print("1. Usar n-gramas (implementação original)")
    print("2. Usar Tucano-160m (modelo de linguagem)")
    print("Para sair, digite '#'.", end="\n\n")

    metodo = input("Escolha o método (1 ou 2): ").strip()

    if metodo not in ['1', '2']:
        print("Opção inválida. Saindo...")
        return

    inputs = []

    while True:
        entrada = input("Digite uma palavra: ").strip().lower()

        if entrada == "#":
            print("Saindo do menu interativo...")
            break

        inputs.append(entrada)

        if len(inputs) == 1:
            palavra1 = '<s>'
            palavra2 = inputs[0]
            contexto = inputs[0]
        else:
            palavra1 = inputs[-2]
            palavra2 = inputs[-1]
            contexto = " ".join(inputs[-2:])

        if metodo == '1':
            previsoes = prever_ngramas(palavra1, palavra2)
        else:
            previsoes = gerar_proximas_palavras_tucano(contexto)

        if previsoes:
            print(f"Próximas palavras mais prováveis: {', '.join(previsoes)}")
        else:
            print("Nenhuma previsão disponível.")

menu_interativo()