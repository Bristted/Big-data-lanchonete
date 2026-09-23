# =====================================================================
# CLASSIFICADOR DE VENDAS - LANCHONETE PONTO CERTO DO LARGO
# =====================================================================
# O QUE ESTE PROGRAMA FAZ:
# 1. Lê o extrato de vendas da maquininha Stone (arquivo CSV).
# 2. Limpa os dados (datas, valores em reais, formas de pagamento).
# 3. Cria informações úteis a partir de cada venda (dia da semana, hora etc.).
# 4. Separa cada venda em uma FAIXA DE TICKET: baixo, medio ou alto.
# 5. Treina dois modelos de Machine Learning (Árvore de Decisão e SVM) para
#    PREVER a faixa de ticket de uma venda a partir do dia, da hora e da
#    forma de pagamento.
# 6. Mostra um menu para ver o desempenho dos modelos, desenhar a árvore,
#    classificar uma venda nova e ver gráficos das vendas.
#
# "Ticket" = valor gasto em uma venda. "Ticket médio" = média desses valores.
# =====================================================================


# ---------------------------------------------------------------------
# IMPORTAÇÃO DAS BIBLIOTECAS
# Bibliotecas são pacotes de código prontos que a gente reaproveita.
# ---------------------------------------------------------------------

# pandas: trabalha com tabelas (como uma planilha do Excel dentro do Python).
# O apelido 'pd' é uma convenção para escrever menos.
import pandas as pd

# matplotlib: faz gráficos. 'plt' é o apelido padrão.
import matplotlib.pyplot as plt

# rich: deixa o texto no terminal mais bonito (cores, caixas, negrito).
from rich import print            # substitui o print normal por um que aceita cores
from rich.console import Console  # "tela" onde desenhamos as caixas
from rich.panel import Panel      # caixa com borda em volta do texto

# scikit-learn (sklearn): a principal biblioteca de Machine Learning em Python.
from sklearn import svm                                           # modelo SVM
from sklearn.metrics import accuracy_score, classification_report # medem o quanto o modelo acerta
from sklearn.model_selection import train_test_split              # divide os dados em treino e teste
from sklearn.pipeline import make_pipeline                        # encadeia etapas (normalizar + modelo)
from sklearn.preprocessing import StandardScaler                  # coloca os números na mesma escala
from sklearn.tree import DecisionTreeClassifier, plot_tree        # Árvore de Decisão e o desenho dela


# =====================================================================
# CONFIGURAÇÃO
# Tudo o que pode mudar de um extrato para outro fica aqui em cima,
# para não precisar mexer no resto do código.
# =====================================================================

# Onde está o extrato da Stone e onde salvar o resultado (no Google Drive, pelo Colab).
CAMINHO_EXTRATO = "/content/drive/MyDrive/extrato_stone.csv"
CAMINHO_SAIDA = "/content/drive/MyDrive/extrato_stone_com_classificacoes.csv"

# Como o arquivo CSV foi gravado.
SEPARADOR = ";"          # caractere que separa as colunas (no Brasil costuma ser ";")
CODIFICACAO = "utf-8"    # tipo de codificação do texto; se os acentos vierem errados, use "latin-1"

# Nomes das colunas no extrato. PRECISAM ser iguais aos do seu arquivo.
COL_DATA = "Data da venda"   # coluna com a data (ou data e hora juntas)
COL_HORA = None              # coluna só com a hora, se vier separada; se não existir, deixe None
COL_VALOR = "Valor bruto"    # coluna com o valor da venda
COL_PAGAMENTO = "Tipo"       # coluna com a forma de pagamento (Pix, Débito, Crédito, Voucher)

# Limites que definem as faixas de ticket (em reais).
LIMITE_BAIXO = 10.0   # até R$10       -> "baixo" (ex.: 1 salgado ou 1 hambúrguer)
LIMITE_MEDIO = 20.0   # de R$10 a R$20 -> "medio"; acima de R$20 -> "alto"

# Listas fixas usadas em vários pontos do programa.
FORMAS_PAGAMENTO = ["pix", "debito", "credito", "voucher"]
DIAS_SEMANA = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

# COLUNAS = as "pistas" que o modelo usa para prever a faixa de ticket.
# No Machine Learning elas se chamam FEATURES (características).
# A última parte cria automaticamente: pag_pix, pag_debito, pag_credito, pag_voucher.
COLUNAS = ["dia_semana", "hora", "fim_de_semana", "dia_mes"] + [f"pag_{f}" for f in FORMAS_PAGAMENTO]

# Cria a "tela" do rich onde vamos desenhar as caixas.
caixa = Console()

# ---------------------------------------------------------------------
# VARIÁVEIS GLOBAIS
# Ficam fora das funções para que todas as funções possam usá-las.
# Começam vazias (None) e só recebem valor dentro de preparar_dados().
# ---------------------------------------------------------------------
df = None                                        # a tabela com todas as vendas
clf = None                                       # o modelo Árvore de Decisão treinado
svm_model = None                                 # o modelo SVM treinado
X_train, X_test, y_train, y_test = None, None, None, None  # dados de treino e de teste


# =====================================================================
# PARTE 1 - LIMPEZA DOS DADOS
# O extrato vem "do jeito do banco". Aqui a gente transforma em dados
# que o computador consegue calcular.
# =====================================================================

def limpar_valor(serie):
    # Recebe a coluna de valores e transforma texto em número.
    # Exemplo: "R$ 1.234,56" vira 1234.56   |   "8,00" vira 8.0
    # (No Brasil a vírgula é decimal; no Python o decimal é ponto.)

    # Transforma tudo em texto, tira o "R$" e os espaços das pontas.
    s = serie.astype(str).str.replace("R$", "", regex=False).str.strip()

    # Só mexe nos valores que usam vírgula (formato brasileiro).
    tem_virgula = s.str.contains(",", regex=False)
    s[tem_virgula] = (s[tem_virgula]
                      .str.replace(".", "", regex=False)    # tira o ponto de milhar: 1.234,56 -> 1234,56
                      .str.replace(",", ".", regex=False))  # troca vírgula por ponto: 1234,56 -> 1234.56

    # Converte para número. Se algum valor não der para converter, vira "vazio" (NaN)
    # em vez de travar o programa (errors="coerce").
    return pd.to_numeric(s, errors="coerce")


def normalizar_pagamento(texto):
    # O extrato pode escrever a forma de pagamento de vários jeitos
    # ("Crédito à vista", "CREDITO", "Débito Visa"...).
    # Esta função padroniza tudo em só 4 nomes: pix, debito, credito, voucher.
    t = str(texto).lower()   # deixa tudo minúsculo para facilitar a comparação
    if "pix" in t:
        return "pix"
    if "déb" in t or "deb" in t:
        return "debito"
    if "créd" in t or "cred" in t:
        return "credito"
    if "voucher" in t or "vale" in t or "refei" in t or "alimenta" in t:
        return "voucher"     # vale-refeição / vale-alimentação
    return "outro"           # qualquer coisa que não se encaixe acima


def adicionar_features(tabela):
    # Cria as colunas que o modelo usa (as FEATURES), a partir da data/hora
    # e da forma de pagamento de cada venda.

    # .dt dá acesso às partes de uma data.
    tabela["dia_semana"] = tabela["data_hora"].dt.dayofweek   # 0 = segunda, 1 = terça ... 6 = domingo
    tabela["hora"] = tabela["data_hora"].dt.hour              # hora da venda (0 a 23)
    tabela["fim_de_semana"] = (tabela["dia_semana"] >= 5).astype(int)  # 1 se sábado/domingo, 0 se não
    tabela["dia_mes"] = tabela["data_hora"].dt.day            # dia do mês (1 a 31) - ajuda a ver efeito de salário no início do mês

    # O modelo só entende números, não palavras como "pix".
    # Então criamos uma coluna para cada forma de pagamento com 1 (foi essa) ou 0 (não foi).
    # Essa técnica se chama ONE-HOT ENCODING.
    # Ex.: venda no pix -> pag_pix=1, pag_debito=0, pag_credito=0, pag_voucher=0
    for forma in FORMAS_PAGAMENTO:
        tabela[f"pag_{forma}"] = (tabela["pagamento"] == forma).astype(int)
    return tabela


def carregar_dados():
    # Lê o extrato, limpa e devolve a tabela pronta para o modelo.

    # Lê o arquivo CSV e coloca numa tabela do pandas (DataFrame).
    tabela = pd.read_csv(CAMINHO_EXTRATO, sep=SEPARADOR, encoding=CODIFICACAO)

    # Se data e hora vierem em colunas separadas, junta as duas num texto só.
    if COL_HORA:
        texto_data = tabela[COL_DATA].astype(str) + " " + tabela[COL_HORA].astype(str)
    else:
        texto_data = tabela[COL_DATA].astype(str)

    # Converte o texto em data de verdade.
    # dayfirst=True avisa que o formato é brasileiro (dia/mês/ano).
    # errors="coerce": datas inválidas viram vazio em vez de travar o programa.
    tabela["data_hora"] = pd.to_datetime(texto_data, dayfirst=True, errors="coerce")

    # Cria colunas limpas de valor e de forma de pagamento usando as funções acima.
    tabela["valor"] = limpar_valor(tabela[COL_VALOR])
    tabela["pagamento"] = tabela[COL_PAGAMENTO].apply(normalizar_pagamento)  # aplica a função linha a linha

    # Remove linhas problemáticas:
    antes = len(tabela)                                  # guarda quantas linhas tinha
    tabela = tabela.dropna(subset=["data_hora", "valor"])  # tira linhas com data ou valor vazio
    tabela = tabela[tabela["valor"] > 0].copy()          # tira estornos/cancelamentos (valor zero ou negativo)
    print(f"[dim]{len(tabela)} vendas válidas ({antes - len(tabela)} linhas descartadas)[/dim]")

    # Cria as features (dia da semana, hora etc.).
    tabela = adicionar_features(tabela)

    # Cria a coluna ALVO: aquilo que o modelo vai aprender a prever.
    # pd.cut divide os valores em intervalos:
    #   de 0 até 10   -> "baixo"
    #   de 10 até 20  -> "medio"
    #   acima de 20   -> "alto"
    tabela["faixa_ticket"] = pd.cut(
        tabela["valor"],
        bins=[0, LIMITE_BAIXO, LIMITE_MEDIO, float("inf")],  # float("inf") = infinito (sem limite de cima)
        labels=["baixo", "medio", "alto"],
    ).astype(str)

    return tabela


# =====================================================================
# PARTE 2 - TREINO DOS MODELOS
# =====================================================================

def preparar_dados():
    # 'global' avisa que vamos alterar as variáveis globais lá de cima,
    # e não criar variáveis novas só dentro desta função.
    global df, X_train, X_test, y_train, y_test, clf, svm_model

    # Carrega e limpa o extrato.
    df = carregar_dados()

    # X (maiúsculo) = as features, ou seja, as pistas para prever.
    # y (minúsculo) = o alvo, ou seja, a resposta certa (faixa de ticket).
    X = df[COLUNAS]
    y = df["faixa_ticket"]

    # Divide os dados em dois grupos:
    #   TREINO (70%): o modelo aprende com esses exemplos.
    #   TESTE  (30%): o modelo NUNCA viu esses; servem para medir se ele aprendeu de verdade
    #                 ou só decorou (como uma prova com questões novas).
    # random_state=1: a divisão é sempre igual, então o resultado é repetível.
    # stratify: mantém a mesma proporção de baixo/medio/alto nos dois grupos.
    #           (Só funciona se cada faixa tiver pelo menos 2 vendas; senão, fica sem.)
    estratificar = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=estratificar
    )

    # ÁRVORE DE DECISÃO: aprende uma sequência de perguntas do tipo "sim ou não"
    # (ex.: "a hora é antes das 10h30?", "é fim de semana?") até chegar numa faixa.
    # max_depth=4: no máximo 4 perguntas em sequência. Isso evita que a árvore
    # "decore" os dados de treino (OVERFITTING) e deixa o desenho legível.
    clf = DecisionTreeClassifier(max_depth=4, random_state=1)
    clf.fit(X_train, y_train)   # .fit = treinar (o modelo aprende com os dados de treino)

    # SVM (Support Vector Machine): tenta traçar "linhas" (fronteiras) que separam
    # as faixas de ticket da melhor forma possível. kernel="linear" = fronteiras retas.
    # O SVM é sensível à escala dos números (hora vai até 23, pag_pix só 0 ou 1),
    # então o StandardScaler coloca tudo na mesma escala antes.
    # make_pipeline junta as duas etapas: primeiro normaliza, depois o SVM.
    svm_model = make_pipeline(StandardScaler(), svm.SVC(kernel="linear"))
    svm_model.fit(X_train, y_train)

    # Usa os dois modelos para classificar TODAS as vendas e guarda o resultado na tabela.
    df["Classificacao_Arvore"] = clf.predict(X)       # .predict = prever
    df["Classificacao_SVM"] = svm_model.predict(X)

    # Marca em cada linha se a venda foi usada no treino ou no teste.
    # Importante: nas vendas de treino o modelo "já viu a resposta",
    # então só as de teste mostram o desempenho real.
    df["conjunto"] = "teste"
    df.loc[X_train.index, "conjunto"] = "treino"

    # Salva a tabela completa num novo CSV. index=False evita uma coluna extra com números de linha.
    df.to_csv(CAMINHO_SAIDA, index=False)
    print(f"[dim]Arquivo salvo em {CAMINHO_SAIDA}[/dim]")


# =====================================================================
# PARTE 3 - FUNÇÕES DE APOIO AO MENU
# =====================================================================

def ler_int(mensagem, minimo, maximo):
    # Pede um número inteiro ao usuário e só aceita se estiver entre minimo e maximo.
    # Se a pessoa digitar letra ou número fora do intervalo, pede de novo
    # em vez de o programa travar.
    while True:                        # repete até receber um valor válido
        try:
            valor = int(input(mensagem))   # int() transforma o texto digitado em número
            if minimo <= valor <= maximo:
                return valor               # valor válido: sai da função devolvendo o número
        except ValueError:
            pass                           # digitou algo que não é número: ignora e pede de novo
        print(f"[red]Digite um número entre {minimo} e {maximo}.[/red]")


def ler_nova_venda():
    # Pergunta os dados de uma venda nova e monta uma tabela de 1 linha
    # exatamente no mesmo formato que o modelo usou no treino.

    # Mostra os dias numerados (0 - Segunda, 1 - Terça...) e pede a escolha.
    for i, nome in enumerate(DIAS_SEMANA):   # enumerate dá o número (i) e o nome de cada item
        print(f"{i} - {nome}")
    dia = ler_int("Dia da semana: ", 0, 6)
    hora = ler_int("Hora (0-23): ", 0, 23)
    dia_mes = ler_int("Dia do mês (1-31): ", 1, 31)

    # Mostra as formas de pagamento numeradas a partir de 1 e pega a escolhida.
    for i, forma in enumerate(FORMAS_PAGAMENTO, start=1):
        print(f"{i} - {forma}")
    # O "- 1" é porque a lista começa na posição 0, mas o menu começa no 1.
    forma = FORMAS_PAGAMENTO[ler_int("Forma de pagamento: ", 1, len(FORMAS_PAGAMENTO)) - 1]

    # Monta um dicionário (pares nome: valor) com as features da venda.
    venda = {
        "dia_semana": dia,
        "hora": hora,
        "fim_de_semana": int(dia >= 5),   # calcula sozinho: 1 se sábado/domingo
        "dia_mes": dia_mes,
    }
    # Mesmo one-hot do treino: 1 na forma escolhida, 0 nas outras.
    for f in FORMAS_PAGAMENTO:
        venda[f"pag_{f}"] = int(f == forma)

    # Transforma em tabela e garante as colunas na mesma ordem do treino.
    return pd.DataFrame([venda])[COLUNAS]


def mostrar_desempenho(modelo):
    # Mede o quanto o modelo acerta usando SÓ os dados de teste (que ele nunca viu).

    y_pred = modelo.predict(X_test)             # o que o modelo previu
    acuracia = accuracy_score(y_test, y_pred)   # % de previsões iguais à resposta certa
    caixa.print(Panel(f"[green]Acurácia: {acuracia*100:.2f}%[/green]"))  # :.2f = 2 casas decimais

    # Relatório por faixa. A acurácia sozinha pode enganar, então vemos cada faixa:
    #   precision (precisão): quando o modelo disse "alto", quantas vezes era alto mesmo?
    #   recall (revocação):   de todas as vendas que eram "alto", quantas ele encontrou?
    #   f1-score:             média equilibrada entre precision e recall
    #   support:              quantas vendas daquela faixa havia no teste
    # zero_division=0 evita aviso quando o modelo nunca prevê alguma faixa.
    print(classification_report(y_test, y_pred, zero_division=0))

    # Mostra quantas vendas há em cada faixa no extrato inteiro.
    print("[dim]Distribuição das faixas no extrato:[/dim]")
    print(df["faixa_ticket"].value_counts().to_string())


# =====================================================================
# PARTE 4 - GRÁFICOS DAS VENDAS
# Análise descritiva: mostra o que aconteceu nas vendas, sem previsão.
# =====================================================================

def reais(valor):
    # Formata um número como dinheiro brasileiro: 41868.5 -> "R$ 41.868,50"
    texto = f"{valor:,.2f}"                    # formato americano: 41,868.50
    texto = texto.replace(",", "X").replace(".", ",").replace("X", ".")  # troca vírgula e ponto
    return f"R$ {texto}"


def mostrar_graficos():
    # Cria uma figura com 4 gráficos (2 linhas x 2 colunas).
    # 'eixos' guarda
