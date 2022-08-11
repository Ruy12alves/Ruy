# Prevendo a Ocorrência de Câncer

## Etapa 1 - Coletando os dados
# http://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Winsconsin+%28Diagnostic%29

# Os dados do câncer da mama incluem 569 observações de biópsias de câncer,
# cada um com 32 características (variáveis). Uma característica é um número de 
# identificação (ID), outro é o diagnóstico de câncer, e 30 são medidas laboratoriais 
# numéricas. O diagnóstico é codificado com "M" para indicar maligno ou "B" para 
# indicar benigno.

dados <- read.csv(file.choose())
str(dados)
head(dados)

## Etapa 2 - Explorando os Dados
# Excluindo a coluna ID
# Independentemente do método de aprendizagem de máquina, deve sempre ser excluída
# variáveis de ID. Caso contrário, isso pode levar a resultados errados porque o ID
# pode ser usado para unicamente "prever" cada exemplo. Por conseguinte, um modelo
# que inclui um identificador pode sofrer de superajuste, e será muito difícil usá-lo para
# generalizar outros dados.

dados <- dados[-1]
str(dados)
any(is.na(dados))

# Muitos classificadores requerem que as variáveis sejam do tipo Fator
table(dados$diagnosis)
dados$diagnosis <- factor(dados$diagnosis, levels = c("B", "M"), labels = c("Benigno", "Maligno"))
str(dados$diagnosis)

# Verificando a proporção
round(prop.table(table(dados$diagnosis)) * 100, digits = 1)

# Medidas de Tendência Central
# Determinamos aqui um problema de escala entre os dados, que então precisam ser normalizados
# O cálculo de distância feito pelo kNN é dependente das medidas de escala nos dados de entrada.
summary(dados[c("radius_mean", "area_mean", "smoothness_mean")])

# Criando um função de normalização
normalizar <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

# Testando a função de normalização - os resultados devem ser idênticos
normalizar(c(1,2,3,4,5))
normalizar(c(10,20,30,40,50))

# Normalizando os dados
dados_norm <- as.data.frame(lapply(dados[2:31], normalizar))

# Confirmando que a normalização funcionou
summary(dados[c("radius_mean", "area_mean", "smoothness_mean")])
summary(dados_norm[c("radius_mean", "area_mean", "smoothness_mean")])

## Etapa 3: Treinando o modelo

# Carregando o pacote library
install.packages("class")
library(class)
?knn

# Criando dados de treino e dados de teste
dados_treino <- dados_norm[1:469, ]
dados_teste <- dados_norm[470:569, ]

# Criando os labels para os dados de treino e de teste
dados_treino_labels <- dados[1:469, 1]
dados_teste_labels <- dados[470:569, 1]

# Criando o modelo
modelo <- knn(train = dados_treino,
              test = dados_teste,
              cl = dados_treino_labels,
              k = 21)

# A função knn() retorna um objeto do tipo fator com as previsões para classe
class(modelo)

## Etapa 4: Avaliando e Interpretando o Modelo

# Carregando o gmodels
install.packages("gmodels")
library(gmodels)

# Criando uma tabela cruzada dos dados previstos x dados atuais
CrossTable(x = dados_teste_labels, y = modelo, prop.chisq = FALSE)

# Interpretando os Resultados
# A tabela cruzada mostra 4 possíveis valores, que representam os falsos 
# A primeira coluna lista os labels originais nos dados observados
# As duas colunas do modelo (Benigno e Maligno) do modelo, mostram os 
# Temos:
# Cenário 1: Célula Benigno (label) X Benigno (Modelo) - 61 casos - true negative (o modelo acerta)
# Cenário 2: Célula Benigno (label) x Maligno (Modelo) - 0 casos - false positive (o modelo acerta)
# Cenário 3: Célula Maligno (label) x Benigno (Modelo) - 2 casos - false negative (o modelo errou)
# Cenário 4: Célula Maligno (label) x Maligno (Modelo) - 37 casos - true positive (o modelo acerta)

# Taxa de acerto do Modelo 98% (acertou  98 em 100)
 
# Consulte a definição de confusion matrix em caso

## Etapa 5: Otimizando a performance do modelo
?scale()
dados_z <- as.data.frame(scale(dados[-1]))

# Confirmando transformação realizada com sucesso
summary(dados_z$area_mean)

# Criando novos datasets de treino e de teste
dados_treino <- dados_z[1:469, ]
dados_teste <- dados_z[470:569, ]

dados_treino_labels <- dados[ 1: 469, 1]
dados_teste_labels <- dados[ 470: 569, 1]

# Reclassificando
modelo_v2 <- knn(train = dados_treino,
                 test = dados_teste,
                 cl = dados_treino_labels,
                 k = 21)

# Criando uma tabela cruzadas dos dados previstos x dados atuais
CrossTable(x = dados_teste_labels, y = modelo_v2, prop.chisq = FALSE)

# Testando diferentes valores para k
# Criando dados de treino e dados de teste
dados_treino <- dados_norm[1:469, ]
dados_teste <- dados_norm[470:569, ]

# Criando os labels para os dados de treino e de teste
dados_treino_labels <- dados[ 1: 469, 1]
dados_teste_labels <- dados[ 470: 569, 1]

# Valores diferentes para k
modelo_v3 <- knn(train = dados_treino,
                 test = dados_teste,
                 cl = dados_treino_labels,
                 k = 1)
CrossTable(x = dados_teste_labels, y = modelo_v3, prop.chisq = FALSE)

modelo_v4 <- knn(train = dados_treino,
                 test = dados_teste,
                 cl = dados_treino_labels,
                 k = 5)
CrossTable(x = dados_teste_labels, y = modelo_v4, prop.chisq = FALSE)

modelo_v5 <- knn(train = dados_treino,
                 test = dados_teste,
                 cl = dados_treino_labels,
                 k = 11)
CrossTable(x = dados_teste_labels, y = modelo_v5, prop.chisq = FALSE)

modelo_v6 <- knn(train = dados_treino,
                 test = dados_teste,
                 cl = dados_treino_labels,
                 k = 15)
CrossTable(x = dados_teste_labels, y = modelo_v6, prop.chisq = FALSE)

modelo_v7 <- knn(train = dados_treino,
                 test = dados_teste,
                 cl = dados_treino_labels,
                 k = 27)
CrossTable(x = dados_teste_labels, y = modelo_v7, prop.chisq = FALSE)

modelo_v2 <- knn(train = dados_treino,
                 test = dados_teste,
                 cl = dados_treino_labels,
                 k = 21)
CrossTable(x = dados_teste_labels, y = modelo_v2, prop.chisq = FALSE)


## Calculando a taxa de erro
prev = NULL
taxa_erro = NULL

for (i in 1:20) {
  set.seed(101)
  prev = knn(train = dados_treino, test = dados_teste, cl = dados_treino_labels, k = i)
  taxa_erro[i] = mean(dados$diagnosis != prev)
}

# Obtendo os valores de k e das taxas de erro
library(ggplot2)
k.values <- 1:20
df_erro <- data.frame(taxa_erro, k.values)
df_erro

# Á medida que aumentamos k, diminuímos a taxa de erro do modelo
ggplot(df_erro, aes(x = k.values, y = taxa_erro)) +
  geom_point() + geom_line(lty = "dotted", color = "red")

# Quando maior o k.values, menor é a taxa_erro














