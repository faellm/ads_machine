# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Carregando o conjunto de dados a partir do arquivo CSV
df = pd.read_csv('C:\Users\2021206415\Documents\fael\treinamento_alunos.csv')

# Separando os dados em features (X) e target (y)
X = df.drop(columns=['Original_NU_NOTA_REDACAO'])  # previsões
y = df['Original_NU_NOTA_REDACAO']  # variável que queremos prever

# Dividindo os dados em conjuntos de treinamento e teste

# (train_test_split) = nos ajuda a dividir nossos dados em conjuntos de treinamento e teste
# (X_train): Conjunto de features para treinamento.
# (X_test:) Conjunto de features para teste.
# (y_train): Conjunto de targets correspondentes ao treinamento.
# (y_test:) Conjunto de targets correspondentes ao teste.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) test_size=0.2, random_state=42)

# Criando e treinando o modelo Gradient Boosting
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = gb_model.predict(X_test)

# Avaliando o desempenho do modelo
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
