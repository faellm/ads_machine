import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Carregando o conjunto de dados a partir do arquivo CSV
df = pd.read_csv("./treinamento_alunos.csv")

# Convertendo a coluna alvo em formato numérico
df['Original_NU_NOTA_REDACAO'] = pd.to_numeric(df['Original_NU_NOTA_REDACAO'])

# Separar features numéricas e categóricas
numeric_features = df.select_dtypes(include=['float64', 'int64'])
categorical_features = df.select_dtypes(include=['object'])

# Aplicar normalização às features numéricas
scaler = StandardScaler()
numeric_features_scaled = scaler.fit_transform(numeric_features)

# Aplicar codificação one-hot às features categóricas
encoder = OneHotEncoder()
categorical_features_encoded = encoder.fit_transform(categorical_features)

# Juntar as features numéricas e categóricas transformadas
X = pd.concat([pd.DataFrame(numeric_features_scaled), pd.DataFrame(categorical_features_encoded.toarray())], axis=1)

# Separando os dados em features (X) e target (y)
y = df['Original_NU_NOTA_REDACAO']  # variável que queremos prever

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)

# Criando e treinando o modelo Gradient Boosting
gb_model = GradientBoostingRegressor()
gb_model.fit(X_train, y_train)

# Fazendo previsões no conjunto de teste
y_pred = gb_model.predict(X_test)

# Avaliando o desempenho do modelo
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
