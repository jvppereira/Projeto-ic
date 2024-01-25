import cx_Oracle
from flask import Flask, jsonify, render_template
import json
from flask_cors import CORS
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

cx_Oracle.init_oracle_client(lib_dir=r"C:\instantclient_21_12")
app = Flask(__name__)
CORS(app)



# -------------------- FUNCAO PARA CONECTAR NO BANCO --------------------
def connect_to_database():
    try:
        #conexao com o bacno, obviamente vai ser outra coisa aqui
        conn = cx_Oracle.connect(user='tasy', password='aloisk', dsn='(DESCRIPTION=(ADDRESS=(PROTOCOL=TCP)(HOST=172.16.149.139)(PORT=1521))(CONNECT_DATA=(SERVICE_NAME=pdbcli03)))')
        return conn
    except Exception as e:
        print(f"Erro na conexão com o banco de dados: {e}")
        return None



# -------------------- FUNCAO PARA ATUALIZAR A BASE A SER TREINADA PELO MODELO COM OS NOVOS PARAMETROS --------------------
def atualizajson(parametro, coluna, base):
    base[coluna] = base[coluna] * parametro
    return base




# -------------------- FUNCAO PARA EXECUTAR A QUERY --------------------
def execute_query(conn, query_type):
    try:
        #INSERE OS DADOS
        cursor = conn.cursor()
        if query_type == "inserirdados":
            sql = 'INSERT'
        #BUSCA OS DADOS
        elif query_type == "buscardados":
            sql = 'SELECT'


        cursor.execute(sql)
        columns = [col[0] for col in cursor.description]
        result = [dict(zip(columns, row)) for row in cursor.fetchall()]
        return result
    except Exception as e:
        print(f"Erro na execução da query: {e}")
        return None
    finally:
        cursor.close()



# -------------------- ROTA PARA CHAMAR UMA QUERY --------------------
@app.route('/api/query_result/<query_type>', methods=['GET'])
def query_result(query_type):
    connection = connect_to_database()
    if connection:
        result = execute_query(connection, query_type)
        if result:
              return jsonify(result)
    return jsonify({'error': 'Falha na execução da query'})



#ROTA PARA TREINAR O MODELO
@app.route('/api/treinarmodelo', methods=['GET'])
def treinarmodelo():
    connection = connect_to_database()
    if connection:
        result = execute_query(connection, "buscardados")
        if result:
                dados = jsonify(result)
                
                # -----------------------PEGANDO OS DADOS -----------------------

                df = pd.json_normalize(dados)

                # ----------------------- TRATANDO OS DADOS -----------------------
                #colunas vazias
                df = df.drop('Valor de Compra', axis=1)
                df = df.drop('Complemento', axis=1)

                #transformando em float
                df['Valor de Venda'] = df['Valor de Venda'].str.replace(',','.').astype(float)


                #transformando todas as colunas em numerica e fazendo categorias
                label_encoder = LabelEncoder()

                colunas_com_strings = df.select_dtypes(include=['object']).columns.difference(['Valor de Venda'])

                # Itera sobre as colunas com strings
                for coluna in colunas_com_strings:
                    df[coluna] = label_encoder.fit_transform(df[coluna])



                # ----------------------- TREINANDO O MODELO -----------------------

                #DEFININDO X E Y
                y = df['Valor de Venda']
                X = df.drop('Valor de Venda', axis=1)

                X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

                def avaliar_modelo(nome_modelo, y_teste, previsao):
                    r2 = r2_score(y_teste, previsao)
                    RSME = np.sqrt(mean_squared_error(y_teste, previsao))
                    return f'Modelo {nome_modelo}:\nR2:{r2:.2%} (Quanto maior melhor \nRSME:{RSME:.2%} (Quanto menor melhor)'


                modelo = RandomForestRegressor()

                #TREINAR MODELO ESCOLHIDO

                #treino
                modelo.fit(X_train, y_train)

                #teste
                previsao = modelo.predict(X_test)
                print(avaliar_modelo('RandomForest', y_test, previsao))


                #return modelo treinado





#ROTA PARA USAR O MODELO
@app.route('/api/treinarmodelo', methods=['POST'])
def treinarmodelo():

    #RECEBE DADOS
    connection = connect_to_database()
    if connection:
        result = execute_query(connection, "buscardados")
        if result:
            dados = jsonify(result)

    df = pd.json_normalize(result)
    dadosatualizados = df
    
    #recebe um json com todos os atributos que serao alterados, entao precisamos percorrer o dataset e em cada coluna alterar o valor baseado no que foi passado
      # Obtém os dados JSON da solicitação
    
    dados_json = request.get_json()
  
    # Verifica se o JSON contém os parâmetros necessários

    if 'parametro1' in dados_json:
        parametro1 = dados_json['parametro1']
        print(parametro1)
        dadosatualizados = atualizajson(parametro1,'parametro1', dadosatualizados)

    if 'parametro2' in dados_json:
        parametro2 = dados_json['parametro2']
        dadosatualizados = atualizajson(parametro2,'parametro2', dadosatualizados)

    if 'parametro3' in dados_json:
        parametro3 = dados_json['parametro3']
        dadosatualizados = atualizajson(parametro3,'parametro3', dadosatualizados)

    if 'parametro4' in dados_json:
        parametro4 = dados_json['parametro4']
        dadosatualizados = atualizajson(parametro4,'parametro4', dadosatualizados)

    #muda os parametros de acordo com o que o usuario passou
    df = pd.json_normalize(dadosatualizados)
  

    #MANDA PRO MODELO
    novos_dados = df

    # Fazendo previsões usando o modelo treinado
    previsoes_novos_dados = modelo.predict(novos_dados)

    # Exibindo as previsões
    return previsoes_novos_dados



if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

