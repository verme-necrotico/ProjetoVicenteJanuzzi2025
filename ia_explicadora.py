##############################################################################################################
##############################################################################################################
######################   INSTRUÇÕES      #####################################################################
### A interface do projeto está sendo feita em Streamlit, ou seja, para executar o projeto, vá até o terminal  
### e rode o comando "streamlit run ia_explicadora.py"                                                         
###                                                                                                            
### Certifique-se de que você tem todas as bibliotecas instaladas (streamlit, sklearn, pandas e requests)      
### Caso não possua alguma, o Streamlit irá avisar mostrando um erro de "sintaxe desconhecida" ou algo to tipo 
### Para instalar alguma biblioteca, abra seu terminal e digite pip install {nome da biblioteca} e aguarde até tudo
### ser instalado corretamente :)
###
##############################################################################################################
##############################################################################################################

import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from streamlit_extras.stylable_container import stylable_container
import streamlit.components.v1 as components





# Função para ler o arquivo CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Carrega o CSS logo no início
load_css("styles.css")


# dados iniciais de treino (fase 1A)
frases = [
    "adoro",
    "amo",
    "maravilhoso",
    "incrível",
    "feliz",
    "bom",
    "bem",
    "gosto",
    "horrível",
    "odeio",
    "ruim",
    "péssimo",
    "triste",
    "mau",
    "mal"

]
sentimentos = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]  # 1 = positivo, 0 = negativo
# treinamento dessa porra ai (etapa 1)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(frases)
modelo = MultinomialNB()
modelo.fit(X, sentimentos)



# controle de navegaçao (pode ignorar)
if 'etapa' not in st.session_state:
    st.session_state.etapa = 1


# botao para avançar
def avancar_etapa():
    st.session_state.etapa += 1



##########################################################################################################################################################################################################################################################################
##########=== Etapa 1A ===
##########################################################################################################################################################################################################################################################################
if st.session_state.etapa == 1:
    
    #container
    with stylable_container(
        key="container1A1",
        css_styles="""
            {
                background-color: rgba(255, 255, 255, 0);
                backdrop-filter: blur(2px);
                -webkit-backdrop-filter: blur(2px);
                border: 2px solid #FF4B4B;
                border-radius: 10px;
                padding: 25px;
            }
            """
    ):

        # resumo da etapa
        st.header("1️⃣ Fase 1A: Análise de Sentimento")
        st.markdown("""
        Nesta fase, eu recebo uma **frase do usuário**, transformo as palavras em **números vetoriais** com base no que aprendi durante o treinamento e então **classifico** se o sentimento é positivo ou negativo.

        A técnica usada é chamada de **classificação de texto**, usando um modelo simples chamado `Naive Bayes`, treinado com frases que representam sentimentos conhecidos.
        """)

        #input do usuario
        entrada1 = st.text_input("Digite uma frase para análise de sentimento:", key="etapa1")

        #coisas que acontecem depois do input do usuario
        if entrada1:

                entrada_vetorizada = vectorizer.transform([entrada1])
                pred = modelo.predict(entrada_vetorizada)[0]
                prob = modelo.predict_proba(entrada_vetorizada)[0]
                palavras_input = entrada1.lower().split()
                vocabulario = vectorizer.get_feature_names_out()
                palavras_reconhecidas = [p for p in palavras_input if p in vocabulario]

                if pred == 1:
                    st.success(f"🌞 Resultado: Frase positiva ({prob[1]*100:.1f}% de confiança)")
                else:
                    st.error(f"🌧️ Resultado: Frase negativa ({prob[0]*100:.1f}% de confiança)")


                #######################################
                ####Explicação do processo
                #######################################
                if st.toggle("👀 **Veja como eu fiz isso:**", key="exet1"):
                    st.write("### **📚Etapas do Processo:**")
                    st.write("- Primeiro eu **dividi** sua frase em **palavras**. Isso está representado no esquema a seguir:", palavras_input)
                    st.write("- Depois, **transformei suas palavras em números** para serem melhor compreendidas por mim. Esse processo é chamado", "**Vetorização**.")
                    st.write("- Após **vetorizar**, eu comparei as palavras com as quais eu conheço, e baseado na quantidade de palavras do seu **input** que se assemelham com as palavras que eu conheço, eu irei retornar a probabilidade da sua frase ser **Positiva** ou **Negativa:**")
                    st.write("✅ Palavras reconhecidas:", palavras_reconhecidas if palavras_reconhecidas else "Nenhuma")
                    st.markdown("""
                    Palavras não reconhecidas são ignoradas, pois não fazem parte do treinamento original. Isso limita a minha compreensão, o que demonstra a importância de ampliar o vocabulário com novos exemplos.
                    """)
                        
    # botao para 1B    
    if st.button("👉 Que tal agora com Machine Learning?", key="1B"):
        st.session_state.etapa = 2


##########################################################################################################################################################################################################################################################################
##########=== Etapa 1B ===
##########################################################################################################################################################################################################################################################################
elif st.session_state.etapa == 2:

     #container
    with stylable_container(
        key="container1A1",
        css_styles="""
            {
                background-color: rgba(255, 255, 255, 0);
                backdrop-filter: blur(2px);
                -webkit-backdrop-filter: blur(2px);
                border: 2px solid #FF4B4B;
                border-radius: 10px;
                padding: 25px;
            }
            """
    ):

        # resumo da etapa
        st.header("1️⃣ Fase 1B: Análise de Sentimento (Avançado)")
        st.write("Agora, ao invés de analisar cada palavra individualmente, eu posso entender o **contexto inteiro da frase**, e após ser treinado com uma **base maior e mais variada de dados**, eu posso dizer se sua frase é negativa ou positiva **mesmo que ela não apareça diretamente no meu banco de dados.**")
        st.write("O nome desse processo de prever as características de uma informação, mesmo que eu não a conheça, é chamado de `Regressão Logística`. É atualmente o processo mais utilizado por grandes IA's, como o ChatGPT e o Gemini.")

        # nerdices blablabla
        import pandas as pd
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics.pairwise import cosine_similarity

        # dataset ampliado
        dados = pd.read_csv("treino_ia.csv")

        pipeline = Pipeline([
            ('vect', CountVectorizer()),
            ('clf', LogisticRegression())
        ])

        pipeline.fit(dados['frase'], dados['sentimento'])
        
        # criar pipeline
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer_transicao = CountVectorizer()
        X_base = vectorizer_transicao.fit_transform(dados['frase'])


        # nem mexe aqui
        modelo_transicao = LogisticRegression()
        modelo_transicao.fit(X_base, dados['sentimento'])

        #input do usuario
        entrada_alt = st.text_input("Escreva uma frase para ser analisada:", key="etapa_1b")
        
        # coisas que acontecem depois do input
        if entrada_alt:
            entrada_vetorizada = vectorizer_transicao.transform([entrada_alt])
            pred_alt = pipeline.predict([entrada_alt])[0]
            prob_alt = pipeline.predict_proba([entrada_alt])[0]

            if pred_alt == 1:
                st.success(f"🌟 Resultado: Frase positiva ({prob_alt[1]*100:.1f}% de confiança)")
            else:
                st.error(f"⚠️ Resultado: Frase negativa ({prob_alt[0]*100:.1f}% de confiança)")
        

            #######################################
            ####Explicação do processo
            #######################################
            if st.toggle("👀 **Veja como eu fiz isso:**"):
                st.write("### **📚Etapas do Processo:**")
                st.write("- Agora, eu fui treinado com uma base de dados maior, ou seja, eu conheço mais frases e consigo ter mais precisão nas minhas respostas.")
                st.write("- Após receber sua frase, eu a comparei com algumas das quais eu conheço. Veja quais são:")

                # Similaridade com frases do treino
                similaridades = cosine_similarity(entrada_vetorizada, X_base)[0]
                top_indices = similaridades.argsort()[::-1][:3]  # Top 3 mais similares
                st.markdown("""
                **Frases semelhantes usadas no aprendizado:**
                A IA encontrou frases no treinamento com estrutura semelhante. Isso ajuda a justificar sua decisão:
                """)
                for i in top_indices:
                    frase = dados.iloc[i]['frase']
                    sim = similaridades[i]
                    st.markdown(f"🔹 **{frase}**  ")
                    st.progress(sim)

    # botoes 
    if st.button("➡ Ir para Fase 2"):
        st.session_state.etapa = 3
    if st.button("⬅ Voltar para Fase 1A"):
        st.session_state.etapa = 1


##########################################################################################################################################################################################################################################################################
##########=== Etapa 2 ===
##########################################################################################################################################################################################################################################################################
elif st.session_state.etapa == 3:

    # container
    with stylable_container(
        key="container1A1",
        css_styles="""
            {
                background-color: rgba(255, 255, 255, 0);
                backdrop-filter: blur(2px);
                -webkit-backdrop-filter: blur(2px);
                border: 2px solid #FF4B4B;
                border-radius: 10px;
                padding: 25px;
            }
            """
    ):
        # resumo etapa2
        st.header("2️⃣ Fase 2: Extração de Informações (Entidades)")
        st.markdown("""
        Nesta etapa, eu tento encontrar **informações específicas** dentro da frase digitada, como **nomes próprios**, **lugares** e **datas**.

        Esse processo é chamado de `Reconhecimento de Entidades Nomeadas (NER)`. Ele é usado por assistentes virtuais e buscadores para identificar elementos importantes em um texto.
        """)

        # input usuario
        entrada2 = st.text_input("Digite uma frase para extrair informações:", key="etapa2")

        # blabla
        if entrada2:
            import re

            # detecção de entidades (possiveis alterações)
            palavras = entrada2.split()
            locais = [p for p in palavras if p.istitle() and p.lower() not in ["eu", "estou"]]
            datas = re.findall(r"\b\d{4}\b", entrada2)

            

            st.write("🔍 **Entidades detectadas:**")
            if locais:
                st.write(f"📍 Local identificado: {', '.join(locais)}")
            if datas:
                st.write(f"📅 Ano(s) identificado(s): {', '.join(datas)}")
            if not locais and not datas:
                st.write("Nenhuma entidade específica detectada.")

            #######################################
            ####Explicação do processo
            #######################################
            if st.toggle("**👀 Veja como eu fiz isso:**", key="exet2"):
                st.write("### **📚Etapas do Processo:**")
                st.write("•  Primeiro eu **dividi** sua frase em **palavras**. Isso está representado no esquema a seguir:", palavras)
                st.write("•  Depois, procurei por palavras que comecem com **letra maiúscula**, indicando **nomes próprios** de lugares (ex: Barra da Tijuca).")
                st.write("•  E então, procurei por numerais que possuem 4 dígitos, indicando **anos** (ex: 2025).")
                st.write("✅ Informações de locais extraídas:", locais if locais else "Nenhum local identificado")
                st.write("✅ Informações de datas extraídas:", datas if datas else "Nenhuma data identificada")
                st.markdown("""
                Lorem ipsum dolor sit amet
                """)

            # explicação contextual
                st.write("Em aplicações reais, esse processo pode ser feito com modelos mais robustos que analisam o **contexto**, como os usados em **assistentes virtuais** e sistemas de busca.")

    # botao 😭
    if st.button("➡ Ir para Fase 3"):
        st.session_state.etapa = 4
    if st.button("⬅ Voltar para Fase 1B"):
        st.session_state.etapa = 2

##########################################################################################################################################################################################################################################################################
##########=== Etapa 3 ===
##########################################################################################################################################################################################################################################################################
elif st.session_state.etapa == 4:

    # container
    with stylable_container(
        key="container1A1",
        css_styles="""
            {
                background-color: rgba(255, 255, 255, 0);
                backdrop-filter: blur(2px);
                -webkit-backdrop-filter: blur(2px);
                border: 2px solid #FF4B4B;
                border-radius: 10px;
                padding: 25px;
            }
            """
    ):
        # mano nao
        st.header("3️⃣ Fase 3: Acesso a Dados Externos (Clima em tempo real)")

        st.markdown("""
        Neste etapa, eu busco certas informações em `API's`, as quais são primordiais para o funcionamento de processos como **Previsão de Tempo**, amplamente utilizado por **assistentes virtuais** como **Siri**, **Alexa** e **Google Assistant**
        """)

        # input
        cidade = st.text_input("Digite o nome da cidade para saber o clima atual:", key="cidade_tempo")

        # coisas que acontecem depois do input
        if cidade:
            # nem mexe nisso aqui
            import requests
            api_key = "12e5db3b5754e99d8cf3e0bf62e8e8e9"
            url = f"http://api.openweathermap.org/data/2.5/weather?q={cidade}&appid={api_key}&units=metric&lang=pt_br"
            hyperlink = "https://openweathermap.org/weathermap"

            # pronto agora pode mexer
            try:
                response = requests.get(url)
                dados = response.json()
                if response.status_code == 200:
                    clima = dados["weather"][0]["description"].capitalize()
                    temp = dados["main"]["temp"]
                    st.success(f"🌦️ Tempo em {cidade}: {clima}, {temp}°C.")

                    if st.toggle("**👀 Veja como eu fiz isso:**", key="exet3"):
                        st.write("### **📚Etapas do Processo:**")
                        st.write("•  Primeiro, eu recebi o nome da cidade que você digitou, neste caso: ",cidade)
                        st.write("•  Depois, peguei o nome da sua cidade e através da API do site [Open Weather](%s) eu recebi as informações referentes à cidade." % hyperlink)
                        st.write("•  Finalmente, exibi as informações recebidas. :)")
                        st.write("Isso demonstra a habilidade da IA em se conectar com **fontes de dados externas** em tempo real.")
                        
                else:
                    st.error("❌ Cidade não encontrada ou erro na API.")
            except:
                st.error("❌ Erro ao acessar a API. Verifique sua conexão e a chave de API.")

    # botao
    if st.button("➡ Ir para Fase 4"):
         st.session_state.etapa = 5

    if st.button("⬅ Voltar para Fase 2"):
         st.session_state.etapa = 3

##########################################################################################################################################################################################################################################################################
##########=== Etapa 4 ===
##########################################################################################################################################################################################################################################################################
elif st.session_state.etapa == 5:
   
    ############################# aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    # container
    with stylable_container(
        key="container1A1",
        css_styles="""
            {
                background-color: rgba(255, 255, 255, 0);
                backdrop-filter: blur(2px);
                -webkit-backdrop-filter: blur(2px);
                border: 2px solid #FF4B4B;
                border-radius: 10px;
                width: 1120px;
                
                padding: 30px;
                margin: auto;

            }
            """
    ):
        st.set_page_config(layout="wide")
        st.header("4️⃣ Fase 4: O que aprendemos até agora?")
        st.subheader("🧠 Após observar os processos pelos quais uma IA percorre para executar suas principais funções, é possível acrescentar uma série de reflexões:")

        html_code = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');

        .grid-container {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            padding: 30px;
            font-family: 'Inter', sans-serif;
        }

        .card {
            background-color: transparent;
            backdrop-filter: blur(2px);
            color: white;
            border-radius: 12px;
            padding: 15px;
            text-align: center;
            transition: height 0.3s ease, background-color 0.4s ease;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            position: relative;
            overflow: hidden;
            height: 130px;
            max-height: 250px;
            border: 2px solid #FF4B4B;
            border-radius: 10px;
        }

        .card .info {
            opacity: 0;
            max-height: 0;
            transition: opacity 0.3s ease, max-height 0.4s ease;
            margin-top: 0px;
            font-size: 16px;
            overflow: hidden;
        }

        .card:hover {
            height: 300px;
            max-height: 600px;
        }

        .card:hover .info {
            opacity: 1;
            max-height: 300px;
            margin-top: 20px;
        }
        </style>

        <div class="grid-container">
            <div class="card">
                <h3>🤖 O nome "Inteligência Artificial" reflete a verdadeira natureza das IA's?</h3>
                <div class="info">• IA's até o presente momento, não possuem qualquer vestígio de autoconsciência, portanto não são capazes de gerar conclusões que não tenham sido a elas direcionadas previamente.</div>
            </div>
            <div class="card">
                <h3>💵 Quando o desenvolvimento de grandes IA's passa a ser somente uma ferramenta na mão de um grupo minoritário?</h3>
                <div class="info">• A forma como IA's são programadas permitem que certos tipos de informações sejam extraídas e armazenadas. Também servem como um meio de comunicação entre o grupo que a controla e os usuários que a utilizam.</div>
            </div>
            <div class="card">
                <h3>🦾 Inteligência Artificial: uma ferramenta ou um substituto humano?</h3>
                <div class="info">• Conforme os processos das IA's se tornaram cada vez mais complexos, a indagação de um futuro onde funções antes exercidas por humanos passem a ser exercidas apenas por máquinas começou a ser mais constante.</div>
            </div>
            <div class="card">
                <h3>🌍 Como o mau desenvolvimento de grandes tecnologias está acelerando o colapso entre humanidade e meio ambiente?</h3>
                <div class="info">• Com o avanço da tecnologia, mais recursos naturais são necessários para a manutenção das máquinas, como a queima de combustíveis fósseis que poluem a atmosfera e o uso exagerado de água limpa para resfriar máquinas.</div>
            </div>
        </div>
        """

        components.html(html_code, height=600, width=1000)


    # fim






















