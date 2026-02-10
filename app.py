import streamlit as st
import pandas as pd
import pickle

st.write("🚀 Streamlit está rodando!")
# carregar pipeline
with open("pipeline_titanic.pkl", "rb") as f:
    pipeline = pickle.load(f)

st.title("🚢 Simulador de Sobrevivência do Titanic")

st.write("Preencha os dados do passageiro fictício:")

# inputs
pclass = st.selectbox("Classe", [1, 2, 3])
sex = st.selectbox("Sexo", ["male", "female"])
age = st.slider("Idade", 0, 80, 25)
sibsp = st.number_input("Irmãos/Cônjuges a bordo (SibSp)", 0, 10, 0)
parch = st.number_input("Pais/Filhos a bordo (Parch)", 0, 10, 0)
fare = st.number_input("Tarifa paga (Fare)", 0.0, 600.0, 30.0)
embarked = st.selectbox("Porto de embarque", ["C", "Q", "S"])

# botão
if st.button("🔮 Prever sobrevivência"):
    passageiro = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }])

    predicao = pipeline.predict(passageiro)[0]
    proba = pipeline.predict_proba(passageiro)[0][1]

    st.subheader("Resultado")

    if predicao == 1:
        st.success("✅ O passageiro SOBREVIVERIA")
    else:
        st.error("❌ O passageiro NÃO sobreviveria")

    st.write(f"**Probabilidade de sobreviver:** {proba*100:.1f}%")
