import streamlit as st
import pandas as pd
import numpy as np
import os
from io import BytesIO
import boto3
from botocore.exceptions import ClientError
import plotly.graph_objects as go
import plotly.express as px
import warnings
from functools import lru_cache

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Dashboard Crédit")

# Supprimer les avertissements
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Définition de palettes de couleurs daltonisme-compatibles personnalisées ---
# Ces couleurs sont choisies pour un bon contraste et sont généralement sûres pour le daltonisme
COLOR_ACCORDE = '#1f77b4'  # Bleu distinctif
COLOR_DEFAUT = '#ff7f0e'   # Orange distinctif
COLOR_BAR_GLOBAL = '#2ca02c' # Vert pour les barres d'importance globale (si utilisé)
COLOR_GAUGE_ACCORDE = '#1f77b4' # Bleu pour la jauge
COLOR_GAUGE_DEFAUT = '#ff7f0e' # Orange pour la jauge
COLOR_CLIENT_HIGHLIGHT = 'red' # Rouge pour la position du client (clairement visible)


# Constantes et variables d'environnement
BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
S3_PREFIX_DATA = "input/"

# URLs des images SHAP statiques
LOCAL_SHAP_IMAGE_URL = "https://modele-regression-streamlit-mlflow-etude-credit.s3.eu-north-1.amazonaws.com/reports2/local-shap.png"
GLOBAL_SHAP_IMAGE_URL = "https://modele-regression-streamlit-mlflow-etude-credit.s3.eu-north-1.amazonaws.com/reports2/global-shap.png"

# Initialisation de l'état de session
if 'client_analysis_submitted' not in st.session_state:
    st.session_state['client_analysis_submitted'] = False
if 'simulation_submitted' not in st.session_state:
    st.session_state['simulation_submitted'] = False
if 'client_id_selected' not in st.session_state:
    st.session_state['client_id_selected'] = None
if 'last_client_id' not in st.session_state:
    st.session_state['last_client_id'] = None
if 'simulation_features' not in st.session_state:
    st.session_state['simulation_features'] = None
if 'selected_features_client' not in st.session_state:
    st.session_state['selected_features_client'] = []
# Ajout pour la simulation si elle n'existait pas encore
if 'selected_features_sim' not in st.session_state:
    st.session_state['selected_features_sim'] = []

# --- Helpers S3 ---
@st.cache_resource
def init_s3():
    """Initialisation sécurisée de S3"""
    try:
        s3 = boto3.client('s3',
                          aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                          aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                          region_name=os.getenv('AWS_REGION', 'eu-north-1'))
        return s3
    except Exception as e:
        st.error(f"Erreur lors de l'initialisation de S3: {str(e)}")
        st.info("Vérifiez que les variables d'environnement AWS sont bien définies.")
        st.stop()

@st.cache_data
def load_s3_parquet(_s3, key):
    """Charge un fichier Parquet depuis S3"""
    if not BUCKET_NAME:
        st.error("Variable d'environnement 'AWS_S3_BUCKET_NAME' non définie.")
        st.stop()
    try:
        obj = _s3.get_object(Bucket=BUCKET_NAME, Key=S3_PREFIX_DATA + key)
        return pd.read_parquet(BytesIO(obj['Body'].read()))
    except ClientError as e:
        st.error(f"Erreur S3 ({key}): {e.response['Error']['Message']}")
        st.info("Vérifiez que le nom du fichier et le chemin S3 sont corrects.")
        st.stop()

# --- Fonctions pour les visualisations ---
def create_interactive_distribution_plot(df, feature, client_value, title):
    """
    Crée un histogramme interactif avec 2 distributions superposées et la ligne du client.
    Améliorations WCAG: Utilisation de couleurs daltonisme-compatibles.
    """
    df_plot = df.dropna(subset=[feature])
    df_accord = df_plot[df_plot['TARGET'] == 0]
    df_defaut = df_plot[df_plot['TARGET'] == 1]
    
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df_accord[feature],
        name='Prêts Accordés (TARGET = 0)',
        marker_color=COLOR_ACCORDE, # Utilisation de la couleur définie
        opacity=0.7
    ))

    fig.add_trace(go.Histogram(
        x=df_defaut[feature],
        name='Prêts en Défaut (TARGET = 1)',
        marker_color=COLOR_DEFAUT, # Utilisation de la couleur définie
        opacity=0.7
    ))

    # Ajouter la ligne pour la valeur du client
    fig.add_vline(x=client_value, line_dash="dash", line_color=COLOR_CLIENT_HIGHLIGHT, 
                  annotation_text="Valeur du client", annotation_position="top right",
                  annotation_font_color=COLOR_CLIENT_HIGHLIGHT) # Couleur pour le texte de l'annotation

    # Mise à jour du layout
    fig.update_layout(
        barmode='overlay',
        title=f"Distribution de '{feature}'",
        xaxis_title=feature,
        yaxis_title="Nombre de clients",
        legend_title="Statut du Prêt", # Légende plus explicite
        height=350, # Hauteur fixe pour cohérence
    )

    return fig

def create_bivariate_plot(df, feature_x, feature_y, client_value_x, client_value_y, title):
    """
    Crée un graphique de dispersion bivarié avec des couleurs et symboles daltonisme-compatibles.
    Améliorations WCAG: Couleurs discrètes, symboles différents pour TARGET, légende claire.
    """
    df_plot = df.dropna(subset=[feature_x, feature_y])
    
    fig = px.scatter(df_plot,
                     x=feature_x,
                     y=feature_y,
                     color='TARGET',
                     symbol='TARGET', # Utilisation de symboles différents (cercle pour 0, croix pour 1)
                     color_discrete_map={0: COLOR_ACCORDE, 1: COLOR_DEFAUT}, # Couleurs définies globalement
                     title=title,
                     labels={'TARGET': 'Statut du prêt (0=Accordé, 1=Défaut)'}, # Labels explicites
                     hover_data=['TARGET'])

    # Ajustement des marqueurs et de la légende pour être plus descriptifs et accessibles
    fig.update_traces(marker=dict(size=8, line=dict(width=0.5, color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    fig.update_layout(
        legend_title_text='Statut du Prêt',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Ajout du point pour le client/simulation
    fig.add_trace(go.Scatter(
        x=[client_value_x],
        y=[client_value_y],
        mode='markers',
        marker=dict(
            color=COLOR_CLIENT_HIGHLIGHT, # Point du client reste rouge pour le distinguer clairement
            size=15,
            symbol='star',
            line=dict(width=2, color='white')
        ),
        name='Position du client' # Ajout d'un nom pour la légende du client
    ))
    
    fig.update_layout(height=450)
    
    return fig

def display_static_shap_images():
    """Affiche des exemples d'images SHAP depuis des URLs."""
    st.markdown("### **Explication du modèle**")
    st.info("Les graphiques ci-dessous sont des exemples de visualisations SHAP et ne sont pas calculés en temps réel. Ils servent à illustrer l'explication du modèle.")
    
    col_waterfall, col_bar = st.columns(2)
    with col_waterfall:
        st.markdown("#### **Contribution des facteurs (SHAP local)**")
        st.image(LOCAL_SHAP_IMAGE_URL, use_container_width=True, caption="Exemple de contribution des variables pour un client")

    with col_bar:
        st.markdown("#### **Importance globale des facteurs (SHAP global)**")
        st.image(GLOBAL_SHAP_IMAGE_URL, use_container_width=True, caption="Importance moyenne des variables sur l'ensemble de la population")


def display_client_analysis(client_id, X_train_raw):
    """Affiche les résultats de l'analyse d'un client."""
    st.subheader(f"Analyse du dossier client n°{client_id}")
    st.markdown("---")
    
    client_data_raw = X_train_raw.loc[[client_id]].drop(columns=['TARGET'], errors='ignore')

    # Simulation de la probabilité de défaut pour la démo
    proba = 0.55 # Maintenu pour la démo
    
    col_score, col_explication = st.columns([1, 2])
    with col_score:
        st.metric("Probabilité de défaut", f"{proba:.2%}")
        threshold = 0.5
        decision = "Refusé" if proba > threshold else "Accordé"
        st.write(f"**Décision :** {decision}")
        
        # Jauge de score améliorée avec les couleurs définies
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Score de Risque de Défaut", 'font': {'size': 20}},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"}, # Couleur de la barre de progression
                       'steps': [{'range': [0, threshold * 100], 'color': COLOR_GAUGE_ACCORDE}, # Bleu pour Accordé
                                 {'range': [threshold * 100, 100], 'color': COLOR_GAUGE_DEFAUT}], # Orange pour Refusé
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold * 100}}
            )
        )
        fig_gauge.update_layout(height=250, title_font_size=18)
        st.plotly_chart(fig_gauge)

    with col_explication:
        display_static_shap_images()

    st.markdown("---")
    st.markdown("#### Profil du client vs. Population")
    
    feature_options = X_train_raw.drop(columns=['TARGET'], errors='ignore').columns.tolist()
    with st.form("client_comparison_form"):
        features_to_compare = st.multiselect(
            "Sélectionnez 2 features à afficher :",
            options=feature_options,
            default=st.session_state.get('selected_features_client', []),
            max_selections=2,
            key='client_multiselect'
        )
        submit_comparison = st.form_submit_button("Afficher les graphiques")
    
    if submit_comparison or (st.session_state['client_multiselect'] and st.session_state['selected_features_client'] == features_to_compare):
        st.session_state['selected_features_client'] = features_to_compare
        if len(st.session_state['selected_features_client']) == 2:
            feature1, feature2 = st.session_state['selected_features_client']
            client_value1 = client_data_raw.loc[client_id, feature1]
            client_value2 = client_data_raw.loc[client_id, feature2]

            st.markdown("##### 1. Distributions univariées")
            col1, col2 = st.columns(2)
            with col1:
                dist_plot1 = create_interactive_distribution_plot(X_train_raw, feature1, client_value1, f"Distribution de '{feature1}'")
                st.plotly_chart(dist_plot1, use_container_width=True, key='client_dist_plot_1')
            with col2:
                dist_plot2 = create_interactive_distribution_plot(X_train_raw, feature2, client_value2, f"Distribution de '{feature2}'")
                st.plotly_chart(dist_plot2, use_container_width=True, key='client_dist_plot_2')

            st.markdown("##### 2. Distribution bivariée")
            bivariate_plot = create_bivariate_plot(X_train_raw, feature1, feature2, client_value1, client_value2, f"Relation entre {feature1} et {feature2}")
            st.plotly_chart(bivariate_plot, use_container_width=True, key='client_bivariate_plot')
        else:
            st.warning("Veuillez sélectionner exactement 2 features pour afficher les graphiques.")


def display_simulator(client_id, X_train_raw):
    """Affiche le simulateur client."""
    st.subheader("Simuler un nouveau dossier client")
    st.markdown("---")
    
    simulation_features_dict = {
        "AMT_CREDIT": "Montant du prêt",
        "AMT_ANNUITY": "Montant Annuité",
        "PAYMENT_RATE": "Ratio Crédit/Annuité",
        "DAYS_EMPLOYED": "Ancienneté emploi (jours)",
        "REGION_POPULATION_RELATIVE": "Taux de population région",
        "EXT_SOURCE_1": "Source Extérieure 1",
        "EXT_SOURCE_2": "Source Extérieure 2",
        "EXT_SOURCE_3": "Source Extérieure 3",
        "CNT_CHILDREN": "Nombre d'enfants",
        "DAYS_BIRTH": "Âge client (jours)"
    }
    
    if 'simulated_features' not in st.session_state or st.session_state.client_id_selected != client_id:
        default_values = X_train_raw.loc[[client_id]].squeeze().to_dict()
        st.session_state.simulated_features = {
            col: default_values.get(col, X_train_raw[col].mean()) for col in simulation_features_dict.keys() if col in X_train_raw.columns
        }
        st.session_state.client_id_selected = client_id

    with st.form("simulation_form"):
        st.write("Modifiez les valeurs des principales variables pour simuler un nouveau client :")
        
        editable_features = {}
        cols_form = st.columns(2)
        
        for i, (col, label) in enumerate(simulation_features_dict.items()):
            if col in X_train_raw.columns:
                with cols_form[i % 2]:
                    min_val = X_train_raw[col].min()
                    max_val = X_train_raw[col].max()
                    
                    if pd.api.types.is_integer_dtype(X_train_raw[col]):
                        step_val = 1
                        value_input = st.number_input(label, value=int(st.session_state.simulated_features[col]), min_value=int(min_val), max_value=int(max_val), key=f'sim_int_{col}', step=step_val)
                    else:
                        step_val = (max_val - min_val) / 100.0 if max_val != min_val else 0.01
                        value_input = st.number_input(label, value=float(st.session_state.simulated_features[col]), min_value=float(min_val), max_value=float(max_val), key=f'sim_float_{col}', step=step_val, format="%.2f")
                    editable_features[col] = value_input
        
        submitted = st.form_submit_button("Calculer le score de ce client")
    
    if submitted:
        st.session_state['simulation_submitted'] = True
        st.session_state['simulation_features'] = editable_features

    if st.session_state['simulation_submitted']:
        st.markdown("---")
        st.markdown("### **Résultat de la simulation**")
        
        # Simulation d'un score
        sim_proba = 0.45 # Maintenu pour la démo
        
        col_score_sim, col_explication_sim = st.columns([1, 2])
        with col_score_sim:
            st.metric("Probabilité de défaut", f"{sim_proba:.2%}")
            threshold = 0.5
            decision = "Refusé" if sim_proba > threshold else "Accordé"
            st.write(f"**Décision :** {decision}")

        with col_explication_sim:
            display_static_shap_images()

        st.markdown("---")
        st.markdown("#### Profil simulé vs. Population")
        
        feature_options = X_train_raw.drop(columns=['TARGET'], errors='ignore').columns.tolist()
        
        with st.form("simulation_comparison_form"):
            features_to_compare_sim = st.multiselect(
                "Sélectionnez 2 features à afficher :",
                options=feature_options,
                default=st.session_state.get('selected_features_sim', []), # Utilisation de 'selected_features_sim'
                max_selections=2,
                key='sim_multiselect'
            )
            submit_sim_comparison = st.form_submit_button("Afficher les graphiques de simulation")

        if submit_sim_comparison or (st.session_state['sim_multiselect'] and st.session_state['selected_features_sim'] == features_to_compare_sim):
            st.session_state['selected_features_sim'] = features_to_compare_sim
            if len(st.session_state['selected_features_sim']) == 2:
                feature1, feature2 = st.session_state['selected_features_sim']
                
                # Pour le simulateur, on utilise les valeurs du formulaire
                sim_value1 = st.session_state['simulation_features'].get(feature1, X_train_raw[feature1].mean())
                sim_value2 = st.session_state['simulation_features'].get(feature2, X_train_raw[feature2].mean())

                st.markdown("##### 1. Distributions univariées")
                col1_sim, col2_sim = st.columns(2)
                with col1_sim:
                    dist_plot1_sim = create_interactive_distribution_plot(X_train_raw, feature1, sim_value1, f"Distribution de '{feature1}'")
                    st.plotly_chart(dist_plot1_sim, use_container_width=True, key='sim_dist_plot_1')
                with col2_sim:
                    dist_plot2_sim = create_interactive_distribution_plot(X_train_raw, feature2, sim_value2, f"Distribution de '{feature2}'")
                    st.plotly_chart(dist_plot2_sim, use_container_width=True, key='sim_dist_plot_2')

                st.markdown("##### 2. Distribution bivariée")
                bivariate_plot_sim = create_bivariate_plot(X_train_raw, feature1, feature2, sim_value1, sim_value2, f"Relation simulée entre {feature1} et {feature2}")
                st.plotly_chart(bivariate_plot_sim, use_container_width=True, key='sim_bivariate_plot')
            else:
                st.warning("Veuillez sélectionner exactement 2 features pour afficher les graphiques.")


# --- Main App ---
def main():
    st.title("📊 Dashboard d'Analyse de Risque de Crédit")
    st.markdown("---")

    with st.spinner("Initialisation de l'application et chargement des ressources..."):
        s3 = init_s3()
        if s3 is None:
            return
            
        try:
            X_train_raw = load_s3_parquet(s3, "X_train.parquet")
            y_train = load_s3_parquet(s3, "y_train.parquet")
            if X_train_raw is None or y_train is None:
                return
                
            if 'TARGET' not in X_train_raw.columns:
                # S'assurer que les index sont alignés avant le merge
                if not X_train_raw.index.equals(y_train.index):
                    # Tentez un reset_index si les index ne sont pas alignés
                    X_train_raw = X_train_raw.reset_index(drop=True)
                    y_train = y_train.reset_index(drop=True)
                X_train_raw = X_train_raw.merge(y_train, left_index=True, right_index=True)

        except Exception as e:
            st.error(f"Erreur lors du chargement des données d'entraînement: {e}")
            return
            
    st.success("Toutes les ressources sont chargées avec succès !")
    st.markdown("---")

    # Sidebar pour la sélection du client
    st.sidebar.header("Sélection Client")
    # Vérifiez si 'SK_ID_CURR' est dans les colonnes ou si l'index est l'ID client
    if 'SK_ID_CURR' in X_train_raw.columns:
        # Si 'SK_ID_CURR' est une colonne, nous devons l'utiliser comme index pour les loc futurs
        if X_train_raw.index.name != 'SK_ID_CURR':
            X_train_raw = X_train_raw.set_index('SK_ID_CURR')
        client_id_list = X_train_raw.index.tolist()
    else:
        client_id_list = X_train_raw.index.tolist()
    
    # Assurez-vous que l'index par défaut pour selectbox est valide
    if client_id_list:
        client_id = st.sidebar.selectbox("Sélectionnez un ID client:", client_id_list, index=0)
    else:
        st.error("Aucun ID client disponible dans les données de formation.")
        return
    
    if st.sidebar.button("Lancer l'analyse complète"):
        st.session_state['client_analysis_submitted'] = True
    
    if client_id != st.session_state.get('last_client_id', None):
        st.session_state['simulation_submitted'] = False
        st.session_state['last_client_id'] = client_id
        st.session_state['selected_features_client'] = []
        st.session_state['selected_features_sim'] = [] # Réinitialiser aussi pour le simulateur

    tab1, tab2 = st.tabs(["🔍 Analyse Client", "🧮 Simulateur"])
    
    with tab1:
        if st.session_state['client_analysis_submitted']:
            display_client_analysis(client_id, X_train_raw)

    with tab2:
        display_simulator(client_id, X_train_raw)

if __name__ == "__main__":
    main()