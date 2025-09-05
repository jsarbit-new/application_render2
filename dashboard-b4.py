import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import os
from io import BytesIO
import boto3
from botocore.exceptions import ClientError
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import matplotlib
import json
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
import warnings
import tempfile
import joblib
from functools import lru_cache
import re
import streamlit.components.v1 as components

# --- Import de Bokeh pour les graphiques interactifs ---
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Legend
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.layouts import gridplot
from bokeh.palettes import Magma, Viridis

# Supprimer les avertissements pour une meilleure lisibilité
warnings.filterwarnings('ignore', category=UserWarning, module='shap')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message="X does not have valid feature names")
matplotlib.use('Agg')

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Dashboard Crédit")

# Constantes et variables d'environnement
BUCKET_NAME = os.getenv('AWS_S3_BUCKET_NAME')
S3_PREFIX_DATA = "input/"
S3_MODEL_KEY = "modele_mlflow_final/model/model.pkl"

# Initialisation de l'état de session si non déjà fait
if 'client_analysis_submitted' not in st.session_state:
    st.session_state['client_analysis_submitted'] = False
if 'simulation_submitted' not in st.session_state:
    st.session_state['simulation_submitted'] = False
if 'simulation_features' not in st.session_state:
    st.session_state['simulation_features'] = None
if 'client_id_selected' not in st.session_state:
    st.session_state['client_id_selected'] = None
if 'last_client_id' not in st.session_state:
    st.session_state['last_client_id'] = None
if 'selected_features_client' not in st.session_state:
    st.session_state['selected_features_client'] = []
if 'selected_features_sim' not in st.session_state:
    st.session_state['selected_features_sim'] = []
if 'shap_values_train' not in st.session_state:
    st.session_state['shap_values_train'] = None

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

# --- Helpers de chargement du modèle depuis S3 ---
@st.cache_resource
def load_s3_model_pipeline():
    """Charge le pipeline MLflow complet depuis S3 en utilisant joblib."""
    s3_client = init_s3()
    if s3_client is None:
        return None, None
    try:
        st.info(f"Chargement du modèle depuis s3://{BUCKET_NAME}/{S3_MODEL_KEY}")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "model.pkl")
            s3_client.download_file(BUCKET_NAME, S3_MODEL_KEY, temp_file_path)
            pipeline = joblib.load(temp_file_path)
        st.success("Pipeline chargé avec succès depuis S3.")
        
        preprocessor = pipeline.named_steps.get('preprocessor')
        feature_names_out = preprocessor.get_feature_names_out()
        feature_names = [re.sub(r'^[a-zA-Z0-9]+__', '', f) for f in feature_names_out]
        
        return pipeline, feature_names
    except (ClientError, Exception) as e:
        st.error(f"Erreur lors du chargement du pipeline depuis S3: {e}")
        st.stop()

# --- Helpers SHAP ---
@st.cache_resource
def train_explainer_model(_preprocessor, X_raw, y):
    """Entraîne un nouveau modèle de régression logistique pour l'explicabilité."""
    st.info("Entraînement d'un modèle local pour l'explicabilité (SHAP)...")
    X_processed = _preprocessor.transform(X_raw)
    feature_names_out = _preprocessor.get_feature_names_out()
    feature_names = [re.sub(r'^[a-zA-Z0-9]+__', '', f) for f in feature_names_out]
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X_raw.index)
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_processed_df, y.squeeze())
    st.success("Modèle SHAP local et données prétraitées pour l'explainer sont prêts.")
    return model, X_processed_df

@st.cache_resource
def create_explainer(_model, _data):
    """Crée l'explainer SHAP une seule fois."""
    return shap.Explainer(_model, _data)

@st.cache_resource
def generate_global_shap_data(_explainer, _data):
    """Génère les données nécessaires aux plots SHAP globaux et retourne les 10 features les plus importantes."""
    shap_values = _explainer(_data)
    feature_importance_df = pd.DataFrame({
        'feature': _data.columns,
        'importance': np.abs(shap_values.values).mean(0)
    }).sort_values(by='importance', ascending=False)
    
    top_10_features = feature_importance_df['feature'].head(10).tolist()
    
    st.session_state['shap_values_train'] = shap_values
    
    return feature_importance_df, top_10_features

# --- Fonctions pour les sections de l'UI ---

def create_interactive_bar_plot(feature_importance_df, title="Importance moyenne globale des features"):
    """Crée un graphique à barres interactif avec Bokeh."""
    source = ColumnDataSource(feature_importance_df)
    p = figure(x_range=feature_importance_df['feature'], height=350, title=title,
               tools="pan,box_zoom,reset,save", toolbar_location="above")
    
    p.vbar(x='feature', top='importance', width=0.9, source=source, 
           color="#007bff", legend_label="Importance moyenne")

    p.add_tools(HoverTool(
        tooltips=[("Feature", "@feature"), ("Importance", "@importance{0.2f}")]
    ))
    
    p.xaxis.major_label_orientation = 1.2
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.xaxis.axis_label = "Fonctionnalité"
    p.yaxis.axis_label = "Importance moyenne SHAP"
    
    return p

def create_interactive_distribution_plot(df, feature, client_value):
    """
    Crée un histogramme interactif avec 2 distributions superposées et la ligne du client.
    df: DataFrame contenant les features et la colonne 'TARGET'.
    feature: Nom de la feature à visualiser.
    client_value: Valeur de cette feature pour le client sélectionné.
    """
    # Séparer les données en deux groupes
    df_accord = df[df['TARGET'] == 0]
    df_defaut = df[df['TARGET'] == 1]
    
    # Définir les limites et les bins
    bins = 25
    min_val = df[feature].min()
    max_val = df[feature].max()
    
    # Calculer les histogrammes pour chaque groupe
    hist_accord, edges_accord = np.histogram(df_accord[feature].dropna(), bins=bins, range=(min_val, max_val))
    hist_defaut, edges_defaut = np.histogram(df_defaut[feature].dropna(), bins=bins, range=(min_val, max_val))
    
    # Créer le graphique
    p = figure(height=300, title=f"Distribution de '{feature}'",
               tools="pan,box_zoom,reset,save")
    
    # Créer les barres d'histogramme pour le groupe "Accordé"
    p.quad(top=hist_accord, bottom=0, left=edges_accord[:-1], right=edges_accord[1:],
           fill_color="green", legend_label="Prêts Accordés", line_color="white", alpha=0.7)
           
    # Créer les barres d'histogramme pour le groupe "Défaut"
    p.quad(top=hist_defaut, bottom=0, left=edges_defaut[:-1], right=edges_defaut[1:],
           fill_color="red", legend_label="Prêts en Défaut", line_color="white", alpha=0.7)
    
    # Ajouter la ligne pour la valeur du client
    max_hist = max(np.max(hist_accord), np.max(hist_defaut))
    p.line(x=[client_value, client_value], y=[0, max_hist * 1.1], line_color="black", line_width=3, line_dash="dashed", legend_label="Valeur du client")
    
    # Configuration supplémentaire
    p.xaxis.axis_label = feature
    p.yaxis.axis_label = "Nombre de clients"
    p.legend.location = "top_right"
    p.legend.title = "Légende"
    p.add_tools(HoverTool(
        tooltips=[("Plage", "$x"), ("Nb. clients", "$y")]
    ))
    
    return p

def create_bivariate_plot(df, feature_x, feature_y, client_value_x, client_value_y, title):
    """
    Crée un graphique de dispersion bivarié avec un dégradé de couleurs et la position du client/simulation.
    df: DataFrame contenant les features et les SHAP values moyennes.
    feature_x, feature_y: Les deux features à visualiser.
    client_value_x, client_value_y: Valeurs des features pour le client/simulation.
    title: Titre du graphique.
    """
    
    # Pour éviter le warning de Plotly, on s'assure que les features sont dans le DataFrame
    df_plot = df.dropna(subset=[feature_x, feature_y])
    
    # Création du nuage de points avec un dégradé de couleurs basé sur TARGET
    fig = px.scatter(df_plot,
                     x=feature_x,
                     y=feature_y,
                     color='TARGET',
                     color_continuous_scale=px.colors.sequential.Viridis,
                     title=title,
                     labels={'TARGET': 'Prêt Accordé (0) / Défaut (1)'},
                     hover_data=['TARGET'])

    # Ajout du point pour le client/simulation
    fig.add_trace(go.Scatter(
        x=[client_value_x],
        y=[client_value_y],
        mode='markers',
        marker=dict(
            color='red',
            size=15,
            symbol='star',
            line=dict(width=2, color='white')
        ),
        name='Position du client'
    ))
    
    fig.update_layout(height=450)
    
    return fig

def display_client_analysis(client_id, X_train_raw, X_train_processed_df, prediction_pipeline, preprocessor, explainer, global_feature_importance_df, top_10_features):
    """Affiche les résultats de l'analyse d'un client."""
    st.subheader(f"Analyse du dossier client n°{client_id}")
    st.markdown("---")

    client_data_raw = X_train_raw.loc[[client_id]].drop(columns=['TARGET'])
    
    with st.spinner("Calcul en cours..."):
        proba = prediction_pipeline.predict_proba(client_data_raw)[0][1]
        client_data_processed = preprocessor.transform(client_data_raw)
        client_data_processed_df = pd.DataFrame(client_data_processed, columns=X_train_processed_df.columns, index=[client_id])
        shap_values_client = explainer(client_data_processed_df)
        
    col_score, col_explication = st.columns([1, 2])
    with col_score:
        st.metric("Probabilité de défaut", f"{proba:.2%}")
        threshold = 0.5
        decision = "Refusé" if proba > threshold else "Accordé"
        st.write(f"**Décision :** {decision}")
        
        # Jauge de score améliorée
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Score de Risque de Défaut", 'font': {'size': 20}},
                gauge={'axis': {'range': [None, 100]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, threshold * 100], 'color': "#00ff7f"},
                                 {'range': [threshold * 100, 100], 'color': "#ff6347"}],
                       'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': threshold * 100}}
            )
        )
        fig_gauge.update_layout(height=250, title_font_size=18)
        st.plotly_chart(fig_gauge)

    with col_explication:
        st.markdown("### **Facteurs déterminants**")
        st.info("Voici les 5 facteurs qui ont le plus influencé le score de ce client. Les flèches indiquent l'impact sur le risque de défaut.")
        
        top_features_values = shap_values_client[0].values
        top_features_abs = np.abs(top_features_values)
        sorted_indices = np.argsort(top_features_abs)[::-1]
        
        for i in sorted_indices[:5]:
            feature_name = shap_values_client[0].feature_names[i]
            shap_value = top_features_values[i]
            if feature_name in client_data_raw.columns:
                feature_value = client_data_raw.loc[client_id, feature_name]
            else:
                feature_value = "N/A"

            impact_emoji = "📈" if shap_value > 0 else "📉"
            impact_text = "augmente le risque" if shap_value > 0 else "diminue le risque"
            
            st.write(f"**{impact_emoji} {feature_name} ({feature_value:.2f}) :** Cette valeur {impact_text}.")

    st.markdown("---")
    st.subheader("Analyse comparative et explicative")
    st.info("Le graphique en cascade ci-dessous (Waterfall Plot) montre la contribution de chaque variable à la prédiction du modèle. Il illustre comment le score final est atteint en ajoutant les impacts positifs (rouge) et négatifs (bleus).")
    
    col_waterfall, col_bar = st.columns(2)
    with col_waterfall:
        st.markdown("#### Détail des facteurs SHAP")
        fig_shap_waterfall = plt.figure(figsize=(10, 5))
        shap.plots.waterfall(shap_values_client[0], max_display=10, show=False)
        plt.title("Explication de la prédiction")
        st.pyplot(fig_shap_waterfall, bbox_inches='tight')
    
    with col_bar:
        st.markdown("#### Importance globale des facteurs (Interactif)")
        st.info("Passez la souris sur les barres pour voir les détails. Vous pouvez aussi zoomer et vous déplacer.")
        bokeh_bar_plot = create_interactive_bar_plot(global_feature_importance_df.head(20))
        bar_plot_html = file_html(bokeh_bar_plot, CDN, "Bar Plot SHAP")
        components.html(bar_plot_html, height=400, scrolling=True)
        
    st.markdown("---")
    st.markdown("#### Profil du client vs. Population (Analyses)")
    
    with st.form("client_comparison_form"):
        features_to_compare = st.multiselect(
            "Sélectionnez 2 features à afficher :",
            options=global_feature_importance_df['feature'].head(20).tolist(),
            default=st.session_state.get('selected_features_client', []),
            max_selections=2
        )
        submit_comparison = st.form_submit_button("Afficher les graphiques")
    
    if submit_comparison:
        st.session_state['selected_features_client'] = features_to_compare
    
    if len(st.session_state.get('selected_features_client', [])) == 2:
        feature1, feature2 = st.session_state['selected_features_client']
        client_value1 = client_data_raw.loc[client_id, feature1]
        client_value2 = client_data_raw.loc[client_id, feature2]

        st.markdown("##### 1. Distributions univariées")
        bokeh_plots = []
        plot1 = create_interactive_distribution_plot(X_train_raw, feature1, client_value1)
        plot2 = create_interactive_distribution_plot(X_train_raw, feature2, client_value2)
        bokeh_plots.append(plot1)
        bokeh_plots.append(plot2)
        grid = gridplot(bokeh_plots, ncols=len(bokeh_plots), sizing_mode="scale_width")
        grid_html = file_html(grid, CDN, "Client Comparison")
        components.html(grid_html, height=400, scrolling=True)

        st.markdown("##### 2. Distribution bivariée")
        bivariate_plot = create_bivariate_plot(X_train_raw, feature1, feature2, client_value1, client_value2, f"Relation entre {feature1} et {feature2}")
        st.plotly_chart(bivariate_plot, use_container_width=True)

    else:
        st.warning("Veuillez sélectionner exactement 2 features pour afficher les graphiques.")
        
def display_simulator(client_id, X_train_raw, X_train_processed_df, prediction_pipeline, preprocessor, explainer, global_feature_importance_df):
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
        with st.spinner("Simulation en cours..."):
            sim_data_raw = pd.DataFrame([st.session_state['simulation_features']])
            sim_data_raw = sim_data_raw.astype(X_train_raw[list(st.session_state['simulation_features'].keys())].dtypes)
            
            full_sim_data = X_train_raw.iloc[0:1].drop(columns=['TARGET']).copy()
            for feat, val in st.session_state['simulation_features'].items():
                full_sim_data.loc[full_sim_data.index[0], feat] = val

            proba = prediction_pipeline.predict_proba(full_sim_data)[0][1]
            sim_data_processed = preprocessor.transform(full_sim_data)
            sim_data_processed_df = pd.DataFrame(sim_data_processed, columns=X_train_processed_df.columns, index=full_sim_data.index)
            shap_values = explainer(sim_data_processed_df)

            st.markdown("---")
            st.markdown("### **Résultat de la simulation**")
            col_score_sim, col_explication_sim = st.columns([1, 2])
            with col_score_sim:
                st.metric("Probabilité de défaut", f"{proba:.2%}")
                threshold = 0.5
                decision = "Refusé" if proba > threshold else "Accordé"
                st.write(f"**Décision :** {decision}")
            
            with col_explication_sim:
                st.markdown("#### Principaux facteurs (simulation)")
                top_features_values = shap_values[0].values
                top_features_abs = np.abs(top_features_values)
                sorted_indices = np.argsort(top_features_abs)[::-1]
                for i in sorted_indices[:5]:
                    feature_name = shap_values[0].feature_names[i]
                    shap_value = top_features_values[i]
                    feature_value = full_sim_data[feature_name].iloc[0]
                    impact_emoji = "📈" if shap_value > 0 else "📉"
                    impact_text = "augmente le risque" if shap_value > 0 else "diminue le risque"
                    st.write(f"**{impact_emoji} {feature_name} ({feature_value:.2f}) :** Cette valeur {impact_text}.")

            st.markdown("---")
            st.subheader("Analyse de la simulation")
            
            col_waterfall_sim, col_bar_sim = st.columns(2)
            with col_waterfall_sim:
                st.markdown("#### Détail des facteurs SHAP")
                fig_shap_waterfall_sim = plt.figure(figsize=(10, 5))
                shap.plots.waterfall(shap_values[0], max_display=10, show=False)
                plt.title("Explication de la prédiction")
                st.pyplot(fig_shap_waterfall_sim, bbox_inches='tight')

            with col_bar_sim:
                st.markdown("#### Importance globale des facteurs (Interactif)")
                st.info("Passez la souris sur les barres pour voir les détails. Vous pouvez aussi zoomer et vous déplacer.")
                bokeh_bar_plot_sim = create_interactive_bar_plot(global_feature_importance_df.head(20))
                bar_plot_html_sim = file_html(bokeh_bar_plot_sim, CDN, "Simulated Bar Plot SHAP")
                components.html(bar_plot_html_sim, height=400, scrolling=True)

            st.markdown("---")
            st.markdown("#### Simulation vs. Population (Analyses)")
            
            features_to_compare_sim = st.multiselect(
                "Sélectionnez 2 features à afficher :",
                options=list(simulation_features_dict.keys()),
                default=st.session_state['selected_features_sim'],
                max_selections=2,
                key='sim_multiselect'
            )
            
            if len(features_to_compare_sim) == 2:
                st.session_state['selected_features_sim'] = features_to_compare_sim
                feature1, feature2 = features_to_compare_sim
                sim_value1 = st.session_state['simulation_features'][feature1]
                sim_value2 = st.session_state['simulation_features'][feature2]

                st.markdown("##### 1. Distributions univariées")
                bokeh_plots_sim = []
                plot1_sim = create_interactive_distribution_plot(X_train_raw, feature1, sim_value1)
                plot2_sim = create_interactive_distribution_plot(X_train_raw, feature2, sim_value2)
                bokeh_plots_sim.append(plot1_sim)
                bokeh_plots_sim.append(plot2_sim)
                grid_sim = gridplot(bokeh_plots_sim, ncols=len(bokeh_plots_sim), sizing_mode="scale_width")
                grid_html_sim = file_html(grid_sim, CDN, "Simulation Comparison")
                components.html(grid_html_sim, height=400, scrolling=True)
                
                st.markdown("##### 2. Distribution bivariée")
                bivariate_plot_sim = create_bivariate_plot(X_train_raw, feature1, feature2, sim_value1, sim_value2, f"Relation simulée entre {feature1} et {feature2}")
                st.plotly_chart(bivariate_plot_sim, use_container_width=True)

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
            
        prediction_pipeline, feature_names = load_s3_model_pipeline()
        if prediction_pipeline is None or feature_names is None:
            return

        preprocessor = prediction_pipeline.named_steps.get('preprocessor')
        
        X_train_raw = load_s3_parquet(s3, "X_train.parquet")
        y_train = load_s3_parquet(s3, "y_train.parquet")
        if X_train_raw is None or y_train is None:
            return
            
        if 'TARGET' not in X_train_raw.columns:
            if not pd.api.types.is_integer_dtype(X_train_raw.index):
                 X_train_raw = X_train_raw.reset_index(drop=True).set_index(X_train_raw.index)
            X_train_raw = X_train_raw.merge(y_train, left_index=True, right_index=True)

        explainer_model, X_train_processed_df = train_explainer_model(preprocessor, X_train_raw.drop(columns=['TARGET']), y_train)
        explainer = create_explainer(explainer_model, X_train_processed_df)
        global_feature_importance_df, top_10_features = generate_global_shap_data(explainer, X_train_processed_df)

    st.success("Toutes les ressources sont chargées avec succès !")
    st.markdown("---")

    # Sidebar pour la sélection du client
    st.sidebar.header("Sélection Client")
    client_id_list = X_train_raw.index.tolist()
    client_id = st.sidebar.selectbox("Sélectionnez un ID client:", client_id_list, index=0)
    
    if st.sidebar.button("Lancer l'analyse complète"):
        st.session_state['client_analysis_submitted'] = True
    
    if client_id != st.session_state.get('last_client_id', None):
        st.session_state['simulation_submitted'] = False
        st.session_state['last_client_id'] = client_id
        st.session_state['selected_features_client'] = []

    tab1, tab2 = st.tabs(["🔍 Analyse Client", "🧮 Simulateur"])
    
    with tab1:
        if st.session_state['client_analysis_submitted']:
            display_client_analysis(client_id, X_train_raw, X_train_processed_df, prediction_pipeline, preprocessor, explainer, global_feature_importance_df, top_10_features)

    with tab2:
        display_simulator(client_id, X_train_raw, X_train_processed_df, prediction_pipeline, preprocessor, explainer, global_feature_importance_df)

if __name__ == "__main__":
    main()