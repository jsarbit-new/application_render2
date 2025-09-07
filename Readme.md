Fichier Readme.md

# Projet 8 : Réalisez un dashboard  et assurer une veille technologique



1. dashboard-b6.py

Ce script est la version complète et en temps réel de l'application Streamlit.

    Fonctionnalités : Il effectue tous les calculs dynamiquement, incluant :

        La probabilité de défaut pour un client existant ou pour la simulation d'un nouveau client.

        Le calcul des valeurs SHAP pour expliquer la prédiction du modèle.

        La génération d'un rapport de dérive (drift) avec Evidently pour 10 lignes de données.

    Exécution : Ce script n'est pas conçu pour les services cloud gratuits. Il doit être exécuté via une console Python sur un ordinateur local, avec les secrets de connexion au bucket AWS S3 correctement définis.

2. dashboard-b6-render.py

Ce script est une version optimisée pour le déploiement sur la plateforme gratuite RENDER. Pour surmonter les limitations de l'offre gratuite, certaines fonctionnalités ont été pré-calculées.

    Stratégie d'optimisation :

        Les graphiques SHAP ont été générés en amont à l'aide du script generate_shap_plots.py et les images résultantes ont été stockées sur le bucket AWS S3.

        Le rapport de dérive Evidently a aussi été généré en local et mis à disposition via un lien de téléchargement directement dans l'application.

