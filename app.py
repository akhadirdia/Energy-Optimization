import os
from flask import Flask, render_template, request, session, jsonify, redirect, url_for
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # Utiliser le backend 'Agg' pour la génération de graphiques sans GUI
import matplotlib.pyplot as plt
import base64
import nbformat
import numpy as np
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Clé secrète pour la session
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def prepare_data(file):
    df1 = pd.read_csv(file, engine='python', sep=None, header=0, encoding='ISO-8859-1')
    for col in df1.columns:
        if df1[col].dtype == 'object':
            df1[col] = df1[col].str.replace(',', '.')
            df1[col] = pd.to_numeric(df1[col], errors='ignore')
    new_var = "Facteur d?utilisation ou FU (%)"
    var_sup = ["Code de relève", "Contrat", "Tarif", "Jour", "Date et heure de la dernière relève", "Facteur de puissance ou FP (%)", new_var]
    df1 = df1.drop(var_sup, axis=1)
    return df1

def prepare_data2(file):
    df2 = pd.read_excel(file, sheet_name='Feuil1', usecols='A:E') #'B:E'
    return df2

def prepare_data3(file):
    df3 = pd.read_excel(file, sheet_name=0, header=0) 
    return df3

def prepare_data_elec(file):
    df = pd.read_excel(file, sheet_name=0, header=0)
    var_supp=["Nbre_borne7.2", "Nbre_borne24", "Nbre_borne50"]
    df=df.drop(var_supp, axis=1) 
    return df
    
def numeric_data(df):
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df

def describe_data(numeric_df):
    # Exemple de traitement pour df1
    if not numeric_df.empty:        
        return numeric_df.describe(include='all').to_html(classes='table table-striped', table_id='myTable')
    return None

def describe_df3(df3):
    var = ["Autonomie_KM_ete", "Autonomie_KM_hiver", "Confirmation Km/Jour" , "Consomation kWh/jour (été)", "Consomation kWh/jour (hiver)", "Confirmation Capacité Batterie (kWh)"]
    # Calcul de la moyenne pour chaque groupe
    mean_df = df3.groupby('Catégorie de véhicule Électrique')[var].mean()
    # Ajout de la colonne du nombre de véhicules par catégorie
    mean_df['Nombre de véhicules'] = df3.groupby('Catégorie de véhicule Électrique').size()
    mean_df = mean_df.round(0).astype(int)
    if not mean_df.empty:
        return mean_df.to_html(classes='table table-striped', table_id='myTable')
    return None

def describe_df3_munic(df):
    var = ["Consomation kWh/jour (été)", "Consomation kWh/jour (hiver)", "Autonomie_KM_ete", "Autonomie_KM_hiver", "Confirmation Km/Jour", "Confirmation autonomie Batterie (kWh)"]
    # Calcul de la moyenne pour chaque groupe
    mean_df = df.groupby('Catégorie de véhicule Électrique')[var].mean()
    mean_df['Nombre de véhicules par catégorie']=df.groupby('Catégorie de véhicule Électrique').size()
    mean_df = mean_df.round(0).astype(int)
    if not mean_df.empty:
        return mean_df.to_html(classes='table table-striped', table_id='myTable')
    return None

def describe_elec(df):
    df_head=df.head(15)
    if not df_head.empty:
        return df_head.to_html(classes='table table-striped table-bordered table-hover', table_id='myTable')
    return None

# def cons_puissance_autobus(df, phase1_start, phase1_end, phase2_start, phase2_end):
    phase_1 = df[(df['Année conversion Electrique (trajet)'] >= phase1_start) & (df['Année conversion Electrique (trajet)'] <= phase1_end)]
    phase_2 = df[(df['Année conversion Electrique (trajet)'] >= phase2_start) & (df['Année conversion Electrique (trajet)'] <= phase2_end)]
    phase_3 = df

    var_phase = ["Consomation kWh/jour (été)", "Consomation kWh/jour (hiver)", "Puissance de la borne calculé pour la recharge (kW)"]

    sum_phase_1 = phase_1[var_phase].sum().round(2)
    sum_phase_2 = phase_2[var_phase].sum().round(2)
    sum_phase_3 = phase_3[var_phase].sum().round(2)

    results_df = pd.DataFrame({
        'Phase 1': sum_phase_1,
        'Phase 2': sum_phase_2,
        'Phase 3': sum_phase_3
    }).transpose()

    if not results_df.empty:
        return results_df.to_html(classes='table table-striped', table_id='myTable')
    return None


# def cons_puissance_autobus(df):
    phase_1 = df[(df['Année conversion Electrique (trajet)'] >= 1) & (df['Année conversion Electrique (trajet)'] <= 3)]
    phase_2 = df[(df['Année conversion Electrique (trajet)'] >= 1) & (df['Année conversion Electrique (trajet)'] <= 6)]
    phase_3 = df
    
    var_phase = ["Consomation kWh/jour (été)", "Consomation kWh/jour (hiver)", "Puissance de la borne calculé pour la recharge (kW)"]
    
    # Calculer les sommes pour chaque phase
    sum_phase_1 = phase_1[var_phase].sum()
    sum_phase_1 = sum_phase_1.round(2)
    
    sum_phase_2 = phase_2[var_phase].sum()
    sum_phase_2 = sum_phase_2.round(2)
    
    sum_phase_3 = phase_3[var_phase].sum()
    sum_phase_3 = sum_phase_3.round(2)
    
    # Créer un DataFrame avec ces résultats
    results_df = pd.DataFrame({'Phase 1': sum_phase_1, 'Phase 2': sum_phase_2, 'Phase 3': sum_phase_3}).transpose()
    
    if not results_df.empty:
        return results_df.to_html(classes='table table-striped', table_id='myTable')
    
    return None

def credit_carbone(df):
    # Prix du carbone
    prix_carbone = 56.61

    # Calcul des G.E.S. émis et réduits
    Ges_diesel_fl = df['G.E.S. Chauffage (diésel) Émis par an C02eq. (tonnes)'].sum()
    Ges_diesel_compl = df['G.E.S. Chauffage (diésel) Émis par an C02eq. (tonnes) étude complète'].sum()
    Ges_fl = df['Réduction G.E.S. (tonnes)'].sum()
    Ges_compl = df['Réduction G.E.S. (tonnes) étude complète'].sum()

    # Calcul des crédits de carbone
    Credit_carbone_flotte = (Ges_fl - Ges_diesel_fl).round(2)
    Credit_carbone_complet = (Ges_compl - Ges_diesel_compl).round(2)

    # Calcul des compensations financières
    Compensation_fin_flotte = (Credit_carbone_flotte * prix_carbone).round(2)
    Compensation_fin_complet = (Credit_carbone_complet * prix_carbone).round(2)

    # Création du DataFrame
    data = {
        'Crédit carbone (tonnes)': [Credit_carbone_flotte, Credit_carbone_complet],
        'Compensation financière ($)': [f"${Compensation_fin_flotte:,.2f}", f"${Compensation_fin_complet:,.2f}"]
    }

    index_labels = ['Flotte/an', 'Étude complète (10 ans)']
    data_s= pd.DataFrame(data, index=index_labels)

    if not data_s.empty:
        return data_s.to_html(classes='table table-striped', table_id='myTable')
    return None

def credit_carbone_munic(df):
    # Prix du carbone
    prix_carbone = 56.61

    # Calcul des G.E.S. émis et réduits
    Ges_diesel_fl = df['G.E.S. Chauffage (diésel) Émis par an C02eq. (tonnes)'].sum()
    Ges_diesel_compl = df['G.E.S. Chauffage (diésel) Émis par an C02eq. (tonnes) étude complète'].sum()
    Ges_fl = df['Réduction G.E.S. (tonnes)'].sum()
    Ges_compl = df['Réduction G.E.S. (tonnes) étude complète'].sum()

    # Calcul des crédits de carbone
    Credit_carbone_flotte = (Ges_fl - Ges_diesel_fl).round(2)
    Credit_carbone_complet = (Ges_compl - Ges_diesel_compl).round(2)

    # Calcul des compensations financières
    Compensation_fin_flotte = (Credit_carbone_flotte * prix_carbone).round(2)
    Compensation_fin_complet = (Credit_carbone_complet * prix_carbone).round(2)

    # Création du DataFrame
    data = {
        'Crédit carbone (tonnes)': [Credit_carbone_flotte, Credit_carbone_complet],
        'Compensation financière ($)': [f"${Compensation_fin_flotte:,.2f}", f"${Compensation_fin_complet:,.2f}"]
    }

    index_labels = ['Flotte/an', 'Étude complète (10 ans)']
    data_s= pd.DataFrame(data, index=index_labels)

    if not data_s.empty:
        return data_s.to_html(classes='table table-striped', table_id='myTable')
    return None
# Fonction générique pour calculer la consommation énergétique et le facteur d'utilisation
def calculer_cout_energetique(df, results_df, phase_index, puissance_col):
    mois_hiver = ['Novembre', 'Décembre', 'Janvier', 'Février', 'Mars']
    #mois_ete = ['Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre']
    consommation_ete = results_df.iloc[phase_index, 0]  # kWh
    consommation_hiver = results_df.iloc[phase_index, 1]  # kWh
    
    def calculer_consommation(row):
        if row['Mois'] in mois_hiver:
            consommation = consommation_hiver
        else:
            consommation = consommation_ete
            # Appliquer un facteur spécifique pour Juillet et Août
            if row['Mois'] in ['Juillet', 'Août']:
                consommation *= 0.10
        cons_energetique = consommation * row['Nombre de jours ouvrés']
        return cons_energetique
    
    # Appliquer la fonction pour créer la nouvelle colonne
    df['Consommation énergétique (kWh)'] = df.apply(calculer_consommation, axis=1)
    
    # Calculer le facteur d'utilisation
    df["Facteur d'utilisation (%)"] = 100 * df['Consommation énergétique (kWh)'] / (df['Nombre de Jours'] * 24 * df[puissance_col])
    
    return df

def calcul_tarif_BR(ligne, phase, opt):
    tarif_1ere_tranche = 0.12844  # Exemple
    tarif_2e_tranche = 0.24070  # Exemple
    tarif_3e_tranche = 0.18929  # Exemple
    seuil_1ere_tranche = 50  # Exemple
    # Extraction des valeurs nécessaires de la ligne
    cons = ligne['Consommation énergétique (kWh)']
    fu = ligne["Facteur d'utilisation (%)"] / 100  # Assurez-vous que fu est un pourcentage correct
    jour_ouvré = ligne['Nombre de jours ouvrés']
    
    # Sélection de la puissance en fonction de la phase et de l'option
    if opt == 0:
        if phase == 1:
            puissance = ligne['Puissance phase 1 (kW)']
        elif phase == 2:
            puissance = ligne['Puissance phase 2 (kW)']
        elif phase == 3:
            puissance = ligne['Puissance phase 3 (kW)']
    else:
        if phase == 1:
            puissance = ligne['Puissance optimale phase 1 (kW)']
        elif phase == 2:
            puissance = ligne['Puissance optimale phase 2 (kW)']
        elif phase == 3:
            puissance = ligne['Puissance optimale phase 3 (kW)']
    
    if puissance <= seuil_1ere_tranche:
        tranche1 = (puissance * fu) * (jour_ouvré * 24)
    
    # Coûts pour la 1ère tranche
    if puissance <= seuil_1ere_tranche:
        tranche1 = (puissance * fu) * (jour_ouvré * 24)
    else:
        tranche1 = (seuil_1ere_tranche * fu) * (jour_ouvré * 24)
    cout_tranche1 = tranche1 * tarif_1ere_tranche
    
    # Coûts pour la 2e tranche
    if puissance > seuil_1ere_tranche:
        tranche2 = ((puissance - seuil_1ere_tranche) * 0.03) * (jour_ouvré * 24)
    else:
        tranche2 = 0
    cout_tranche2 = tranche2 * tarif_2e_tranche
    
    # Coûts pour la 3e tranche
    tranche3 = cons - (tranche1 + tranche2) if (tranche1 + tranche2) < cons else 0
    cout_tranche3 = tranche3 * tarif_3e_tranche

    cout_total = cout_tranche1 + cout_tranche2 + cout_tranche3
    
    return cout_total

def calcul_tarif_M(ligne, phase, opt):

    tarif_1ere_tranche_210000kwh = 0.05851  # Exemple
    tarif_reste_energie = 0.04339  # Exemple
    tarif_puissance_m = 16.962  # Exemple
    # Extraction des valeurs nécessaires de la ligne
    cons = ligne['Consommation énergétique (kWh)']
    fu = ligne["Facteur d'utilisation (%)"] / 100  # Assurez-vous que fu est un pourcentage correct
    jour_ouvré = ligne['Nombre de jours ouvrés']
    
    if opt == 0:
        if phase == 1:
            puissance = ligne['Puissance phase 1 (kW)']
        elif phase == 2:
            puissance = ligne['Puissance phase 2 (kW)']
        elif phase == 3:
            puissance = ligne['Puissance phase 3 (kW)']
    else:
        if phase == 1:
            puissance = ligne['Puissance optimale phase 1 (kW)']
        elif phase == 2:
            puissance = ligne['Puissance optimale phase 2 (kW)']
        elif phase == 3:
            puissance = ligne['Puissance optimale phase 3 (kW)']
    
   
    if puissance is None:
        raise ValueError(f"Puissance not defined for phase {phase} and opt {opt}")

    # Coûts pour la 1ère tranche de 210 000 kWh  
    cout_tranche1_m=cons * tarif_1ere_tranche_210000kwh
    
    # Coûts pour le reste de l'énergie consommée
    if cons > 210000:
        cout_tranche2_m = cons * tarif_reste_energie 
    else:
        cout_tranche2_m = 0
    
    # Coûts puissance  
    cout_puissance = puissance * tarif_puissance_m    

    cout_total_m = cout_tranche1_m + cout_tranche2_m + cout_puissance
    
    return cout_total_m

def calcul_tarif_G(ligne, phase, opt):
    tarif_acces_reseau=14.344
    tarif_tranche1_15090kwh=0.11518
    tarif_tranche2=0.08650
    tarif_puissance_g=20.522
    # Extraction des valeurs nécessaires de la ligne
    cons = ligne['Consommation énergétique (kWh)']
    fu = ligne["Facteur d'utilisation (%)"] / 100  # Assurez-vous que fu est un pourcentage correct
    jour_ouvré = ligne['Nombre de jours ouvrés']
    
    if opt == 0:
        if phase == 1:
            puissance = ligne['Puissance phase 1 (kW)']
        elif phase == 2:
            puissance = ligne['Puissance phase 2 (kW)']
        elif phase == 3:
            puissance = ligne['Puissance phase 3 (kW)']
    else:
        if phase == 1:
            puissance = ligne['Puissance optimale phase 1 (kW)']
        elif phase == 2:
            puissance = ligne['Puissance optimale phase 2 (kW)']
        elif phase == 3:
            puissance = ligne['Puissance optimale phase 3 (kW)']

    if puissance is None:
        raise ValueError(f"Puissance not defined for phase {phase} and opt {opt}")

    # Frais accès au réseau
    frais_acces_reseau=jour_ouvré*tarif_acces_reseau/30
    
    #coût tranhce 1
    if cons>15090:
        cout_tranche1_g=15090*tarif_tranche1_15090kwh
    else :
        cout_tranche1_g=cons*tarif_tranche1_15090kwh
    # coût tranche 2
    if cons>15090:
        cout_tranche2_g=(cons-15090)*tarif_tranche2
    else :
        cout_tranche2_g=0
    # puissance
    if puissance>50:
        cout_puissance_g=(puissance-50)*tarif_puissance_g
    else :
        cout_puissance_g=0
    # cout total
    cout_total_g = frais_acces_reseau+ cout_tranche1_g + cout_tranche2_g + cout_puissance_g
    
    return cout_total_g

def calcul_tarif_G9(ligne, phase, opt):
    tarif_energie=0.11726
    tarif_puissance_g9=4.921
    # Extraction des valeurs nécessaires de la ligne
    cons = ligne['Consommation énergétique (kWh)']
    fu = ligne["Facteur d'utilisation (%)"] /100  # Assurez-vous que fu est un pourcentage correct
    jour_ouvré = ligne['Nombre de jours ouvrés']
    
    if opt == 0:
        if phase == 1:
            puissance = ligne['Puissance phase 1 (kW)']
        elif phase == 2:
            puissance = ligne['Puissance phase 2 (kW)']
        elif phase == 3:
            puissance = ligne['Puissance phase 3 (kW)']
    else:
        if phase == 1:
            puissance = ligne['Puissance optimale phase 1 (kW)']
        elif phase == 2:
            puissance = ligne['Puissance optimale phase 2 (kW)']
        elif phase == 3:
            puissance = ligne['Puissance optimale phase 3 (kW)']
  
    if puissance is None:
        raise ValueError(f"Puissance not defined for phase {phase} and opt {opt}")
    
    # cout energie
    cout_energie_g9=cons*tarif_energie
    
    #coût puissance 
    cout_puissance_g9=(puissance*tarif_puissance_g9*jour_ouvré)/30
    
    # cout total
    cout_total_g9 = cout_energie_g9+cout_puissance_g9
    
    return cout_total_g9

#def calculer_cout_energetique(df, results_df, phase_index, puissance_col):

    mois_hiver = ['Novembre', 'Décembre', 'Janvier', 'Février', 'Mars']
    mois_ete = ['Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre']

    # Récupérer la consommation pour l'été et l'hiver
    consommation_ete = results_df.iloc[phase_index, 0]  # kWh
    consommation_hiver = results_df.iloc[phase_index, 1]  # kWh
    
    # Fonction interne pour calculer la consommation énergétique par ligne
    def calculer_consommation(row):
        if row['Mois'] in mois_hiver:
            consommation = consommation_hiver
        else:
            consommation = consommation_ete
            # Appliquer un facteur spécifique pour Juillet et Août
            if row['Mois'] in ['Juillet', 'Août']:
                consommation *= 0.10
        cons_energetique = consommation * row['Nombre de jours ouvrés']
        return cons_energetique
    
    # Appliquer la fonction pour créer la nouvelle colonne
    df['Consommation énergétique (kWh)'] = df.apply(calculer_consommation, axis=1)
    
    # Calculer le facteur d'utilisation
    df["Facteur d'utilisation (%)"] = 100 * df['Consommation énergétique (kWh)'] / (df['Nombre de Jours'] * 24 * df[puissance_col])
    
    return df

#def calculer_cout_tarif(tarif_df1, tarif_df2, tarif_df3, results_df):
    # Fonction pour appliquer les calculs tarifaires sur un DataFrame donné
    def appliquer_calculs(tarif_df, phase):
        global opt
        opt = 0
        global phase_global
        phase_global = phase

        # Appliquer les fonctions de calcul des tarifs
        tarif_df['Coût tarif M'] = tarif_df.apply(calcul_tarif_M, axis=1)
        tarif_df['Coût tarif BR'] = tarif_df.apply(calcul_tarif_BR, axis=1)
        tarif_df['Coût tarif G'] = tarif_df.apply(calcul_tarif_G, axis=1)
        tarif_df['Coût tarif G9'] = tarif_df.apply(calcul_tarif_G9, axis=1)
        return tarif_df

    # Appliquer les calculs tarifaires et énergétiques aux trois phases
    tarif_df1 = appliquer_calculs(calculer_cout_energetique(tarif_df1, results_df, 0, 'Puissance phase 1 (kW)'), 1)
    tarif_df2 = appliquer_calculs(calculer_cout_energetique(tarif_df2, results_df, 1, 'Puissance phase 2 (kW)'), 2)
    tarif_df3 = appliquer_calculs(calculer_cout_energetique(tarif_df3, results_df, 2, 'Puissance phase 3 (kW)'), 3)

    # Calculer les coûts totaux pour chaque phase
    var_plot = ["Coût tarif BR", "Coût tarif M", "Coût tarif G", "Coût tarif G9"]
    cout_tarif1 = tarif_df1[var_plot].sum()
    cout_tarif2 = tarif_df2[var_plot].sum()
    cout_tarif3 = tarif_df3[var_plot].sum()

    # Créer un DataFrame avec les résultats
    results_tarif_df = pd.DataFrame({
        'Phase 1': cout_tarif1,
        'Phase 2': cout_tarif2,
        'Phase 3': cout_tarif3
    }).transpose()  # Transpose pour avoir les phases en lignes et les variables en colonnes

    # Retourner le DataFrame en HTML
    if not results_tarif_df.empty:
        return results_tarif_df.to_html(classes='table table-striped', table_id='myTable')
    return None

def appliquer_calculs(tarif_df, phase, opt):
    tarif_df['Coût tarif M'] = tarif_df.apply(lambda row: calcul_tarif_M(row, phase, opt), axis=1)
    tarif_df['Coût tarif BR'] = tarif_df.apply(lambda row: calcul_tarif_BR(row, phase, opt), axis=1)
    tarif_df['Coût tarif G'] = tarif_df.apply(lambda row: calcul_tarif_G(row, phase, opt), axis=1)
    tarif_df['Coût tarif G9'] = tarif_df.apply(lambda row: calcul_tarif_G9(row, phase, opt), axis=1)
    return tarif_df

def calculer_cout_tarif(tarif_df1, tarif_df2, tarif_df3, results_df, opt):
    tarif_df1 = appliquer_calculs(calculer_cout_energetique(tarif_df1, results_df, 0, 'Puissance phase 1 (kW)'), 1, opt)
    tarif_df2 = appliquer_calculs(calculer_cout_energetique(tarif_df2, results_df, 1, 'Puissance phase 2 (kW)'), 2, opt)
    tarif_df3 = appliquer_calculs(calculer_cout_energetique(tarif_df3, results_df, 2, 'Puissance phase 3 (kW)'), 3, opt)

    var_plot = ["Coût tarif BR", "Coût tarif M", "Coût tarif G", "Coût tarif G9"]
    
    cout_tarif1 = tarif_df1[var_plot].sum().round(2)
    cout_tarif2 = tarif_df2[var_plot].sum().round(2)
    cout_tarif3 = tarif_df3[var_plot].sum().round(2)

    # Ajouter le symbole de dollar
    cout_tarif1 = cout_tarif1.apply(lambda x: f"${x:,.2f}")
    cout_tarif2 = cout_tarif2.apply(lambda x: f"${x:,.2f}")
    cout_tarif3 = cout_tarif3.apply(lambda x: f"${x:,.2f}")

    results_tarif_df = pd.DataFrame({
        'Phase 1': cout_tarif1,
        'Phase 2': cout_tarif2,
        'Phase 3': cout_tarif3
    }).transpose()

    if not results_tarif_df.empty:
        return results_tarif_df.to_html(classes='table table-striped', table_id='myTable')
    return None

def calculer_cout_tarif_opt(tarif_df1_opt, tarif_df2_opt, tarif_df3_opt, results_df_opt, opt):
    tarif_df1_opt = appliquer_calculs(calculer_cout_energetique(tarif_df1_opt, results_df_opt, 0, 'Puissance optimale phase 1 (kW)'), 1, opt)
    tarif_df2_opt = appliquer_calculs(calculer_cout_energetique(tarif_df2_opt, results_df_opt, 1, 'Puissance optimale phase 2 (kW)'), 2, opt)
    tarif_df3_opt = appliquer_calculs(calculer_cout_energetique(tarif_df3_opt, results_df_opt, 2, 'Puissance optimale phase 3 (kW)'), 3, opt)

    var_plot = ["Coût tarif BR", "Coût tarif M", "Coût tarif G", "Coût tarif G9"]
    
    cout_tarif1_opt = tarif_df1_opt[var_plot].sum().round(2)
    cout_tarif2_opt = tarif_df2_opt[var_plot].sum().round(2)
    cout_tarif3_opt = tarif_df3_opt[var_plot].sum().round(2)

    # Ajouter le symbole de dollar
    cout_tarif1_opt = cout_tarif1_opt.apply(lambda x: f"${x:,.2f}")
    cout_tarif2_opt = cout_tarif2_opt.apply(lambda x: f"${x:,.2f}")
    cout_tarif3_opt = cout_tarif3_opt.apply(lambda x: f"${x:,.2f}")

    results_tarif_df_opt= pd.DataFrame({
        'Phase 1': cout_tarif1_opt,
        'Phase 2': cout_tarif2_opt,
        'Phase 3': cout_tarif3_opt
    }).transpose()

    if not results_tarif_df_opt.empty:
        return results_tarif_df_opt.to_html(classes='table table-striped', table_id='myTable')
    return None

def heatmap(numeric_df):
    #Correlation usg heatmap
    # Afficher la heatmap de la matrice de corrélation
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="YlGnBu")
    img1 = BytesIO()
    plt.savefig(img1, format='png', bbox_inches='tight')
    img1.seek(0)
    return base64.b64encode(img1.getvalue()).decode('utf8')

def plot_profil_puissance(df1):
    # Génération du second graphique
    # Convertir Date de début comme variable de temps
    df1["Date de début"] = pd.to_datetime(df1["Date de début"])
    # Définir le nom des colonnes
    col_puissance_reelle = 'Puissance réelle (kW)'
    col_pfm = 'Puissance à facturer minimale (PFM)'
    col_temp = 'Température moyenne (°C)'
    # Initialiser la figure et les axes
    plt.figure(figsize=(12, 6))
    # Créer le premier axe
    ax1 = plt.gca()
    # Tracer les lignes pour 'Puissance réelle (kW)' et 'Puissance à facturer minimale (PFM)'
    ax1.plot(df1['Date de début'], df1[col_puissance_reelle], label=col_puissance_reelle, color='blue', linestyle='--', linewidth=4)
    ax1.plot(df1['Date de début'], df1[col_pfm], label=col_pfm, color='orange', linestyle='-', linewidth=4)
    # Paramétrer l'axe des X et des Y pour le premier axe
    #ax1.set_xlabel('Date',fontsize=14)  # Taille de la police pour l'axe X
    ax1.set_ylabel('Kilowatt (kW)', fontsize=14)  # Taille de la police pour l'axe Y
    ax1.tick_params(axis='x', rotation=0)
    ax1.tick_params(axis='y', labelsize=14)  # Taille de la police pour les étiquettes de l'axe Y
    ax1.tick_params(axis='x', labelsize=14) 
    # Créer un second axe pour la température
    ax2 = ax1.twinx()
    # Tracer l'aire pour 'Température moyenne (°C)' sur le second axe
    ax2.fill_between(df1['Date de début'], df1[col_temp], alpha=0.5, color='grey', label=col_temp)
    ax2.set_ylabel('Degrés Celsius (°C)', fontsize=14, color='black')  # Taille de la police pour l'axe Y secondaire
    ax2.tick_params(axis='y', labelcolor='black', labelsize=14)  # Taille de la police pour les étiquettes de l'axe Y secondaire
    # # Ajouter les légendes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=14)
    # Taille de la police pour la légende
    plt.tight_layout()  # Ajuste automatiquement le sous-graphique pour qu'il rentre dans la figure 
    img2 = BytesIO()
    plt.savefig(img2, format='png', bbox_inches='tight')
    img2.seek(0)
    return base64.b64encode(img2.getvalue()).decode('utf8')

def heure_pointe(df1):
    # Convertir la colonne de date et heure en type datetime
    df1['Date et heure de la mesure de la puissance maximale réelle'] = pd.to_datetime(df1['Date et heure de la mesure de la puissance maximale réelle'])
    # Créer une nouvelle colonne 'Heures de pointe' en format float
    df1['Heures de pointe'] = df1['Date et heure de la mesure de la puissance maximale réelle'].dt.hour + \
                         df1['Date et heure de la mesure de la puissance maximale réelle'].dt.minute / 60.0
    
    # Imputer les NaN dans 'Heures de pointe' par la médiane
    median_heures_pointe = df1['Heures de pointe'].median()
    df1['Heures de pointe'].fillna(median_heures_pointe, inplace=True)
    
    
    df1["Intervalle min"]=df1['Heures de pointe'].min() 
    df1["Intervalle max"]=df1['Heures de pointe'].max() 

     # Initialiser la figure et les axes
    plt.figure(figsize=(10, 5))

    # Créer le premier axe
    ax = plt.gca()

    # Tracer les lignes pour 'Puissance réelle (kW)' et 'Puissance à facturer minimale (PFM)'
    ax.plot(df1['Date de début'], df1["Intervalle min"], label="Intervalle min", color='blue', linestyle='-', linewidth=4)
    ax.plot(df1['Date de début'], df1["Intervalle max"], label="Intervalle max", color='orange', linestyle='-', linewidth=4)
    ax.plot(df1['Date de début'], df1["Heures de pointe"], label="Heures de pointe", color='black', linestyle='--', linewidth=4)

    # Paramétrer l'axe des X et des Y pour le premier axe
    #ax1.set_xlabel('Date')
    ax.set_ylabel('Heures', fontsize=14)
    ax.tick_params(axis='x', rotation=0, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, 24)

    # Réglage de la légende comme sur la photo jointe
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=14)

    plt.tight_layout()  # Ajuste automatiquement le sous-graphique pour qu'il rentre dans la figure
    img3 = BytesIO()
    plt.savefig(img3, format='png', bbox_inches='tight')
    img3.seek(0)
    return base64.b64encode(img3.getvalue()).decode('utf8')

def box_plot(df):
    plt.figure(figsize=(10, 6))
    boxprops = dict(linestyle='-', linewidth=3, color='darkgoldenrod')
    medianprops = dict(linestyle='-', linewidth=2.5, color='firebrick')
    flierprops = dict(marker='o', color='green', markersize=8)
    bp = plt.boxplot(df['Heures de pointe'], patch_artist=True, showfliers=True,
                    boxprops=boxprops, medianprops=medianprops, flierprops=flierprops)
    # Calculer les statistiques
    quartiles = np.percentile(df['Heures de pointe'], [25, 50, 75])
    min_val = df['Heures de pointe'].min()
    max_val = df['Heures de pointe'].max()
    whiskers = [item.get_ydata() for item in bp['whiskers']]
    # Ajouter les valeurs des statistiques sur le graphique
    x_position = 1.08  # Légèrement à droite de la boîte
    plt.text(x_position, min_val, f'Min: {min_val:.2f}', ha='left', va='center', fontsize=10)
    plt.text(x_position, max_val, f'Max: {max_val:.2f}', ha='left', va='center', fontsize=10)
    plt.text(x_position, quartiles[0], f'Q1: {quartiles[0]:.2f}', ha='left', va='center', fontsize=10)
    plt.text(x_position, quartiles[2], f'Q3: {quartiles[2]:.2f}', ha='left', va='center', fontsize=10)
    plt.text(x_position, quartiles[1], f'Médiane: {quartiles[1]:.2f}', ha='left', va='center', fontsize=10, color='firebrick')
    # Paramètres de la figure
    plt.ylabel('Heures')
    plt.xticks([1], ['Heures de pointe'])
    img6 = BytesIO()
    plt.savefig(img6, format='png', bbox_inches='tight')
    img6.seek(0)
    return base64.b64encode(img6.getvalue()).decode('utf8')

def heatmap_df2(df2):
    #Correlation using heatmap
    plt.figure(figsize=(8, 5))
    sns.heatmap(df2.corr(), annot = True, cmap="YlGnBu")
    img4 = BytesIO()
    plt.savefig(img4, format='png', bbox_inches='tight')
    img4.seek(0)
    return base64.b64encode(img4.getvalue()).decode('utf8')

def appel_puissance(df2):
    # Convertir la colonne de date et heure en type datetime
    df2['Date et heure'] = pd.to_datetime(df2['Date et heure'])
    # Créer une nouvelle colonne 'Heures de pointe' en format float
    df2['Heures'] = df2['Date et heure'].dt.hour + \
                         df2['Date et heure'].dt.minute / 60.0
    # Initialiser la figure et les axes
    plt.figure(figsize=(12, 6))
    # Créer le premier axe
    ax = plt.gca()
    # Tracer les lignes pour chaque saison
    ax.plot(df2['Heures'], df2["Hiver"], label="Hiver", color='red', linestyle='--', linewidth=4)
    ax.plot(df2['Heures'], df2["Printemps"], label="Printemps", color='black', linestyle='-', linewidth=4)
    ax.plot(df2['Heures'], df2["Été"], label="Été", color='blue', linestyle=':', linewidth=4)
    ax.plot(df2['Heures'], df2["Automne"], label="Automne", color='green', linestyle='-.', linewidth=4)
    # Paramétrer l'axe des Y pour le premier axe
    ax.set_ylabel('Puissance (kW)', fontsize=14)
    ax.set_xlabel('Heures', fontsize=14)
    ax.tick_params(axis='x', rotation=0, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    # Trouver et marquer les valeurs maximales pour chaque saison avec une transparence
    for saison, couleur in zip(["Hiver", "Printemps", "Été", "Automne"], ['red', 'black', 'blue', 'green']):
        # Trouver l'index de l'heure avec la valeur maximale pour la saison
        max_idx = df2[saison].idxmax()
        # Trouver l'heure correspondante
        max_heure = df2['Heures'][max_idx]
        # Tracer une ligne verticale à cette heure avec une transparence (alpha) de 0.5 par exemple
        plt.axvline(x=max_heure, color=couleur, linestyle='-', linewidth=5, alpha=0.5, label=f'Max {saison}')
    # Réglage de la légende
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=4, fontsize=14)
    # Vous pouvez ajuster 'start' et 'end' en fonction de votre plage de données spécifique
    start = int(df2['Heures'].min())  # Début de l'axe des X basé sur vos données
    end = int(df2['Heures'].max()) + 1  # Fin de l'axe des X, +1 pour inclure la dernière valeur
    plt.xticks(np.arange(start, end, 1))  # Générer des marques d'intervalle de 1 entre 'start' et 'end'
    plt.tight_layout()  # Ajuste automatiquement le sous-graphique pour qu'il rentre dans la figure
    img5 = BytesIO()
    plt.savefig(img5, format='png', bbox_inches='tight')
    img5.seek(0)
    return base64.b64encode(img5.getvalue()).decode('utf8')

def cons_puissance_autobus(df):
    # Création de phase 
    phase_1 = df[(df['Année conversion Electrique (trajet)'] >= 1) & (df['Année conversion Electrique (trajet)'] <= 3)]
    phase_2 = df[(df['Année conversion Electrique (trajet)'] >= 1) & (df['Année conversion Electrique (trajet)'] <= 6)]
    phase_3 = df
    var_phase=["Consomation kWh/jour (été)", "Consomation kWh/jour (hiver)", "Puissance de la borne calculé pour la recharge (kW)"]
    # Calculer les sommes pour chaque phase
    sum_phase_1 = phase_1[var_phase].sum()
    sum_phase_1 =sum_phase_1.round(2)
    sum_phase_2 = phase_2[var_phase].sum()
    sum_phase_2 =sum_phase_2.round(2)
    sum_phase_3 = phase_3[var_phase].sum()
    sum_phase_3 =sum_phase_3.round(2)
    # Créer un DataFrame avec ces résultats
    results_df = pd.DataFrame({
        'Phase 1': sum_phase_1,
        'Phase 2': sum_phase_2,
        'Phase 3': sum_phase_3
    }).transpose()  # Transpose pour avoir les phases en lignes et les variables en colonnes
    if not results_df.empty:
        return results_df.to_html(classes='table table-striped', table_id='myTable', justify='center')
    return None

def cons_puissance_autobus_opt(df):  
    #
    df["Temps de recharge dispo AM"]=df["Début du trajet de l'après-midi"]-df["Fin du trajet du matin"]
    df["Temps de recharge dispo PM"]=24-(df["Fin du trajet de l'après-midi"]-df["Début du trajet du matin"])
    #
    df["Durée trajet AM"]=df["Fin du trajet du matin"]-df["Début du trajet du matin"]
    df["Durée trajet PM"]=df["Fin du trajet de l'après-midi"]-df["Début du trajet de l'après-midi"]
    df["Durée trajet journée"]=df["Durée trajet AM"] + df["Durée trajet PM"]
    #
    df["Km parcouru AM"]=df["Confirmation Km/Jour"]*df["Durée trajet AM"]/df["Durée trajet journée"]
    df["Km parcouru PM"]=df["Confirmation Km/Jour"]*df["Durée trajet PM"]/df["Durée trajet journée"]
    #
    df["kWh/km été"]=df["Consomation kWh/jour (été)"]/df["Confirmation Km/Jour"]
    df["kWh/km hiver"]=df["Consomation kWh/jour (hiver)"]/df["Confirmation Km/Jour"]
    #
    df["Consommation AM hiver"]=df["kWh/km été"]*df["Km parcouru AM"]
    df["Consommation PM hiver"]=df["kWh/km hiver"]*df["Km parcouru PM"]

    #
    def puissance_bornes_AM(ligne):
        cons_AM=ligne["Consommation AM hiver"]
        temps_rech_AM=ligne["Temps de recharge dispo AM"]
        puissance_borne=ligne["Puissance de la borne calculé pour la recharge (kW)"]
        if (cons_AM/temps_rech_AM)>puissance_borne:
            puissance_AM=puissance_borne
        else :
            puissance_AM=cons_AM/temps_rech_AM
        return puissance_AM
    df["Puissance nécessaire de bornes AM"]=df.apply(puissance_bornes_AM, axis=1)
    #
    def puissance_bornes_PM(ligne):
        cons_PM=ligne["Consommation PM hiver"]
        temps_rech_PM=ligne["Temps de recharge dispo PM"]
        puissance_borne=ligne["Puissance de la borne calculé pour la recharge (kW)"]
        if (cons_PM/temps_rech_PM)>puissance_borne:
            puissance_PM=puissance_borne
        else :
            puissance_PM=cons_PM/temps_rech_PM
        return puissance_PM
    df["Puissance nécessaire de bornes PM"]=df.apply(puissance_bornes_PM, axis=1)
    #
    df["Puissance appelée optimale"]=df[["Puissance nécessaire de bornes AM","Puissance nécessaire de bornes PM"]].max(axis=1)


    # Création de phase 
    phase_1_opt = df[(df['Année conversion Electrique (trajet)'] >= 1) & (df['Année conversion Electrique (trajet)'] <= 3)]
    phase_2_opt = df[(df['Année conversion Electrique (trajet)'] >= 1) & (df['Année conversion Electrique (trajet)'] <= 6)]
    phase_3_opt = df
    var_phase=["Consomation kWh/jour (été)", "Consomation kWh/jour (hiver)", "Puissance appelée optimale"]
    # Calculer les sommes pour chaque phase
    sum_phase_1_opt = phase_1_opt[var_phase].sum()
    sum_phase_1_opt =sum_phase_1_opt.round(2)
    sum_phase_2_opt = phase_2_opt[var_phase].sum()
    sum_phase_2_opt =sum_phase_2_opt.round(2)
    sum_phase_3_opt = phase_3_opt[var_phase].sum()
    sum_phase_3_opt =sum_phase_3_opt.round(2)
    # Créer un DataFrame avec ces résultats
    results_df = pd.DataFrame({
        'Phase 1': sum_phase_1_opt,
        'Phase 2': sum_phase_2_opt,
        'Phase 3': sum_phase_3_opt
    }).transpose()  # Transpose pour avoir les phases en lignes et les variables en colonnes
    if not results_df.empty:
        return results_df.to_html(classes='table table-striped', table_id='myTable')
    return None

def cal_result_df(df):
    # Création de phase 
    phase_1 = df[(df['Année conversion Electrique (trajet)'] >= 1) & (df['Année conversion Electrique (trajet)'] <= 3)]
    phase_2 = df[(df['Année conversion Electrique (trajet)'] >= 1) & (df['Année conversion Electrique (trajet)'] <= 6)]
    phase_3 = df
    var_phase=["Consomation kWh/jour (été)", "Consomation kWh/jour (hiver)", "Puissance de la borne calculé pour la recharge (kW)"]
    # Calculer les sommes pour chaque phase
    sum_phase_1 = phase_1[var_phase].sum()
    sum_phase_1 =sum_phase_1.round(2)
    sum_phase_2 = phase_2[var_phase].sum()
    sum_phase_2 =sum_phase_2.round(2)
    sum_phase_3 = phase_3[var_phase].sum()
    sum_phase_3 =sum_phase_3.round(2)
    # Créer un DataFrame avec ces résultats
    results_df = pd.DataFrame({
        'Phase 1': sum_phase_1,
        'Phase 2': sum_phase_2,
        'Phase 3': sum_phase_3
    }).transpose()  # Transpose pour avoir les phases en lignes et les variables en colonnes
    return results_df

def cal_result_df_opt(df):
    # Création de phase 
    phase_1 = df[(df['Année conversion Electrique (trajet)'] >= 1) & (df['Année conversion Electrique (trajet)'] <= 3)]
    phase_2 = df[(df['Année conversion Electrique (trajet)'] >= 1) & (df['Année conversion Electrique (trajet)'] <= 6)]
    phase_3 = df
    var_phase=["Consomation kWh/jour (été)", "Consomation kWh/jour (hiver)", "Puissance appelée optimale"]
    # Calculer les sommes pour chaque phase
    sum_phase_1 = phase_1[var_phase].sum()
    sum_phase_1 =sum_phase_1.round(2)
    sum_phase_2 = phase_2[var_phase].sum()
    sum_phase_2 =sum_phase_2.round(2)
    sum_phase_3 = phase_3[var_phase].sum()
    sum_phase_3 =sum_phase_3.round(2)
    # Créer un DataFrame avec ces résultats
    results_df = pd.DataFrame({
        'Phase 1': sum_phase_1,
        'Phase 2': sum_phase_2,
        'Phase 3': sum_phase_3
    }).transpose()  # Transpose pour avoir les phases en lignes et les variables en colonnes
    return results_df

def cal_result_df_munic(df):
    # Création de phase 
    phase_1 = df[(df['Année conversion Electrique'] >= 1) & (df['Année conversion Electrique'] <= 3)]
    phase_2 = df[(df['Année conversion Electrique'] >= 1) & (df['Année conversion Electrique'] <= 6)]
    phase_3 = df
    var_phase=["Consomation kWh/jour (été)", "Consomation kWh/jour (hiver)", "Puissance de la borne calculé pour la recharge (kW)"]
    # Calculer les sommes pour chaque phase
    sum_phase_1 = phase_1[var_phase].sum()
    sum_phase_1 =sum_phase_1.round(2)
    sum_phase_2 = phase_2[var_phase].sum()
    sum_phase_2 =sum_phase_2.round(2)
    sum_phase_3 = phase_3[var_phase].sum()
    sum_phase_3 =sum_phase_3.round(2)
    # Créer un DataFrame avec ces résultats
    results_df = pd.DataFrame({
        'Phase 1': sum_phase_1,
        'Phase 2': sum_phase_2,
        'Phase 3': sum_phase_3
    }).transpose()  # Transpose pour avoir les phases en lignes et les variables en colonnes
    return results_df

def cons_puissance_munic(df):
    # Création de phase 
    phase_1 = df[(df['Année conversion Electrique'] >= 1) & (df['Année conversion Electrique'] <= 3)]
    phase_2 = df[(df['Année conversion Electrique'] >= 1) & (df['Année conversion Electrique'] <= 6)]
    phase_3 = df
    var_phase=["Consomation kWh/jour (été)", "Consomation kWh/jour (hiver)", "Puissance de la borne calculé pour la recharge (kW)"]
    # Calculer les sommes pour chaque phase
    sum_phase_1 = phase_1[var_phase].sum()
    sum_phase_1 =sum_phase_1.round(2)
    sum_phase_2 = phase_2[var_phase].sum()
    sum_phase_2 =sum_phase_2.round(2)
    sum_phase_3 = phase_3[var_phase].sum()
    sum_phase_3 =sum_phase_3.round(2)
    # Créer un DataFrame avec ces résultats
    results_df = pd.DataFrame({
        'Phase 1': sum_phase_1,
        'Phase 2': sum_phase_2,
        'Phase 3': sum_phase_3
    }).transpose()  # Transpose pour avoir les phases en lignes et les variables en colonnes
    if not results_df.empty:
        return results_df.to_html(classes='table table-striped', table_id='myTable')
    return None

def tab_ap_puissance(data, results_df):
    # Initialiser le DataFrame final
    final_results = pd.DataFrame()
    # Liste des phases
    phases = ['Phase 1', 'Phase 2', 'Phase 3']
    # Traiter chaque phase
    for i, phase in enumerate(phases):
        phase_data = data.copy()  # Copie du DataFrame pour chaque phase
        # Calcul des bornes pour chaque phase
        coeff = results_df.iloc[i, 0] * 0.9
        phase_data['Bornes'] = np.where((phase_data['Heures'] >= 23.25) & (phase_data['Heures'] <= 23.75), coeff, 0)
        phase_data['Bornes'] = np.where((phase_data['Heures'] >= 22.75) & (phase_data['Heures'] <= 23.00), coeff, phase_data['Bornes'])
        # Sauvegarder les valeurs maximales des bornes
        final_results.loc[phase, 'Bornes'] = phase_data['Bornes'].max()
        # Calculer la consommation totale pour chaque saison
        for season in ['Hiver', 'Printemps', 'Été', 'Automne']:
            phase_data[f'Batiment+Bornes {season}'] = phase_data['Bornes'] + phase_data[season]
            final_results.loc[phase, season] = phase_data[season].max()  # Puissance maximale saisonnière
            final_results.loc[phase, f'Batiment+Bornes {season}'] = phase_data[f'Batiment+Bornes {season}'].max()  # Puissance maximale combinée
    if not final_results.empty:
        return final_results.to_html(classes='table table-striped', table_id='myTable')
    return None

def tarif_plot(tarif_df1, tarif_df2, tarif_df3):
    plt.figure(figsize=(12, 6))
    var_plot = ["Coût tarif BR", "Coût tarif M", "Coût tarif G", "Coût tarif G9"]
    cout_tarif1 = tarif_df1[var_plot].sum()
    cout_tarif2 = tarif_df2[var_plot].sum()
    cout_tarif3 = tarif_df3[var_plot].sum()

    # Création d'un DataFrame avec les résultats
    results_tarif_df = pd.DataFrame({
        'phase 1': cout_tarif1,
        'phase 2': cout_tarif2,
        'phase 3': cout_tarif3
    }).transpose()

    fig, ax = plt.subplots(figsize=(10, 6))  # Augmentation de la taille de la figure

    width = 0.25  # Largeur des barres
    ind = np.arange(len(var_plot))  # les emplacements x pour les groupes

    # Tracé des barres
    bars1 = ax.bar(ind - width, results_tarif_df.loc['phase 1', :], width, label='phase 1')
    bars2 = ax.bar(ind, results_tarif_df.loc['phase 2', :], width, label='phase 2')
    bars3 = ax.bar(ind + width, results_tarif_df.loc['phase 3', :], width, label='phase 3')

    ax.set_ylabel('Coût ($)', fontsize=14)
    ax.set_title('Coûts énergétiques annuels sans contrôle de puissance', fontsize=14)
    ax.set_xticks(ind)
    ax.set_xticklabels(var_plot, rotation=0)  # Rotation des étiquettes pour éviter le chevauchement

    max_height = results_tarif_df.max().max()
    ax.set_ylim(0, max_height * 1.2)  # Ajout d'une marge supérieure pour les annotations

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'${height:,.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Décalage vertical des annotations
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    # Réglage de la légende
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=14)

    plt.tight_layout()  # Ajuste automatiquement le sous-graphique pour qu'il rentre dans la figure
    img7 = BytesIO()
    plt.savefig(img7, format='png', bbox_inches='tight')
    img7.seek(0)
    return base64.b64encode(img7.getvalue()).decode('utf8')

def tarif_plot_opt(tarif_df1, tarif_df2, tarif_df3):
    plt.figure(figsize=(12, 6))
    var_plot = ["Coût tarif BR", "Coût tarif M", "Coût tarif G", "Coût tarif G9"]
    cout_tarif1 = tarif_df1[var_plot].sum()
    cout_tarif2 = tarif_df2[var_plot].sum()
    cout_tarif3 = tarif_df3[var_plot].sum()

    # Création d'un DataFrame avec les résultats
    results_tarif_df = pd.DataFrame({
        'phase 1': cout_tarif1,
        'phase 2': cout_tarif2,
        'phase 3': cout_tarif3
    }).transpose()

    fig, ax = plt.subplots(figsize=(10, 6))  # Augmentation de la taille de la figure

    width = 0.25  # Largeur des barres
    ind = np.arange(len(var_plot))  # les emplacements x pour les groupes

    # Tracé des barres
    bars1 = ax.bar(ind - width, results_tarif_df.loc['phase 1', :], width, label='phase 1')
    bars2 = ax.bar(ind, results_tarif_df.loc['phase 2', :], width, label='phase 2')
    bars3 = ax.bar(ind + width, results_tarif_df.loc['phase 3', :], width, label='phase 3')

    ax.set_ylabel('Coût ($)', fontsize=14)
    ax.set_title('Coûts énergétiques annuels avec contrôle de puissance', fontsize=14)
    ax.set_xticks(ind)
    ax.set_xticklabels(var_plot, rotation=0)  # Rotation des étiquettes pour éviter le chevauchement

    max_height = results_tarif_df.max().max()
    ax.set_ylim(0, max_height * 1.2)  # Ajout d'une marge supérieure pour les annotations

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'${height:,.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Décalage vertical des annotations
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    # Réglage de la légende
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=14)

    plt.tight_layout()  # Ajuste automatiquement le sous-graphique pour qu'il rentre dans la figure
    img8 = BytesIO()
    plt.savefig(img8, format='png', bbox_inches='tight')
    img8.seek(0)
    return base64.b64encode(img8.getvalue()).decode('utf8')

def tarif_plot_munic(tarif_df1, tarif_df2, tarif_df3):
    plt.figure(figsize=(12, 6))
    var_plot = ["Coût tarif BR", "Coût tarif M", "Coût tarif G", "Coût tarif G9"]
    cout_tarif1 = tarif_df1[var_plot].sum()
    cout_tarif2 = tarif_df2[var_plot].sum()
    cout_tarif3 = tarif_df3[var_plot].sum()

    # Création d'un DataFrame avec les résultats
    results_tarif_df = pd.DataFrame({
        'phase 1': cout_tarif1,
        'phase 2': cout_tarif2,
        'phase 3': cout_tarif3
    }).transpose()

    fig, ax = plt.subplots(figsize=(10, 6))  # Augmentation de la taille de la figure

    width = 0.25  # Largeur des barres
    ind = np.arange(len(var_plot))  # les emplacements x pour les groupes

    # Tracé des barres
    bars1 = ax.bar(ind - width, results_tarif_df.loc['phase 1', :], width, label='phase 1')
    bars2 = ax.bar(ind, results_tarif_df.loc['phase 2', :], width, label='phase 2')
    bars3 = ax.bar(ind + width, results_tarif_df.loc['phase 3', :], width, label='phase 3')

    ax.set_ylabel('Coût ($)', fontsize=14)
    ax.set_title('Coûts énergétiques annuels sans contrôle de puissance', fontsize=14)
    ax.set_xticks(ind)
    ax.set_xticklabels(var_plot, rotation=0)  # Rotation des étiquettes pour éviter le chevauchement

    max_height = results_tarif_df.max().max()
    ax.set_ylim(0, max_height * 1.2)  # Ajout d'une marge supérieure pour les annotations

    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'${height:,.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Décalage vertical des annotations
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)

    # Réglage de la légende
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=14)

    plt.tight_layout()  # Ajuste automatiquement le sous-graphique pour qu'il rentre dans la figure
    img7 = BytesIO()
    plt.savefig(img7, format='png', bbox_inches='tight')
    img7.seek(0)
    return base64.b64encode(img7.getvalue()).decode('utf8')

#def generate_phase_results(data, df2):
    def process_phase(data, value, phase_label):
        # Convertir Date et heure en datetime si ce n'est pas déjà fait
        data['Date et heure'] = pd.to_datetime(data['Date et heure'])
        data['Heures'] = data['Date et heure'].dt.hour + data['Date et heure'].dt.minute / 60.0
        
        # S'assurer que la valeur est numérique
        try:
            numeric_value = pd.to_numeric(value)
        except ValueError:
            raise ValueError(f"La valeur {value} pour {phase_label} n'est pas convertible en numérique.")

        # Calculs des bornes
        data[f'Bornes {phase_label}'] = np.where((data['Heures'] >= 23.25) & (data['Heures'] <= 23.75), numeric_value * 0.9, 0)
        data[f'Bornes {phase_label}'] = np.where((data['Heures'] >= 22.75) & (data['Heures'] <= 23.00), numeric_value * 0.9, data[f'Bornes {phase_label}'])

        # Calculs pour chaque saison
        for season in ['Hiver', 'Printemps', 'Été', 'Automne']:
            data[f'Batiment+Bornes {phase_label} {season}'] = data[f'Bornes {phase_label}'] + data[season]

        max_power = data[[f'Batiment+Bornes {phase_label} {season}' for season in ['Hiver', 'Printemps', 'Été', 'Automne']]].max().max()
        return pd.DataFrame({f'Puissance maximale {phase_label}': [max_power]})

    results = {}
    for i, phase in enumerate(['Phase 1', 'Phase 2', 'Phase 3'], start=0):
        results[phase] = process_phase(data.copy(), df2.iloc[i, 0], phase)

    final_results = pd.concat(results, axis=1)
    return final_results.to_html(classes='table table-striped', table_id='myTable')

#def heure_pointe(df1):
    # Tenter de convertir en datetime, ignorer les erreurs pour les valeurs non convertibles
    df1['Date et heure de la mesure de la puissance maximale réelle'] = pd.to_datetime(df1['Date et heure de la mesure de la puissance maximale réelle'], errors='coerce')
    
    # Calculer les heures de pointe à partir des valeurs datetime valides
    df1['Heures de pointe'] = df1['Date et heure de la mesure de la puissance maximale réelle'].dt.hour + \
                         df1['Date et heure de la mesure de la puissance maximale réelle'].dt.minute / 60.0
    
    # Imputer les NaN dans 'Heures de pointe' par la médiane
    median_heures_pointe = df1['Heures de pointe'].median()
    df1['Heures de pointe'].fillna(median_heures_pointe, inplace=True)
    
    # Calculer les intervalles min et max
    df1["Intervalle min"] = df1['Heures de pointe'].min()
    df1["Intervalle max"] = df1['Heures de pointe'].max()

    # Initialiser la figure et les axes
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    
    # Tracer les lignes pour les intervalles
    ax.plot(df1['Date et heure de la mesure de la puissance maximale réelle'], df1["Intervalle min"], label="Intervalle min", color='blue', linestyle='-', linewidth=2)
    ax.plot(df1['Date et heure de la mesure de la puissance maximale réelle'], df1["Intervalle max"], label="Intervalle max", color='orange', linestyle='-', linewidth=2)
    ax.plot(df1['Date et heure de la mesure de la puissance maximale réelle'], df1["Heures de pointe"], label="Heures de pointe", color='black', linestyle='--', linewidth=2)

    # Paramétrer les axes
    ax.set_ylabel('Heures', fontsize=14)
    ax.tick_params(axis='x', rotation=0, labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(0, 24)
    
    # Réglage de la légende
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=14)

    plt.tight_layout()  # Ajuste automatiquement le sous-graphique pour qu'il rentre dans la figure
    img3 = BytesIO()
    plt.savefig(img3, format='png', bbox_inches='tight')
    img3.seek(0)
    return base64.b64encode(img3.getvalue()).decode('utf8')

@app.route('/')
def accueuil():
    # Générer ou récupérer le HTML du tableau
    return render_template('accueuil.html')

@app.route('/autobus', methods=['GET', 'POST'])
def autobus():
    plot_url = plot_url2 = plot_url3 = plot_url4 = plot_url5 =plot_url6= plot_url_7 = plot_url_8 =  stats = stats2 = stats3= stats4= results_tarif_html = results_tarif_html_opt = carbone=None
    if request.method == 'POST':
    # Initialisation des dataframes
        df1 = df2 = df3 = pd.DataFrame()
        numeric_df = pd.DataFrame()

        file_type = request.form['fileType']
        file = request.files.get('file')      
        if file:
            if file_type == 'filePeriode':          
                df1 = prepare_data(file)
                numeric_df=numeric_data(df1)
                stats = describe_data(numeric_df)
                plot_url = heatmap(numeric_df)
                plot_url2 = plot_profil_puissance(df1)
                plot_url3 = heure_pointe(df1)  
                plot_url6=box_plot(df1)            
            elif file_type== 'fileAppel':
                df2= prepare_data2(file)
                numeric_df2=numeric_data(df2)
                plot_url4= heatmap_df2(numeric_df2)
                plot_url5=appel_puissance(df2)               
            elif file_type== 'fileFlotte':     
                # Obtenir les années de conversion depuis le formulaire
                df3 = prepare_data3(file)
                stats2=describe_df3(df3)
                stats3=cons_puissance_autobus(df3) #resultat_df
                result_df=cal_result_df(df3)
                carbone=credit_carbone(df3)
                stats4=cons_puissance_autobus_opt(df3)
                result_df_opt=cal_result_df_opt(df3)
                # Initialiser les données de tarif_df
                data_cal = {
                    'Mois': ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre'],
                    'Nombre de Jours': [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                    'Nombre de Jours ouvrables 2024': [22, 21, 21, 21, 19, 20, 23, 21, 21, 23, 19, 21]
                }
                tarif_df = pd.DataFrame(data_cal)
                tarif_df['Nombre de jours ouvrés'] = round(180 * tarif_df['Nombre de Jours ouvrables 2024'] / tarif_df['Nombre de Jours ouvrables 2024'].sum())
                # Créer des copies pour chaque phase
                tarif_df1 = tarif_df.copy()
                tarif_df2 = tarif_df.copy()
                tarif_df3 = tarif_df.copy()

                tarif_df1_opt = tarif_df.copy()
                tarif_df2_opt = tarif_df.copy()
                tarif_df3_opt = tarif_df.copy()

                # Ajouter les puissances pour chaque phase
                tarif_df1["Puissance phase 1 (kW)"] = result_df.iloc[0, 2]
                tarif_df2["Puissance phase 2 (kW)"] = result_df.iloc[1, 2]
                tarif_df3["Puissance phase 3 (kW)"] = result_df.iloc[2, 2]

                tarif_df1_opt["Puissance optimale phase 1 (kW)"] = result_df_opt.iloc[0, 2]
                tarif_df2_opt["Puissance optimale phase 2 (kW)"] = result_df_opt.iloc[1, 2]
                tarif_df3_opt["Puissance optimale phase 3 (kW)"] = result_df_opt.iloc[2, 2]

                # Calculer les coûts tarifaires
                results_tarif_html = calculer_cout_tarif(tarif_df1, tarif_df2, tarif_df3, result_df, 0)
                plot_url_7=tarif_plot(tarif_df1, tarif_df2, tarif_df3)
                results_tarif_html_opt = calculer_cout_tarif_opt(tarif_df1_opt, tarif_df2_opt, tarif_df3_opt, result_df_opt, 1)
                plot_url_8=tarif_plot_opt(tarif_df1_opt, tarif_df2_opt, tarif_df3_opt)

    return render_template('autobus.html', 
                           stats=stats, 
                           stats2=stats2,
                           stats3=stats3,
                           stats4=stats4,
                           results_tarif_html=results_tarif_html,
                           results_tarif_html_opt=results_tarif_html_opt,
                           carbone=carbone,
                           plot_url=plot_url, 
                           plot_url2=plot_url2, 
                           plot_url3=plot_url3, 
                           plot_url4=plot_url4, 
                           plot_url5=plot_url5,
                           plot_url6=plot_url6,
                           plot_url_7=plot_url_7,
                           plot_url_8=plot_url_8
                           )
    
@app.route('/municipalite', methods=['GET', 'POST'])
def municipalite():
    plot_url = plot_url2 = plot_url3 = plot_url4 = plot_url5 = plot_url6 = plot_url7=stats = stats2 = stats3= carbone=results_tarif_html=results_tarif_html_opt=resultats_html_ap=None
    if request.method == 'POST':
        # Initialisation des dataframes
        df1 = df2 = df3 = pd.DataFrame()
        numeric_df = pd.DataFrame()    

        file_type = request.form['fileType']
        file = request.files.get('file')      
        if file:
            if file_type == 'filePeriode':          
                df1 = prepare_data(file)
                numeric_df=numeric_data(df1)
                stats = describe_data(numeric_df)
                plot_url = heatmap(numeric_df)
                plot_url2 = plot_profil_puissance(df1)
                plot_url3 = heure_pointe(df1)  
                plot_url6=box_plot(df1)            
            elif file_type== 'fileAppel':
                df2= prepare_data2(file)
                numeric_df2=numeric_data(df2)
                plot_url4= heatmap_df2(numeric_df2)
                plot_url5=appel_puissance(df2)  
                results_df = {
                    'Puissance de Bornes': [72, 120, 200]
                }
                index_labels = ['Phase 1', 'Phase 2', 'Phase 3']
                results_df = pd.DataFrame(results_df, index=index_labels)
                data=df2       
                data['Date et heure'] = pd.to_datetime(
                data['Date et heure'])
                # Créer une nouvelle colonne 'Heures de pointe' en format float
                data['Heures'] = data['Date et heure'].dt.hour + \
                         data['Date et heure'].dt.minute / 60.0
                data=data.drop('Date et heure', axis=1)
                resultats_html_ap = tab_ap_puissance(data, results_df)
            elif file_type== 'fileFlotte':
                df3 = prepare_data3(file)
                stats2=describe_df3_munic(df3)
                stats3=cons_puissance_munic(df3)
                result_df=cal_result_df_munic(df3)
                carbone=credit_carbone_munic(df3)
                # Initialiser les données de tarif_df
                data_cal = {
                    'Mois': ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre'],
                    'Nombre de Jours': [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
                    'Nombre de Jours ouvrables 2024': [22, 21, 21, 21, 19, 20, 23, 21, 21, 23, 19, 21]
                }
                tarif_df = pd.DataFrame(data_cal)
                tarif_df['Nombre de jours ouvrés'] = round(180 * tarif_df['Nombre de Jours ouvrables 2024'] / tarif_df['Nombre de Jours ouvrables 2024'].sum())
                # Créer des copies pour chaque phase
                tarif_df1 = tarif_df.copy()
                tarif_df2 = tarif_df.copy()
                tarif_df3 = tarif_df.copy()
                 # Ajouter les puissances pour chaque phase
                tarif_df1["Puissance phase 1 (kW)"] = result_df.iloc[0, 2]
                tarif_df2["Puissance phase 2 (kW)"] = result_df.iloc[1, 2]
                tarif_df3["Puissance phase 3 (kW)"] = result_df.iloc[2, 2]

                tarif_df1_opt = tarif_df.copy()
                tarif_df2_opt = tarif_df.copy()
                tarif_df3_opt = tarif_df.copy()

                result_df_opt=cal_result_df_munic(df)
                 # Calculer les coûts tarifaires
                results_tarif_html = calculer_cout_tarif(tarif_df1, tarif_df2, tarif_df3, result_df, 0)
                results_tarif_html_opt = calculer_cout_tarif(tarif_df1_opt, tarif_df2_opt, tarif_df3_opt, result_df_opt, 1)
                plot_url7=tarif_plot(tarif_df1, tarif_df2, tarif_df3)
    # Générer ou récupérer le HTML du graphique
    return render_template('municipalite.html', 
                           stats=stats,
                           stats2=stats2,
                           stats3=stats3, 
                           results_tarif_html=results_tarif_html,
                           results_html_ap=resultats_html_ap,
                           results_tarif_html_opt=results_tarif_html_opt,
                           carbone=carbone,
                           plot_url=plot_url, 
                           plot_url2=plot_url2, 
                           plot_url3=plot_url3,
                           plot_url4=plot_url4,
                           plot_url5=plot_url5,
                           plot_url6=plot_url6,
                           plot_url7=plot_url7
                           )

@app.route('/concessionnaire', methods=['GET', 'POST'])
def concessionnaire():
    plot_url = plot_url2 = plot_url3 = plot_url4 = plot_url5 = plot_url6 = stats = stats2 = stats3= None
    if request.method == 'POST':
        # Initialisation des dataframes
        df1 = df2 = df3 = pd.DataFrame()
        numeric_df = pd.DataFrame()    

        file_type = request.form['fileType']
        file = request.files.get('file')      
        if file:
            if file_type == 'filePeriode':          
                df1 = prepare_data(file)
                numeric_df=numeric_data(df1)
                stats = describe_data(numeric_df)
                plot_url = heatmap(numeric_df)
                plot_url2 = plot_profil_puissance(df1)
                plot_url3 = heure_pointe(df1)  
                plot_url6=box_plot(df1)            
            elif file_type== 'fileAppel':
                df2= prepare_data2(file)
                numeric_df2=numeric_data(df2)
                plot_url4= heatmap_df2(numeric_df2)
                plot_url5=appel_puissance(df2)               
            elif file_type== 'fileElec':
                df3 = prepare_data_elec(file)
                stats2=describe_elec(df3)
    # Générer ou récupérer le HTML du graphique
    return render_template('concessionnaire.html', 
                           stats=stats,
                           stats2=stats2,
                           stats3=stats3, 
                           plot_url=plot_url, 
                           plot_url2=plot_url2, 
                           plot_url3=plot_url3,
                           plot_url4=plot_url4,
                           plot_url5=plot_url5,
                           plot_url6=plot_url6)

@app.route('/immobilier', methods=['GET', 'POST'])
def immobilier():
    plot_url = plot_url2 = plot_url3 = plot_url4 = plot_url5 = plot_url6 = stats = stats2 = stats3= None
    if request.method == 'POST':
        # Initialisation des dataframes
        df1 = df2 = df3 = pd.DataFrame()
        numeric_df = pd.DataFrame()    

        file_type = request.form['fileType']
        file = request.files.get('file')      
        if file:
            if file_type == 'filePeriode':          
                df1 = prepare_data(file)
                numeric_df=numeric_data(df1)
                stats = describe_data(numeric_df)
                plot_url = heatmap(numeric_df)
                plot_url2 = plot_profil_puissance(df1)
                plot_url3 = heure_pointe(df1)  
                plot_url6=box_plot(df1)            
            elif file_type== 'fileAppel':
                df2= prepare_data2(file)
                numeric_df2=numeric_data(df2)
                plot_url4= heatmap_df2(numeric_df2)
                plot_url5=appel_puissance(df2)               
    # Générer ou récupérer le HTML du graphique
    return render_template('immobilier.html', 
                           stats=stats,
                           stats2=stats2,
                           stats3=stats3, 
                           plot_url=plot_url, 
                           plot_url2=plot_url2, 
                           plot_url3=plot_url3,
                           plot_url4=plot_url4,
                           plot_url5=plot_url5,
                           plot_url6=plot_url6)

@app.route('/commercial', methods=['GET', 'POST'])
def commercial():
    plot_url = plot_url2 = plot_url3 = plot_url4 = plot_url5 = plot_url6 = stats = stats2 = stats3= None
    if request.method == 'POST':
        # Initialisation des dataframes
        df1 = df2 = df3 = pd.DataFrame()
        numeric_df = pd.DataFrame()    

        file_type = request.form['fileType']
        file = request.files.get('file')      
        if file:
            if file_type == 'filePeriode':          
                df1 = prepare_data(file)
                numeric_df=numeric_data(df1)
                stats = describe_data(numeric_df)
                plot_url = heatmap(numeric_df)
                plot_url2 = plot_profil_puissance(df1)
                plot_url3 = heure_pointe(df1)  
                plot_url6=box_plot(df1)            
            elif file_type== 'fileAppel':
                df2= prepare_data2(file)
                numeric_df2=numeric_data(df2)
                plot_url4= heatmap_df2(numeric_df2)
                plot_url5=appel_puissance(df2)               
    # Générer ou récupérer le HTML du graphique
    return render_template('commercial.html', 
                           stats=stats,
                           stats2=stats2,
                           stats3=stats3, 
                           plot_url=plot_url, 
                           plot_url2=plot_url2, 
                           plot_url3=plot_url3,
                           plot_url4=plot_url4,
                           plot_url5=plot_url5,
                           plot_url6=plot_url6)

if __name__ == '__main__':
    app.run(debug=True)
