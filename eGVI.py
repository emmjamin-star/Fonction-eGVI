import streamlit as st
import ezc3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import tempfile

st.set_page_config(page_title="Score eGVI", layout="centered")
st.title("🦿 Score eGVI - Interface interactive")

# 1. Upload des fichiers .c3d
st.header("1. Importer un ou plusieurs fichiers .c3d dont au moins un fichier d'essai statique et un d'essai dynamique")
uploaded_files = st.file_uploader("Choisissez un ou plusieurs fichiers .c3d", type="c3d", accept_multiple_files=True)

if uploaded_files:
    selected_file_statique = st.selectbox("Choisissez un fichier statique pour l'analyse", uploaded_files, format_func=lambda x: x.name)
    selected_file_dynamique = st.selectbox("Choisissez un fichier dynamique pour l'analyse", uploaded_files, format_func=lambda x: x.name)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_statique.read())
        tmp_path = tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".c3d") as tmp:
        tmp.write(selected_file_dynamique.read())
        tmpd_path = tmp.name

    acq1 = ezc3d.c3d(tmpd_path)  # acquisition dynamique
    labels = acq1['parameters']['POINT']['LABELS']['value']
    freq = acq1['header']['points']['frame_rate']
    first_frame = acq1['header']['points']['first_frame']
    n_frames = acq1['data']['points'].shape[2]
    time_offset = first_frame / freq
    time = np.arange(n_frames) / freq + time_offset
  
    # Valeur population contrôle Arnaud Gouelle
    cn = [0.80, 0.93, 0.92, 0.90, 0.89, 0.73, 0.82, 0.85, 0.86, 0.90]
    cn = np.array(cn)
    m_ln_CTRL = 1.386573
    sd_ln_CTRL = 0.619334
    Projection_CTRL = 20.38
  
    statique = ezc3d.c3d(tmp_path)  # acquisition statique
    labelsStat = statique['parameters']['POINT']['LABELS']['value']
    first_frameStat = statique['header']['points']['first_frame']
    n_framesStat = statique['data']['points'].shape[2]
    time_offsetStat = first_frameStat / freq
    timeStat = np.arange(n_framesStat) / freq + time_offsetStat
    
    markersStat  = statique['data']['points']
    markers1 = acq1['data']['points']
    data1 = acq1['data']['points']

if st.button("Lancer le calcul du score eGVI"):
    try:
        # Extraction des coordonnées
        a1, a2, b1, b2, c1, c2 = markersStat[:,labels.index('LASI'),:][0, 0], markersStat[:,labels.index('LANK'),:][0, 0], markersStat[:,labels.index('LASI'),:][1, 0], markersStat[:,labels.index('LANK'),:][1, 0], markersStat[:,labels.index('LASI'),:][2, 0], markersStat[:,labels.index('LANK'),:][2, 0]
        LgJambeL = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
      
        a1, a2, b1, b2, c1, c2 = markersStat[:,labels.index('RASI'),:][0, 0], markersStat[:,labels.index('RANK'),:][0, 0], markersStat[:,labels.index('RASI'),:][1, 0], markersStat[:,labels.index('RANK'),:][1, 0], markersStat[:,labels.index('RASI'),:][2, 0], markersStat[:,labels.index('RANK'),:][2, 0]
        LgJambeR = np.sqrt((a2-a1)*(a2-a1)+(b2-b1)*(b2-b1)+(c2-c1)*(c2-c1))
      
        LargeurPelvis = np.abs(markersStat[:,labels.index('RASI'),:][1, 0] - markersStat[:,labels.index('LASI'),:][1, 0])
      
        # Détection event gauche
        # Détection des cycles à partir du marqueur LHEE (talon gauche)
        points = acq1['data']['points']
        if "LHEE" in labels:
            idx_lhee = labels.index("LHEE")
            z_lhee = points[2, idx_lhee, :]
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z = -z_lhee
            min_distance = int(freq * 0.8)
      
            # Détection pics
            peaks, _ = find_peaks(inverted_z, distance = min_distance, prominence = 1)
      
            # Début et fin des cycles = entre chaque pic
            lhee_cycle_start_indices = peaks[:-1]
            lhee_cycle_end_indices = peaks[1:]
            min_lhee_cycle_duration = int(0.5 * freq)
            lhee_valid_cycles = [(start, end) for start, end in zip(lhee_cycle_start_indices, lhee_cycle_end_indices) if (end - start) >= min_lhee_cycle_duration]
            lhee_n_cycles = len(lhee_valid_cycles)
      
        # Détection event droite
        # Détection des cycles à partir du marqueur RHEE (talon droite)
        points = acq1['data']['points']
        if "RHEE" in labels:
            idx_rhee = labels.index("RHEE")
            z_rhee = points[2, idx_rhee, :]
      
            # Inversion signal pour détecter les minima (contacts au sol)
            inverted_z = -z_rhee
            min_distance = int(freq * 0.8)
      
            # Détection pics
            peaks, _ = find_peaks(inverted_z, distance = min_distance, prominence = 1)
      
            # Début et fin des cycles = entre chaque pic
            rhee_cycle_start_indices = peaks[:-1]
            rhee_cycle_end_indices = peaks[1:]
            min_rhee_cycle_duration = int(0.5 * freq)
            rhee_valid_cycles = [(start, end) for start, end in zip(rhee_cycle_start_indices, rhee_cycle_end_indices) if (end - start) >= min_rhee_cycle_duration]
            rhee_n_cycles = len(rhee_valid_cycles)
        # COTE DROIT
        PA_D = []
        SA_D = []
        LPas_D = []
        Vitesse_D = []
        DPas_D = []
        correction = [1,1,2,2,3,3]
      
        for i,j in rhee_valid_cycles :
            PA_D.append((j-i)/100)
            n = int(i+((j-i)/2))
            LPas_D.append(np.abs((data1[:,labels.index('RTOE'),:][1, n] - data1[:,labels.index('LTOE'),:][1, n])/10))
            Vitesse_D.append(np.abs(((data1[:,labels.index('RTOE'),:][1, n] - data1[:,labels.index('RTOE'),:][1, i])/10)/(n-i))*100)
            for k,m in lhee_valid_cycles :
              DPas_D.append(np.abs(k - i)/100)
              SA_D.append(np.abs(k-m))
      
        for i in correction :
            DPas_D.pop(i)
            SA_D.pop(i)
      
      
        # Valeur moyenne 
        PA_D_m = np.mean(PA_D)
        LPas_D_m = np.mean(LPas_D)
        DPas_D_m = np.mean(DPas_D)
        Vitesse_D_m = np.mean(Vitesse_D)
        SA_D_m = np.mean(SA_D)
      
        # En pourcentage de la valeur moyenne 
        pPA_Dm = []
        pDPas_Dm = []
        pLPas_Dm = []
        pVitesse_Dm = []
        pSA_Dm = []
        for i in [0, 1, 2] : 
            pPA_Dm.append(PA_D[i] * 100 / PA_D_m) 
            pDPas_Dm.append(DPas_D[i] * 100 / DPas_D_m)
            pLPas_Dm.append(LPas_D[i] * 100 / LPas_D_m)
            pVitesse_Dm.append(Vitesse_D[i] * 100 / Vitesse_D_m)
            pSA_Dm.append(SA_D[i] * 100 / SA_D_m)
      
        # Différence absolue
        PA_D_f = []
        DPas_D_f = []
        LPas_D_f = []
        Vitesse_D_f = []
        SA_D_f = []
      
        for i in [0, 1]:
            PA_D_f.append(np.abs(pPA_Dm[i+1] - pPA_Dm[i]))
            DPas_D_f.append(np.abs(pDPas_Dm[i+1] - pDPas_Dm[i]))
            LPas_D_f.append(np.abs(pLPas_Dm[i+1] - pLPas_Dm[i]))
            Vitesse_D_f.append(np.abs(pVitesse_Dm[i+1] - pVitesse_Dm[i]))
            SA_D_f.append(np.abs(pSA_Dm[i+1] - pSA_Dm[i]))
      
      # COTE GAUCHE
        PA_G = []
        SA_G = []
        LPas_G = []
        Vitesse_G = []
        DPas_G = []
        correction = [1,1,2,2,3,3]
      
        for i,j in lhee_valid_cycles :
            PA_G.append((j-i)/100)
            n = int(i+((j-i)/2))
            LPas_G.append(np.abs((data1[:,labels.index('LTOE'),:][1, n] - data1[:,labels.index('RTOE'),:][1, n])/10))
            Vitesse_G.append(np.abs(((data1[:,labels.index('LTOE'),:][1, n] - data1[:,labels.index('LTOE'),:][1, i])/10)/(n-i))*100)
            for k,m in rhee_valid_cycles :
              DPas_G.append(np.abs(k - i)/100)
              SA_G.append(np.abs(k-m))
      
        for i in correction :
            DPas_G.pop(i)
            SA_G.pop(i)
      
      
        # Valeur moyenne 
        PA_G_m = np.mean(PA_G)
        LPas_G_m = np.mean(LPas_G)
        DPas_G_m = np.mean(DPas_G)
        Vitesse_G_m = np.mean(Vitesse_G)
        SA_G_m = np.mean(SA_G)
      
        # En pourcentage de la valeur moyenne 
        pPA_Gm = []
        pDPas_Gm = []
        pLPas_Gm = []
        pVitesse_Gm = []
        pSA_Gm = []
        for i in [0, 1, 2] : 
            pPA_Gm.append(PA_G[i] * 100 / PA_G_m) 
            pDPas_Gm.append(DPas_G[i] * 100 / DPas_G_m)
            pLPas_Gm.append(LPas_G[i] * 100 / LPas_G_m)
            pVitesse_Gm.append(Vitesse_G[i] * 100 / Vitesse_G_m)
            pSA_Gm.append(SA_G[i] * 100 / SA_G_m)
      
        # Différence absolue
        PA_G_f = []
        DPas_G_f = []
        LPas_G_f = []
        Vitesse_G_f = []
        SA_G_f = []
      
        for i in [0, 1]:
            PA_G_f.append(np.abs(pPA_Gm[i+1] - pPA_Gm[i]))
            DPas_G_f.append(np.abs(pDPas_Gm[i+1] - pDPas_Gm[i]))
            LPas_G_f.append(np.abs(pLPas_Gm[i+1] - pLPas_Gm[i]))
            Vitesse_G_f.append(np.abs(pVitesse_Gm[i+1] - pVitesse_Gm[i]))
            SA_G_f.append(np.abs(pSA_Gm[i+1] - pSA_Gm[i]))
      
        # Moyenne des différences absolues 
        # Droit : 
        mean_PA_D_f = np.mean(PA_D_f)
        mean_DPas_D_f = np.mean(DPas_D_f)
        mean_LPas_D_f = np.mean(LPas_D_f)
        mean_Vitesse_D_f = np.mean(Vitesse_D_f)
        mean_SA_D_f = np.mean(SA_D_f)
      
        # Gauche : 
        mean_PA_G_f = np.mean(PA_G_f)
        mean_DPas_G_f = np.mean(DPas_G_f)
        mean_LPas_G_f = np.mean(LPas_G_f)
        mean_Vitesse_G_f = np.mean(Vitesse_G_f)
        mean_SA_G_f = np.mean(SA_G_f)
      
        # Ecart-type des différences absolues 
        # Droit :
        std_PA_D_f = np.std(PA_D_f)
        std_DPas_D_f = np.std(DPas_D_f)
        std_LPas_D_f = np.std(LPas_D_f)
        std_Vitesse_D_f = np.std(Vitesse_D_f)
        std_SA_D_f = np.std(SA_D_f)
      
        # Gauche :
        std_PA_G_f = np.std(PA_G_f)
        std_DPas_G_f = np.std(DPas_G_f)
        std_LPas_G_f = np.std(LPas_G_f)
        std_Vitesse_G_f = np.std(Vitesse_G_f)
        std_SA_G_f = np.std(SA_G_f)
      
        # Création des vecteurs droit et gauche 
        Vect_D = [mean_DPas_D_f, mean_LPas_D_f, mean_Vitesse_D_f, mean_SA_D_f, mean_PA_D_f, std_DPas_D_f, std_LPas_D_f, std_PA_D_f, std_SA_D_f, std_Vitesse_D_f]
        Vect_G = [mean_DPas_G_f, mean_LPas_G_f, mean_Vitesse_G_f, mean_SA_G_f, mean_PA_G_f, std_DPas_G_f, std_LPas_G_f, std_PA_G_f, std_SA_G_f, std_Vitesse_G_f]
      
        # Réalisation des somprod et obtention de la projection du sujet 
        SP_D = np.sum(Vect_D * cn, axis=0)
        SP_G = np.sum(Vect_G * cn, axis=0)
      
        Diff_Sujet_CTRL_G = (SP_G-Projection_CTRL)
        Diff_Sujet_CTRL_G2 = abs(Diff_Sujet_CTRL_G)+1
        Diff_Sujet_CTRL_D = (SP_D-Projection_CTRL);
        Diff_Sujet_CTRL_D2 = abs(Diff_Sujet_CTRL_D)+1
      
        # Fin et calcul eGVI
        if Diff_Sujet_CTRL_G < 0 :
            ln_sujet_G = -np.log(Diff_Sujet_CTRL_G2)
        else :
            ln_sujet_G = np.log(Diff_Sujet_CTRL_G2)
      
        if Diff_Sujet_CTRL_D < 0 :
            ln_sujet_D = -np.log(Diff_Sujet_CTRL_D2)
        else :
            ln_sujet_D = np.log(Diff_Sujet_CTRL_D2)
      
        # z gauche
        z_G = []
      
        if ln_sujet_G < -m_ln_CTRL :
            z_G = (ln_sujet_G + m_ln_CTRL) / sd_ln_CTRL
      
        if ln_sujet_G > m_ln_CTRL :
            z_G = (ln_sujet_G - m_ln_CTRL) / sd_ln_CTRL
      
        if -m_ln_CTRL < ln_sujet_G and ln_sujet_G < m_ln_CTRL :
            z_G = 0
      
        # z droit
        z_D =  []
      
        if ln_sujet_D < -m_ln_CTRL :
            z_D = (ln_sujet_D + m_ln_CTRL) / sd_ln_CTRL
      
        if ln_sujet_D > m_ln_CTRL :
            z_D = (ln_sujet_D - m_ln_CTRL) / sd_ln_CTRL
      
        if -m_ln_CTRL < ln_sujet_D and  ln_sujet_D < m_ln_CTRL :
            z_D = 0
      
        eGVI_G = 100+z_G
        eGVI_D = 100+z_D
        eGVI = (eGVI_D + eGVI_G)/2
     
        st.markdown("### 📊 Résultats du score eGVI")
        st.write(f"**Score eGVI** : {eGVI:.2f}")
        st.write(f"**Lecture du test** : Un individu présentant une marche saine aura un score compris entre 98 et 102. Tout score en-dehors indique une atteinte à la variabilité de la marche.")
       
    except Exception as e:
        st.error(f"Erreur pendant l'analyse : {e}")
