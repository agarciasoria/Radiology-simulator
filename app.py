# app.py
# ============================================
# Física de Imagen Radiológica - Simulador Interactivo
# Para Técnico Superior en Imagen para el Diagnóstico
# ============================================

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import io

# ---------- Utilities for X-ray Physics ----------

def calculate_bremsstrahlung_spectrum(energies, kVp, mA, filtration_mm_al):
    """
    Calculate Bremsstrahlung (continuous) X-ray spectrum
    Kramer's formula approximation
    """
    spectrum = np.zeros_like(energies)
    
    # Only photons up to max energy (kVp)
    valid_energies = energies <= kVp
    E = energies[valid_energies]
    
    # Kramer's formula: I(E) ∝ Z(E_max - E)
    # Simplified model
    Z = 74  # Tungsten atomic number
    spectrum[valid_energies] = Z * mA * (kVp - E)
    
    # Apply filtration (exponential attenuation)
    # μ/ρ approximation for aluminum at diagnostic energies
    mu_al = 0.5 * (70 / E)**3 + 0.02  # Simplified attenuation coefficient
    filtration_cm = filtration_mm_al / 10
    attenuation = np.exp(-mu_al * 2.7 * filtration_cm)  # 2.7 g/cm³ Al density
    spectrum[valid_energies] *= attenuation
    
    return spectrum

def add_characteristic_peaks(energies, spectrum, kVp):
    """
    Add characteristic radiation peaks for Tungsten
    K-alpha: 59.3 keV, K-beta: 67.2 keV
    """
    if kVp < 69.5:  # K-edge of tungsten
        return spectrum
    
    # Add K-alpha peak at 59.3 keV
    k_alpha_idx = np.argmin(np.abs(energies - 59.3))
    spectrum[k_alpha_idx] += 0.15 * spectrum.max()
    
    # Add K-beta peak at 67.2 keV
    k_beta_idx = np.argmin(np.abs(energies - 67.2))
    spectrum[k_beta_idx] += 0.08 * spectrum.max()
    
    return spectrum

def calculate_xray_spectrum(kVp, mA, filtration_mm_al):
    """
    Complete X-ray spectrum calculation
    """
    energies = np.linspace(1, 150, 500)
    
    # Bremsstrahlung component
    spectrum = calculate_bremsstrahlung_spectrum(energies, kVp, mA, filtration_mm_al)
    
    # Add characteristic peaks
    spectrum = add_characteristic_peaks(energies, spectrum, kVp)
    
    return energies, spectrum

def calculate_hvl(kVp):
    """
    Calculate Half-Value Layer (HVL) in mm Al
    Empirical formula for diagnostic X-rays
    """
    # Simplified empirical relationship
    hvl = 0.001 * kVp**2 + 0.01 * kVp + 1.5
    return hvl

def calculate_entrance_dose(kVp, mAs, SID_cm=100):
    """
    Estimate entrance skin dose (ESD) in mGy
    Simplified model
    """
    # Empirical formula: dose proportional to kVp² and mAs, inverse square with distance
    reference_dose = 0.05  # mGy at 100cm, 80kVp, 10mAs
    dose = reference_dose * (kVp/80)**2 * (mAs/10) * (100/SID_cm)**2
    return dose

def calculate_effective_energy(kVp, filtration):
    """
    Calculate effective (mean) energy of spectrum
    """
    # Empirical: effective energy is roughly 1/3 to 1/2 of kVp
    # Increases with filtration
    eff_energy = kVp * (0.33 + 0.05 * filtration)
    return eff_energy

def calculate_contrast_index(kVp):
    """
    Contrast potential decreases with kVp
    Higher kVp = more penetration = less differential absorption = less contrast
    """
    # Normalized contrast index (arbitrary units, 100 at 60kVp)
    contrast = 100 * np.exp(-0.01 * (kVp - 60))
    return max(10, min(100, contrast))

# ---------- Utilities for Image Formation (Tab 2) ----------

def photoelectric_cross_section(energy_keV, Z):
    """
    Photoelectric effect cross-section (simplified)
    Strongly dependent on Z^3 and E^-3
    """
    # Simplified model
    if energy_keV < 10:
        energy_keV = 10  # Avoid division by zero
    sigma = (Z**3.8) / (energy_keV**3.2)
    return sigma

def compton_cross_section(energy_keV):
    """
    Compton scattering cross-section (Klein-Nishina formula, simplified)
    Nearly independent of Z, decreases with energy
    """
    # Simplified Klein-Nishina
    E = energy_keV / 511  # Normalize to electron rest mass
    sigma = 1 / (1 + 2*E)
    return sigma

def coherent_scattering_cross_section(energy_keV, Z):
    """
    Coherent (Rayleigh) scattering - minor contribution
    """
    sigma = (Z**2) / (energy_keV**2) * 0.1  # Small contribution
    return sigma

def total_attenuation_coefficient(energy_keV, Z, density=1.0):
    """
    Total linear attenuation coefficient μ
    Sum of all interactions
    """
    photo = photoelectric_cross_section(energy_keV, Z)
    compton = compton_cross_section(energy_keV)
    coherent = coherent_scattering_cross_section(energy_keV, Z)
    
    # Normalize and scale by density
    mu = (photo + compton + coherent) * density * 0.001
    return mu

def calculate_transmission(thickness_cm, mu):
    """
    Beer-Lambert law: I = I0 * exp(-μ * x)
    """
    transmission = np.exp(-mu * thickness_cm)
    return transmission

def calculate_contrast(I1, I2):
    """
    Contrast between two regions
    """
    if I1 + I2 == 0:
        return 0
    contrast = abs(I1 - I2) / (I1 + I2)
    return contrast

# Tissue properties database
TISSUES = {
    "Aire": {"Z_eff": 7.6, "density": 0.001, "color": "lightblue"},
    "Pulmón": {"Z_eff": 7.5, "density": 0.3, "color": "lightyellow"},
    "Tejido blando": {"Z_eff": 7.4, "density": 1.0, "color": "pink"},
    "Músculo": {"Z_eff": 7.5, "density": 1.05, "color": "salmon"},
    "Grasa": {"Z_eff": 6.3, "density": 0.95, "color": "yellow"},
    "Hueso cortical": {"Z_eff": 13.8, "density": 1.92, "color": "white"},
    "Hueso trabecular": {"Z_eff": 12.3, "density": 1.18, "color": "lightgray"},
    "Contraste yodado": {"Z_eff": 50, "density": 1.5, "color": "purple"},
    "Metal (implante)": {"Z_eff": 26, "density": 7.8, "color": "silver"},
    "Agua": {"Z_eff": 7.4, "density": 1.0, "color": "lightblue"}
}

def create_phantom_layer(tissue_type, thickness_cm):
    """
    Create a phantom layer with tissue properties
    """
    tissue = TISSUES[tissue_type]
    return {
        "type": tissue_type,
        "thickness": thickness_cm,
        "Z_eff": tissue["Z_eff"],
        "density": tissue["density"],
        "color": tissue["color"]
    }

def simulate_photon_path(layers, energy_keV, num_photons=1000):
    """
    Monte Carlo simulation of photon paths through layers
    Returns: transmitted, absorbed, scattered counts
    """
    transmitted = 0
    absorbed = 0
    scattered = 0
    
    for _ in range(num_photons):
        current_energy = energy_keV
        photon_alive = True
        
        for layer in layers:
            if not photon_alive:
                break
            
            mu = total_attenuation_coefficient(current_energy, layer["Z_eff"], layer["density"])
            
            # Probability of interaction in this layer
            prob_interaction = 1 - np.exp(-mu * layer["thickness"])
            
            if np.random.random() < prob_interaction:
                # Interaction occurred - determine type
                photo_prob = photoelectric_cross_section(current_energy, layer["Z_eff"])
                compton_prob = compton_cross_section(current_energy)
                total_prob = photo_prob + compton_prob
                
                if np.random.random() < (photo_prob / total_prob):
                    # Photoelectric - photon absorbed
                    absorbed += 1
                    photon_alive = False
                else:
                    # Compton - photon scattered
                    scattered += 1
                    # Reduce energy (simplified)
                    current_energy *= 0.7
                    if current_energy < 20:
                        photon_alive = False
        
        if photon_alive:
            transmitted += 1
    
    return transmitted, absorbed, scattered

def calculate_grid_transmission(grid_ratio, grid_frequency, is_scatter=False):
    """
    Calculate transmission through anti-scatter grid
    """
    if is_scatter:
        # Scatter is blocked more effectively
        transmission = 1 / (1 + grid_ratio * 0.8)
    else:
        # Primary beam is partially blocked by septa
        transmission = 1 / (1 + 0.1 * grid_ratio / 10)
    
    return transmission

# ---------- Utilities for Radiation Protection (Tab 3) ----------

def calculate_dose_at_distance(dose_at_reference, distance_reference, distance_target):
    """
    Inverse square law for dose calculation
    """
    dose_target = dose_at_reference * (distance_reference / distance_target) ** 2
    return dose_target

def calculate_transmission_through_shielding(hvl_mm, thickness_mm):
    """
    Calculate transmission through shielding material using HVL
    """
    n_hvls = thickness_mm / hvl_mm
    transmission = 0.5 ** n_hvls
    return transmission

def get_hvl_for_material(material, kVp):
    """
    Get HVL for different shielding materials at different kVp
    Simplified empirical relationships
    """
    hvl_data = {
        "Plomo": {
            60: 0.15,
            80: 0.25,
            100: 0.35,
            120: 0.45,
            150: 0.60
        },
        "Hormigón": {
            60: 10,
            80: 12,
            100: 14,
            120: 16,
            150: 18
        },
        "Vidrio plomado": {
            60: 0.20,
            80: 0.30,
            100: 0.42,
            120: 0.55,
            150: 0.70
        },
        "Acero": {
            60: 0.50,
            80: 0.70,
            100: 1.00,
            120: 1.30,
            150: 1.60
        }
    }
    
    # Interpolate if exact kVp not in table
    if material in hvl_data:
        kVps = sorted(hvl_data[material].keys())
        hvls = [hvl_data[material][k] for k in kVps]
        hvl = np.interp(kVp, kVps, hvls)
        return hvl
    return 1.0  # Default

def calculate_shielding_thickness(initial_dose_mSv_week, target_dose_mSv_week, hvl_mm):
    """
    Calculate required shielding thickness to achieve target dose
    """
    if target_dose_mSv_week >= initial_dose_mSv_week:
        return 0
    
    attenuation_needed = target_dose_mSv_week / initial_dose_mSv_week
    n_hvls = -np.log2(attenuation_needed)
    thickness_mm = n_hvls * hvl_mm
    return thickness_mm

def calculate_workload(patients_per_day, mAs_per_patient, days_per_week):
    """
    Calculate weekly workload in mA·min
    """
    total_mAs = patients_per_day * mAs_per_patient * days_per_week
    workload_mAmin = total_mAs / 60  # Convert to mA·min
    return workload_mAmin

def calculate_tenth_value_layer(hvl_mm):
    """
    Tenth Value Layer (TVL) - thickness that reduces to 10%
    TVL ≈ 3.32 × HVL
    """
    tvl = 3.32 * hvl_mm
    return tvl

# Dose limit constants (Spain/EU)
DOSE_LIMITS = {
    "Trabajador_expuesto": {
        "anual": 20,  # mSv/year (averaged over 5 years)
        "5_años": 100,  # mSv in 5 years
        "cristalino": 20,  # mSv/year (new limit)
        "piel": 500,  # mSv/year
        "extremidades": 500  # mSv/year
    },
    "Embarazada": {
        "superficie_abdomen": 2,  # mSv durante embarazo
        "feto": 1  # mSv durante embarazo
    },
    "Publico": {
        "anual": 1,  # mSv/year
        "cristalino": 15,  # mSv/year
        "piel": 50  # mSv/year
    },
    "Aprendiz_16-18": {
        "anual": 6,  # mSv/year
        "cristalino": 20,
        "piel": 150,
        "extremidades": 150
    }
}

# Typical organ doses for common procedures (mGy)
ORGAN_DOSES = {
    "Tórax PA": {"entrada": 0.2, "efectiva": 0.02},
    "Tórax LAT": {"entrada": 0.4, "efectiva": 0.04},
    "Abdomen AP": {"entrada": 4.0, "efectiva": 0.7},
    "Pelvis AP": {"entrada": 3.0, "efectiva": 0.6},
    "Columna Lumbar AP": {"entrada": 4.0, "efectiva": 0.7},
    "Columna Lumbar LAT": {"entrada": 8.0, "efectiva": 1.3},
    "Cráneo AP": {"entrada": 2.0, "efectiva": 0.05},
    "Mamografía (2 proyecciones)": {"entrada": 3.0, "efectiva": 0.4},
    "TC Tórax": {"CTDI": 8.0, "efectiva": 7.0},
    "TC Abdomen": {"CTDI": 10.0, "efectiva": 10.0},
    "TC Cráneo": {"CTDI": 60.0, "efectiva": 2.0}
}

# ---------- Utilities for Technical Parameters ----------(tab 4)

def calculate_15_percent_rule(kVp_initial, mAs_initial, direction="increase"):
    """
    Regla del 15%: Aumentar kVp en 15% duplica la exposición
    Si aumentas kVp → reduces mAs a la mitad
    Si reduces kVp → duplicas mAs
    """
    if direction == "increase":
        kVp_new = kVp_initial * 1.15
        mAs_new = mAs_initial / 2
    else:
        kVp_new = kVp_initial / 1.15
        mAs_new = mAs_initial * 2
    return kVp_new, mAs_new

def inverse_square_law(intensity_initial, distance_initial, distance_new):
    """
    Ley del cuadrado inverso: I₁/I₂ = (d₂/d₁)²
    """
    intensity_new = intensity_initial * (distance_initial / distance_new) ** 2
    return intensity_new

def calculate_grid_conversion_factor(grid_ratio):
    """
    Factor de conversión de rejilla (Bucky factor)
    """
    grid_factors = {
        "Sin rejilla": 1,
        "5:1": 2,
        "6:1": 3,
        "8:1": 4,
        "10:1": 5,
        "12:1": 5,
        "16:1": 6
    }
    return grid_factors.get(grid_ratio, 1)

def body_habitus_factor(habitus):
    """
    Factores de corrección según morfología del paciente
    """
    factors = {
        "Pediátrico (< 5 años)": 0.25,
        "Niño (5-12 años)": 0.5,
        "Adolescente": 0.75,
        "Adulto asténico (delgado)": 0.8,
        "Adulto hiposténico": 0.9,
        "Adulto esténico (normal)": 1.0,
        "Adulto hiperesténico": 1.2,
        "Adulto obeso": 1.5,
        "Adulto obeso mórbido": 2.0
    }
    return factors.get(habitus, 1.0)

def calculate_snr_cnr(kVp, mAs, thickness_cm):
    """
    Estimar SNR (Signal-to-Noise Ratio) y CNR (Contrast-to-Noise Ratio)
    """
    # SNR aumenta con √(fotones) ∝ √(mAs)
    snr = np.sqrt(mAs) * 10 * np.exp(-0.03 * thickness_cm)
    
    # CNR = Contraste × SNR (contraste disminuye con kVp)
    contrast_factor = 100 / kVp
    cnr = snr * contrast_factor
    
    return snr, cnr

def get_technique_chart():
    """
    Tabla de técnicas radiográficas estándar
    """
    
    data = {
        "Región Anatómica": [
            "Cráneo AP/PA", "Cráneo Lateral", "Senos paranasales",
            "Columna cervical AP", "Columna cervical Lateral", 
            "Columna torácica AP", "Columna torácica Lateral",
            "Columna lumbar AP", "Columna lumbar Lateral",
            "Tórax PA", "Tórax Lateral", "Parrilla costal",
            "Abdomen AP", "Abdomen Lateral",
            "Pelvis AP", "Cadera AP", "Fémur",
            "Rodilla AP/Lateral", "Tibia-Peroné", "Tobillo", "Pie",
            "Hombro AP", "Húmero", "Codo", "Antebrazo", "Muñeca", "Mano"
        ],
        "kVp": [
            70, 70, 70,
            75, 75,
            75, 85,
            80, 90,
            120, 120, 70,
            75, 85,
            75, 75, 70,
            65, 60, 60, 55,
            70, 65, 60, 55, 55, 50
        ],
        "mAs": [
            32, 20, 20,
            15, 10,
            25, 40,
            40, 80,
            3, 12, 10,
            25, 50,
            32, 32, 10,
            8, 5, 4, 3,
            12, 6, 5, 4, 3, 2.5
        ],
        "DFI (cm)": [
            100, 100, 100,
            180, 180,
            100, 100,
            100, 100,
            180, 180, 100,
            100, 100,
            100, 100, 100,
            100, 100, 100, 100,
            100, 100, 100, 100, 100, 100
        ],
        "Rejilla": [
            "8:1", "8:1", "No",
            "No", "No",
            "8:1", "8:1",
            "10:1", "10:1",
            "12:1", "12:1", "8:1",
            "10:1", "10:1",
            "10:1", "10:1", "8:1",
            "No", "No", "No", "No",
            "8:1", "No", "No", "No", "No", "No"
        ],
        "Grosor (cm)": [
            15, 15, 20,
            11, 11,
            20, 25,
            23, 28,
            23, 30, 20,
            22, 28,
            20, 18, 12,
            10, 8, 8, 6,
            14, 9, 8, 6, 5, 4
        ]
    }
    
    return pd.DataFrame(data)

# ---------- Page setup ----------
st.set_page_config(
    page_title="Física de Imagen Radiológica", 
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# ---------- Header and Introduction ----------
st.title("⚡ Física de Imagen Radiológica")
st.markdown("### Simulador Interactivo para Técnicos en Imagen para el Diagnóstico")
st.markdown("---")

# Introduction
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    ### ¡Bienvenido al simulador de física radiológica!
    
    Esta aplicación está diseñada específicamente para estudiantes y profesionales del 
    **Técnico Superior en Imagen para el Diagnóstico y Medicina Nuclear**.
    
    Aquí podrás explorar de manera **interactiva y visual** los conceptos físicos fundamentales 
    que necesitas dominar en tu práctica profesional diaria:
    
    - ⚡ **Producción de Rayos X**: Cómo funciona el tubo y qué parámetros controlan el haz
    - 🎯 **Formación de Imagen**: Cómo los rayos X interactúan con los tejidos
    - 🛡️ **Protección Radiológica**: Principios ALARA y cálculos de dosis
    - 🔧 **Optimización Técnica**: Cómo elegir kVp, mAs y otros factores
    - 📊 **Calidad de Imagen**: Balance entre calidad diagnóstica y dosis al paciente
    
    **Todo con simulaciones en tiempo real** para que veas inmediatamente el efecto 
    de cada parámetro técnico.
    """)

with col2:
    st.info("""
    **📚 Basado en el Currículo Oficial**
    
    Este simulador cubre contenidos del módulo:
    
    **"Formación de la imagen radiográfica"**
    
    Incluyendo:
    - Fundamentos físicos de rayos X
    - Parámetros técnicos
    - Formación y calidad de imagen
    - Protección radiológica
    - Normativa y seguridad
    
    ✅ Perfecto para **estudio**  
    ✅ Perfecto para **repaso**  
    ✅ Perfecto para **práctica**
    """)

# Module context expander
with st.expander("📖 Sobre este Módulo Formativo", expanded=False):
    st.markdown("""
    ### Imagen para el Diagnóstico - Contexto Educativo
    
    #### 🎓 Ciclo Formativo de Grado Superior
    
    El Técnico Superior en Imagen para el Diagnóstico y Medicina Nuclear es un profesional 
    sanitario especializado en la obtención de imágenes médicas mediante diferentes técnicas:
    
    - **Radiología Convencional** (Rayos X)
    - **Tomografía Computarizada (TC/CT)**
    - **Resonancia Magnética (RM/MRI)**
    - **Medicina Nuclear**
    - **Ultrasonidos** (en algunas comunidades)
    
    #### 📚 Competencias Profesionales
    
    Este simulador te ayuda a desarrollar competencias clave:
    
    1. **Competencia técnica**: Dominar los parámetros de exposición
    2. **Competencia en protección radiológica**: Aplicar el principio ALARA
    3. **Competencia en calidad**: Optimizar la relación calidad-dosis
    4. **Competencia analítica**: Entender el "por qué" detrás de cada protocolo
    
    #### 🏥 Aplicación Práctica
    
    En tu trabajo diario, estos conocimientos te permitirán:
    
    - Seleccionar parámetros técnicos apropiados para cada exploración
    - Adaptar técnicas a pacientes especiales (pediátricos, obesos, politraumatizados)
    - Minimizar la dosis manteniendo calidad diagnóstica
    - Resolver problemas técnicos (artefactos, repeticiones)
    - Cumplir con la normativa de protección radiológica
    - Comunicarte efectivamente con radiólogos y médicos prescriptores
    
    #### ⚖️ Marco Legal
    
    Trabajarás bajo regulación estricta:
    
    - **Real Decreto 1085/2009**: Instalaciones de rayos X con fines diagnósticos
    - **Real Decreto 783/2001**: Protección sanitaria contra radiaciones ionizantes
    - **Directiva 2013/59/EURATOM**: Normas de seguridad europeas
    - **Guías de protocolos clínicos** de cada comunidad autónoma
    
    🎯 **Este simulador te prepara para aplicar estos conocimientos de forma segura y efectiva.**
    """)

st.markdown("---")

# ---------- Sidebar ----------
st.sidebar.title("⚡ Física Radiológica")

# Navigation helper
st.sidebar.markdown("### 🧭 Guía de Navegación")
st.sidebar.markdown("""
- **¿Nuevo en radiología?** Empieza por el Tubo de Rayos X
- **¿Preparando examen?** Revisa cada sección en orden
- **¿Quieres practicar?** Prueba los casos clínicos
""")

# About section
with st.sidebar.expander("👤 Acerca de", expanded=True):
    st.markdown("""
    **Simulador Educativo**
    
    Desarrollado para apoyar la formación de 
    técnicos en imagen diagnóstica.
    
    Basado en:
    - 📖 Física aplicada
    - 🏥 Protocolos clínicos
    - 🛡️ Normativa de protección radiológica
    - 👨‍⚕️ Experiencia profesional
    
    **Versión**: 1.0  
    **Última actualización**: 2024
    """)

# Quick tips
with st.sidebar.expander("💡 Consejos de Uso"):
    st.markdown("""
    - **Interactúa**: Mueve los controles y observa cambios en tiempo real
    - **Zoom**: Usa la rueda del ratón en los gráficos
    - **Información**: Despliega las secciones "📚 Aprende más"
    - **Exporta**: Descarga datos para tus informes
    - **Experimenta**: No tengas miedo de probar valores extremos
    """)

# Safety reminder
st.sidebar.warning("""
⚠️ **Recordatorio de Seguridad**

Este es un simulador educativo. 
En la práctica real:
- Siempre sigue protocolos establecidos
- Verifica parámetros antes de exponer
- Usa protecciones para paciente y personal
- Cuando dudes, consulta
""")

# ---------- Main content tabs ----------
tabs = st.tabs([
    "⚡ Tubo de Rayos X",
    "🎯 Formación de Imagen",
    "🛡️ Protección Radiológica",
    "🔧 Parámetros Técnicos",
    "📊 Calidad de Imagen",
    "🏥 Casos Clínicos"
])

# ============================================
# TAB 1: TUBO DE RAYOS X (X-RAY TUBE PHYSICS)
# ============================================
with tabs[0]:
    st.header("⚡ El Tubo de Rayos X: Producción de Radiación")
    st.markdown("""
    Comprende cómo se generan los rayos X y cómo los parámetros del tubo afectan 
    al haz de radiación, la dosis al paciente y la calidad de la imagen.
    """)
    
    st.markdown("---")
    
    # Main controls
    st.subheader("🎛️ Parámetros del Tubo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### ⚡ Tensión del Tubo")
        kVp = st.slider(
            "kVp (kilovoltaje pico)", 
            min_value=40, 
            max_value=150, 
            value=80, 
            step=1,
            help="Controla la ENERGÍA de los fotones (penetración y contraste)"
        )
        st.caption(f"✓ Rango típico: 50-90 kVp (Rx simple), 100-140 kVp (TC)")
        
    with col2:
        st.markdown("##### 🔌 Corriente del Tubo")
        mA = st.slider(
            "mA (miliamperios)", 
            min_value=50, 
            max_value=500, 
            value=200, 
            step=10,
            help="Controla la CANTIDAD de fotones (densidad óptica/brillo)"
        )
        time_ms = st.slider(
            "Tiempo (ms)",
            min_value=1,
            max_value=1000,
            value=100,
            step=1,
            help="Tiempo de exposición en milisegundos"
        )
        mAs = mA * (time_ms / 1000)
        st.metric("mAs total", f"{mAs:.1f}", help="mAs = mA × tiempo(s)")
        st.caption(f"✓ Rango típico: 1-100 mAs (Rx), 100-500 mAs (TC)")
        
    with col3:
        st.markdown("##### 🔰 Filtración")
        filtration = st.slider(
            "Filtración adicional (mm Al)", 
            min_value=0.5, 
            max_value=5.0, 
            value=2.5, 
            step=0.5,
            help="Elimina fotones de baja energía (poco penetrantes pero que aumentan dosis)"
        )
        st.caption("✓ Filtración inherente: ~1mm Al")
        st.caption(f"✓ Filtración total: ~{filtration + 1:.1f} mm Al")
        
        # Anode material selector
        anode_material = st.selectbox(
            "Material del ánodo",
            ["Tungsteno (W)", "Molibdeno (Mo)", "Rodio (Rh)"],
            help="Tungsteno es el estándar. Mo y Rh se usan en mamografía"
        )

    st.markdown("---")
    
    # Calculate spectrum and parameters
    energies, spectrum = calculate_xray_spectrum(kVp, mA, filtration)
    hvl = calculate_hvl(kVp)
    entrance_dose = calculate_entrance_dose(kVp, mAs)
    eff_energy = calculate_effective_energy(kVp, filtration)
    contrast_index = calculate_contrast_index(kVp)
    
    # Display key metrics
    st.subheader("📊 Características del Haz")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric(
            "Energía Efectiva", 
            f"{eff_energy:.1f} keV",
            help="Energía promedio del espectro"
        )
        
    with metric_col2:
        st.metric(
            "HVL (Capa Hemirreductora)", 
            f"{hvl:.2f} mm Al",
            help="Espesor de aluminio que reduce la intensidad a la mitad"
        )
        
    with metric_col3:
        st.metric(
            "Dosis de Entrada Estimada", 
            f"{entrance_dose:.2f} mGy",
            delta=f"{entrance_dose - 2.0:.2f} mGy" if entrance_dose > 2.0 else None,
            delta_color="inverse",
            help="Dosis en la superficie de entrada del paciente (a 100cm)"
        )
        
    with metric_col4:
        contrast_color = "🟢" if contrast_index > 70 else "🟡" if contrast_index > 40 else "🔴"
        st.metric(
            "Índice de Contraste", 
            f"{contrast_color} {contrast_index:.0f}%",
            help="Potencial de contraste (mayor = más contraste tejido blando)"
        )
    
    st.markdown("---")
    
    # Spectrum visualization
    st.subheader("📈 Espectro de Rayos X")
    
    # Create spectrum plot
    fig_spectrum = go.Figure()
    
    # Add spectrum
    fig_spectrum.add_trace(go.Scatter(
        x=energies,
        y=spectrum,
        mode='lines',
        name='Espectro',
        fill='tozeroy',
        line=dict(color='royalblue', width=2),
        fillcolor='rgba(65, 105, 225, 0.3)'
    ))
    
    # Add markers for important energies
    fig_spectrum.add_vline(
        x=kVp, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Energía máxima ({kVp} keV)",
        annotation_position="top"
    )
    
    fig_spectrum.add_vline(
        x=eff_energy, 
        line_dash="dot", 
        line_color="green",
        annotation_text=f"Energía efectiva ({eff_energy:.1f} keV)",
        annotation_position="bottom"
    )
    
    # Mark characteristic peaks if present
    if kVp >= 69.5 and anode_material == "Tungsteno (W)":
        fig_spectrum.add_annotation(
            x=59.3, y=spectrum.max() * 0.9,
            text="K-α (W)<br>59.3 keV",
            showarrow=True,
            arrowhead=2,
            arrowcolor="orange",
            font=dict(color="orange", size=10)
        )
        fig_spectrum.add_annotation(
            x=67.2, y=spectrum.max() * 0.7,
            text="K-β (W)<br>67.2 keV",
            showarrow=True,
            arrowhead=2,
            arrowcolor="darkorange",
            font=dict(color="darkorange", size=10)
        )
    
    fig_spectrum.update_layout(
        title="Distribución de Energías del Haz de Rayos X",
        xaxis_title="Energía (keV)",
        yaxis_title="Intensidad Relativa (unidades arbitrarias)",
        hovermode='x unified',
        height=500,
        showlegend=True
    )
    
    fig_spectrum.update_xaxes(range=[0, min(150, kVp + 20)])
    
    st.plotly_chart(fig_spectrum, use_container_width=True)
    
    # Explanation of what we're seeing
    col_exp1, col_exp2 = st.columns(2)
    
    with col_exp1:
        st.info("""
        **📊 Interpretando el espectro:**
        
        - **Espectro continuo**: Radiación de frenado (Bremsstrahlung)
        - **Picos pronunciados**: Radiación característica del ánodo
        - **Energía máxima**: Limitada por el kVp aplicado
        - **Energía efectiva**: ~40-50% del kVp (con filtración)
        """)
        
    with col_exp2:
        st.warning("""
        **⚠️ Efecto de los parámetros:**
        
        - **↑ kVp**: Desplaza el espectro hacia energías más altas → más penetración
        - **↑ mA**: Aumenta la intensidad (altura) pero no cambia la forma
        - **↑ Filtración**: Elimina energías bajas, "endurece" el haz
        - **Material ánodo**: Cambia la posición de los picos característicos
        """)
    
    st.markdown("---")
    
    # Interactive comparison
    st.subheader("🔄 Comparación de Espectros")
    
    compare_col1, compare_col2 = st.columns(2)
    
    with compare_col1:
        compare_mode = st.checkbox("Activar comparación de espectros", value=False)
        
    if compare_mode:
        with compare_col2:
            comparison_param = st.radio(
                "Comparar efecto de:",
                ["kVp", "Filtración", "mA"],
                horizontal=True
            )
        
        # Create comparison plot
        fig_compare = go.Figure()
        
        if comparison_param == "kVp":
            kVp_values = [60, 80, 100, 120]
            colors_comp = ['blue', 'green', 'orange', 'red']
            for kv, color in zip(kVp_values, colors_comp):
                e, s = calculate_xray_spectrum(kv, mA, filtration)
                fig_compare.add_trace(go.Scatter(
                    x=e, y=s,
                    mode='lines',
                    name=f'{kv} kVp',
                    line=dict(color=color, width=2)
                ))
            fig_compare.update_layout(title="Efecto del kVp en el Espectro")
            
        elif comparison_param == "Filtración":
            filt_values = [0.5, 1.5, 3.0, 5.0]
            colors_comp = ['red', 'orange', 'green', 'blue']
            for filt, color in zip(filt_values, colors_comp):
                e, s = calculate_xray_spectrum(kVp, mA, filt)
                fig_compare.add_trace(go.Scatter(
                    x=e, y=s,
                    mode='lines',
                    name=f'{filt} mm Al',
                    line=dict(color=color, width=2)
                ))
            fig_compare.update_layout(title="Efecto de la Filtración en el Espectro")
            
        else:  # mA
            mA_values = [100, 200, 300, 400]
            colors_comp = ['lightblue', 'blue', 'darkblue', 'navy']
            for ma, color in zip(mA_values, colors_comp):
                e, s = calculate_xray_spectrum(kVp, ma, filtration)
                fig_compare.add_trace(go.Scatter(
                    x=e, y=s,
                    mode='lines',
                    name=f'{ma} mA',
                    line=dict(color=color, width=2)
                ))
            fig_compare.update_layout(title="Efecto del mA en el Espectro (solo intensidad)")
        
        fig_compare.update_layout(
            xaxis_title="Energía (keV)",
            yaxis_title="Intensidad Relativa",
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_compare, use_container_width=True)
    
    st.markdown("---")
    
    # Tube diagram and animation
    st.subheader("🔬 Funcionamiento del Tubo de Rayos X")
    
    show_animation = st.checkbox("Mostrar animación del tubo", value=True)
    
    if show_animation:
        # Create simplified tube diagram
        fig_tube = go.Figure()
        
        # Cathode (filament)
        fig_tube.add_trace(go.Scatter(
            x=[1, 1.5],
            y=[2, 2],
            mode='lines+markers',
            line=dict(color='red', width=4),
            marker=dict(size=10, color='red'),
            name='Cátodo (-)',
            showlegend=True
        ))
        
        # Anode (target)
        fig_tube.add_trace(go.Scatter(
            x=[8, 8.5, 8.5, 8],
            y=[3, 3.5, 1.5, 1],
            mode='lines',
            fill='toself',
            fillcolor='gray',
            line=dict(color='darkgray', width=2),
            name='Ánodo (+)',
            showlegend=True
        ))
        
        # Electron beam (animated with arrows)
        num_electrons = 5
        for i in range(num_electrons):
            x_start = 1.5
            x_end = 8
            y_pos = 2 + (i - num_electrons//2) * 0.15
            
            fig_tube.add_annotation(
                x=x_end,
                y=y_pos,
                ax=x_start,
                ay=y_pos,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='yellow',
            )
        
        # X-ray beam (emanating from anode)
        x_ray_angles = [30, 35, 40, 45, 50]
        for angle in x_ray_angles:
            angle_rad = np.radians(angle)
            x_end = 8 + 2 * np.cos(angle_rad)
            y_end = 2 + 2 * np.sin(angle_rad)
            
            fig_tube.add_annotation(
                x=x_end,
                y=y_end,
                ax=8,
                ay=2,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=3,
                arrowcolor='cyan',
            )
        
        # Glass envelope
        theta = np.linspace(0, 2*np.pi, 100)
        x_envelope = 4.75 + 4 * np.cos(theta)
        y_envelope = 2 + 1.5 * np.sin(theta)
        fig_tube.add_trace(go.Scatter(
            x=x_envelope,
            y=y_envelope,
            mode='lines',
            line=dict(color='lightblue', width=2, dash='dash'),
            name='Ampolla de vidrio',
            showlegend=True
        ))
        
        # Labels
        fig_tube.add_annotation(x=1.25, y=2.5, text="<b>Filamento</b><br>(emisión termoiónica)", showarrow=False, font=dict(size=10))
        fig_tube.add_annotation(x=8.25, y=3.8, text=f"<b>Ánodo ({anode_material.split()[0]})</b>", showarrow=False, font=dict(size=10))
        fig_tube.add_annotation(x=4.5, y=2, text="Electrones acelerados<br>↑ velocidad ∝ kVp", showarrow=False, font=dict(size=9, color='yellow'))
        fig_tube.add_annotation(x=9, y=3, text="<b>Rayos X</b><br>(1% energía)", showarrow=False, font=dict(size=10, color='cyan'))
        fig_tube.add_annotation(x=4.75, y=4, text=f"<b>Vacío</b> (~10⁻⁶ torr)", showarrow=False, font=dict(size=9))
        
        # Voltage indicator
        fig_tube.add_annotation(x=4.75, y=0.2, text=f"<b>{kVp} kV</b>", showarrow=False, font=dict(size=14, color='red'))
        fig_tube.add_annotation(x=4.75, y=-0.2, text=f"<b>{mA} mA</b>", showarrow=False, font=dict(size=12, color='orange'))
        
        fig_tube.update_layout(
            title="Esquema Simplificado del Tubo de Rayos X",
            xaxis=dict(range=[0, 11], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-0.5, 4.5], showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x'),
            height=400,
            plot_bgcolor='black',
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        st.plotly_chart(fig_tube, use_container_width=True)
        
        # Process explanation
        process_col1, process_col2, process_col3 = st.columns(3)
        
        with process_col1:
            st.markdown("""
            **1️⃣ Emisión Termoiónica**
            - Filamento calentado (~2000°C)
            - Libera electrones
            - Cantidad ∝ corriente (mA)
            """)
            
        with process_col2:
            st.markdown("""
            **2️⃣ Aceleración**
            - Diferencia de potencial (kVp)
            - Electrones ganan energía cinética
            - Velocidad hasta 60% velocidad luz
            """)
            
        with process_col3:
            st.markdown("""
            **3️⃣ Producción RX**
            - Impacto en ánodo
            - 99% → Calor ♨️
            - 1% → Rayos X ⚡
            """)
    
    st.markdown("---")
    
    # Heat management
    st.subheader("🌡️ Gestión Térmica del Tubo")
    
    heat_col1, heat_col2 = st.columns([2, 1])
    
    with heat_col1:
        # Calculate heat units
        heat_units = kVp * mAs  # Simplified HU calculation
        max_heat_capacity = 300000  # Typical modern tube in HU
        heat_percentage = (heat_units / max_heat_capacity) * 100
        
        # Heat dissipation rate (HU/s)
        cooling_rate = 1000  # HU per second
        cooling_time = heat_units / cooling_rate
        
        st.markdown(f"""
        **Unidades de Calor (HU - Heat Units):**
        
        Para un tubo monofásico: HU = kVp × mAs
        
        - Exposición actual: **{heat_units:.0f} HU**
        - Capacidad máxima del ánodo: **{max_heat_capacity:,} HU**
        - Porcentaje de capacidad: **{heat_percentage:.2f}%**
        - Tiempo de enfriamiento estimado: **{cooling_time:.1f} segundos**
        """)
        
        # Heat capacity bar
        fig_heat = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=heat_units,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Carga Térmica del Ánodo (HU)", 'font': {'size': 16}},
            delta={'reference': max_heat_capacity * 0.8, 'increasing': {'color': "red"}},
            gauge={
                'axis': {'range': [None, max_heat_capacity], 'tickformat': ',.0f'},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, max_heat_capacity * 0.6], 'color': "lightgreen"},
                    {'range': [max_heat_capacity * 0.6, max_heat_capacity * 0.8], 'color': "yellow"},
                    {'range': [max_heat_capacity * 0.8, max_heat_capacity], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_heat_capacity * 0.9
                }
            }
        ))
        
        fig_heat.update_layout(height=300)
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with heat_col2:
        st.info("""
        **⚠️ Límites Térmicos**
        
        El calor excesivo puede:
        - Fundir el ánodo
        - Agrietar el disco
        - Dañar los rodamientos
        - Romper la ampolla
        
        **Prevención:**
        - ✅ Respetar tiempo entre exposiciones
        - ✅ Usar ánodo giratorio
        - ✅ Monitorizar HU
        - ✅ Ventilación adecuada
        
        **Regla práctica:**
        - Serie corta: OK
        - Serie larga: Esperar
        - Fluoroscopia: Pulsada
        """)
    
    # Warning if heat is too high
    if heat_percentage > 80:
        st.error(f"""
        🚨 **ADVERTENCIA: Carga térmica elevada ({heat_percentage:.1f}%)**
        
        Esta configuración puede sobrecalentar el tubo. Recomendaciones:
        - Reducir mAs si es posible
        - Aumentar tiempo entre exposiciones
        - Considerar usar técnica de alta tensión (↑kVp, ↓mAs)
        """)
    elif heat_percentage > 60:
        st.warning(f"""
        ⚠️ **Atención: Carga térmica moderada-alta ({heat_percentage:.1f}%)**
        
        Monitoriza la temperatura del tubo si realizas múltiples exposiciones.
        """)
    
    st.markdown("---")
    
    # Practical applications
    st.subheader("💼 Aplicación Práctica: Selección de Parámetros")
    
    clinical_scenario = st.selectbox(
        "Selecciona un escenario clínico:",
        [
            "Tórax PA (adulto estándar)",
            "Abdomen AP (adulto)",
            "Extremidad (mano/pie)",
            "Pediatría (tórax)",
            "Paciente obeso (IMC > 35)",
            "Radiografía portátil (UCI)"
        ]
    )
    
    # Provide recommended values for each scenario
    recommendations = {
        "Tórax PA (adulto estándar)": {
            "kVp": (110, 125),
            "mAs": (2, 5),
            "filtration": 2.5,
            "rationale": """
            **Protocolo Alta Tensión:**
            - Alto kVp → Buena penetración del mediastino
            - Bajo mAs → Menor dosis, menor movimiento
            - Filtración estándar
            - Distancia: 180 cm (mejor geometría pulmonar)
            """
        },
        "Abdomen AP (adulto)": {
            "kVp": (70, 80),
            "mAs": (20, 40),
            "rationale": """
            **Protocolo Contraste Medio:**
            - kVp medio → Buen contraste tejidos blandos
            - mAs alto → Compensar densidad abdominal
            - Usar rejilla obligatorio (alta dispersión)
            - Control respiratorio (espiración)
            """
        },
        "Extremidad (mano/pie)": {
            "kVp": (50, 60),
            "mAs": (2, 5),
            "filtration": 1.5,
            "rationale": """
            **Protocolo Detalle Óseo:**
            - Bajo kVp → Máximo contraste óseo
            - Bajo mAs → Parte delgada
            - Sin rejilla (poca dispersión)
            - Foco fino si disponible
            """
        },
        "Pediatría (tórax)": {
            "kVp": (65, 75),
            "mAs": (1, 3),
            "filtration": 2.5,
            "rationale": """
            **Protocolo Pediátrico:**
            - kVp moderado → Balance contraste/dosis
            - mAs muy bajo → ALARA (niños más radiosensibles)
            - Tiempo mínimo → Evitar movimiento
            - ⚠️ Protección gonadal obligatoria
            """
        },
        "Paciente obeso (IMC > 35)": {
            "kVp": (90, 110),
            "mAs": (30, 60),
            "rationale": """
            **Adaptación Obesidad:**
            - ↑ kVp → Mayor penetración
            - ↑ mAs → Compensar atenuación
            - Considerar técnica AEC (control automático)
            - Puede requerir dos exposiciones (ancho detector)
            """
        },
        "Radiografía portátil (UCI)": {
            "kVp": (70, 90),
            "mAs": (2, 6),
            "filtration": 2.5,
            "rationale": """
            **Limitaciones Equipo Portátil:**
            - kVp limitado por potencia equipo
            - mAs bajo (batería limitada)
            - Mayor distancia → Reduce dosis al personal
            - Calidad subóptima aceptable (urgencia)
            - ⚠️ Dispersión elevada: usar protección
            """
        }
    }
    
    rec = recommendations[clinical_scenario]
    
    rec_col1, rec_col2 = st.columns([1, 1])
    
    with rec_col1:
        st.success(f"""
        **📋 Parámetros Recomendados:**
        
        - **kVp**: {rec['kVp'][0]}-{rec['kVp'][1]} kVp
        - **mAs**: {rec['mAs'][0]}-{rec['mAs'][1]} mAs
        {f"- **Filtración**: {rec.get('filtration', 2.5)} mm Al" if 'filtration' in rec else ""}
        """)
        
        # Check if current settings match
        kVp_ok = rec['kVp'][0] <= kVp <= rec['kVp'][1]
        mAs_ok = rec['mAs'][0] <= mAs <= rec['mAs'][1]
        
        if kVp_ok and mAs_ok:
            st.success("✅ Tus parámetros actuales están dentro del rango recomendado")
        else:
            st.warning("⚠️ Tus parámetros actuales están fuera del rango típico para este escenario")
            if not kVp_ok:
                st.write(f"- kVp actual: {kVp} (recomendado: {rec['kVp'][0]}-{rec['kVp'][1]})")
            if not mAs_ok:
                st.write(f"- mAs actual: {mAs:.1f} (recomendado: {rec['mAs'][0]}-{rec['mAs'][1]})")
    
    with rec_col2:
        st.info(rec['rationale'])
    
    st.markdown("---")
    
    # Download section
    st.subheader("📥 Exportar Datos")
    
    enable_downloads = st.checkbox("Habilitar descargas", value=False)
    
    if enable_downloads:
        download_col1, download_col2 = st.columns(2)
        
        with download_col1:
            # Prepare CSV data
            csv_data = pd.DataFrame({
                'Energía (keV)': energies,
                'Intensidad Relativa': spectrum
            })
            
            csv_buffer = io.StringIO()
            csv_data.to_csv(csv_buffer, index=False)
            
            st.download_button(
                label="📊 Descargar Espectro (CSV)",
                data=csv_buffer.getvalue(),
                file_name=f"espectro_{kVp}kVp_{mAs}mAs.csv",
                mime="text/csv"
            )
        
        with download_col2:
            # Prepare parameters report
            report = f"""
INFORME DE PARÁMETROS DEL TUBO DE RAYOS X
==========================================

PARÁMETROS DE EXPOSICIÓN:
- Tensión (kVp): {kVp} kVp
- Corriente (mA): {mA} mA
- Tiempo: {time_ms} ms
- mAs total: {mAs:.2f} mAs
- Filtración: {filtration} mm Al
- Material ánodo: {anode_material}

CARACTERÍSTICAS DEL HAZ:
- Energía efectiva: {eff_energy:.2f} keV
- HVL: {hvl:.2f} mm Al
- Índice de contraste: {contrast_index:.1f}%
- Dosis de entrada (estimada): {entrance_dose:.2f} mGy

CARGA TÉRMICA:
- Unidades de calor: {heat_units:.0f} HU
- Porcentaje capacidad: {heat_percentage:.2f}%
- Tiempo enfriamiento: {cooling_time:.1f} s

ESCENARIO CLÍNICO:
{clinical_scenario}

Generado por: Simulador de Física Radiológica
Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            
            st.download_button(
                label="📄 Descargar Informe (TXT)",
                data=report,
                file_name=f"informe_tubo_{kVp}kVp.txt",
                mime="text/plain"
            )
    
    st.markdown("---")
    
    # Comprehensive theory section
    with st.expander("📚 Teoría Completa: Física del Tubo de Rayos X", expanded=False):
        st.markdown("""
        ## 🔬 Fundamentos Físicos de la Producción de Rayos X
        
        ### 1. Historia y Descubrimiento
        
        Los rayos X fueron descubiertos accidentalmente por **Wilhelm Conrad Röntgen** el 8 de noviembre 
        de 1895 en Würzburg, Alemania. Mientras experimentaba con tubos de rayos catódicos, observó una 
        fluorescencia en una pantalla de platinocianuro de bario situada cerca del tubo. Röntgen denominó 
        a esta radiación desconocida "Rayos X".
        
        **Primera radiografía de la historia**: La mano de su esposa Anna Bertha, 22 de diciembre de 1895.
        
        ---
        
        ### 2. Componentes del Tubo de Rayos X Moderno
        
        #### 🔴 Cátodo (Polo Negativo)
        
        **Función**: Emisión de electrones mediante **emisión termoiónica**
        
        **Componentes**:
        - **Filamento**: Alambre de tungsteno enrollado (0.1-0.2 mm diámetro)
        - **Copa focalizadora**: Concentra el haz de electrones hacia el ánodo
        - **Circuito de baja tensión**: 10-12 V, pero corriente alta (3-5 A)
        
        **Física de la emisión termoiónica**:
        
        Cuando el filamento se calienta (~2000°C), los electrones adquieren suficiente energía 
        para vencer la **función trabajo** del tungsteno (4.5 eV) y escapar del metal.
        
        La corriente de emisión sigue la **ecuación de Richardson-Dushman**:
        """)
        
        st.latex(r"J = AT^2 e^{-\frac{W}{kT}}")
        
        st.markdown("""
        Donde:
        - J = densidad de corriente de emisión
        - A = constante de Richardson (material)
        - T = temperatura absoluta
        - W = función trabajo
        - k = constante de Boltzmann
        
        **¿Por qué tungsteno?**
        - Alto punto de fusión (3422°C)
        - Bajo trabajo de extracción
        - Baja presión de vapor
        - Alta resistencia mecánica
        
        ---
        
        #### 🔵 Ánodo (Polo Positivo)
        
        **Función**: Blanco donde impactan los electrones, produciendo rayos X
        
        **Tipos**:
        
        1. **Ánodo fijo**: Para equipos de baja potencia (dental, portátil)
        2. **Ánodo giratorio**: Estándar en radiología moderna (3,400-10,000 rpm)
        
        **Material del ánodo**:
        - **Tungsteno (W)**: Estándar en radiodiagnóstico general
          - Z = 74 (alto número atómico → eficiente producción RX)
          - Punto de fusión: 3422°C
          - Conductividad térmica excelente
        - **Aleación W-Renio**: Mayor durabilidad (5-10% Re)
        - **Molibdeno/Rodio**: Mamografía (energías más bajas)
        
        **Ángulo del ánodo**: Típicamente 7-20°
        - Menor ángulo → foco efectivo más pequeño → mejor resolución
        - Mayor ángulo → mayor área de cobertura → menor efecto anódico
        
        **Efecto anódico (heel effect)**:
        La intensidad del haz es menor en el lado del ánodo debido a la auto-absorción en el material.
        - Diferencia puede ser hasta 45% entre lado cátodo y ánodo
        - **Aplicación práctica**: Orientar lado cátodo hacia la parte más densa del paciente
        
        ---
        
        ### 3. Producción de Rayos X: Dos Mecanismos
        
        Cuando un electrón acelerado (energía cinética = e × kVp) impacta el ánodo, puede ocurrir:
        
        #### ⚡ A) Radiación de Frenado (Bremsstrahlung) - ~80% de los RX
        
        **Proceso físico**:
        - El electrón se desvía por el campo eléctrico del núcleo del átomo de tungsteno
        - Pierde energía cinética que se emite como fotón de RX
        - Puede ocurrir en múltiples interacciones (frenado gradual)
        
        **Características**:
        - **Espectro continuo** de energías (0 a kVp máximo)
        - Energía del fotón = Energía perdida por el electrón
        - La mayoría de electrones sufren múltiples frenados parciales
        """)
        
        st.latex(r"E_{fotón} \leq E_{electrón} = e \times kVp")
        
        st.markdown("""
        **Dependencia con Z y kVp**:
        """)
        
        st.latex(r"I_{Brems} \propto Z \times kVp^2")
        
        st.markdown("""
        Por eso se usa tungsteno (Z alto) y aumentar kVp produce muchos más fotones.
        
        #### 🎯 B) Radiación Característica - ~20% de los RX
        
        **Proceso físico**:
        1. Electrón incidente ioniza un átomo del ánodo, expulsando un electrón de capa interna (K, L, M)
        2. Un electrón de capa superior cae para llenar el hueco
        3. La diferencia de energía se emite como fotón de energía **exacta y característica** del material
        
        **Para Tungsteno**:
        - Transición L→K: **K-α = 59.3 keV**
        - Transición M→K: **K-β = 67.2 keV**
        
        **Energía umbral**: Solo ocurre si el electrón tiene energía suficiente para ionizar la capa K
        - Tungsteno: Energía de enlace K = 69.5 keV
        - Por tanto, se necesita **kVp ≥ 70** para ver picos característicos
        
        """)
        
        st.latex(r"E_{característica} = E_{capa\_superior} - E_{capa\_inferior}")
        
        st.markdown("""
        **Picos característicos en el espectro**:
        - Aparecen como líneas discretas superpuestas al continuo
        - Su intensidad aumenta rápidamente por encima del umbral
        - Son específicos del material del ánodo (huella dactilar)
        
        ---
        
        ### 4. Eficiencia de Producción de RX
        
        ⚠️ **Solo ~1% de la energía se convierte en rayos X**
        
        El otro **99% se convierte en CALOR** en el ánodo. Esto explica:
        - La necesidad de ánodos giratorios (distribuir calor)
        - Sistemas de refrigeración (aceite, radiadores)
        - Límites de carga térmica (HU)
        
        **Eficiencia aproximada**:
        """)
        
        st.latex(r"\eta \approx 10^{-9} \times Z \times kVp")
        
        st.markdown("""
        Para tungsteno (Z=74) a 100 kVp:
        η ≈ 0.74% (menos del 1%)
        
        **Implicaciones prácticas**:
        - El tubo se calienta muchísimo
        - Necesidad de tiempo de enfriamiento entre exposiciones
        - Limitación en fluoroscopia continua (usar modo pulsado)
        
        ---
        
        ### 5. Factores que Modifican el Espectro
        
        #### 📈 Aumento de kVp
        
        **Efectos**:
        1. ↑ Energía máxima de fotones (límite superior del espectro)
        2. ↑ Intensidad total (más fotones) - proporcional a kVp²
        3. ↑ Energía promedio del haz (desplazamiento hacia la derecha)
        4. ↑ Penetración (fotones más energéticos)
        5. ↓ Contraste (menos diferencia de absorción entre tejidos)
        
        **Regla del 15%**:
        Aumentar kVp en 15% tiene aproximadamente el mismo efecto en densidad 
        que duplicar el mAs (pero con MENOR contraste).
        
        #### 📊 Aumento de mA (o tiempo)
        
        **Efectos**:
        1. ↑ Número de electrones → ↑ número de fotones
        2. **NO cambia** la forma del espectro (solo escala vertical)
        3. **NO cambia** la penetración ni energías
        4. ↑ Proporcionalmente la dosis al paciente
        
        **mAs = mA × tiempo (segundos)**
        
        Es la carga total. 100mA × 0.1s = 10mAs = mismo efecto que 200mA × 0.05s
        
        #### 🔰 Aumento de Filtración
        
        **Efectos**:
        1. ↓ Fotones de baja energía (absorbidos por filtro)
        2. ↑ Energía promedio (espectro "endurecido")
        3. ↓ Dosis en piel (elimina fotones no útiles)
        4. Ligera ↓ en intensidad total
        
        **Concepto de endurecimiento del haz**:
        """)
        
        st.latex(r"I(E, x) = I_0(E) \times e^{-\mu(E) \cdot x}")
        
        st.markdown("""
        Los fotones de baja energía tienen mayor μ (coeficiente de atenuación), 
        por lo que son absorbidos preferentemente por el filtro.
        
        **Filtración en radiología**:
        - **Inherente**: Ampolla de vidrio, aceite, ventana (~1 mm Al equivalente)
        - **Adicional**: Láminas de Al añadidas (1-3 mm típico)
        - **Total**: Inherente + Adicional (mínimo 2.5 mm Al en España)
        
        **Filtros especiales**:
        - **Filtro en cuña**: Compensa diferencias de espesor (AP de tórax)
        - **Filtro de compensación**: Para partes anatómicas irregulares
        
        ---
        
        ### 6. Capa Hemirreductora (HVL - Half Value Layer)
        
        **Definición**: Espesor de material que reduce la intensidad del haz a la MITAD
        
        """)
        
        st.latex(r"I = I_0 \times 0.5^{x/HVL} = I_0 \times e^{-\mu \cdot x}")
        
        st.markdown("""
        **Relación con μ**:
        """)
        
        st.latex(r"HVL = \frac{ln(2)}{\mu} \approx \frac{0.693}{\mu}")
        
        st.markdown("""
        **Importancia de HVL**:
        - **Control de calidad**: Mide la "dureza" del haz
        - **Normativa**: HVL mínimo requerido por ley (protección paciente)
        - **Optimización**: Haces más duros = menos dosis superficial
        
        **Valores típicos en diagnóstico**:
        - 60 kVp: ~1.5 mm Al
        - 80 kVp: ~2.5 mm Al
        - 100 kVp: ~3.5 mm Al
        - 120 kVp: ~4.5 mm Al
        
        **Endurecimiento del haz**: 
        Conforme el haz penetra tejido, HVL aumenta (fotones blandos absorbidos primero).
        - HVL₁: Primera capa hemirreductora
        - HVL₂: Segunda capa hemirreductora (HVL₂ > HVL₁)
        - **Coeficiente de homogeneidad**: HVL₁/HVL₂ (ideal = 1, real ~0.7-0.9)
        
        ---
        
        ### 7. Calidad vs Cantidad de Radiación
        
        | Parámetro | Afecta Calidad (penetración) | Afecta Cantidad (nº fotones) |
        |-----------|------------------------------|------------------------------|
        | **kVp**   | ✅ SÍ (principal factor)     | ✅ SÍ (proporcional a kVp²)  |
        | **mAs**   | ❌ NO                        | ✅ SÍ (proporcional)         |
        | **Filtración** | ✅ SÍ (endurece)       | ⬇️ Reduce ligeramente        |
        | **Distancia** | ❌ NO                    | ⬇️ Reduce (ley inversa cuadrado) |
        
        **Estrategias de optimización**:
        
        1. **Técnica de alto kVp** (ej. tórax):
           - Alto kVp (110-120) + Bajo mAs (2-5)
           - Ventajas: Menos dosis, menos movimiento
           - Desventajas: Menos contraste
        
        2. **Técnica de bajo kVp** (ej. extremidades):
           - Bajo kVp (50-60) + Bajo mAs (2-5)
           - Ventajas: Máximo contraste óseo
           - Aplicable solo en partes delgadas
        
        3. **Técnica equilibrada** (ej. abdomen):
           - kVp medio (70-80) + mAs moderado (20-40)
           - Balance contraste/penetración
        
        ---
        
        ### 8. Gestión Térmica: Un Reto de Ingeniería
        
        #### 🌡️ Carga Térmica (Heat Units)
        
        **Cálculo simplificado**:
        - Monofásico: HU = kVp × mAs
        - Trifásico: HU = 1.35 × kVp × mAs
        - Alta frecuencia: HU = 1.4 × kVp × mAs
        
        **Capacidad típica de ánodos modernos**:
        - Tubo estándar: 200,000-400,000 HU
        - TC de alta gama: 5,000,000+ HU
        
        #### 🔄 Ánodo Giratorio: Solución Elegante
        
        **Principio**: Distribuir el calor sobre un área mayor
        
        **Ventajas sobre ánodo fijo**:
        - Área de contacto efectiva ~100 veces mayor
        - Permite cargas instantáneas mucho mayores
        - Enfriamiento más eficiente
        
        **Velocidad de rotación**:
        - Estándar: 3,400 rpm (~60 Hz)
        - Alta velocidad: 10,000 rpm (para TC y angiografía)
        - Activación mediante motor de inducción electromagnético
        
        **Pista focal**:
        - Diámetro típico: 50-100 mm
        - Ancho de pista: 3-10 mm
        - Material: Tungsteno-Renio sobre disco de molibdeno-grafito
        
        #### ❄️ Sistemas de Refrigeración
        
        1. **Aceite dieléctrico**: Rodea la ampolla, absorbe calor por convección
        2. **Carcasa metálica**: Disipa calor al ambiente por radiación
        3. **Ventilación forzada**: Algunos equipos tienen ventiladores
        4. **Refrigeración líquida**: TC de última generación
        
        **Curvas de enfriamiento**:
        - Enfriamiento exponencial: rápido al inicio, lento al final
        - Tiempo para enfriar completamente: 10-15 minutos típico
        - Nunca apagar el equipo inmediatamente tras uso intensivo
        
        #### ⚠️ Sobrecalentamiento: Consecuencias
        
        1. **Inmediatas**:
           - Interruptor térmico (protección del tubo)
           - Equipo bloqueado hasta enfriamiento
           - Pérdida de tiempo en sala
        
        2. **Daños a medio plazo**:
           - Microfisuras en el ánodo
           - Desgasificación de componentes
           - Pérdida de vacío
        
        3. **Daños graves (tubo inutilizable)**:
           - Fusión localizada del ánodo
           - Desequilibrio del disco (vibración)
           - Rotura de la ampolla
           - **Coste**: 15,000-80,000€ reemplazar tubo
        
        ---
        
        ### 9. Tamaño del Foco y Resolución
        
        #### 🎯 Concepto de Foco
        
        **Foco real**: Área del ánodo donde impactan los electrones
        
        **Foco efectivo**: Proyección del foco real perpendicular al haz central
        
        Gracias al **principio del foco de línea** (ángulo del ánodo):
        """)
        
        st.latex(r"\text{Foco efectivo} = \text{Foco real} \times \sin(\text{ángulo ánodo})")
        
        st.markdown("""
        **Ejemplo**: 
        - Foco real: 7 mm
        - Ángulo ánodo: 12°
        - Foco efectivo: 7 × sin(12°) ≈ 1.5 mm
        
        #### 📐 Doble Foco
        
        La mayoría de tubos modernos tienen **dos filamentos**:
        
        | Característica | Foco Fino | Foco Grueso |
        |---------------|-----------|-------------|
        | Tamaño efectivo | 0.3-0.6 mm | 1.0-1.5 mm |
        | Resolución espacial | ⭐⭐⭐ Alta | ⭐⭐ Moderada |
        | Capacidad carga | ⭐ Baja (1-10 mAs) | ⭐⭐⭐ Alta (>10 mAs) |
        | Aplicación | Extremidades, detalles | Tórax, abdomen, partes gruesas |
        
        **Regla práctica de selección**:
        - mAs < 10 → Foco fino (si disponible)
        - mAs ≥ 10 → Foco grueso (obligatorio)
        - Algunos equipos lo seleccionan automáticamente
        
        #### 🔍 Penumbra Geométrica
        
        La borrosidad en los bordes de las estructuras depende del tamaño del foco:
        """)
        
        st.latex(r"Penumbra = \frac{\text{Foco} \times \text{DOI}}{\text{DFI}}")
        
        st.markdown("""
        Donde:
        - DOI = Distancia Objeto-Imagen (Detector)
        - DFI = Distancia Foco-Imagen (SID/FFD)
        
        **Para minimizar penumbra**:
        - ✅ Usar foco pequeño
        - ✅ Aumentar DFI (distancia foco-detector)
        - ✅ Minimizar DOI (pegar parte al detector)
        
        ---
        
        ### 10. Control Automático de Exposición (AEC)
        
        #### 🤖 Sistema de Control Automático
        
        **Función**: Termina la exposición automáticamente cuando se alcanza la densidad correcta
        
        **Componentes**:
        - **Cámaras de ionización**: Detrás del detector, miden radiación transmitida
        - **Circuito de control**: Corta exposición al alcanzar umbral predeterminado
        - Típicamente 3 cámaras (seleccionables): derecha, centro, izquierda
        
        **Ventajas**:
        - ✅ Consistencia de calidad entre pacientes
        - ✅ Compensa variaciones de espesor/densidad
        - ✅ Reduce repeticiones por sobre/sub-exposición
        - ✅ Menos carga cognitiva para el técnico
        
        **Limitaciones**:
        - ❌ Requiere correcta selección de cámara
        - ❌ No funciona bien con prótesis metálicas sobre cámara
        - ❌ Puede fallar con pacientes muy delgados/obesos extremos
        - ❌ El técnico debe seguir eligiendo kVp correctamente
        
        **Errores comunes**:
        1. **Cámara mal posicionada**: No sobre la anatomía de interés
        2. **Colimación excesiva**: Cámara fuera del campo → tiempo infinito
        3. **Material denso sobre cámara**: Sobre-exposición del resto
        
        ---
        
        ### 11. Factores de Conversión y Reglas Prácticas
        
        #### 📏 Ley del Inverso del Cuadrado de la Distancia
        
        La intensidad del haz disminuye con el cuadrado de la distancia:
        """)
        
        st.latex(r"I_2 = I_1 \times \left(\frac{D_1}{D_2}\right)^2")
        
        st.markdown("""
        **Aplicación práctica**: Si cambias la distancia, debes compensar el mAs
        """)
        
        st.latex(r"\frac{mAs_2}{mAs_1} = \left(\frac{D_2}{D_1}\right)^2")
        
        st.markdown("""
        **Ejemplo**:
        - Técnica a 100 cm: 80 kVp, 10 mAs
        - Cambio a 150 cm: necesito 10 × (150/100)² = 10 × 2.25 = **22.5 mAs**
        
        #### 📊 Regla del 15% (kVp vs mAs)
        
        **Aumentar kVp en 15% ≈ Duplicar el mAs** (en términos de densidad/brillo)
        
        **Ejemplo**:
        - Original: 70 kVp, 20 mAs
        - Opción A: 70 × 1.15 = **80 kVp, 10 mAs** (misma densidad, ¡menos dosis!)
        - Pero: menor contraste
        
        **Uso práctico**:
        - Paciente obeso: ↑ kVp 15% en vez de duplicar mAs
        - Repetición por movimiento: ↑ kVp, ↓ tiempo
        
        #### 🔢 Otras Reglas Útiles
        
        **Regla del 5 kVp** (contraste):
        - Cambio perceptible en contraste: ±5 kVp
        - Menos de 5 kVp: cambio poco visible
        
        **Regla del 30% mAs** (densidad):
        - Cambio perceptible en densidad: ±30% mAs
        - Ajustes menores: imperceptibles
        
        **Reciprocidad mA-tiempo**:
        - 200 mA × 0.1 s = 100 mA × 0.2 s = 20 mAs
        - **Pero**: Tiempo corto mejor para evitar movimiento
        
        ---
        
        ### 12. Consideraciones de Protección Radiológica
        
        #### 🛡️ Optimización desde el Tubo
        
        **Principios para minimizar dosis**:
        
        1. **Filtración adecuada**:
           - Elimina fotones "blandos" (baja energía)
           - No contribuyen a imagen pero sí a dosis cutánea
           - Mínimo legal: 2.5 mm Al (≥70 kVp)
        
        2. **Colimación estricta**:
           - Reduce área irradiada → reduce dosis integral
           - Reduce dispersión → mejora contraste
           - **Nunca** irradiar más allá del detector
        
        3. **Técnica apropiada**:
           - No usar más mAs del necesario
           - Preferir alto kVp cuando contraste lo permita
           - Evitar repeticiones (técnica correcta a la primera)
        
        4. **Mantenimiento preventivo**:
           - Calibración periódica
           - Verificación de HVL
           - Control de calidad trimestral
        
        #### ⚡ Exposición Ocupacional del Técnico
        
        **Fuentes de exposición**:
        1. **Radiación de fuga**: < 1 mGy/h a 1 metro (normativa)
        2. **Radiación dispersa**: Del paciente (principal fuente)
        3. **Durante fluoroscopia**: Exposición continua
        
        **Protección del profesional**:
        - 🚪 **Distancia**: Sala de control con biombo plomado
        - ⏱️ **Tiempo**: Minimizar tiempo en sala durante exposición
        - 🦺 **Blindaje**: Delantal plomado (0.25-0.5 mm Pb eq) si necesario permanecer
        - 📊 **Dosimetría**: Obligatoria (cuerpo + cristalino para fluoroscopia)
        
        ---
        
        ### 13. Innovaciones Tecnológicas
        
        #### 💎 Ánodos de Diamante
        
        - En investigación/prototipos
        - Conductividad térmica 5× mayor que cobre
        - Permitiría cargas mucho mayores
        - Coste elevado
        
        #### 🔬 Tubos de Emisión de Campo
        
        - Nanotubos de carbono como emisores fríos
        - No necesitan calentamiento → respuesta instantánea
        - Múltiples fuentes controlables independientemente
        - Aplicación: TC de "fuentes estacionarias"
        
        #### 🌟 Tubos de Alta Potencia para TC
        
        - Capacidad calorífica: >5 MHU
        - Refrigeración líquida integrada
        - Velocidad >10,000 rpm
        - Permite TC cardíaco de alta resolución temporal
        
        ---
        
        ### 14. Resolución de Problemas Comunes
        
        #### ❓ Imagen Demasiado Clara (Subexposición)
        
        **Causas posibles**:
        - mAs insuficiente
        - kVp demasiado bajo (absorción excesiva)
        - Distancia aumentada sin compensar
        - AEC con cámara mal seleccionada
        - Fallo técnico (generador)
        
        **Solución**:
        - ↑ mAs (duplicar si muy clara)
        - O ↑ kVp 15% + mantener mAs
        - Verificar distancia y compensar
        
        #### ❓ Imagen Demasiado Oscura (Sobreexposición)
        
        **Causas**:
        - mAs excesivo
        - kVp muy alto
        - Distancia reducida sin compensar
        - AEC terminó demasiado tarde
        - Paciente más delgado de lo estimado
        
        **Solución**:
        - ↓ mAs a la mitad
        - O ↓ kVp 15% si contraste permite
        - **Importante**: En digital, sobreexposición = dosis innecesaria al paciente
        
        #### ❓ Falta de Contraste
        
        **Causas**:
        - kVp demasiado alto (penetración excesiva)
        - Mucha radiación dispersa (falta rejilla, colimación inadecuada)
        - Procesado digital incorrecto
        
        **Solución**:
        - ↓ kVp 10-15%
        - Compensar con ↑ mAs
        - Usar rejilla anti-dispersión
        - Colimar estrictamente
        
        #### ❓ Borrosidad de Movimiento
        
        **Causas**:
        - Tiempo de exposición largo
        - Paciente no colaborador (pediatría, confusión)
        - Movimiento involuntario (respiración, peristaltismo)
        
        **Solución**:
        - ↑ mA y ↓ tiempo (mantener mAs constante o ligeramente ↑)
        - ↑ kVp 15% para permitir menor mAs (tiempo más corto)
        - Inmovilización adecuada
        - Indicaciones claras al paciente
        
        #### ❓ Tubo No Dispara o Falla
        
        **Causas comunes**:
        1. **Sobrecalentamiento**: HU excedidas
           - Esperar enfriamiento (10-15 min)
           - Revisar curva de carga
        
        2. **Fallo de filamento**: Filamento roto
           - Cambiar a otro filamento (si dual)
           - Reemplazar tubo
        
        3. **Pérdida de vacío**: Entra aire a la ampolla
           - Tubo inservible, reemplazo necesario
           - Signo: chispas visibles, ruido
        
        4. **Error de parámetros**: Combinación no permitida
           - Verificar límites técnicos (tabla de técnicas)
           - ej: mAs demasiado alto para kVp bajo
        
        5. **Problema eléctrico**: Generador, cables
           - Servicio técnico
        
        ---
        
        ### 15. Mantenimiento y Control de Calidad
        
        #### 🔧 Mantenimiento Preventivo (Periódico)
        
        **Diario** (Técnico/TSID):
        - ✅ Inspección visual del equipo
        - ✅ Verificar movimientos del tubo
        - ✅ Limpieza de colimador
        - ✅ Test de exposición (phantom)
        
        **Mensual** (Técnico/TSID + Supervisor):
        - ✅ Test de reproducibilidad (kVp y mAs constantes)
        - ✅ Verificación de colimación (alineación luz/RX)
        - ✅ Test de AEC (si aplica)
        - ✅ Registro de HU acumulados
        
        **Trimestral** (Radiofísico):
        - ✅ Medición de kVp real
        - ✅ Medición de tiempo de exposición
        - ✅ Medición de HVL (calidad del haz)
        - ✅ Medición de dosis de entrada
        - ✅ Test de resolución espacial
        
        **Anual** (Radiofísico + Servicio Técnico):
        - ✅ Control de calidad completo
        - ✅ Verificación de seguridades
        - ✅ Calibración completa
        - ✅ Medición de radiación de fuga
        - ✅ Informe oficial para autoridad competente
        
        #### 📋 Valores de Referencia (Aceptabilidad)
        
        | Parámetro | Tolerancia |
        |-----------|------------|
        | **Exactitud kVp** | ±5% del valor nominal |
        | **Reproducibilidad kVp** | <2% variación |
        | **Exactitud tiempo** | ±5% o ±5ms |
        | **Linealidad mAs** | ±10% |
        | **HVL mínimo** | Según normativa (ej: 2.5mm Al a 80kVp) |
        | **Colimación luz-RX** | <2% DFI |
        | **Fuga de radiación** | <1 mGy/h a 1m |
        
        **Acciones correctivas**:
        - Fuera de tolerancia → Ajuste/calibración
        - Fuera de límites legales → **Paralización del equipo**
        - Documentar todas las intervenciones
        
        #### 📊 Registro de Tubo
        
        **Libro de vida del tubo** (obligatorio):
        - Fecha de instalación
        - HU acumulados totales
        - Exposiciones realizadas
        - Mantenimientos e incidencias
        - Fecha de reemplazo
        
        **Vida útil típica**:
        - Radiología general: 3-7 años
        - TC de alta carga: 1-3 años
        - Mamografía: 5-10 años (carga menor)
        
        ---
        
        ### 16. Casos Especiales y Adaptaciones
        
        #### 👶 Pediatría
        
        **Consideraciones especiales**:
        - Mayor radiosensibilidad (células en desarrollo)
        - Menor espesor → requiere menos radiación
        - Dificultad de inmovilización
        - Expectativa de vida larga (riesgo estocástico)
        
        **Adaptaciones técnicas**:
        - ↓↓ mAs máximo posible
        - Tiempo mínimo (movimiento)
        - kVp moderado (balance dosis/contraste)
        - Colimación estricta (órganos en desarrollo)
        - **Protección gonadal obligatoria** (< 30 años o si útil)
        - Considerar técnicas alternativas (US, MRI)
        
        **Ejemplo tórax pediátrico** (1-5 años):
        - 65-75 kVp (vs 110-120 adulto)
        - 1-3 mAs (vs 5-10 adulto)
        - Tiempo: <0.02s si posible
        
        #### 🤰 Embarazo
        
        **Principio fundamental**: Evitar RX si posible
        
        **Si absolutamente necesario**:
        1. **Verificar embarazo**: Siempre preguntar (10-50 años)
        2. **Justificación estricta**: Beneficio >> riesgo
        3. **Optimización extrema**:
           - Mínima área irradiada
           - Protección fetal (delantal plomado si fuera del campo)
           - Técnica de mínima dosis efectiva
        4. **Documentación**: Consentimiento informado
        5. **Estimación de dosis fetal**: Por radiofísico
        
        **Trimestre crítico**: 2-8 semanas (organogénesis)
        
        **Dosis de referencia**:
        - <100 mGy al útero: Riesgo insignificante
        - >100 mGy: Considerar alternativas/consejo
        
        #### 🦴 Pacientes con Prótesis Metálicas
        
        **Problema**: Metal atenúa fuertemente → sombras, artefactos
        
        **Estrategias**:
        - ↑ kVp (mayor penetración)
        - ↑ mAs compensatorio
        - Proyecciones alternativas (evitar superposición)
        - TC con algoritmos de reducción de artefactos metálicos
        
        #### 🏃 Portátiles/UCI
        
        **Limitaciones del equipo**:
        - Menor potencia (30-50 kVp máximo típico)
        - Batería limitada (mAs restringido)
        - Geometría subóptima (distancia corta)
        
        **Compensaciones**:
        - Usar máximo kVp disponible
        - mAs ajustado a límite del equipo
        - Aceptar menor calidad (contexto urgencia)
        - **Protección personal crítica** (dispersión elevada)
        - Alejar personal no necesario >2 metros
        
        ---
        
        ### 17. Normativa Legal (España)
        
        #### ⚖️ Marco Regulatorio Principal
        
        **Real Decreto 1085/2009**:
        - Instalaciones de rayos X con fines diagnósticos
        - Requisitos técnicos y de calidad
        - Control de calidad obligatorio
        
        **Real Decreto 783/2001**:
        - Protección sanitaria contra radiaciones ionizantes
        - Límites de dosis ocupacionales y público
        - Obligaciones del titular
        
        **Directiva 2013/59/EURATOM** (implementación española):
        - Principio de justificación
        - Principio de optimización (ALARA)
        - Limitación de dosis
        - Niveles de referencia diagnósticos (DRL)
        
        #### 📜 Responsabilidades del TSID
        
        Como técnico, eres responsable de:
        1. ✅ Aplicar protocolos de protección radiológica
        2. ✅ Verificar identidad del paciente y prescripción
        3. ✅ Optimizar técnica (mínima dosis, máxima calidad)
        4. ✅ Usar protecciones (colimación, protectores)
        5. ✅ Mantener dosímetro personal
        6. ✅ Participar en control de calidad
        7. ✅ Informar de anomalías o incidencias
        8. ✅ Formación continuada (actualización)
        
        **No eres responsable de**:
        - ❌ Justificación médica (responsabilidad del médico prescriptor)
        - ❌ Mantenimiento técnico complejo (servicio especializado)
        - ❌ Control de calidad oficial (radiofísico)
        
        Pero **sí debes colaborar** en todo lo anterior.
        
        ---
        
        ### 18. Conclusión y Mensaje Final
        
        El tubo de rayos X es el **corazón del sistema de imagen radiológica**. Entender 
        su funcionamiento físico te permite:
        
        🎯 **Optimizar técnicas**: Elegir parámetros apropiados para cada situación
        
        🛡️ **Proteger**: Minimizar dosis al paciente y a ti mismo
        
        🔧 **Resolver problemas**: Entender qué ajustar cuando algo va mal
        
        📈 **Mejorar calidad**: Saber qué factores afectan cada aspecto de la imagen
        
        💼 **Profesionalidad**: Comunicarte efectivamente con radiólogos y físicos
        
        **Recuerda siempre**:
        - Cada exposición debe estar **justificada**
        - Cada técnica debe estar **optimizada** (ALARA)
        - La calidad diagnóstica es prioritaria, pero no a cualquier dosis
        - La protección radiológica no es opcional, es **obligatoria**
        
        ---
        
        ### 📚 Recursos Adicionales Recomendados
        
        **Libros de referencia**:
        - *Radiologic Science for Technologists* - Bushong
        - *Física para Radiólogos* - Sprawls
        - *Manual SEPR de Protección Radiológica*
        
        **Organismos oficiales**:
        - **SEFM**: Sociedad Española de Física Médica
        - **SEPR**: Sociedad Española de Protección Radiológica
        - **CSN**: Consejo de Seguridad Nuclear (España)
        - **ICRP**: International Commission on Radiological Protection
        
        **Normativa actualizada**:
        - Portal del CSN: www.csn.es
        - BOE: Legislación vigente
        
        ¡Sigue explorando y experimentando con este simulador para afianzar estos conceptos! 
        La física no es solo teoría - **véla en acción** con los controles interactivos.
        """)
    
    # Additional interactive quiz/self-test section
    with st.expander("🎓 Autoevaluación: Pon a Prueba tus Conocimientos", expanded=False):
        st.markdown("""
        ### Preguntas de Repaso
        
        Intenta responder estas preguntas basándote en lo que has aprendido:
        """)
        
        quiz_col1, quiz_col2 = st.columns(2)
        
        with quiz_col1:
            st.markdown("""
            **Pregunta 1**: Si aumentas el kVp de 70 a 80 (≈15%), ¿qué debes hacer con el mAs 
            para mantener la misma densidad de imagen?
            
            <details>
            <summary>👉 Ver respuesta</summary>
            
            **Reducir el mAs a la mitad**. La regla del 15% indica que aumentar kVp en 15% 
            equivale a duplicar el mAs, por lo que para mantener la densidad constante 
            debes reducir el mAs al 50%.
            
            Ejemplo: 70 kVp + 20 mAs → 80 kVp + 10 mAs
            
            *Ventaja*: Menor dosis al paciente
            *Desventaja*: Menor contraste
            </details>
            
            ---
            
            **Pregunta 2**: ¿Por qué la radiación característica del tungsteno solo aparece 
            cuando usamos kVp ≥ 70?
            
            <details>
            <summary>👉 Ver respuesta</summary>
            
            Porque la **energía de enlace de la capa K del tungsteno es 69.5 keV**. 
            Los electrones incidentes deben tener al menos esa energía para ionizar 
            la capa K y producir radiación característica K-α (59.3 keV) y K-β (67.2 keV).
            
            Por debajo de 70 kVp, solo se produce radiación de frenado (Bremsstrahlung).
            </details>
            
            ---
            
            **Pregunta 3**: Realizas una técnica a 100 cm con 10 mAs. Por necesidades 
            del paciente, debes alejarte a 150 cm. ¿Qué mAs necesitas?
            
            <details>
            <summary>👉 Ver respuesta</summary>
            
            **22.5 mAs**
            
            Usando la ley del inverso del cuadrado:
            
            mAs₂ = mAs₁ × (D₂/D₁)²
            mAs₂ = 10 × (150/100)²
            mAs₂ = 10 × 2.25 = 22.5 mAs
            
            La intensidad disminuye con el cuadrado de la distancia, por lo que 
            necesitas más del doble de mAs para compensar.
            </details>
            """, unsafe_allow_html=True)
        
        with quiz_col2:
            st.markdown("""
            **Pregunta 4**: ¿Qué porcentaje de la energía de los electrones se convierte 
            en rayos X en un tubo típico?
            
            <details>
            <summary>👉 Ver respuesta</summary>
            
            **Solo ~1%**
            
            El otro 99% se convierte en calor en el ánodo. Esta baja eficiencia explica:
            - Por qué se necesitan ánodos giratorios
            - Por qué hay límites de carga térmica (HU)
            - Por qué el tubo necesita enfriarse entre series
            
            Eficiencia aproximada: η ≈ 10⁻⁹ × Z × kVp
            Para tungsteno (Z=74) a 100 kVp: η ≈ 0.74%
            </details>
            
            ---
            
            **Pregunta 5**: ¿Qué parámetro del tubo afecta SOLO a la cantidad de fotones 
            pero NO a su energía/penetración?
            
            <details>
            <summary>👉 Ver respuesta</summary>
            
            **El mAs (miliamperios-segundo)**
            
            - ↑ mAs → Más fotones (mayor intensidad)
            - Pero NO cambia la distribución de energías
            - NO afecta la penetración ni el contraste
            - Solo afecta la "cantidad" de radiación, no la "calidad"
            
            En cambio, el kVp afecta tanto cantidad (∝kVp²) como calidad (energía).
            </details>
            
            ---
            
            **Pregunta 6**: ¿Cuál es la principal ventaja de usar filtración adicional 
            en el tubo?
            
            <details>
            <summary>👉 Ver respuesta</summary>
            
            **Reducir la dosis en la piel del paciente sin perder calidad diagnóstica**
            
            La filtración:
            - Elimina fotones de baja energía (rayos "blandos")
            - Estos fotones no penetran al paciente → no contribuyen a la imagen
            - Pero SÍ depositan dosis en la piel
            - "Endurece" el haz (aumenta energía promedio)
            - Aumenta HVL
            
            Es un requisito legal: mínimo 2.5 mm Al total para equipos ≥70 kVp
            </details>
            """, unsafe_allow_html=True)
    
    # Key takeaways box
    st.markdown("---")
    st.success("""
    ### 🎯 Puntos Clave para Recordar
    
    1. **kVp controla ENERGÍA** (penetración, contraste) → Afecta "calidad" del haz
    2. **mAs controla CANTIDAD** (número de fotones) → Afecta "cantidad" del haz
    3. **Solo ~1% energía → RX**, 99% → calor (gestión térmica crítica)
    4. **Dos tipos de radiación**: Frenado (~80%) + Característica (~20%)
    5. **Filtración reduce dosis** cutánea eliminando fotones de baja energía
    6. **HVL mide "dureza"** del haz (mayor HVL = haz más penetrante)
    7. **Regla 15%**: ↑kVp 15% ≈ duplicar mAs (pero menos contraste)
    8. **Ley inversa cuadrado**: Dosis ∝ 1/distancia²
    9. **Foco pequeño** = mejor resolución (pero menor capacidad de carga)
    10. **ALARA siempre**: Mínima dosis compatible con calidad diagnóstica
    """)
    
    # Pro tips
    st.info("""
    ### 💡 Consejos Profesionales
    
    **Para optimizar tus técnicas**:
    - 📋 Consulta protocolos establecidos en tu centro
    - 🎯 Colima siempre al mínimo necesario
    - 👥 Adapta según morfología del paciente (delgado/obeso)
    - ⚡ Usa el menor tiempo posible (evita movimiento)
    - 🔄 Aprovecha AEC cuando esté disponible
    - 📊 Revisa tus imágenes críticamente (aprende de cada caso)
    - 📚 Actualízate constantemente (técnicas evolucionan)
    
    **Para protegerte**:
    - 🛡️ Usa siempre tu dosímetro personal
    - 🚪 Sal de la sala durante exposiciones (si es posible)
    - 🦺 Delantal plomado en portátiles/fluoroscopia (obligatorio)
    - 📏 Máxima distancia posible del paciente durante exposición
    - ⏱️ Minimiza tiempo de exposición a radiación
    - 🧤 Nunca sujetes pacientes durante exposiciones
    """)
    
    # Footer for this tab
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>⚡ <strong>Tab 1: Tubo de Rayos X</strong> | 
        Simulador de Física Radiológica | 
        Formación Profesional en Imagen para el Diagnóstico</p>
        <p>Experimenta con los controles superiores para ver cómo cada parámetro 
        afecta al espectro, dosis y características del haz</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# TAB 2: FORMACIÓN DE IMAGEN
# ============================================
with tabs[1]:
    st.header("🎯 Formación de Imagen Radiográfica")
    st.markdown("""
    Descubre cómo los rayos X interactúan con los tejidos del paciente para crear la imagen radiográfica.
    Experimenta con diferentes tejidos, energías y configuraciones.
    """)
    
    st.markdown("---")
    
    # Section selector
    section = st.radio(
        "Selecciona el concepto a explorar:",
        ["📊 Interacciones RX-Materia", "🔬 Constructor de Phantom", "🎯 Rejillas Anti-Dispersión", "📈 Contraste y Calidad"],
        horizontal=False
    )
    
    # ============================================
    # SECTION 1: X-RAY INTERACTIONS
    # ============================================
    if section == "📊 Interacciones RX-Materia":
        st.subheader("📊 Interacciones de Rayos X con la Materia")
        
        st.markdown("""
        Los rayos X pueden interactuar con la materia de tres formas principales. 
        La **probabilidad de cada interacción** depende de la energía del fotón y del material.
        """)
        
        # Controls
        interact_col1, interact_col2 = st.columns(2)
        
        with interact_col1:
            st.markdown("##### Parámetros del Haz")
            energy_interact = st.slider("Energía del fotón (keV)", 20, 150, 60, 1)
            
        with interact_col2:
            st.markdown("##### Tipo de Tejido")
            tissue_interact = st.selectbox(
                "Selecciona tejido",
                list(TISSUES.keys()),
                index=2  # Tejido blando
            )
        
        tissue_props = TISSUES[tissue_interact]
        Z_eff = tissue_props["Z_eff"]
        density = tissue_props["density"]
        
        # Calculate cross sections
        energies_range = np.linspace(20, 150, 200)
        photo_values = [photoelectric_cross_section(E, Z_eff) for E in energies_range]
        compton_values = [compton_cross_section(E) for E in energies_range]
        coherent_values = [coherent_scattering_cross_section(E, Z_eff) for E in energies_range]
        
        # Normalize for visualization
        max_val = max(max(photo_values), max(compton_values), max(coherent_values))
        photo_norm = np.array(photo_values) / max_val * 100
        compton_norm = np.array(compton_values) / max_val * 100
        coherent_norm = np.array(coherent_values) / max_val * 100
        total_interaction = photo_norm + compton_norm + coherent_norm
        
        # Plot interactions
        fig_interactions = go.Figure()
        
        fig_interactions.add_trace(go.Scatter(
            x=energies_range,
            y=photo_norm,
            mode='lines',
            name='Efecto Fotoeléctrico',
            line=dict(color='red', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.2)'
        ))
        
        fig_interactions.add_trace(go.Scatter(
            x=energies_range,
            y=compton_norm,
            mode='lines',
            name='Dispersión Compton',
            line=dict(color='blue', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 0, 255, 0.2)'
        ))
        
        fig_interactions.add_trace(go.Scatter(
            x=energies_range,
            y=coherent_norm,
            mode='lines',
            name='Dispersión Coherente',
            line=dict(color='green', width=3),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 0, 0.2)'
        ))
        
        # Mark current energy
        current_photo = photoelectric_cross_section(energy_interact, Z_eff) / max_val * 100
        current_compton = compton_cross_section(energy_interact) / max_val * 100
        current_coherent = coherent_scattering_cross_section(energy_interact, Z_eff) / max_val * 100
        
        fig_interactions.add_vline(
            x=energy_interact,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"Energía actual: {energy_interact} keV",
            annotation_position="top"
        )
        
        fig_interactions.update_layout(
            title=f"Probabilidad Relativa de Interacciones en {tissue_interact}",
            xaxis_title="Energía del Fotón (keV)",
            yaxis_title="Probabilidad Relativa (%)",
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(x=0.7, y=0.95)
        )
        
        st.plotly_chart(fig_interactions, use_container_width=True)
        
        # Show percentages at current energy
        total_current = current_photo + current_compton + current_coherent
        
        st.markdown("### 📊 A la Energía Actual")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            photo_percent = (current_photo / total_current * 100) if total_current > 0 else 0
            st.metric(
                "Efecto Fotoeléctrico",
                f"{photo_percent:.1f}%",
                help="Absorción completa del fotón"
            )
            
        with metric_col2:
            compton_percent = (current_compton / total_current * 100) if total_current > 0 else 0
            st.metric(
                "Dispersión Compton",
                f"{compton_percent:.1f}%",
                help="Dispersión inelástica, reduce energía"
            )
            
        with metric_col3:
            coherent_percent = (current_coherent / total_current * 100) if total_current > 0 else 0
            st.metric(
                "Dispersión Coherente",
                f"{coherent_percent:.1f}%",
                help="Dispersión elástica, sin pérdida energía"
            )
        
        # Tissue properties
        st.markdown("### 🧬 Propiedades del Tejido Seleccionado")
        prop_col1, prop_col2 = st.columns(2)
        
        with prop_col1:
            st.info(f"""
            **{tissue_interact}**
            
            - **Z efectivo**: {Z_eff}
            - **Densidad**: {density} g/cm³
            - **Coeficiente atenuación** (a {energy_interact} keV): {total_attenuation_coefficient(energy_interact, Z_eff, density):.3f} cm⁻¹
            """)
            
        with prop_col2:
            # Visual representation of interactions
            st.markdown("**Interacción Dominante:**")
            if photo_percent > compton_percent:
                st.error(f"🔴 **Fotoeléctrico** ({photo_percent:.0f}%)")
                st.caption("Absorción completa → Contribuye al contraste")
            else:
                st.info(f"🔵 **Compton** ({compton_percent:.0f}%)")
                st.caption("Dispersión → Reduce contraste, aumenta dosis")
        
        # Explanations
        st.markdown("---")
        
        expl_col1, expl_col2, expl_col3 = st.columns(3)
        
        with expl_col1:
            st.markdown("""
            #### 🔴 Efecto Fotoeléctrico
            
            **Proceso:**
            1. Fotón impacta electrón orbital
            2. Transfiere TODA su energía
            3. Electrón es expulsado
            4. Fotón desaparece (absorbido)
            
            **Dependencia:**
            - ∝ Z³ (fuerte dependencia del material)
            - ∝ 1/E³ (dominante a bajas energías)
            
            **Importancia:**
            - ✅ Genera CONTRASTE
            - ✅ Útil para imagen
            - ⚠️ Aumenta dosis
            """)
            
        with expl_col2:
            st.markdown("""
            #### 🔵 Dispersión Compton
            
            **Proceso:**
            1. Fotón choca con electrón libre
            2. Transfiere PARTE de su energía
            3. Fotón se desvía (dispersa)
            4. Continúa con menor energía
            
            **Dependencia:**
            - Casi independiente de Z
            - Disminuye con energía
            - Dominante a energías medias (60-100 keV)
            
            **Importancia:**
            - ❌ Reduce CONTRASTE (niebla)
            - ❌ Dosis al personal
            - Requiere rejilla anti-dispersión
            """)
            
        with expl_col3:
            st.markdown("""
            #### 🟢 Dispersión Coherente (Rayleigh)
            
            **Proceso:**
            1. Fotón interactúa con átomo completo
            2. NO hay transferencia energía
            3. Solo cambia dirección
            4. Mantiene misma energía
            
            **Dependencia:**
            - ∝ Z²
            - ∝ 1/E²
            - Contribución menor (<5%)
            
            **Importancia:**
            - Efecto pequeño en diagnóstico
            - Ignorado en cálculos simplificados
            """)
        
        # Interactive comparison
        st.markdown("---")
        st.subheader("🔄 Comparación entre Tejidos")
        
        compare_tissues = st.multiselect(
            "Selecciona tejidos para comparar",
            list(TISSUES.keys()),
            default=["Tejido blando", "Hueso cortical", "Pulmón"]
        )
        
        if len(compare_tissues) > 0:
            fig_compare_tissues = go.Figure()
            
            for tissue in compare_tissues:
                props = TISSUES[tissue]
                mu_values = [total_attenuation_coefficient(E, props["Z_eff"], props["density"]) 
                           for E in energies_range]
                
                fig_compare_tissues.add_trace(go.Scatter(
                    x=energies_range,
                    y=mu_values,
                    mode='lines',
                    name=tissue,
                    line=dict(width=3)
                ))
            
            fig_compare_tissues.update_layout(
                title="Coeficiente de Atenuación μ por Tipo de Tejido",
                xaxis_title="Energía del Fotón (keV)",
                yaxis_title="μ (cm⁻¹)",
                hovermode='x unified',
                height=450,
                showlegend=True,
                yaxis_type="log"  # Logarithmic scale for better visualization
            )
            
            st.plotly_chart(fig_compare_tissues, use_container_width=True)
            
            st.caption("📊 Escala logarítmica: los tejidos con mayor Z y densidad atenúan más la radiación")
    
    # ============================================
    # SECTION 2: PHANTOM CONSTRUCTOR
    # ============================================
    elif section == "🔬 Constructor de Phantom":
        st.subheader("🔬 Constructor de Paciente Virtual (Phantom)")
        
        st.markdown("""
        Construye tu propio "paciente virtual" apilando capas de diferentes tejidos.
        Observa cómo los rayos X se atenúan al atravesar cada capa y cómo se forma la imagen final.
        """)
        
        # kVp selection for phantom
        phantom_col1, phantom_col2 = st.columns(2)
        
        with phantom_col1:
            phantom_kVp = st.slider("kVp del haz", 40, 150, 80, 1, key="phantom_kvp")
            phantom_mAs = st.slider("mAs", 1, 100, 10, 1, key="phantom_mas")
            
        with phantom_col2:
            use_grid = st.checkbox("Usar rejilla anti-dispersión", value=True)
            if use_grid:
                grid_ratio = st.select_slider("Ratio de rejilla", options=[5, 8, 10, 12, 16], value=10)
            else:
                grid_ratio = 0
        
        # Phantom construction
        st.markdown("### 🏗️ Construye tu Phantom")
        st.caption("Añade capas de tejido de adelante hacia atrás (como atraviesa el haz de RX)")
        
        # Initialize phantom layers in session state
        if 'phantom_layers' not in st.session_state:
            st.session_state.phantom_layers = [
                create_phantom_layer("Tejido blando", 5),
                create_phantom_layer("Hueso cortical", 2),
                create_phantom_layer("Tejido blando", 8)
            ]
        
        # Add layer interface
        add_col1, add_col2, add_col3 = st.columns([2, 2, 1])
        
        with add_col1:
            new_tissue = st.selectbox("Tipo de tejido", list(TISSUES.keys()), key="new_tissue")
        
        with add_col2:
            new_thickness = st.number_input("Espesor (cm)", 0.1, 20.0, 2.0, 0.5, key="new_thickness")
        
        with add_col3:
            st.markdown("")
            st.markdown("")
            if st.button("➕ Añadir capa"):
                st.session_state.phantom_layers.append(create_phantom_layer(new_tissue, new_thickness))
                st.rerun()
        
        # Display current layers
        st.markdown("### 📋 Capas Actuales del Phantom")
        
        if len(st.session_state.phantom_layers) == 0:
            st.warning("No hay capas. Añade al menos una capa de tejido.")
        else:
            for idx, layer in enumerate(st.session_state.phantom_layers):
                layer_col1, layer_col2, layer_col3, layer_col4 = st.columns([3, 2, 2, 1])
                
                with layer_col1:
                    st.markdown(f"**{idx + 1}.** {layer['type']}")
                
                with layer_col2:
                    st.text(f"Espesor: {layer['thickness']:.1f} cm")
                
                with layer_col3:
                    st.text(f"Z={layer['Z_eff']:.1f}, ρ={layer['density']:.2f}")
                
                with layer_col4:
                    if st.button("🗑️", key=f"delete_{idx}"):
                        st.session_state.phantom_layers.pop(idx)
                        st.rerun()
            
            # Clear all button
            if st.button("🗑️ Limpiar todo"):
                st.session_state.phantom_layers = []
                st.rerun()
            
            # Calculate total thickness
            total_thickness = sum(layer['thickness'] for layer in st.session_state.phantom_layers)
            st.info(f"**Espesor total del phantom**: {total_thickness:.1f} cm")
        
        # Simulate if layers exist
        if len(st.session_state.phantom_layers) > 0:
            st.markdown("---")
            
            # Calculate effective energy from kVp
            eff_energy = phantom_kVp * 0.4  # Approximation
            
            # Calculate transmission through each layer
            transmissions = []
            intensities = [100]  # Start with 100% intensity
            
            for layer in st.session_state.phantom_layers:
                mu = total_attenuation_coefficient(eff_energy, layer['Z_eff'], layer['density'])
                trans = calculate_transmission(layer['thickness'], mu)
                transmissions.append(trans)
                intensities.append(intensities[-1] * trans)
            
            # Monte Carlo simulation
            transmitted, absorbed, scattered = simulate_photon_path(
                st.session_state.phantom_layers, 
                eff_energy, 
                num_photons=1000
            )
            
            # Visualization
            vis_col1, vis_col2 = st.columns([2, 1])
            
            with vis_col1:
                st.markdown("### 📊 Atenuación del Haz")
                
                # Create attenuation diagram
                fig_atten = go.Figure()
                
                # Plot intensity through layers
                positions = [0]
                current_pos = 0
                for layer in st.session_state.phantom_layers:
                    current_pos += layer['thickness']
                    positions.append(current_pos)
                
                fig_atten.add_trace(go.Scatter(
                    x=positions,
                    y=intensities,
                    mode='lines+markers',
                    name='Intensidad del haz',
                    line=dict(color='blue', width=3),
                    marker=dict(size=10)
                ))
                
                # Add shaded regions for each layer
                current_pos = 0
                for idx, layer in enumerate(st.session_state.phantom_layers):
                    fig_atten.add_vrect(
                        x0=current_pos,
                        x1=current_pos + layer['thickness'],
                        fillcolor=TISSUES[layer['type']]['color'],
                        opacity=0.3,
                        layer="below",
                        line_width=0,
                        annotation_text=layer['type'],
                        annotation_position="top"
                    )
                    current_pos += layer['thickness']
                
                fig_atten.update_layout(
                    title="Atenuación del Haz a través del Phantom",
                    xaxis_title="Posición (cm)",
                    yaxis_title="Intensidad Relativa (%)",
                    height=400,
                    showlegend=True
                )
                
                st.plotly_chart(fig_atten, use_container_width=True)
            
            with vis_col2:
                st.markdown("### 🎯 Resultados")
                
                final_intensity = intensities[-1]
                
                st.metric(
                    "Transmisión Total",
                    f"{final_intensity:.1f}%",
                    delta=f"-{100-final_intensity:.1f}%",
                    delta_color="inverse"
                )
                
                st.metric(
                    "Fotones Transmitidos",
                    f"{transmitted}/1000",
                    help="Simulación Monte Carlo"
                )
                
                st.metric(
                    "Fotones Absorbidos",
                    f"{absorbed}/1000",
                    help="Efecto fotoeléctrico principalmente"
                )
                
                st.metric(
                    "Fotones Dispersos",
                    f"{scattered}/1000",
                    help="Dispersión Compton principalmente"
                )
                
                # Grid effect
                if use_grid:
                    primary_through_grid = final_intensity * calculate_grid_transmission(grid_ratio, 40, is_scatter=False)
                    scatter_through_grid = (scattered/1000*100) * calculate_grid_transmission(grid_ratio, 40, is_scatter=True)
                    
                    st.markdown("---")
                    st.markdown("**Con Rejilla:**")
                    st.metric(
                        "Primarios en detector",
                        f"{primary_through_grid:.1f}%"
                    )
                    st.metric(
                        "Dispersión en detector",
                        f"{scatter_through_grid:.1f}%",
                        delta=f"-{(scattered/1000*100 - scatter_through_grid):.1f}%",
                        delta_color="inverse"
                    )
            
            # Photon paths visualization
            st.markdown("---")
            st.markdown("### 🌟 Simulación de Trayectorias de Fotones")
            
            if st.button("🎬 Simular Trayectorias de Fotones"):
                # Create visual simulation
                fig_photons = go.Figure()
                
                # Draw layers
                current_x = 0
                for layer in st.session_state.phantom_layers:
                    fig_photons.add_shape(
                        type="rect",
                        x0=current_x, x1=current_x + layer['thickness'],
                        y0=0, y1=10,
                        fillcolor=TISSUES[layer['type']]['color'],
                        opacity=0.3,
                        line=dict(color="black", width=1)
                    )
                    # Label
                    fig_photons.add_annotation(
                        x=current_x + layer['thickness']/2,
                        y=9,
                        text=layer['type'],
                        showarrow=False,
                        font=dict(size=10)
                    )
                    current_x += layer['thickness']
                
                # Simulate some photon paths
                np.random.seed(42)
                num_visual_photons = 20
                
                for i in range(num_visual_photons):
                    y_start = i * (10 / num_visual_photons)
                    x_path = [0]
                    y_path = [y_start]
                    
                    current_x = 0
                    photon_alive = True
                    color = 'green'
                    
                    for layer in st.session_state.phantom_layers:
                        if not photon_alive:
                            break
                        
                        mu = total_attenuation_coefficient(eff_energy, layer['Z_eff'], layer['density'])
                        prob_interaction = 1 - np.exp(-mu * layer['thickness'])
                        
                        if np.random.random() < prob_interaction:
                            # Interaction occurs
                            interaction_point = current_x + np.random.random() * layer['thickness']
                            x_path.append(interaction_point)
                            y_path.append(y_start + np.random.normal(0, 0.5))
                            
                            # Determine if absorbed or scattered
                            if np.random.random() < 0.4:  # Absorbed
                                color = 'red'
                                photon_alive = False
                            else:  # Scattered
                                color = 'orange'
                                y_start += np.random.normal(0, 1)
                        
                        current_x += layer['thickness']
                    
                    if photon_alive:
                        x_path.append(current_x)
                        y_path.append(y_start)
                        color = 'green'
                    
                    fig_photons.add_trace(go.Scatter(
                        x=x_path,
                        y=y_path,
                        mode='lines',
                        line=dict(color=color, width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                
                fig_photons.update_layout(
                    title="Trayectorias de Fotones (Simulación)",
                    xaxis_title="Posición (cm)",
                    yaxis_title="",
                    height=400,
                    yaxis=dict(showticklabels=False),
                    showlegend=False
                )
                
                st.plotly_chart(fig_photons, use_container_width=True)
                
                st.caption("""
                🟢 Verde = Transmitido | 🟠 Naranja = Dispersado | 🔴 Rojo = Absorbido
                """)
            
            # Beer-Lambert calculation
            st.markdown("---")
            st.markdown("### 📐 Ley de Beer-Lambert")
            
            st.latex(r"I = I_0 \times e^{-\mu x}")
            
            st.markdown("""
            Donde:
            - **I** = Intensidad tras atravesar el material
            - **I₀** = Intensidad inicial
            - **μ** = Coeficiente de atenuación lineal (cm⁻¹)
            - **x** = Espesor del material (cm)
            """)
            
            # Show calculation for each layer
            with st.expander("📊 Ver cálculos detallados por capa"):
                calc_data = []
                current_intensity = 100
                
                for idx, layer in enumerate(st.session_state.phantom_layers):
                    mu = total_attenuation_coefficient(eff_energy, layer['Z_eff'], layer['density'])
                    trans = calculate_transmission(layer['thickness'], mu)
                    intensity_after = current_intensity * trans
                    attenuation_percent = (1 - trans) * 100
                    
                    calc_data.append({
                        "Capa": f"{idx+1}. {layer['type']}",
                        "Espesor (cm)": f"{layer['thickness']:.1f}",
                        "μ (cm⁻¹)": f"{mu:.3f}",
                        "Transmisión": f"{trans*100:.1f}%",
                        "Atenuación": f"{attenuation_percent:.1f}%",
                        "I antes": f"{current_intensity:.1f}%",
                        "I después": f"{intensity_after:.1f}%"
                    })
                    
                    current_intensity = intensity_after
                
                df_calc = pd.DataFrame(calc_data)
                st.dataframe(df_calc, use_container_width=True)
        
        # Theory section
        with st.expander("📚 Teoría: Formación de la Imagen Radiográfica"):
            st.markdown("""
            ## 🎯 Cómo se Forma la Imagen
            
            ### Principio Fundamental
            
            La imagen radiográfica se basa en la **atenuación diferencial** de los rayos X 
            al atravesar tejidos de diferente densidad y composición.
            
            **Proceso**:
            1. **Haz uniforme** sale del tubo
            2. **Atraviesa el paciente** → diferentes tejidos atenúan diferente
            3. **Haz modulado** (con información anatómica) llega al detector
            4. **Detector** convierte RX en señal eléctrica/imagen
            
            ### Factores que Determinan la Atenuación
            
            #### 1. Número Atómico (Z)
            
            - Mayor Z → Mayor atenuación (especialmente fotoeléctrico)
            - **Ejemplos**:
              - Hueso (Ca, Z=20): Alta atenuación → Blanco en imagen
              - Tejido blando (C, H, O, N; Z~7): Media atenuación → Gris
              - Aire (Z~7.6 pero baja densidad): Baja atenuación → Negro
            
            #### 2. Densidad Física (ρ)
            
            - Más átomos por cm³ → Mayor probabilidad de interacción
            - **Ejemplos**:
              - Hueso cortical: 1.92 g/cm³
              - Músculo: 1.05 g/cm³
              - Pulmón: 0.3 g/cm³
              - Aire: 0.001 g/cm³
            
            #### 3. Espesor del Tejido (x)
            
            - Relación exponencial: duplicar espesor NO duplica atenuación
            - Ley de Beer-Lambert
            
            #### 4. Energía del Haz (kVp)
            
            - Mayor energía → Menor atenuación (más penetración)
            - Afecta el contraste entre tejidos
            
            ### Contraste Radiográfico
            
            **Definición**: Diferencia de intensidad entre dos regiones adyacentes
            """)
            
            st.latex(r"Contraste = \frac{|I_1 - I_2|}{I_1 + I_2}")
            
            st.markdown("""
            **Tipos de contraste**:
            
            1. **Contraste de Sujeto**: Inherente al paciente
               - Depende de diferencias anatómicas
               - No podemos modificarlo
            
            2. **Contraste Radiográfico**: En el haz que sale
               - Depende de kVp, espesor, Z
               - Lo controlamos con técnica
            
            3. **Contraste de la Imagen**: Lo que vemos
               - Depende también del detector y procesado
               - Post-procesado digital puede modificarlo
            
            ### Optimización del Contraste
            
            **Para MAXIMIZAR contraste**:
            - ✅ Usar kVp bajo (↑ efecto fotoeléctrico)
            - ✅ Aplicable solo en partes delgadas
            - ✅ Ejemplo: Extremidades (50-60 kVp)
            
            **Para PARTES GRUESAS**:
            - ↑ kVp para penetración (sacrifica contraste)
            - Compensar con procesado digital
            - Usar rejilla (elimina dispersión)
            
            ### Ley de Beer-Lambert Extendida
            
            Para múltiples capas de diferentes materiales:
            """)
            
            st.latex(r"I = I_0 \times e^{-(\mu_1 x_1 + \mu_2 x_2 + ... + \mu_n x_n)}")
            
            st.markdown("""
            O equivalente:
            """)
            
            st.latex(r"I = I_0 \times e^{-\mu_1 x_1} \times e^{-\mu_2 x_2} \times ... \times e^{-\mu_n x_n}")
            
            st.markdown("""
            Cada capa de tejido atenúa el haz de forma independiente y multiplicativa.
            
            ### Dispersión: El Enemigo del Contraste
            
            **Problema**: Los fotones dispersados (Compton) llegan al detector desde 
            direcciones incorrectas, creando una "niebla" que reduce el contraste.
            
            **Cantidad de dispersión depende de**:
            - ↑ Volumen irradiado (área × espesor)
            - ↑ kVp (más Compton)
            - Tipo de tejido (agua/tejido blando genera más)
            
            **Relación dispersión/primarios**:
            - Extremidad: ~0.5:1 (tolerable sin rejilla)
            - Abdomen: ~5:1 (requiere rejilla)
            - Paciente obeso: >10:1 (rejilla obligatoria)
            
            **Soluciones**:
            1. **Colimación estricta**: ↓ volumen irradiado
            2. **Rejilla anti-dispersión**: Elimina fotones oblicuos
            3. **Air gap**: Distancia entre paciente y detector
            4. **Procesado digital**: Reducción software (limitado)
            
            ### Aplicación Práctica
            
            **Caso típico: Tórax PA**
            
            Atraviesa:
            1. Tejido blando (pared torácica): ~2 cm
            2. Pulmón (aire): ~20 cm
            3. Mediastino (tejido + sangre): ~8 cm
            4. Pulmón (aire): ~20 cm
            5. Tejido blando (pared posterior): ~2 cm
            
            **Resultado**:
            - Campos pulmonares: Baja atenuación → Negro (estructura vascular visible)
            - Mediastino: Alta atenuación → Blanco
            - Costillas: Muy alta atenuación → Blanco brillante
            - Contraste natural excelente (alto kVp posible: 110-125)
            
            **Caso típico: Abdomen AP**
            
            Atraviesa:
            - Principalmente tejido blando/agua
            - Espesor variable (15-30 cm)
            - Poco contraste natural (todo similar Z y ρ)
            
            **Resultado**:
            - Bajo contraste inherente
            - Requiere kVp moderado (70-80) para contraste
            - Gas intestinal proporciona contraste natural
            - Contraste artificial (Ba, I) a veces necesario
            """)
    
    # ============================================
    # SECTION 3: ANTI-SCATTER GRIDS
    # ============================================
    elif section == "🎯 Rejillas Anti-Dispersión":
        st.subheader("🎯 Rejillas Anti-Dispersión")
        
        st.markdown("""
        Las rejillas eliminan la radiación dispersa que degrada el contraste de la imagen.
        Explora cómo diferentes configuraciones de rejilla afectan la calidad de imagen y la dosis.
        """)
        
        # Grid parameters
        grid_col1, grid_col2, grid_col3 = st.columns(3)
        
        with grid_col1:
            st.markdown("##### Parámetros de la Rejilla")
            grid_ratio_section = st.select_slider(
                "Ratio de rejilla (r)",
                options=[5, 6, 8, 10, 12, 16],
                value=10,
                help="Relación altura de las láminas / distancia entre ellas"
            )
            
        with grid_col2:
            grid_frequency = st.slider(
                "Frecuencia (líneas/cm)",
                20, 80, 40, 5,
                help="Número de líneas de plomo por centímetro"
            )
            
        with grid_col3:
            grid_type = st.selectbox(
                "Tipo de rejilla",
                ["Lineal", "Cruzada"],
                help="Lineal: líneas en una dirección. Cruzada: dos direcciones perpendiculares"
            )
        
        # Scenario selection
        st.markdown("### 📋 Escenario Clínico")
        grid_scenario = st.selectbox(
            "Selecciona anatomía",
            ["Tórax PA", "Abdomen AP", "Pelvis AP", "Columna Lumbar LAT", "Extremidad (sin rejilla)"]
        )
        
        # Define scenarios with scatter-to-primary ratios
        scenarios_data = {
            "Tórax PA": {"thickness": 25, "scatter_ratio": 1.5, "kVp_typical": 120},
            "Abdomen AP": {"thickness": 25, "scatter_ratio": 5.0, "kVp_typical": 75},
            "Pelvis AP": {"thickness": 25, "scatter_ratio": 6.0, "kVp_typical": 80},
            "Columna Lumbar LAT": {"thickness": 35, "scatter_ratio": 8.0, "kVp_typical": 90},
            "Extremidad (sin rejilla)": {"thickness": 8, "scatter_ratio": 0.3, "kVp_typical": 55}
        }
        
        scenario_params = scenarios_data[grid_scenario]
        scatter_to_primary = scenario_params["scatter_ratio"]
        
        # Calculate grid performance
        primary_transmission = calculate_grid_transmission(grid_ratio_section, grid_frequency, is_scatter=False)
        scatter_transmission = calculate_grid_transmission(grid_ratio_section, grid_frequency, is_scatter=True)
        
        # With and without grid
        primary_signal = 100
        scatter_signal = primary_signal * scatter_to_primary
        
        # Without grid
        total_without_grid = primary_signal + scatter_signal
        contrast_without = primary_signal / total_without_grid
        
        # With grid
        primary_through_grid = primary_signal * primary_transmission
        scatter_through_grid = scatter_signal * scatter_transmission
        total_with_grid = primary_through_grid + scatter_through_grid
        contrast_with = primary_through_grid / total_with_grid if total_with_grid > 0 else 0
        
        # Contrast improvement factor
        contrast_improvement = contrast_with / contrast_without if contrast_without > 0 else 1
        
        # Bucky factor (dose increase needed)
        bucky_factor = 1 / primary_transmission
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Resultados")
        
        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        
        with result_col1:
            st.metric(
                "Mejora de Contraste",
                f"{contrast_improvement:.2f}x",
                help="Factor de mejora del contraste con rejilla vs sin rejilla"
            )
            
        with result_col2:
            st.metric(
                "Factor Bucky",
                f"{bucky_factor:.2f}x",
                delta="Aumento de dosis necesario",
                delta_color="inverse",
                help="Factor de aumento de mAs necesario para compensar absorción de la rejilla"
            )
            
        with result_col3:
            st.metric(
                "Dispersión Eliminada",
                f"{(1-scatter_transmission)*100:.0f}%",
                help="Porcentaje de radiación dispersa bloqueada"
            )
            
        with result_col4:
            selectivity = scatter_transmission / primary_transmission if primary_transmission > 0 else 0
            st.metric(
                "Selectividad",
                f"{selectivity:.2f}",
                help="Ratio scatter_trans/primary_trans. Menor es mejor"
            )
        
        # Visualization
        vis_col1, vis_col2 = st.columns(2)
        
        with vis_col1:
            st.markdown("#### Sin Rejilla")
            
            fig_without = go.Figure()
            
            fig_without.add_trace(go.Bar(
                x=['Primaria', 'Dispersa'],
                y=[primary_signal, scatter_signal],
                marker_color=['blue', 'red'],
                text=[f'{primary_signal:.0f}', f'{scatter_signal:.0f}'],
                textposition='auto'
            ))
            
            fig_without.update_layout(
                title=f"Radiación en Detector (Sin Rejilla)",
                yaxis_title="Intensidad Relativa",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_without, use_container_width=True)
            
            st.info(f"""
            **Contraste**: {contrast_without*100:.1f}%
            
            La dispersión ({scatter_signal:.0f}) degrada significativamente 
            la calidad de la imagen.
            """)
            
        with vis_col2:
            st.markdown("#### Con Rejilla")
            
            fig_with = go.Figure()
            
            fig_with.add_trace(go.Bar(
                x=['Primaria', 'Dispersa'],
                y=[primary_through_grid, scatter_through_grid],
                marker_color=['darkblue', 'darkred'],
                text=[f'{primary_through_grid:.0f}', f'{scatter_through_grid:.0f}'],
                textposition='auto'
            ))
            
            fig_with.update_layout(
                title=f"Radiación en Detector (Con Rejilla {grid_ratio_section}:1)",
                yaxis_title="Intensidad Relativa",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_with, use_container_width=True)
            
            st.success(f"""
            **Contraste**: {contrast_with*100:.1f}%
            
            Mejora de contraste: **{contrast_improvement:.2f}x**
            
            ⚠️ Pero requiere **{bucky_factor:.2f}x** más dosis
            """)
        
        # Grid diagram
        st.markdown("---")
        st.markdown("### 🔬 Estructura de la Rejilla")
        
        # Visual representation of grid
        fig_grid = go.Figure()
        
        # Draw grid lines
        num_lines = 15
        for i in range(num_lines):
            x_pos = i * (10 / num_lines)
            # Grid septa
            fig_grid.add_shape(
                type="rect",
                x0=x_pos, x1=x_pos + 0.1,
                y0=0, y1=grid_ratio_section,
                fillcolor="gray",
                line=dict(color="black", width=1)
            )
        
        # Add some photon paths
        # Primary (vertical)
        for i in range(3):
            x_primary = 2 + i * 3
            fig_grid.add_annotation(
                x=x_primary,
                y=grid_ratio_section + 1,
                ax=x_primary,
                ay=-1,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=3,
                arrowcolor='blue',
            )
            fig_grid.add_annotation(
                x=x_primary,
                y=grid_ratio_section + 1.5,
                text="Primario ✓",
                showarrow=False,
                font=dict(color='blue', size=10)
            )
        
        # Scattered (oblique) - blocked
        for i in range(2):
            x_start = 1.5 + i * 4
            x_end = x_start + 2
            fig_grid.add_annotation(
                x=x_end,
                y=-0.5,
                ax=x_start,
                ay=grid_ratio_section + 1,
                xref='x', yref='y',
                axref='x', ayref='y',
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=3,
                arrowcolor='red',
            )
            fig_grid.add_annotation(
                x=x_start + 1,
                y=grid_ratio_section/2,
                text="✗",
                showarrow=False,
                font=dict(color='red', size=20)
            )
        
        # Add labels
        fig_grid.add_annotation(
            x=5, y=-2,
            text=f"Ratio = Altura (h) / Distancia (d) = {grid_ratio_section}:1<br>Frecuencia = {grid_frequency} líneas/cm",
            showarrow=False,
            font=dict(size=12)
        )
        
        fig_grid.add_annotation(
            x=0.5, y=grid_ratio_section/2,
            text=f"h = {grid_ratio_section}",
            showarrow=True,
            arrowhead=2,
            ax=0.5,
            ay=0,
            font=dict(size=10)
        )
        
        fig_grid.update_layout(
            title="Principio de Funcionamiento de la Rejilla",
            xaxis=dict(range=[-0.5, 10.5], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(range=[-3, grid_ratio_section + 2], showgrid=False, zeroline=False, showticklabels=False),
            height=400,
            showlegend=False,
            plot_bgcolor='white'
        )
        
        st.plotly_chart(fig_grid, use_container_width=True)
        
        st.caption("""
        🔵 **Azul**: Radiación primaria (perpendicular) → Pasa a través de la rejilla
        🔴 **Rojo**: Radiación dispersa (oblicua) → Bloqueada por las láminas de plomo
        """)
        
        # Recommendations table
        st.markdown("---")
        st.markdown("### 📋 Recomendaciones por Anatomía")
        
        recommendations = {
            "Anatomía": ["Extremidades", "Tórax PA", "Abdomen", "Pelvis", "Columna Lumbar LAT"],
            "Espesor (cm)": ["< 10", "20-25", "20-30", "20-25", "30-40"],
            "Rejilla Necesaria": ["No", "Sí", "Sí", "Sí", "Sí"],
            "Ratio Recomendado": ["-", "8:1 - 10:1", "10:1 - 12:1", "10:1 - 12:1", "12:1 - 16:1"],
            "Frecuencia": ["-", "40-50", "40-50", "40-50", "40-60"],
            "Factor Bucky Típico": ["1.0", "3-4", "4-5", "4-5", "5-6"]
        }
        
        df_recommendations = pd.DataFrame(recommendations)
        st.dataframe(df_recommendations, use_container_width=True)
        
        # Grid selection tool
        st.markdown("---")
        st.markdown("### 🎯 Asistente de Selección de Rejilla")
        
        assist_col1, assist_col2 = st.columns(2)
        
        with assist_col1:
            patient_thickness = st.slider("Espesor del paciente (cm)", 5, 45, 25)
            exam_kVp = st.slider("kVp de la técnica", 40, 150, 80)
            
        with assist_col2:
            # Recommendation logic
            if patient_thickness < 10:
                recommended_grid = "Sin rejilla"
                recommended_ratio = "-"
                reason = "Parte delgada, poca dispersión generada"
            elif patient_thickness < 20:
                recommended_grid = "Rejilla 8:1"
                recommended_ratio = "8:1"
                reason = "Espesor moderado, rejilla de ratio bajo suficiente"
            elif patient_thickness < 30:
                recommended_grid = "Rejilla 10:1 o 12:1"
                recommended_ratio = "10:1 - 12:1"
                reason = "Espesor estándar, rejilla de ratio medio óptimo"
            else:
                recommended_grid = "Rejilla 12:1 o 16:1"
                recommended_ratio = "12:1 - 16:1"
                reason = "Parte muy gruesa, alta dispersión, necesita ratio alto"
            
            st.success(f"""
            **Recomendación**: {recommended_grid}
            
            **Ratio**: {recommended_ratio}
            
            **Razón**: {reason}
            """)
            
            # Additional considerations
            if exam_kVp > 100:
                st.warning("⚠️ Alto kVp aumenta dispersión Compton. Considera ratio mayor.")
            
            if exam_kVp < 60:
                st.info("💡 Bajo kVp (menos dispersión). Rejilla de ratio bajo o sin rejilla puede ser suficiente.")
        
        # Theory expander
        with st.expander("📚 Teoría: Rejillas Anti-Dispersión"):
            st.markdown("""
            ## 🎯 Fundamentos de las Rejillas Anti-Dispersión
            
            ### Problema: La Dispersión Compton
            
            Cuando los rayos X atraviesan el paciente:
            - **Radiación primaria**: Sale en línea recta, porta información anatómica
            - **Radiación dispersa**: Sale en todas direcciones, NO porta información útil
            
            La dispersión crea una **"niebla"** uniforme que:
            - ❌ Reduce el contraste
            - ❌ Degrada la calidad de imagen
            - ❌ Dificulta el diagnóstico
            
            ### Solución: La Rejilla
            
            **Principio**: Láminas de plomo muy finas y paralelas que actúan como "filtro direccional"
            
            - ✅ Radiación perpendicular (primaria) → Pasa entre las láminas
            - ❌ Radiación oblicua (dispersa) → Bloqueada por las láminas
            
            ### Parámetros de la Rejilla
            
            #### 1. Ratio de Rejilla (r)
            
            Relación entre altura (h) de las láminas y distancia (d) entre ellas:
            """)
            
            st.latex(r"r = \frac{h}{d}")
            
            st.markdown("""
            **Ejemplos**:
            - **5:1**: Rejilla "suave" (poco selectiva)
            - **8:1**: Estándar para radiología general
            - **12:1**: Alta selectividad (partes gruesas)
            - **16:1**: Muy selectiva (máxima eliminación dispersión)
            
            **Efecto del ratio**:
            - ↑ Ratio → ↑ Eliminación de dispersión → ↑ Contraste
            - ↑ Ratio → ↑ Absorción de primarios → ↑ Dosis necesaria
            - ↑ Ratio → ↑ Criticidad de alineación (más errores si mal centrado)
            
            #### 2. Frecuencia (líneas/cm)
            
            Número de láminas de plomo por centímetro.
            
            **Rangos típicos**:
            - **Baja frecuencia** (30-40 líneas/cm): Láminas visibles, económica
            - **Alta frecuencia** (60-80 líneas/cm): Láminas invisibles, mejor estética
            
            **Trade-off**:
            - ↑ Frecuencia → Líneas menos visibles → Mejor apariencia
            - ↑ Frecuencia → Más plomo → Mayor absorción primarios
            
            #### 3. Tipo de Rejilla
            
            **Lineal**:
            - Láminas en una sola dirección
            - Permite angulación perpendicular a las líneas
            - Estándar en radiología general
            
            **Cruzada (Crossed)**:
            - Dos rejillas lineales perpendiculares
            - Elimina dispersión en todas direcciones
            - Mayor selectividad pero NO permite angulación
            - Usado en TC (detector fijo)
            
            **Enfocada vs Paralela**:
            - **Enfocada**: Láminas convergen hacia foco del tubo
            - **Paralela**: Láminas paralelas (portátiles, fluoroscopia)
            
            ### Factor Bucky (Factor de Rejilla)
            
            **Definición**: Factor de aumento de dosis necesario para compensar 
            la absorción de radiación primaria por la rejilla.
            """)
            
            st.latex(r"B = \frac{\text{mAs con rejilla}}{\text{mAs sin rejilla}}")
            
            st.markdown("""
            **Valores típicos**:
            - Rejilla 5:1 → B ≈ 2
            - Rejilla 8:1 → B ≈ 3-4
            - Rejilla 10:1 → B ≈ 4-5
            - Rejilla 12:1 → B ≈ 5-6
            - Rejilla 16:1 → B ≈ 6-8
            
            **Implicación práctica**: Si usas rejilla, debes multiplicar el mAs por el factor Bucky.
            
            ### Selectividad de la Rejilla
            
            Mide la capacidad de discriminar entre primaria y dispersa:
            """)
            
            st.latex(r"\Sigma = \frac{T_p}{T_s}")
            
            st.markdown("""
            Donde:
            - T_p = Transmisión de primaria
            - T_s = Transmisión de dispersa
            
            **Mayor selectividad** (Σ alto) = Mejor rejilla
            
            ### Contrast Improvement Factor (CIF)
            
            Medida de mejora de contraste:
            """)
            
            st.latex(r"CIF = \frac{C_{\text{con rejilla}}}{C_{\text{sin rejilla}}}")
            
            st.markdown("""
            Valores típicos: 1.5 - 4.0 dependiendo de:
            - Ratio de rejilla
            - Cantidad de dispersión (espesor, kVp)
            - Anatomía
            
            ### Errores Comunes con Rejillas
            
            #### 1. Error de Centrado (Grid Cut-Off)
            
            **Causa**: Rayo central no perpendicular al centro de rejilla
            
            **Efecto**: 
            - Pérdida de densidad en uno o ambos lados
            - Imagen más clara en zonas periféricas
            
            **Prevención**:
            - Centrar correctamente haz y rejilla
            - Distancia correcta (focal distance)
            
            #### 2. Inversión de Rejilla (Upside-Down)
            
            **Causa**: Rejilla enfocada instalada al revés
            
            **Efecto**:
            - Imagen muy clara (subexpuesta)
            - Bordes oscuros característicos
            
            **Prevención**:
            - Verificar marca de "tube side"
            - No ocurre con rejillas paralelas
            
            #### 3. Grid Lines Visibles
            
            **Causa**:
            - Rejilla estacionaria (no Bucky)
            - Baja frecuencia
            - Imagen digital sobre-procesada
            
            **Solución**:
            - Usar Bucky móvil (mueve rejilla durante exposición)
            - Alta frecuencia (>60 líneas/cm)
            - Ajustar procesado
            
            #### 4. Error de Distancia Focal
            
            **Causa**: Usar rejilla enfocada a distancia incorrecta
            
            **Efecto**: Cut-off periférico
            
            **Prevención**: Respetar distancia focal de rejilla (ej: 100 cm)
            
            ### Alternativas a la Rejilla
            
            #### 1. Air Gap Technique
            
            Aumentar distancia paciente-detector (15-20 cm):
            - Dispersión "falla" el detector (divergencia geométrica)
            - No requiere rejilla
            - Pero: ↑ Magnificación, ↓ Resolución
            - Usado en: Radiografía lateral de columna cervical
            
            #### 2. Colimación Estricta
            
            - Reduce volumen irradiado → Menos dispersión generada
            - Siempre primer paso de optimización
            - Complementa (no sustituye) rejilla
            
            #### 3. kVp Óptimo
            
            - kVp bajo → Menos Compton → Menos dispersión
            - Pero: Solo aplicable en partes delgadas
            
            ### Decisión: ¿Usar o No Rejilla?
            
            **Usar rejilla SI**:
            - ✅ Espesor > 10-12 cm
            - ✅ kVp > 70
            - ✅ Área grande (> 20×20 cm)
            - ✅ Anatomía densa (abdomen, pelvis)
            
            **NO usar rejilla SI**:
            - ✅ Espesor < 10 cm
            - ✅ Pediatría (minimizar dosis)
            - ✅ Extremidades distales
            - ✅ Mamografía (técnica especial)
            
            ### Rejillas en Modalidades Especiales
            
            #### TC (Tomografía Computarizada)
            
            - **Rejillas lineales** enfocadas al foco
            - **Alta frecuencia** (>60 líneas/cm) para invisibilidad
            - **Ratio moderado** (8:1 - 10:1)
            - Orientación 1D permite geometría helicoidal
            - Algunos sistemas modernos: Sin rejilla (colimación post-paciente)
            
            #### Fluoroscopia
            
            - **Rejilla móvil** (reciprocating Bucky) obligatoria
            - **Ratio bajo-medio** (6:1 - 8:1) para permitir angulación
            - Movimiento durante exposición elimina líneas
            - Crítico por tiempo de exposición largo
            
            #### Radiografía Digital
            
            - Mismos principios que analógica
            - **Ventaja**: Post-procesado puede mejorar contraste
            - **Riesgo**: "Creep" de dosis (sobreexposición no visible)
            - Rejilla sigue siendo necesaria
            
            #### Mamografía
            
            - **Rejilla especial** de ratio bajo (4:1 - 5:1)
            - **Muy alta frecuencia** (>70 líneas/cm)
            - Material: Fibra de carbono (radiotransparente)
            - Móvil para eliminar líneas
            - Crítico: Máximo contraste en tejido blando
            
            ### Mantenimiento y Control de Calidad
            
            **Verificaciones periódicas**:
            1. ✅ Alineación rejilla-detector
            2. ✅ Integridad física (sin dobleces)
            3. ✅ Uniformidad de transmisión
            4. ✅ Test de cut-off con desalineación intencional
            5. ✅ Factor Bucky experimental
            
            **Vida útil**: 
            - Rejilla fija: 10+ años
            - Bucky móvil: Mantenimiento motor cada 2-3 años
            
            ### Conclusión Práctica
            
            La rejilla es un **compromiso**:
            - ✅ Ganas: Contraste, calidad diagnóstica
            - ❌ Pierdes: Dosis aumentada, complejidad técnica
            
            **Regla de oro**: Usa rejilla cuando la dispersión degrada la imagen más 
            de lo que el aumento de dosis perjudica al paciente.
            
            Para partes gruesas (>12 cm) y áreas grandes, ¡la rejilla es esencial!
            """)
    
    # ============================================
    # SECTION 4: CONTRAST AND QUALITY
    # ============================================
    elif section == "📈 Contraste y Calidad":
        st.subheader("📈 Contraste Radiográfico y Calidad de Imagen")
        
        st.markdown("""
        El contraste es la diferencia visible entre estructuras anatómicas. 
        Explora cómo los parámetros técnicos afectan el contraste y la calidad de imagen.
        """)
        
        # Parameter controls
        contrast_col1, contrast_col2, contrast_col3 = st.columns(3)
        
        with contrast_col1:
            st.markdown("##### Técnica")
            contrast_kVp = st.slider("kVp", 40, 150, 70, 1, key="contrast_kvp")
            contrast_mAs = st.slider("mAs", 1, 100, 20, 1, key="contrast_mas")
            
        with contrast_col2:
            st.markdown("##### Objeto")
            object_type_1 = st.selectbox("Tejido 1", list(TISSUES.keys()), index=2, key="obj1")
            object_thickness_1 = st.slider("Espesor 1 (cm)", 0.5, 10.0, 5.0, 0.5, key="thick1")
            
        with contrast_col3:
            st.markdown("##### Comparación")
            object_type_2 = st.selectbox("Tejido 2", list(TISSUES.keys()), index=5, key="obj2")
            object_thickness_2 = st.slider("Espesor 2 (cm)", 0.5, 10.0, 2.0, 0.5, key="thick2")
        
        # Calculate transmissions
        eff_energy_contrast = contrast_kVp * 0.4
        
        # Tissue 1
        tissue1_props = TISSUES[object_type_1]
        mu1 = total_attenuation_coefficient(eff_energy_contrast, tissue1_props["Z_eff"], tissue1_props["density"])
        trans1 = calculate_transmission(object_thickness_1, mu1)
        intensity1 = 100 * trans1
        
        # Tissue 2
        tissue2_props = TISSUES[object_type_2]
        mu2 = total_attenuation_coefficient(eff_energy_contrast, tissue2_props["Z_eff"], tissue2_props["density"])
        trans2 = calculate_transmission(object_thickness_2, mu2)
        intensity2 = 100 * trans2
        
        # Calculate contrast
        contrast_value = calculate_contrast(intensity1, intensity2)
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Análisis de Contraste")
        
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric(
                f"Intensidad - {object_type_1}",
                f"{intensity1:.1f}%",
                help=f"Radiación transmitida a través de {object_thickness_1} cm de {object_type_1}"
            )
            
        with result_col2:
            st.metric(
                f"Intensidad - {object_type_2}",
                f"{intensity2:.1f}%",
                help=f"Radiación transmitida a través de {object_thickness_2} cm de {object_type_2}"
            )
            
        with result_col3:
            # Contrast quality indicator
            if contrast_value > 0.3:
                contrast_quality = "🟢 Excelente"
                contrast_color = "success"
            elif contrast_value > 0.15:
                contrast_quality = "🟡 Bueno"
                contrast_color = "info"
            elif contrast_value > 0.05:
                contrast_quality = "🟠 Moderado"
                contrast_color = "warning"
            else:
                contrast_quality = "🔴 Pobre"
                contrast_color = "error"
            
            st.metric(
                "Contraste",
                f"{contrast_value:.3f}",
                help="Contraste = |I1-I2|/(I1+I2)"
            )
            
            if contrast_color == "success":
                st.success(contrast_quality)
            elif contrast_color == "info":
                st.info(contrast_quality)
            elif contrast_color == "warning":
                st.warning(contrast_quality)
            else:
                st.error(contrast_quality)
        
        # Visual comparison
        st.markdown("### 🎨 Visualización de Contraste")
        
        fig_contrast_vis = go.Figure()
        
        # Create two rectangles representing the image intensity
        fig_contrast_vis.add_trace(go.Bar(
            x=[object_type_1, object_type_2],
            y=[intensity1, intensity2],
            marker=dict(
                color=[intensity1, intensity2],
                colorscale='Greys',
                showscale=False,
                line=dict(color='black', width=2)
            ),
            text=[f"{intensity1:.1f}%", f"{intensity2:.1f}%"],
            textposition='auto',
            showlegend=False
        ))
        
        fig_contrast_vis.update_layout(
            title="Intensidad en el Detector (Escala de Grises)",
            xaxis_title="Tejido",
            yaxis_title="Intensidad Relativa (%)",
            yaxis=dict(range=[0, 100]),
            height=400
        )
        
        st.plotly_chart(fig_contrast_vis, use_container_width=True)
        
        # Effect of kVp on contrast
        st.markdown("---")
        st.markdown("### 📉 Efecto del kVp en el Contraste")
        
        # Calculate contrast at different kVps
        kVp_range = np.arange(40, 151, 5)
        contrasts_at_kVps = []
        
        for kvp in kVp_range:
            eff_e = kvp * 0.4
            mu1_temp = total_attenuation_coefficient(eff_e, tissue1_props["Z_eff"], tissue1_props["density"])
            mu2_temp = total_attenuation_coefficient(eff_e, tissue2_props["Z_eff"], tissue2_props["density"])
            
            i1_temp = 100 * calculate_transmission(object_thickness_1, mu1_temp)
            i2_temp = 100 * calculate_transmission(object_thickness_2, mu2_temp)
            
            contrast_temp = calculate_contrast(i1_temp, i2_temp)
            contrasts_at_kVps.append(contrast_temp)
        
        fig_contrast_kvp = go.Figure()
        
        fig_contrast_kvp.add_trace(go.Scatter(
            x=kVp_range,
            y=contrasts_at_kVps,
            mode='lines',
            name='Contraste',
            line=dict(color='purple', width=3)
        ))
        
        # Mark current kVp
        fig_contrast_kvp.add_vline(
            x=contrast_kVp,
            line_dash="dash",
            line_color="red",
            annotation_text=f"kVp actual: {contrast_kVp}",
            annotation_position="top"
        )
        
        fig_contrast_kvp.update_layout(
            title=f"Contraste entre {object_type_1} y {object_type_2} vs kVp",
            xaxis_title="kVp",
            yaxis_title="Contraste",
            hovermode='x',
            height=400
        )
        
        st.plotly_chart(fig_contrast_kvp, use_container_width=True)
        
        st.info("""
        📉 **Observación clave**: El contraste **disminuye** al aumentar kVp
        
        **Por qué**: 
        - Mayor energía → Más penetración → Menos diferencia en atenuación
        - Efecto fotoeléctrico (dependiente de Z) disminuye
        - Efecto Compton (independiente de Z) domina
        
        **Implicación práctica**: 
        - Bajo kVp = Alto contraste (pero solo en partes delgadas)
        - Alto kVp = Bajo contraste (pero mejor penetración)
        """)
        
        # Latitude (exposure latitude)
        st.markdown("---")
        st.markdown("### 📏 Latitud de Exposición")
        
        st.markdown("""
        La **latitud** es el rango de exposiciones que produce una imagen diagnóstica aceptable.
        """)
        
        lat_col1, lat_col2 = st.columns(2)
        
        with lat_col1:
            st.markdown("#### Alto kVp (Técnica Larga)")
            st.success("""
            **Ventajas**:
            - ✅ Mayor latitud (más "perdona" errores)
            - ✅ Menor dosis al paciente
            - ✅ Menos sensible a variaciones de espesor
            - ✅ Mejor para partes gruesas
            
            **Desventajas**:
            - ❌ Menor contraste
            - ❌ Más dispersión
            """)
            
        with lat_col2:
            st.markdown("#### Bajo kVp (Técnica Corta)")
            st.warning("""
            **Ventajas**:
            - ✅ Mayor contraste
            - ✅ Mejor detalle tejidos blandos
            - ✅ Menos dispersión
            
            **Desventajas**:
            - ❌ Menor latitud (menos margen error)
            - ❌ Mayor dosis al paciente
            - ❌ Solo aplicable en partes delgadas
            - ❌ Más repeticiones por error técnico
            """)
        
        # SNR simulation
        st.markdown("---")
        st.markdown("### 📊 Relación Señal-Ruido (SNR)")
        
        st.markdown("""
        El ruido cuántico depende del número de fotones detectados.
        """)
        
        # Calculate SNR (simplified)
        # SNR proportional to sqrt(number of photons)
        # Number of photons proportional to mAs
        snr_value = np.sqrt(contrast_mAs * intensity1) / 10  # Normalized
        
        snr_col1, snr_col2 = st.columns([1, 2])
        
        with snr_col1:
            st.metric(
                "SNR Estimado",
                f"{snr_value:.2f}",
                help="Mayor SNR = Menos ruido, mejor calidad"
            )
            
            if snr_value > 5:
                st.success("🟢 SNR Excelente")
            elif snr_value > 3:
                st.info("🟡 SNR Adecuado")
            else:
                st.warning("🔴 SNR Bajo (imagen ruidosa)")
        
        with snr_col2:
            st.markdown("""
            **Para mejorar SNR**:
            - ↑ mAs (más fotones)
            - ↑ kVp (más fotones llegan al detector)
            - Filtrado/procesado digital
            
            **Trade-off**: 
            - Más mAs = Más dosis
            - Más kVp = Menos contraste
            """)
        
        # Optimization tool
        st.markdown("---")
        st.markdown("### 🎯 Optimización Técnica")
        
        st.markdown("Encuentra el balance óptimo entre contraste, dosis y calidad")
        
        optimize_col1, optimize_col2 = st.columns(2)
        
        with optimize_col1:
            priority = st.radio(
                "Prioridad",
                ["Máximo Contraste", "Mínima Dosis", "Balance"],
                index=2
            )
            
        with optimize_col2:
            part_thickness = st.slider("Espesor de la parte (cm)", 5, 40, 20)
        
        # Optimization recommendations
        if priority == "Máximo Contraste":
            if part_thickness < 12:
                rec_kVp = 55
                rec_mAs = 10
                explanation = "Parte delgada: Bajo kVp factible para máximo contraste"
            else:
                rec_kVp = 70
                rec_mAs = 30
                explanation = "Parte gruesa: kVp mínimo necesario para penetración"
        
        elif priority == "Mínima Dosis":
            if part_thickness < 12:
                rec_kVp = 70
                rec_mAs = 5
                explanation = "Alto kVp, bajo mAs: Técnica de mínima dosis"
            else:
                rec_kVp = 90
                rec_mAs = 10
                explanation = "Alto kVp permite reducir mAs significativamente"
        
        else:  # Balance
            if part_thickness < 12:
                rec_kVp = 60
                rec_mAs = 8
                explanation = "Balance óptimo para parte delgada"
            elif part_thickness < 25:
                rec_kVp = 75
                rec_mAs = 15
                explanation = "Balance estándar: contraste adecuado y dosis razonable"
            else:
                rec_kVp = 85
                rec_mAs = 25
                explanation = "Parte gruesa: kVp suficiente, mAs compensatorio"
        
        st.success(f"""
        ### 💡 Recomendación Optimizada
        
        **kVp**: {rec_kVp}
        **mAs**: {rec_mAs}
        
        **Explicación**: {explanation}
        """)
        
        # Compare current vs optimized
        compare_col1, compare_col2 = st.columns(2)
        
        with compare_col1:
            st.markdown("**Tu Técnica Actual**")
            current_dose_index = contrast_kVp * contrast_mAs / 100
            st.write(f"- kVp: {contrast_kVp}")
            st.write(f"- mAs: {contrast_mAs}")
            st.write(f"- Índice dosis: {current_dose_index:.1f}")
            
        with compare_col2:
            st.markdown("**Técnica Optimizada**")
            optimized_dose_index = rec_kVp * rec_mAs / 100
            st.write(f"- kVp: {rec_kVp}")
            st.write(f"- mAs: {rec_mAs}")
            st.write(f"- Índice dosis: {optimized_dose_index:.1f}")
            
            dose_reduction = ((current_dose_index - optimized_dose_index) / current_dose_index * 100) if current_dose_index > 0 else 0
            if dose_reduction > 0:
                st.success(f"🎉 Reducción de dosis: {dose_reduction:.0f}%")
            elif dose_reduction < -10:
                st.warning(f"⚠️ Aumento de dosis: {abs(dose_reduction):.0f}%")
        
        # Theory expander
        with st.expander("📚 Teoría: Contraste y Calidad de Imagen"):
            st.markdown("""
            ## 📈 Contraste Radiográfico: Fundamentos
            
            ### Definición de Contraste
            
            El contraste es la diferencia en la intensidad de radiación entre dos áreas adyacentes de la imagen.
            """)
            
            st.latex(r"C = \frac{|I_1 - I_2|}{I_1 + I_2}")
            
            st.markdown("""
            Donde:
            - **I₁, I₂**: Intensidades de dos regiones
            - **C**: Contraste (0 a 1, donde 1 = máximo contraste)
            
            ### Tipos de Contraste
            
            #### 1. Contraste de Sujeto (Subject Contrast)
            
            Depende de las **diferencias anatómicas** inherentes al paciente:
            
            **Factores**:
            - **Número atómico efectivo (Z)**: Hueso (↑Z) vs tejido blando (↓Z)
            - **Densidad física (ρ)**: Hueso (↑ρ) vs pulmón (↓ρ)
            - **Espesor**: Estructuras gruesas vs delgadas
            
            **No podemos modificarlo** (es anatomía del paciente), pero podemos **optimizarlo** con técnica.
            
            #### 2. Contraste Radiográfico (Radiographic Contrast)
            
            El contraste en el **haz de rayos X emergente** del paciente.
            
            **Factores que lo afectan**:
            - **kVp**: Factor más importante
            - **Espesor del paciente**
            - **Dispersión**: La reduce significativamente
            - **Filtración del haz**
            
            #### 3. Contraste de la Imagen (Image Contrast)
            
            El contraste **visible en la imagen final**.
            
            **Factores adicionales**:
            - Características del detector
            - Procesado digital
            - Ventanas (W/L) en imagen digital
            - Calidad del monitor
            
            ### Factores que Afectan el Contraste
            
            #### 📉 kVp: El Factor Más Crítico
            
            **Relación inversa**: ↑ kVp → ↓ Contraste
            
            **Mecanismo físico**:
            
            A **bajo kVp** (50-70 keV):
            - Domina efecto **fotoeléctrico** (∝ Z³)
            - Gran diferencia entre tejidos de diferente Z
            - **Máximo contraste**
            - Pero: Poca penetración, alta dosis
            
            A **alto kVp** (>90 keV):
            - Domina dispersión **Compton** (independiente de Z)
            - Poca diferencia entre tejidos
            - **Bajo contraste** (todo se ve gris)
            - Pero: Buena penetración, baja dosis
            
            **Ejemplo práctico**:
            - **Mamografía**: 25-30 kVp (máximo contraste en tejido blando)
            - **Extremidades**: 50-60 kVp (alto contraste óseo)
            - **Tórax**: 110-125 kVp (penetrar mediastino, sacrifica contraste)
            - **Abdomen**: 70-80 kVp (balance)
            
            #### 🌫️ Dispersión: El Enemigo
            
            La radiación dispersa añade una "niebla" uniforme que **reduce el contraste**.
            
            **Efecto cuantitativo**:
            """)
            
            st.latex(r"C_{\text{real}} = \frac{C_{\text{primaria}}}{1 + SPR}")
            
            st.markdown("""
            Donde **SPR** = Scatter-to-Primary Ratio (ratio dispersión/primaria)
            
            **Ejemplos**:
            - Extremidad: SPR = 0.3 → Contraste reducido 23%
            - Abdomen sin rejilla: SPR = 5 → Contraste reducido 83% (!!)
            - Abdomen con rejilla 10:1: SPR = 0.5 → Contraste reducido 33%
            
            **Por eso las rejillas son esenciales en partes gruesas.**
            
            #### 📏 Espesor del Paciente
            
            Mayor espesor → Más material → Más atenuación → Menos contraste
            
            Además: Mayor espesor → Más dispersión generada
            
            **Compensación**:
            - Partes delgadas: Bajo kVp, alto contraste
            - Partes gruesas: Alto kVp necesario (sacrifica contraste)
            
            ### Calidad de Imagen: Parámetros Objetivos
            
            #### 1. Resolución Espacial
            
            Capacidad de distinguir objetos pequeños cercanos.
            
            **Medida**: Pares de líneas por milímetro (pl/mm)
            
            **Factores limitantes**:
            - Tamaño del foco (principal)
            - Píxel del detector
            - Movimiento del paciente
            - Penumbra geométrica
            
            **Valores típicos**:
            - Radiología digital: 2.5-5 pl/mm
            - Mamografía digital: 8-12 pl/mm
            - Radiología analógica (película): 10+ pl/mm
            
            #### 2. Resolución de Contraste
            
            Capacidad de distinguir diferencias pequeñas de densidad.
            
            **Factores**:
            - Ruido de la imagen
            - Contraste de sujeto
            - Dispersión
            - Procesado digital
            
            **Digital vs Analógica**:
            - Digital: Mejor resolución de contraste (mayor rango dinámico)
            - Analógica: Mejor resolución espacial
            
            #### 3. Ruido Cuántico
            
            Variación aleatoria en el número de fotones detectados.
            
            **Naturaleza**: Estadística de Poisson
            """)
            
            st.latex(r"\sigma = \sqrt{N}")
            
            st.markdown("""
            Donde N = número de fotones
            
            **Relación Señal-Ruido**:
            """)
            
            st.latex(r"SNR = \frac{S}{\sigma} = \frac{N}{\sqrt{N}} = \sqrt{N}")
            
            st.markdown("""
            **Conclusión**: SNR ∝ √(mAs)
            
            Para **duplicar el SNR** (reducir ruido a la mitad), necesitas **cuadruplicar el mAs**.
            
            **Implicación práctica**:
            - Imagen ruidosa → ↑ mAs (pero ↑ dosis)
            - Imagen muy ruidosa → Verificar detector, no solo ↑ mAs
            
            #### 4. Nitidez (Sharpness)
            
            Definición clara de bordes y estructuras.
            
            **Factores que reducen nitidez**:
            - Movimiento (paciente, órganos)
            - Penumbra geométrica (foco grande)
            - Dispersión no eliminada
            - Píxel grande del detector
            
            **Mejoras**:
            - ✅ Foco pequeño
            - ✅ Tiempo de exposición corto
            - ✅ Inmovilización adecuada
            - ✅ Distancia foco-detector grande
            - ✅ Objeto pegado al detector
            
            ### Trade-offs en Radiología
            
            En radiología **todo es un compromiso**:
            
            #### Contraste vs Dosis
            
            | Objetivo | kVp | mAs | Resultado |
            |----------|-----|-----|-----------|
            | **Máximo contraste** | ↓↓ | ↑↑ | Alta dosis, aplicable solo en partes delgadas |
            | **Mínima dosis** | ↑↑ | ↓↓ | Bajo contraste, compensar con procesado |
            | **Balance** | Medio | Medio | Compromiso razonable |
            
            #### Contraste vs Penetración
            
            - Bajo kVp → Máximo contraste pero mala penetración
            - Alto kVp → Buena penetración pero bajo contraste
            - **Solución**: kVp óptimo según anatomía
            
            #### SNR vs Dosis
            
            - Más mAs → Mejor SNR (menos ruido)
            - Más mAs → Más dosis al paciente
            - **Solución**: mAs mínimo compatible con calidad diagnóstica
            
            #### Resolución vs Capacidad de Carga
            
            - Foco fino → Mejor resolución
            - Foco fino → Baja capacidad térmica (mAs limitado)
            - **Solución**: Foco fino solo para técnicas de bajo mAs
            
            ### Optimización Práctica
            
            #### Paso 1: Determinar kVp
            
            **Basado en anatomía**:
            
            | Anatomía | Espesor típico | kVp recomendado | Razón |
            |----------|---------------|-----------------|-------|
            | **Dedos/mano** | 2-5 cm | 50-55 | Detalle óseo, máximo contraste |
            | **Muñeca/tobillo** | 5-8 cm | 55-60 | Balance contraste-penetración |
            | **Rodilla** | 10-12 cm | 65-70 | Penetración suficiente |
            | **Hombro/pelvis** | 15-20 cm | 70-80 | Partes densas |
            | **Abdomen** | 20-30 cm | 70-80 | Contraste tejido blando |
            | **Tórax PA** | 20-25 cm | 110-125 | Penetrar mediastino |
            | **Columna lumbar LAT** | 30-40 cm | 90-100 | Máxima penetración |
            
            #### Paso 2: Calcular mAs
            
            **Fórmula empírica** (punto de partida):
            """)
            
            st.latex(r"mAs = k \times \text{Espesor}^2")
            
            st.markdown("""
            Donde k = constante según anatomía (determinar por experiencia/tablas)
            
            **Ajustar por**:
            - Morfología del paciente (obeso → ↑ mAs)
            - Uso de rejilla (con rejilla → ×Bucky factor)
            - Distancia (si ≠100cm → ley inversa del cuadrado)
            - Detector (algunos requieren más/menos)
            
            #### Paso 3: Verificar y Ajustar
            
            **En imagen digital**:
            - Verificar índice de exposición (EI/DI)
            - Objetivo: Dentro del rango óptimo
            - Si fuera de rango → Ajustar técnica
            
            **Regla de oro ALARA**:
            - Usar **mínimo mAs** que produzca calidad diagnóstica
            - No sobreexponer "por si acaso"
            - En digital, sobreexposición no se ve (¡pero la dosis sí!)
            
            ### Índices de Exposición en Digital
            
            Diferentes fabricantes usan diferentes métricas:
            
            #### Exposure Index (EI) - IEC Standard
            
            Valor objetivo: **Depende del detector y fabricante**
            
            **Interpretación**:
            - EI correcto → Imagen óptima
            - EI bajo → Subexposición (ruido excesivo)
            - EI alto → Sobreexposición (dosis innecesaria)
            
            #### Deviation Index (DI)
            
            Desviación respecto al valor objetivo.
            """)
            
            st.latex(r"DI = 10 \times \log_{10}\left(\frac{EI}{EI_{target}}\right)")
            
            st.markdown("""
            **Interpretación**:
            - **DI = 0**: Perfecto (EI = target)
            - **DI = +1**: 25% sobreexposición
            - **DI = +3**: 2× sobreexposición
            - **DI = -1**: 20% subexposición
            - **DI = -3**: 50% subexposición
            
            **Rango aceptable**: DI entre -1 y +1
            
            #### Fabricantes Específicos
            
            **Agfa**: Log of Median (lgM)
            - Objetivo: ~2.5
            - Rango: 1.9-2.8
            
            **Carestream**: Exposure Index (EI)
            - Objetivo: ~2000
            - Rango: 1800-2200
            
            **Fuji**: S value
            - Objetivo: ~200
            - ⚠️ **Inverso**: Menor S = más exposición
            
            **Philips**: Exposure Index (EI)
            - Objetivo: ~400-600
            
            **Importante**: Consultar manual de tu equipo específico.
            
            ### Control de Calidad del Contraste
            
            #### Tests Periódicos
            
            **Test de penetrómetro (step wedge)**:
            - Objeto con escalones de diferentes espesores
            - Verificar que se distinguen todos los escalones
            - Mide rango dinámico y contraste
            
            **Test de bajo contraste**:
            - Phantom con objetos de diferente tamaño y contraste
            - Verificar detectabilidad mínima
            - Asegura capacidad de ver lesiones sutiles
            
            **Test de uniformidad**:
            - Exposición de campo uniforme
            - Verificar que no hay variaciones de densidad
            - Detecta problemas de calibración
            
            ### Artefactos que Afectan al Contraste
            
            #### 1. Velo por Dispersión (Fog)
            
            **Causa**: Dispersión no eliminada
            
            **Efecto**: Reduce contraste globalmente
            
            **Solución**: 
            - Rejilla adecuada
            - Colimación estricta
            - Evitar objetos dispersores cerca del detector
            
            #### 2. Artefactos de Procesado
            
            **Causa**: Algoritmos de mejora agresivos
            
            **Efecto**: 
            - Halo alrededor de estructuras densas
            - "Edge enhancement" excesivo
            - Contraste artificial
            
            **Solución**: Ajustar parámetros de procesado
            
            #### 3. Saturación del Detector
            
            **Causa**: Sobreexposición extrema en zonas
            
            **Efecto**: Pérdida de información (área blanca sin detalle)
            
            **Solución**: 
            - Compensar técnica
            - Usar filtros compensadores
            
            ### Casos Especiales
            
            #### Pacientes Pediátricos
            
            **Consideraciones**:
            - Menor espesor → Menos mAs
            - Mayor contraste natural (menos grasa)
            - Prioridad absoluta: **Mínima dosis**
            
            **Técnica**:
            - kVp ligeramente menor (mejor contraste)
            - mAs mínimo (↑ ruido aceptable vs dosis)
            - Tiempo mínimo (evitar movimiento)
            
            #### Pacientes Obesos
            
            **Problemas**:
            - Gran espesor → Necesita penetración
            - Mucha dispersión → Reduce contraste
            - Mayor dosis inevitable
            
            **Técnica**:
            - ↑ kVp significativamente (90-100+)
            - ↑ mAs proporcionalmente
            - Rejilla obligatoria (ratio alto: 12:1-16:1)
            - Considerar proyecciones alternativas
            
            #### Estudios con Contraste Artificial
            
            **Bario (Ba, Z=56)** o **Yodo (I, Z=53)**:
            - Alto Z → Máxima atenuación
            - Excelente contraste natural
            - Permite kVp más bajo
            
            **Optimización**:
            - kVp justo por encima de K-edge del contraste
            - Ba: K-edge = 37 keV → Usar ~70-80 kVp
            - I: K-edge = 33 keV → Usar ~60-70 kVp
            - Maximiza absorción fotoeléctrica del contraste
            
            ### Herramientas Digitales de Mejora
            
            #### Post-procesado
            
            **Ventanas (Windowing)**:
            - W/L = Window/Level
            - Ajusta contraste y brillo sin reexposición
            - Permite "recuperar" imágenes de bajo contraste
            
            **Ecualización de histograma**:
            - Redistribuye niveles de gris
            - Mejora visualización de regiones específicas
            
            **Filtros de realce de bordes**:
            - Mejora percepción de estructuras pequeñas
            - Complementa (no sustituye) técnica correcta
            
            #### Limitaciones del Post-procesado
            
            **No puede**:
            - Eliminar ruido cuántico (información no está)
            - Recuperar estructuras saturadas
            - Compensar movimiento
            - Crear información que no fue captada
            
            **Puede**:
            - Optimizar visualización de información existente
            - Ajustar contraste y brillo
            - Reducir artefactos menores
            - Mejorar percepción visual
            
            ### Conclusión: El Arte del Balance
            
            La radiología diagnóstica es encontrar el **punto óptimo** entre:
            
            1. ✅ **Calidad diagnóstica suficiente** (no perfecta, suficiente)
            2. ✅ **Dosis mínima razonable** (ALARA)
            3. ✅ **Eficiencia clínica** (no repeticiones)
            4. ✅ **Comodidad del paciente** (tiempo, posicionamiento)
            
            **No existe "la técnica perfecta"** - existe la técnica **apropiada para cada situación**.
            
            Tu trabajo como TSID es **dominar estos principios** para tomar decisiones 
            informadas caso por caso, siempre priorizando:
            
            🎯 **Calidad diagnóstica + Protección radiológica**
            """)
    
    # Final section summary
    st.markdown("---")
    st.success("""
    ### 🎯 Puntos Clave - Formación de Imagen
    
    1. **Tres interacciones**: Fotoeléctrico (contraste), Compton (dispersión), Coherente (menor)
    2. **Beer-Lambert**: I = I₀ × e^(-μx) - Atenuación exponencial
    3. **Contraste**: Depende de ΔZ, Δρ, espesor, y kVp
    4. **↑ kVp → ↓ Contraste** pero ↑ penetración y ↓ dosis
    5. **Dispersión**: Principal enemigo del contraste (niebla)
    6. **Rejillas**: Eliminan dispersión pero ↑ dosis (Factor Bucky)
    7. **Ratio rejilla**: Mayor ratio = más selectiva pero más dosis
    8. **SNR ∝ √mAs**: Duplicar SNR requiere 4× mAs
    9. **Optimización**: Balance entre contraste, dosis y calidad
    10. **ALARA siempre**: Mínima dosis compatible con calidad diagnóstica
    """)
    
    # Pro tips for this tab
    st.info("""
    ### 💡 Consejos Profesionales - Formación de Imagen
    
    **Para maximizar contraste**:
    - 🎯 Usa el kVp más bajo que permita la penetración
    - 🔍 Colima estrictamente (menos volumen = menos dispersión)
    - 🛡️ Usa rejilla en partes >10-12 cm
    - 📏 Comprime suavemente si es posible (reduce espesor)
    
    **Para minimizar dosis manteniendo calidad**:
    - ⚡ Aplica regla del 15% (↑kVp 15% = ½ mAs)
    - 📊 Verifica índices de exposición (EI/DI)
    - 🎯 No sobreexpongas "por si acaso"
    - 📱 Usa AEC cuando disponible
    
    **Para reducir dispersión**:
    - ✂️ Colimación al mínimo necesario
    - 📏 Usa rejilla apropiada (ratio según espesor)
    - 🔄 Considera air gap en lateral de C-spine
    - 🎯 Elimina objetos innecesarios del campo
    
    **Para mejorar calidad general**:
    - 👤 Posicionamiento correcto (primera vez)
    - ⏱️ Tiempo mínimo (evita movimiento)
    - 📍 Parte pegada al detector (↓ penumbra)
    - 🎚️ Usa foco fino si mAs lo permite
    """)
    
    # Footer for this tab
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>🎯 <strong>Tab 2: Formación de Imagen</strong> | 
        Simulador de Física Radiológica | 
        Formación Profesional en Imagen para el Diagnóstico</p>
        <p>Experimenta con diferentes tejidos y parámetros para entender cómo se forma la imagen radiográfica</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================
# TAB 3: PROTECCIÓN RADIOLÓGICA
# ============================================
with tabs[2]:
    st.header("🛡️ Protección Radiológica")
    st.markdown("""
    La protección radiológica es **fundamental** en tu práctica diaria. Aprende a aplicar 
    los principios ALARA, calcular dosis, y diseñar estrategias de protección efectivas.
    """)
    
    st.markdown("---")
    
    # Section selector
    protection_section = st.radio(
        "Selecciona el tema:",
        [
            "⏱️ Principios ALARA",
            "📊 Límites y Dosimetría",
            "🧱 Cálculo de Blindajes",
            "🏥 Diseño de Instalaciones",
            "📈 Niveles de Referencia (DRL)"
        ],
        horizontal=False
    )
    
    # ============================================
    # SECTION 1: ALARA PRINCIPLES
    # ============================================
    if protection_section == "⏱️ Principios ALARA":
        st.subheader("⏱️ Principios ALARA: As Low As Reasonably Achievable")
        
        st.info("""
        **ALARA** = **T**an **B**ajo **C**omo **R**azonablemente **P**osible
        
        Los tres pilares de la protección radiológica:
        1. ⏱️ **TIEMPO**: Minimizar tiempo de exposición
        2. 📏 **DISTANCIA**: Maximizar distancia a la fuente
        3. 🧱 **BLINDAJE**: Interponer material protector
        """)
        
        st.markdown("---")
        
        # Interactive ALARA demonstration
        st.markdown("### 🎯 Simulador Interactivo ALARA")
        
        # Input parameters
        alara_col1, alara_col2, alara_col3 = st.columns(3)
        
        with alara_col1:
            st.markdown("#### ⏱️ Factor 1: TIEMPO")
            exposure_time_s = st.slider(
                "Tiempo de exposición (segundos)",
                0.01, 10.0, 0.1, 0.01,
                help="Tiempo que el técnico está expuesto a radiación dispersa"
            )
            num_procedures = st.number_input(
                "Procedimientos por día",
                1, 50, 10,
                help="Número de exposiciones realizadas"
            )
            
        with alara_col2:
            st.markdown("#### 📏 Factor 2: DISTANCIA")
            distance_m = st.slider(
                "Distancia a la fuente (metros)",
                0.5, 5.0, 2.0, 0.1,
                help="Distancia entre el punto de dispersión y el técnico"
            )
            dose_rate_at_1m = st.number_input(
                "Tasa de dosis a 1m (µSv/h)",
                1.0, 1000.0, 100.0,
                help="Tasa de dosis dispersa a 1 metro del paciente"
            )
            
        with alara_col3:
            st.markdown("#### 🧱 Factor 3: BLINDAJE")
            use_shielding = st.checkbox("Usar protección", value=True)
            if use_shielding:
                shielding_type = st.selectbox(
                    "Tipo de protección",
                    ["Delantal 0.25mm Pb", "Delantal 0.5mm Pb", "Biombo 2mm Pb", "Mampara plomada"]
                )
                # Attenuation factors
                shielding_factors = {
                    "Delantal 0.25mm Pb": 0.10,  # 90% reduction
                    "Delantal 0.5mm Pb": 0.01,   # 99% reduction
                    "Biombo 2mm Pb": 0.001,      # 99.9% reduction
                    "Mampara plomada": 0.0001    # 99.99% reduction
                }
                shielding_transmission = shielding_factors[shielding_type]
            else:
                shielding_transmission = 1.0
                shielding_type = "Ninguna"
        
        # Calculate doses
        # Time factor
        total_time_h = (exposure_time_s * num_procedures) / 3600
        
        # Distance factor (inverse square law)
        dose_rate_at_distance = calculate_dose_at_distance(dose_rate_at_1m, 1.0, distance_m)
        
        # Combined dose
        dose_per_day_no_shield = dose_rate_at_distance * total_time_h
        dose_per_day_with_shield = dose_per_day_no_shield * shielding_transmission
        
        # Annual projection (250 working days)
        dose_per_year_no_shield = dose_per_day_no_shield * 250
        dose_per_year_with_shield = dose_per_day_with_shield * 250
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Resultados de la Simulación")
        
        results_col1, results_col2, results_col3, results_col4 = st.columns(4)
        
        with results_col1:
            st.metric(
                "Dosis Diaria (sin protección)",
                f"{dose_per_day_no_shield:.2f} µSv",
                help="Dosis que recibirías sin ninguna protección"
            )
            
        with results_col2:
            st.metric(
                "Dosis Diaria (con protección)",
                f"{dose_per_day_with_shield:.3f} µSv",
                delta=f"-{(1-shielding_transmission)*100:.1f}%",
                delta_color="inverse",
                help="Dosis con las medidas de protección seleccionadas"
            )
            
        with results_col3:
            st.metric(
                "Proyección Anual (sin protección)",
                f"{dose_per_year_no_shield/1000:.2f} mSv",
                help="Extrapolación a 250 días laborables"
            )
            
        with results_col4:
            limit_percentage = (dose_per_year_with_shield/1000) / 20 * 100
            st.metric(
                "% del Límite Anual",
                f"{limit_percentage:.2f}%",
                help="Porcentaje del límite de 20 mSv/año para trabajadores"
            )
        
        # Visual comparison
        st.markdown("### 📈 Impacto de Cada Factor ALARA")
        
        # Calculate scenarios
        scenarios = {
            "Sin protección": dose_per_day_no_shield,
            "Solo Tiempo (50%)": dose_per_day_no_shield * 0.5,
            "Solo Distancia (×2)": calculate_dose_at_distance(dose_rate_at_1m, 1.0, distance_m*2) * total_time_h,
            "Solo Blindaje": dose_per_day_no_shield * shielding_transmission,
            "Combinado (actual)": dose_per_day_with_shield
        }
        
        fig_alara = go.Figure()
        
        colors_alara = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        
        fig_alara.add_trace(go.Bar(
            x=list(scenarios.keys()),
            y=list(scenarios.values()),
            marker=dict(color=colors_alara),
            text=[f"{v:.2f} µSv" for v in scenarios.values()],
            textposition='auto'
        ))
        
        fig_alara.update_layout(
            title="Comparación de Estrategias de Protección",
            yaxis_title="Dosis Diaria (µSv)",
            height=450,
            showlegend=False
        )
        
        st.plotly_chart(fig_alara, use_container_width=True)
        
        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Recomendaciones Personalizadas")
        
        if limit_percentage < 10:
            st.success(f"""
            ✅ **Excelente**: Tu dosis proyectada es solo el {limit_percentage:.1f}% del límite legal.
            
            Tus medidas de protección son muy efectivas. Continúa con estas buenas prácticas.
            """)
        elif limit_percentage < 50:
            st.info(f"""
            ℹ️ **Adecuado**: Tu dosis proyectada es el {limit_percentage:.1f}% del límite legal.
            
            Dentro de rangos aceptables, pero siempre busca optimizar siguiendo ALARA.
            """)
        else:
            st.warning(f"""
            ⚠️ **Atención**: Tu dosis proyectada es el {limit_percentage:.1f}% del límite legal.
            
            Considera mejorar tus medidas de protección:
            - Aumenta la distancia cuando sea posible
            - Verifica que usas blindaje adecuado
            - Minimiza tiempo de exposición
            - Consulta con tu supervisor de protección radiológica
            """)
        
        # Practical tips by modality
        st.markdown("---")
        st.markdown("### 🏥 Consejos Prácticos por Modalidad")
        
        modality_tips = st.selectbox(
            "Selecciona modalidad",
            ["Radiografía Convencional", "Radiografía Portátil", "Fluoroscopia", "TC", "Intervencionismo"]
        )
        
        tips_dict = {
            "Radiografía Convencional": """
            **⏱️ Tiempo**:
            - Sal de la sala durante la exposición
            - Si debes permanecer (pediatría, paciente no colaborador): mínimo tiempo necesario
            - Nunca sujetes al paciente durante la exposición
            
            **📏 Distancia**:
            - Mínimo 2 metros del tubo y paciente
            - Usa el biombo plomado de la sala de control
            - En sala: sitúate lo más lejos posible (esquina opuesta)
            
            **🧱 Blindaje**:
            - Biombo plomado obligatorio
            - Si estás en sala: delantal 0.5mm Pb eq mínimo
            - Protección tiroidea si exposiciones frecuentes
            - Gafas plomadas para proteger cristalino (nueva normativa)
            """,
            
            "Radiografía Portátil": """
            **⏱️ Tiempo**:
            - Minimiza número de exposiciones (técnica correcta primera vez)
            - No permanezcas en habitación más tiempo del necesario
            
            **📏 Distancia**:
            - ⚠️ **CRÍTICO**: Mínimo 2 metros del tubo (idealmente 3m)
            - NUNCA sujetes el chasis/detector durante exposición
            - Sal de habitación si es posible
            - Distancia es tu MEJOR protección en portátiles
            
            **🧱 Blindaje**:
            - Delantal plomado 0.5mm Pb eq OBLIGATORIO
            - Protección tiroidea obligatoria
            - Si hay otros pacientes: biombo portátil entre ellos y el equipo
            - Advertir a personal de la habitación
            
            **⚠️ ESPECIAL ATENCIÓN**:
            - Radiografía portátil = Mayor exposición ocupacional
            - Dispersión en todas direcciones (sin blindaje de sala)
            - Cumplir estrictamente protocolos de distancia
            """,
            
            "Fluoroscopia": """
            **⏱️ Tiempo**:
            - Modo pulsado en lugar de continuo (reduce dosis 50-90%)
            - Última imagen guardada (LIH) en lugar de fluoro continua
            - Mínimo tiempo de fluoro necesario
            - Monitorizar tiempo acumulado
            
            **📏 Distancia**:
            - Máxima distancia compatible con el procedimiento
            - No acercar cara al campo (dispersión máxima cerca del paciente)
            - Bajo mesa mejor que sobre mesa (menos dispersión)
            
            **🧱 Blindaje**:
            - Delantal 0.5mm Pb eq obligatorio (considerar 0.25 adicional frontal)
            - Protección tiroidea obligatoria
            - Gafas plomadas OBLIGATORIAS (protección cristalino - límite reducido)
            - Faldones plomados de la mesa (uso correcto)
            - Mamparas suspendidas
            
            **⚠️ ALTO RIESGO**:
            - Fluoroscopia = Mayor exposición ocupacional en radiología
            - Dosímetro adicional sobre delantal recomendado
            - Control dosimétrico estricto
            """,
            
            "TC": """
            **⏱️ Tiempo**:
            - Estar en sala solo durante posicionamiento (sin emisión RX)
            - Salir ANTES de iniciar escaneo
            - Monitorización desde sala de control
            
            **📏 Distancia**:
            - Sala de control con biombo plomado
            - Si debes entrar durante escaneo (emergencia): rápido y con protección
            
            **🧱 Blindaje**:
            - Biombo de sala de control (vidrio plomado)
            - Delantal si excepcionalmente debes estar en sala
            - Verificar indicadores de emisión (luces de aviso)
            
            **✅ BAJO RIESGO**:
            - TC bien protegido (túnel colimado, sala blindada)
            - Exposición ocupacional mínima si sigues protocolo
            - Dispersión muy baja fuera del gantry
            """,
            
            "Intervencionismo": """
            **⏱️ Tiempo**:
            - Procedimientos largos: rotación de personal si posible
            - Modo pulsado (fluoro) cuando sea factible
            - Minimizar tiempo con RX activo
            
            **📏 Distancia**:
            - Máxima distancia del tubo compatible con rol
            - Lado del detector si posible (menos dispersión)
            - Nunca directamente en línea del haz primario
            
            **🧱 Blindaje**:
            - Delantal doble capa (0.5mm frontal + 0.25mm posterior)
            - Protección tiroidea obligatoria
            - Gafas plomadas obligatorias
            - Guantes plomados si manos cerca del campo
            - Mamparas móviles posicionadas correctamente
            - Faldones bajo mesa
            
            **⚠️ MÁXIMO RIESGO**:
            - Intervencionismo = Más alta exposición ocupacional
            - Dosímetro de anillo (extremidades)
            - Dosímetro adicional sobre delantal
            - Control dosimétrico mensual recomendado
            - Formación específica obligatoria
            """
        }
        
        st.info(tips_dict[modality_tips])
        
        # ALARA checklist
        st.markdown("---")
        st.markdown("### ✅ Checklist ALARA Pre-Exposición")
        
        check_col1, check_col2 = st.columns(2)
        
        with check_col1:
            st.markdown("""
            **Antes de cada exposición verifica**:
            
            ☑️ ¿La exploración está justificada?
            
            ☑️ ¿He optimizado parámetros técnicos? (kVp/mAs)
            
            ☑️ ¿He colimado al mínimo necesario?
            
            ☑️ ¿He usado protecciones al paciente? (gonadal si aplica)
            
            ☑️ ¿Estoy a distancia segura? (≥2m)
            
            ☑️ ¿Llevo dosímetro personal?
            
            ☑️ ¿Uso protección si debo estar en sala?
            """)
            
        with check_col2:
            st.markdown("""
            **Para el paciente**:
            
            ☑️ ¿He explicado el procedimiento?
            
            ☑️ ¿He verificado posible embarazo? (mujeres 10-50 años)
            
            ☑️ ¿He registrado datos para trazabilidad?
            
            ☑️ ¿Posicionamiento correcto primera vez?
            
            ☑️ ¿Inmovilización adecuada? (evitar repetición)
            
            ☑️ ¿He retirado objetos radiopacos innecesarios?
            
            ☑️ ¿He informado de resultados/seguimiento?
            """)
        
        # Theory expander
        with st.expander("📚 Teoría: Fundamentos de Protección Radiológica"):
            st.markdown("""
            ## 🛡️ Bases Científicas de la Protección Radiológica
            
            ### Efectos Biológicos de las Radiaciones Ionizantes
            
            #### Mecanismos de Daño
            
            **Ionización directa**:
            - RX ioniza moléculas biológicas (especialmente ADN)
            - Ruptura de enlaces químicos
            - Daño directo a estructuras celulares
            
            **Ionización indirecta** (70% del daño):
            - RX ioniza agua → Radicales libres (OH·, H·)
            - Radicales atacan moléculas biológicas
            - Daño oxidativo
            
            #### Tipos de Efectos
            
            **1. Efectos Deterministas (antes "no estocásticos")**:
            
            Características:
            - **Umbral de dosis**: No ocurren por debajo de cierta dosis
            - **Severidad proporcional a dosis**: Mayor dosis → Mayor efecto
            - **Predecibles**: Ocurren en todos los expuestos por encima del umbral
            - **Corto plazo**: Días a semanas
            
            Ejemplos:
            - Eritema cutáneo: Umbral ~2 Gy
            - Depilación temporal: ~3 Gy
            - Síndrome agudo radiación: >1 Gy cuerpo entero
            - Cataratas: >0.5 Gy (acumulado)
            - Esterilidad temporal: 0.15 Gy
            
            **En diagnóstico**: Dosis muy por debajo de umbrales (excepto intervencionismo prolongado)
            
            **2. Efectos Estocásticos** (probabilísticos):
            
            Características:
            - **Sin umbral**: Cualquier dosis implica riesgo (por pequeña que sea)
            - **Probabilidad proporcional a dosis**: Mayor dosis → Mayor probabilidad
            - **Severidad independiente de dosis**: Si ocurre, gravedad no depende de dosis
            - **Largo plazo**: Años a décadas
            - **No predecibles individualmente**: Cuestión de probabilidad
            
            Ejemplos:
            - **Cáncer inducido por radiación**
            - **Efectos genéticos** (mutaciones heredables)
            
            **Modelo LNT** (Linear No-Threshold):
            - Asunción conservadora: No hay dosis segura
            - Relación lineal dosis-riesgo
            - Base de los límites de dosis actuales
            
            ### Principios de Protección Radiológica (ICRP 103)
            
            #### 1. Justificación
            
            **Definición**: Toda exposición debe estar justificada - beneficio > riesgo
            
            **En diagnóstico**:
            - Responsabilidad del **médico prescriptor**
            - Debe existir indicación clínica clara
            - Considerar alternativas sin radiación (US, MRI)
            - No exploraciones "rutinarias" o "de screening" sin justificación
            
            **El TSID debe**:
            - Verificar que existe prescripción médica
            - Confirmar identidad del paciente
            - Ante duda sobre justificación: consultar con radiólogo
            - NUNCA realizar estudio sin prescripción
            
            #### 2. Optimización (ALARA)
            
            **Definición**: Mantener dosis tan baja como razonablemente posible, 
            compatible con el objetivo diagnóstico.
            
            **Aplicación práctica**:
            
            **Para el paciente**:
            - Técnica óptima (no excesiva)
            - Colimación estricta
            - Protecciones (gonadal, tiroides si aplica)
            - Evitar repeticiones (técnica correcta primera vez)
            - Protocolos específicos (pediátricos, embarazo)
            
            **Para el trabajador**:
            - **Tiempo**: Mínimo necesario
            - **Distancia**: Máxima posible
            - **Blindaje**: Apropiado al riesgo
            
            **Para el público**:
            - Blindaje de instalaciones
            - Señalización adecuada
            - Control de accesos
            
            #### 3. Limitación de Dosis
            
            **Definición**: No superar límites establecidos legalmente
            
            **Aplicable a**:
            - Trabajadores expuestos
            - Aprendices y estudiantes
            - Público
            
            **NO aplicable a**:
            - Pacientes (justificación y optimización, sin límite absoluto)
            - Cuidadores/confortantes de pacientes (límites especiales)
            - Exposición médica voluntaria en investigación
            
            ### Magnitudes y Unidades Dosimétricas
            
            #### Dosis Absorbida (D)
            
            **Definición**: Energía absorbida por unidad de masa
            """)
            
            st.latex(r"D = \frac{dE}{dm}")
            
            st.markdown("""
            **Unidad**: Gray (Gy) = 1 J/kg
            
            **Antigua**: rad = 0.01 Gy
            
            **Características**:
            - Magnitud física objetiva
            - Medible directamente
            - No considera tipo de radiación
            - No considera radiosensibilidad del tejido
            
            #### Dosis Equivalente (H_T)
            
            **Definición**: Dosis absorbida ponderada por tipo de radiación
            """)
            
            st.latex(r"H_T = \sum_R w_R \times D_{T,R}")
            
            st.markdown("""
            Donde:
            - **w_R**: Factor de ponderación de la radiación
            - **D_T,R**: Dosis absorbida en tejido T por radiación R
            
            **Unidad**: Sievert (Sv) = 1 J/kg (misma dimensión que Gy, pero concepto diferente)
            
            **Antigua**: rem = 0.01 Sv
            
            **Factores w_R**:
            - Fotones (RX, γ): w_R = 1
            - Electrones, muones: w_R = 1
            - Neutrones: w_R = 2.5-20 (depende de energía)
            - Partículas α: w_R = 20
            
            **En radiodiagnóstico**: Solo fotones (RX) → w_R = 1 → **H_T = D** (numéricamente)
            
            #### Dosis Efectiva (E)
            
            **Definición**: Dosis equivalente ponderada por radiosensibilidad del tejido
            """)
            
            st.latex(r"E = \sum_T w_T \times H_T")
            
            st.markdown("""
            Donde:
            - **w_T**: Factor de ponderación del tejido
            - **H_T**: Dosis equivalente en tejido T
            
            **Unidad**: Sievert (Sv)
            
            **Factores w_T** (ICRP 103):
            - Médula ósea, colon, pulmón, estómago: 0.12 cada uno
            - Gónadas: 0.08
            - Vejiga, esófago, hígado, tiroides: 0.04 cada uno
            - Piel, superficie ósea: 0.01 cada uno
            - Resto: 0.12 (distribuido)
            - **Suma total: 1.0**
            
            **Utilidad**:
            - Comparar riesgo entre diferentes exposiciones
            - Sumar exposiciones de diferentes órganos
            - Aplicar límites de dosis
            - Estimación de riesgo de cáncer
            
            **Limitación**: 
            - No es medible directamente (se calcula)
            - Concepto de protección, no para diagnóstico individual
            
            ### Límites de Dosis (Legislación Española/UE)
            
            #### Trabajadores Expuestos (Categoría A)
            
            **Límite efectivo**:
            - **20 mSv/año** (promediado en 5 años)
            - **50 mSv en un solo año** (máximo)
            - **100 mSv en 5 años consecutivos**
            
            **Límites equivalentes (órganos)**:
            - **Cristalino**: 20 mSv/año (¡reducido desde 150!)
            - **Piel**: 500 mSv/año (promediado en 1 cm²)
            - **Manos, pies**: 500 mSv/año
            
            ⚠️ **Nueva normativa (2018)**: Límite de cristalino reducido drásticamente
            → Gafas plomadas obligatorias en fluoroscopia/intervencionismo
            
            #### Trabajadoras Embarazadas
            
            **Obligatorio**: Declarar embarazo a supervisor de protección radiológica
            
            **Límites desde declaración**:
            - **Superficie de abdomen**: 2 mSv durante resto de embarazo
            - **Feto**: 1 mSv durante embarazo
            
            **Medidas prácticas**:
            - Reasignación temporal de funciones
            - Evitar fluoroscopia, intervencionismo, portátiles
            - Dosímetro adicional a nivel de abdomen
            - Seguimiento dosimétrico mensual
            
            #### Aprendices y Estudiantes (16-18 años)
            
            **Límites reducidos**:
            - **6 mSv/año** (efectiva)
            - **Cristalino**: 20 mSv/año
            - **Piel y extremidades**: 150 mSv/año
            
            **Supervisión obligatoria** durante prácticas
            
            #### Público General
            
            **Límite efectivo**:
            - **1 mSv/año** (adicional al fondo natural y exposiciones médicas)
            
            **Límites equivalentes**:
            - **Cristalino**: 15 mSv/año
            - **Piel**: 50 mSv/año
            
            **Aplicación**:
            - Diseño de blindajes de instalaciones
            - Áreas controladas vs vigiladas
            - Acompañantes de pacientes (límites especiales)
            
            ### Ley Inversa del Cuadrado de la Distancia
            
            **Principio fundamental**: La intensidad disminuye con el cuadrado de la distancia
            """)
            
            st.latex(r"I(d) = \frac{I_0}{d^2}")
            
            st.markdown("""
            O, para calcular dosis a diferentes distancias:
            """)
            
            st.latex(r"D_2 = D_1 \times \left(\frac{d_1}{d_2}\right)^2")
            
            st.markdown("""
            **Ejemplo práctico**:
            - Dosis a 1m: 100 µSv/h
            - Dosis a 2m: 100 × (1/2)² = 25 µSv/h (**4 veces menos**)
            - Dosis a 3m: 100 × (1/3)² = 11 µSv/h (**9 veces menos**)
            
            **Conclusión crítica**: **Duplicar la distancia reduce dosis a ¼**
            
            → En radiografía portátil, pasar de 1m a 2m reduce tu dosis **75%**
            
            ### Atenuación por Blindaje
            
            **Ley exponencial**:
            """)
            
            st.latex(r"I = I_0 \times e^{-\mu x} \approx I_0 \times 0.5^{x/HVL}")
            
            st.markdown("""
            **Capa Hemirreductora (HVL)**:
            - Espesor que reduce intensidad a la mitad
            - Cada HVL adicional → reduce a la mitad otra vez
            
            **Ejemplo**:
            - 0 HVL: 100% (sin blindaje)
            - 1 HVL: 50%
            - 2 HVL: 25%
            - 3 HVL: 12.5%
            - 4 HVL: 6.25%
            - 5 HVL: 3.125%
            - 10 HVL: 0.1% (**factor 1000**)
            
            **HVL típicas** (plomo):
            - 60 kVp: 0.15 mm Pb
            - 80 kVp: 0.25 mm Pb
            - 100 kVp: 0.35 mm Pb
            - 150 kVp: 0.60 mm Pb
            
            **Delantal 0.5mm Pb** a 80 kVp:
            - 0.5 / 0.25 = **2 HVL**
            - Atenúa **75%** de radiación dispersa
            
            ### Efectividad de Medidas de Protección
            
            **Tabla comparativa** (reducción de dosis):
            
            | Medida | Factor de Reducción |
            |--------|---------------------|
            | **Salir de la sala** | ∞ (dosis = 0) |
            | **Distancia 1m → 2m** | 4× |
            | **Distancia 1m → 3m** | 9× |
            | **Biombo 2mm Pb** | ~1000× |
            | **Delantal 0.25mm Pb** | ~2× |
            | **Delantal 0.5mm Pb** | ~4× |
            | **Gafas plomadas** | 5-10× (cristalino) |
            | **Protección tiroidea** | 10× (tiroides) |
            | **Colimación (½ campo)** | 2× (paciente) |
            | **Modo pulsado vs continuo** | 2-10× (fluoroscopia) |
            
            **Conclusión**: La combinación de medidas es multiplicativa
            
            Ejemplo: Distancia ×2 + Delantal 0.5mm + Biombo = 4 × 4 × 1000 = **16,000× reducción**
            
            ### Radiación Natural de Fondo
            
            **Fuentes naturales** (promedio España: ~2.5 mSv/año):
            
            - **Radón** (gas): ~1.3 mSv/año (50%)
            - **Radiación cósmica**: ~0.4 mSv/año (varía con altitud)
            - **Radiación terrestre**: ~0.5 mSv/año (varía con geología)
            - **Interna** (K-40, C-14): ~0.3 mSv/año
            
            **Variabilidad geográfica**:
            - Nivel del mar: ~2 mSv/año
            - Madrid (600m altitud): ~2.5 mSv/año
            - Zonas graníticas (Galicia): hasta 5-6 mSv/año
            - Vuelo trasatlántico: +0.05 mSv
            
            **Comparación con medicina**:
            - Fondo natural: 2.5 mSv/año
            - RX tórax: 0.02 mSv (= 3 días de fondo)
            - RX abdomen: 0.7 mSv (= 3-4 meses de fondo)
            - TC abdomen: 10 mSv (= 4 años de fondo)
            
            ### Clasificación de Zonas (RD 783/2001)
            
            #### Zona Controlada
            
            **Definición**: Zona donde puede superarse 6 mSv/año o 3/10 de límites de órgano
            
            **Características**:
            - Acceso restringido (señalización)
            - Solo trabajadores expuestos o autorizados
            - Dosimetría individual obligatoria
            - Vigilancia médica especial
            
            **Ejemplos**:
            - Salas de RX durante funcionamiento
            - Salas de fluoroscopia/intervencionismo
            - Salas de TC
            - Bunkers de aceleradores lineales
            
            #### Zona Vigilada
            
            **Definición**: Zona donde puede superarse 1 mSv/año pero no criterios de controlada
            
            **Características**:
            - Señalización menos restrictiva
            - Acceso regulado
            - Dosimetría recomendada pero no siempre obligatoria
            
            **Ejemplos**:
            - Salas de control (tras biombo)
            - Pasillos adyacentes a salas de RX
            - Zonas cercanas a fuentes
            
            #### Zona de Libre Acceso
            
            **Definición**: Dosis <1 mSv/año
            
            - Público general puede acceder
            - No requiere medidas especiales
            
            ### Clasificación de Trabajadores
            
            #### Categoría A
            
            **Criterio**: Puede superar 6 mSv/año o 3/10 de límites de órgano
            
            **Obligaciones**:
            - Dosimetría individual obligatoria (mensual)
            - Vigilancia médica específica (anual)
            - Formación específica (20h inicial + actualización)
            - Historial dosimétrico
            
            **Ejemplos**:
            - TSID en intervencionismo
            - TSID en fluoroscopia intensiva
            - Físicos médicos
            - Médicos intervencionistas
            
            #### Categoría B
            
            **Criterio**: No supera criterios de Cat. A
            
            **Obligaciones**:
            - Dosimetría recomendada
            - Vigilancia médica general
            - Formación básica
            
            **Ejemplos**:
            - TSID en radiología convencional
            - TSID en TC
            - Personal administrativo en zonas vigiladas
            
            ### Riesgo de Cáncer Inducido por Radiación
            
            **Estimación ICRP** (modelo LNT):
            """)
            
            st.latex(r"\text{Riesgo} \approx 5\% \text{ por Sv}")
            
            st.markdown("""
            O más precisamente: **5.5% por Sv** (población general)
            
            **Interpretación**:
            - 1 Sv (1000 mSv) → ~5.5% probabilidad adicional de cáncer mortal
            - 10 mSv → ~0.055% = 1 en 1,800
            - 1 mSv → ~0.0055% = 1 en 18,000
            
            **Contexto**:
            - Riesgo base de cáncer (España): ~25% (1 de cada 4)
            - 10 mSv aumenta riesgo a: 25.055% (cambio imperceptible individualmente)
            - Pero: Significativo en poblaciones grandes
            
            **Ejemplos prácticos**:
            
            | Exploración | Dosis Efectiva | Riesgo Adicional | Equivalente a |
            |-------------|----------------|------------------|---------------|
            | **RX Tórax PA** | 0.02 mSv | 1 en 1,000,000 | 3 días de fondo natural |
            | **RX Abdomen** | 0.7 mSv | 1 en 26,000 | 4 meses de fondo |
            | **Mamografía** | 0.4 mSv | 1 en 45,000 | 2 meses de fondo |
            | **TC Tórax** | 7 mSv | 1 en 2,600 | 3 años de fondo |
            | **TC Abdomen** | 10 mSv | 1 en 1,800 | 4 años de fondo |
            | **PET-TC** | 15 mSv | 1 en 1,200 | 6 años de fondo |
            
            **Importante**: 
            - Estos son riesgos **muy bajos**
            - Casi siempre el beneficio diagnóstico >> riesgo
            - La NO realización de estudio necesario tiene más riesgo
            - Pero: **Justificación y optimización siempre obligatorias**
            
            ### Gestión del Riesgo: Principio de Proporcionalidad
            
            **Balance riesgo/beneficio** según situación:
            
            **Alta justificación** (beneficio muy alto):
            - Trauma severo → TC inmediato sin dudar
            - Sospecha cáncer → Estudios necesarios
            - Emergencia vital → Dosis no es limitante
            
            **Justificación moderada**:
            - Seguimiento de patología conocida → Optimizar frecuencia
            - Síntomas inespecíficos → Considerar alternativas (US, MRI)
            - Chequeos → Individualizar necesidad
            
            **Baja/nula justificación**:
            - Screening sin indicación → NO realizar
            - "Por si acaso" → NO justificado
            - Repetición por curiosidad → NO ético
            
            ### Principio de Proporcionalidad en Acción
            
            **Caso 1: Niño con traumatismo craneal leve**
            - Riesgo radiación: Mayor (niño más radiosensible)
            - Beneficio: Bajo si criterios clínicos no indican TC
            - **Decisión**: Observación clínica, evitar TC si no indicado
            
            **Caso 2: Adulto mayor con sospecha de cáncer pulmonar**
            - Riesgo radiación: Bajo (menor expectativa de vida, menor radiosensibilidad)
            - Beneficio: Alto (diagnóstico precoz puede ser curativo)
            - **Decisión**: TC tórax claramente justificado
            
            **Caso 3: Mujer joven con dolor abdominal inespecífico**
            - Riesgo radiación: Moderado (edad fértil)
            - Beneficio: Depende de clínica
            - **Decisión**: Ecografía primero, TC solo si indicación clara
            
            ### Conclusión Práctica
            
            Como TSID, tu rol es:
            
            1. ✅ **Verificar justificación** (prescripción médica)
            2. ✅ **Optimizar técnica** (ALARA para el paciente)
            3. ✅ **Protegerte** (ALARA ocupacional)
            4. ✅ **Documentar** (trazabilidad de dosis)
            5. ✅ **Comunicar** (explicar al paciente, reportar incidentes)
            
            **No eres responsable de** justificar la exploración (médico prescriptor),
            **pero sí de** cuestionar si hay dudas razonables.
            
            **Ante duda**: Consultar con radiólogo o supervisor de protección radiológica.
            """)
    
    # ============================================
    # SECTION 2: DOSE LIMITS AND DOSIMETRY
    # ============================================
    elif protection_section == "📊 Límites y Dosimetría":
        st.subheader("📊 Límites de Dosis y Dosimetría Personal")
        
        st.markdown("""
        Comprende los límites legales de dosis y cómo interpretar tu dosimetría personal.
        """)
        
        # Interactive dose limit comparison
        st.markdown("### 📏 Límites de Dosis Legales")
        
        # Visual comparison of limits
        limits_data = {
            "Categoría": ["Trabajador\n(efectiva)", "Trabajador\n(cristalino)", "Trabajador\n(piel)", 
                         "Embarazada\n(abdomen)", "Aprendiz\n16-18 años", "Público"],
            "Límite Anual (mSv)": [20, 20, 500, 2, 6, 1]
        }
        
        fig_limits = go.Figure()
        
        colors_limits = ['blue', 'orange', 'red', 'purple', 'green', 'lightblue']
        
        fig_limits.add_trace(go.Bar(
            x=limits_data["Categoría"],
            y=limits_data["Límite Anual (mSv)"],
            marker=dict(color=colors_limits),
            text=limits_data["Límite Anual (mSv)"],
            textposition='auto'
        ))
        
        fig_limits.update_layout(
            title="Límites de Dosis Anuales (Legislación Española)",
            yaxis_title="Dosis (mSv/año)",
            yaxis_type="log",  # Logarithmic scale due to wide range
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_limits, use_container_width=True)
        
        st.info("""
        📌 **Nota importante**: El límite de cristalino se **redujo drásticamente** de 150 a 20 mSv/año 
        con la transposición de Directiva 2013/59/EURATOM (RD 1029/2022).
        
        Esto hace **obligatorio** el uso de gafas plomadas en fluoroscopia e intervencionismo.
        """)
        
        # Personal dosimetry simulator
        st.markdown("---")
        st.markdown("### 🔬 Simulador de Dosimetría Personal")
        
        dosim_col1, dosim_col2 = st.columns(2)
        
        with dosim_col1:
            st.markdown("#### Tu Perfil")
            worker_category = st.selectbox(
                "Categoría de trabajador",
                ["Categoría A (intervencionismo/fluoro)", "Categoría B (RX convencional/TC)", 
                 "Estudiante en prácticas", "Embarazada (declarada)"]
            )
            
            work_area = st.selectbox(
                "Área de trabajo principal",
                ["Radiología convencional", "TC", "Fluoroscopia", "Intervencionismo vascular",
                 "Radiología portátil", "Mixto"]
            )
            
            hours_per_week = st.slider("Horas de trabajo por semana", 10, 60, 40, 5)
            
        with dosim_col2:
            st.markdown("#### Dosimetría Mensual (últimos 3 meses)")
            month1 = st.number_input("Mes 1 (mSv)", 0.0, 5.0, 0.2, 0.01, help="Lectura dosímetro mes 1")
            month2 = st.number_input("Mes 2 (mSv)", 0.0, 5.0, 0.15, 0.01, help="Lectura dosímetro mes 2")
            month3 = st.number_input("Mes 3 (mSv)", 0.0, 5.0, 0.18, 0.01, help="Lectura dosímetro mes 3")
        
        # Calculate projections
        avg_monthly = (month1 + month2 + month3) / 3
        projected_annual = avg_monthly * 12
        
        # Determine applicable limit
        if "Embarazada" in worker_category:
            applicable_limit = 2  # mSv resto de embarazo (~6-7 meses)
            limit_period = "resto de embarazo"
        elif "Estudiante" in worker_category:
            applicable_limit = 6
            limit_period = "año"
        else:
            applicable_limit = 20
            limit_period = "año"
        
        # Calculate percentage
        percentage_of_limit = (projected_annual / applicable_limit) * 100
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Análisis de Tu Dosimetría")
        
        result_col1, result_col2, result_col3, result_col4 = st.columns(4)
        
        with result_col1:
            st.metric(
                "Promedio Mensual",
                f"{avg_monthly:.2f} mSv",
                help="Promedio de los últimos 3 meses"
            )
            
        with result_col2:
            st.metric(
                "Proyección Anual",
                f"{projected_annual:.2f} mSv",
                help="Extrapolación a 12 meses"
            )
            
        with result_col3:
            st.metric(
                "Límite Aplicable",
                f"{applicable_limit} mSv/{limit_period}",
                help="Límite legal según tu categoría"
            )
            
        with result_col4:
            st.metric(
                "% del Límite",
                f"{percentage_of_limit:.1f}%",
                delta=f"{percentage_of_limit - 100:.1f}%" if percentage_of_limit > 100 else None,
                delta_color="inverse"
            )
        
        # Interpretation and recommendations
        st.markdown("### 💡 Interpretación y Recomendaciones")
        
        if percentage_of_limit < 10:
            st.success(f"""
            ✅ **Excelente control de dosis**
            
            Tu dosis proyectada ({projected_annual:.2f} mSv/año) representa solo el {percentage_of_limit:.1f}% del límite.
            
            **Situación**: Óptima
            - Protección muy efectiva
            - Prácticas de trabajo seguras
            - Continúa con las medidas actuales
            
            **Acción**: Ninguna necesaria, mantener buenas prácticas
            """)
            
        elif percentage_of_limit < 30:
            st.info(f"""
            ℹ️ **Control adecuado**
            
            Tu dosis proyectada ({projected_annual:.2f} mSv/año) es el {percentage_of_limit:.1f}% del límite.
            
            **Situación**: Dentro de rangos normales para tu área
            - Protección efectiva
            - Prácticas correctas
            
            **Acción**: Continuar con protección habitual, revisar ALARA periódicamente
            """)
            
        elif percentage_of_limit < 60:
            st.warning(f"""
            ⚠️ **Atención - Revisión recomendada**
            
            Tu dosis proyectada ({projected_annual:.2f} mSv/año) es el {percentage_of_limit:.1f}% del límite.
            
            **Situación**: Elevada pero dentro del límite
            - Revisar prácticas de protección
            - Identificar fuentes principales de exposición
            - Optimizar técnicas
            
            **Acciones recomendadas**:
            1. Revisar uso correcto de protecciones (delantal, biombo)
            2. Verificar distancias de trabajo
            3. Consultar con supervisor de protección radiológica
            4. Formación de actualización en protección
            5. Considerar rotación de tareas si posible
            """)
            
        else:  # >= 60%
            st.error(f"""
            🚨 **Alerta - Acción inmediata requerida**
            
            Tu dosis proyectada ({projected_annual:.2f} mSv/año) es el {percentage_of_limit:.1f}% del límite.
            
            **Situación**: Riesgo de superar límite legal
            
            **ACCIONES OBLIGATORIAS**:
            1. ⚠️ **Informar inmediatamente** a supervisor de protección radiológica
            2. 🔍 **Investigación** de causas (dosímetro correcto, prácticas, equipos)
            3. 🛡️ **Refuerzo** de medidas de protección
            4. 📋 **Evaluación** puesto de trabajo
            5. 🔄 **Reasignación temporal** si es necesario
            6. 📊 **Seguimiento** dosimétrico más frecuente
            
            ⚠️ **Superar el límite** es una no conformidad legal grave
            """)
        
        # Dosimetry comparison chart
        st.markdown("---")
        st.markdown("### 📈 Historial Dosimétrico")
        
        # Create a simple trend chart
        months = ['Mes 1', 'Mes 2', 'Mes 3', 'Proyección\nanual']
        doses = [month1, month2, month3, projected_annual]
        
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Bar(
            x=months[:3],
            y=doses[:3],
            name='Dosis mensual',
            marker=dict(color='lightblue')
        ))
        
        fig_trend.add_trace(go.Scatter(
            x=months,
            y=[avg_monthly, avg_monthly, avg_monthly, projected_annual],
            mode='lines+markers',
            name='Promedio/Proyección',
            line=dict(color='blue', dash='dash')
        ))
        
        # Add limit line
        fig_trend.add_hline(
            y=applicable_limit,
            line_dash="dot",
            line_color="red",
            annotation_text=f"Límite legal: {applicable_limit} mSv",
            annotation_position="right"
        )
        
        fig_trend.update_layout(
            title="Evolución de Dosis y Proyección Anual",
            yaxis_title="Dosis (mSv)",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Dosimeter types explanation
        st.markdown("---")
        st.markdown("### 🔬 Tipos de Dosímetros")
        
        dosim_type_col1, dosim_type_col2 = st.columns(2)
        
        with dosim_type_col1:
            st.markdown("""
            #### TLD (Thermoluminescent Dosimeter)
            
            **Principio**: 
            - Material (LiF) almacena energía de radiación
            - Al calentar, emite luz proporcional a dosis
            
            **Características**:
            - ✅ Reutilizable
            - ✅ Amplio rango de medida
            - ✅ Relativamente económico
            - ❌ Lectura destructiva (debe enviarse)
            - ❌ No lectura inmediata
            
            **Uso**: Dosimetría oficial mensual/trimestral
            
            **Colocación**: Solapa o pecho (representativo de cuerpo)
            """)
            
        with dosim_type_col2:
            st.markdown("""
            #### OSL (Optically Stimulated Luminescence)
            
            **Principio**:
            - Material (Al₂O₃:C) estimulado con luz láser
            - Emite luz proporcional a dosis
            
            **Características**:
            - ✅ Lectura no destructiva (puede releerse)
            - ✅ Mayor sensibilidad que TLD
            - ✅ Menos sensible a calor/luz ambiental
            - ✅ Más estable
            - ❌ Más costoso
            
            **Uso**: Cada vez más estándar en dosimetría oficial
            
            **Ventaja**: Relectura posible en caso de duda
            """)
        
        st.markdown("""
        #### Dosímetros Electrónicos (EPD - Electronic Personal Dosimeter)
        
        **Principio**: Detector de semiconductor + electrónica
        
        **Características**:
        - ✅ **Lectura inmediata** (tiempo real)
        - ✅ Alarmas programables
        - ✅ Registro continuo (trazabilidad)
        - ✅ Útil para formación (feedback inmediato)
        - ❌ Más costoso
        - ❌ Requiere baterías/mantenimiento
        - ❌ No sustituye dosimetría oficial (complementario)
        
        **Uso**: Intervencionismo, fluoroscopia (alto riesgo)
        
        **Ventaja principal**: Permite optimización inmediata de prácticas
        """)
        
        # Dosimeter placement
        st.markdown("---")
        st.markdown("### 📍 Colocación Correcta del Dosímetro")
        
        placement_col1, placement_col2 = st.columns(2)
        
        with placement_col1:
            st.markdown("""
            #### Sin Delantal Plomado
            
            **Posición**: Parte frontal del torso, entre pecho y cintura
            
            **Razón**: Representa dosis a órganos del tronco (más radiosensibles)
            
            **Típico en**:
            - Radiología convencional (trabajo tras biombo)
            - TC (sala de control)
            - Cuando NO hay exposición directa
            """)
            
        with placement_col2:
            st.markdown("""
            #### Con Delantal Plomado
            
            **Configuración estándar**: 1 dosímetro
            - **Posición**: Bajo el delantal (pecho)
            - **Mide**: Dosis efectiva real tras protección
            
            **Configuración completa**: 2 dosímetros
            - **Uno bajo delantal** (pecho): Dosis a tronco protegido
            - **Uno sobre delantal** (cuello): Dosis a tiroides, cristalino
            - **Cálculo**: Dosis efectiva ponderada
            """)
        
        st.info("""
        **⚠️ Importante en Fluoroscopia/Intervencionismo**:
        
        Debido al nuevo límite de cristalino (20 mSv/año), se recomienda:
        - **Dosímetro de anillo** (manos cerca del campo)
        - **Dosímetro sobre delantal** (estimar dosis a cristalino)
        - **Gafas plomadas** (obligatorias)
        - **Protección tiroidea** (recomendada)
        """)
        
        # Dosimetry record keeper
        st.markdown("---")
        st.markdown("### 📋 Registro Dosimétrico")
        
        st.markdown("""
        **Tu derecho como trabajador expuesto**:
        
        ✅ Acceso a tu historial dosimétrico completo
        
        ✅ Información sobre dosis recibidas (mensual)
        
        ✅ Copia del historial al cambiar de empleo
        
        ✅ Conservación del historial (mínimo hasta 30 años tras cese actividad)
        
        **Obligación del empleador**:
        - Mantener registro actualizado
        - Comunicar lecturas al trabajador
        - Informar si se superan niveles de investigación
        - Enviar datos a Registro Nacional de Dosis (CSN)
        """)
        
        # Dose comparison tool
        st.markdown("---")
        st.markdown("### 🔢 Comparador de Dosis")
        
        st.markdown("Compara tu dosis ocupacional con otras fuentes de exposición")
        
        your_annual_dose = projected_annual
        
        comparisons = {
            "Tu dosis anual proyectada": your_annual_dose,
            "Fondo natural (España)": 2.5,
            "Límite público general": 1.0,
            "Vuelo Madrid-New York (ida/vuelta)": 0.1,
            "Mamografía (paciente)": 0.4,
            "TC abdomen (paciente)": 10.0,
            "Límite trabajador expuesto": applicable_limit
        }
        
        fig_comparison = go.Figure()
        
        colors_comparison = ['red' if 'Tu dosis' in k else 'blue' if 'Límite trabajador' in k else 'gray' 
                            for k in comparisons.keys()]
        
        fig_comparison.add_trace(go.Bar(
            y=list(comparisons.keys()),
            x=list(comparisons.values()),
            orientation='h',
            marker=dict(color=colors_comparison),
            text=[f"{v:.2f} mSv" for v in comparisons.values()],
            textposition='auto'
        ))
        
        fig_comparison.update_layout(
            title="Comparación de Dosis (mSv/año)",
            xaxis_title="Dosis (mSv)",
            height=450,
            showlegend=False
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Action items
        st.markdown("---")
        st.markdown("### ✅ Checklist de Buenas Prácticas Dosimétricas")
        
        checklist_col1, checklist_col2 = st.columns(2)
        
        with checklist_col1:
            st.markdown("""
            **Uso del dosímetro**:
            
            ☑️ Llevar dosímetro durante TODA la jornada laboral
            
            ☑️ Colocación correcta (según protocolo)
            
            ☑️ NO olvidarlo en vestuario/taquilla
            
            ☑️ NO dejarlo cerca de fuentes de radiación cuando no lo llevas
            
            ☑️ NO compartir con otros (es personal)
            
            ☑️ Protegerlo de daños físicos
            
            ☑️ NO lavar (puede dañarlo)
            """)
            
        with checklist_col2:
            st.markdown("""
            **Gestión dosimétrica**:
            
            ☑️ Revisar lecturas mensuales
            
            ☑️ Investigar aumentos inusuales
            
            ☑️ Reportar pérdida o daño inmediatamente
            
            ☑️ Mantener registro personal (complementario)
            
            ☑️ Informar de embarazo (mujeres)
            
            ☑️ Actualizar datos de contacto con servicio dosimétrico
            
            ☑️ Solicitar historial al cambiar de empleo
            """)
        
        # Theory expander
        with st.expander("📚 Teoría: Dosimetría y Límites"):
            st.markdown("""
            ## 📊 Fundamentos de Dosimetría Personal
            
            ### ¿Por Qué Dosimetría Individual?
            
            **Objetivos**:
            1. **Verificar** que límites no se superan
            2. **Detectar** exposiciones anómalas
            3. **Optimizar** prácticas de protección
            4. **Documentar** historial (evidencia médico-legal)
            5. **Vigilancia** de salud laboral
            
            ### Características de un Buen Dosímetro
            
            **Requisitos técnicos**:
            - **Sensibilidad**: Medir desde ~0.01 mSv
            - **Rango dinámico**: Hasta varios Sv
            - **Independencia energética**: Respuesta similar a diferentes keV
            - **Independencia direccional**: Isotropía razonable
            - **Linealidad**: Respuesta proporcional a dosis
            - **Estabilidad**: No degradación con tiempo
            
            **Requisitos prácticos**:
            - Pequeño y ligero
            - Robusto
            - No interferir con trabajo
            - Fácil identificación
            - Económico (reutilizable)
            
            ### Dosímetros TLD: Funcionamiento Detallado
            
            **Material**: LiF:Mg,Ti (Fluoruro de litio dopado)
            
            **Proceso**:
            
            1. **Exposición**: Radiación ioniza átomos del cristal
            2. **Atrapamiento**: Electrones quedan atrapados en defectos del cristal
            3. **Almacenamiento**: Electrones permanecen atrapados (semanas/meses)
            4. **Lectura**: Calentamiento (~300°C) libera electrones
            5. **Emisión**: Electrones emiten luz al volver a nivel base
            6. **Medida**: Fotomultiplicador mide luz total ∝ dosis
            7. **Borrado**: Calentamiento completo resetea el dosímetro
            
            **Ventajas**:
            - Integra dosis durante periodo completo
            - Reutilizable indefinidamente
            - Pequeño (chip de 3×3×1 mm típico)
            
            **Limitaciones**:
            - Lectura destructiva (no relectura)
            - Desvanecimiento (fading) ~5% en 3 meses
            - Sensible a luz y calor extremos
            
            ### Dosímetros OSL: Ventajas sobre TLD
            
            **Material**: Al₂O₃:C (Óxido de aluminio dopado con carbono)
            
            **Diferencias clave**:
            - **Estimulación**: Luz láser (no calor)
            - **Lectura**: NO destructiva (solo consume ~0.1% señal)
            - **Relectura**: Posible múltiples veces
            - **Estabilidad**: Mejor (menos fading)
            - **Sensibilidad**: 3-5× mayor que TLD
            
            **Proceso de lectura**:
            1. Láser verde (532 nm) estimula el dosímetro
            2. Electrones atrapados se liberan
            3. Emiten luz UV-azul (420 nm)
            4. Filtro óptico separa luz láser de señal
            5. PMT mide señal ∝ dosis
            6. Solo se consume pequeña fracción de señal
            
            ### Dosímetros Electrónicos: Tiempo Real
            
            **Tecnologías**:
            - **Diodo de silicio**: Sensible, económico
            - **Cámara de ionización miniatura**: Referencia gold-standard
            - **MOSFET**: Alta sensibilidad, compacto
            
            **Funcionalidades**:
            - Display en tiempo real
            - Alarmas (dosis rate y dosis acumulada)
            - Memoria de eventos
            - Comunicación (Bluetooth, IR)
            - Registro temporal (gráficas)
            
            **Aplicaciones ideales**:
            - **Formación**: Feedback inmediato mejora prácticas
            - **Alto riesgo**: Fluoroscopia, intervencionismo
            - **Investigación**: Análisis detallado de exposiciones
            - **Emergencias**: Gestión en tiempo real
            
            **Limitación**: NO sustituye dosimetría oficial (TLD/OSL)
            → Complementario, no alternativo
            
            ### Niveles de Registro e Investigación
            
            **Nivel de Registro** (Recording Level):
            - Dosis mínima que debe registrarse oficialmente
            - Típicamente: **0.1 mSv** en periodo de medida
            - Por debajo: Se registra como "< nivel de registro" o "0"
            
            **Nivel de Investigación** (Investigation Level):
            - Dosis que dispara investigación de causas
            - Típicamente: **3/10 del límite anual**
            - Para trabajador Cat. A: ~**6 mSv** en periodo
            
            **Si se supera nivel de investigación**:
            1. Verificar dosímetro (¿uso correcto?, ¿daño?)
            2. Analizar prácticas de trabajo (¿cambios?)
            3. Evaluar equipos (¿mal funcionamiento?)
            4. Revisar protecciones (¿adecuadas?)
            5. Documentar hallazgos
            6. Implementar acciones correctivas
            7. Seguimiento reforzado
            
            ### Interpretación de Lecturas Anómalas
            
            **Lectura muy alta (ej: 10 mSv en 1 mes)**:
            
            Posibles causas:
            1. **Exposición real**: Procedimientos complejos, emergencias
            2. **Uso incorrecto**: Dosímetro dejado cerca de fuente
            3. **Contaminación radiactiva**: Poco probable en RX (no en medicina nuclear)
            4. **Fallo del dosímetro**: Exposición a luz/calor extremo
            5. **Intercambio de dosímetros**: Con colega de área de mayor riesgo
            
            **Investigación**:
            - Entrevista al trabajador (¿recuerda algo inusual?)
            - Revisión de registro de trabajo (¿procedimientos especiales?)
            - Verificación dosimétrica (¿otros trabajadores también elevados?)
            - Lectura de dosímetro electrónico si existe
            
            **Lectura cero constante**:
            
            Posibles causas:
            1. **No uso del dosímetro** (¡incumplimiento!)
            2. **Excelente protección** (poco probable si es siempre cero)
            3. **Trabajo exclusivo sin exposición** (¿realista?)
            
            **Acción**: Verificar que el trabajador lleva el dosímetro
            
            ### Dosimetría de Extremidades
            
            **Cuándo necesaria**:
            - Manos cerca del haz primario (< 5 cm)
            - Fluoroscopia/intervencionismo con manos en campo
            - Sujeción de pacientes (¡NO debería ocurrir!)
            - Braquiterapia
            
            **Dosímetro de anillo**:
            - Se lleva en dedo (base, no punta)
            - Mano dominante (más expuesta)
            - Lado palmar (hacia la fuente)
            - TLD de chip único o múltiple
            
            **Interpretación**:
            - Dosis en anillo >> dosis en torso (normal)
            - Límite: 500 mSv/año
            - Si >100 mSv/año: Revisar técnica
            
            ### Dosimetría de Cristalino
            
            **Nuevo límite (20 mSv/año)** ha cambiado paradigma:
            
            **Estimación de dosis a cristalino**:
            
            Método 1: **Dosímetro sobre delantal** (cuello)
            - Aproximación: Dosis_cristalino ≈ 0.75 × Dosis_cuello
            - Con gafas plomadas: ÷ 10 adicional
            
            Método 2: **Dosímetro específico** (cerca de ojo)
            - Clip en gafas o diadema
            - Más preciso pero menos práctico
            
            Método 3: **Cálculo desde cuerpo entero**
            - Dosis_cristalino ≈ 3 × Dosis_sobre_delantal
            - O: Dosis_cristalino ≈ 10-100 × Dosis_bajo_delantal
            - Muy variable según geometría
            
            **Protección obligatoria**:
            - Gafas plomadas (0.5-0.75 mm Pb eq)
            - Reducción típica: Factor 5-10
            - Con protección lateral: Factor >10
            
            ### Algoritmo de Cálculo de Dosis Efectiva
            
            **Configuración: Dosímetro bajo delantal**
            
            Dosis efectiva ≈ Lectura dosímetro
            
            (El delantal ya ha atenuado, dosímetro mide dosis real a órganos del tronco)
            
            **Configuración: Dos dosímetros (bajo y sobre delantal)**
            
            Método NCRP Report 122:
            """)
            
            st.latex(r"E = 0.5 \times H_B + 0.025 \times H_O")
            
            st.markdown("""
            Donde:
            - E = Dosis efectiva
            - H_B = Lectura bajo delantal (cuerpo)
            - H_O = Lectura sobre delantal (cuello)
            - Coeficientes reflejan: 50% órganos protegidos, 2.5% no protegidos
            
            **Ejemplo**:
            - Bajo delantal: 0.1 mSv/mes
            - Sobre delantal: 2.0 mSv/mes
            - E = 0.5 × 0.1 + 0.025 × 2.0 = 0.05 + 0.05 = **0.10 mSv/mes**
            
            ### Historial Dosimétrico: Valor Legal
            
            **Información que debe contener**:
            - Datos personales del trabajador
            - Periodo de medida
            - Dosis efectiva
            - Dosis equivalentes (si aplicable)
            - Tipo de dosímetro
            - Instalación/empresa
            - Tipo de trabajo
            
            **Conservación**:
            - Hasta 30 años tras fin de actividad
            - O hasta 75 años de edad del trabajador
            - La que sea más larga
            
            **Registro centralizado**:
            - España: **Banco de Datos de Dosis** (CSN)
            - Todas las empresas deben enviar datos
            - Permite seguimiento si trabajador cambia de empleo
            - Accesible para vigilancia médica
            
            **Derechos del trabajador**:
            - Acceso completo a su historial
            - Copia al finalizar relación laboral
            - Información comprensible (no solo números)
            - Explicación si valores inusuales
            
            ### Casos Especiales
            
            #### Mujeres en Edad Fértil
            
            **Normativa**: Sin restricciones especiales
            - Mismo límite que hombres (20 mSv/año)
            - **Pero**: Obligación de declarar embarazo lo antes posible
            
            **Buena práctica**:
            - Información previa sobre importancia de declaración
            - Ambiente laboral que facilite comunicación
            - Plan de contingencia preparado
            
            #### Embarazo Declarado
            
            **Límites desde declaración**:
            - **Feto**: 1 mSv resto de embarazo
            - **Superficie abdomen**: 2 mSv/mes
            
            **Gestión práctica**:
            
            Opción 1: **Reasignación temporal**
            - A trabajo sin exposición (TC, administrativa)
            - Preferible y habitual
            
            Opción 2: **Continuar con restricciones**
            - Solo radiología convencional (no portátil)
            - Dosímetro adicional a nivel de abdomen
            - Seguimiento mensual estricto
            - **NO** fluoroscopia, intervencionismo, portátiles
            
            Opción 3: **Baja laboral**
            - Si no es posible reasignación
            - Según valoración médica
            
            #### Estudiantes en Prácticas
            
            **Límites** (16-18 años):
            - Dosis efectiva: 6 mSv/año
            - 3/10 del límite de trabajadores
            
            **Requisitos**:
            - Supervisión directa obligatoria
            - Formación específica previa
            - Dosimetría individual
            - Autorización de padres/tutores (menores)
            - Vigilancia médica
            
            **Prohibiciones**:
            - NO tareas de máximo riesgo sin supervisión
            - NO exposiciones no justificadas (formativas puras)
            
            ### Auditorías e Inspecciones
            
            **Inspecciones del CSN**:
            - Periódicas (cada 2-5 años según instalación)
            - Verifican cumplimiento normativo
            - Revisan historiales dosimétricos
            - Pueden solicitar documentación
            
            **Preparación para inspección**:
            - Historiales actualizados
            - Acreditación servicio dosimétrico
            - Registros de entrega/recogida dosímetros
            - Formación del personal al día
            - Protocolos escritos disponibles
            
            **No conformidades típicas**:
            - Dosimetría no actualizada
            - Trabajadores sin dosímetro
            - Formación caducada
            - Vigilancia médica no realizada
            - Documentación incompleta
            
            **Consecuencias de incumplimiento**:
            - Advertencia y plazo para corrección
            - Multas (según gravedad)
            - Suspensión de autorización (casos graves)
            - Responsabilidad penal (negligencia grave)
            
            ### Buenas Prácticas en Gestión Dosimétrica
            
            **Para el TSID**:
            1. Llevar dosímetro SIEMPRE que estés en zona controlada/vigilada
            2. Colocación correcta según protocolo
            3. NO dejar cerca de fuentes cuando no lo llevas
            4. Revisar lecturas mensuales
            5. Reportar anomalías inmediatamente
            6. Mantener tu propio registro (Excel, cuaderno)
            7. Solicitar explicaciones si no entiendes lecturas
            
            **Para el supervisor de protección radiológica**:
            1. Sistema fiable de distribución/recogida
            2. Análisis rutinario de lecturas
            3. Investigación proactiva de desviaciones
            4. Feedback a trabajadores
            5. Formación continua
            6. Cultura de seguridad (no punitiva ante reporte)
            
            ### Tecnologías Emergentes
            
            **Dosimetría en tiempo real con IA**:
            - Algoritmos predicen dosis basándose en parámetros de equipo
            - Sin necesidad de dosímetro físico
            - Útil para optimización inmediata
            - **Limitación**: No sustituye dosimetría legal (aún)
            
            **Dosimetría 3D**:
            - Reconstrucción de distribución de dosis en cuerpo
            - Basada en geometría del procedimiento
            - Permite optimizar posición del trabajador
            
            **Blockchain para historial dosimétrico**:
            - Registro inmutable y descentralizado
            - Acceso controlado por el trabajador
            - Portabilidad entre empleadores
            - En fase de investigación
            
            ### Conclusión Práctica
            
            La dosimetría personal es:
            - **Obligación legal** (no opcional)
            - **Herramienta de protección** (te informa de tu exposición)
            - **Evidencia médico-legal** (historial para toda la vida)
            - **Instrumento de optimización** (mejora prácticas)
            
            **Tu responsabilidad**:
            - Usarla correctamente
            - Revisarla regularmente
            - Actuar si valores preocupantes
            - Conservar tu historial
            
            **Recuerda**: La mejor dosis es la que NO recibes (ALARA).
            """)
    
    # ============================================
    # SECTION 3: SHIELDING CALCULATIONS
    # ============================================
    elif protection_section == "🧱 Cálculo de Blindajes":
        st.subheader("🧱 Cálculo de Blindajes")
        
        st.markdown("""
        Aprende a calcular el blindaje necesario para proteger áreas adyacentes a salas de rayos X.
        """)
        
        # Interactive shielding calculator
        st.markdown("### 🔧 Calculadora de Blindaje")
        
        shield_col1, shield_col2, shield_col3 = st.columns(3)
        
        with shield_col1:
            st.markdown("#### Parámetros del Equipo")
            shield_kVp = st.slider("kVp del equipo", 40, 150, 100, 5, key="shield_kvp")
            workload_patients_day = st.number_input("Pacientes/día", 1, 200, 50)
            avg_mAs_per_patient = st.number_input("mAs promedio/paciente", 1, 100, 20)
            days_per_week_operation = st.number_input("Días/semana operación", 1, 7, 5)
            
        with shield_col2:
            st.markdown("#### Geometría")
            distance_to_point = st.slider("Distancia a punto de interés (m)", 1.0, 10.0, 3.0, 0.5)
            occupancy_factor = st.select_slider(
                "Factor de ocupación",
                options=[1, 0.5, 0.2, 0.05, 0.025],
                value=1,
                help="1=siempre ocupado, 0.5=50% tiempo, 0.2=20%, 0.05=ocasional, 0.025=raro"
            )
            use_factor = st.select_slider(
                "Factor de uso",
                options=[1, 0.5, 0.25, 0.1],
                value=0.25,
                help="Fracción del tiempo que el haz apunta a esa barrera"
            )
            
        with shield_col3:
            st.markdown("#### Objetivo de Protección")
            area_type = st.selectbox(
                "Tipo de área a proteger",
                ["Área controlada (trabajadores)", "Área pública", "Exterior edificio"]
            )
            
            if "pública" in area_type.lower() or "Exterior" in area_type:
                target_dose_mSv_week = st.number_input(
                    "Dosis objetivo (mSv/semana)",
                    0.001, 0.1, 0.02,
                    help="Típico: 0.02 mSv/semana (= 1 mSv/año)"
                )
            else:
                target_dose_mSv_week = st.number_input(
                    "Dosis objetivo (mSv/semana)",
                    0.001, 1.0, 0.4,
                    help="Típico: 0.4 mSv/semana (= 20 mSv/año)"
                )
        
        # Calculate workload
        workload_mAmin = calculate_workload(workload_patients_day, avg_mAs_per_patient, days_per_week_operation)
        
        # Estimate unshielded dose at 1m (very simplified model)
        # Typical: ~1 µGy per mAs at 1m for scatter
        dose_rate_per_mAs = 0.001  # mSv per mAs at 1m (scatter approximation)
        
        # Weekly dose at 1m without shielding
        weekly_exposure_mAs = workload_patients_day * avg_mAs_per_patient * days_per_week_operation
        dose_at_1m_week = weekly_exposure_mAs * dose_rate_per_mAs
        
        # Apply inverse square law
        dose_at_distance = calculate_dose_at_distance(dose_at_1m_week, 1.0, distance_to_point)
        
        # Apply use and occupancy factors
        dose_at_point_unshielded = dose_at_distance * use_factor * occupancy_factor
        
        # Calculate required attenuation
        if dose_at_point_unshielded > target_dose_mSv_week:
            attenuation_needed = target_dose_mSv_week / dose_at_point_unshielded
        else:
            attenuation_needed = 1.0  # No shielding needed
        
        # Get HVL for lead at this kVp
        hvl_lead = get_hvl_for_material("Plomo", shield_kVp)
        hvl_concrete = get_hvl_for_material("Hormigón", shield_kVp)
        
        # Calculate thickness needed
        if attenuation_needed < 1.0:
            n_hvls_needed = -np.log2(attenuation_needed)
            thickness_lead_mm = n_hvls_needed * hvl_lead
            thickness_concrete_cm = n_hvls_needed * hvl_concrete / 10
        else:
            n_hvls_needed = 0
            thickness_lead_mm = 0
            thickness_concrete_cm = 0
        
        # Display results
        st.markdown("---")
        st.markdown("### 📊 Resultados del Cálculo")
        
        calc_col1, calc_col2, calc_col3, calc_col4 = st.columns(4)
        
        with calc_col1:
            st.metric(
                "Carga de trabajo",
                f"{workload_mAmin:.1f} mA·min/semana",
                help="Carga total semanal"
            )
            
        with calc_col2:
            st.metric(
                "Dosis sin blindaje",
                f"{dose_at_point_unshielded:.3f} mSv/semana",
                help="Dosis en el punto sin protección"
            )
            
        with calc_col3:
            st.metric(
                "Atenuación necesaria",
                f"Factor {1/attenuation_needed:.1f}" if attenuation_needed < 1 else "No necesaria",
                help="Factor de reducción requerido"
            )
            
        with calc_col4:
            st.metric(
                "HVL necesarias",
                f"{n_hvls_needed:.1f}",
                help="Número de capas hemirreductoras"
            )
        
        # Shielding recommendations
        st.markdown("### 🛡️ Recomendaciones de Blindaje")
        
        if attenuation_needed >= 1.0:
            st.success("""
            ✅ **No se requiere blindaje adicional**
            
            La distancia y los factores de uso/ocupación son suficientes para cumplir objetivos de dosis.
            
            **Recomendaciones**:
            - Verificar que puertas/ventanas no permitan exposición directa
            - Señalización adecuada de zona controlada
            - Mantener protocolos de acceso
            """)
        else:
            recommend_col1, recommend_col2 = st.columns(2)
            
            with recommend_col1:
                st.info(f"""
                **Opción 1: Plomo**
                
                Espesor mínimo: **{thickness_lead_mm:.2f} mm Pb**
                
                Redondeado: **{np.ceil(thickness_lead_mm*4)/4:.2f} mm Pb**
                
                (Láminas comerciales: 0.5, 1, 1.5, 2, 2.5, 3 mm)
                
                **Ventajas**:
                - ✅ Menor espesor
                - ✅ Ocupa menos espacio
                - ✅ Ideal para retrofitting
                
                **Desventajas**:
                - ❌ Más costoso
                - ❌ Pesado (11.3 kg/m² por mm)
                - ❌ Requiere soporte estructural
                """)
                
            with recommend_col2:
                st.info(f"""
                **Opción 2: Hormigón**
                
                Espesor mínimo: **{thickness_concrete_cm:.1f} cm hormigón**
                
                Redondeado: **{np.ceil(thickness_concrete_cm):.0f} cm**
                
                (Densidad estándar: 2.35 g/cm³)
                
                **Ventajas**:
                - ✅ Más económico
                - ✅ Estructural (pared portante)
                - ✅ Estándar en construcción
                
                **Desventajas**:
                - ❌ Mucho más grueso
                - ❌ Solo viable en construcción nueva
                - ❌ Reduce espacio útil
                """)
            
            # Additional materials
            st.markdown("**Opciones Alternativas:**")
            
            materials_comparison = {
                "Material": ["Plomo", "Acero", "Hormigón baritado", "Hormigón normal", "Ladrillo macizo"],
                "Espesor (mm)": [
                    thickness_lead_mm,
                    thickness_lead_mm * (get_hvl_for_material("Acero", shield_kVp) / hvl_lead),
                    thickness_concrete_cm * 10 * 0.7,  # Barite concrete is denser
                    thickness_concrete_cm * 10,
                    thickness_concrete_cm * 10 * 1.3
                ],
                "Peso (kg/m²)": [
                    thickness_lead_mm * 11.3,
                    thickness_lead_mm * (get_hvl_for_material("Acero", shield_kVp) / hvl_lead) * 7.8,
                    thickness_concrete_cm * 10 * 0.7 * 0.35,
                    thickness_concrete_cm * 10 * 0.235,
                    thickness_concrete_cm * 10 * 1.3 * 0.18
                ],
                "Coste relativo": [4, 2, 1.5, 1, 0.8]
            }
            
            df_materials = pd.DataFrame(materials_comparison)
            df_materials["Espesor (mm)"] = df_materials["Espesor (mm)"].round(1)
            df_materials["Peso (kg/m²)"] = df_materials["Peso (kg/m²)"].round(1)
            
            st.dataframe(df_materials, use_container_width=True)
        
        # Visualization of attenuation
        st.markdown("---")
        st.markdown("### 📉 Curva de Atenuación")
        
        # Calculate dose vs shielding thickness
        thicknesses_mm_pb = np.linspace(0, thickness_lead_mm * 1.5 if thickness_lead_mm > 0 else 5, 50)
        doses_vs_thickness = []
        
        for thick in thicknesses_mm_pb:
            transmission = calculate_transmission_through_shielding(hvl_lead, thick)
            dose = dose_at_point_unshielded * transmission
            doses_vs_thickness.append(dose)
        
        fig_attenuation = go.Figure()
        
        fig_attenuation.add_trace(go.Scatter(
            x=thicknesses_mm_pb,
            y=doses_vs_thickness,
            mode='lines',
            name='Dosis vs espesor',
            line=dict(color='blue', width=3)
        ))
        
        # Add target line
        fig_attenuation.add_hline(
            y=target_dose_mSv_week,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Objetivo: {target_dose_mSv_week} mSv/semana",
            annotation_position="right"
        )
        
        # Mark required thickness
        if thickness_lead_mm > 0:
            fig_attenuation.add_vline(
                x=thickness_lead_mm,
                line_dash="dot",
                line_color="red",
                annotation_text=f"Requerido: {thickness_lead_mm:.2f} mm",
                annotation_position="top"
            )
        
        fig_attenuation.update_layout(
            title="Atenuación vs Espesor de Plomo",
            xaxis_title="Espesor de Plomo (mm)",
            yaxis_title="Dosis (mSv/semana)",
            yaxis_type="log",
            height=450,
            hovermode='x'
        )
        
        st.plotly_chart(fig_attenuation, use_container_width=True)
        
        # Factors explanation
        st.markdown("---")
        st.markdown("### 📖 Factores de Cálculo")
        
        factor_col1, factor_col2 = st.columns(2)
        
        with factor_col1:
            st.markdown(f"""
            #### Factor de Uso (U = {use_factor})
            
            **Definición**: Fracción del tiempo que el haz primario apunta hacia la barrera
            
            **Valores típicos**:
            - **1.0**: Suelo (haz siempre hacia abajo)
            - **0.25**: Paredes (haz horizontal ~25% del tiempo)
            - **0.1**: Techo (haz hacia arriba raramente)
            - **0.5**: Pared con bucky mural (uso frecuente)
            
            **Tu selección**: {use_factor} = {['Uso completo', 'Uso frecuente (50%)', 'Uso medio (25%)', 'Uso ocasional (10%)'][{1:0, 0.5:1, 0.25:2, 0.1:3}[use_factor]]}
            """)
            
        with factor_col2:
            st.markdown(f"""
            #### Factor de Ocupación (T = {occupancy_factor})
            
            **Definición**: Fracción del tiempo que el área está ocupada
            
            **Valores típicos**:
            - **1.0**: Área siempre ocupada (oficinas, salas de espera)
            - **0.5**: Ocupación media (pasillos con tráfico)
            - **0.2**: Ocupación baja (almacenes, cuartos técnicos)
            - **0.05**: Ocupación ocasional (escaleras, baños)
            - **0.025**: Ocupación rara (aparcamientos, azoteas)
            
            **Tu selección**: {occupancy_factor} = {['Siempre', 'Media (50%)', 'Baja (20%)', 'Ocasional (5%)', 'Rara (2.5%)'][{1:0, 0.5:1, 0.2:2, 0.05:3, 0.025:4}[occupancy_factor]]}
            """)
        
        # Practical considerations
        st.markdown("---")
        st.markdown("### 🔨 Consideraciones Prácticas de Instalación")
        
        pract_col1, pract_col2 = st.columns(2)
        
        with pract_col1:
            st.warning("""
            **⚠️ Puntos Críticos - NO olvidar**:
            
            1. **Juntas y solapamientos**:
               - Solapar láminas mínimo 1 cm
               - Sellar juntas (no dejar huecos)
               - Especial atención en esquinas
            
            2. **Penetraciones**:
               - Puertas: Equivalente a pared (mismo blindaje)
               - Ventanas: Vidrio plomado + marco plomado
               - Conductos: Laberintos o blindaje adicional
               - Cables/tuberías: Sellar con material plomado
            
            3. **Áreas vulnerables**:
               - Bajo puertas (umbral plomado)
               - Marcos de ventanas
               - Falsos techos/suelos técnicos
               - Cajas de enchufes/interruptores
            
            4. **Carga estructural**:
               - Plomo es MUY pesado (11.3 kg/m² por mm)
               - Verificar capacidad estructural
               - Refuerzo puede ser necesario
               - Consultar ingeniero estructural
            """)
            
        with pract_col2:
            st.info("""
            **✅ Buenas Prácticas**:
            
            1. **Documentación**:
               - Planos "as-built" con blindajes
               - Certificados de materiales
               - Medidas de verificación post-instalación
               - Conservar para inspecciones
            
            2. **Verificación**:
               - Medidas con detector calibrado
               - Con equipo en funcionamiento
               - En todas las áreas críticas
               - Por empresa acreditada
            
            3. **Señalización**:
               - Zona controlada (trébol)
               - Zona vigilada (si aplica)
               - Advertencias en puertas
               - Luces de aviso en funcionamiento
            
            4. **Mantenimiento**:
               - Inspección visual periódica (grietas, oxidación)
               - Verificar puertas (cierre hermético)
               - Revisar tras reformas/obras
               - Re-evaluación si cambio de equipo/uso
            """)
        
        # Quick reference table
        st.markdown("---")
        st.markdown("### 📋 Tabla de Referencia Rápida")
        
        st.markdown("""
        **Espesores típicos de plomo para diferentes escenarios** (orientativo):
        """)
        
        reference_data = {
            "Escenario": [
                "RX dental (70 kVp)",
                "RX general (80-100 kVp)",
                "RX alta tensión (120-150 kVp)",
                "Fluoroscopia (100-120 kVp)",
                "TC (120-140 kVp)",
                "Mamografía (25-30 kVp)"
            ],
            "Pared primaria (mm Pb)": ["1.5-2.0", "2.0-2.5", "2.5-3.0", "2.5-3.0", "2.5-3.0", "0.5-1.0"],
            "Pared secundaria (mm Pb)": ["0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0-2.5", "1.5-2.0", "0.5"],
            "Puerta (mm Pb)": ["1.0", "1.5-2.0", "2.0", "2.0-2.5", "2.0", "0.5"],
            "Visor (mm Pb eq)": ["1.0", "1.5-2.0", "2.0", "2.0", "2.0", "0.5"]
        }
        
        df_reference = pd.DataFrame(reference_data)
        st.dataframe(df_reference, use_container_width=True)
        
        st.caption("""
        ⚠️ **Advertencia**: Estos son valores orientativos. El cálculo exacto debe realizarse 
        por un experto en protección radiológica considerando todos los factores específicos 
        de la instalación.
        """)
        
        # Download calculation report
        if thickness_lead_mm > 0:
            st.markdown("---")
            st.markdown("### 📄 Informe de Cálculo")
            
            report_text = f"""
INFORME DE CÁLCULO DE BLINDAJE
================================

PARÁMETROS DE ENTRADA:
----------------------
Equipo: Rayos X de diagnóstico
kVp máximo: {shield_kVp} kVp
Carga de trabajo: {workload_mAmin:.1f} mA·min/semana
  - Pacientes/día: {workload_patients_day}
  - mAs/paciente: {avg_mAs_per_patient}
  - Días/semana: {days_per_week_operation}

GEOMETRÍA:
----------
Distancia al punto: {distance_to_point} m
Factor de uso (U): {use_factor}
Factor de ocupación (T): {occupancy_factor}

OBJETIVO DE PROTECCIÓN:
-----------------------
Tipo de área: {area_type}
Dosis objetivo: {target_dose_mSv_week} mSv/semana

RESULTADOS:
-----------
Dosis sin blindaje: {dose_at_point_unshielded:.3f} mSv/semana
Atenuación requerida: {1/attenuation_needed:.1f}×
Número de HVL: {n_hvls_needed:.2f}

BLINDAJE RECOMENDADO:
--------------------
Plomo: {thickness_lead_mm:.2f} mm Pb (redondear a {np.ceil(thickness_lead_mm*4)/4:.2f} mm)
Hormigón: {thickness_concrete_cm:.1f} cm (redondear a {np.ceil(thickness_concrete_cm):.0f} cm)

HVL utilizado:
- Plomo: {hvl_lead:.3f} mm
- Hormigón: {hvl_concrete:.1f} mm

VERIFICACIÓN POST-INSTALACIÓN:
------------------------------
Se recomienda verificar mediante medidas directas que la dosis en el punto
de interés no supera {target_dose_mSv_week} mSv/semana con el equipo en 
condiciones de carga máxima.

NORMATIVA APLICABLE:
-------------------
- Real Decreto 1085/2009
- Real Decreto 783/2001
- Guía de Seguridad del CSN nº 5.10

Fecha: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

NOTA: Este cálculo es orientativo. El diseño definitivo debe ser realizado
por un experto cualificado en protección radiológica.
"""
            
            st.download_button(
                label="📥 Descargar Informe (TXT)",
                data=report_text,
                file_name=f"informe_blindaje_{shield_kVp}kVp.txt",
                mime="text/plain"
            )
    
    # ============================================
    # SECTION 4: FACILITY DESIGN
    # ============================================
    elif protection_section == "🏥 Diseño de Instalaciones":
        st.subheader("🏥 Diseño de Instalaciones Radiológicas")
        
        st.markdown("""
        Principios de diseño y distribución de una sala de rayos X para optimizar 
        la protección radiológica.
        """)
        
        # Facility type selector
        facility_type = st.selectbox(
            "Tipo de instalación",
            ["Sala de Radiología Convencional", "Sala de Fluoroscopia", "Sala de TC", 
             "Sala de Mamografía", "Radiología Dental"]
        )
        
        st.markdown(f"### 📐 Diseño de: {facility_type}")
        
        # Show specific design for each type
        if facility_type == "Sala de Radiología Convencional":
            st.markdown("""
            #### Requisitos Mínimos (RD 1085/2009)
            
            **Dimensiones**:
            - Superficie mínima: **20 m²** (recomendado 25-30 m²)
            - Altura mínima: **2.5 m**
            - Distancia tubo-bucky: Ajustable 100-150 cm
            
            **Elementos de protección**:
            """)
            
            design_col1, design_col2 = st.columns(2)
            
            with design_col1:
                st.info("""
                **Barreras Primarias**:
                
                ✅ Paredes donde incide haz directo
                - Típicamente: 2 paredes (bucky mural + camilla)
                - Blindaje: 2-2.5 mm Pb eq (100 kVp)
                - Hasta 2.1 m de altura mínimo
                
                ✅ Suelo (si hay sala debajo)
                - Blindaje: 2-2.5 mm Pb eq
                - Considerar peso del equipo
                """)
                
            with design_col2:
                st.info("""
                **Barreras Secundarias**:
                
                ✅ Resto de paredes (dispersión)
                - Blindaje: 1-1.5 mm Pb eq
                - Altura completa (hasta techo)
                
                ✅ Techo (si hay sala encima)
                - Blindaje: 0.5-1 mm Pb eq (factor uso bajo)
                
                ✅ Puertas
                - Blindaje: 2 mm Pb eq
                - Cierre hermético
                - Marco plomado
                """)
            
            st.markdown("""
            **Sala de Control**:
            - Biombo plomado: 2 mm Pb eq
            - Visor: Vidrio plomado 2 mm Pb eq (40×40 cm mínimo)
            - Visibilidad completa del paciente
            - Distancia mínima: 2 m del tubo
            - Intercom bidireccional
            """)
            
            st.markdown("""
            **Señalización y Seguridad**:
            - 🚨 Luz de aviso roja "RAYOS X" en puertas
            - ⚠️ Señal trébol radiación en accesos
            - 🔴 Pulsador de emergencia (corta RX)
            - 📋 Cartel identificativo de instalación
            - 🚪 Apertura de puertas desde interior sin llave
            """)
            
            st.markdown("""
            **Equipamiento adicional**:
            - Delantales plomados (0.5 mm Pb eq) - mínimo 2
            - Protecciones gonadales (adulto y pediátrico)
            - Protección tiroidea
            - Colimador luminoso calibrado
            - Dosímetro de área (opcional pero recomendado)
            """)
            
        elif facility_type == "Sala de Fluoroscopia":
            st.markdown("""
            #### Requisitos Específicos para Fluoroscopia
            
            ⚠️ **Mayor riesgo de exposición** - Requisitos más estrictos
            """)
            
            fluoro_col1, fluoro_col2 = st.columns(2)
            
            with fluoro_col1:
                st.warning("""
                **Blindaje Reforzado**:
                
                - Paredes: **2.5-3 mm Pb eq** (todas)
                - Puertas: **2.5 mm Pb eq**
                - Visor control: **2.5 mm Pb eq**
                - Suelo/techo: **2 mm Pb eq** mínimo
                
                **Protecciones en Sala**:
                
                - Faldones plomados en mesa (obligatorio)
                - Mamparas suspendidas (techo)
                - Cortinas plomadas laterales
                - Todos mínimo 0.5 mm Pb eq
                """)
                
            with fluoro_col2:
                st.error("""
                **Protección Personal Obligatoria**:
                
                - Delantal 0.5 mm Pb eq (uso continuo)
                - Protección tiroidea (obligatoria)
                - Gafas plomadas 0.5 mm Pb eq (¡OBLIGATORIAS!)
                - Guantes plomados si manos cerca campo
                
                **Dosimetría Reforzada**:
                
                - Dosímetro cuerpo (bajo delantal)
                - Dosímetro adicional (sobre delantal) recomendado
                - Dosímetro de anillo si procedente
                - Lectura mensual obligatoria
                """)
            
            st.info("""
            **Equipamiento Específico**:
            
            - Modo pulsado (obligatorio en equipos nuevos)
            - Control automático dosis (CAD)
            - Registro de dosis por procedimiento
            - Última imagen guardada (LIH)
            - Colimación automática
            - Filtros adicionales (Cu)
            """)
            
        elif facility_type == "Sala de TC":
            st.success("""
            #### Ventajas del TC en Protección Radiológica
            
            ✅ **Exposición ocupacional mínima**:
            - Personal NUNCA en sala durante escaneo
            - Sala de control completamente separada
            - Muy bajo riesgo para técnicos
            
            **Blindaje típico** (120-140 kVp):
            - Paredes: 2-2.5 mm Pb eq
            - Puerta: 2 mm Pb eq
            - Visor: 2 mm Pb eq
            - Laberinto en entrada (reduce blindaje puerta)
            
            **Características de Sala**:
            
            📏 **Dimensiones**:
            - Mínimo 25-30 m² (depende del gantry)
            - Altura: 2.7-3.0 m (paso de camilla alta)
            
            🔧 **Requisitos técnicos**:
            - Climatización reforzada (calor del equipo)
            - Suelo reforzado (equipo muy pesado: 1-3 ton)
            - Instalación eléctrica dedicada
            - Sistema de inyección de contraste
            
            🛡️ **Protección del paciente**:
            - Colimación automática estricta
            - Control automático exposición (AEC)
            - Protocolos pediátricos específicos
            - Registro de dosis (obligatorio)
            - DRL (Diagnostic Reference Levels)
            
            ℹ️ **Sala de control**:
            - Completamente aislada
            - Visión directa de la sala
            - Comunicación bidireccional
            - Monitor de vídeo del paciente
            - Acceso fácil en emergencias
            """)
            
        elif facility_type == "Sala de Mamografía":
            st.markdown("""
            #### Particularidades de Mamografía
            
            🟣 **Energías muy bajas** (25-35 kVp) → Blindaje más sencillo
            """)
            
            mamo_col1, mamo_col2 = st.columns(2)
            
            with mamo_col1:
                st.info("""
                **Blindaje Reducido**:
                
                - Paredes: **0.5-1 mm Pb eq** (suficiente)
                - Puerta: **0.5 mm Pb eq**
                - Visor: **0.5 mm Pb eq**
                - Hormigón estándar puede ser suficiente
                
                **Ventajas**:
                
                ✅ Menor coste de blindaje
                ✅ Menor peso estructural
                ✅ Más flexible en ubicación
                ✅ Retrofitting más sencillo
                """)
                
            with mamo_col2:
                st.warning("""
                **Consideraciones Especiales**:
                
                - Programas de screening (alto volumen)
                - Cálculo con carga alta
                - Compresión = Repeticiones (optimizar)
                - Calidad de imagen crítica (↓dosis difícil)
                
                **Control de Calidad Estricto**:
                
                - Pruebas diarias (phantom)
                - Calibración semestral
                - Mantenimiento preventivo
                - Registro exhaustivo
                """)
            
            st.success("""
            **Protección de la Paciente**:
            
            - ⚠️ Verificar embarazo (mujeres <50 años)
            - Técnica optimizada (compresión adecuada)
            - kVp mínimo necesario
            - Rejilla de alta frecuencia
            - Filtros específicos (Mo, Rh)
            - AGD (Average Glandular Dose) <2.5 mGy
            
            **Protección del Personal**:
            
            - Exposición ocupacional muy baja
            - Técnico tras biombo siempre
            - Delantal no necesario habitualmente
            - Dosimetría estándar suficiente
            """)
            
        else:  # Radiología Dental
            st.markdown("""
            #### Radiología Dental - Casos Especiales
            
            🦷 **Características únicas** del entorno dental
            """)
            
            dental_col1, dental_col2 = st.columns(2)
            
            with dental_col1:
                st.info("""
                **Intraoral (periapical, bite-wing)**:
                
                📏 Dimensiones:
                - Sala pequeña: 6-9 m² suficiente
                - Puede ser gabinete polivalente
                
                🛡️ Blindaje:
                - Paredes: 1-1.5 mm Pb eq (60-70 kVp)
                - Puerta: 1 mm Pb eq
                - A veces: blindaje parcial (hasta 2m)
                
                👤 Técnico:
                - Distancia mínima: 2 m
                - Ángulo 90-135° respecto haz
                - Tras biombo si disponible
                - O fuera de sala
                """)
                
            with dental_col2:
                st.info("""
                **Panorámica / CBCT**:
                
                📏 Dimensiones:
                - 8-12 m² recomendado
                - Espacio para rotación equipo
                
                🛡️ Blindaje:
                - Panorámica: Similar intraoral
                - CBCT: Como RX convencional (2 mm Pb eq)
                
                ⚠️ CBCT (Cone Beam CT):
                - Mayor volumen irradiado
                - Dosis mayor (0.1-0.5 mSv)
                - Justificación estricta
                - Protocolos optimizados
                """)
            
            st.warning("""
            **Particularidades del entorno dental**:
            
            ⚠️ **Riesgos específicos**:
            - Múltiples exposiciones por sesión (serie periapical: 4-18 Rx)
            - Sujeción de película por paciente (buena práctica) o asistente (EVITAR)
            - Espacios pequeños → Difícil mantener distancia
            
            ✅ **Soluciones**:
            - Posicionadores de película (sin sujeción manual)
            - Colimación rectangular (reduce área 60%)
            - Sensores digitales (reducen dosis 50-80%)
            - Técnico fuera de sala o tras biombo
            - Dosimetría si >100 Rx/semana
            
            📋 **Regulación específica**:
            - Puede no requerir supervisor de PR (instalación tipo II)
            - Control de calidad simplificado
            - Pero: Normativa de protección igual de estricta
            """)
        
        # Common elements for all facilities
        st.markdown("---")
        st.markdown("### 🔍 Verificación Post-Instalación")
        
        verification_col1, verification_col2 = st.columns(2)
        
        with verification_col1:
            st.markdown("""
            #### Pruebas Obligatorias
            
            **Antes de uso clínico**:
            
            1. ✅ **Estudio de blindajes**
               - Medidas con detector calibrado
               - Carga de trabajo máxima
               - Todos los puntos críticos
               - Informe por experto cualificado
            
            2. ✅ **Pruebas de aceptación**
               - Fabricante o servicio técnico
               - Verificar especificaciones
               - Seguridades funcionales
               - Calibración inicial
            
            3. ✅ **Estado de referencia**
               - Control de calidad completo
               - Establece valores de referencia
               - Por física médica
               - Base para controles periódicos
            """)
            
        with verification_col2:
            st.markdown("""
            #### Controles Periódicos
            
            **Mantenimiento de la protección**:
            
            📅 **Diario** (TSID):
            - Inspección visual equipos
            - Luces de aviso funcionando
            - Intercom operativo
            
            📅 **Mensual** (TSID + Supervisor):
            - Integridad de blindajes
            - Puertas y cierres
            - Protecciones plomadas (grietas)
            
            📅 **Anual** (Experto + Física Médica):
            - Control de calidad completo
            - Verificación de blindajes
            - Actualización del estudio
            - Informe para CSN
            """)
        
        # Layout best practices
        st.markdown("---")
        st.markdown("### 💡 Mejores Prácticas en Distribución")
        
        st.success("""
        **Principios de diseño óptimo**:
        
        1. 🚪 **Accesos**:
           - Evitar puertas directas a zonas públicas
           - Laberintos reducen blindaje de puertas
           - Doble puerta en zonas de alto tráfico
           - Apertura hacia exterior (evacuación)
        
        2. 📏 **Distancias**:
           - Máxima distancia entre sala y áreas sensibles
           - Considerar vertical (pisos superiores/inferiores)
           - Zona de espera NO adyacente a pared primaria
           - Oficinas administrativas alejadas
        
        3. 🏗️ **Agrupación**:
           - Agrupar salas de RX (optimiza blindajes)
           - Áreas controladas continuas (simplifica gestión)
           - Servicios comunes centralizados
           - Vestuarios y dosimetría cerca
        
        4. 🔄 **Flujos**:
           - Separar flujo pacientes / personal
           - Evitar cruces innecesarios
           - Circuito claro: espera → sala → recuperación → salida
           - Acceso equipos/materiales independiente
        
        5. 🎯 **Futuro**:
           - Prever ampliaciones
           - Flexibilidad para cambio de equipos
           - Instalaciones sobredimensionadas (eléctrica, clima)
           - Blindajes genéricos (no solo para equipo actual)
        """)
        
        # Common mistakes
        st.markdown("---")
        st.markdown("### ⚠️ Errores Comunes a Evitar")
        
        mistake_col1, mistake_col2 = st.columns(2)
        
        with mistake_col1:
            st.error("""
            **En diseño**:
            
            ❌ No considerar dispersión
            ❌ Olvidar blindaje de suelo/techo
            ❌ Ventanas sin protección
            ❌ Marcos de puertas no plomados
            ❌ Cajas eléctricas sin blindar
            ❌ Conductos sin laberinto
            ❌ No prever carga estructural (plomo pesa)
            ❌ Sala muy pequeña (imposible mantener distancia)
            """)
            
        with mistake_col2:
            st.error("""
            **En instalación**:
            
            ❌ Juntas mal selladas
            ❌ Solapamientos insuficientes
            ❌ Fijaciones inadecuadas (plomo se deforma)
            ❌ No proteger durante obra (daños)
            ❌ No verificar post-instalación
            ❌ No documentar (planos as-built)
            ❌ Señalización incorrecta/insuficiente
            ❌ No formar al personal antes del uso
            """)
        
        st.markdown("---")
        st.info("""
        💡 **Consejo final**: El diseño de una instalación radiológica debe involucrar desde el inicio a:
        
        - Experto en protección radiológica cualificado
        - Arquitecto con experiencia en instalaciones sanitarias
        - Ingeniero estructural (cargas)
        - Radiólogos/técnicos (flujos de trabajo)
        - Servicio de mantenimiento (accesibilidad)
        - Responsable de compras (presupuesto realista)
        
        **Un buen diseño inicial ahorra problemas y costes futuros.**
        """)
    
    # ============================================
    # SECTION 5: DIAGNOSTIC REFERENCE LEVELS (DRL)
    # ============================================
    elif protection_section == "📈 Niveles de Referencia (DRL)":
        st.subheader("📈 Niveles de Referencia Diagnósticos (DRL)")
        
        st.markdown("""
        Los Diagnostic Reference Levels (DRL) son herramientas de optimización para 
        comparar las dosis de tu centro con estándares nacionales/internacionales.
        """)
        
        # Explanation
        st.info("""
        ### ¿Qué son los DRL?
        
        **NO son**:
        - ❌ Límites de dosis (no son obligatorios estrictamente)
        - ❌ Valores óptimos (son valores altos del percentil 75)
        - ❌ Aplicables a pacientes individuales
        
        **SÍ son**:
        - ✅ Herramienta de **optimización**
        - ✅ Valores de **referencia** para comparación
        - ✅ Basados en **buenas prácticas** (percentil 75)
        - ✅ Aplicables a **grupos de pacientes** estándar
        - ✅ **Indicadores** de que algo puede mejorarse si se superan
        
        **Principio**: Si tu centro supera sistemáticamente los DRL, debes:
        1. Investigar las causas
        2. Optimizar protocolos
        3. Formar al personal
        4. Revisar equipos
        5. Documentar acciones
        """)
        
        # DRL comparison tool
        st.markdown("---")
        st.markdown("### 🔍 Comparador de Dosis con DRL")
        
        drl_col1, drl_col2 = st.columns(2)
        
        with drl_col1:
            st.markdown("#### Selecciona Exploración")
            exam_type = st.selectbox(
                "Tipo de examen",
                ["Tórax PA", "Tórax LAT", "Abdomen AP", "Pelvis AP", "Columna Lumbar AP", 
                 "Columna Lumbar LAT", "Cráneo AP/PA", "Mamografía", "TC Cráneo", "TC Tórax", "TC Abdomen"]
            )
            
            patient_type = st.selectbox(
                "Tipo de paciente",
                ["Adulto estándar (70 kg)", "Pediátrico (5 años)", "Pediátrico (10 años)"]
            )
            
        with drl_col2:
            st.markdown("#### Datos de Tu Centro")
            
            if "TC" in exam_type:
                your_dose_metric = "CTDIvol (mGy)"
                your_dose = st.number_input(your_dose_metric, 0.0, 100.0, 10.0, 0.1)
                your_dlp = st.number_input("DLP (mGy·cm)", 0.0, 2000.0, 500.0, 10.0)
            else:
                your_dose_metric = "Dosis entrada (mGy)" if "Mamografía" not in exam_type else "AGD (mGy)"
                your_dose = st.number_input(your_dose_metric, 0.0, 20.0, 2.0, 0.1)
        
        # DRL values (Spain/Europe - approximate values)
        DRL_VALUES = {
            "Tórax PA": {"adulto": {"entrada": 0.3, "efectiva": 0.02}, "pediátrico_5": {"entrada": 0.1}, "pediátrico_10": {"entrada": 0.15}},
            "Tórax LAT": {"adulto": {"entrada": 1.5, "efectiva": 0.04}},
            "Abdomen AP": {"adulto": {"entrada": 10.0, "efectiva": 0.7}, "pediátrico_5": {"entrada": 2.0}, "pediátrico_10": {"entrada": 4.0}},
            "Pelvis AP": {"adulto": {"entrada": 10.0, "efectiva": 0.7}},
            "Columna Lumbar AP": {"adulto": {"entrada": 10.0, "efectiva": 0.7}},
            "Columna Lumbar LAT": {"adulto": {"entrada": 30.0, "efectiva": 1.3}},
            "Cráneo AP/PA": {"adulto": {"entrada": 5.0, "efectiva": 0.07}},
            "Mamografía": {"adulto": {"AGD": 2.5}},
            "TC Cráneo": {"adulto": {"CTDIvol": 60.0, "DLP": 1000, "efectiva": 2.0}},
            "TC Tórax": {"adulto": {"CTDIvol": 15.0, "DLP": 600, "efectiva": 7.0}},
            "TC Abdomen": {"adulto": {"CTDIvol": 15.0, "DLP": 700, "efectiva": 10.0}}
        }
        
        # Get applicable DRL
        patient_key = "adulto" if "estándar" in patient_type else "pediátrico_5" if "5 años" in patient_type else "pediátrico_10"
        
        if exam_type in DRL_VALUES and patient_key in DRL_VALUES[exam_type]:
            drl_data = DRL_VALUES[exam_type][patient_key]
            
            if "TC" in exam_type:
                drl_value = drl_data.get("CTDIvol", 0)
                drl_dlp = drl_data.get("DLP", 0)
                metric_name = "CTDIvol"
            elif "Mamografía" in exam_type:
                drl_value = drl_data.get("AGD", 0)
                metric_name = "AGD"
            else:
                drl_value = drl_data.get("entrada", 0)
                metric_name = "Dosis entrada"
            
            # Compare
            percentage_of_drl = (your_dose / drl_value * 100) if drl_value > 0 else 0
            
            # Display results
            st.markdown("---")
            st.markdown("### 📊 Análisis Comparativo")
            
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.metric(
                    "Tu Dosis",
                    f"{your_dose:.2f} {'mGy' if 'mGy' in your_dose_metric else 'mGy'}",
                    help="Dosis típica en tu centro para este examen"
                )
                
            with comp_col2:
                st.metric(
                    "DRL (España/EU)",
                    f"{drl_value:.2f} {'mGy' if 'mGy' in your_dose_metric else 'mGy'}",
                    help="Nivel de referencia diagnóstico (percentil 75)"
                )
                
            with comp_col3:
                delta_text = f"{percentage_of_drl - 100:.1f}%" if percentage_of_drl > 100 else f"{100 - percentage_of_drl:.1f}%"
                st.metric(
                    "% del DRL",
                    f"{percentage_of_drl:.1f}%",
                    delta=delta_text if percentage_of_drl > 100 else f"-{delta_text}",
                    delta_color="inverse"
                )
            
            # Visual comparison
            fig_drl = go.Figure()
            
            fig_drl.add_trace(go.Bar(
                x=["Tu Centro", "DRL"],
                y=[your_dose, drl_value],
                marker=dict(color=['red' if your_dose > drl_value else 'green', 'blue']),
                text=[f"{your_dose:.2f}", f"{drl_value:.2f}"],
                textposition='auto'
            ))
            
            fig_drl.update_layout(
                title=f"Comparación con DRL: {exam_type}",
                yaxis_title=metric_name + " (mGy)",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_drl, use_container_width=True)
            
            # Interpretation
            st.markdown("### 💡 Interpretación")
            
            if percentage_of_drl < 50:
                st.success(f"""
                ✅ **Excelente - Muy por debajo del DRL** ({percentage_of_drl:.0f}%)
                
                Tu centro está aplicando técnicas muy optimizadas. Estás en el rango de mejores prácticas.
                
                **Mantén**:
                - Protocolos actuales
                - Formación continua del personal
                - Control de calidad riguroso
                
                **Considera**:
                - Compartir tu protocolo con otros centros
                - Verificar que calidad diagnóstica es adecuada (no sub-optimizar)
                """)
                
            elif percentage_of_drl < 75:
                st.info(f"""
                ℹ️ **Bueno - Por debajo del DRL** ({percentage_of_drl:.0f}%)
                
                Tu centro está dentro de buenas prácticas. La mayoría de centros están en este rango.
                
                **Mantén**:
                - Vigilancia de dosis
                - Revisión periódica de protocolos
                
                **Considera**:
                - Pequeñas optimizaciones aún posibles
                - Benchmarking con centros de referencia
                """)
                
            elif percentage_of_drl <= 100:
                st.warning(f"""
                ⚠️ **Atención - Cerca del DRL** ({percentage_of_drl:.0f}%)
                
                Tu centro está cerca o en el DRL. Es el momento de revisar y optimizar.
                
                **Acciones recomendadas**:
                1. Revisar protocolos (kVp, mAs, colimación)
                2. Verificar calibración de equipos
                3. Formación específica del personal
                4. Control de calidad exhaustivo
                5. Considerar actualización de equipos si son antiguos
                """)
                
            else:  # > 100%
                st.error(f"""
                🚨 **Acción Requerida - Superas el DRL** ({percentage_of_drl:.0f}%)
                
                Tu centro supera el nivel de referencia. Es **obligatorio** investigar y optimizar.
                
                **Acciones OBLIGATORIAS** (según normativa):
                
                1. 📋 **Análisis de causas**:
                   - ¿Equipos descalibrados?
                   - ¿Protocolos inadecuados?
                   - ¿Falta de formación?
                   - ¿Pacientes atípicos? (muy obesos, patología especial)
                
                2. 🔧 **Optimización**:
                   - Ajustar parámetros técnicos
                   - Revisar técnicas de posicionamiento
                   - Verificar AEC si disponible
                   - Considerar filtros adicionales
                
                3. 📚 **Formación**:
                   - Actualización TSID
                   - Radiólogos informados
                   - Protocolos escritos y accesibles
                
                4. 🔍 **Seguimiento**:
                   - Medición continua post-optimización
                   - Documentar mejoras
                   - Informe a supervisor de PR
                
                5. 📊 **Documentación**:
                   - Registrar acciones tomadas
                   - Evidencia de mejora
                   - Para auditorías/inspecciones
                
                ⚠️ **Nota importante**: Superar DRL NO es ilegal per se, pero requiere justificación 
                documentada y demostración de esfuerzos de optimización.
                """)
            
            # Additional DLP comparison for CT
            if "TC" in exam_type and drl_dlp > 0:
                st.markdown("---")
                st.markdown("#### Comparación DLP (Producto Dosis-Longitud)")
                
                percentage_dlp = (your_dlp / drl_dlp * 100) if drl_dlp > 0 else 0
                
                dlp_col1, dlp_col2, dlp_col3 = st.columns(3)
                
                with dlp_col1:
                    st.metric("Tu DLP", f"{your_dlp:.0f} mGy·cm")
                with dlp_col2:
                    st.metric("DRL DLP", f"{drl_dlp:.0f} mGy·cm")
                with dlp_col3:
                    st.metric("% DRL", f"{percentage_dlp:.0f}%")
                
                st.caption("""
                **DLP** (Dose-Length Product) considera la longitud escaneada.
                Es un mejor indicador de dosis total que CTDIvol solo.
                """)
        
        else:
            st.warning("DRL no disponible para esta combinación de examen y paciente")
        
        # DRL table reference
        st.markdown("---")
        st.markdown("### 📋 Tabla de Referencia DRL (España/Europa)")
        
        drl_table_data = {
            "Exploración": [
                "Tórax PA", "Abdomen AP", "Pelvis AP", "Columna Lumbar LAT",
                "Cráneo", "Mamografía (2 proyecciones)", "TC Cráneo", "TC Tórax", "TC Abdomen"
            ],
            "DRL Adulto": [
                "0.3 mGy", "10 mGy", "10 mGy", "30 mGy",
                "5 mGy", "2.5 mGy (AGD)", "60 mGy (CTDI)", "15 mGy (CTDI)", "15 mGy (CTDI)"
            ],
            "DLP (TC)": [
                "-", "-", "-", "-", "-", "-", "1000 mGy·cm", "600 mGy·cm", "700 mGy·cm"
            ],
            "Fuente": [
                "EU RP 180", "EU RP 180", "EU RP 180", "EU RP 180",
                "EU RP 180", "EU RP 180", "EU RP 180", "EU RP 180", "EU RP 180"
            ]
        }
        
        df_drl = pd.DataFrame(drl_table_data)
        st.dataframe(df_drl, use_container_width=True)
        
        st.caption("""
        📌 **Fuente**: European Commission RP 180 (2014) - Diagnostic Reference Levels in Thirty-six European Countries
        
        ⚠️ **Nota**: Estos son valores orientativos. Consultar DRL nacionales actualizados en documentos oficiales del CSN.
        """)
        
        # Optimization strategies
        st.markdown("---")
        st.markdown("### 🎯 Estrategias de Optimización")
        
        optim_col1, optim_col2 = st.columns(2)
        
        with optim_col1:
            st.markdown("""
            #### Para Radiología Convencional
            
            **1. Técnica**:
            - ✅ Aplicar regla del 15% (↑kVp, ↓mAs)
            - ✅ Colimación estricta
            - ✅ Usar AEC si disponible
            - ✅ Evitar repeticiones (técnica correcta primera vez)
            
            **2. Equipamiento**:
            - ✅ Filtración adicional (Cu)
            - ✅ Digital en lugar de analógico
            - ✅ Generadores alta frecuencia
            - ✅ Rejillas apropiadas (ratio correcto)
            - ✅ Mantenimiento preventivo
            
            **3. Personal**:
            - ✅ Formación continua
            - ✅ Protocolos escritos visibles
            - ✅ Feedback regular (dosis registradas)
            - ✅ Cultura de optimización
            
            **4. Paciente**:
            - ✅ Posicionamiento óptimo
            - ✅ Inmovilización adecuada
            - ✅ Preparación correcta
            - ✅ Compresión suave (abdomen)
            """)
            
        with optim_col2:
            st.markdown("""
            #### Para TC
            
            **1. Protocolos**:
            - ✅ Modulación de corriente (AEC)
            - ✅ Rango de escaneo mínimo necesario
            - ✅ Pitch optimizado
            - ✅ kVp según indicación (bajo para yodo)
            - ✅ Reconstructión iterativa
            
            **2. Tecnología**:
            - ✅ Equipos modernos (iterativa, dual-energy)
            - ✅ Algoritmos de reducción de dosis
            - ✅ Filtros de configuración (bow-tie)
            - ✅ Actualización de software
            
            **3. Indicación**:
            - ✅ Justificación estricta
            - ✅ Evitar TC "rutinarios"
            - ✅ Considerar alternativas (MRI, US)
            - ✅ Protocolos específicos por indicación
            
            **4. Pediátrico**:
            - ✅ Protocolos específicos obligatorios
            - ✅ Reducción dosis 50-80% vs adulto
            - ✅ Justificación aún más estricta
            - ✅ Alternative imaging first
            """)
        
        # DRL monitoring program
        st.markdown("---")
        st.markdown("### 📊 Programa de Monitorización de Dosis")
        
        st.info("""
        **Cómo implementar un programa DRL en tu centro**:
        
        **Paso 1: Recopilación de Datos**
        - Registrar dosis de TODOS los exámenes (DICOM dose reports)
        - Sistema informático automatizado (PACS, RIS)
        - Mínimo 20-50 pacientes por protocolo
        - Pacientes estándar (excluir extremos)
        
        **Paso 2: Análisis**
        - Calcular percentiles (25, 50, 75)
        - Tu DRL local = percentil 75
        - Comparar con DRL nacionales/europeos
        - Identificar outliers (valores extremos)
        
        **Paso 3: Evaluación**
        - ¿Tu P75 > DRL nacional? → Investigar
        - ¿Gran variabilidad? → Falta estandarización
        - ¿Muchos outliers? → Problemas técnicos o formación
        
        **Paso 4: Optimización**
        - Ajustar protocolos donde sea necesario
        - Formación específica
        - Revisión equipos
        - Documentar cambios
        
        **Paso 5: Re-evaluación**
        - Repetir medidas post-optimización
        - Verificar reducción de dosis
        - Mantener calidad diagnóstica
        - Documentar mejoras
        
        **Periodicidad**: Anual (mínimo) o tras cambios significativos
        """)
        
        # Theory expander
        with st.expander("📚 Teoría: Niveles de Referencia Diagnósticos"):
            st.markdown("""
            ## 📈 Fundamentos de los DRL
            
            ### Historia y Origen
            
            **Introducción**: ICRP 73 (1996)
            - Concepto: "Investigation levels" → "Diagnostic Reference Levels"
            - Objetivo: Identificar niveles inusualmente altos de dosis
            - NO son límites, son herramientas de optimización
            
            **Evolución**:
            - ICRP 73 (1996): Introducción del concepto
            - ICRP 103 (2007): Refuerzo y expansión
            - Directiva 2013/59/EURATOM: Obligatoriedad en EU
            - ICRP 135 (2017): Actualización y nuevas modalidades
            
            ### Marco Legal (España)
            
            **Real Decreto 1085/2009**:
            - Artículo 9: Obligación de establecer y usar DRL
            - Titular debe garantizar su aplicación
            - Supervisión por experto en PR
            
            **Real Decreto 783/2001**:
            - Marco general de protección radiológica
            - Principio de optimización (ALARA)
            - DRL como herramienta de optimización
            
            **Guía CSN 5.10**:
            - Orientación práctica
            - Valores de referencia españoles
            - Metodología de implementación
            
            ### Metodología de Establecimiento
            
            **Nivel Nacional/Regional**:
            
            1. **Recopilación de datos**:
               - Encuesta a centros representativos
               - Mínimo 10-20 centros
               - Pacientes estándar definidos
               - Equipos en buen estado
            
            2. **Análisis estadístico**:
               - Calcular percentiles de distribución
               - **Percentil 75** como DRL
               - No media (sesgo por valores altos)
               - No percentil 50 (sería "típico", no "alto")
            
            3. **Publicación**:
               - Documentos oficiales (CSN, EU)
               - Accesible a todos los centros
               - Actualización periódica (3-5 años)
            
            **Nivel Local (tu centro)**:
            
            1. **DRL local = Percentil 75 de tu centro**
            2. Comparar con DRL nacional
            3. Si P75 local > DRL nacional → Optimizar
            4. Objetivo: Reducir P75 local por debajo de DRL nacional
            
            ### Magnitudes Dosimétricas Usadas
            
            #### Radiología Convencional
            
            **Dosis Entrada en Superficie (ESD)**:
            - Medida en superficie de entrada del paciente
            - Incluye radiación dispersa retrógrada
            - Fácil de medir (TLD, cámara de ionización)
            - Usado en la mayoría de DRL de Rx simple
            
            **Producto Dosis-Área (DAP/PKA)**:
            - Integral de dosis sobre área del haz
            - Unidades: Gy·cm² o cGy·cm²
            - Medido automáticamente (cámara en colimador)
            - Mejor para fluoroscopia y procedimientos largos
            
            #### Tomografía Computarizada
            
            **CTDIvol** (CT Dose Index volume):
            - Dosis promedio en volumen escaneado
            - Para un único corte o serie
            - Unidades: mGy
            - Mostrado en consola del TC
            
            **DLP** (Dose-Length Product):
            - CTDIvol × Longitud escaneada
            - Unidades: mGy·cm
            - Mejor correlación con riesgo
            - Usado para calcular dosis efectiva
            
            **SSDE** (Size-Specific Dose Estimate):
            - Ajusta CTDIvol según tamaño del paciente
            - Más preciso (phantom estándar no representa a todos)
            - Emergente como métrica preferida
            
            #### Mamografía
            
            **AGD** (Average Glandular Dose):
            - Dosis promedio al tejido glandular
            - Calculada (no medida directamente)
            - Basada en kVp, HVL, compresión, composición mama
            - Unidades: mGy
            - Métrica estándar en mamografía
            
            ### Paciente Estándar
            
            **Definición necesaria para comparabilidad**:
            
            **Adulto estándar**:
            - Peso: 70 kg
            - Altura: 170 cm
            - IMC: 24 kg/m²
            - Espesores específicos según región
            
            **Pediátrico**:
            - Grupos de edad: 0, 1, 5, 10, 15 años
            - O grupos de peso
            - DRL específicos (mucho menores que adulto)
            
            **Exclusiones**:
            - Pacientes con IMC extremo (<18 o >30)
            - Patologías que requieren parámetros especiales
            - Prótesis metálicas extensas
            - Estudios no estándar
            
            ### Interpretación Estadística
            
            **¿Por qué percentil 75?**
            
            - No demasiado alto (99% sería muy permisivo)
            - No demasiado bajo (50% no indica "alto")
            - 75% = "Cuartil superior" = Límite de lo aceptable
            - El 25% más alto debe investigarse
            
            **Distribución típica**:
            """)
            
            st.latex(r"\text{P25} < \text{P50 (mediana)} < \text{P75 (DRL)} < \text{P95}")
            
            st.markdown("""
            **Interpretación**:
            - Si estás en P25: Excelente (pero verifica calidad diagnóstica)
            - Si estás en P50: Bueno (típico)
            - Si estás en P75 (DRL): Límite aceptable
            - Si estás >P75: Debes optimizar
            
            ### Acciones según Resultado
            
            **Tu dosis < DRL**:
            - ✅ Mantener protocolo
            - ✅ Verificar calidad diagnóstica adecuada
            - ✅ Documentar para auditorías
            - ✅ Considerar compartir protocolo
            
            **Tu dosis ≈ DRL** (90-110%):
            - ℹ️ Monitorización estrecha
            - ℹ️ Pequeñas optimizaciones
            - ℹ️ Revisión protocolo preventiva
            
            **Tu dosis > DRL** (>110%):
            - ⚠️ Investigación obligatoria
            - ⚠️ Análisis de causas
            - ⚠️ Plan de optimización
            - ⚠️ Documentación completa
            - ⚠️ Seguimiento post-optimización
            - ⚠️ Informe a autoridad si persiste
            
            **Tu dosis >> DRL** (>150%):
            - 🚨 Acción inmediata
            - 🚨 Suspender protocolo hasta resolver
            - 🚨 Investigación exhaustiva
            - 🚨 Posible problema grave (equipo, formación)
            - 🚨 Notificación a CSN recomendada
            
            ### Limitaciones de los DRL
            
            **NO sustituyen el juicio clínico**:
            - Paciente específico puede requerir dosis mayor
            - Indicación compleja justifica superación
            - Calidad diagnóstica prioritaria
            
            **NO son aplicables a**:
            - Procedimientos intervencionistas complejos
            - Pacientes con características extremas
            - Investigación (protocolos experimentales)
            - Emergencias vitales
            
            **Variabilidad**:
            - Entre países (diferente equipamiento, prácticas)
            - Entre centros (tecnología, formación)
            - Temporal (equipos envejecen o se modernizan)
            
            ### DRL y Calidad de Imagen
            
            **Concepto erróneo**: "Menos dosis siempre mejor"
            
            **Realidad**: Debe existir balance
            """)
            
            st.latex(r"\text{Dosis óptima} = \text{mín}\{\text{Dosis} : \text{Calidad diagnóstica adecuada}\}")
            
            st.markdown("""
            **Sobre-optimización (dosis demasiado baja)**:
            - Ruido excesivo
            - Contraste insuficiente
            - Artefactos
            - Diagnóstico imposible o incierto
            - Repeticiones (¡más dosis total!)
            
            **Por tanto**:
            - DRL es límite superior, NO objetivo a alcanzar
            - Objetivo: Mínima dosis compatible con calidad diagnóstica
            - Control de calidad imagen tan importante como control dosis
            
            ### Futuro de los DRL
            
            **Tendencias emergentes**:
            
            1. **DRL más específicos**:
               - Por indicación clínica (no solo anatomía)
               - Por tecnología (iterativa vs FBP)
               - Por tamaño paciente (SSDE en TC)
            
            2. **Monitorización automatizada**:
               - Software extrae datos de DICOM automáticamente
               - Alertas en tiempo real si >DRL
               - Dashboard para gestión
               - Big data y AI para benchmarking
            
            3. **DRL para nuevas modalidades**:
               - CBCT (dental, intervencionismo)
               - PET-CT
               - Dual-energy CT
               - Spectral imaging
            
            4. **Individualización**:
               - DRL ajustados por tamaño paciente
               - Considerar riesgo individual (edad, genética)
               - Medicina personalizada en dosimetría
            
            ### Programa Nacional de DRL
            
            **España - Registro de dosis**:
            - Centros deben enviar datos periódicamente
            - CSN analiza y actualiza DRL nacionales
            - Publicación en Guías de Seguridad
            - Comparación con EU
            
            **Beneficios**:
            - Benchmarking entre centros
            - Identificación de mejores prácticas
            - Detección de problemas sistémicos
            - Base para formación y guías
            
            ### Conclusión Práctica
            
            Los DRL son:
            - Herramienta de **optimización continua**
            - **NO punitivos** (si superas, optimizas, no te multan)
            - Requieren **cultura de seguridad** (reporte sin miedo)
            - Efectivos solo con **uso consistente**
            
            **Tu rol como TSID**:
            1. Conocer DRL de tu centro
            2. Monitorizar tus técnicas
            3. Reportar valores inusuales
            4. Participar en optimización
            5. Formación continua
            
            **Recuerda**: El objetivo final es **proteger al paciente** sin comprometer el diagnóstico.
            """)
    
    # Final section summary
    st.markdown("---")
    st.success("""
    ### 🎯 Puntos Clave - Protección Radiológica
    
    1. **ALARA es obligatorio**: Tiempo, Distancia, Blindaje - tres pilares fundamentales
    2. **Límites de dosis**: 20 mSv/año (trabajador), 1 mSv/año (público)
    3. **Cristalino**: Nuevo límite 20 mSv/año - Gafas plomadas obligatorias en fluoro
    4. **Embarazo**: Declarar inmediatamente - 1 mSv al feto durante embarazo
    5. **Dosimetría personal**: Obligatoria, individual, intransferible
    6. **Blindajes**: Calcular correctamente - No olvidar penetraciones
    7. **HVL**: Cada HVL reduce dosis a la mitad
    8. **Ley inversa cuadrado**: Duplicar distancia = ¼ de dosis
    9. **DRL**: Herramienta optimización, no límite - Investigar si superas
    10. **Justificación + Optimización + Limitación**: Tres principios de PR
    """)
    
    # Pro tips for this tab
    st.info("""
    ### 💡 Consejos Profesionales - Protección Radiológica
    
    **Para protegerte (ocupacional)**:
    - 🚪 Sal de la sala durante exposición (radiología convencional)
    - 📏 Mínimo 2 metros en portátiles (idealmente 3m)
    - 🦺 Delantal + gafas + protección tiroidea en fluoroscopia (OBLIGATORIO)
    - 📊 Revisa tu dosimetría mensualmente
    - 🤰 Si embarazo: declarar inmediatamente
    
    **Para proteger al paciente**:
    - 🎯 Justificación: ¿Es realmente necesaria la exploración?
    - ⚙️ Optimización: Mínimos kVp/mAs compatibles con calidad
    - ✂️ Colimación estricta: Solo área de interés
    - 🛡️ Protecciones: Gonadal si útil y no interfiere
    - 📋 Técnica correcta primera vez: Evitar repeticiones
    
    **Para cumplir normativa**:
    - 📋 Documentación completa y actualizada
    - 🎓 Formación específica vigente (renovar cada 5 años)
    - 🔍 Participar en controles de calidad
    - 📊 Conocer DRL y comparar tus técnicas
    - ⚠️ Reportar incidentes y no conformidades
    
    **Cultura de seguridad**:
    - 🗣️ Comunicación abierta sobre seguridad
    - ❓ Preguntar sin miedo si hay dudas
    - 📢 Reportar problemas (no punitivo)
    - 🤝 Trabajo en equipo (PR es responsabilidad de todos)
    - 📚 Formación continua (normativa cambia)
    """)
    
    # Footer for this tab
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.9em;'>
        <p>🛡️ <strong>Tab 3: Protección Radiológica</strong> | 
        Simulador de Física Radiológica | 
        Formación Profesional en Imagen para el Diagnóstico</p>
        <p>La protección radiológica no es opcional - Es tu responsabilidad profesional y legal</p>
    </div>
    """, unsafe_allow_html=True)


            
            

# ============================================
# TAB 4: PARÁMETROS TÉCNICOS
# ============================================
with tabs[3]:
    st.header("🔧 Parámetros Técnicos y Optimización")
    
    st.markdown("""
    Los **parámetros técnicos** son las variables que el técnico en radiología ajusta para optimizar 
    la calidad diagnóstica de la imagen mientras minimiza la dosis al paciente. Esta sección explora 
    las principales reglas de conversión, factores de exposición y herramientas de cálculo.
    """)
    
    # ============================================
    # SECTION 1: Factores Fundamentales
    # ============================================
    st.markdown("---")
    st.subheader("📋 Factores Técnicos Fundamentales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### ⚡ kVp (Kilovoltaje Pico)
        **Controla la CALIDAD del haz**
        
        - **Penetración**: Mayor kVp → mayor penetración
        - **Contraste**: Mayor kVp → menor contraste
        - **Dosis**: Mayor kVp → menor dosis (más eficiente)
        
        **Rango típico**: 40-150 kVp
        """)
        
    with col2:
        st.markdown("""
        ### 🔋 mAs (Miliamperios-segundo)
        **Controla la CANTIDAD de RX**
        
        - **Densidad**: Mayor mAs → mayor densidad
        - **Ruido**: Mayor mAs → menor ruido
        - **Dosis**: Mayor mAs → mayor dosis (proporcional)
        
        **Cálculo**: mAs = mA × tiempo (s)
        """)
        
    with col3:
        st.markdown("""
        ### 📏 DFI (Distancia Foco-Imagen)
        **Distancia tubo-receptor**
        
        - **Magnificación**: Menor DFI → mayor magnificación
        - **Definición**: Mayor DFI → mejor definición
        - **Intensidad**: Ley del cuadrado inverso
        
        **Estándar**: 100 cm (general), 180 cm (tórax)
        """)
    
    # ============================================
    # SECTION 2: Tabla de Técnicas
    # ============================================
    st.markdown("---")
    st.subheader("📊 Tabla de Técnicas Radiográficas")
    
    st.markdown("""
    Esta tabla muestra los **parámetros técnicos recomendados** para las proyecciones más comunes. 
    Son valores de referencia que deben ajustarse según el equipo, el paciente y el protocolo del centro.
    """)
    
    # Get technique chart
    df_techniques = get_technique_chart()
    
    # Add filter by body region
    col1, col2 = st.columns([1, 3])
    with col1:
        region_filter = st.selectbox(
            "Filtrar por región:",
            ["Todas", "Cráneo/Columna", "Tórax/Abdomen", "Extremidades Superiores", "Extremidades Inferiores"]
        )
    
    # Filter dataframe
    if region_filter == "Cráneo/Columna":
        df_filtered = df_techniques[df_techniques["Región Anatómica"].str.contains("Cráneo|Columna|Senos")]
    elif region_filter == "Tórax/Abdomen":
        df_filtered = df_techniques[df_techniques["Región Anatómica"].str.contains("Tórax|Abdomen|Pelvis|Parrilla")]
    elif region_filter == "Extremidades Superiores":
        df_filtered = df_techniques[df_techniques["Región Anatómica"].str.contains("Hombro|Húmero|Codo|Antebrazo|Muñeca|Mano")]
    elif region_filter == "Extremidades Inferiores":
        df_filtered = df_techniques[df_techniques["Región Anatómica"].str.contains("Cadera|Fémur|Rodilla|Tibia|Tobillo|Pie")]
    else:
        df_filtered = df_techniques
    
    # Display table
    st.dataframe(df_filtered, use_container_width=True, height=400)
    
    st.info("""
    💡 **Nota importante**: Estos valores son orientativos. Siempre consulta el protocolo específico 
    de tu centro y ajusta según:
    - Morfología del paciente (delgado, obeso, pediátrico)
    - Tipo de receptor (CR, DR, sensibilidad)
    - Estado del equipo (antigüedad, calibración)
    - Patología sospechada (puede requerir técnica especial)
    """)
    
    # ============================================
    # SECTION 3: Calculadoras de Conversión
    # ============================================
    st.markdown("---")
    st.subheader("🧮 Calculadoras de Conversión")
    
    calc_tabs = st.tabs([
        "📐 Regla del 15%", 
        "📏 Ley del Cuadrado Inverso", 
        "🔲 Factor de Rejilla",
        "👤 Morfología del Paciente"
    ])
    
    # --- Calculator 1: 15% Rule ---
    with calc_tabs[0]:
        st.markdown("""
        ### 📐 Regla del 15% (kVp ↔ mAs)
        
        **Principio físico**: Aumentar el kVp en un **15%** duplica la exposición del receptor de imagen, 
        lo que permite reducir el mAs **a la mitad** manteniendo la densidad óptica constante.
        
        **¿Cuándo usarla?**
        - ✅ Reducir dosis al paciente
        - ✅ Reducir tiempo de exposición (pacientes con movimiento)
        - ✅ Mejorar penetración en pacientes obesos
        - ⚠️ Cuidado: reduce el contraste de la imagen
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📥 Técnica Inicial")
            kvp_init = st.number_input("kVp inicial", 40, 150, 70, 5, key="kvp_15_init")
            mas_init = st.number_input("mAs inicial", 0.5, 500.0, 20.0, 0.5, key="mas_15_init")
            
            direction = st.radio(
                "Modificación:",
                ["Aumentar kVp (+15%)", "Disminuir kVp (-15%)"],
                key="direction_15"
            )
        
        with col2:
            st.markdown("#### 📤 Técnica Modificada")
            
            if "Aumentar" in direction:
                kvp_new, mas_new = calculate_15_percent_rule(kvp_init, mas_init, "increase")
                st.success(f"**kVp nuevo**: {kvp_new:.1f} kVp (+15%)")
                st.success(f"**mAs nuevo**: {mas_new:.1f} mAs (-50%)")
            else:
                kvp_new, mas_new = calculate_15_percent_rule(kvp_init, mas_init, "decrease")
                st.warning(f"**kVp nuevo**: {kvp_new:.1f} kVp (-15%)")
                st.warning(f"**mAs nuevo**: {mas_new:.1f} mAs (+100%)")
            
            # Calculate dose comparison
            dose_init = calculate_entrance_dose(kvp_init, mas_init)
            dose_new = calculate_entrance_dose(kvp_new, mas_new)
            
            st.metric("Dosis aproximada", f"{dose_new:.2f} mGy", 
                     delta=f"{((dose_new/dose_init - 1)*100):.1f}%")
        
        # Visualization
        st.markdown("#### 📊 Comparación Visual")
        
        # Bar chart comparison
        params = ['kVp', 'mAs', 'Dosis (mGy)']
        initial_values = [kvp_init, mas_init, dose_init]
        new_values = [kvp_new, mas_new, dose_new]
        
        fig_comparison = go.Figure()
        
        fig_comparison.add_trace(go.Bar(
            x=params,
            y=initial_values,
            name='Inicial',
            marker_color='#3498db',
            opacity=0.8
        ))
        
        fig_comparison.add_trace(go.Bar(
            x=params,
            y=new_values,
            name='Modificada',
            marker_color='#e74c3c',
            opacity=0.8
        ))
        
        fig_comparison.update_layout(
            title='Comparación de Parámetros',
            yaxis_title='Valor',
            barmode='group',
            height=400,
            hovermode='x unified',
            showlegend=True
        )
        
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Contrast comparison
        contrast_init = calculate_contrast_index(kvp_init)
        contrast_new = calculate_contrast_index(kvp_new)
        
        fig_contrast = go.Figure()
        
        fig_contrast.add_trace(go.Bar(
            y=['Inicial', 'Modificada'],
            x=[contrast_init, contrast_new],
            orientation='h',
            marker_color=['#3498db', '#e74c3c'],
            opacity=0.8,
            text=[f'{contrast_init:.1f}', f'{contrast_new:.1f}'],
            textposition='auto'
        ))
        
        fig_contrast.update_layout(
            title='Efecto en el Contraste',
            xaxis_title='Índice de Contraste (unidades arbitrarias)',
            xaxis_range=[0, 100],
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig_contrast, use_container_width=True)
        
        st.info("""
        💡 **Interpretación**:
        - Si aumentas kVp: ✅ Menos dosis, ✅ Menos mAs, ⚠️ Menos contraste
        - Si disminuyes kVp: ✅ Más contraste, ⚠️ Más dosis, ⚠️ Más mAs
        """)
    
    # --- Calculator 2: Inverse Square Law ---
    with calc_tabs[1]:
        st.markdown("""
        ### 📏 Ley del Cuadrado Inverso
        
        **Principio físico**: La intensidad de la radiación es **inversamente proporcional al cuadrado de la distancia**.
        
        $$I_1 / I_2 = (d_2 / d_1)^2$$
        
        **Aplicación práctica**: Si cambias la distancia foco-imagen (DFI), debes ajustar el mAs 
        para mantener la misma densidad en la imagen.
        
        **Fórmula de compensación**:
        $$mAs_2 = mAs_1 \\times (d_2 / d_1)^2$$
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📥 Condiciones Iniciales")
            dfi_init = st.number_input("DFI inicial (cm)", 50, 250, 100, 10, key="dfi_init")
            mas_init_dfi = st.number_input("mAs inicial", 0.5, 500.0, 20.0, 0.5, key="mas_dfi_init")
            
            st.markdown("#### 🎯 Nueva Distancia")
            dfi_new = st.number_input("DFI nueva (cm)", 50, 250, 180, 10, key="dfi_new")
        
        with col2:
            st.markdown("#### 📤 mAs Compensado")
            
            # Calculate new mAs
            mas_new_dfi = mas_init_dfi * (dfi_new / dfi_init) ** 2
            
            ratio = (dfi_new / dfi_init) ** 2
            
            if dfi_new > dfi_init:
                st.success(f"**mAs nuevo**: {mas_new_dfi:.1f} mAs")
                st.info(f"Aumenta mAs ×{ratio:.2f} (distancia mayor → menos intensidad)")
            elif dfi_new < dfi_init:
                st.warning(f"**mAs nuevo**: {mas_new_dfi:.1f} mAs")
                st.info(f"Reduce mAs ×{ratio:.2f} (distancia menor → más intensidad)")
            else:
                st.info("**Sin cambios** (misma distancia)")
            
            # Calculate relative intensity
            st.markdown("#### 💡 Intensidad Relativa")
            intensity_rel = inverse_square_law(100, dfi_init, dfi_new)
            st.metric("Intensidad", f"{intensity_rel:.1f}%", 
                     delta=f"{(intensity_rel - 100):.1f}%")
        
        # Visualization: Inverse square law curve
        st.markdown("#### 📊 Visualización de la Ley del Cuadrado Inverso")
        
        distances = np.linspace(50, 250, 100)
        intensities = inverse_square_law(100, 100, distances)
        required_mas = mas_init_dfi * (distances / dfi_init) ** 2
        
        # Create subplot with 2 columns
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            # Intensity vs distance
            fig_intensity = go.Figure()
            
            fig_intensity.add_trace(go.Scatter(
                x=distances,
                y=intensities,
                mode='lines',
                name='Intensidad relativa',
                line=dict(color='#3498db', width=3),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)'
            ))
            
            fig_intensity.add_vline(
                x=dfi_init,
                line_dash="dash",
                line_color="green",
                annotation_text=f"DFI inicial ({dfi_init} cm)"
            )
            
            fig_intensity.add_vline(
                x=dfi_new,
                line_dash="dash",
                line_color="red",
                annotation_text=f"DFI nueva ({dfi_new} cm)"
            )
            
            fig_intensity.add_hline(
                y=100,
                line_dash="dot",
                line_color="gray",
                opacity=0.5
            )
            
            fig_intensity.update_layout(
                title='Intensidad vs Distancia',
                xaxis_title='Distancia (cm)',
                yaxis_title='Intensidad Relativa (%)',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_intensity, use_container_width=True)
        
        with col_plot2:
            # Required mAs vs distance
            fig_mas = go.Figure()
            
            fig_mas.add_trace(go.Scatter(
                x=distances,
                y=required_mas,
                mode='lines',
                name='mAs requerido',
                line=dict(color='#e74c3c', width=3),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.2)'
            ))
            
            fig_mas.add_vline(
                x=dfi_init,
                line_dash="dash",
                line_color="green"
            )
            
            fig_mas.add_vline(
                x=dfi_new,
                line_dash="dash",
                line_color="red"
            )
            
            fig_mas.add_scatter(
                x=[dfi_init, dfi_new],
                y=[mas_init_dfi, mas_new_dfi],
                mode='markers',
                marker=dict(size=12, color=['green', 'red'], 
                           line=dict(color='white', width=2)),
                name='Puntos actuales',
                showlegend=False
            )
            
            fig_mas.update_layout(
                title='Compensación de mAs según Distancia',
                xaxis_title='Distancia (cm)',
                yaxis_title='mAs necesario',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_mas, use_container_width=True)
        
        st.success("""
        🎯 **Ejemplo práctico**: 
        - Radiografía de tórax PA: DFI = 180 cm (reduce magnificación cardiaca)
        - Radiografía general: DFI = 100 cm (estándar)
        - Si cambias de 100→180 cm: mAs debe multiplicarse por (180/100)² = **3.24 veces**
        """)
    
    # --- Calculator 3: Grid Factor ---
    with calc_tabs[2]:
        st.markdown("""
        ### 🔲 Factor de Conversión de Rejilla
        
        Las **rejillas antidifusión** eliminan radiación dispersa mejorando el contraste, 
        pero absorben también radiación primaria, requiriendo **aumentar el mAs**.
        
        **Ratio de rejilla**: Relación altura/separación de las láminas (ej: 10:1, 12:1)
        - Mayor ratio → elimina más dispersión → mejor contraste → requiere más mAs
        
        **Factor Bucky**: Factor multiplicador del mAs al usar rejilla
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📥 Técnica Sin Rejilla")
            mas_no_grid = st.number_input("mAs sin rejilla", 0.5, 200.0, 10.0, 0.5, key="mas_no_grid")
            
            st.markdown("#### 🔲 Seleccionar Rejilla")
            grid_ratio = st.selectbox(
                "Ratio de rejilla:",
                ["Sin rejilla", "5:1", "6:1", "8:1", "10:1", "12:1", "16:1"],
                index=4,
                key="grid_ratio"
            )
        
        with col2:
            st.markdown("#### 📤 mAs Con Rejilla")
            
            grid_factor = calculate_grid_conversion_factor(grid_ratio)
            mas_with_grid = mas_no_grid * grid_factor
            
            if grid_ratio != "Sin rejilla":
                st.success(f"**Factor de conversión**: {grid_factor}×")
                st.success(f"**mAs con rejilla**: {mas_with_grid:.1f} mAs")
                st.info(f"Debes aumentar el mAs ×{grid_factor} al usar rejilla {grid_ratio}")
            else:
                st.info("Sin rejilla seleccionada")
            
            # Dose comparison
            st.markdown("#### ⚠️ Impacto en Dosis")
            dose_no_grid = mas_no_grid * 0.1  # Arbitrary units
            dose_with_grid = mas_with_grid * 0.1
            
            st.metric("Dosis al paciente", f"{dose_with_grid:.1f} u.a.", 
                     delta=f"+{((grid_factor - 1) * 100):.0f}%")
        
        # Grid comparison table
        st.markdown("#### 📊 Tabla de Factores de Rejilla")
        
        grid_data = {
            "Ratio Rejilla": ["Sin rejilla", "5:1", "6:1", "8:1", "10:1", "12:1", "16:1"],
            "Factor Bucky": [1, 2, 3, 4, 5, 5, 6],
            "Frecuencia típica": ["-", "25-40 líneas/cm", "40 líneas/cm", "40 líneas/cm", 
                                 "40-60 líneas/cm", "60-70 líneas/cm", "70-80 líneas/cm"],
            "Aplicación": ["Extremidades", "Portátiles", "General", "General/Mesa", 
                          "General/Bucky", "Bucky/Alta calidad", "Alta energía"]
        }
        
        df_grid = pd.DataFrame(grid_data)
        st.dataframe(df_grid, use_container_width=True)
        
        # Visualization
        fig_grid = go.Figure()
        
        ratios = ["Sin rejilla", "5:1", "6:1", "8:1", "10:1", "12:1", "16:1"]
        factors = [1, 2, 3, 4, 5, 5, 6]
        colors_grid = ['#2ecc71' if r == grid_ratio else '#3498db' for r in ratios]
        
        fig_grid.add_trace(go.Bar(
            x=ratios,
            y=factors,
            marker_color=colors_grid,
            opacity=0.8,
            text=factors,
            texttemplate='× %{text}',
            textposition='outside',
            textfont=dict(size=14, color='white')
        ))
        
        fig_grid.update_layout(
            title='Factores de Conversión por Tipo de Rejilla',
            xaxis_title='Ratio de Rejilla',
            yaxis_title='Factor de Conversión (mAs)',
            yaxis_range=[0, 7],
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig_grid, use_container_width=True)
        
        st.warning("""
        ⚠️ **Consideraciones clínicas**:
        - **Sin rejilla**: Solo para extremidades finas (< 10 cm grosor)
        - **8:1 a 10:1**: Estándar para radiografía general
        - **12:1 a 16:1**: Alta energía (> 100 kVp), cuerpo grueso
        - **Móviles**: Rejillas de bajo ratio (5:1, 6:1) o sin rejilla
        """)
    
    # --- Calculator 4: Body Habitus ---
    with calc_tabs[3]:
        st.markdown("""
        ### 👤 Ajuste por Morfología del Paciente
        
        La **morfología del paciente** (habitus corporal) afecta significativamente la atenuación 
        del haz de rayos X. Es necesario ajustar los parámetros técnicos según el grosor y 
        composición corporal.
        
        **Factores a considerar**:
        - **Grosor del paciente**: A mayor grosor → más atenuación → más mAs
        - **Composición**: Músculo atenúa más que grasa
        - **Edad**: Pediátricos requieren técnicas significativamente menores
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📥 Técnica Base (Adulto Normal)")
            kvp_base = st.number_input("kVp base", 40, 150, 75, 5, key="kvp_habitus")
            mas_base = st.number_input("mAs base", 0.5, 200.0, 25.0, 0.5, key="mas_habitus")
            
            st.markdown("#### 👤 Morfología del Paciente")
            habitus = st.selectbox(
                "Seleccionar tipo:",
                ["Pediátrico (< 5 años)", "Niño (5-12 años)", "Adolescente",
                 "Adulto asténico (delgado)", "Adulto hiposténico", "Adulto esténico (normal)",
                 "Adulto hiperesténico", "Adulto obeso", "Adulto obeso mórbido"],
                index=5,
                key="habitus_type"
            )
        
        with col2:
            st.markdown("#### 📤 Técnica Ajustada")
            
            habitus_factor = body_habitus_factor(habitus)
            mas_adjusted = mas_base * habitus_factor
            
            # Show adjustment
            if habitus_factor < 1:
                st.success(f"**Factor de ajuste**: {habitus_factor}× (reducir)")
                st.success(f"**kVp**: {kvp_base} kVp (sin cambio)")
                st.success(f"**mAs ajustado**: {mas_adjusted:.1f} mAs")
            elif habitus_factor > 1:
                st.warning(f"**Factor de ajuste**: {habitus_factor}× (aumentar)")
                st.warning(f"**kVp**: {kvp_base} kVp (considerar +10-15%)")
                st.warning(f"**mAs ajustado**: {mas_adjusted:.1f} mAs")
            else:
                st.info("**Sin ajuste** (técnica estándar)")
            
            # Calculate dose
            st.markdown("#### 💊 Dosis Estimada")
            dose_base = calculate_entrance_dose(kvp_base, mas_base)
            dose_adjusted = calculate_entrance_dose(kvp_base, mas_adjusted)
            
            st.metric("Dosis de entrada", f"{dose_adjusted:.2f} mGy",
                     delta=f"{((habitus_factor - 1) * 100):.0f}%")
        
        # Visualization
        st.markdown("#### 📊 Factores de Ajuste por Morfología")
        
        habitus_types = ["Pediátrico\n(< 5 años)", "Niño\n(5-12 años)", "Adolescente",
                        "Asténico", "Hiposténico", "Esténico\n(normal)",
                        "Hiperesténico", "Obeso", "Obeso\nmórbido"]
        factors_all = [0.25, 0.5, 0.75, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]
        colors_habitus = ['#3498db' if f <= 1 else '#e74c3c' for f in factors_all]
        
        # Highlight selected
        habitus_map = {
            "Pediátrico (< 5 años)": 0, "Niño (5-12 años)": 1, "Adolescente": 2,
            "Adulto asténico (delgado)": 3, "Adulto hiposténico": 4, "Adulto esténico (normal)": 5,
            "Adulto hiperesténico": 6, "Adulto obeso": 7, "Adulto obeso mórbido": 8
        }
        
        if habitus in habitus_map:
            colors_habitus[habitus_map[habitus]] = '#2ecc71'
        
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            # Bar chart of all habitus factors
            fig_habitus_bar = go.Figure()
            
            fig_habitus_bar.add_trace(go.Bar(
                y=habitus_types,
                x=factors_all,
                orientation='h',
                marker_color=colors_habitus,
                opacity=0.8,
                text=[f'{f}×' for f in factors_all],
                textposition='auto'
            ))
            
            fig_habitus_bar.add_vline(
                x=1.0,
                line_dash="dash",
                line_color="gray",
                line_width=2,
                annotation_text="Estándar (1.0×)"
            )
            
            fig_habitus_bar.update_layout(
                title='Factores de Ajuste por Tipo de Paciente',
                xaxis_title='Factor de Conversión',
                xaxis_range=[0, 2.2],
                height=450,
                showlegend=False
            )
            
            st.plotly_chart(fig_habitus_bar, use_container_width=True)
        
        with col_plot2:
            # mAs comparison
            mas_values = [mas_base * f for f in factors_all]
            
            fig_mas_comp = go.Figure()
            
            fig_mas_comp.add_trace(go.Scatter(
                x=factors_all,
                y=mas_values,
                mode='lines+markers',
                name='mAs requerido',
                line=dict(color='#3498db', width=3),
                marker=dict(size=8)
            ))
            
            fig_mas_comp.add_scatter(
                x=[habitus_factor],
                y=[mas_adjusted],
                mode='markers',
                marker=dict(size=15, color='#e74c3c', 
                           line=dict(color='white', width=3)),
                name='Selección actual',
                showlegend=False
            )
            
            fig_mas_comp.add_hline(
                y=mas_base,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Base ({mas_base} mAs)"
            )
            
            fig_mas_comp.update_layout(
                title='Relación Factor-mAs',
                xaxis_title='Factor de Morfología',
                yaxis_title='mAs Requerido',
                height=450,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_mas_comp, use_container_width=True)
        
        # Clinical recommendations
        st.markdown("#### 🏥 Recomendaciones Clínicas")
        
        recommendations = {
            "Pediátrico (< 5 años)": {
                "icon": "👶",
                "advice": "Técnica muy baja. Priorizar velocidad (movimiento). Considerar inmovilización.",
                "kvp": "Reducir 10-15 kVp respecto adulto",
                "safety": "⚠️ Extremar ALARA - tejidos en desarrollo"
            },
            "Niño (5-12 años)": {
                "icon": "🧒",
                "advice": "Técnica reducida. Explicar procedimiento para colaboración.",
                "kvp": "Reducir 5-10 kVp respecto adulto",
                "safety": "⚠️ Alta radiosensibilidad"
            },
            "Adolescente": {
                "icon": "👦",
                "advice": "Técnica ligeramente reducida. Similar a adulto delgado.",
                "kvp": "Similar a adulto",
                "safety": "⚠️ Proteger gónadas si aplica"
            },
            "Adulto asténico (delgado)": {
                "icon": "🧍",
                "advice": "Reducir técnica. Buen contraste natural por poco tejido blando.",
                "kvp": "Estándar o -5 kVp",
                "safety": "✅ Menor dosis necesaria"
            },
            "Adulto hiposténico": {
                "icon": "🧍",
                "advice": "Técnica estándar con ligera reducción.",
                "kvp": "Estándar",
                "safety": "✅ Técnica habitual"
            },
            "Adulto esténico (normal)": {
                "icon": "🧍‍♂️",
                "advice": "Técnica estándar de referencia.",
                "kvp": "Estándar según tabla",
                "safety": "✅ Protocolo estándar"
            },
            "Adulto hiperesténico": {
                "icon": "🧍‍♂️",
                "advice": "Aumentar técnica. Considerar aumento de kVp además de mAs.",
                "kvp": "+5 a +10 kVp",
                "safety": "⚠️ Optimizar kVp para reducir dosis"
            },
            "Adulto obeso": {
                "icon": "🧍‍♂️",
                "advice": "Aumentar significativamente. Preferir aumento de kVp (15%) antes que mAs.",
                "kvp": "+15 a +20 kVp",
                "safety": "⚠️ Alto riesgo de dosis elevada"
            },
            "Adulto obeso mórbido": {
                "icon": "🧍‍♂️",
                "advice": "Técnica muy alta. Considerar técnicas alternativas (TC si disponible).",
                "kvp": "+20 a +30 kVp",
                "safety": "⚠️⚠️ Riesgo muy alto - Evaluar beneficio/riesgo"
            }
        }
        
        if habitus in recommendations:
            rec = recommendations[habitus]
            
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"### {rec['icon']}")
            with col2:
                st.info(f"**{habitus}**\n\n{rec['advice']}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Ajuste de kVp**: {rec['kvp']}")
            with col2:
                st.markdown(f"**Consideración de seguridad**: {rec['safety']}")
    
    # ============================================
    # SECTION 4: Control Automático de Exposición (AEC)
    # ============================================
    st.markdown("---")
    st.subheader("🤖 Control Automático de Exposición (AEC/Phototimer)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        El **AEC (Automatic Exposure Control)** o **Phototimer** es un sistema que mide automáticamente 
        la cantidad de radiación que atraviesa el paciente y **detiene la exposición** cuando se alcanza 
        la densidad óptima.
        
        **Ventajas**:
        - ✅ Densidad consistente independiente de la morfología del paciente
        - ✅ Reduce errores del operador
        - ✅ Optimiza la dosis (no sobreexpone)
        - ✅ Mejora el flujo de trabajo (menos repeticiones)
        
        **Componentes**:
        - **Cámaras de ionización**: Detectores bajo la mesa (generalmente 3)
        - **Circuito de control**: Analiza la señal y corta el disparo
        - **Selector de cámaras**: Permite elegir qué cámaras usar
        """)
    
    with col2:
        st.image("https://via.placeholder.com/300x300.png?text=AEC+Chambers", 
                caption="Disposición típica de cámaras AEC", use_container_width=True)
        # En tu implementación real, reemplaza con una imagen real de las cámaras AEC
    
    # AEC Chamber Selection Simulator
    st.markdown("#### 🎯 Simulador de Selección de Cámaras")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        region_aec = st.selectbox(
            "Región a radiografiar:",
            ["Tórax PA", "Tórax Lateral", "Columna Lumbar AP", "Columna Lumbar Lateral",
             "Abdomen AP", "Pelvis AP"],
            key="region_aec"
        )
        
        st.markdown("**Selecciona cámaras activas:**")
        camera_left = st.checkbox("🟦 Cámara Izquierda", value=True, key="cam_left")
        camera_center = st.checkbox("🟨 Cámara Central", value=True, key="cam_center")
        camera_right = st.checkbox("🟥 Cámara Derecha", value=True, key="cam_right")
    
    with col2:
        st.markdown("#### 🎯 Diagrama de Cámaras")
        
        # Create simple representation
        camera_status = []
        if camera_left:
            camera_status.append("🟦 Izquierda: ACTIVA")
        else:
            camera_status.append("⬜ Izquierda: Inactiva")
            
        if camera_center:
            camera_status.append("🟨 Central: ACTIVA")
        else:
            camera_status.append("⬜ Central: Inactiva")
            
        if camera_right:
            camera_status.append("🟥 Derecha: ACTIVA")
        else:
            camera_status.append("⬜ Derecha: Inactiva")
        
        st.info(f"**Región: {region_aec}**\n\n" + "\n\n".join(camera_status))
        
        st.markdown("```")
        st.markdown("     Vista desde el tubo de RX")
        st.markdown("   ┌─────────────────────────┐")
        st.markdown(f"   │  {'🟦' if camera_left else '⬜'}      {'🟨' if camera_center else '⬜'}      {'🟥' if camera_right else '⬜'}  │")
        st.markdown("   │   (I)    (C)    (D)     │")
        st.markdown("   │                         │")
        st.markdown("   │      DETECTOR           │")
        st.markdown("   └─────────────────────────┘")
        st.markdown("```")
    
    # AEC Recommendations by region
    aec_recommendations = {
        "Tórax PA": {
            "cameras": "Izquierda + Derecha (ambos pulmones)",
            "avoid": "⚠️ NO usar cámara central (mediastino muy denso)",
            "kvp": "120-125 kVp",
            "tips": "Asegurar simetría del paciente. Centrar el tórax."
        },
        "Tórax Lateral": {
            "cameras": "Central (o combinación según equipo)",
            "avoid": "✅ Verificar que brazos estén elevados",
            "kvp": "120-125 kVp",
            "tips": "Mayor mAs necesario que PA. Verificar campo."
        },
        "Columna Lumbar AP": {
            "cameras": "Las 3 cámaras",
            "avoid": "⚠️ Verificar centrado (no debe salirse del campo)",
            "kvp": "75-85 kVp",
            "tips": "Considerar morfología. Obesos pueden requerir +15 kVp."
        },
        "Columna Lumbar Lateral": {
            "cameras": "Central",
            "avoid": "⚠️ Difícil con AEC - considerar técnica manual en obesos",
            "kvp": "85-95 kVp",
            "tips": "Flexionar rodillas. Zona muy densa."
        },
        "Abdomen AP": {
            "cameras": "Las 3 cámaras",
            "avoid": "✅ Verificar que vejiga esté vacía si es posible",
            "kvp": "75-80 kVp",
            "tips": "Exposición al final de espiración."
        },
        "Pelvis AP": {
            "cameras": "Las 3 cámaras",
            "avoid": "✅ Rotación interna de pies",
            "kvp": "75-80 kVp",
            "tips": "Densidad homogénea - funciona bien con AEC."
        }
    }
    
    if region_aec in aec_recommendations:
        rec_aec = aec_recommendations[region_aec]
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Cámaras recomendadas**: {rec_aec['cameras']}")
            st.info(f"**kVp sugerido**: {rec_aec['kvp']}")
        with col2:
            st.warning(rec_aec['avoid'])
            st.markdown(f"💡 **Consejo**: {rec_aec['tips']}")
    
    # AEC Common Errors
    with st.expander("⚠️ Errores Comunes con AEC y Cómo Evitarlos"):
        st.markdown("""
        ### Problemas Frecuentes:
        
        #### 1. **Imagen Sobreexpuesta (muy oscura en film)**
        **Causas**:
        - Cámaras mal seleccionadas (detectan zona muy radiolúcida)
        - Colimación excesiva (cámaras parcialmente fuera del campo)
        - Paciente no centrado correctamente
        
        **Solución**: Verificar cámaras activas, centrado y colimación.
        
        ---
        
        #### 2. **Imagen Subexpuesta (muy clara)**
        **Causas**:
        - Cámaras detectan zona muy densa (ej: prótesis metálica)
        - Material radiopaco sobre las cámaras
        - kVp insuficiente (AEC llega a tiempo máximo sin densidad adecuada)
        
        **Solución**: Desactivar cámara sobre prótesis, aumentar kVp, verificar que no hay objetos sobre el detector.
        
        ---
        
        #### 3. **Disparo Mínimo (imagen inmediata)**
        **Causas**:
        - No hay paciente en la mesa
        - Chasis/detector no insertado correctamente
        - Cámaras fuera del campo de colimación
        
        **Solución**: Verificar presencia del paciente, correcta inserción del receptor, colimación adecuada.
        
        ---
        
        #### 4. **Tiempo Máximo Alcanzado**
        **Causas**:
        - kVp demasiado bajo para el grosor del paciente
        - mA muy bajo
        - Paciente muy obeso sin ajuste de técnica
        
        **Solución**: Aumentar kVp (+15%), verificar mA, considerar técnica manual en casos extremos.
        
        ---
        
        ### ✅ Reglas de Oro del AEC:
        
        1. **Selección de cámaras**: Deben estar bajo la anatomía de interés
        2. **Centrado**: El paciente debe estar perfectamente centrado
        3. **Colimación**: Las cámaras activas deben estar dentro del campo
        4. **kVp adecuado**: Debe ser suficiente para penetrar al paciente
        5. **Densidad/Backup time**: Configurar tiempo máximo de seguridad (3-5 segundos típico)
        6. **Material sobre detector**: Eliminar ropa con metales, cables, etc.
        """)
    
    # ============================================
    # SECTION 5: Optimización de Calidad de Imagen
    # ============================================
    st.markdown("---")
    st.subheader("📈 Optimización de Calidad: SNR y CNR")
    
    st.markdown("""
    La calidad de una imagen radiográfica digital se evalúa principalmente mediante:
    
    - **SNR (Signal-to-Noise Ratio)**: Relación entre señal útil y ruido estadístico
    - **CNR (Contrast-to-Noise Ratio)**: Capacidad de distinguir estructuras diferentes
    - **Resolución espacial**: Capacidad de ver detalles finos
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Parámetros de Simulación")
        kvp_snr = st.slider("kVp", 40, 150, 75, 5, key="kvp_snr")
        mas_snr = st.slider("mAs", 1, 100, 20, 1, key="mas_snr")
        thickness = st.slider("Grosor del paciente (cm)", 5, 40, 20, 1, key="thickness_snr")
        
        # Calculate SNR and CNR
        snr, cnr = calculate_snr_cnr(kvp_snr, mas_snr, thickness)
        
    with col2:
        st.markdown("#### 📈 Métricas de Calidad")
        
        # SNR metric
        snr_color = "normal"
        if snr < 20:
            snr_color = "off"
            snr_quality = "⚠️ Bajo - Imagen ruidosa"
        elif snr < 40:
            snr_color = "normal"
            snr_quality = "✅ Aceptable"
        else:
            snr_quality = "✅ Excelente"
        
        st.metric("SNR (Relación Señal-Ruido)", f"{snr:.1f}", help="Mayor es mejor. >30 es óptimo")
        st.caption(snr_quality)
        
        # CNR metric
        if cnr < 5:
            cnr_quality = "⚠️ Bajo contraste"
        elif cnr < 10:
            cnr_quality = "✅ Contraste adecuado"
        else:
            cnr_quality = "✅ Alto contraste"
        
        st.metric("CNR (Contraste-Ruido)", f"{cnr:.1f}", help="Mayor es mejor. >8 es óptimo")
        st.caption(cnr_quality)
        
        # Dose estimation
        dose_snr = calculate_entrance_dose(kvp_snr, mas_snr)
        st.metric("Dosis estimada", f"{dose_snr:.2f} mGy")
    
    # Interactive plot: SNR vs mAs
    st.markdown("#### 📊 Efecto de los Parámetros en SNR/CNR")
    
    tab_snr1, tab_snr2 = st.tabs(["SNR vs mAs", "CNR vs kVp"])
    
    with tab_snr1:
        # SNR increases with sqrt(mAs)
        mas_range = np.linspace(1, 100, 50)
        snr_range = [calculate_snr_cnr(kvp_snr, m, thickness)[0] for m in mas_range]
        dose_range = [calculate_entrance_dose(kvp_snr, m) for m in mas_range]
        
        col_snr1, col_snr2 = st.columns(2)
        
        with col_snr1:
            # SNR vs mAs
            fig_snr = go.Figure()
            
            fig_snr.add_trace(go.Scatter(
                x=mas_range,
                y=snr_range,
                mode='lines',
                name='SNR',
                line=dict(color='#3498db', width=3),
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.2)'
            ))
            
            fig_snr.add_hline(
                y=30,
                line_dash="dash",
                line_color="green",
                annotation_text="Objetivo (SNR=30)"
            )
            
            fig_snr.add_vline(
                x=mas_snr,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Actual ({mas_snr} mAs)"
            )
            
            fig_snr.add_scatter(
                x=[mas_snr],
                y=[snr],
                mode='markers',
                marker=dict(size=15, color='red', 
                           line=dict(color='white', width=2)),
                showlegend=False
            )
            
            fig_snr.update_layout(
                title='SNR vs mAs (√mAs)',
                xaxis_title='mAs',
                yaxis_title='SNR',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_snr, use_container_width=True)
        
        with col_snr2:
            # Dose vs mAs (linear)
            fig_dose = go.Figure()
            
            fig_dose.add_trace(go.Scatter(
                x=mas_range,
                y=dose_range,
                mode='lines',
                name='Dosis',
                line=dict(color='#e74c3c', width=3),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.2)'
            ))
            
            fig_dose.add_vline(
                x=mas_snr,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Actual ({mas_snr} mAs)"
            )
            
            fig_dose.add_scatter(
                x=[mas_snr],
                y=[dose_snr],
                mode='markers',
                marker=dict(size=15, color='red', 
                           line=dict(color='white', width=2)),
                showlegend=False
            )
            
            fig_dose.update_layout(
                title='Dosis vs mAs (Lineal)',
                xaxis_title='mAs',
                yaxis_title='Dosis (mGy)',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_dose, use_container_width=True)
        
        st.info("""
        💡 **Interpretación**:
        - SNR aumenta con √mAs → **duplicar mAs mejora SNR en 41%**
        - Dosis aumenta linealmente con mAs → **duplicar mAs duplica la dosis**
        - Compromiso: Buscar mínimo mAs que dé SNR aceptable (>30)
        """)
    
    with tab_snr2:
        # CNR decreases with kVp (less contrast)
        kvp_range = np.linspace(50, 130, 50)
        cnr_range = [calculate_snr_cnr(k, mas_snr, thickness)[1] for k in kvp_range]
        contrast_range = [calculate_contrast_index(k) for k in kvp_range]
        
        col_cnr1, col_cnr2 = st.columns(2)
        
        with col_cnr1:
            # CNR vs kVp
            fig_cnr = go.Figure()
            
            fig_cnr.add_trace(go.Scatter(
                x=kvp_range,
                y=cnr_range,
                mode='lines',
                name='CNR',
                line=dict(color='#2ecc71', width=3),
                fill='tozeroy',
                fillcolor='rgba(46, 204, 113, 0.2)'
            ))
            
            fig_cnr.add_hline(
                y=8,
                line_dash="dash",
                line_color="green",
                annotation_text="Objetivo (CNR=8)"
            )
            
            fig_cnr.add_vline(
                x=kvp_snr,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Actual ({kvp_snr} kVp)"
            )
            
            fig_cnr.add_scatter(
                x=[kvp_snr],
                y=[cnr],
                mode='markers',
                marker=dict(size=15, color='red', 
                           line=dict(color='white', width=2)),
                showlegend=False
            )
            
            fig_cnr.update_layout(
                title='CNR vs kVp',
                xaxis_title='kVp',
                yaxis_title='CNR',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_cnr, use_container_width=True)
        
        with col_cnr2:
            # Contrast Index vs kVp
            fig_contrast_idx = go.Figure()
            
            fig_contrast_idx.add_trace(go.Scatter(
                x=kvp_range,
                y=contrast_range,
                mode='lines',
                name='Índice de Contraste',
                line=dict(color='#f39c12', width=3),
                fill='tozeroy',
                fillcolor='rgba(243, 156, 18, 0.2)'
            ))
            
            fig_contrast_idx.add_vline(
                x=kvp_snr,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Actual ({kvp_snr} kVp)"
            )
            
            current_contrast = calculate_contrast_index(kvp_snr)
            fig_contrast_idx.add_scatter(
                x=[kvp_snr],
                y=[current_contrast],
                mode='markers',
                marker=dict(size=15, color='red', 
                           line=dict(color='white', width=2)),
                showlegend=False
            )
            
            fig_contrast_idx.update_layout(
                title='Contraste vs kVp',
                xaxis_title='kVp',
                yaxis_title='Índice de Contraste',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_contrast_idx, use_container_width=True)
        
        st.warning("""
        ⚠️ **Interpretación**:
        - Mayor kVp → **menor contraste** (más penetración, menos absorción diferencial)
        - Menor kVp → **mayor contraste** pero mayor dosis y menos penetración
        - **Regla del 15%**: Subir kVp 15% permite bajar mAs a la mitad (reducir dosis 30%)
        - Compromiso: Elegir kVp que dé contraste adecuado con mínima dosis
        """)
    
    # ============================================
    # SECTION 6: Casos Prácticos
    # ============================================
    st.markdown("---")
    st.subheader("🎓 Casos Prácticos y Ejercicios")
    
    caso_tabs = st.tabs(["Caso 1: Tórax", "Caso 2: Lumbar", "Caso 3: Pediátrico", "Caso 4: Obeso"])
    
    with caso_tabs[0]:
        st.markdown("""
        ### 📋 Caso Clínico 1: Radiografía de Tórax PA
        
        **Escenario**:
        Paciente adulto, 70 kg, complexión normal, sin patología respiratoria conocida.
        Tu equipo tiene:
        - AEC disponible (3 cámaras)
        - Rejilla 12:1 en el Bucky
        - DFI estándar: 180 cm
        
        **Técnica habitual**: 125 kVp, AEC, cámaras laterales
        
        **Problema**: El paciente tiene marcapasos metálico en región superior izquierda.
        """)
        
        st.markdown("#### ❓ Preguntas:")
        
        q1 = st.radio(
            "1. ¿Qué cámaras AEC deberías usar?",
            ["Izquierda + Central", "Derecha + Central", "Solo Derecha", "Las 3 cámaras"],
            key="q1_caso1"
        )
        
        if q1:
            if q1 == "Solo Derecha":
                st.success("✅ **Correcto**: Usa solo la cámara derecha para evitar que el marcapasos cause subexposición.")
            elif q1 == "Derecha + Central":
                st.warning("⚠️ Parcialmente correcto, pero el mediastino (central) es muy denso para tórax PA.")
            else:
                st.error("❌ Incorrecto: El marcapasos en la izquierda causaría lectura incorrecta.")
        
        q2 = st.radio(
            "2. ¿Necesitas ajustar el kVp?",
            ["No, 125 kVp es correcto", "Sí, reducir a 110 kVp", "Sí, aumentar a 140 kVp"],
            key="q2_caso1"
        )
        
        if q2:
            if q2 == "No, 125 kVp es correcto":
                st.success("✅ **Correcto**: 120-125 kVp es el estándar para tórax PA (bajo contraste necesario).")
            else:
                st.error("❌ El kVp estándar para tórax es adecuado.")
        
        with st.expander("📖 Ver Explicación Completa"):
            st.markdown("""
            **Análisis del caso**:
            
            1. **Selección de cámaras**: 
               - El marcapasos es muy denso (metal) y bloqueará muchos fotones
               - Si la cámara izquierda está activa, recibirá menos radiación de la esperada
               - El AEC pensará que necesita más exposición → **sobreexposición**
               - Solución: **Desactivar cámara izquierda**, usar solo la derecha
            
            2. **kVp para tórax**:
               - Tórax requiere **alto kVp (120-125 kVp)** para:
                 - Penetrar estructuras mediastínicas densas (corazón, vasos)
                 - Conseguir **bajo contraste** (escala de grises larga)
                 - Visualizar tanto parénquima pulmonar como mediastino
                 - Reducir dosis al paciente (más eficiente)
               - NO reducir kVp: aumentaría contraste excesivo y dosis
            
            3. **DFI 180 cm**:
               - Reduce magnificación cardiaca (importante para valorar índice cardiotorácico)
               - Mejora nitidez (menor penumbra geométrica)
            
            4. **Precauciones adicionales**:
               - Marcar la presencia del marcapasos en la solicitud
               - Informar al radiólogo de su ubicación
               - Evitar colimación excesiva que deje cámaras fuera del campo
            """)
    
    with caso_tabs[1]:
        st.markdown("""
        ### 📋 Caso Clínico 2: Columna Lumbar Lateral
        
        **Escenario**:
        Paciente varón, 95 kg, IMC 32 (obesidad grado I), dolor lumbar crónico.
        
        **Técnica estándar**: 90 kVp, 80 mAs, rejilla 10:1, DFI 100 cm
        
        **Primera exposición**: Imagen muy subexpuesta (clara), técnica insuficiente.
        """)
        
        st.markdown("#### ❓ Preguntas:")
        
        q1_c2 = st.radio(
            "1. ¿Qué modificación harías primero?",
            ["Aumentar mAs a 160 mAs (×2)", 
             "Aumentar kVp a 104 kVp (+15%)", 
             "Ambas: +15% kVp y +50% mAs",
             "Cambiar a rejilla 6:1"],
            key="q1_caso2"
        )
        
        if q1_c2:
            if q1_c2 == "Aumentar kVp a 104 kVp (+15%)":
                st.success("✅ **Correcto**: Aplicar regla del 15% es más eficiente. Aumenta penetración y permite reducir mAs.")
            elif q1_c2 == "Ambas: +15% kVp y +50% mAs":
                st.warning("⚠️ Podría funcionar, pero aumenta dosis innecesariamente. Prueba primero solo kVp.")
            elif q1_c2 == "Aumentar mAs a 160 mAs (×2)":
                st.error("❌ Funcionaría pero DUPLICA la dosis. Mejor aumentar kVp primero (regla 15%).")
            else:
                st.error("❌ Cambiar rejilla es complejo. Ajusta primero kVp/mAs.")
        
        q2_c2 = st.radio(
            "2. Si aumentas kVp de 90 a 104 kVp (+15%), ¿cuánto mAs necesitas?",
            ["40 mAs (la mitad)", "80 mAs (igual)", "120 mAs (+50%)"],
            key="q2_caso2"
        )
        
        if q2_c2:
            if q2_c2 == "40 mAs (la mitad)":
                st.success("✅ **Correcto**: Regla del 15% → +15% kVp = duplica exposición → mAs a la mitad.")
            else:
                st.error("❌ Revisa la regla del 15%: +15% kVp duplica la exposición del receptor.")
        
        with st.expander("📖 Ver Solución Completa"):
            st.markdown("""
            **Análisis del caso**:
            
            **Problema**: Paciente obeso → mayor atenuación → técnica estándar insuficiente
            
            **Solución paso a paso**:
            
            1. **Primera opción (RECOMENDADA)**: Aumentar kVp
               - De 90 kVp → 104 kVp (+15%)
               - Permite reducir mAs de 80 → 40 mAs
               - Mayor penetración del haz
               - **Dosis neta**: Reducción ~30%
               - Contraste ligeramente menor (aceptable en lumbar)
            
            2. **Segunda opción**: Aumentar solo mAs
               - De 80 → 160 mAs (×2)
               - Mantener 90 kVp
               - Mayor contraste
               - **Dosis**: Duplicada ⚠️
               - Mayor carga térmica del tubo
            
            3. **Opción óptima para obesos**:
               - kVp: 90 → 104 kVp (+15%)
               - mAs: Ajustar por morfología (~100 mAs con factor 1.5×)
               - Resultado: Buena penetración + dosis controlada
            
            **Cálculos**:
            ```
            Técnica estándar: 90 kVp, 80 mAs
            Factor obesidad: 1.5× (paciente 95kg, obeso)
            
            Opción A (solo mAs):
            - 90 kVp, 120 mAs (80 × 1.5)
            - Dosis: +50%
            
            Opción B (Regla 15%):
            - 104 kVp (+15%), 60 mAs (40 base × 1.5 morfología)
            - Dosis: +12% respecto estándar
            
            Opción B es claramente superior ✅
            ```
            
            **Consideraciones adicionales**:
            - Verificar límites térmicos del tubo
            - Lumbar lateral es proyección muy exigente
            - En obesos mórbidos, considerar TC si disponible
            - Flexión de rodillas reduce grosor del paciente
            """)
    
    with caso_tabs[2]:
        st.markdown("""
        ### 📋 Caso Clínico 3: Radiografía de Abdomen Pediátrico
        
        **Escenario**:
        Niño de 4 años, 18 kg, sospecha de obstrucción intestinal.
        
        **Técnica adulto**: 75 kVp, 25 mAs, rejilla 10:1
        
        **Pregunta**: ¿Cómo adaptas la técnica?
        """)
        
        st.markdown("#### ❓ Preguntas:")
        
        q1_c3 = st.radio(
            "1. Ajuste de kVp:",
            ["Mantener 75 kVp", "Reducir a 65-70 kVp", "Aumentar a 85 kVp"],
            key="q1_caso3"
        )
        
        if q1_c3:
            if q1_c3 == "Reducir a 65-70 kVp":
                st.success("✅ **Correcto**: Menor grosor del paciente requiere menos penetración.")
            else:
                st.error("❌ Niños requieren kVp menor por su menor grosor corporal.")
        
        q2_c3 = st.radio(
            "2. Ajuste de mAs:",
            ["6 mAs (factor 0.25×)", "12 mAs (factor 0.5×)", "25 mAs (igual)", "50 mAs (factor 2×)"],
            key="q2_caso3"
        )
        
        if q2_c3:
            if q2_c3 == "6 mAs (factor 0.25×)":
                st.success("✅ **Correcto**: Pediátrico <5 años → factor 0.25× según tabla de morfología.")
            else:
                st.error("❌ Pediátricos requieren reducción drástica: factor 0.25× para <5 años.")
        
        q3_c3 = st.radio(
            "3. ¿Usar rejilla antidifusión?",
            ["Sí, rejilla 10:1", "Sí, pero reducir a 6:1", "NO usar rejilla"],
            key="q3_caso3"
        )
        
        if q3_c3:
            if q3_c3 == "NO usar rejilla":
                st.success("✅ **Correcto**: Grosor <10-12 cm → sin rejilla (menos dosis, suficiente calidad).")
            else:
                st.error("❌ Pacientes pediátricos delgados NO requieren rejilla. Aumentaría dosis innecesariamente.")
        
        with st.expander("📖 Ver Protocolo Pediátrico Completo"):
            st.markdown("""
            **Protocolo optimizado para niño 4 años (18 kg)**:
            
            ```
            ⚡ kVp: 65-70 kVp (reducción de 5-10 kVp)
            🔋 mAs: 6 mAs (25 × 0.25)
            🔲 Rejilla: NO (grosor <12 cm)
            📏 DFI: 100 cm (estándar)
            ⏱️ Tiempo: Mínimo posible (movimiento)
            ```
            
            **Justificación**:
            
            1. **ALARA es crítico en pediatría**:
               - Tejidos en desarrollo = mayor radiosensibilidad
               - Mayor expectativa de vida = más tiempo para efectos tardíos
               - Objetivo: **Dosis mínima diagnóstica**
            
            2. **Sin rejilla**:
               - Grosor abdominal ~10-12 cm
               - Poca radiación dispersa generada
               - Ahorro de factor Bucky (5×) = **80% menos dosis**
            
            3. **Menor kVp**:
               - Menor penetración necesaria
               - Mantiene contraste adecuado
               - Reduce dosis de salida
            
            4. **Mínimo mAs**:
               - Factor 0.25× para <5 años
               - Suficiente SNR en digital moderno
               - Reduce tiempo de exposición (menor movimiento)
            
            **Precauciones adicionales**:
            
            - 🎯 **Colimación estricta**: Solo área de interés
            - 🛡️ **Protección gonadal**: Siempre que no interfiera con diagnóstico
            - 👶 **Inmovilización**: Sábanas, dispositivos, acompañante si necesario
            - 📋 **Preparación**: Explicar al niño (si edad apropiada) y padres
            - ⚡ **Técnica rápida**: Minimizar tiempo en sala
            - 🔄 **Evitar repeticiones**: Verificar posicionamiento antes de disparar
            
            **Comparación de dosis**:
            
            | Técnica | kVp | mAs | Rejilla | Dosis estimada |
            |---------|-----|-----|---------|----------------|
            | Adulto estándar | 75 | 25 | 10:1 | 1.0 mGy (ref) |
            | Pediátrico SUB-ÓPTIMO | 75 | 12 | 10:1 | 0.48 mGy |
            | **Pediátrico ÓPTIMO** | **70** | **6** | **NO** | **0.09 mGy** ✅ |
            
            **Reducción: >90% de dosis** 🎉
            """)
    
    with caso_tabs[3]:
        st.markdown("""
        ### 📋 Caso Clínico 4: Paciente Obeso Mórbido
        
        **Escenario**:
        Mujer de 45 años, 145 kg, IMC 48 (obesidad mórbida), dolor abdominal agudo.
        Radiografía de abdomen AP en urgencias.
        
        **Técnica estándar**: 75 kVp, 25 mAs, rejilla 10:1
        
        **Primer intento con AEC**: Tiempo máximo alcanzado (6 segundos), imagen subexpuesta.
        """)
        
        st.markdown("#### ❓ Preguntas:")
        
        q1_c4 = st.radio(
            "1. ¿Qué falló en el primer intento?",
            ["mAs insuficiente", 
             "kVp insuficiente (baja penetración)", 
             "Rejilla inadecuada",
             "AEC mal configurado"],
            key="q1_caso4"
        )
        
        if q1_c4:
            if q1_c4 == "kVp insuficiente (baja penetración)":
                st.success("✅ **Correcto**: 75 kVp es insuficiente para penetrar ~35-40 cm de tejido. El AEC no pudo compensar.")
            else:
                st.warning("⚠️ El problema principal es penetración. Con bajo kVp, ni el AEC puede compensar.")
        
        q2_c4 = st.radio(
            "2. Técnica optimizada:",
            ["75 kVp, 100 mAs (×4)", 
             "90 kVp (+20%), 50 mAs (×2)", 
             "105 kVp (+40%), 50 mAs (×2)",
             "Técnica manual imposible, usar TC"],
            key="q2_caso4"
        )
        
        if q2_c4:
            if q2_c4 == "90 kVp (+20%), 50 mAs (×2)":
                st.success("✅ **Correcto**: Equilibrio entre penetración y dosis. Aumentar kVp es prioritario en obesos.")
            elif q2_c4 == "105 kVp (+40%), 50 mAs (×2)":
                st.warning("⚠️ Funcionaría, pero kVp muy alto puede generar mucha dispersión. 90-95 kVp suele ser suficiente.")
            elif q2_c4 == "75 kVp, 100 mAs (×4)":
                st.error("❌ Dosis excesiva sin resolver el problema de penetración. Siempre aumentar kVp primero.")
            else:
                st.info("💡 TC puede ser mejor opción diagnóstica, pero RX optimizada es posible.")
        
        q3_c4 = st.radio(
            "3. ¿Qué más puedes hacer?",
            ["Comprimir el abdomen con banda", 
             "Usar técnica de dos disparos", 
             "Cambiar a proyección lateral",
             "Aumentar DFI a 150 cm"],
            key="q3_caso4"
        )
        
        if q3_c4:
            if q3_c4 == "Comprimir el abdomen con banda":
                st.success("✅ **Correcto**: La compresión reduce grosor efectivo (pero con precaución en abdomen agudo).")
            elif q3_c4 == "Aumentar DFI a 150 cm":
                st.warning("⚠️ Reduce magnificación pero requiere mucho más mAs (ley cuadrado inverso). Contraproducente.")
            else:
                st.error("❌ No son estrategias estándar para este problema.")
        
        with st.expander("📖 Ver Estrategia Completa para Pacientes Obesos"):
            st.markdown("""
            **Análisis del caso**:
            
            **Problema principal**: Obesidad mórbida (IMC 48) → grosor abdominal ~35-40 cm
            
            **Estrategia de optimización**:
            
            ### 1️⃣ **Aumentar kVp (PRIORIDAD)**
            
            ```
            Técnica estándar:  75 kVp, 25 mAs
            Obesidad factor 2× (mAs): 75 kVp, 50 mAs
            
            Problema: 75 kVp no penetra 35+ cm
            ❌ AEC alcanza tiempo máximo (6s)
            ❌ Imagen subexpuesta incluso con mAs alto
            
            Solución: AUMENTAR kVp primero
            ✅ Técnica optimizada: 90-95 kVp, 50 mAs
            ```
            
            **Justificación kVp alto**:
            - Mayor penetración del haz
            - Reduce absorción fotoelectrica (proporcional a Z³/E³)
            - Permite al AEC funcionar en rango normal
            - **Dosis efectiva menor** que forzar mAs alto con kVp bajo
            
            ### 2️⃣ **Ajuste de mAs**
            
            Factor obesidad mórbida: **2.0-2.5×**
            ```
            mAs base: 25 mAs
            mAs obeso mórbido: 50-60 mAs
            ```
            
            Con regla del 15%:
            ```
            Si 75 kVp, 50 mAs → subexpuesta
            Entonces 90 kVp (+20% = 1.15²), 50 mAs → CORRECTA ✅
            
            Justificación:
            75 → 86 kVp (+15%) = duplica exposición
            86 → 90 kVp (+5% adicional) = +10% más
            Total: ~2.2× exposición manteniendo 50 mAs
            ```
            
            ### 3️⃣ **Optimizaciones adicionales**
            
            **A. Compresión abdominal** (si es seguro):
            - Banda de compresión reduce 3-5 cm de grosor
            - ⚠️ **Precaución**: NO en abdomen agudo con sospecha de perforación
            - ⚠️ Requiere consentimiento y colaboración del paciente
            
            **B. Rejilla adecuada**:
            - Obesidad genera MUCHA radiación dispersa
            - Usar rejilla 12:1 o 16:1 (si disponible)
            - Mejora contraste (crítico con alto kVp)
            
            **C. Configuración AEC**:
            - Activar las 3 cámaras (abdomen es homogéneo)
            - Aumentar tiempo máximo de backup a 8-10 segundos (si equipo lo permite)
            - Verificar que cámaras están dentro del campo
            
            **D. Posicionamiento**:
            - Centrado perfecto (crítico con AEC)
            - Considerar decúbito lateral (reduce grosor AP)
            - Elevar brazos (reducir atenuación adicional)
            
            ### 4️⃣ **Técnica final propuesta**
            
            ```
            📊 TÉCNICA OPTIMIZADA:
            
            ⚡ kVp: 90-95 kVp (+20-27%)
            🔋 mAs: 50-60 mAs (×2-2.5)
            🔲 Rejilla: 12:1 o 16:1
            📏 DFI: 100 cm (estándar)
            🤖 AEC: 3 cámaras, backup 8-10s
            🎯 Colimación: Estricta
            
            Dosis estimada: 3.5-4.0 mGy
            (vs 8-10 mGy con técnica no optimizada)
            ```
            
            ### 5️⃣ **Comparación de estrategias**
            
            | Estrategia | kVp | mAs | Penetración | Dosis | Viabilidad |
            |------------|-----|-----|-------------|-------|------------|
            | Estándar | 75 | 25 | ❌ Insuficiente | 1.0× | ❌ Falla |
            | Solo ↑mAs | 75 | 100 | ❌ Insuficiente | 4.0× | ❌ Falla + alta dosis |
            | Solo ↑kVp | 90 | 25 | ✅ Buena | 0.8× | ⚠️ Puede ser corto |
            | **ÓPTIMA** | **90** | **50** | ✅ **Excelente** | **2.0×** | ✅ **Funciona** |
            
            ### 6️⃣ **Consideraciones especiales**
            
            **Límites del equipo**:
            - Verificar capacidad térmica del tubo
            - mA máximo disponible (puede limitar tiempo mínimo)
            - Generador de alta potencia preferible (>50 kW)
            
            **Alternativas diagnósticas**:
            - **Ecografía**: Primera línea en muchos casos abdominales
            - **TC**: Mejor calidad diagnóstica, dosis similar o menor
            - **RM**: Sin radiación, pero disponibilidad y coste
            
            **Comunicación**:
            - Informar al radiólogo de la dificultad técnica
            - Documentar parámetros utilizados
            - Si imagen es subóptima, explicar limitaciones técnicas
            - Considerar protocolo alternativo con el clínico
            
            ### 7️⃣ **Principios ALARA aplicados**
            
            ✅ **Justificación**: ¿Es realmente necesaria la RX?
            - En abdomen agudo: Valorar eco primero
            - Si RX imprescindible: Optimizar técnica
            
            ✅ **Optimización**: Técnica que minimiza dosis para diagnóstico adecuado
            - Preferir ↑kVp sobre ↑mAs
            - Colimación estricta
            - Evitar repeticiones (verificar antes de disparar)
            
            ✅ **Limitación**: Protección y blindaje
            - Personal: Salir de sala
            - Paciente: Protección gonadal si no interfiere
            
            **Resultado esperado**:
            Con técnica optimizada (90 kVp, 50 mAs):
            - ✅ Penetración adecuada
            - ✅ Densidad diagnóstica
            - ✅ Dosis controlada (~50% menos que técnica forzada con bajo kVp)
            - ✅ Contraste aceptable (con rejilla apropiada)
            """)
    
    # ============================================
    # SECTION 7: Resumen y Recursos
    # ============================================
    st.markdown("---")
    st.subheader("📚 Resumen de Conceptos Clave")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### ⚡ Reglas Fundamentales
        
        **Regla del 15%**
        - +15% kVp = ×2 exposición
        - Permite ÷2 mAs
        - Reduce dosis ~30%
        
        **Ley Cuadrado Inverso**
        - I₁/I₂ = (d₂/d₁)²
        - Duplicar distancia = ÷4 intensidad
        - Ajustar mAs proporcionalmente
        
        **Factor de Rejilla**
        - Sin rejilla: ×1
        - 8:1 → ×4
        - 10:1 → ×5
        - 12:1 → ×5
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Optimización
        
        **Prioridades**:
        1. Calidad diagnóstica
        2. Mínima dosis (ALARA)
        3. Eficiencia técnica
        
        **Estrategias**:
        - Pacientes delgados: ↓mAs, kVp estándar
        - Pacientes obesos: ↑kVp primero
        - Pediátricos: ↓↓mAs, sin rejilla si <12cm
        - Uso de AEC siempre que sea posible
        """)
    
    with col3:
        st.markdown("""
        ### 🔧 Resolución de Problemas
        
        **Imagen muy clara**:
        - ↑kVp (+15%) o ↑mAs (×2)
        - Verificar morfología paciente
        
        **Imagen muy oscura**:
        - ↓mAs o ↓kVp
        - Verificar AEC
        
        **Bajo contraste**:
        - ↓kVp (-10-15%)
        - Verificar rejilla
        
        **Imagen ruidosa**:
        - ↑mAs (mejora SNR)
        """)
    
    # Final tips box
    st.info("""
    💡 **Consejos del Técnico Experto**:
    
    1. **Siempre piensa en kVp primero**: Es el parámetro más influyente en calidad y dosis
    2. **AEC es tu amigo**: Úsalo siempre que sea apropiado, pero entiende cómo funciona
    3. **Morfología importa**: No hay "técnica única" - adapta siempre al paciente
    4. **ALARA constante**: Cada mAs cuenta - usa el mínimo diagnósticamente aceptable
    5. **Verifica antes de disparar**: Posición, centrado, colimación, protección
    6. **Documenta técnicas**: Especialmente en casos difíciles, aprende de la experiencia
    7. **Pregunta si dudas**: Mejor consultar que repetir (dosis adicional innecesaria)
    """)
    
    # Download summary
    with st.expander("📥 Descargar Resumen de Parámetros Técnicos"):
        st.markdown("""
        ### Tabla de Referencia Rápida
        
        #### Conversiones Básicas
        
        | Cambio | Factor | Efecto en Exposición |
        |--------|--------|----------------------|
        | kVp +15% | 1.15 | ×2 exposición |
        | kVp -15% | 0.87 | ÷2 exposición |
        | mAs ×2 | 2.0 | ×2 exposición |
        | mAs ÷2 | 0.5 | ÷2 exposición |
        | DFI ×2 | 2.0 | ÷4 intensidad → ×4 mAs |
        | DFI ÷2 | 0.5 | ×4 intensidad → ÷4 mAs |
            
        #### Factores de Morfología
        
        | Tipo de Paciente | Factor mAs | kVp Ajuste |
        |------------------|------------|------------|
        | Pediátrico < 5 años | 0.25× | -10 kVp |
        | Niño 5-12 años | 0.5× | -5 kVp |
        | Adolescente | 0.75× | Estándar |
        | Adulto asténico | 0.8× | Estándar |
        | Adulto normal | 1.0× | Estándar |
        | Adulto hiperesténico | 1.2× | +5 kVp |
        | Adulto obeso | 1.5× | +10-15 kVp |
        | Obeso mórbido | 2.0-2.5× | +20-30 kVp |
        
        #### Factores de Rejilla
        
        | Rejilla | Factor Bucky | Uso Típico |
        |---------|--------------|------------|
        | Sin rejilla | 1 | <10 cm grosor |
        | 5:1 | 2 | Portátiles |
        | 6:1 | 3 | Portátiles/General |
        | 8:1 | 4 | General/Mesa |
        | 10:1 | 5 | General/Bucky |
        | 12:1 | 5 | Bucky/Alta calidad |
        | 16:1 | 6 | Alta energía |
        
        #### Rangos de kVp por Región
        
        | Región Anatómica | kVp Típico | Contraste |
        |------------------|------------|-----------|
        | Extremidades | 50-65 | Alto |
        | Cráneo | 70-80 | Medio-Alto |
        | Columna cervical | 70-75 | Medio |
        | Columna lumbar | 80-95 | Medio |
        | Tórax PA | 120-125 | Bajo |
        | Abdomen | 75-85 | Medio |
        | Pelvis | 75-80 | Medio |
        
        #### Distancias Estándar
        
        | Proyección | DFI Estándar | Motivo |
        |------------|--------------|--------|
        | Tórax PA/PA | 180 cm | Reduce magnificación cardiaca |
        | General | 100 cm | Estándar universal |
        | Portátiles | 100-120 cm | Limitación práctica |
        | Extremidades | 100 cm | Estándar |
        
        #### Fórmulas Útiles
        
        ```
        1. Regla del 15%:
           kVp_nuevo = kVp_inicial × 1.15
           mAs_nuevo = mAs_inicial ÷ 2
        
        2. Ley del Cuadrado Inverso:
           mAs₂ = mAs₁ × (DFI₂ / DFI₁)²
        
        3. Factor de Rejilla:
           mAs_con_rejilla = mAs_sin_rejilla × Factor_Bucky
        
        4. Compensación de Morfología:
           mAs_ajustado = mAs_base × Factor_morfología
        
        5. SNR (aproximado):
           SNR ∝ √(mAs)
           Duplicar mAs → SNR aumenta 41%
        
        6. Dosis de Entrada (aproximada):
           ESD ∝ kVp² × mAs / DFI²
        ```
        
        #### Checklist Pre-Exposición
        
        ✅ **Identificación**: Paciente correcto, solicitud verificada
        ✅ **Posicionamiento**: Centrado, alineación, inmovilización
        ✅ **Protección**: Blindaje gonadal, colimación estricta
        ✅ **Técnica**: kVp/mAs apropiados para morfología
        ✅ **AEC**: Cámaras correctas si aplica
        ✅ **Rejilla**: Apropiada para región/grosor
        ✅ **DFI**: Correcta para proyección
        ✅ **Colimación**: Solo área de interés
        ✅ **Respiración**: Instrucciones claras al paciente
        ✅ **Personal**: Fuera de sala o protegido
        
        ---
        
        **Documento generado por el Simulador de Radiología**
        *Valores orientativos - Consulta siempre protocolos específicos de tu centro*
        """)
        
        # Create downloadable content
        summary_text = """
        RESUMEN DE PARÁMETROS TÉCNICOS EN RADIOLOGÍA
        =============================================
        
        REGLAS FUNDAMENTALES:
        - Regla del 15%: +15% kVp = ×2 exposición, permite ÷2 mAs
        - Ley Cuadrado Inverso: Intensidad ∝ 1/distancia²
        - Factor Rejilla: Multiplicar mAs según ratio de rejilla
        
        PRIORIDADES EN OPTIMIZACIÓN:
        1. Calidad diagnóstica adecuada
        2. Mínima dosis al paciente (ALARA)
        3. Eficiencia del flujo de trabajo
        
        ESTRATEGIAS POR TIPO DE PACIENTE:
        - Pediátrico: Reducir drásticamente (0.25-0.5×), sin rejilla si <12cm
        - Delgado: Reducir mAs (0.8×), kVp estándar
        - Normal: Técnica de referencia (1.0×)
        - Obeso: Aumentar kVp primero (+15-20%), luego mAs (1.5-2×)
        
        RESOLUCIÓN DE PROBLEMAS:
        - Imagen clara: ↑kVp o ↑mAs
        - Imagen oscura: ↓kVp o ↓mAs
        - Bajo contraste: ↓kVp
        - Mucho ruido: ↑mAs
        
        Generado por Simulador Educativo de Radiología
        """
        
        st.download_button(
            label="📄 Descargar Resumen en TXT",
            data=summary_text,
            file_name="resumen_parametros_tecnicos_radiologia.txt",
            mime="text/plain"
        )

# ============================================
# TAB 5: CALIDAD DE IMAGEN (to be completed)
# ============================================
with tabs[4]:
    st.header("📊 Calidad de Imagen")
    st.info("⚠️ Esta sección está en desarrollo. Será completada en la siguiente iteración.")
    st.markdown("""
    ### Próximamente en esta sección:
    
    - **Contraste**: Alto contraste vs bajo contraste
    - **Resolución espacial**: MTF, pares de líneas
    - **Ruido**: Cuántico, electrónico, estructurado
    - **Relación señal-ruido (SNR)**
    - **Artefactos**: Identificación y solución
    - **Métricas de calidad**: DQE, NEQ
    - **Balance dosis-calidad**: Curvas ROC
    
    Continúa con las otras pestañas disponibles...
    """)

# ============================================
# TAB 6: CASOS CLÍNICOS (to be completed)
# ============================================
with tabs[5]:
    st.header("🏥 Casos Clínicos Prácticos")
    st.info("⚠️ Esta sección está en desarrollo. Será completada en la siguiente iteración.")
    st.markdown("""
    ### Próximamente en esta sección:
    
    **Casos interactivos donde deberás**:
    - Seleccionar parámetros técnicos apropiados
    - Resolver problemas de calidad de imagen
    - Optimizar dosis manteniendo calidad diagnóstica
    - Adaptar técnicas a pacientes especiales
    - Identificar y corregir artefactos
    - Aplicar principios ALARA en situaciones reales
    
    **Escenarios incluirán**:
    - 👶 Radiografía pediátrica (tórax, abdomen)
    - 🦴 Trauma (extremidades, cráneo)
    - 🫁 Tórax en diferentes condiciones (obesidad, neumotórax)
    - 🤰 Paciente embarazada (consideraciones especiales)
    - 🏥 Portátiles en UCI
    - 🔧 Resolución de problemas técnicos
    
    Continúa con las otras pestañas disponibles...
    """)

# ============================================
# GLOBAL FOOTER
# ============================================
st.markdown("---")
st.markdown("---")

# About and credits
footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("""
    ### 📚 Recursos
    
    - [CSN - Consejo de Seguridad Nuclear](https://www.csn.es)
    - [SEFM - Sociedad Española de Física Médica](https://www.sefm.es)
    - [SEPR - Sociedad Española de Protección Radiológica](https://www.sepr.es)
    """)

with footer_col2:
    st.markdown("""
    ### ⚖️ Normativa
    
    - Real Decreto 1085/2009
    - Real Decreto 783/2001
    - Directiva 2013/59/EURATOM
    """)

with footer_col3:
    st.markdown("""
    ### 🎓 Formación
    
    - Ciclo FP: TSID
    - Módulo: Imagen para el Diagnóstico
    - Contenido: Física aplicada
    """)

st.markdown("---")

st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>⚡ Física de Imagen Radiológica - Simulador Interactivo</strong></p>
    <p>Herramienta educativa para Técnicos Superiores en Imagen para el Diagnóstico</p>
    <p style='font-size: 0.85em; margin-top: 10px;'>
        ⚠️ <strong>Disclaimer</strong>: Este simulador es una herramienta educativa. 
        En la práctica clínica real, siempre sigue los protocolos establecidos por tu centro 
        y la normativa vigente. Los valores y cálculos son aproximaciones simplificadas 
        con fines didácticos.
    </p>
    <p style='font-size: 0.8em; margin-top: 10px; color: #999;'>
        Versión 1.0 | 2024 | Basado en normativa española y europea vigente
    </p>
</div>
""", unsafe_allow_html=True)

