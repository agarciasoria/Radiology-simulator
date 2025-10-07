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
# TAB 3: PROTECCIÓN RADIOLÓGICA (to be completed)
# ============================================
with tabs[2]:
    st.header("🛡️ Protección Radiológica")
    st.info("⚠️ Esta sección está en desarrollo. Será completada en la siguiente iteración.")
    st.markdown("""
    ### Próximamente en esta sección:
    
    - **Principios ALARA**: Tiempo, Distancia, Blindaje (interactivo)
    - **Cálculo de dosis**: Paciente, profesional, público
    - **Límites legales**: Trabajadores, embarazo, público
    - **Dosimetría personal**: TLD, OSL, interpretación
    - **Blindajes**: Cálculo de espesores de plomo/hormigón
    - **Simulador de sala**: Diseño de instalaciones
    - **Niveles de referencia diagnósticos (DRL)**
    
    Continúa con las otras pestañas disponibles...
    """)

# ============================================
# TAB 4: PARÁMETROS TÉCNICOS (to be completed)
# ============================================
with tabs[3]:
    st.header("🔧 Parámetros Técnicos y Optimización")
    st.info("⚠️ Esta sección está en desarrollo. Será completada en la siguiente iteración.")
    st.markdown("""
    ### Próximamente en esta sección:
    
    - **Tabla de técnicas**: Por anatomía y proyección
    - **Calculadora de conversión**: kVp-mAs, distancia, grid
    - **Factor de exposición**: Ajustes por morfología
    - **Rejillas anti-difusión**: Ratios, frecuencia, tipos
    - **AEC (Control automático)**: Selección de cámaras
    - **Calidad de imagen**: SNR, CNR, resolución espacial
    
    Continúa con las otras pestañas disponibles...
    """)

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

