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
# TAB 2: FORMACIÓN DE IMAGEN (to be completed)
# ============================================
with tabs[1]:
    st.header("🎯 Formación de Imagen Radiográfica")
    st.info("⚠️ Esta sección está en desarrollo. Será completada en la siguiente iteración.")
    st.markdown("""
    ### Próximamente en esta sección:
    
    - **Interacción de rayos X con la materia**: Efecto fotoeléctrico, Compton, dispersión
    - **Ley de Beer-Lambert**: Atenuación exponencial
    - **Contraste radiográfico**: Diferencias de densidad y número atómico
    - **Dispersión**: Rejillas anti-difusión
    - **Simulador de phantom**: Construye tu propio paciente virtual
    - **Factor de exposición**: Cálculo para diferentes anatomías
    
    Continúa con las otras pestañas disponibles...
    """)

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
    """)
        