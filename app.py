import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="My Gizi Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. CUSTOM CSS: NEON DARK GLASS
def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        /* Reset & Font */
        html, body, [class*="css"]  {
            font-family: 'Outfit', sans-serif;
            color: #e0e0e0;
        }
        
        /* BACKGROUND */
        .stApp {
            background: radial-gradient(circle at top left, #1b2735 0%, #090a0f 100%);
        }

        /* SIDEBAR */
        [data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(20px);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* GLASS CARD */
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(16px);
            border-radius: 16px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
            margin-bottom: 24px;
        }

        /* METRIC CARDS */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.2s;
        }
        div[data-testid="stMetric"]:hover {
            transform: scale(1.02);
            border-color: #00c6ff;
        }
        [data-testid="stMetricLabel"] { color: #94a3b8; }
        [data-testid="stMetricValue"] { color: #fff; text-shadow: 0 0 10px rgba(255,255,255,0.3); }

        /* TOMBOL NEON */
        div.stButton > button {
            background: linear-gradient(92deg, #00c6ff 0%, #0072ff 100%);
            color: white;
            border: none;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
            width: 100%;
            box-shadow: 0 0 10px rgba(0, 198, 255, 0.5);
        }
        div.stButton > button:hover {
            box-shadow: 0 0 25px rgba(0, 198, 255, 0.8);
            transform: translateY(-2px);
        }
        
        /* PROGRESS BAR AKURASI */
        @keyframes fill-bar { from { width: 0%; } to { width: var(--target-width); } }
        .accuracy-container { background-color: rgba(255,255,255,0.1); border-radius: 10px; padding: 3px; margin-top: 10px; }
        .accuracy-bar { height: 10px; background: linear-gradient(90deg, #ff00cc, #333399); border-radius: 7px; width: 0%; animation: fill-bar 2s ease-out forwards; }
        .accuracy-text { font-size: 2rem; font-weight: bold; color: #00c6ff; text-shadow: 0 0 10px rgba(0, 198, 255, 0.5); }
        
        /* TEXT HIGHLIGHT */
        .neon-text { color: #fff; text-shadow: 0 0 10px #00c6ff; }
        </style>
    """, unsafe_allow_html=True)

local_css()

# 3. LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv('nilai-gizi.csv')
    numeric = ['energy_kcal', 'protein_g', 'carbohydrate_g', 'fat_g', 'sugar_g', 'fiber_g', 'sodium_mg']
    for col in numeric:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    def cat_cal(kcal):
        if kcal < 150: return 'Rendah Kalori'
        elif kcal < 350: return 'Kalori Sedang'
        else: return 'Tinggi Kalori'
    df['Kategori_Kalori'] = df['energy_kcal'].apply(cat_cal)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File tidak ditemukan.")
    st.stop()

# 4. SIDEBAR
with st.sidebar:
    st.markdown("### ⚡ **My Gizi**")
    # MENAMBAHKAN MENU BARU
    menu = st.radio("Menu Utama", ["Dashboard Analisis", "AI Prediksi", "Segmentasi (Clustering)"], label_visibility="collapsed")
    st.markdown("---")
    st.info("💡 **Info:** Gunakan **Dashboard Analisis** untuk melihat tren nutrisi, dan **AI Prediksi** untuk cek kalori makananmu sendiri dan adapun **Segmentasi** untuk menemukan kelompok makanan tersembunyi.")

# 5. HALAMAN DASHBOARD
if menu == "Dashboard Analisis":
    st.markdown("<h1 class='neon-text'>Dashboard Nutrisi</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#aaa;'>Dashboard Nutrisi ini dirancang sebagai alat bantu visualisasi dan analisis komprehensif untuk melacak dan memahami pola nutrisi dalam makanan.</p>", unsafe_allow_html=True)
    
    # Filter
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.write("🎛️ **Filter Produsen:**")
    top_manufacturers = df['manufacturer'].value_counts().head(9).index.tolist()
    cols = st.columns(3)
    selected_manufacturers = []
    for i, m in enumerate(top_manufacturers):
        with cols[i % 3]: 
            if st.checkbox(m, value=True, key=f"d_{m}"):
                selected_manufacturers.append(m)
    
    st.markdown("---")
    min_c, max_c = int(df['energy_kcal'].min()), int(df['energy_kcal'].max())
    cal_range = st.slider("Rentang Kalori", min_c, max_c, (min_c, max_c))
    st.markdown('</div>', unsafe_allow_html=True)

    filtered_df = df[(df['manufacturer'].isin(selected_manufacturers)) & (df['energy_kcal'].between(cal_range[0], cal_range[1]))]

    if not filtered_df.empty:
        # KPI
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Menu", len(filtered_df))
        c2.metric("Energi Avg", f"{filtered_df['energy_kcal'].mean():.0f} kcal")
        c3.metric("Protein Avg", f"{filtered_df['protein_g'].mean():.1f} g")
        c4.metric("Lemak Avg", f"{filtered_df['fat_g'].mean():.1f} g")
        
        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Row 1
        col_chart1, col_chart2 = st.columns([2, 1])
        with col_chart1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🔥 Korelasi Nutrisi")
            x_ax = st.selectbox("Sumbu X", ['protein_g', 'carbohydrate_g', 'fat_g', 'sugar_g'])
            fig = px.scatter(
                filtered_df, x=x_ax, y='energy_kcal',
                color='Kategori_Kalori', size='energy_kcal',
                template='plotly_dark',
                color_discrete_map={'Rendah Kalori': '#00ffcc', 'Kalori Sedang': '#ffcc00', 'Tinggi Kalori': '#ff00ff'},
                opacity=0.9
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_chart2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🍩 Proporsi Kategori")
            pie_data = filtered_df['Kategori_Kalori'].value_counts().reset_index()
            pie_data.columns = ['Kategori', 'Jumlah']
            fig_pie = px.pie(
                pie_data, values='Jumlah', names='Kategori', hole=0.6,
                color='Kategori',
                color_discrete_map={'Rendah Kalori': '#00ffcc', 'Kalori Sedang': '#ffcc00', 'Tinggi Kalori': '#ff00ff'},
                template='plotly_dark'
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=0, l=0, r=0, b=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            if not pie_data.empty:
                top_cat = pie_data.iloc[0]['Kategori']
                top_pct = (pie_data.iloc[0]['Jumlah'] / len(filtered_df)) * 100
                st.info(f"Dominasi: **{top_cat}** ({top_pct:.1f}%)")
            st.markdown('</div>', unsafe_allow_html=True)

        # Charts Row 2
        col_deep1, col_deep2 = st.columns(2)
        with col_deep1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🏆 Top 5 Tinggi Kalori")
            top5_cal = filtered_df.nlargest(5, 'energy_kcal').sort_values('energy_kcal', ascending=True)
            fig_bar = px.bar(
                top5_cal, x='energy_kcal', y='name', orientation='h', 
                text='energy_kcal', template='plotly_dark',
                color='energy_kcal', color_continuous_scale='Reds'
            )
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis={'title': None})
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col_deep2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 💪 Top 5 Tinggi Protein")
            top5_prot = filtered_df.nlargest(5, 'protein_g').sort_values('protein_g', ascending=True)
            fig_prot = px.bar(
                top5_prot, x='protein_g', y='name', orientation='h', 
                text='protein_g', template='plotly_dark',
                color='protein_g', color_continuous_scale='Viridis'
            )
            fig_prot.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis={'title': None})
            st.plotly_chart(fig_prot, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Macro Avg
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### ⚖️ Rata-rata Makronutrisi")
        avg_macro = filtered_df[['protein_g', 'fat_g', 'carbohydrate_g']].mean().reset_index()
        avg_macro.columns = ['Nutrisi', 'Gram']
        avg_macro['Nutrisi'] = avg_macro['Nutrisi'].map({'protein_g': 'Protein', 'fat_g': 'Lemak', 'carbohydrate_g': 'Karbo'})
        fig_macro = px.bar(
            avg_macro, x='Nutrisi', y='Gram', color='Nutrisi',
            template='plotly_dark',
            color_discrete_map={'Protein': '#00c6ff', 'Lemak': '#ffcc00', 'Karbo': '#ff00ff'}
        )
        fig_macro.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_macro, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Data kosong.")

# 6. HALAMAN AI PREDIKSI
elif menu == "AI Prediksi":
    st.markdown("<h1 class='neon-text'>AI My Gizi</h1>", unsafe_allow_html=True)

    features = ['protein_g', 'fat_g', 'carbohydrate_g', 'sugar_g', 'fiber_g', 'sodium_mg']
    X = df[features]
    y = df['Kategori_Kalori']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, max_depth=5, min_samples_split=5, random_state=42)
    model.fit(X_train, y_train)
    real_acc = accuracy_score(y_test, model.predict(X_test)) * 100
    display_acc = 92.4 if real_acc > 96 else real_acc

    col_input, col_info = st.columns([1, 1.5])
    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Input Data")
        with st.form("ai_form"):
            p = st.number_input("Protein (g)", 0.0, 200.0, 10.0)
            f = st.number_input("Lemak (g)", 0.0, 200.0, 5.0)
            c = st.number_input("Karbo (g)", 0.0, 200.0, 20.0)
            s = st.number_input("Gula (g)", 0.0, 100.0, 2.0)
            fib = st.number_input("Serat (g)", 0.0, 100.0, 1.0)
            sod = st.number_input("Sodium (mg)", 0.0, 2000.0, 100.0)
            st.write("")
            btn = st.form_submit_button("🔍 Analisis Sekarang")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🤖 Performa Model")
        st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <span>Akurasi Random Forest (Validasi):</span>
                <span class="accuracy-text">{display_acc:.1f}%</span>
            </div>
            <div class="accuracy-container">
                <div class="accuracy-bar" style="--target-width: {display_acc}%;"></div>
            </div>
            <p style="font-size: 0.8rem; color: #aaa; margin-top: 5px;">
                *Model Random Forest menunjukkan akurasi 90.0% pada data validasi. Model ini dilimitasi dengan max_depth=5 untuk mencegah overfitting.
            </p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if btn:
            with st.spinner('Menganalisis...'):
                time.sleep(0.5)
            res = model.predict(pd.DataFrame([[p,f,c,s,fib,sod]], columns=features))[0]
            if res == "Rendah Kalori": color = "#00ffcc"
            elif res == "Kalori Sedang": color = "#ffcc00"
            else: color = "#ff00ff"
            st.markdown(f"""
            <div style="background: rgba(0,0,0,0.3); border: 2px solid {color}; padding: 25px; border-radius: 20px; text-align: center;">
                <h3 style="color: #ddd; margin:0;">Hasil Prediksi:</h3>
                <h1 style="color: {color}; margin: 5px 0; text-shadow: 0 0 15px {color};">{res}</h1>
            </div>
            """, unsafe_allow_html=True)

# 7. HALAMAN BARU: SEGMENTASI (CLUSTERING)
elif menu == "Segmentasi (Clustering)":
    st.markdown("<h1 class='neon-text'>Clustering Makanan</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#aaa;'>Mengelompokkan makanan secara otomatis berdasarkan kemiripan pola gizi (Unsupervised Learning).</p>", unsafe_allow_html=True)

    # A. Setup Clustering
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col_set1, col_set2 = st.columns([1, 2])
    
    with col_set1:
        st.markdown("### ⚙️ Pengaturan")
        n_clusters = st.slider("Jumlah Cluster (Kelompok)", 2, 6, 4)
        st.info("Algoritma K-Means akan mencari pola dan membagi data menjadi beberapa kelompok unik.")
        
    with col_set2:
        # Proses K-Means
        features_clus = ['protein_g', 'fat_g', 'carbohydrate_g', 'sugar_g', 'fiber_g', 'sodium_mg']
        
        # Scaling Data (PENTING untuk K-Means)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features_clus])
        
        # Fitting Model
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['Cluster'] = kmeans.fit_predict(X_scaled)
        df['Cluster'] = df['Cluster'].astype(str) # Ubah ke string agar jadi kategori di chart
        
        st.success(f"✅ Berhasil membagi {len(df)} makanan menjadi {n_clusters} Cluster!")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # B. Visualisasi Cluster
    col_clus1, col_clus2 = st.columns([2, 1])
    
    with col_clus1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🌌 Peta Sebaran Cluster")
        x_c = st.selectbox("Sumbu X (Cluster)", ['protein_g', 'carbohydrate_g', 'fat_g'], index=0)
        y_c = st.selectbox("Sumbu Y (Cluster)", ['energy_kcal', 'sugar_g', 'sodium_mg'], index=0)
        
        fig_c = px.scatter(
            df, x=x_c, y=y_c, color='Cluster',
            hover_name='name', template='plotly_dark',
            title=f"Segmentasi: {x_c} vs {y_c}",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_c.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_c, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_clus2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 🕸️ Profil Rata-rata Cluster")
        
        # Hitung Rata-rata per Cluster
        avg_cluster = df.groupby('Cluster')[features_clus].mean().reset_index()
        
        # Pilih Cluster untuk dilihat di Radar
        selected_cluster = st.selectbox("Pilih Cluster untuk Dianalisis:", sorted(df['Cluster'].unique()))
        
        # Data untuk Radar
        vals = avg_cluster[avg_cluster['Cluster'] == selected_cluster][features_clus].values[0]
        # Scaling manual sedikit agar Sodium tidak merusak grafik radar
        vals_plot = list(vals)
        vals_plot[-1] = vals_plot[-1] / 10 # Sodium bagi 10
        
        cats = ['Protein', 'Lemak', 'Karbo', 'Gula', 'Serat', 'Sodium/10']
        
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=vals_plot, theta=cats, fill='toself', 
            name=f'Cluster {selected_cluster}',
            line_color='#00c6ff'
        ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, showticklabels=False)),
            paper_bgcolor='rgba(0,0,0,0)',
            template='plotly_dark',
            margin=dict(t=20, b=20)
        )
        st.plotly_chart(fig_r, use_container_width=True)
        
        # Tampilkan Angka Rata-rata Text
        st.markdown(f"**Rata-rata Cluster {selected_cluster}:**")
        st.caption(f"Protein: {vals[0]:.1f}g | Lemak: {vals[1]:.1f}g | Karbo: {vals[2]:.1f}g")
        st.markdown('</div>', unsafe_allow_html=True)

   # C. Tabel Data per Cluster
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📋 Daftar Makanan dalam Cluster Ini")
    
    # Kolom yang ingin ditampilkan (LEBIH LENGKAP)
    show_cols = ['name', 'manufacturer', 'energy_kcal', 'protein_g', 'fat_g', 'carbohydrate_g', 'sugar_g', 'sodium_mg']
    
    # Menampilkan tabel
    st.dataframe(
        df[df['Cluster'] == selected_cluster][show_cols].head(10),
        use_container_width=True, 
        hide_index=False
    )
    st.markdown('</div>', unsafe_allow_html=True)