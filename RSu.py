import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pandas as pd
import plotly.express as px
from collections import Counter
import re
import plotly.graph_objects as go
import concurrent.futures
import html2text
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import validators
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image

# Set page config first
st.set_page_config(
    page_title="Analyse SEO Pro",
    page_icon="Logo.png", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Import CSS from interface.css
with open('interface.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ---- Configuration des requêtes HTTP ----

# Configuration des retries pour les requêtes HTTP
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("https://", adapter)
http.mount("http://", adapter)

# ---- Fonctions d'extraction de données ----

def get_page_content(url):
    try:
        if not validators.url(url):
            st.error("URL invalide")
            return None
            
        response = http.get(url, timeout=5)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de la récupération de la page {url}: {e}")
        return None

def get_internal_links(base_url, content):
    if not content:
        return []
        
    soup = BeautifulSoup(content, "html.parser")
    links = soup.find_all("a", href=True)
    internal_links = []
    base_domain = urlparse(base_url).netloc
    
    for link in links:
        href = link['href']
        full_url = urljoin(base_url, href)
        if urlparse(full_url).netloc == base_domain and "#" not in full_url:
            internal_links.append(full_url)
            
    return list(set(internal_links))  # Return all internal links

# ---- Fonctions d'analyse robots.txt ----

def check_robots_txt(url):
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    
    try:
        response = http.get(robots_url, timeout=5)
        response.raise_for_status()
        
        robots_content = response.text
        if "User-agent" in robots_content:
            return {"status": "present", "content": robots_content}
        else:
            return {"status": "invalid", "content": "Le fichier robots.txt ne contient pas de directives valides."}
    except requests.exceptions.HTTPError:
        return {"status": "not_found", "content": None}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "content": str(e)}

def analyze_robots_txt(content):
    if not isinstance(content, str):
        return []

    blocked_paths = []
    lines = content.splitlines()
    current_agent = "*"
    
    for line in lines:
        line = line.strip().lower()
        if line.startswith("user-agent:"):
            current_agent = line.split(":", 1)[1].strip()
        elif line.startswith("disallow:") and current_agent in ["*", "googlebot"]:
            path = line.split(":", 1)[1].strip()
            if path:
                blocked_paths.append(path)
    
    return blocked_paths

# ---- Fonctions de vérification technique ----

def check_meta_tags(content):
    soup = BeautifulSoup(content, "html.parser")
    meta_title = soup.find("title")
    meta_description = soup.find("meta", attrs={"name": "description"})
    meta_robots = soup.find("meta", attrs={"name": "robots"})
    meta_keywords = soup.find("meta", attrs={"name": "keywords"})
    meta_viewport = soup.find("meta", attrs={"name": "viewport"})
    
    meta_score = 0
    meta_issues = []
    
    if meta_title and len(meta_title.text) >= 10:
        meta_score += 25
    else:
        meta_issues.append("Titre manquant ou trop court")
        
    if meta_description and len(meta_description.get("content", "")) >= 50:
        meta_score += 25
    else:
        meta_issues.append("Description manquante ou trop courte")
        
    if meta_keywords:
        meta_score += 25
    else:
        meta_issues.append("Mots-clés manquants")
        
    if meta_viewport:
        meta_score += 25
    else:
        meta_issues.append("Viewport manquant")
        
    return meta_score, meta_issues

def check_security(url):
    security_score = 0
    security_issues = []
    
    # Vérification HTTPS
    if url.startswith("https://"):
        security_score += 30
    else:
        security_issues.append("Le site n'utilise pas HTTPS")
    
    try:
        response = http.head(url, timeout=5)
        headers = response.headers
        
        # Vérification des en-têtes de sécurité essentiels
        security_headers = {
            "Strict-Transport-Security": 20,
            "Content-Security-Policy": 20,
            "X-Frame-Options": 10,
            "X-Content-Type-Options": 10,
            "Referrer-Policy": 10
        }
        
        for header, points in security_headers.items():
            if header in headers:
                security_score += points
            else:
                security_issues.append(f"En-tête {header} manquant")
                
        return security_score, security_issues
        
    except requests.exceptions.RequestException as e:
        security_issues.append(f"Erreur lors de la vérification de sécurité: {str(e)}")
        return 0, security_issues

def check_page_load_time(url):
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(15)

        start_time = time.time()
        driver.get(url)
        
        # Attendre que la page soit complètement chargée
        load_time = time.time() - start_time
        
        # Récupérer les métriques de performance
        navigation_timing = driver.execute_script(
            "return window.performance.timing.loadEventEnd - window.performance.timing.navigationStart;"
        )
        
        driver.quit()
        
        # Utiliser la plus grande des deux valeurs
        final_load_time = max(load_time, navigation_timing / 1000 if navigation_timing > 0 else load_time)
        return final_load_time
        
    except Exception as e:
        st.warning(f"Erreur lors de la mesure du temps de chargement: {str(e)}")
        return None
    finally:
        try:
            driver.quit()
        except:
            pass
    
def check_mobile_friendly(content):
    soup = BeautifulSoup(content, "html.parser")
    
    mobile_score = 0
    mobile_issues = []
    
    # Vérification de la balise viewport
    viewport = soup.find("meta", attrs={"name": "viewport"})
    if viewport and "width=device-width" in viewport.get("content", ""):
        mobile_score += 50
    else:
        mobile_issues.append("Pas de viewport responsive")
    
    # Vérification des tailles de police
    font_sizes = []
    for element in soup.find_all(["p", "span", "div"]):
        font_size = element.get("style", "")
        if "font-size" in font_size:
            try:
                size = int(re.search(r'font-size:\s*(\d+)px', font_size).group(1))
                font_sizes.append(size)
            except:
                pass
    
    if font_sizes:
        avg_font_size = sum(font_sizes) / len(font_sizes)
        if avg_font_size >= 14:
            mobile_score += 25
        else:
            mobile_issues.append("Taille de police trop petite")
    
    # Vérification des éléments tactiles
    touch_elements = soup.find_all(["button", "a", "input", "select"])
    for element in touch_elements:
        style = element.get("style", "")
        if "width" in style and "height" in style:
            try:
                width = int(re.search(r'width:\s*(\d+)px', style).group(1))
                height = int(re.search(r'height:\s*(\d+)px', style).group(1))
                if width < 44 or height < 44:
                    mobile_issues.append("Éléments tactiles trop petits")
                    break
            except:
                pass
    
    if not mobile_issues:
        mobile_score += 25
        
    return mobile_score, mobile_issues

def check_content_quality(content):
    # Convertir le HTML en texte
    h = html2text.HTML2Text()
    h.ignore_links = True
    text = h.handle(content)
    
    # Tokenization et nettoyage
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text.lower())
    
    # Filtrer les stop words et les mots courts
    french_stop_words = set(stopwords.words('french'))
    words = [word for word in tokens if word not in french_stop_words and len(word) > 3]
    
    word_count = len(words)
    
    # Calcul du score
    content_score = 0
    content_issues = []
    # Longueur du contenu
    if word_count >= 300:
        content_score += 30
    else:
        content_issues.append(f"Contenu trop court ({word_count} mots, minimum recommandé: 300)")
    
    # Analyse des mots-clés significatifs
    keyword_density = Counter(words)
    top_keywords = keyword_density.most_common(10)  # Changed to 10 keywords
    
    if top_keywords:
        max_density = top_keywords[0][1] / word_count * 100
        if max_density <= 5:
            content_score += 20
        else:
            content_issues.append("Densité de mots-clés trop élevée")
    
    # Structure du contenu
    soup = BeautifulSoup(content, "html.parser")
    headings = soup.find_all(["h1", "h2", "h3"])
    if len(headings) >= 3:
        content_score += 25
    else:
        content_issues.append("Pas assez de titres et sous-titres")
    
    # Images avec alt text
    images = soup.find_all("img")
    images_with_alt = [img for img in images if img.get("alt")]
    if images and len(images_with_alt) / len(images) >= 0.8:
        content_score += 25
    else:
        content_issues.append("Images sans texte alternatif")
    
    return content_score, content_issues, word_count, top_keywords

# ---- Fonctions de calcul de score SEO ----

def calculate_seo_score(url):
    content = get_page_content(url)
    if content is None:
        return 0, {}, {}, pd.DataFrame(), []
    
    # Analyse parallèle des différents aspects
    with concurrent.futures.ThreadPoolExecutor() as executor:
        load_time_future = executor.submit(check_page_load_time, url)
        meta_future = executor.submit(check_meta_tags, content)
        mobile_future = executor.submit(check_mobile_friendly, content)
        content_future = executor.submit(check_content_quality, content)
        security_future = executor.submit(check_security, url)
        
        # Récupération des résultats
        load_time = load_time_future.result()
        meta_score, meta_issues = meta_future.result()
        mobile_score, mobile_issues = mobile_future.result()
        content_score, content_issues, word_count, top_keywords = content_future.result()
        security_score, security_issues = security_future.result()
    
    # Calcul du score de vitesse
    load_time_score = max(0, 100 - (load_time * 10)) if load_time else 0
    
    # Structure des données pour chaque critère
    criteria_data = []
    
    # Score final pondéré avec les nouveaux poids
    weights = {
        "Vitesse de chargement": 0.25,
        "Balises meta": 0.20,
        "Compatibilité mobile": 0.20,
        "Qualité du contenu": 0.20,
        "Sécurité": 0.15
    }
    
    final_score = (
        (load_time_score * weights["Vitesse de chargement"]) +
        (meta_score * weights["Balises meta"]) +
        (mobile_score * weights["Compatibilité mobile"]) +
        (content_score * weights["Qualité du contenu"]) +
        (security_score * weights["Sécurité"])
    )
    
    scores = {
        "Vitesse de chargement": load_time_score,
        "Balises meta": meta_score,
        "Compatibilité mobile": mobile_score,
        "Qualité du contenu": content_score,
        "Sécurité": security_score
    }
    
    issues = {
        "Vitesse de chargement": [f"Temps de chargement: {load_time:.2f}s"] if load_time > 3 else [],
        "Balises meta": meta_issues,
        "Compatibilité mobile": mobile_issues,
        "Qualité du contenu": content_issues,
        "Sécurité": security_issues
    }
    
    # Création du DataFrame avec les détails
    for criterion, score in scores.items():
        criteria_data.append({
            "Critère": criterion,
            "Score (%)": f"{score:.2f}",
            "Importance (%)": f"{int(weights[criterion] * 100)}",
            "Problèmes détectés": ", ".join(issues[criterion]) if issues[criterion] else "Aucun problème détecté"
        })
    
    df = pd.DataFrame(criteria_data)
    
    return final_score, scores, issues, df, top_keywords

# ---- Fonctions d'analyse globale ----

def analyze_site_seo(base_url):
    main_score, main_scores, issues, _, top_keywords = calculate_seo_score(base_url)
    st.write(f"Note globale pour {base_url}: {main_score:.2f}%")
    st.progress(main_score / 100)
    
    # Afficher les problèmes détectés
    for category, category_issues in issues.items():
        if category_issues:
            st.write(f"**{category}:**")
            for issue in category_issues:
                st.write(f"- {issue}")
    
    # Analyse des sous-pages en parallèle
    html_content = get_page_content(base_url)
    if html_content:  # Ensure html_content is not None
        internal_links = get_internal_links(base_url, html_content)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_url = {executor.submit(calculate_seo_score, link): link for link in internal_links}
            subpage_scores = []
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    sub_score, sub_scores, sub_issues, _, _ = future.result()
                    subpage_scores.append((url, sub_score, sub_scores))
                    st.write(f"Page {url} - Score SEO : {sub_score:.2f}%")
                    st.progress(sub_score / 100)
                except Exception as e:
                    st.error(f"Erreur lors de l'analyse de {url}: {str(e)}")
    
    total_score = (main_score + sum([score for _, score, _ in subpage_scores])) / (len(subpage_scores) + 1)
    return total_score, subpage_scores

# ---- Fonctions d'affichage ----

def display_circular_score(total_score):
    colors = ['#497fc5' if total_score >= 80 else '#497fc5' if total_score >= 50 else '#224e87', '#E5ECF6']
    
    fig = go.Figure(go.Pie(
        labels=["Score SEO", "Potentiel d'amélioration"],
        values=[total_score, 100 - total_score],
        hole=0.7,
        marker=dict(colors=colors),
        textinfo="percent",
        insidetextorientation="radial",
        hoverinfo="label+percent"
    ))

    fig.update_layout(
        title=dict(
            text=f"Score SEO: {total_score:.1f}%",
            y=0.5,
            x=0.5,
            xanchor='center',
            yanchor='middle',
            font=dict(size=24, color='#497fc5')
        ),
        showlegend=False,
        height=400,
        margin=dict(t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    st.plotly_chart(fig, use_container_width=True)

def display_seo_explanation():
    st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <h1>Analyse SEO Pro</h1>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🎯 Objectifs de l'analyse SEO

    Notre outil analyse en profondeur votre site web selon 5 critères clés :

    #### 1. Vitesse de chargement (25%)
    - Temps de chargement optimal < 3 secondes
    - Impact direct sur l'expérience utilisateur
    - Facteur crucial pour le référencement mobile

    #### 2. Balises Meta (20%)
    - Title : 50-60 caractères
    - Description : 150-160 caractères
    - Mots-clés pertinents
    - Balises robots et viewport

    #### 3. Compatibilité Mobile (20%)
    - Design responsive
    - Taille de texte lisible
    - Boutons et liens adaptés au tactile
    - Navigation fluide

    #### 4. Qualité du Contenu (20%)
    - Minimum 300 mots par page
    - Structure claire (H1, H2, H3)
    - Densité de mots-clés 2-5%
    - Images optimisées avec alt text

    #### 5. Sécurité (15%)
    - Certificat HTTPS
    - En-têtes de sécurité
    - Protection contre les attaques
    - Politique de confidentialité

    ### 📊 Interprétation des scores

    - 90-100% : Excellent
    - 80-89% : Très bon
    - 70-79% : Bon
    - 60-69% : Moyen
    - < 60% : Nécessite des améliorations

    ### 🔄 Processus d'analyse

    1. Scan initial de la page
    2. Analyse technique approfondie
    3. Vérification du contenu
    4. Tests de performance
    5. Génération du rapport

    ### 💡 Conseils d'optimisation

    - Optimisez vos images
    - Utilisez un cache navigateur
    - Minimisez CSS/JavaScript
    - Créez du contenu unique
    - Structurez vos données
    """)

# ---- Application principale ----

def main():
    # Sidebar pour l'explication SEO
    with st.sidebar:
        # Afficher le logo
        Logo = Image.open("Logo.png")
        st.image(Logo, use_container_width=True)
        display_seo_explanation()

    # En-tête
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(Logo, use_container_width=True) 
        st.markdown("""
            <div style='text-align: center; margin-bottom: 20px;'>
                <h1>Analyse SEO Pro</h1>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("""
            <div style='text-align: center; color: #497fc5;'>
            Optimisez votre visibilité en ligne avec notre outil d'analyse SEO professionnel
            </div>
        """, unsafe_allow_html=True)

    # Conteneur principal
    with st.container():
        st.markdown("---")
        
        # Zone de saisie URL avec style
        url = st.text_input(
            "🌐 Entrez l'URL du site à analyser",
            placeholder="Entrer votre lien",
            help="Entrez l'URL complète incluant https:// ou http://"
        )

        if url:
            if not validators.url(url):
                st.error("⚠️ Veuillez entrer une URL valide")
                return

            # Boutons d'analyse
            col1, col2 = st.columns(2)
            with col1:
                analyse_complete = st.button("🔍 Analyse complète du site")
            with col2:
                analyse_page = st.button("📄 Analyse de la page")

            if analyse_complete:
                with st.spinner("🔄 Analyse approfondie du site en cours..."):
                    main_score, main_scores, issues, df, top_keywords = calculate_seo_score(url)
                    html_content = get_page_content(url)
                    internal_links = get_internal_links(url, html_content)
                    
                    # Affichage des résultats dans des onglets
                    tab1, tab2, tab3, tab4 = st.tabs(["📊 Vue d'ensemble", "🔍 Détails", "🔗 Pages internes", "📈 Mots-clés"])
                    
                    with tab1:
                        display_circular_score(main_score)
                        
                    with tab2:
                        st.dataframe(
                            df.style.background_gradient(subset=['Score (%)'], cmap='RdYlGn'),
                            use_container_width=True
                        )
                        
                        # Affichage des problèmes avec des icônes
                        for category, category_issues in issues.items():
                            if category_issues:
                                with st.expander(f"⚠️ {category}"):
                                    for issue in category_issues:
                                        st.markdown(f"• {issue}")
                                        
                    with tab3:
                        if internal_links:
                            st.success(f"🔗 {len(internal_links)} pages internes trouvées")
                            for link in internal_links:
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    score, _, _, _, _ = calculate_seo_score(link)
                                    st.markdown(f"""
                                        <span style='color: var(--text-color);'>
                                        • <a href='{link}'>{link}</a> - Score SEO: {score:.2f}%
                                        </span>
                                    """, unsafe_allow_html=True)
                                with col2:
                                    if st.button("Analyser", key=f"analyze_{link}"):
                                        st.session_state.url_to_analyze = link
                                        st.session_state.analyze_page = True
                                        st.experimental_rerun()
                            
                    with tab4:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            if top_keywords:
                                # Word Cloud avec les couleurs du logo
                                word_freq = {word: freq for word, freq in top_keywords}
                                wordcloud = WordCloud(
                                    width=800,
                                    height=400,
                                    background_color='white',
                                    colormap='Blues',
                                    color_func=lambda *args, **kwargs: '#497fc5',
                                    max_words=10
                                ).generate_from_frequencies(word_freq)
                                
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wordcloud, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig)
                        
                        with col2:
                            st.markdown("""
                            ### Comprendre les mots-clés
                            
                            Les mots-clés affichés sont les termes les plus fréquents de votre contenu, 
                            filtrés pour exclure les mots courants (articles, prépositions, etc.).
                            
                            Une bonne stratégie de mots-clés devrait :
                            - Être naturelle et pertinente
                            - Avoir une densité équilibrée (2-5%)
                            - Inclure des variations sémantiques
                            - Cibler votre audience
                            """)

    # Pied de page
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #497fc5; padding: 20px;'>
        Développé avec ❤️ pour optimiser votre présence en ligne
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
