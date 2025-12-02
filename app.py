import streamlit as st
import sys
from pathlib import Path
from textwrap import dedent

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from image_analyzer import ImageMoodAnalyzer
from song_recommender import SongRecommender


# Page configuration
st.set_page_config(
    page_title="Imagify - Image-to-Song Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS STYLING
st.markdown("""
<style>
    /* ===== HEADER ===== */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(180deg, #1DB954 0%, #191414 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }

    /* ===== MOOD CARD ===== */
    .mood-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #282828;
        margin: 1rem 0;
    }
    
    .stat-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #282828;
        margin: 1rem 0;
        text-align: center;
        box-shadow: 0 0 0 1px #333;
    }

    /* ===== SONG CARD (Cylindrical Glow Effect) ===== */
    .song-card {
        position: relative;
        border-radius: 9999px;
        background-color: #181818;
        margin: 0.75rem 0;
        overflow: hidden;
        box-shadow: 0 0 0 1px #333;
    }

    /* Blurred album background */
    .song-card-bg {
        position: absolute;
        inset: 0;
        background-size: cover;
        background-position: center;
        filter: blur(30px);
        transform: scale(1.1);
        opacity: 0.55;
    }

    /* Dark overlay gradient */
    .song-card-overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(
            90deg,
            rgba(0, 0, 0, 0.85) 0%,
            rgba(0, 0, 0, 0.35) 40%,
            rgba(0, 0, 0, 0.85) 100%
        );
    }

    /* Inner content alignment */
    .song-card-inner {
        position: relative;
        z-index: 1;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1.5rem;
        padding: 1rem 1.5rem;
    }

    .song-card-left {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .song-card-right {
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        gap: 8px;
        margin-right: 10px;
    }

    /* ===== ALBUM COVER ===== */
    .song-cover, .song-cover-placeholder {
        width: 80px;
        height: 80px;
        border-radius: 12px;
    }

    .song-cover {
        object-fit: cover;
    }

    .song-cover-placeholder {
        background-color: #333;
    }

    /* ===== SONG INFO TEXT ===== */
    .song-info {
        display: flex;
        flex-direction: column;
        gap: 0.1rem;
        font-size: 0.95rem;
    }

    .song-title {
        font-weight: 700;
    }

    /* ===== ICON BUTTONS (Spotify / YouTube) ===== */
    .song-icon-btn {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: #222;
        border: 1px solid #444;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: 0.2s ease;
        text-decoration: none;
    }

    .song-icon-btn img {
        width: 22px;
        height: 22px;
        user-select: none;
    }

    .song-icon-btn:hover {
        border-color: #1DB954;
        background-color: #1DB95422;
    }

    .song-youtube-btn:hover {
        border-color: #FF0000;
        background-color: #FF000022;
    }
    
    .disabled-btn {
        opacity: 0.3;
        pointer-events: none;
    }
</style>
""", unsafe_allow_html=True)

# SESSION STATE INITIALIZATION
def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'recommender' not in st.session_state:
        st.session_state.recommender = None
    if 'mood_result' not in st.session_state:
        st.session_state.mood_result = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = None
    if 'dataset_loaded' not in st.session_state:
        st.session_state.dataset_loaded = False

# UI COMPONENTS
def render_header():
    """Render the application header."""
    st.markdown("""
    <div class="main-header">
        <h1>🎵 Imagify 🖼️</h1>
        <p>Upload an image and discover songs that match its mood!</p>
        <p style="font-size: 0.9em; opacity: 0.8;">✨ Over 800+ Most Streamed Songs</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with information and stats."""
    with st.sidebar:
        # Load dataset and show stats
        if not st.session_state.dataset_loaded:
            with st.spinner("Loading song database..."):
                if st.session_state.recommender is None:
                    st.session_state.recommender = SongRecommender()
                st.session_state.dataset_loaded = True
        
        if st.session_state.recommender:
            st.header("📊 Dataset Info")
            stats = st.session_state.recommender.get_dataset_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Songs", f"{stats['total_songs']:,}")
            with col2:
                st.metric("Artists", f"{stats['artists']:,}")

            st.write(f"**Year Range:** {stats['date_range']['earliest']} - {stats['date_range']['latest']}")
        
        st.divider()
        
        # How it works
        with st.expander("🔍 How It Works"):
            st.write("""
            1. **Upload** an image
            2. **AI analyzes** the mood using OpenAI's CLIP
            3. **Match** songs by valence & energy
            4. **Get** instant recommendations!
            """)


def render_image_upload():
    """Render image upload section."""
    st.header("📸 Upload Your Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['png', 'jpg', 'jpeg'],
        )
    
    with col2:
        st.info("**💡 Tips:**\n- Use clear images\n- Nature, people, cities work best\n- Try the samples below!")
    
    if uploaded_file is not None:
        # Display the uploaded image
        col_img, col_btn = st.columns([2, 1])
        
        with col_img:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        with col_btn:
            st.write("")  
            st.write("")
            if st.button("🔍 Analyze Mood", type="primary", use_container_width=True):
                analyze_image(uploaded_file)
    
    else:
        # Show sample images
        st.subheader("📁 Or try a sample image:")
        col1, col2, col3 = st.columns(3)
        
        sample_images = {
            "Happy Beach": "samples/happy_beach.jpg",
            "Calm Forest": "samples/calm_forest.jpg",
            "Dark City": "samples/dark_city.jpg"
        }
        
        for col, (name, path) in zip([col1, col2, col3], sample_images.items()):
            with col:
                if Path(path).exists():
                    st.image(path, caption=name, use_container_width=True)
                    if st.button(f"Use {name}", key=name, use_container_width=True):
                        analyze_sample_image(path)
    
    return uploaded_file


def render_mood_results():
    """Render mood analysis results."""
    if not st.session_state.mood_result:
        return
    
    result = st.session_state.mood_result
    
    st.header("🎨 Mood Analysis Results")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="mood-card" style="border-left: 5px solid {result['color']}">
            <h2>{result['emoji']} {result['mood'].title()}</h2>
            <p style="font-size: 1.1em; margin-top: 1rem;">
                <strong>Confidence:</strong> {result['confidence']:.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <h3>💃 Valence</h3>
            <p style="font-size: 2em; margin: 0;">{result['valence']:.2f}</p>
            <p style="font-size: 0.9em; opacity: 0.8;">Positivity</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="stat-box">
            <h3>⚡ Energy</h3>
            <p style="font-size: 2em; margin: 0;">{result['energy']:.2f}</p>
            <p style="font-size: 0.9em; opacity: 0.8;">Intensity</p>
        </div>
        """, unsafe_allow_html=True)


def render_recommendations():
    """Render song recommendations with custom card design."""
    if not st.session_state.recommendations:
        return
    
    st.header("🎵 Recommended Songs")
    
    if len(st.session_state.recommendations) == 0:
        st.warning("⚠️ No songs found with 98%+ match. Try:")
        st.info("""
        - Using a different image with clearer mood
        - The dataset has 950+ most streamed songs
        - Adjust match threshold in code if needed
        """)
        return
    
    st.write(f"Found **{len(st.session_state.recommendations)} songs** that match this image's mood:")
    st.write("")  # Spacing
    
    # Track last valid cover for background glow effect
    last_cover_url = None
    
    for i, song in enumerate(st.session_state.recommendations, 1):
        cover_url = song.get("cover_url")
        spotify_url = song.get("spotify_url", "#")
        youtube_url = song.get("youtube_url", "#")
        
        # Album cover HTML
        if cover_url and cover_url != "Not Found":
            cover_html = f"<img class='song-cover' src='{cover_url}' alt='Album Cover' />"
            last_cover_url = cover_url
        else:
            cover_html = "<div class='song-cover-placeholder'></div>"
        
        # Background glow effect (uses last valid cover)
        if last_cover_url:
            bg_html = f"<div class='song-card-bg' style=\"background-image: url('{last_cover_url}');\"></div>"
        else:
            bg_html = ""
        
        # Determine if buttons should be disabled
        spotify_disabled = "disabled-btn" if spotify_url == "#" else ""
        youtube_disabled = "disabled-btn" if youtube_url == "#" else ""
        
        # Icon buttons HTML
        song_icons_html = f"""
            <a class='song-icon-btn {spotify_disabled}' href='{spotify_url}' target='_blank'>
                <img src='https://upload.wikimedia.org/wikipedia/commons/1/19/Spotify_logo_without_text.svg' alt='Spotify'>
            </a>
            <a class='song-icon-btn song-youtube-btn {youtube_disabled}' href='{youtube_url}' target='_blank'>
                <img src='https://upload.wikimedia.org/wikipedia/commons/e/ef/Youtube_logo.png' alt='YouTube'>
            </a>
        """
        
        # Complete song card HTML
        raw_html = dedent(f"""
        <div class="song-card">
            {bg_html}
            <div class="song-card-overlay"></div>
            <div class="song-card-inner">
                <div class="song-card-left">
                    {cover_html}
                    <div class="song-info">
                        <div class="song-title">{i}. {song['name']}</div>
                        <div>👤 {song['artist']}</div>
                        <div>📅 {song['release_date']}</div>
                        <div class="match-badge">{song['match_score']:.0%} match</div>
                    </div>
                </div>
                <div class="song-card-right">
                    {song_icons_html}
                </div>
            </div>
        </div>
        """)
        
        # Remove leading spaces to prevent Markdown code block interpretation
        card_html = "\n".join(line.lstrip() for line in raw_html.splitlines() if line.strip())
        
        st.markdown(card_html, unsafe_allow_html=True)

# IMAGE ANALYSIS FUNCTION
def analyze_image(uploaded_file):
    """Analyze the uploaded image for mood."""
    # Initialize analyzer if needed
    if st.session_state.analyzer is None:
        with st.spinner("🔄 Loading AI model (first time only, ~1 minute)..."):
            st.session_state.analyzer = ImageMoodAnalyzer()
            st.session_state.analyzer.load_model()
    
    # Analyze the image
    with st.spinner("🎨 Analyzing image mood..."):
        image_bytes = uploaded_file.getvalue()
        result = st.session_state.analyzer.analyze_image(image_bytes)
        st.session_state.mood_result = result
        
        # Get song recommendations
        get_recommendations(result)
    
    st.success("✅ Analysis complete!")
    st.rerun()


def analyze_sample_image(image_path):
    """Analyze a sample image."""
    # Initialize analyzer if needed
    if st.session_state.analyzer is None:
        with st.spinner("🔄 Loading AI model (first time only, ~1 minute)..."):
            st.session_state.analyzer = ImageMoodAnalyzer()
            st.session_state.analyzer.load_model()
    
    # Analyze the image
    with st.spinner("🎨 Analyzing image mood..."):
        result = st.session_state.analyzer.analyze_image(image_path)
        st.session_state.mood_result = result
        
        # Get song recommendations
        get_recommendations(result)
    
    st.success("✅ Analysis complete!")
    st.rerun()

def get_recommendations(mood_result):
    """Get song recommendations based on mood analysis."""
    # Initialize recommender if needed
    if st.session_state.recommender is None:
        st.session_state.recommender = SongRecommender()
    
    with st.spinner("🎵 Finding perfect songs for this mood..."):
        recommendations = st.session_state.recommender.recommend_songs(
            mood=mood_result['mood'],
            valence=mood_result['valence'],
            energy=mood_result['energy'],
            limit=10
        )
        
        st.session_state.recommendations = recommendations

# MAIN 
def main():
    """Main application function."""
    # Initialize session state
    initialize_session_state()
    
    # Render UI components
    render_header()
    render_sidebar()
    
    # Main content area
    uploaded_file = render_image_upload()
    
    # Show results if available
    if st.session_state.mood_result:
        render_mood_results()
        render_recommendations()


if __name__ == "__main__":
    main()
