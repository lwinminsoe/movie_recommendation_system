import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
import time

# 🔵 NEXON BLUE THEME PAGE CONFIG
st.set_page_config(
    page_title="CineNex AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🌊 NEXON BLUE & BLACK CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 50%, #16213e 100%);
        color: #e6f7ff;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    
    .nexon-header {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(45deg, #0096ff, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .nexon-card {
        background: rgba(10, 25, 47, 0.85);
        border: 1px solid rgba(0, 150, 255, 0.3);
        border-radius: 16px;
        padding: 24px;
        margin: 12px 0;
        backdrop-filter: blur(20px);
    }
    
    .success-message {
        background: linear-gradient(135deg, rgba(0, 255, 150, 0.1), rgba(0, 200, 120, 0.05));
        border: 1px solid rgba(0, 255, 150, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
    }
    
    .error-message {
        background: linear-gradient(135deg, rgba(255, 50, 50, 0.1), rgba(200, 40, 40, 0.05));
        border: 1px solid rgba(255, 50, 50, 0.3);
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class MovieRecommenderSystem:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.links_df = None
        self.tags_df = None
        self.cosine_sim = None
        self.indices = None
        self.tfidf = None
        self.popular_movies_df = None
        
    def load_movies_data(self, file):
        """Load and validate movies data"""
        try:
            df = pd.read_csv(file)
            
            # Check required columns
            if 'title' not in df.columns:
                return None, "❌ Missing 'title' column in movies data"
            
            # Create movieId if missing
            if 'movieId' not in df.columns:
                df['movieId'] = range(1, len(df) + 1)
                st.info("ℹ️ Created movieId column automatically")
                
            # Clean data
            df['title'] = df['title'].fillna('Unknown Movie')
            
            if 'genres' in df.columns:
                df['genres'] = df['genres'].fillna('Unknown')
                df['genres'] = (
                    df['genres']
                    .str.replace('|', ' ')
                    .str.replace('(no genres listed)', 'Unknown', case=False)
                    .str.strip()
                )
            else:
                df['genres'] = 'Unknown'
                st.info("ℹ️ No genres column found - using default")
                
            return df, None
            
        except Exception as e:
            return None, f"Error loading movies data: {str(e)}"
    
    def load_ratings_data(self, file):
        """Load and validate ratings data"""
        try:
            df = pd.read_csv(file)
            required_cols = ['userId', 'movieId', 'rating']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return None, f"Missing columns in ratings: {missing_cols}"
                
            return df, None
            
        except Exception as e:
            return None, f"Error loading ratings data: {str(e)}"
    
    def load_tags_data(self, file):
        """Load and validate tags data"""
        try:
            df = pd.read_csv(file)
            return df, None
        except Exception as e:
            return None, f"Error loading tags data: {str(e)}"
    
    def load_links_data(self, file):
        """Load and validate links data"""
        try:
            df = pd.read_csv(file)
            return df, None
        except Exception as e:
            return None, f"Error loading links data: {str(e)}"
    
    def process_ratings_data(self):
        """Process ratings data to connect with movies"""
        try:
            if self.ratings_df is None or self.movies_df is None:
                return False, "No ratings or movies data available"
            
            # Check if we have movieId in both datasets
            if 'movieId' not in self.movies_df.columns:
                return False, "No movieId in movies data"
            
            if 'movieId' not in self.ratings_df.columns:
                return False, "No movieId in ratings data"
            
            # Calculate movie statistics
            movie_stats = self.ratings_df.groupby('movieId').agg({
                'rating': ['mean', 'count', 'std']
            }).round(3)
            
            movie_stats.columns = ['avg_rating', 'rating_count', 'rating_std']
            movie_stats = movie_stats.reset_index()
            
            # Filter movies with sufficient ratings
            popular_movies = movie_stats[movie_stats['rating_count'] >= 5]
            popular_movies = popular_movies.sort_values(['avg_rating', 'rating_count'], ascending=[False, False])
            
            # Merge with movie titles and genres
            self.popular_movies_df = popular_movies.merge(
                self.movies_df[['movieId', 'title', 'genres']], 
                on='movieId', 
                how='inner'
            )
            
            st.success(f"✅ Processed {len(self.popular_movies_df)} popular movies from ratings")
            return True, f"Found {len(self.popular_movies_df)} popular movies"
            
        except Exception as e:
            return False, f"Error processing ratings: {str(e)}"
    
    def build_recommendation_model(self, movies_df):
        """Build the recommendation model"""
        try:
            if movies_df is None or movies_df.empty:
                return False, "No movies data available"
                
            # Create TF-IDF vectorizer for genres
            self.tfidf = TfidfVectorizer(stop_words='english', min_df=1)
            tfidf_matrix = self.tfidf.fit_transform(movies_df['genres'])
            
            # Compute cosine similarity
            self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # Create movie indices
            self.indices = pd.Series(movies_df.index, index=movies_df['title'].str.lower()).drop_duplicates()
            
            return True, "Model built successfully"
            
        except Exception as e:
            return False, f"Error building model: {str(e)}"
    
    def get_recommendations(self, title, top_n=10):
        """Get movie recommendations based on title"""
        try:
            if self.cosine_sim is None or self.indices is None:
                return [], "Recommendation model not built"
                
            title_lower = title.lower()
            
            # Find matching title
            if title_lower not in self.indices:
                # Try fuzzy matching
                matches = [t for t in self.indices.index if title_lower in t]
                if matches:
                    title_lower = matches[0]
                else:
                    return [], f"Movie '{title}' not found in dataset"
            
            idx = self.indices[title_lower]
            sim_scores = list(enumerate(self.cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
            
            movie_indices = [i[0] for i in sim_scores]
            recommendations = self.movies_df.iloc[movie_indices].copy()
            recommendations['similarity_score'] = [sim_scores[i][1] for i in range(len(movie_indices))]
            
            return recommendations, None
            
        except Exception as e:
            return [], f"Error getting recommendations: {str(e)}"
    
    def get_popular_movies(self, top_n=10):
        """Get popular movies based on ratings"""
        try:
            if self.popular_movies_df is not None and not self.popular_movies_df.empty:
                return self.popular_movies_df.head(top_n)
            else:
                # Fallback: return random movies from dataset
                return self.movies_df.sample(min(top_n, len(self.movies_df)))
                
        except Exception as e:
            # Final fallback
            return self.movies_df.head(top_n) if self.movies_df is not None else None

# Initialize the recommender system
recommender = MovieRecommenderSystem()

# 🎬 NEXON HEADER
st.markdown("""
<div style="text-align: center; padding: 2rem 0 1rem 0;">
    <h1 class="nexon-header">CineNex AI</h1>
    <p style="color: #88ccff; font-size: 1.2rem;">
        Advanced Movie Recommendation System
    </p>
</div>
""", unsafe_allow_html=True)

# 📁 MULTI-FILE UPLOAD SYSTEM
with st.sidebar:
    st.markdown("""
    <div class="nexon-card">
        <h3 style="color: #00d4ff; margin-bottom: 1rem;">📁 Upload Movie Datasets</h3>
        <p style="color: #88ccff; font-size: 0.9rem;">
            Start with movies.csv, then add other files for enhanced features
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Multiple file uploaders
    movies_file = st.file_uploader(
        "🎬 Movies CSV (REQUIRED)", 
        type=["csv"],
        help="Must contain: title, Optional: movieId, genres"
    )
    
    ratings_file = st.file_uploader(
        "⭐ Ratings CSV (OPTIONAL)", 
        type=["csv"],
        help="Should contain: userId, movieId, rating"
    )
    
    tags_file = st.file_uploader(
        "🏷️ Tags CSV (OPTIONAL)", 
        type=["csv"],
        help="Optional: userId, movieId, tag, timestamp"
    )
    
    links_file = st.file_uploader(
        "🔗 Links CSV (OPTIONAL)", 
        type=["csv"],
        help="Optional: movieId, imdbId, tmdbId"
    )
    
    # Process uploaded files
    if movies_file is not None:
        with st.spinner("Loading movies data..."):
            movies_df, error = recommender.load_movies_data(movies_file)
            if error:
                st.error(error)
            else:
                recommender.movies_df = movies_df
                st.success(f"✅ Loaded {len(movies_df)} movies")
                
                # Build recommendation model
                success, message = recommender.build_recommendation_model(movies_df)
                if success:
                    st.success("🎯 Recommendation model ready!")
                else:
                    st.warning(message)
    
    if ratings_file is not None:
        with st.spinner("Loading ratings data..."):
            ratings_df, error = recommender.load_ratings_data(ratings_file)
            if error:
                st.error(error)
            else:
                recommender.ratings_df = ratings_df
                st.success(f"✅ Loaded {len(ratings_df)} ratings")
                
                # Process ratings to connect with movies
                with st.spinner("Connecting ratings with movies..."):
                    success, message = recommender.process_ratings_data()
                    if success:
                        st.success(message)
                    else:
                        st.warning(f"Ratings processing: {message}")
    
    if tags_file is not None:
        with st.spinner("Loading tags data..."):
            tags_df, error = recommender.load_tags_data(tags_file)
            if error:
                st.error(error)
            else:
                recommender.tags_df = tags_df
                st.success(f"✅ Loaded {len(tags_df)} tags")
    
    if links_file is not None:
        with st.spinner("Loading links data..."):
            links_df, error = recommender.load_links_data(links_file)
            if error:
                st.error(error)
            else:
                recommender.links_df = links_df
                st.success(f"✅ Loaded {len(links_df)} links")

# 🎯 MAIN APPLICATION INTERFACE
try:
    if recommender.movies_df is not None:
        # 🎭 DASHBOARD OVERVIEW
        st.markdown("""
        <div class="nexon-card">
            <h2 style="color: #00d4ff; margin-bottom: 1rem;">📊 Movie Dashboard</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Animated metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_movies = len(recommender.movies_df)
            st.metric("🎬 Total Movies", total_movies)
        
        with col2:
            if recommender.ratings_df is not None:
                total_ratings = len(recommender.ratings_df)
                st.metric("⭐ Total Ratings", total_ratings)
            else:
                unique_titles = recommender.movies_df['title'].nunique()
                st.metric("📝 Unique Titles", unique_titles)
        
        with col3:
            if recommender.popular_movies_df is not None:
                popular_count = len(recommender.popular_movies_df)
                st.metric("🏆 Popular Movies", popular_count)
            elif recommender.tags_df is not None:
                total_tags = len(recommender.tags_df)
                st.metric("🏷️ Total Tags", total_tags)
            else:
                if 'genres' in recommender.movies_df.columns:
                    unique_genres = len(set(' '.join(recommender.movies_df['genres']).split()))
                    st.metric("🎭 Unique Genres", unique_genres)
                else:
                    st.metric("⚡ AI Engine", "Active")
        
        with col4:
            if recommender.cosine_sim is not None:
                st.metric("💫 Status", "Ready", delta="Online")
            else:
                st.metric("🔧 Status", "Building...")
        
        # 🎯 RECOMMENDATION SYSTEM
        st.markdown("""
        <div class="nexon-card">
            <h2 style="color: #00d4ff; margin-bottom: 1rem;">🎯 Intelligent Recommendations</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div class="nexon-card">
                <h4 style="color: #00d4ff;">Select Your Movie</h4>
            </div>
            """, unsafe_allow_html=True)
            
            movie_titles = recommender.movies_df['title'].dropna().unique()
            selected_movie = st.selectbox(
                "Choose a movie you like:",
                movie_titles,
                index=min(10, len(movie_titles) - 1)
            )
            
            top_n = st.slider("Number of recommendations:", 3, 20, 10)
            
            if st.button("🚀 Get Recommendations", use_container_width=True):
                with st.spinner("Finding similar movies..."):
                    time.sleep(0.5)
                    recommendations, error = recommender.get_recommendations(selected_movie, top_n)
                    
                    if error:
                        st.error(error)
                    else:
                        st.markdown(f"""
                        <div class="success-message">
                            <h4 style="color: #00ffaa; margin: 0;">✨ Found {len(recommendations)} Recommendations</h4>
                            <p style="color: #88ccff; margin: 0.5rem 0 0 0;">Similar to: <strong>{selected_movie}</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for idx, movie in recommendations.iterrows():
                            with st.container():
                                st.markdown('<div class="nexon-card">', unsafe_allow_html=True)
                                
                                col_a, col_b = st.columns([3, 1])
                                with col_a:
                                    st.markdown(f"### 🎬 **{movie['title']}**")
                                    if 'genres' in movie and movie['genres']:
                                        st.markdown(f"**🎭** `{movie['genres']}`")
                                with col_b:
                                    similarity = movie.get('similarity_score', 0)
                                    progress = int(similarity * 100)
                                    st.metric("Match", f"{progress}%")
                                
                                st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="nexon-card">
                <h4 style="color: #00d4ff;">🎪 Popular Movies</h4>
                <p style="color: #88ccff; font-size: 0.9rem;">
                    Based on user ratings data
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            popular_movies = recommender.get_popular_movies(8)
            if popular_movies is not None:
                for idx, movie in popular_movies.iterrows():
                    with st.container():
                        st.markdown('<div class="nexon-card" style="padding: 1rem; margin: 0.5rem 0;">', unsafe_allow_html=True)
                        
                        # Movie title
                        st.markdown(f"<p style='color: #ffffff; margin: 0 0 8px 0; font-weight: bold;'>🎬 {movie['title']}</p>", unsafe_allow_html=True)
                        
                        # Rating info if available
                        if 'avg_rating' in movie:
                            rating_count = movie.get('rating_count', 0)
                            st.markdown(f"<p style='color: #ffd700; margin: 0 0 5px 0;'>⭐ {movie['avg_rating']}/5.0 ({rating_count} ratings)</p>", unsafe_allow_html=True)
                        
                        # Genres if available
                        if 'genres' in movie and movie['genres']:
                            st.markdown(f"<p style='color: #88ccff; margin: 0; font-size: 0.9em;'>{movie['genres']}</p>", unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No popular movies data available. Upload ratings.csv to see popular movies.")
        
        # 🔍 DATA EXPLORATION TABS
        tab1, tab2, tab3 = st.tabs(["📁 Dataset Explorer", "🔍 Search Movies", "📈 Statistics"])
        
        with tab1:
            st.markdown("""
            <div class="nexon-card">
                <h4 style="color: #00d4ff;">Dataset Preview</h4>
            </div>
            """, unsafe_allow_html=True)
            
            dataset_choice = st.selectbox("Choose dataset to view:", 
                                         ["Movies", "Ratings", "Tags", "Links", "Popular Movies"])
            
            if dataset_choice == "Movies" and recommender.movies_df is not None:
                rows_to_show = st.slider("Rows to display:", 5, 50, 10)
                st.dataframe(recommender.movies_df.head(rows_to_show), use_container_width=True)
            
            elif dataset_choice == "Ratings" and recommender.ratings_df is not None:
                rows_to_show = st.slider("Rows to display:", 5, 50, 10)
                st.dataframe(recommender.ratings_df.head(rows_to_show), use_container_width=True)
            
            elif dataset_choice == "Tags" and recommender.tags_df is not None:
                rows_to_show = st.slider("Rows to display:", 5, 50, 10)
                st.dataframe(recommender.tags_df.head(rows_to_show), use_container_width=True)
            
            elif dataset_choice == "Links" and recommender.links_df is not None:
                rows_to_show = st.slider("Rows to display:", 5, 50, 10)
                st.dataframe(recommender.links_df.head(rows_to_show), use_container_width=True)
            
            elif dataset_choice == "Popular Movies" and recommender.popular_movies_df is not None:
                rows_to_show = st.slider("Rows to display:", 5, 50, 10)
                st.dataframe(recommender.popular_movies_df.head(rows_to_show), use_container_width=True)
            else:
                st.info(f"No {dataset_choice} data available")
        
        with tab2:
            st.markdown("""
            <div class="nexon-card">
                <h4 style="color: #00d4ff;">Search Movie Database</h4>
            </div>
            """, unsafe_allow_html=True)
            
            search_query = st.text_input("Enter movie title or keyword:")
            if search_query:
                results = recommender.movies_df[
                    recommender.movies_df['title'].str.contains(search_query, case=False, na=False)
                ]
                st.write(f"Found {len(results)} matches")
                if len(results) > 0:
                    st.dataframe(results[['title', 'genres']] if 'genres' in results.columns else results[['title']])
        
        with tab3:
            st.markdown("""
            <div class="nexon-card">
                <h4 style="color: #00d4ff;">Dataset Statistics</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Movies Data Info:**")
                if recommender.movies_df is not None:
                    st.write(f"Total movies: {len(recommender.movies_df)}")
                    st.write(f"Columns: {list(recommender.movies_df.columns)}")
                    if 'genres' in recommender.movies_df.columns:
                        unique_genres = len(set(' '.join(recommender.movies_df['genres']).split()))
                        st.write(f"Unique genres: {unique_genres}")
                else:
                    st.write("No movies data")
            
            with col2:
                if recommender.ratings_df is not None:
                    st.write("**Ratings Data Info:**")
                    st.write(f"Total ratings: {len(recommender.ratings_df)}")
                    st.write(f"Average rating: {recommender.ratings_df['rating'].mean():.2f}")
                    st.write(f"Unique users: {recommender.ratings_df['userId'].nunique()}")
                    if recommender.popular_movies_df is not None:
                        st.write(f"Popular movies: {len(recommender.popular_movies_df)}")
                elif recommender.tags_df is not None:
                    st.write("**Tags Data Info:**")
                    st.write(f"Total tags: {len(recommender.tags_df)}")
                    if 'userId' in recommender.tags_df.columns:
                        st.write(f"Unique users: {recommender.tags_df['userId'].nunique()}")
                    if 'movieId' in recommender.tags_df.columns:
                        st.write(f"Tagged movies: {recommender.tags_df['movieId'].nunique()}")
                else:
                    st.write("No additional data")
    
    else:
        # 🏠 SIMPLIFIED WELCOME PAGE
        st.markdown("""
        <div style="text-align: center; padding: 5rem 2rem;">
            <h1 class="nexon-header">Welcome to CineNex AI</h1>
            <p style="color: #88ccff; font-size: 1.4rem; line-height: 1.6; margin: 2rem 0;">
                Upload your movies.csv file to get started with AI-powered recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Application error: {str(e)}")

# 🌊 FOOTER
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #88ccff; padding: 2rem 0 1rem 0;">
    <p style="margin: 0.5rem 0; font-size: 1.1rem;">CineNex AI • Advanced Movie Recommendation System</p>
    <p style="margin: 0.5rem 0; font-size: 0.9rem; color: #66aaff;">
        Built with Streamlit • Powered by Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)