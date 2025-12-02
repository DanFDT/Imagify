"""
Song Recommender Module
Recommends songs from Spotify's most streamed songs based on mood scores.
No API authentication required - uses local CSV dataset.
"""

import pandas as pd
from pathlib import Path


# SONG RECOMMENDER CLASS
class SongRecommender:
    """
    Recommends songs based on valence (positivity) and energy (intensity) scores.
    Uses a local dataset of Spotify's most streamed songs.
    """
    
    # Dataset configuration
    DATASET_PATH = Path("data/SpotifySongsDatabase.csv")
    
    def __init__(self):
        """Initialize the recommender and load the dataset."""
        self.df = None
        self.load_dataset()
    
    
    # DATASET LOADING & PREPROCESSING
    def load_dataset(self):
        """Load and preprocess the song dataset from CSV."""
        # Check if dataset exists
        if not self.DATASET_PATH.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.DATASET_PATH}. "
                f"Please ensure SpotifySongsDatabase.csv is in the data/ folder."
            )
        
        # Load CSV with latin-1 encoding (handles special characters)
        print(f"Loading dataset from {self.DATASET_PATH}...")
        self.df = pd.read_csv(self.DATASET_PATH, encoding="latin-1")
        
        # Convert percentage columns (0-100) to decimal scale (0-1)
        percentage_columns = {
            'valence_%': 'valence',
            'energy_%': 'energy',
            'danceability_%': 'danceability'
        }
        
        for old_col, new_col in percentage_columns.items():
            if old_col in self.df.columns:
                self.df[new_col] = self.df[old_col] / 100.0
        
        # Rename columns to standardized names
        column_mapping = {
            'artist(s)_name': 'artist_name',
            'in_spotify_charts': 'popularity',
            'cover_url': 'album_cover',
            'spotify_search_link': 'spotify_url'
        }
        self.df = self.df.rename(columns=column_mapping)
        
        # Clean data - remove rows with missing fields
        self.df = self.df.dropna(subset=['valence', 'energy', 'track_name', 'artist_name'])
        
        # Convert streams to numeric 
        if 'streams' in self.df.columns:
            self.df['streams'] = pd.to_numeric(self.df['streams'], errors='coerce')
        
        print(f"✓ Loaded {len(self.df)} songs from dataset")
    
    
    # SONG RECOMMENDATION
    def recommend_songs(self, mood, valence, energy, limit=10, min_streams=None):
        """
        Recommend songs based on mood characteristics.
        
        Args:
            mood (str): Mood category (for display purposes)
            valence (float): Target valence/positivity (0.0 to 1.0)
            energy (float): Target energy/intensity (0.0 to 1.0)
            limit (int): Number of recommendations to return
            min_streams (int, optional): Minimum stream count filter
            
        Returns:
            list: List of recommended tracks with metadata
        """
        if self.df is None:
            self.load_dataset()
        
        # Calculate mood distance (how far each song is from target mood)
        self.df['mood_distance'] = (
            abs(self.df['valence'] - valence) + 
            abs(self.df['energy'] - energy)
        )
        
        # Filter for high-quality matches
        # Match score = 1 - (distance / 2)
        # For 75%+ match: distance <= 0.50
        max_distance = 0.25
        
        # Build filter conditions
        filters = (self.df['mood_distance'] <= max_distance)
        
        # Optional: filter by minimum streams
        if min_streams and 'streams' in self.df.columns:
            filters = filters & (self.df['streams'] >= min_streams)
        
        high_quality_matches = self.df[filters]
        
        # Remove duplicate songs (same name + artist)
        high_quality_matches = high_quality_matches.drop_duplicates(
            subset=['track_name', 'artist_name'], 
            keep='first'
        )
        
        # Get recommendations
        recommendations = self._select_recommendations(
            high_quality_matches, 
            limit
        )
        
        # Format and return results
        return self._format_results(recommendations)
    
    
    def _select_recommendations(self, matches, limit):
        """
        Select recommendations from matches with randomization.
        
        Args:
            matches (DataFrame): Filtered song matches
            limit (int): Number of songs to return
            
        Returns:
            DataFrame: Selected recommendations
        """
        if len(matches) >= limit:
            # Randomly sample from high-quality matches
            recommendations = matches.sample(n=limit, random_state=None)
            # Sort by match quality for display
            recommendations = recommendations.sort_values('mood_distance')
        
        elif len(matches) > 0:
            # Return all matches if less than limit
            recommendations = matches.sort_values('mood_distance')
        
        else:
            # Fallback: no high-quality matches found
            # Get top 10% of dataset as pool
            pool_size = max(50, len(self.df) // 10)
            closest_matches = self.df.nsmallest(pool_size, 'mood_distance')
            
            # Remove duplicates
            closest_matches = closest_matches.drop_duplicates(
                subset=['track_name', 'artist_name'], 
                keep='first'
            )
            
            # Sample or return all
            if len(closest_matches) > limit:
                recommendations = closest_matches.sample(n=limit, random_state=None)
                recommendations = recommendations.sort_values('mood_distance')
            else:
                recommendations = closest_matches.head(limit)
        
        return recommendations
    
    
    def _format_results(self, recommendations):
        """
        Format recommendation results with all metadata.
        
        Args:
            recommendations (DataFrame): Selected songs
            
        Returns:
            list: Formatted track information dictionaries
        """
        results = []
        
        for _, row in recommendations.iterrows():
            # Calculate match score (0-1, higher is better)
            match_score = 1 - (row['mood_distance'] / 2)
            
            # Format release date
            release_date = self._format_release_date(row)
            
            # Format streams
            streams_formatted = self._format_streams(row.get('streams'))
            
            # Build track info dictionary
            track_info = {
                "name": row['track_name'],
                "artist": row['artist_name'],
                "valence": float(row['valence']),
                "energy": float(row['energy']),
                "danceability": float(row.get('danceability', 0)),
                "tempo": float(row.get('bpm', 0)),
                "match_score": float(match_score),
                "release_date": release_date,
                "streams": streams_formatted,
                "spotify_playlists": int(row.get('in_spotify_playlists', 0)),
                "spotify_charts": int(row.get('popularity', 0)),
                "key": row.get('key', 'Unknown'),
                "mode": row.get('mode', 'Unknown'),
                "cover_url": row.get('album_cover', 'Not Found'),
                "spotify_url": row.get('spotify_url', ''),
                "youtube_url": row.get('youtube_url', '')
            }
            
            results.append(track_info)
        
        return results
    
    
    # HELPER METHODS
    def _format_release_date(self, row):
        """Format release date from row data."""
        year = row.get('released_year', 'Unknown')
        
        # Try to include month and day if available
        if 'released_month' in row and pd.notna(row['released_month']):
            day = int(row['released_day'])
            month = int(row['released_month'])
            year = int(year)
            return f"{day}/{month}/{year}"
        
        return str(year)
    
    
    def _format_streams(self, streams):
        """Format stream count with commas."""
        if pd.notna(streams):
            return f"{int(streams):,}"
        return "N/A"
    

    # DATASET STATISTICS
    def get_dataset_stats(self):
        """
        Get statistics about the loaded dataset.
        
        Returns:
            dict: Dataset statistics including songs, artists, and averages
        """
        if self.df is None:
            self.load_dataset()
        
        stats = {
            "total_songs": len(self.df),
            "artists": self.df['artist_name'].nunique(),
            "avg_valence": float(self.df['valence'].mean()),
            "avg_energy": float(self.df['energy'].mean()),
            "avg_streams": self._get_avg_streams(),
            "total_streams": self._get_total_streams(),
            "date_range": self._get_date_range()
        }
        
        return stats
    
    
    def _get_avg_streams(self):
        """Get average streams from dataset."""
        if 'streams' in self.df.columns:
            return int(self.df['streams'].mean())
        return 0
    
    
    def _get_total_streams(self):
        """Get total streams from dataset."""
        if 'streams' in self.df.columns:
            return int(self.df['streams'].sum())
        return 0
    
    
    def _get_date_range(self):
        """Get earliest and latest release years."""
        if 'released_year' in self.df.columns:
            return {
                "earliest": int(self.df['released_year'].min()),
                "latest": int(self.df['released_year'].max())
            }
        return {"earliest": "Unknown", "latest": "Unknown"}
    
    
    # SEARCH FUNCTIONALITY
    def search_songs(self, query, limit=10):
        """
        Search for songs by name or artist.
        
        Args:
            query (str): Search query string
            limit (int): Number of results to return
            
        Returns:
            list: Search results with song info
        """
        if self.df is None:
            self.load_dataset()
        
        # Case-insensitive search in track name and artist
        query_lower = query.lower()
        matches = self.df[
            self.df['track_name'].str.lower().str.contains(query_lower, na=False) |
            self.df['artist_name'].str.lower().str.contains(query_lower, na=False)
        ]
        
        # Format results
        results = []
        for _, row in matches.head(limit).iterrows():
            results.append({
                "name": row['track_name'],
                "artist": row['artist_name'],
                "streams": self._format_streams(row.get('streams'))
            })
        
        return results


# MODULE TEST
if __name__ == "__main__":
    """Test the song recommender module."""
    print("Song Recommender (Most Streamed Songs) module")
    print("=" * 50)
    
    try:
        # Initialize recommender
        recommender = SongRecommender()
        print("\n✓ Recommender initialized successfully!")
        
        # Display dataset statistics
        stats = recommender.get_dataset_stats()
        print(f"\nDataset Statistics:")
        print(f"  Total songs: {stats['total_songs']:,}")
        print(f"  Unique artists: {stats['artists']:,}")
        print(f"  Average valence: {stats['avg_valence']:.2f}")
        print(f"  Average energy: {stats['avg_energy']:.2f}")
        print(f"  Average streams: {stats['avg_streams']:,}")
        print(f"  Year range: {stats['date_range']['earliest']} - {stats['date_range']['latest']}")
        
        # Test recommendation
        print("\nTesting recommendation for happy/energetic mood (valence=0.9, energy=0.8):")
        songs = recommender.recommend_songs(
            mood="happy and energetic",
            valence=0.9,
            energy=0.8,
            limit=3
        )
        
        for i, song in enumerate(songs, 1):
            print(f"{i}. {song['name']} by {song['artist']} "
                  f"(match: {song['match_score']:.1%}, streams: {song['streams']})")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
