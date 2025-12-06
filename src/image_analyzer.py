"""
Image Mood Analyzer Module
Uses CLIP (Contrastive Language-Image Pre-training) for zero-shot mood classification.
Analyzes images and extracts mood characteristics (valence, energy) for song matching.
"""

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import io


# IMAGE MOOD ANALYZER CLASS
class ImageMoodAnalyzer:
    """
    Analyzes images and extracts mood scores using OpenAI's CLIP model.
    Maps images to mood categories with valence (positivity) and energy (intensity) scores.
    """
    
    # MOOD CATEGORY DEFINITIONS
    # Each mood has valence (0-1, sad to happy) and energy (0-1, calm to intense)
    MOOD_CATEGORIES = {
        "happy and excited": {
            "valence": 0.8,  # Very positive
            "energy": 0.8,   # High energy
            "emoji": "😄",
            "color": "#FFD700"  
        },
        "calm and peaceful": {
            "valence": 0.5,
            "energy": 0.2,
            "emoji": "😌",
            "color": "#87CEEB"  
        },
        "sad and melancholic": {
            "valence": 0.2,
            "energy": 0.2,
            "emoji": "😢",
            "color": "#4682B4"  
        },
        "dark and mysterious": {
            "valence": 0.3,
            "energy": 0.6,
            "emoji": "🌙",
            "color": "#2F4F4F"  
        },
        "romantic and gentle": {
            "valence": 0.2,
            "energy": 0.2,
            "emoji": "💕",
            "color": "#FFB6C1"  
        },
        "angry and aggressive": {
            "valence": 0.9,
            "energy": 0.9,
            "emoji": "😠",
            "color": "#DC143C"  
        },
        "nostalgic and reflective": {
            "valence": 0.4,
            "energy": 0.3,
            "emoji": "🍂",
            "color": "#D2691E"  
        }
    }
    
    # INITIALIZATION
    def __init__(self, model_name="openai/clip-vit-base-patch32", progress_callback=None):
        """
        Initialize the CLIP-based mood analyzer.
        
        Args:
            model_name (str): CLIP model identifier from Hugging Face
            progress_callback (callable, optional): Function to call with progress updates
        """
        self.model_name = model_name
        self.progress_callback = progress_callback
        
        # Model components 
        self.model = None
        self.processor = None
        
        # Device selection (GPU if available, else CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    

    # MODEL LOADING
    def load_model(self):
        """
        Load the CLIP model and processor.
        Called separately to allow progress display in GUI.
        """
        # Skip if already loaded
        if self.model is not None:
            return
        
        self._update_progress("Loading CLIP model...")
        
        # Load model and processor from Hugging Face
        self.model = CLIPModel.from_pretrained(self.model_name)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        # Move model to appropriate device (GPU/CPU)
        self.model.to(self.device)
        
        self._update_progress(f"Model loaded on {self.device}")
    
    # IMAGE ANALYSIS

    def analyze_image(self, image_input):
        """
        Analyze an image and determine its mood.
        
        Args:
            image_input: PIL Image, file path (str), or bytes
            
        Returns:
            dict: Mood analysis results with keys:
                - mood: Detected mood category (str)
                - valence: Positivity score 0-1 (float)
                - energy: Intensity score 0-1 (float)
                - confidence: Model confidence 0-1 (float)
                - emoji: Mood emoji (str)
                - color: Mood color hex code (str)
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Load and prepare image
        image = self._load_image(image_input)
        
        self._update_progress("Analyzing image mood...")
        
        # Get mood predictions
        predicted_mood, confidence = self._predict_mood(image)
        
        # Get mood metadata
        mood_data = self.MOOD_CATEGORIES[predicted_mood]
        
        # Build result dictionary
        result = {
            "mood": predicted_mood,
            "valence": mood_data["valence"],
            "energy": mood_data["energy"],
            "confidence": confidence,
            "emoji": mood_data["emoji"],
            "color": mood_data["color"]
        }
        
        self._update_progress(f"Detected: {predicted_mood}")
        
        return result
    
    
    def get_all_mood_probabilities(self, image_input):
        """
        Get probability scores for all mood categories.
        Useful for showing alternative moods or confidence distribution.
        
        Args:
            image_input: PIL Image, file path (str), or bytes
            
        Returns:
            dict: Mood categories mapped to their metadata and probabilities
        """
        # Ensure model is loaded
        if self.model is None:
            self.load_model()
        
        # Load and prepare image
        image = self._load_image(image_input)
        
        # Get probabilities for all moods
        probabilities = self._get_mood_probabilities(image)
        
        # Build result dictionary with metadata
        mood_probs = {}
        for i, mood in enumerate(self.MOOD_CATEGORIES.keys()):
            mood_data = self.MOOD_CATEGORIES[mood]
            mood_probs[mood] = {
                "probability": probabilities[i],
                "emoji": mood_data["emoji"],
                "color": mood_data["color"],
                "valence": mood_data["valence"],
                "energy": mood_data["energy"]
            }
        
        return mood_probs
    
    # HELPER METHODS
    def _load_image(self, image_input):
        """
        Load image from various input types.
        
        Args:
            image_input: PIL Image, file path (str), or bytes
            
        Returns:
            PIL.Image: RGB image ready for processing
        """
        if isinstance(image_input, str):
            # File path
            return Image.open(image_input).convert("RGB")
        
        elif isinstance(image_input, bytes):
            # Bytes (from Streamlit file uploader)
            return Image.open(io.BytesIO(image_input)).convert("RGB")
        
        elif isinstance(image_input, Image.Image):
            # PIL Image
            return image_input.convert("RGB")
        
        else:
            raise ValueError(
                f"Unsupported image input type: {type(image_input)}. "
                f"Expected str (path), bytes, or PIL.Image.Image"
            )
    
    
    def _predict_mood(self, image):
        """
        Predict the most likely mood for an image.
        
        Args:
            image (PIL.Image): RGB image
            
        Returns:
            tuple: (predicted_mood, confidence)
        """
        # Get probabilities for all moods
        probabilities = self._get_mood_probabilities(image)
        
        # Find the mood with highest probability
        max_prob_idx = probabilities.index(max(probabilities))
        predicted_mood = list(self.MOOD_CATEGORIES.keys())[max_prob_idx]
        confidence = probabilities[max_prob_idx]
        
        return predicted_mood, confidence
    
    
    def _get_mood_probabilities(self, image):
        """
        Get probability scores for all mood categories using CLIP.
        
        Args:
            image (PIL.Image): RGB image
            
        Returns:
            list: Probability scores for each mood category
        """
        # Create text descriptions for each mood
        text_descriptions = [
            f"a photo that feels {mood}" 
            for mood in self.MOOD_CATEGORIES.keys()
        ]
        
        # Process image and text through CLIP
        inputs = self.processor(
            text=text_descriptions,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Convert to list of probabilities
        return probs[0].tolist()
    
    
    def _update_progress(self, message):
        """
        Send progress update through callback if available.
        
        Args:
            message (str): Progress message
        """
        if self.progress_callback:
            self.progress_callback(message)
    
    # UTILITY METHODS
    @classmethod
    def get_mood_list(cls):
        """
        Get list of all available mood categories.
        
        Returns:
            list: Mood category names
        """
        return list(cls.MOOD_CATEGORIES.keys())
    
    
    @classmethod
    def get_mood_info(cls, mood):
        """
        Get metadata for a specific mood category.
        
        Args:
            mood (str): Mood category name
            
        Returns:
            dict: Mood metadata (valence, energy, emoji, color)
        """
        return cls.MOOD_CATEGORIES.get(mood)


# MODULE TEST
if __name__ == "__main__":
    """Test the image mood analyzer module."""
    print("Image Mood Analyzer Module")
    print("=" * 50)
    print("\n✓ Module loaded successfully!")
    
    print(f"\nAvailable mood categories ({len(ImageMoodAnalyzer.MOOD_CATEGORIES)}):")
    for i, mood in enumerate(ImageMoodAnalyzer.get_mood_list(), 1):
        info = ImageMoodAnalyzer.get_mood_info(mood)
        print(f"  {i}. {info['emoji']} {mood.title()}")
        print(f"     Valence: {info['valence']:.1f}, Energy: {info['energy']:.1f}")
    
    print("\n💡 To use:")
    print("  analyzer = ImageMoodAnalyzer()")
    print("  analyzer.load_model()")
    print("  result = analyzer.analyze_image('path/to/image.jpg')")
