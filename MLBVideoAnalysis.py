import streamlit as st
import pandas as pd
import requests
import json
import spacy
import os
import tempfile
import urllib.request
from moviepy.editor import VideoFileClip, vfx
import math
from google.cloud import videointelligence
from PIL import Image
from io import BytesIO
from google.cloud import videointelligence
import numpy as np
from typing import List, Dict

# Set page configuration as the FIRST Streamlit command
st.set_page_config(
    page_title="MLB Home Run Analysis",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .player-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
""", unsafe_allow_html=True)



import streamlit as st

# Access any field like this
credentials = {
    "type": st.secrets["type"],
    "project_id": st.secrets["project_id"],
    "private_key_id": st.secrets["private_key_id"],
    "private_key": st.secrets["private_key"],
    "client_email": st.secrets["client_email"],
    "client_id": st.secrets["client_id"],
    "auth_uri": st.secrets["auth_uri"],
    "token_uri": st.secrets["token_uri"],
    "auth_provider_x509_cert_url": st.secrets["auth_provider_x509_cert_url"],
    "client_x509_cert_url": st.secrets["client_x509_cert_url"],
    "universe_domain": st.secrets["universe_domain"]
}

# Write credentials to a temporary file
credentials_path = "temp_credentials.json"
with open(credentials_path, "w") as f:
    json.dump(credentials, f)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path

# Initialize spaCy model
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# Helper Functions
def process_endpoint_url(endpoint_url, pop_key=None):
    """Process MLB API endpoints and return DataFrame"""
    try:
        response = requests.get(endpoint_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if pop_key:
            df_result = pd.json_normalize(data.pop(pop_key), sep='_')
        else:
            df_result = pd.json_normalize(data)
        return df_result
    except requests.exceptions.RequestException as e:
        st.error(f"Error accessing MLB API: {str(e)}")
        return pd.DataFrame()

def extract_player_name(title):
    """Extract player name from title using spaCy"""
    if isinstance(title, str):
        doc = nlp(title)
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
    return None

def get_player_id(player_name, season):
    """Get player ID from MLB API"""
    if isinstance(player_name, str):
        url = f'https://statsapi.mlb.com/api/v1/sports/1/players?season={season}'
        players_df = process_endpoint_url(url, 'people')
        if not players_df.empty:
            player = players_df[players_df['fullName'].str.contains(player_name, case=False, na=False)]
            if not player.empty:
                return player.iloc[0]['id']
    return None

def get_player_details(player_id):
    """Get player details from MLB API"""
    try:
        url = f'https://statsapi.mlb.com/api/v1/people/{player_id}/'
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting player details: {str(e)}")
        return None

def get_player_headshot(player_id):
    """Get player headshot from MLB.com"""
    try:
        url = f'https://securea.mlb.com/mlb/images/players/head_shot/{player_id}.jpg'
        response = requests.get(url)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content))
        return None
    except Exception:
        return None

@st.cache_data
def load_hr_data():
    """Load MLB home runs data from CSV files"""
    mlb_hr_csvs_list = [
        'https://storage.googleapis.com/gcp-mlb-hackathon-2025/datasets/2016-mlb-homeruns.csv',
        'https://storage.googleapis.com/gcp-mlb-hackathon-2025/datasets/2017-mlb-homeruns.csv',
        'https://storage.googleapis.com/gcp-mlb-hackathon-2025/datasets/2024-mlb-homeruns.csv',
        'https://storage.googleapis.com/gcp-mlb-hackathon-2025/datasets/2024-postseason-mlb-homeruns.csv'
    ]
    
    dfs = []
    for csv_file in mlb_hr_csvs_list:
        try:
            season = csv_file.split('/')[-1].split('-')[0]
            df = pd.read_csv(csv_file)
            df['season'] = season
            dfs.append(df)
        except Exception as e:
            st.error(f"Error loading data from {csv_file}: {str(e)}")
    
    combined_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    # Remove the first entry
    return combined_df.iloc[1:] if not combined_df.empty else combined_df

def process_video(video_url, title):
    """Process video and perform analysis"""
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download video
            input_video = os.path.join(temp_dir, f"{title[:7]}.mp4")
            urllib.request.urlretrieve(video_url, input_video)
            
            # Process video with slower speed
            output_video = os.path.join(temp_dir, "slowed_video.mp4")
            clip = VideoFileClip(input_video)
            speed_factor = 0.4
            slowed_clip = clip.fx(vfx.speedx, speed_factor)
            slowed_clip = slowed_clip.set_audio(clip.audio.fx(vfx.speedx, speed_factor))
            slowed_clip.write_videofile(output_video, codec="libx264")
            
            # Display video
            with open(output_video, 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)
            
            # Perform video intelligence analysis
            video_client = videointelligence.VideoIntelligenceServiceClient()
            features = [
                videointelligence.Feature.LABEL_DETECTION,
                videointelligence.Feature.OBJECT_TRACKING
            ]
            
            with open(output_video, "rb") as video_file:
                input_content = video_file.read()
            
            operation = video_client.annotate_video(
                request={"features": features, "input_content": input_content}
            )
            
            with st.spinner("Processing video for object annotations..."):
                result = operation.result(timeout=500)
            st.success("Video processing complete!")
            
            return result

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None

def calculate_bat_metrics(bat_frames):
    """Calculate bat swing metrics"""
    if not bat_frames or len(bat_frames) < 2:
        return {
            "swing_speed": 0,
            "swing_angle": 0,
            "total_distance": 0,
            "swing_duration": 0,
            "horizontal_distance": 0,
            "vertical_distance": 0,
        }

    positions = []
    timestamps = []
    for frame in bat_frames:
        box = frame.normalized_bounding_box
        center_x = (box.left + box.right) / 2
        center_y = (box.top + box.bottom) / 2
        positions.append((center_x, center_y))
        timestamps.append(frame.time_offset.seconds + frame.time_offset.microseconds / 1e6)

    horizontal_distance = abs(positions[-(min(len(positions)-2,3))][0] - positions[0][0])*10
    vertical_distance = abs(positions[-(min(len(positions)-2,3))][1] - positions[0][1])*10

    total_distance = (horizontal_distance**2 + vertical_distance**2)**0.5
    swing_duration = (timestamps[-(min(len(positions)-2,3))] - timestamps[0])
    swing_speed = total_distance/swing_duration if swing_duration > 0 else 0

    swing_angle = math.degrees(math.atan(vertical_distance / horizontal_distance)) if horizontal_distance > 0 else 0

    return {
        "swing_speed": swing_speed*41.33*5.5,
        "swing_angle": swing_angle,
        "total_distance": total_distance*32,
        "swing_duration": swing_duration*0.4,
        "horizontal_distance": horizontal_distance*32,
        "vertical_distance": vertical_distance*32
    }

def main():
    st.title("⚾ MLB Home Run Analysis")
    
    with st.spinner("Loading MLB home run data..."):
        all_mlb_hrs = load_hr_data()
    
    if all_mlb_hrs.empty:
        st.error("Failed to load home run data. Please try again later.")
        return
    
    # Add a dropdown for home run titles
    hr_titles = all_mlb_hrs['title'].tolist()
    selected_title = st.selectbox(
        "Select a Home Run Play",
        options=hr_titles,
        help="Search for a home run play by its title",
        index=0
    )
    
    # Add Generate Analysis button
    if st.button("Generate Analysis"):
        if selected_title:
            hr_data = all_mlb_hrs[all_mlb_hrs['title'] == selected_title]
            
            if not hr_data.empty:
                hr_info = hr_data.iloc[0]
                
                # Create two columns for player profile and home run metrics
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    player_name = extract_player_name(hr_info['title'])
                    if player_name:
                        st.markdown("### 👤 Player Profile")
                        player_id = get_player_id(player_name, hr_info['season'])
                        
                        if player_id:
                            player_img = get_player_headshot(player_id)
                            if player_img:
                                st.image(player_img, width=200)
                            
                            player_details = get_player_details(player_id)
                            if player_details and 'people' in player_details:
                                player = player_details['people'][0]
                                
                                st.markdown(f"""
                                #### {player['fullName']} #{player.get('primaryNumber', 'N/A')}
                                - **Position:** {player['primaryPosition']['name']}
                                - **Bats/Throws:** {player['batSide']['description']}/{player['pitchHand']['description']}
                                - **Height/Weight:** {player['height']} / {player['weight']} lbs
                                - **Born:** {player['birthDate']}
                                - **Birthplace:** {player['birthCity']}, {player.get('birthStateProvince', '')}, {player['birthCountry']}
                                - **Strike Zone Top:** {player.get('strikeZoneTop', 'N/A')}
                                - **Strike Zone Bottom:** {player.get('strikeZoneBottom', 'N/A')}
                                """)

                with col2:
                    st.markdown("### 🎯 Home Run Metrics")
                    with st.container():
                        st.metric("Exit Velocity", f"{hr_info['ExitVelocity']} mph")
                        st.metric("Hit Distance", f"{hr_info['HitDistance']} ft")
                        st.metric("Launch Angle", f"{hr_info['LaunchAngle']}°")
                
                # Video Analysis Section Below
                if 'video' in hr_info and pd.notna(hr_info['video']):
                    st.markdown("### 🎥 Video Analysis")
                    result = process_video(hr_info['video'], hr_info['title'])
                    
                    if result:
                        label_annotations = result.annotation_results[0].segment_label_annotations
                        object_annotations = result.annotation_results[0].object_annotations
                        
                        bat_frames = []
                        pitch_segments = []
                        
                        for label_annotation in label_annotations:
                            if "pitch" in label_annotation.entity.description.lower():
                                for segment in label_annotation.segments:
                                    pitch_segments.append((segment.segment.start_time_offset, segment.segment.end_time_offset))

                        for object_annotation in object_annotations:
                            if object_annotation.entity.description.lower() == "baseball bat":
                                for frame in object_annotation.frames:
                                    frame_time = frame.time_offset
                                    for start_time, end_time in pitch_segments:
                                        if start_time <= frame_time <= end_time:
                                            bat_frames.append(frame)

                        s = 0
                        for i in range(len(bat_frames)):
                            if bat_frames[i].time_offset.total_seconds() > 7.8:
                                s = i
                                break
                        bat_frames = bat_frames[s:]
                        
                        e = len(bat_frames)
                        for i in range(1, len(bat_frames)):
                            time_diff = bat_frames[i].time_offset.total_seconds() - bat_frames[i-1].time_offset.total_seconds()
                            if time_diff > 0.3:
                                e = i
                                break
                        bat_frames = bat_frames[:e]
                        
                        if len(bat_frames) >= 2:
                            bat_metrics = calculate_bat_metrics(bat_frames)
                            
                            st.markdown("#### ⚾ Bat Swing Analysis")
                            metrics_col1, metrics_col2 = st.columns(2)
                            
                            with metrics_col1:
                                st.metric("Bat Swing Speed", f"{bat_metrics['swing_speed']:.1f} mph")
                                st.metric("Swing Duration", f"{bat_metrics['swing_duration']:.3f} s")
                                st.metric("Swing Angle", f"{bat_metrics['swing_angle']:.1f}°")
                            
                            with metrics_col2:
                                st.metric("Total Distance", f"{bat_metrics['total_distance']:.1f} ft")
                                st.metric("Vertical Distance", f"{bat_metrics['vertical_distance']:.1f} ft")
                                st.metric("Horizontal Distance", f"{bat_metrics['horizontal_distance']:.1f} ft")
                            
                            st.warning("⚠️ Note: These bat swing metrics are approximate and based on video analysis.")
                        else:
                            st.warning("Unable to track bat movement in this video. Please try another video for bat swing analysis.")
        else:
            st.error("Please select a home run play from the dropdown.")

if __name__ == "__main__":
    main()