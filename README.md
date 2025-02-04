# MLB Home Run Analysis

<a href="https://img.shields.io/badge/license-MIT-blue.svg" target="_blank" rel="noopener noreferrer">![Open Source License](LICENSE)</a>

## Introduction

Welcome to the MLB Home Run Analysis app! This application provides an in-depth analysis of Major League Baseball home runs using data from various sources, including video analysis and player statistics.

## Website

Visit the app: <a href="https://diamond-eye.streamlit.app/" target="_blank" rel="noopener noreferrer">MLB Home Run Analysis</a>

## Methodology

### Data Collection

1. **Home Run Data**: The app collects home run data from multiple CSV files stored in Google Cloud Storage. These files contain information about home runs, including the title, season, exit velocity, hit distance, launch angle, and video URLs.

2. **Player Data**: Player information is fetched from the MLB API, which includes details such as player ID, full name, position, batting and throwing hands, height, weight, birth date, birthplace, and strike zone dimensions.

### Data Processing

1. **Data Normalization**: The JSON data from the MLB API is normalized into a structured format using the `pandas` library for easy manipulation and analysis.

2. **Player Name Extraction**: The app uses the spaCy NLP library to extract player names from the home run titles.

3. **Video Processing**: The app downloads and processes home run videos using the `moviepy` library. Videos are slowed down for better analysis, and the Google Cloud Video Intelligence API is used to annotate and track objects (specifically, the baseball bat).

### Metrics Calculation

1. **Bat Swing Metrics**: The app calculates various metrics related to the bat swing, including swing speed, swing angle, total distance, swing duration, horizontal distance, and vertical distance. These metrics provide insights into the player's performance and are displayed in the app.

2. **Home Run Metrics**: Key home run metrics such as exit velocity, hit distance, and launch angle are displayed for each selected home run play.

### Visualization

1. **Player Profile**: The app displays a detailed player profile, including a headshot, player details, and metrics.

2. **Video Analysis**: The app provides a slowed-down version of the home run video along with annotations and bat swing analysis.

## How to Use

1. Select a home run play from the dropdown menu.
2. Click the "Generate Analysis" button to display the player profile and home run metrics.
3. If available, watch the processed video and review the bat swing analysis.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
