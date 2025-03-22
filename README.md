# 🌟 Star Brightness Predictor

Welcome to the **Star Brightness Predictor**, a web application that analyzes light curves from Kepler stars to predict their variability type, such as Transiting Exoplanets or Irregular Variables. This project uses machine learning and the `lightkurve` library to process and visualize stellar light curves, providing insights into the behavior of stars observed by the Kepler Space Telescope.


## 📋 Project Overview

The Star Brightness Predictor is a Flask-based web application that:
- Downloads light curve data for a given Kepler star using the `lightkurve` library.
- Extracts features from the light curve (e.g., period, amplitude, skewness).
- Uses a pre-trained machine learning model to classify the star’s variability type.
- Visualizes the folded light curve using an interactive Plotly plot, showing the star’s brightness variation over phase.

This project is ideal for astronomy enthusiasts, researchers, and students interested in stellar variability and exoplanet detection.

## ✨ Features

- **Star Classification**: Predicts whether a star exhibits behavior like a Transiting Exoplanet, Irregular Variable, or other variability types.
- **Interactive Visualization**: Displays an interactive Plotly plot of the star’s folded light curve, allowing users to zoom and pan.
- **User-Friendly Interface**: Simple web interface to input a Kepler star name (e.g., `Kepler-8`) and view the results.
- **Real-Time Analysis**: Downloads and processes light curve data in real-time from the Kepler archive.

## 🛠️ Installation

Follow these steps to set up the project on your local machine.

### Prerequisites
- **Python 3.8+**: Ensure Python is installed on your system.
- **Git**: To clone the repository.
- **Virtual Environment (Optional but Recommended)**: To manage dependencies.

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/star-brightness-predictor.git
   cd star-brightness-predictor


   
---

### Step 2: Copy and Paste into GitHub
**What We’re Doing**: You can now copy the entire snippet above and paste it into your GitHub repository’s `README.md` file.

**Action**:
1. **Go to Your GitHub Repository**:
   - Navigate to `https://github.com/your-username/star-brightness-predictor` in your browser.
   - If you don’t have a `README.md` file yet, GitHub will prompt you to create one. If you already have one, click on `README.md` to edit it.

2. **Edit the `README.md` File**:
   - Click the pencil icon to edit the file.
   - Delete the existing content (if any).
   - Paste the entire snippet from above into the editor.

3. **Update Placeholder Values**:
   - Replace `https://github.com/your-username/star-brightness-predictor.git` with the actual URL of your GitHub repository.
   - Replace `path/to/star-brightness-predictor` in the "Running the App Locally" section with the actual path to your project directory (e.g., `X:\Download\star-brightness-predictor`).

4. **Commit the Changes**:
   - Scroll to the bottom of the page.
   - Add a commit message, such as "Updated README with complete instructions".
   - Click "Commit changes".

---

### Step 3: Verify the `requirements.txt` File
**What We’re Doing**: The `README.md` references a `requirements.txt` file. Let’s ensure it exists in your repository.

**Action**:
- **Check for `requirements.txt` in Your Project Directory**:
  - Navigate to `X:\Download\star_brightness_predictor`.
  - If `requirements.txt` doesn’t exist, create it with the following content:
    ```plaintext
    flask==2.0.1
    lightkurve==2.4.2
    numpy==1.23.5
    pandas==1.5.0
    joblib==1.2.0
    plotly==5.10.0
    scikit-learn==1.1.2
    scipy==1.9.1
