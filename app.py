from flask import Flask, render_template, request
import lightkurve as lk
import numpy as np
import pandas as pd
import joblib
import plotly
import plotly.graph_objs as go
import json
from scipy.stats import skew, kurtosis

app = Flask(__name__)

# Load the trained model
model = joblib.load('star_classifier.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    plot_data = None
    star_name = None
    error = None

    if request.method == 'POST':
        star_name = request.form['star_name']
        print(f"Processing star: {star_name}")

        try:
            # Download the light curve
            search_result = lk.search_lightcurve(star_name, mission='Kepler')
            if len(search_result) == 0:
                error = f"No data found for {star_name}"
            else:
                lc = search_result[0].download()
                lc = lc.normalize()

                # Extract flux and time, and clean the data
                flux = np.array(lc.flux.value, dtype=np.float64)  # Convert MaskedNDArray to numpy array
                time = np.array(lc.time.value, dtype=np.float64)
                # Remove nan and inf values
                valid_mask = np.isfinite(flux) & np.isfinite(time)
                flux = flux[valid_mask]
                time = time[valid_mask]

                if len(flux) == 0:
                    error = f"No valid data points after cleaning for {star_name}"
                else:
                    # Extract features (10 features to match the model)
                    periodogram = lc.to_periodogram()
                    period = float(periodogram.period_at_max_power.value)
                    amplitude = float(np.max(flux) - np.min(flux))
                    mean_flux = float(np.mean(flux))
                    std_flux = float(np.std(flux))
                    skewness = float(skew(flux))
                    kurt = float(kurtosis(flux))
                    max_power = float(periodogram.power.value.max())
                    freq_at_max_power = float(periodogram.frequency_at_max_power.value)
                    mad_flux = float(np.median(np.abs(flux - np.median(flux))))
                    # Count significant peaks in the periodogram
                    power_threshold = np.mean(periodogram.power.value) + 2 * np.std(periodogram.power.value)
                    num_peaks = float(np.sum(periodogram.power.value > power_threshold))

                    # Prepare features for prediction (10 features) with feature names in the correct order
                    feature_names = ['period', 'amplitude', 'mean_flux', 'std_flux', 'skewness', 'kurtosis', 'max_power', 'freq_at_max_power', 'mad_flux', 'num_peaks']
                    features = np.array([[period, amplitude, mean_flux, std_flux, skewness, kurt, max_power, freq_at_max_power, mad_flux, num_peaks]])
                    features_df = pd.DataFrame(features, columns=feature_names)
                    print(f"Features extracted: {features}")

                    # Make prediction
                    prediction = model.predict(features_df)[0]
                    print(f"Prediction for {star_name}: {prediction}")

                    # Option 1: Fold the light curve using the period (Phase plot)
                    phase = (time / period) % 1  # Normalize to [0, 1]
                    phase = phase - 0.5  # Shift to [-0.5, 0.5] for better centering

                    # Sort the data by phase for plotting with lines
                    sorted_indices = np.argsort(phase)
                    phase = phase[sorted_indices]
                    flux_phase = flux[sorted_indices]

                    # Option 2: Use the raw light curve (Time plot)
                    time_sorted_indices = np.argsort(time)
                    time_sorted = time[time_sorted_indices]
                    flux_time = flux[time_sorted_indices]

                    # Create an interactive Plotly plot (choose one of the options below)
                    # Option 1: Phase plot (folded light curve)
                    x_data = phase.tolist()  # Use phase for folded light curve
                    flux_data = flux_phase.tolist()
                    x_label = 'Phase'

                    # Option 2: Time plot (raw light curve) - Uncomment to use this instead
                    # x_data = time_sorted.tolist()  # Use time for raw light curve
                    # flux_data = flux_time.tolist()
                    # x_label = 'Time (days)'

                    print(f"Number of data points: {len(x_data)}")
                    if len(x_data) > 0 and len(flux_data) > 0:
                        trace = go.Scatter(
                            x=x_data,
                            y=flux_data,
                            mode='lines+markers',  # Connect points with lines to match the desired style
                            marker=dict(
                                size=6,  # Larger markers for a smoother look
                                opacity=0.8  # Higher opacity to blend points together
                            ),
                            line=dict(
                                width=1  # Slightly thicker lines for a cohesive look
                            ),
                            name='Light Curve'
                        )
                        layout = go.Layout(
                            title=f'Light Curve for {star_name}',
                            xaxis=dict(title=x_label),
                            yaxis=dict(title='Normalized Flux'),
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            showlegend=True
                        )
                        fig = go.Figure(data=[trace], layout=layout)
                        plot_data = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                        print(f"Plot data generated: {plot_data[:100]}...")  # Log first 100 characters of plot_data
                    else:
                        error = f"No valid data points to plot for {star_name}"

        except Exception as e:
            error = str(e)
            print(f"Error: {error}")

    return render_template('index.html', prediction=prediction, star_name=star_name, plot_data=plot_data, error=error)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)