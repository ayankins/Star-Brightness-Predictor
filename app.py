# app.py (production-ready for Render)
import os
from flask import Flask, render_template, request
import lightkurve as lk
import numpy as np
import joblib
import plotly.graph_objects as go
import json
import logging
import pandas as pd
from scipy.stats import kurtosis
from scipy.signal import find_peaks

app = Flask(__name__)

# Load the model
try:
    clf = joblib.load("star_classifier.pkl")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise e

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cache directory (ephemeral on Render)
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cached_lightcurve(star_name):
    cache_file = os.path.join(CACHE_DIR, f"{star_name.replace(' ', '_')}.fits")
    if os.path.exists(cache_file):
        app.logger.info(f"Loading cached light curve for {star_name}")
        try:
            # Read with 'time' as the time column
            lc = lk.LightCurve.read(cache_file, time_column='time')
            return lc
        except Exception as e:
            app.logger.error(f"Error reading cached file for {star_name}: {str(e)}")
            # Inspect the FITS file structure
            from astropy.io import fits
            with fits.open(cache_file) as hdul:
                app.logger.debug("FITS file structure: %s", hdul.info())
                if len(hdul) > 1:
                    app.logger.debug("Columns in HDU 1: %s", hdul[1].columns.names)
            # If reading fails, delete the cached file and re-download
            os.remove(cache_file)
            app.logger.info(f"Deleted corrupted cache file for {star_name}, re-downloading")
    # Download the light curve if cache doesn't exist or was deleted
    app.logger.info(f"Downloading light curve for {star_name}")
    lc = lk.search_lightcurve(star_name, mission="Kepler")[0].download()
    # Rename the time column in the underlying table to 'time'
    table = lc.to_table()
    if 'time' not in table.colnames:
        # Kepler light curves typically use 'TIME' as the time column
        if 'TIME' in table.colnames:
            table.rename_column('TIME', 'time')
        else:
            app.logger.warning("No 'TIME' column found in light curve table. Columns: %s", table.colnames)
    # Convert back to LightCurve and save
    lc = lk.LightCurve(time=table['time'], flux=table['flux'], meta=lc.meta)
    lc.write(cache_file, format='fits', overwrite=True)
    return lc

def get_features(lc):
    try:
        # Initial cleaning: remove NaNs and normalize
        lc = lc.remove_nans().normalize()
        app.logger.debug("Light curve after remove_nans: Time length %d, Flux length %d", len(lc.time.value), len(lc.flux.value))

        # Ensure lengths match after initial cleaning
        if len(lc.time.value) != len(lc.flux.value):
            raise ValueError(f"Light curve arrays mismatch after remove_nans: Time length {len(lc.time.value)}, Flux length {len(lc.flux.value)}")

        # Additional cleaning: remove infinite values
        flux = lc.flux.value
        time = lc.time.value
        mask = np.isfinite(flux) & np.isfinite(time)
        lc = lc[mask]
        app.logger.debug("Light curve after additional cleaning: Time length %d, Flux length %d", len(lc.time.value), len(lc.flux.value))

        # Check if the light curve is empty after cleaning
        if len(lc.time.value) == 0 or len(lc.flux.value) == 0:
            raise ValueError("Light curve is empty after cleaning")

        # Ensure lengths match after additional cleaning
        if len(lc.time.value) != len(lc.flux.value):
            raise ValueError(f"Light curve arrays mismatch after additional cleaning: Time length {len(lc.time.value)}, Flux length {len(lc.flux.value)}")

        # Periodogram and folding
        pg = lc.to_periodogram(method='lombscargle', minimum_period=0.1, maximum_period=100)
        period = pg.period_at_max_power
        app.logger.debug("Detected period: %s days", period.value)
        folded_lc = lc.fold(period=period, normalize_phase=True)
        flux = folded_lc.flux.value
        phase = folded_lc.time.value
        app.logger.debug("Folded light curve before cleaning: Phase length %d, Flux length %d", len(phase), len(flux))
        app.logger.debug("Folded phase sample: %s", phase[:5])
        app.logger.debug("Folded flux sample: %s", flux[:5])

        # Additional cleaning after folding
        mask = np.isfinite(flux) & np.isfinite(phase)
        flux = flux[mask]
        phase = phase[mask]
        app.logger.debug("Folded light curve after finite cleaning: Phase length %d, Flux length %d", len(phase), len(flux))

        # Force length alignment if there's a mismatch
        if len(phase) != len(flux):
            app.logger.warning("Mismatch after folding: Phase length %d, Flux length %d", len(phase), len(flux))
            min_len = min(len(phase), len(flux))
            phase = phase[:min_len]
            flux = flux[:min_len]
            app.logger.debug("Forced length alignment after folding: Phase length %d, Flux length %d", len(phase), len(flux))

        # Update folded_lc with cleaned arrays
        # Since folded_lc.time is a Quantity after folding (dimensionless phase), create a new Quantity
        from astropy.units import Quantity
        folded_lc.time = Quantity(phase)  # Phase is dimensionless after normalize_phase=True
        folded_lc.flux = Quantity(flux, unit=folded_lc.flux.unit)  # Preserve the flux unit

        # Final check before feature extraction
        app.logger.debug("Final folded light curve: Time length %d, Flux length %d", len(folded_lc.time.value), len(folded_lc.flux.value))
        if len(folded_lc.time.value) != len(folded_lc.flux.value):
            raise ValueError(f"Final folded light curve arrays mismatch: Time length {len(folded_lc.time.value)}, Flux length {len(folded_lc.flux.value)}")

        # Feature extraction
        period_val = period.value
        amplitude = (np.max(flux) - np.min(flux)) / 2
        mean_flux = np.mean(flux)
        std_flux = np.std(flux)
        skewness = np.mean((flux - mean_flux)**3) / (std_flux**3)
        kurt = kurtosis(flux)
        max_power = pg.power.max().value
        freq_at_max_power = pg.frequency_at_max_power.value
        mad_flux = np.median(np.abs(flux - np.median(flux)))
        peaks, _ = find_peaks(flux, height=mean_flux + std_flux)
        num_peaks = len(peaks)

        transit_depth = None
        transit_duration = None
        if len(flux) > 0:
            transit_depth = np.max(flux) - np.min(flux)
            dip_indices = np.where(flux < (mean_flux - std_flux))[0]
            if len(dip_indices) > 0:
                dip_start = phase[dip_indices[0]]
                dip_end = phase[dip_indices[-1]]
                transit_duration = (dip_end - dip_start) * period_val * 24

        return (np.array([[
            period_val, amplitude, mean_flux, std_flux, skewness,
            kurt, max_power, freq_at_max_power, mad_flux, num_peaks
        ]]), folded_lc, period_val, transit_depth, transit_duration)
    except Exception as e:
        app.logger.error(f"Error in get_features: {str(e)}")
        raise e

def create_plotly_plot(folded_lc):
    try:
        app.logger.debug("Creating Plotly plot for %s", folded_lc.meta.get('OBJECT', 'Unknown'))
        time_data = folded_lc.time.value
        flux_data = folded_lc.flux.value
        app.logger.debug("Initial Time length: %d, Flux length: %d", len(time_data), len(flux_data))
        app.logger.debug("Time data sample: %s", time_data[:5])
        app.logger.debug("Flux data sample: %s", flux_data[:5])

        # Ensure initial lengths match
        if len(time_data) != len(flux_data):
            app.logger.error("Initial array size mismatch: Time length %d, Flux length %d", len(time_data), len(flux_data))
            raise ValueError(f"Initial array size mismatch: Time length {len(time_data)}, Flux length {len(flux_data)}")

        # Filter out non-finite values
        time_data = np.array(time_data)
        flux_data = np.array(flux_data)
        time_mask = np.isfinite(time_data)
        flux_mask = np.isfinite(flux_data)
        combined_mask = time_mask & flux_mask
        time_data = time_data[combined_mask]
        flux_data = flux_data[combined_mask]
        app.logger.debug("After filtering Time length: %d, Flux length: %d", len(time_data), len(flux_data))

        # Check lengths after filtering
        if len(time_data) != len(flux_data):
            app.logger.error("Array size mismatch after filtering: Time length %d, Flux length %d", len(time_data), len(flux_data))
            raise ValueError(f"Array size mismatch after filtering: Time length {len(time_data)}, Flux length {len(flux_data)}")

        # Check if arrays are empty
        if len(time_data) == 0 or len(flux_data) == 0:
            app.logger.error("Empty arrays after filtering: Time length %d, Flux length %d", len(time_data), len(flux_data))
            raise ValueError(f"Empty arrays after filtering: Time length {len(time_data)}, Flux length {len(flux_data)}")

        # Downsample if necessary
        if len(time_data) > 500:
            indices = np.linspace(0, len(time_data) - 1, 500, dtype=int)
            indices = indices[indices < len(time_data)]  # Ensure indices are valid
            time_data = time_data[indices]
            flux_data = flux_data[indices]
            app.logger.debug("Downsampled Time length: %d, Flux length: %d", len(time_data), len(flux_data))

        # Final length check before plotting
        if len(time_data) != len(flux_data):
            app.logger.error("Array size mismatch after downsampling: Time length %d, Flux length %d", len(time_data), len(flux_data))
            raise ValueError(f"Array size mismatch after downsampling: Time length {len(time_data)}, Flux length {len(flux_data)}")

        # Convert to lists for Plotly
        time_data = time_data.tolist()
        flux_data = flux_data.tolist()
        app.logger.debug("Final Time length: %d, Flux length: %d", len(time_data), len(flux_data))

        # Double-check for any Plotly-specific issues
        if len(time_data) == 0 or len(flux_data) == 0:
            app.logger.error("Empty arrays before plotting: Time length %d, Flux length %d", len(time_data), len(flux_data))
            raise ValueError(f"Empty arrays before plotting: Time length {len(time_data)}, Flux length {len(flux_data)}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_data,
            y=flux_data,
            mode='lines+markers',
            marker=dict(size=4, color='#6b48ff'),
            line=dict(color='#6b48ff', width=1),
            name='Flux'
        ))
        fig.update_layout(
            title=f"Light Curve for {folded_lc.meta.get('OBJECT', 'Unknown')}",
            xaxis_title="Phase",
            yaxis_title="Normalized Flux",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            showlegend=True,
            hovermode="closest",
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.2)')
        )
        plot_json = json.dumps(fig.to_dict())
        app.logger.debug("Plot JSON: %s", plot_json[:100])
        return plot_json
    except Exception as e:
        app.logger.error(f"Error in create_plotly_plot: {str(e)}")
        raise e

@app.route("/", methods=["GET", "POST"])
def index():
    app.logger.info("Accessing index route")
    if request.method == "POST":
        star_name = request.form["star_name"]
        app.logger.info("Processing star: %s", star_name)
        try:
            lc = get_cached_lightcurve(star_name)
            features, folded_lc, period, transit_depth, transit_duration = get_features(lc)
            feature_names = ['period', 'amplitude', 'mean_flux', 'std_flux', 'skewness',
                             'kurtosis', 'max_power', 'freq_at_max_power', 'mad_flux', 'num_peaks']
            features_df = pd.DataFrame(features, columns=feature_names)
            prediction = clf.predict(features_df)[0]
            confidence = clf.predict_proba(features_df)[0].max()
            importances = clf.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values(by='importance', ascending=False).to_dict('records')
            plot_data = create_plotly_plot(folded_lc)
            # Check if plot_data is empty (indicating a failure)
            if plot_data == json.dumps({}):
                return render_template("index.html", prediction=prediction,
                                     star_name=star_name, feature_importance=feature_importance,
                                     period=period, transit_depth=transit_depth,
                                     transit_duration=transit_duration, confidence=confidence,
                                     error="Unable to generate plot: No valid data points after processing.")
            return render_template("index.html", prediction=prediction, plot_data=plot_data,
                                 star_name=star_name, feature_importance=feature_importance,
                                 period=period, transit_depth=transit_depth,
                                 transit_duration=transit_duration, confidence=confidence)
        except Exception as e:
            app.logger.error(f"Error processing {star_name}: {str(e)}")
            return render_template("index.html", error=f"Error processing {star_name}: {str(e)}")
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render uses PORT env variable
    app.run(host="0.0.0.0", port=port)