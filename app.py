from flask import Flask, render_template, request
import lightkurve as lk
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.io as pio
import logging
import timeout_decorator 

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the trained model
try:
    model = joblib.load('star_classifier.pkl')
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    plot = None
    star_name = None

    if request.method == 'POST':
        star_name = request.form['star_name']
        logger.info(f"Processing star: {star_name}")

        try:
            # Download the light curve with a timeout
            @timeout_decorator.timeout(20, timeout_exception=TimeoutError)
            def download_lightcurve():
                search_result = lk.search_lightcurve(star_name, mission='Kepler')
                if len(search_result) == 0:
                    return None
                lc = search_result[0].download()
                return lc.normalize()

            lc = download_lightcurve()
            if lc is None:
                prediction = f"No data found for {star_name}"
                logger.warning(f"No data found for {star_name}")
            else:
                # Extract features (period and amplitude)
                periodogram = lc.to_periodogram()
                period = periodogram.period_at_max_power.value
                flux = lc.flux.value
                amplitude = np.max(flux) - np.min(flux)

                # Prepare features for prediction
                features = np.array([[period, amplitude]])
                prediction = model.predict(features)[0]
                logger.info(f"Prediction for {star_name}: {prediction}")

                # Create a Plotly plot of the light curve
                fig = px.scatter(x=lc.time.value, y=lc.flux.value, labels={'x': 'Time (days)', 'y': 'Normalized Flux'},
                                 title=f'Light Curve for {star_name}')
                plot = pio.to_html(fig, full_html=False)

        except TimeoutError:
            prediction = f"Error: Processing {star_name} took too long. Please try again later."
            logger.error(f"Timeout while processing {star_name}")
        except Exception as e:
            prediction = f"Error processing {star_name}: {str(e)}"
            logger.error(f"Error processing {star_name}: {str(e)}")

    return render_template('index.html', prediction=prediction, star_name=star_name, plot=plot)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=False)