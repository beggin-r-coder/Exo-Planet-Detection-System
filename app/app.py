import os
import json
import io
import base64
from typing import Dict, Any, Tuple
from datetime import datetime

from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

try:
    import lightkurve as lk
    LIGHTKURVE_AVAILABLE = True
except ImportError:
    LIGHTKURVE_AVAILABLE = False
    print("Warning: lightkurve library not available. Install it with: pip install lightkurve")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_CACHE_PATH = os.path.join(DATA_DIR, "upload_last.csv")
SYNTHETIC_DATA_PATH = os.path.join(DATA_DIR, "synthetic_tess_data.csv")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
MODEL_PATH = os.path.join(MODELS_DIR, "trained_model.pkl")
EXOPLANETS_DATA_PATH = os.path.join(DATA_DIR, "exoplanets.csv")

# Load exoplanets data
exoplanets_df = None


app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))
app.config["SECRET_KEY"] = "dev"


def load_exoplanets_data():
    """Load exoplanets dataset"""
    global exoplanets_df
    try:
        exoplanets_df = pd.read_csv(EXOPLANETS_DATA_PATH)
        print(f"Loaded {len(exoplanets_df)} exoplanet records")
    except Exception as e:
        print(f"Error loading exoplanets data: {e}")
        exoplanets_df = pd.DataFrame()

def search_stars(query):
    """Search for stars by name"""
    if exoplanets_df is None or exoplanets_df.empty:
        print("Exoplanets dataframe is empty or None")
        return []
    
    try:
        # Search in kepler_name, kepoi_name columns
        query_lower = query.lower()
        
        # Create masks for searching
        kepler_mask = exoplanets_df['kepler_name'].str.lower().str.contains(query_lower, na=False)
        kepoi_mask = exoplanets_df['kepoi_name'].str.lower().str.contains(query_lower, na=False)
        
        # Combine masks
        combined_mask = kepler_mask | kepoi_mask
        
        # Get results
        results_df = exoplanets_df[combined_mask].head(20)  # Limit to 20 results
        
        print(f"Found {len(results_df)} results for query: {query}")
        
        # Replace NaN values with None to make it JSON serializable
        results = results_df.to_dict('records')
        # Replace all NaN values with None in each record
        for record in results:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
        
        return results
        
    except Exception as e:
        print(f"Error in search_stars: {e}")
        import traceback
        traceback.print_exc()
        return []

def get_star_details(star_id):
    """Get detailed information for a specific star"""
    if exoplanets_df is None or exoplanets_df.empty:
        return None
    
    try:
        # Search by kepler_name or kepoi_name
        star_data = exoplanets_df[
            (exoplanets_df['kepler_name'] == star_id) |
            (exoplanets_df['kepoi_name'] == star_id)
        ]
        
        if star_data.empty:
            return None
        
        # Replace NaN values with None to make it JSON serializable
        star_dict = star_data.iloc[0].to_dict()
        # Replace all NaN values with None
        star_dict = {k: (None if pd.isna(v) else v) for k, v in star_dict.items()}
        
        return star_dict
    except Exception as e:
        print(f"Error in get_star_details: {e}")
        import traceback
        traceback.print_exc()
        return None


def count_exoplanets_for_kic(kepic_id):
    """Count the number of confirmed exoplanets for a given Kepler ID (KIC)"""
    if exoplanets_df is None or exoplanets_df.empty:
        return 0, 0, 0
    
    try:
        # Convert kepic_id to string/int for matching
        if isinstance(kepic_id, str):
            try:
                kepic_id = int(kepic_id.strip())
            except ValueError:
                pass
        
        # Filter by kepid column
        matching_planets = exoplanets_df[exoplanets_df['kepid'] == kepic_id]
        
        if matching_planets.empty:
            return 0, 0, 0
        
        # Count by disposition
        total_count = len(matching_planets)
        confirmed_count = len(matching_planets[matching_planets['koi_disposition'] == 'CONFIRMED'])
        candidate_count = len(matching_planets[matching_planets['koi_disposition'] == 'CANDIDATE'])
        
        return total_count, confirmed_count, candidate_count
        
    except Exception as e:
        print(f"Error counting exoplanets for KIC {kepic_id}: {e}")
        return 0, 0, 0


last_stats: Dict[str, Any] = {}


def ensure_directories() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)


def ensure_synthetic_dataset() -> None:
    if os.path.exists(SYNTHETIC_DATA_PATH):
        return
    # Create a small synthetic light curve with a subtle transit-like dip
    rng = np.random.default_rng(42)
    time = np.arange(0, 20, 0.1)
    flux = 1.0 + rng.normal(0, 0.0015, size=time.shape[0])
    # Inject a transit dip between t in [8, 10]
    transit_mask = (time >= 8.0) & (time <= 10.0)
    flux[transit_mask] -= 0.004 + rng.normal(0, 0.0005, size=transit_mask.sum())
    df = pd.DataFrame({"time": time, "flux": flux})
    df.to_csv(SYNTHETIC_DATA_PATH, index=False)


def bootstrap_model_and_scaler() -> Tuple[LogisticRegression, StandardScaler]:
    """
    Load existing scaler/model if available; otherwise train a very simple model
    using the synthetic dataset as a stand-in and persist them for reuse.
    The model is intentionally simple and is not a scientific detector.
    """
    ensure_directories()
    ensure_synthetic_dataset()

    if os.path.exists(SCALER_PATH) and os.path.exists(MODEL_PATH):
        scaler: StandardScaler = joblib.load(SCALER_PATH)
        model: LogisticRegression = joblib.load(MODEL_PATH)
        return model, scaler

    # Train a trivial model: classify points as transit (1) if flux < mean - k*std
    data = pd.read_csv(SYNTHETIC_DATA_PATH)
    flux = data["flux"].to_numpy()
    mean_flux = float(np.mean(flux))
    std_flux = float(np.std(flux) + 1e-8)
    k = 0.75
    labels = (flux < (mean_flux - k * std_flux)).astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(flux.reshape(-1, 1))

    # Logistic regression on single feature (scaled flux)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(model, MODEL_PATH)
    return model, scaler


model, scaler = bootstrap_model_and_scaler()


def load_dataframe_for_visualization() -> pd.DataFrame:
    if os.path.exists(UPLOAD_CACHE_PATH):
        try:
            return pd.read_csv(UPLOAD_CACHE_PATH)
        except Exception:
            pass
    return pd.read_csv(SYNTHETIC_DATA_PATH)


def compute_stats(df: pd.DataFrame, proba: np.ndarray) -> Dict[str, Any]:
    detection_confidence = float(np.mean(proba) * 100.0)
    min_flux = float(df["flux"].min())
    avg_flux = float(df["flux"].mean())
    detected = detection_confidence >= 50.0
    # Transit parameters
    transit = estimate_transit_parameters(df)
    return {
        "detected": detected,
        "detection_confidence": round(detection_confidence, 2),
        "min_flux": round(min_flux, 6),
        "avg_flux": round(avg_flux, 6),
        "count": int(df.shape[0]),
        "estimated_period": transit.get("period"),
        "transit_depth": transit.get("depth"),
        "transit_duration": transit.get("duration"),
    }


def estimate_transit_parameters(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Heuristic estimation of transit parameters from a single light curve:
    - period: median spacing between detected dip centers
    - depth: median depth of dips below baseline
    - duration: median dip width (in time units of the input "time")

    This is a simplistic method intended for demo purposes, not for scientific use.
    """
    try:
        time = df["time"].to_numpy()
        flux = df["flux"].to_numpy()
        if len(time) < 10:
            return {"period": None, "depth": None, "duration": None}

        # Estimate baseline from upper quantile to avoid dip influence
        upper_q = np.quantile(flux, 0.8)
        baseline_candidates = flux[flux >= upper_q]
        baseline = float(np.median(baseline_candidates)) if baseline_candidates.size else float(np.median(flux))

        # Estimate noise level from non-dip region
        noise_region = flux[flux >= upper_q]
        noise_std = float(np.std(noise_region)) if noise_region.size else float(np.std(flux))
        noise_std = max(noise_std, 1e-6)

        threshold = baseline - 3.0 * noise_std

        # Identify contiguous dip segments where flux < threshold
        below = flux < threshold
        if not np.any(below):
            return {"period": None, "depth": None, "duration": None}

        segments = []  # list of (start_idx, end_idx)
        in_seg = False
        start = 0
        for i, flag in enumerate(below):
            if flag and not in_seg:
                in_seg = True
                start = i
            elif not flag and in_seg:
                in_seg = False
                segments.append((start, i - 1))
        if in_seg:
            segments.append((start, len(below) - 1))

        if not segments:
            return {"period": None, "depth": None, "duration": None}

        centers = []
        durations = []
        depths = []
        for s, e in segments:
            seg_flux = flux[s : e + 1]
            seg_time = time[s : e + 1]
            if seg_time.size == 0:
                continue
            min_idx = int(np.argmin(seg_flux))
            centers.append(float(seg_time[min_idx]))
            durations.append(float(seg_time[-1] - seg_time[0]))
            depths.append(float(baseline - np.min(seg_flux)))

        period = None
        if len(centers) >= 2:
            centers_sorted = np.sort(np.array(centers))
            diffs = np.diff(centers_sorted)
            if diffs.size:
                period = float(np.median(diffs))

        depth = float(np.median(depths)) if depths else None
        duration = float(np.median(durations)) if durations else None

        # Round for presentation
        def r(x):
            return None if x is None else round(x, 6)

        return {"period": r(period), "depth": r(depth), "duration": r(duration)}
    except Exception:
        return {"period": None, "depth": None, "duration": None}


@app.route("/search")
def search_page():
    return render_template("search.html")

@app.route("/api/search_stars")
def api_search_stars():
    query = request.args.get('q', '').strip()
    print(f"Search query received: '{query}'")
    
    if not query:
        return jsonify([])
    
    try:
        results = search_stars(query)
        print(f"Returning {len(results)} results")
        return jsonify(results)
    except Exception as e:
        print(f"Error in api_search_stars: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/star/<star_id>")
def star_details(star_id):
    star_data = get_star_details(star_id)
    if not star_data:
        return render_template("star_not_found.html", star_id=star_id)
    
    return render_template("star_details.html", star=star_data)


@app.route("/kepler")
def kepler_search():
    return render_template("kepler_search.html")


def fetch_kepler_lightcurve(kepler_id):
    """Fetch light curve data from Kepler dataset using lightkurve"""
    if not LIGHTKURVE_AVAILABLE:
        return None, "Lightkurve library is not installed. Please install it with: pip install lightkurve"
    
    try:
        # Search for the target
        search_result = lk.search_targetpixelfile(f"KIC {kepler_id}", mission="Kepler")
        
        if len(search_result) == 0:
            return None, f"No data found for Kepler ID {kepler_id}"
        
        # Download the first result
        tpf = search_result.download(quality_bitmask='default')
        
        # Create light curve
        lc = tpf.to_lightcurve(aperture_mask=tpf.pipeline_mask)
        
        # Remove outliers and normalize
        lc = lc.remove_outliers()
        lc = lc.normalize()
        
        # Get time and flux arrays
        time = lc.time.value.tolist()
        flux = lc.flux.value.tolist()
        
        # Get star metadata
        try:
            magnitude = lc.meta.get('KEPLERMAG') if hasattr(lc.meta, 'get') else None
            if magnitude is not None:
                try:
                    magnitude = float(magnitude)
                except (ValueError, TypeError):
                    magnitude = None
        except:
            magnitude = None
        
        # Count exoplanets for this KIC
        total_exoplanets, confirmed_exoplanets, candidate_exoplanets = count_exoplanets_for_kic(kepler_id)
        
        star_info = {
            'kepler_id': kepler_id,
            'ra': float(tpf.ra) if hasattr(tpf, 'ra') and tpf.ra is not None else None,
            'dec': float(tpf.dec) if hasattr(tpf, 'dec') and tpf.dec is not None else None,
            'magnitude': magnitude,
            'data_points': len(time),
            'time_range': {
                'start': float(min(time)) if time else None,
                'end': float(max(time)) if time else None
            },
            'exoplanets': {
                'total': total_exoplanets,
                'confirmed': confirmed_exoplanets,
                'candidates': candidate_exoplanets
            },
            'flux_stats': {
                'mean': float(np.nanmean(flux)) if flux else None,
                'min': float(np.nanmin(flux)) if flux else None,
                'max': float(np.nanmax(flux)) if flux else None,
                'std': float(np.nanstd(flux)) if flux else None
            }
        }
        
        # Replace NaN values
        star_info = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in star_info.items()}
        
        return {
            'time': time[:500],  # Limit to 500 points for performance
            'flux': flux[:500],
            'star_info': star_info
        }, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error fetching Kepler data: {str(e)}"


@app.route("/api/kepler_lightcurve")
def api_kepler_lightcurve():
    """API endpoint to fetch Kepler light curve data"""
    kepler_id = request.args.get('id', '').strip()
    
    if not kepler_id:
        return jsonify({"error": "Kepler ID is required"}), 400
    
    if not LIGHTKURVE_AVAILABLE:
        return jsonify({"error": "Lightkurve library is not available. Please install it first."}), 503
    
    try:
        data, error = fetch_kepler_lightcurve(kepler_id)
        
        if error:
            return jsonify({"error": error}), 404
        
        # Replace NaN values in time and flux arrays
        data['time'] = [t if not (isinstance(t, float) and np.isnan(t)) else None for t in data['time']]
        data['flux'] = [f if not (isinstance(f, float) and np.isnan(f)) else None for f in data['flux']]
        
        return jsonify(data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route("/")
def index():
    return render_template("index.html", result=None, stats=None, error=None)


@app.route("/predict", methods=["POST"])
def predict():
    global last_stats
    if "file" not in request.files:
        return render_template("index.html", result=None, stats=None, error="No file part in request."), 400

    file = request.files["file"]
    if file.filename == "":
        return render_template("index.html", result=None, stats=None, error="No file selected."), 400

    if not file.filename.lower().endswith(".csv"):
        return render_template("index.html", result=None, stats=None, error="Invalid file type. Please upload a .csv file."), 400

    try:
        df = pd.read_csv(file)
    except Exception as exc:
        return render_template("index.html", result=None, stats=None, error=f"Failed to parse CSV: {exc}"), 400

    if "flux" not in df.columns:
        return render_template("index.html", result=None, stats=None, error="CSV must contain a 'flux' column."), 400

    if "time" not in df.columns:
        df["time"] = np.arange(len(df))

    # Clean data: drop NaNs and sort by time if present
    df = df[["time", "flux"]].dropna()
    try:
        df = df.sort_values("time")
    except Exception:
        pass

    # Persist for visualization
    try:
        df.to_csv(UPLOAD_CACHE_PATH, index=False)
    except Exception:
        pass

    X = scaler.transform(df["flux"].to_numpy().reshape(-1, 1))
    # Probability of class 1 (transit)
    proba = model.predict_proba(X)[:, 1]

    stats = compute_stats(df, proba)
    last_stats = stats

    result_text = "Exoplanet Detected" if stats["detected"] else "No Exoplanet Detected"
    return render_template("index.html", result=result_text, stats=stats, error=None)


@app.route("/results")
def results_page():
    return render_template("results.html")


@app.route("/visualize")
def visualize():
    try:
        df = load_dataframe_for_visualization()
        df = df[["time", "flux"]].dropna()
        df = df.sort_values("time")
        df = df.head(200)
        payload = {
            "time": df["time"].tolist(),
            "flux": df["flux"].tolist(),
        }
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": f"Failed to load data for visualization: {exc}"}), 500


@app.route("/summary")
def summary():
    global last_stats
    # If we don't have last_stats yet, derive from current visualization dataframe
    if not last_stats:
        try:
            df = load_dataframe_for_visualization()
            X = scaler.transform(df["flux"].to_numpy().reshape(-1, 1))
            proba = model.predict_proba(X)[:, 1]
            last_stats = compute_stats(df, proba)
        except Exception:
            last_stats = {"detected": False, "detection_confidence": 0.0, "min_flux": None, "avg_flux": None}
    return jsonify(last_stats)


def generate_chart_image():
    """Generate a matplotlib chart and return as base64 string"""
    try:
        df = load_dataframe_for_visualization()
        df = df[["time", "flux"]].dropna()
        df = df.sort_values("time")
        df = df.head(200)
        
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot flux data
        ax.plot(df["time"], df["flux"], color='#60a5fa', linewidth=2, alpha=0.8, label='Flux')
        
        # Add average line
        avg_flux = df["flux"].mean()
        ax.axhline(y=avg_flux, color='#94a3b8', linestyle='--', alpha=0.7, label=f'Average: {avg_flux:.4f}')
        
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('Flux', color='white')
        ax.set_title('Flux vs Time - Light Curve Analysis', color='white', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Save to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', 
                   facecolor='#0f172a', edgecolor='none')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return img_base64
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None


@app.route("/download_sample/<sample_type>")
def download_sample(sample_type):
    """Download sample CSV files for testing"""
    try:
        sample_files = {
            'strong': 'synthetic_tess_data_planet.csv',
            'shallow': 'synthetic_tess_data_planet_shallow.csv', 
            'baseline': 'synthetic_tess_data.csv'
        }
        
        if sample_type not in sample_files:
            return jsonify({"error": "Invalid sample type"}), 400
            
        file_path = os.path.join(DATA_DIR, sample_files[sample_type])
        
        if not os.path.exists(file_path):
            return jsonify({"error": "Sample file not found"}), 404
            
        return send_file(
            file_path,
            as_attachment=True,
            download_name=f"sample_{sample_type}_data.csv",
            mimetype='text/csv'
        )
        
    except Exception as e:
        return jsonify({"error": f"Failed to download sample: {str(e)}"}), 500


@app.route("/download_report")
def download_report():
    """Generate and download PDF report"""
    try:
        # Get current stats
        global last_stats
        if not last_stats:
            try:
                df = load_dataframe_for_visualization()
                X = scaler.transform(df["flux"].to_numpy().reshape(-1, 1))
                proba = model.predict_proba(X)[:, 1]
                last_stats = compute_stats(df, proba)
            except Exception:
                last_stats = {"detected": False, "detection_confidence": 0.0, "min_flux": None, "avg_flux": None}
        
        # Generate chart
        chart_img = generate_chart_image()
        
        # Create PDF
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, 
                              topMargin=72, bottomMargin=18)
        story = []
        
        # Styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                   fontSize=24, spaceAfter=30, alignment=1, textColor=colors.darkblue)
        heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], 
                                    fontSize=16, spaceAfter=12, textColor=colors.darkblue)
        normal_style = ParagraphStyle('CustomNormal', parent=styles['Normal'], 
                                   fontSize=12, spaceAfter=6)
        
        # Title
        story.append(Paragraph("Exoplanet Detection Analysis Report", title_style))
        story.append(Spacer(1, 20))
        
        # Report info
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        story.append(Spacer(1, 20))
        
        # Analysis Results
        story.append(Paragraph("Analysis Results", heading_style))
        story.append(Paragraph(f"<b>Detection Status:</b> {'Exoplanet Detected' if last_stats.get('detected', False) else 'No Exoplanet Detected'}", normal_style))
        story.append(Paragraph(f"<b>Detection Confidence:</b> {last_stats.get('detection_confidence', 0):.2f}%", normal_style))
        story.append(Paragraph(f"<b>Average Flux:</b> {last_stats.get('avg_flux', 0):.6f}", normal_style))
        story.append(Paragraph(f"<b>Minimum Flux:</b> {last_stats.get('min_flux', 0):.6f}", normal_style))
        story.append(Paragraph(f"<b>Data Points Analyzed:</b> {last_stats.get('count', 0)}", normal_style))
        
        if last_stats.get('estimated_period'):
            story.append(Paragraph(f"<b>Estimated Period:</b> {last_stats.get('estimated_period', 0):.6f}", normal_style))
        if last_stats.get('transit_depth'):
            story.append(Paragraph(f"<b>Transit Depth:</b> {last_stats.get('transit_depth', 0):.6f}", normal_style))
        if last_stats.get('transit_duration'):
            story.append(Paragraph(f"<b>Transit Duration:</b> {last_stats.get('transit_duration', 0):.6f}", normal_style))
        
        story.append(Spacer(1, 20))
        
        # Chart
        if chart_img:
            story.append(Paragraph("Light Curve Visualization", heading_style))
            img_data = base64.b64decode(chart_img)
            img_buffer = io.BytesIO(img_data)
            img = Image(img_buffer, width=6*inch, height=3.6*inch)
            story.append(img)
            story.append(Spacer(1, 20))
        
        # Analysis
        story.append(Paragraph("Analysis Summary", heading_style))
        conf_val = last_stats.get('detection_confidence', 0)
        interpretation = "High detection probability. Periodic dips suggest a likely transit." if conf_val >= 50 else "Low detection probability. Flux variations may be due to stellar activity."
        story.append(Paragraph(f"<b>Interpretation:</b> {interpretation}", normal_style))
        story.append(Paragraph("This analysis uses transit photometry to detect exoplanets by identifying periodic dips in stellar brightness caused by planetary transits.", normal_style))
        
        # Build PDF
        doc.build(story)
        buffer.seek(0)
        
        return send_file(
            io.BytesIO(buffer.getvalue()),
            as_attachment=True,
            download_name=f"exoplanet_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        return jsonify({"error": f"Failed to generate report: {str(e)}"}), 500


if __name__ == "__main__":
    ensure_directories()
    ensure_synthetic_dataset()
    load_exoplanets_data()
    app.run(host="127.0.0.1", port=5000, debug=True)


