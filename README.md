🌌 Exoplanet Detection System using Machine Learning & Flask

A machine learning–powered web application that analyzes stellar light curve data to detect potential exoplanets using the transit photometry method.
Built with Python, scikit-learn, Flask, Matplotlib, and ReportLab, this project demonstrates how data science and astronomy can be combined to identify planetary transit events.

🚀 Features

📁 CSV Upload Interface to analyze custom light curve datasets
🤖 Machine Learning Model (Logistic Regression) for transit detection
📉 Flux vs Time Visualization with automatic graph generation
📊 Statistical Analysis: transit depth, duration, period, min/avg flux
📄 Automated PDF Report Generation with plots and confidence metrics
🔍 Search Functionality for Kepler/TESS star catalog metadata
💻 Flask Web Application with an intuitive user interface

🪐 How It Works
This project uses the transit photometry technique, which detects exoplanets by identifying dips in a star’s brightness when a planet crosses in front of it.

🔭 Workflow
User uploads a .csv file containing time and flux columns
Data is validated, cleaned, and scaled
The logistic regression model predicts probabilities of transit events
Transit-related metrics (depth, period, duration) are calculated
A light curve plot is generated

The system displays results + offers a downloadable PDF report

⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/your-username/Exoplanet-Detection-System.git
cd Exoplanet-Detection-System

2. Create a Virtual Environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt

4. Run the Application
python app.py

Visit the app at http://127.0.0.1:5000/

📘 Example Usage
Upload any .csv light curve dataset
View the plotted graph of flux vs time
Check detection confidence & transit parameters
Download the automatically generated PDF report for documentation

🧠 Machine Learning Model
The system uses a Logistic Regression classifier trained on synthetic TESS-like data containing artificial transit dips.

Why Logistic Regression?
Simple, interpretable, and lightweight
Performs well on clean synthetic signals
Ideal for educational and demonstration purposes

Future versions will integrate:
CNNs for pattern detection
RNNs/LSTMs for time-series learning
Real Kepler/TESS telescope datasets

💡 Future Enhancements
Integrate real NASA Kepler/TESS datasets
Improve accuracy with deep learning models
Cloud deployment (AWS/GCP/Azure)
Real-time data stream support
Advanced 3D visualizations

🙌 Acknowledgments
NASA Exoplanet Archive
TESS & Kepler Mission Teams
scikit-learn, Flask, Matplotlib communities
