import streamlit as st
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

# Load trained model
model = tf.keras.models.load_model("49_categories.h5")

# Define class names
class_names = [
    'BATTERY-BATTERY-BATTERY',
 'BEARING-WHEEL BEARING-FRONT WHEEL HUB',
 'BEARING-WHEEL BEARING-REAR WHEEL HUB ASSEMBLY',
 'BODY PARTS-BODY TRIM-FOG LAMP COVER',
 'BODY PARTS-DICKEY-DICKY SHOCK ABSORBER',
 'BRAKE SYSTEM-BRAKE CALIPER-BRAKE CALIPER',
 'BRAKE SYSTEM-BRAKE DISC-BRAKE DISC',
 'BRAKE SYSTEM-BRAKE PAD-BRAKE PAD',
 'CHILD PARTS-CAP-FUEL TANK CAP',
 'ELECTRICALS AND ELECTRONICS-IGNITION COILIGNITOR-IGNITION COIL',
 'ELECTRICALS AND ELECTRONICS-RELAY AND FUSE-FUSE',
 'ELECTRICALS AND ELECTRONICS-ROTATING MACHINE-ALTERNATOR',
 'ELECTRICALS AND ELECTRONICS-ROTATING MACHINE-STARTER',
 'ELECTRICALS AND ELECTRONICS-SENSOR-SENSOR',
 'ENGINE-COOLING SYSTEM-THERMOSTAT ASSEMBLY',
 'ENGINE-COOLING SYSTEM-WATER PUMP',
 'ENGINE-CYLINDERS AND PISTONS-CAMSHAFT',
 'ENGINE-CYLINDERS AND PISTONS-CRANKSHAFT',
 'ENGINE-CYLINDERS AND PISTONS-PISTON',
 'ENGINE-ENGINE BLOCK-CYLINDER HEAD',
 'ENGINE-INLET AND EXHAUST SYSTEM-OXYGEN SENSOR',
 'ENGINE-INLET AND EXHAUST SYSTEM-SILENCER',
 'ENGINE-OIL SUMP-OIL SUMP',
 'ENGINE-SPARK GLOW PLUGS-SPARK PLUG',
 'FILTERS-AIR FILTER-AIR FILTER',
 'FILTERS-CABIN FILTER-CABIN FILTER',
 'FILTERS-FUEL FILTER-FUEL FILTER',
 'FILTERS-OIL FILTER-OIL FILTER',
 'FUEL SYSTEM-INJECTOR-INJECTOR',
 'HORNS-HORNS-HORN',
 'HVACTHERMAL-COMPRESSOR-AC COMPRESSOR',
 'HVACTHERMAL-RADIATOR-RADIATOR',
 'HVACTHERMAL-RADIATOR-RADIATOR FAN ASSEMBLY',
 'LIGHTING-BULBS-BULB',
 'LIGHTING-BULBS-HEAD LIGHT',
 'LIGHTING-FOG LAMP-FOG LAMP',
 'LIGHTING-TAIL LAMP-TAIL LAMP ASSEMBLY - RH',
 'RUBBERS HOSES AND MOUNTINGS-STRUT MOUNTING-STRUT MOUNTING',
 'RUBBERS HOSES AND MOUNTINGS-THERMAL HOSE-RADIATOR HOSE',
 'SUSPENSION-BALL JOINT-BALL JOINT',
 'SUSPENSION-CONTROL ARM-LOWER TRACK CONTROL ARM',
 'SUSPENSION-LINKS AND BUSHES-FRONT STABILIZER LINK',
 'SUSPENSION-LINKS AND BUSHES-LEAF SPRING',
 'SUSPENSION-SHOCK ABSORBER-COIL SPRING',
 'SUSPENSION-SHOCK ABSORBER-FRONT SHOCK ABSORBER',
 'SUSPENSION-SHOCK ABSORBER-REAR SHOCK ABSORBER',
 'TRANSMISSION-TORQUE CONVERTER-TORQUE CONVERTER',
 'WHEELS AND TYRES-RIMS AND ALLOYS-WHEEL RIM',
 'WIPER SYSTEM-WIPER BLADE-WIPER BLADE']

# Create folder to store images for analysis
os.makedirs("saved_images", exist_ok=True)

# Load previous results if available
csv_file = "results.csv"
if os.path.exists(csv_file):
    df_results = pd.read_csv(csv_file)
else:
    df_results = pd.DataFrame(columns=["Image Path", "Predicted Class", "Confidence"])

# Streamlit UI
st.title("🚗 Car Spare Parts Classification - Model Analysis")

# File uploader
uploaded_files = st.file_uploader("Upload Images", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)

if uploaded_files:
    new_results = []
    y_pred = []

    st.subheader("Uploaded Images & Predictions")
    cols = st.columns(5)  # Display images in 5 columns

    for idx, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file)
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Model Prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
        predicted_label = class_names[predicted_class]

        # Save image for analysis
        img_path = os.path.join("saved_images", uploaded_file.name)
        img.save(img_path)

        # Store result
        new_results.append([img_path, predicted_label, confidence])
        y_pred.append(predicted_label)

        # Display image in structured layout
        with cols[idx % 5]:
            st.image(img, caption=f"{predicted_label} ({confidence:.2f})", use_container_width=True)

    # Append new results to CSV
    df_new = pd.DataFrame(new_results, columns=["Image Path", "Predicted Class", "Confidence"])
    df_results = pd.concat([df_results, df_new], ignore_index=True)
    df_results.to_csv(csv_file, index=False)

    st.success("Results saved to `results.csv` 📁")

    # Display structured results
    st.subheader("📊 Prediction Results")
    st.dataframe(df_results)

    # Category-wise Prediction Count
    st.subheader("📌 Category-wise Prediction Count")
    category_counts = df_results["Predicted Class"].value_counts()
    st.bar_chart(category_counts)

    # Generate HTML Report
    report_html = "<h1>Prediction Report</h1><table border='1' cellpadding='10'>"
    report_html += "<tr><th>Image</th><th>Predicted Class</th><th>Confidence</th></tr>"

    for _, row in df_results.iterrows():
        img_tag = f'<img src="{row["Image Path"]}" width="100">'
        report_html += f"<tr><td>{img_tag}</td><td>{row['Predicted Class']}</td><td>{row['Confidence']:.2f}</td></tr>"

    report_html += "</table>"
    with open("prediction_report.html", "w") as f:
        f.write(report_html)

    with open("prediction_report.html", "rb") as f:
        st.download_button("📄 Download Report", f, file_name="prediction_report.html", mime="text/html")

      # Provide a download button for the CSV file
        st.subheader("📥 Download Predictions CSV")
        csv_data = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
          label="📄 Download CSV File",
          data=csv_data,
          file_name="classification_results.csv",
          mime="text/csv",
)

    # Confusion Matrix
    if "True Label" in df_results.columns:
        y_true = df_results["True Label"].tolist()
        y_pred = df_results["Predicted Class"].tolist()

        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(y_true, y_pred, labels=class_names)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(cm, annot=False, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)
