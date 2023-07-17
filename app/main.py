import streamlit as st
import pickle5 as pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np


def get_clean_data():
    data = pd.read_csv("Data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    
    data = get_clean_data()
    # Define the labels and features
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]
    
    input_dict = {}
    
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label, 
            min_value=float(data[key].min()),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
            )
    return input_dict

def get_scaled_values_dict(input_dict):
    
    data = get_clean_data()
    
    X = data.drop(['diagnosis'], axis=1)
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        min_val = X[key].min()
        max_val = X[key].max()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
        
    return scaled_dict

def get_radar_chart(input_data):
    
    input_data = get_scaled_values_dict(input_data)
    
    # Create the categories list
    categories = ['Radius', 'Texture', 'Perimeter',
                  'Area', 'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension'
                  ]
    
    # Create the radar chart
    fig = go.Figure()

    # Add the traces
    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
                input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
                input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
                input_data['fractal_dimension_mean']],
            theta=categories,
            fill='toself',
            name='Mean'
        )
    )

    fig.add_trace(
        go.Scatterpolar(
            r=[input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
                input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
                input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']],
            theta=categories,
            fill='toself',
            name='Standard Error'
        )
    )

    fig.add_trace(go.Scatterpolar(
            r=[input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
                input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
                input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
                input_data['fractal_dimension_worst']],
            theta=categories,
            fill='toself',
            name='Worst'
        )
    )

    # Update the layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        autosize=True
    )

    return fig

def add_predicitons(input_data):
    model = pickle.load(open('Model/model.pkl', 'rb'))
    scalar = pickle.load(open('Model/scalar.pkl', 'rb'))
    
    # Create the array of values using numpy
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    
    input_array_scaled = scalar.transform(input_array)
    
    prediction = model.predict(input_array_scaled)
    
    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster is: ")
    
    if prediction == 1:
        st.write("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)
    if prediction == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
   
    st.write("The probability of being benign is: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("The probability of being malignant is: ", model.predict_proba(input_array_scaled)[0][1])
    
    st.write("This app can assist doctors in diagnosing breast cancer from tissue samples. It is not a replacement for a doctor's opinion.")
    
    

# Creating the streamlit application frontend
def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction", 
        page_icon=":female-doctor:",
        layout = "wide",
        initial_sidebar_state="expanded"
        )
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    # Create the sidebar and load it with the data columns from above
    input_data = add_sidebar()
    
    
    
    with st.container():
        st.title("Breast Cancer Predictor")
        st.write("Please connect this app to your cryptology lab to help diagnose breast cancer from your tissue sample. This app is powered by machine learning and can predict whether a tumor is malignant or benign from the measurements it recieves. You can also update by hand using the sliders in the sidebar.")


    col1, col2 = st.columns([4,1]) 
     
    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)
    with col2:
        add_predicitons(input_data)
    
      
if __name__ == '__main__':
    main()    