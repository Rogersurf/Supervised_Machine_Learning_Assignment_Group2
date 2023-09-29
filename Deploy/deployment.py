import pandas as pd
import gradio as gr
from Pipe.pipeline import load_and_split_data, train_model  # assuming Pipe is in the import path

# Load and split the data
X_train, X_test, y_train, y_test = load_and_split_data()

# Train the model and get the pipeline
model_name = 'random_forest'  # Change to whatever model you need
pipeline = train_model(model_name, X_train, y_train)

def predict(area, parking, bedrooms, stories, furnishingstatus):
    input_df = pd.DataFrame({
        'area': [area],
        'parking': [parking],
        'bedrooms': [bedrooms],
        'stories': [stories],
        'furnishingstatus': [furnishingstatus]
    })
    prediction = pipeline.predict(input_df)
    return prediction[0]

# Define Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.inputs.Slider(minimum=0, maximum=15000, step=1, default=1000, label='Area (0 - 15000 m²)'),
        gr.inputs.Slider(minimum=0, maximum=5, step=1, default=1, label='Parking (0 - 5)'),
        gr.inputs.Slider(minimum=0, maximum=6, step=1, default=1, label='Bedrooms (0 - 6)'),
        gr.inputs.Slider(minimum=0, maximum=20, step=1, default=1, label='Stories (0 - 20)'),
        gr.inputs.Dropdown(choices=['2 - Furnished', '1 - Semi-Furnished', '0 - Unfurnished'], label='Furnishing Status')
    ],
    outputs=gr.outputs.Textbox(label="Predicted Price")
)

# Launch Gradio Interface
iface.launch()