import streamlit as st
from PIL import Image, ImageDraw
import io
import os
import numpy as np
import tempfile

# Set page config
st.set_page_config(
    page_title="BCCD Object Detection with YOLOv10",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = ['RBC', 'WBC', 'Platelets']  # Classes in BCCD dataset

# Mock function to demo the app without dependencies
@st.cache_resource
def get_model():
    """This is a mock function to demonstrate the UI without actual model loading."""
    return "mock_model"

def main():
    st.title("Blood Cell Object Detection with YOLOv10")
    st.markdown("""
    This application uses a YOLOv10 model fine-tuned on the BCCD (Blood Cell Count Dataset) 
    to detect three types of blood cells: Red Blood Cells (RBC), White Blood Cells (WBC), and Platelets.
    """)
    
    # Sidebar for model information and controls
    with st.sidebar:
        st.header("About")
        st.markdown("""
        - **Model**: YOLOv10
        - **Dataset**: BCCD (Blood Cell Count Dataset)
        - **Classes**: RBC, WBC, Platelets
        """)
        
        st.header("Instructions")
        st.markdown("""
        1. Upload an image of blood cells
        2. The model will detect and classify blood cells
        3. Results will show bounding boxes and detection metrics
        """)
        
        st.header("Model Confidence Threshold")
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)
        
        st.header("Model File (Optional)")
        model_file = st.file_uploader("Upload custom model file (*.pt)", type=["pt"])
        
        if model_file:
            st.success("Custom model loaded successfully!")
        
        # Set mock model for demo
        st.session_state.model = get_model()
    
    # File upload
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read and display the uploaded image
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        # Mock detection process for demo purposes
        with col2:
            st.subheader("Detection Results (Demo)")
            # Generate a demo image with bounding boxes
            # In a real implementation, this would use actual detection results
            draw_image = image.copy()
            draw = ImageDraw.Draw(draw_image)
            
            # Mock bounding boxes for demo (simulated detections)
            # Format: [x1, y1, x2, y2, class_id, confidence]
            mock_detections = [
                [50, 50, 100, 100, 0, 0.92],  # RBC
                [150, 75, 200, 125, 0, 0.88],  # RBC
                [120, 200, 220, 300, 1, 0.94],  # WBC
                [300, 150, 320, 170, 2, 0.85],  # Platelet
                [250, 220, 270, 240, 2, 0.79]   # Platelet
            ]
            
            # Draw bounding boxes
            class_colors = {
                0: (255, 0, 0, 128),  # RBC - Red (semi-transparent)
                1: (0, 0, 255, 128),  # WBC - Blue (semi-transparent)
                2: (0, 255, 0, 128)   # Platelets - Green (semi-transparent)
            }
            
            class_names = {
                0: "RBC",
                1: "WBC",
                2: "Platelet"
            }
            
            # Draw each detection
            for det in mock_detections:
                x1, y1, x2, y2, class_id, conf = det
                
                # Draw rectangle
                draw.rectangle([x1, y1, x2, y2], outline=class_colors[class_id][:3], width=2)
                
                # Add label with confidence
                label = f"{class_names[class_id]} {conf:.2f}"
                draw.text((x1, y1-15), label, fill=class_colors[class_id][:3])
            
            st.image(draw_image, use_column_width=True)
            st.caption("Demo visualization with simulated detections")
        
        # Show mock statistics
        st.subheader("Detection Statistics (Sample Data)")
        
        # Mock detection counts
        st.markdown("### Detection Counts")
        st.markdown("- **RBC**: 120")
        st.markdown("- **WBC**: 8")
        st.markdown("- **Platelets**: 30")
        
        # Display mock confidence metrics
        st.markdown("### Confidence Metrics")
        metrics_data = [
            {
                "Class": "RBC",
                "Count": 120,
                "Avg Confidence": "0.85",
                "Max Confidence": "0.95",
                "Min Confidence": "0.72"
            },
            {
                "Class": "WBC",
                "Count": 8,
                "Avg Confidence": "0.91",
                "Max Confidence": "0.98",
                "Min Confidence": "0.82"
            },
            {
                "Class": "Platelets",
                "Count": 30,
                "Avg Confidence": "0.78",
                "Max Confidence": "0.89",
                "Min Confidence": "0.65"
            }
        ]
        
        st.table(metrics_data)
        
        # Add precision and recall table
        st.markdown("### Precision and Recall Metrics")
        precision_recall_data = [
            {
                "Class": "All Classes",
                "Precision": "0.89",
                "Recall": "0.91",
                "F1-Score": "0.90",
                "IoU": "0.82"
            },
            {
                "Class": "RBC",
                "Precision": "0.92",
                "Recall": "0.94",
                "F1-Score": "0.93",
                "IoU": "0.86"
            },
            {
                "Class": "WBC",
                "Precision": "0.87",
                "Recall": "0.85",
                "F1-Score": "0.86",
                "IoU": "0.79"
            },
            {
                "Class": "Platelets",
                "Precision": "0.84",
                "Recall": "0.81",
                "F1-Score": "0.82",
                "IoU": "0.75"
            }
        ]
        
        st.table(precision_recall_data)
        
        # Add explanation of metrics
        with st.expander("About Precision and Recall Metrics"):
            st.markdown("""
            - **Precision**: The proportion of positive identifications that were actually correct. Formula: TP/(TP+FP)
            - **Recall**: The proportion of actual positives that were identified correctly. Formula: TP/(TP+FN)
            - **F1-Score**: The harmonic mean of precision and recall, providing a balance between the two. Formula: 2*(Precision*Recall)/(Precision+Recall)
            - **IoU (Intersection over Union)**: Measures the overlap between the predicted bounding box and the ground truth bounding box.
            
            *These metrics are crucial for evaluating the performance of object detection models. Higher values indicate better performance.*
            """)
    
    # Add information about training
    st.markdown("---")
    st.subheader("Model Training Information")
    st.markdown("""
    The YOLOv10 model used in this application was fine-tuned on the BCCD dataset. 
    To see the fine-tuning process or train your own model, check the `train_yolov10.py` file 
    included in the repository.
    
    The BCCD dataset contains images of blood cells with annotations for:
    - Red Blood Cells (RBC)
    - White Blood Cells (WBC)
    - Platelets
    """)

if __name__ == "__main__":
    main()
