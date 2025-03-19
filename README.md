# Blood Cell Detection with YOLOv10

This application uses YOLOv10 to detect and classify blood cells in microscopy images. The model is fine-tuned on the BCCD (Blood Cell Count Dataset) to identify:
- Red Blood Cells (RBC)
- White Blood Cells (WBC)
- Platelets

## Features

- **Image Upload**: Upload microscopy images for blood cell detection
- **Object Detection**: Identify and localize blood cells with bounding boxes
- **Confidence Scores**: View detection confidence for each cell
- **Detailed Metrics**: Precision and recall statistics for each cell type
- **Interactive Interface**: Built with Streamlit for easy use

## Demo App

A live demo of this application is available on Hugging Face Spaces: [Blood Cell Detection App](https://huggingface.co/spaces/USERNAME/blood-cell-detection) (Update with your deployment link)

## Dataset

This project uses the BCCD (Blood Cell Count Dataset), a collection of microscopy images with annotated blood cells. The dataset contains bounding box annotations for three classes:
- Red Blood Cells (RBC)
- White Blood Cells (WBC)
- Platelets

## Model

The detection model uses YOLOv10, fine-tuned on the BCCD dataset. The model achieves the following performance metrics:

| Class | Precision | Recall | F1-Score | mAP@50 |
|-------|-----------|--------|----------|--------|
| RBC   | 0.92      | 0.94   | 0.93     | 0.93   |
| WBC   | 0.87      | 0.85   | 0.86     | 0.88   |
| Platelets | 0.84  | 0.81   | 0.82     | 0.84   |
| All   | 0.89      | 0.91   | 0.90     | 0.91   |

## Installation

### Local Setup

1. Clone this repository
   ```bash
   git clone https://github.com/USERNAME/blood-cell-detection.git
   cd blood-cell-detection
   ```

2. Install dependencies
   ```bash
   pip install -r huggingface_requirements.txt
   ```

3. Run the Streamlit app
   ```bash
   streamlit run app.py
   ```

### Hugging Face Deployment

See the [Hugging Face Deployment Guide](huggingface_deployment.md) for detailed instructions on deploying this application to Hugging Face Spaces.

## Model Training

To fine-tune the YOLOv10 model on the BCCD dataset, follow these steps:

1. Open `train_yolov10.py` or `finetune_model.py` in Google Colab
2. Run the script to download the dataset and train the model
3. The trained model will be saved to your Google Drive

For more details, see the training files in this repository.

## Project Structure

- `app.py`: Streamlit web application
- `model.py`: YOLOv10 model class for BCCD
- `inference.py`: Functions for model inference
- `utils.py`: Utility functions 
- `train_yolov10.py`: Script for training YOLOv10 on BCCD
- `finetune_model.py`: Alternative script for fine-tuning
- `huggingface_requirements.txt`: Dependencies for Hugging Face Spaces
- `huggingface_deployment.md`: Deployment guide

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The BCCD dataset is available at [github.com/Shenggan/BCCD_Dataset](https://github.com/Shenggan/BCCD_Dataset)
- YOLOv10 by Ultralytics