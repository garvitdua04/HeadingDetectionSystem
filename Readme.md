# PDF Heading Detection System

A comprehensive machine learning system for automatically detecting and classifying headings in PDF documents.

## Features

- **Multi-stage ML Classification**: Binary classifier + hierarchical classifier
- **Synthetic Data Training**: Generates training data automatically
- **Web Interface**: Streamlit-based user-friendly interface
- **Command Line Interface**: Batch processing capabilities
- **Multiple Output Formats**: JSON results with confidence scores

## Installation


1. **Clone or download this project**
2. **Create virtual environment** (recommended):


    python -m venv .venv



    source .venv/bin/activate 
  


    On Windows: .venv\Scripts\activate

 



3. **Install dependencies**:


    pip install -r requirements.txt


## Quick Start

### Step 1: Train the Models (First Time Only)

Train with synthetic data

    python main.py --train-synthetic --samples 15000 --optimize-params

Or quick training (faster, slightly lower accuracy)

    python main.py --train-synthetic --samples 5000


### Step 2: Process PDFs

#### Single PDF:


    python main.py input/document.pdf -o output/result.json


#### Batch Processing:

    python main.py --batch-dir ./input --batch-out ./output --pattern "*.pdf"


#### Web Interface:


    streamlit run streamlit_app.py


## Usage Examples



### Command Line Examples



Check system status


    python main.py --system-info

Process single PDF with verbose output

    python main.py input/report.pdf -o output/report_headings.json -v

Batch process all PDFs in a directory

    python main.py --batch-dir ./input --batch-out ./output

Force retrain models

    python main.py --train-synthetic --force-retrain --samples 10000



### Web Interface

1. Start the Streamlit app:

        streamlit run streamlit_app.py
        
2. Open your browser to `http://localhost:8501`
3. Upload a PDF file
4. View extracted headings and document structure
5. Download results as JSON

## Project Structure

├── input/                     # Place your PDF files here

├── output/                    # Generated JSON results  

├── models/                    # Trained ML models (auto-generated)

├── training_data/             # Synthetic training data (auto-generated)

├── main.py                   # Main 

├── streamlit_app.py          # Web interface  

├── classifiers.py            # ML classification system

├── feature_extractor.py     # Feature engineering

├── config.py                # Configuration settings

├── pdf_processor.py         # PDF text extraction

├── output_formatter.py      # Result formatting

├── synthetic_data_generator.py # Training data generation

├── utils.py                 # Utility functions

├── Readme.md                # Project documentation

├── requirements.txt         # Python dependencies



## Output Format

{
"title": "Document Title",

"outline": [

{

"level": "H1",

"text": "Introduction",

"page": 1

},

{

"level": "H2",

"text": "Background",

"page": 2

}

]

}


## Troubleshooting

- **Models not found**: Run training first with `--train-synthetic`
- **No text extracted**: PDF might be scanned/image-based
- **Low accuracy**: Try retraining with more samples
- **Import errors**: Check if all requirements are installed

