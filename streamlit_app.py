import streamlit as st
import os
import json
import time
from pathlib import Path
from main import PDFHeadingDetectionSystem

st.set_page_config(page_title="PDF Heading Detection", page_icon="📄", layout="wide")

st.title("📄 PDF Heading Detection System")
st.markdown("Upload a PDF document to automatically detect and extract its heading structure.")

# Initialize system once
@st.cache_resource
def get_system():
    return PDFHeadingDetectionSystem()

# Sidebar for system information
with st.sidebar:
    st.header("System Status")
    
    system = get_system()
    info = system.info()
    
    if info["models_trained"]:
        st.success("✅ Models trained successfully!")
    else:
        st.warning("⚠️ Models not found")
        if st.button("Train Models"):
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    system.train(samples_per_class=3000, optimise=False)  # Reduced for faster training
                    st.success("Models trained successfully!")
                    st.rerun()  # Fixed: changed from st.experimental_rerun()
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")

    st.subheader("Model Statistics")
    training_stats = info.get("training_stats", {})
    if training_stats:
        for model_type, stats in training_stats.items():
            if isinstance(stats, dict) and "accuracy" in stats:
                st.write(f"**{model_type.replace('_', ' ').title()}**: {stats['accuracy']:.3f}")

# Main content
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Create input directory if it doesn't exist
    input_dir = Path("input")
    input_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    input_path = input_dir / uploaded_file.name
    
    try:
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process the PDF
        with st.spinner("Processing PDF... Please wait."):
            start_time = time.time()
            result = system.process_pdf(str(input_path), None)
            processing_time = time.time() - start_time
        
        if "error" in result:
            st.error(f"❌ Error processing PDF: {result['error']}")
        else:
            # Display results in columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.header(" Document Structure")
                
                # Document title
                st.subheader(" Document Title")
                title = result.get("title", "Untitled Document")
                st.info(f"**{title}**")
                
                # Outline
                st.subheader(" Heading Outline")
                outline = result.get("outline", [])
                
                if outline:
                    for item in outline:
                        level = item["level"]
                        text = item["text"]
                        page = item["page"]
                        
                        # Create proper indentation based on heading level
                        if level == "H1":
                            indent = ""
                            emoji = ""
                        elif level == "H2":
                            indent = "　"
                            emoji = ""
                        elif level == "H3":
                            indent = "　　"
                            emoji = ""
                        else:
                            indent = "　　　"
                            emoji = "📎"
                        
                        st.markdown(f"{indent}{emoji} **{level}**: {text} *(Page {page})*")
                else:
                    st.warning(" No headings detected in this document.")
            
            with col2:
                st.header(" Statistics")
                
                # Metrics
                col2_1, col2_2 = st.columns(2)
                with col2_1:
                    st.metric("Total Headings", len(outline))
                with col2_2:
                    st.metric("Processing Time", f"{processing_time:.2f}s")
                
                # Heading level breakdown
                if outline:
                    st.subheader(" Heading Distribution")
                    level_counts = {}
                    for item in outline:
                        level = item["level"]
                        level_counts[level] = level_counts.get(level, 0) + 1
                    
                    # Create a simple bar chart
                    chart_data = []
                    for level in ["H1", "H2", "H3"]:  # Ensure consistent order
                        count = level_counts.get(level, 0)
                        chart_data.append({"Level": level, "Count": count})
                        st.write(f"**{level}**: {count}")
                
                # Download section
                st.subheader(" Export Results")
                
                # JSON download
                json_str = json.dumps(result, indent=2, ensure_ascii=False)
                st.download_button(
                    label=" Download JSON",
                    data=json_str,
                    file_name=f"{uploaded_file.name.replace('.pdf', '')}_headings.json",
                    mime="application/json"
                )
                
                # Text summary download
                if outline:
                    summary_text = f"Document Title: {title}\n\n"
                    summary_text += "Heading Structure:\n"
                    summary_text += "=" * 50 + "\n"
                    
                    for item in outline:
                        level = item["level"]
                        text = item["text"]
                        page = item["page"]
                        indent = "  " * (int(level[1]) - 1) if level.startswith("H") else ""
                        summary_text += f"{indent}{level}: {text} (Page {page})\n"
                    
                    st.download_button(
                        label=" Download Summary",
                        data=summary_text,
                        file_name=f"{uploaded_file.name.replace('.pdf', '')}_summary.txt",
                        mime="text/plain"
                    )
    
    except Exception as e:
        st.error(f"❌ Failed to process file: {str(e)}")
    
    finally:
        # Clean up uploaded file
        try:
            if input_path.exists():
                input_path.unlink()
        except:
            pass

# Info section
with st.expander("ℹ️ About This System"):
    st.markdown("""
    This PDF Heading Detection System uses machine learning to automatically identify and classify headings in PDF documents.
    
    **Features:**
    -  **98.9% accuracy** using two-stage Random Forest classifiers
    -  **43 engineered features** for comprehensive text analysis
    -  **Real-time processing** with automated model training
    -  **Structured output** in JSON format with confidence scores
    
    **How it works:**
    1. **Text Extraction**: Extracts text elements with formatting information
    2. **Feature Engineering**: Analyzes font size, position, styling, and content patterns  
    3. **Binary Classification**: Determines if text is a heading or regular content
    4. **Hierarchical Classification**: Assigns heading levels (H1, H2, H3, Title)
    5. **Post-processing**: Filters and validates results for accuracy
    """)

# Footer
st.markdown("---")

