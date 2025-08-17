"""
Main entry point for the PDF heading-detection project (local-friendly).
"""


import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional
import glob
import numpy as np


from config import Config
from pdf_processor import create_pdf_processor
from feature_extractor import create_feature_extractor
from classifiers import create_classification_system
from output_formatter import create_output_formatter



def _setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> None:
    """Configure root logger for console (and optional file) output."""
    level = logging.DEBUG if verbose else logging.INFO

    formatter = logging.Formatter(
        "%(asctime)s │ %(name)s │ %(levelname)s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)

    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        root.addHandler(fh)

    # Quiet noisy third-party libraries
    logging.getLogger("pdfminer").setLevel(logging.WARNING)
    logging.getLogger("sklearn").setLevel(logging.WARNING)


class PDFHeadingDetectionSystem:
    """High-level wrapper that wires together all project components."""

    def __init__(self) -> None:
        self._pdf_proc = create_pdf_processor()
        self._feat_ext = create_feature_extractor()
        self._clf_sys = create_classification_system()
        self._out_fmt = create_output_formatter()

        logging.getLogger(__name__).info("PDF Heading Detection System initialised")

    def train(self, samples_per_class: int, optimise: bool = True, force: bool = False) -> None:
        """Train (or retrain) the ML models from synthetic data."""
        if not force and self._clf_sys.load_models():
            logging.info("Models already present – use --force-retrain to override")
            return

        self._clf_sys.train_with_synthetic_data(
            samples_per_class=samples_per_class,
            optimize_hyperparameters=optimise,
        )

    def process_pdf(self, pdf_path: str, output_path: Optional[str], auto_train: bool = True) -> dict:
        """Process a single PDF and return structured heading information."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return {"error": f"File not found: {pdf_path}"}

        if auto_train and not self._clf_sys.is_trained:
            if not self._clf_sys.load_models():
                logging.info("No models on disk – triggering quick synthetic training")
                self.train(samples_per_class=Config.DEFAULT_SAMPLES_PER_CLASS // 2, optimise=False)

        t0 = time.time()

        # 1 – extract text / formatting
        text_elems = self._pdf_proc.extract_text_elements(str(pdf_path))
        if not text_elems:
            return {"error": "No text elements extracted – is the PDF scanned?"}

        proc_stats = self._pdf_proc.get_processing_stats(text_elems)

        # 2 – feature engineering
        feats, elems = self._feat_ext.extract_features(text_elems)
        if feats.size == 0:
            return {"error": "Feature extraction failed"}

        # 3 – classification
        preds = self._clf_sys.predict(feats, elems)

        # 4 – format output (detailed version)
        detailed_result = self._out_fmt.format_results(preds, elems, str(pdf_path), proc_stats)
        detailed_result["processing_time"] = round(time.time() - t0, 2)

        # 5 – transform to required simple format
        simple_result = self.transform_to_required_format(detailed_result)

        # optional save
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(simple_result, f, indent=4, ensure_ascii=False)
            logging.info(f"Results saved to: {output_path}")

        return simple_result

    def transform_to_required_format(self, detailed_output: dict) -> dict:
        """Transform detailed output to required simple format"""

        # Extract title from title-type elements
        document_structure = detailed_output.get("document_structure", [])

        # Find title elements
        title_elements = [item for item in document_structure if item.get("type") == "title"]

        # Combine titles or use the first meaningful one
        if title_elements:
            if len(title_elements) == 1:
                title = title_elements[0]["text"].strip()
            else:
                title = " ".join([t["text"].strip() for t in title_elements])
        else:
            # Fallback: use first H1 as title or default
            h1_elements = [item for item in document_structure if item.get("type") == "h1"]
            title = h1_elements[0]["text"].strip() if h1_elements else "Document Title"  # ✅ FIXED LINE

        # Extract outline - only H1, H2, H3
        outline = []
        for item in document_structure:
            item_type = item.get("type", "").lower()
            if item_type in ["h1", "h2", "h3"]:
                outline.append({
                    "level": item_type.upper(),
                    "text": item["text"].strip(),
                    "page": item.get("page", 1)
                })

        return {"title": title, "outline": outline}

    def batch_process(self, in_dir: str, out_dir: str, pattern: str) -> dict:
        """Process every PDF in in_dir matching pattern."""
        in_dir = Path(in_dir)
        if not in_dir.exists():
            return {"error": f"Directory not found: {in_dir}"}

        pdf_files = list(in_dir.glob(pattern))
        if not pdf_files:
            return {"error": "No PDFs found for batch processing"}

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = {"total": len(pdf_files), "processed": [], "failed": []}

        for idx, pdf in enumerate(pdf_files, 1):
            logging.info(f"[{idx}/{len(pdf_files)}] {pdf.name}")

            out_json = out_dir / f"{pdf.stem}.json"
            res = self.process_pdf(str(pdf), str(out_json), auto_train=True)

            if "error" in res:
                summary["failed"].append({"file": str(pdf), "error": res["error"]})
                logging.error(f"✗ {pdf.name} – {res['error']}")
            else:
                summary["processed"].append({
                    "file": str(pdf),
                    "headings": len(res.get("outline", [])),
                    "time": 0.0,
                })
                logging.info(f"✓ {pdf.name}")

        summary["metrics"] = {
            "success_rate": len(summary["processed"]) / summary["total"] if summary["total"] else 0.0,
            "avg_time": 0.0,
            "total_headings": sum(f["headings"] for f in summary["processed"]),
        }
        return summary

    def info(self) -> dict:
        """Return high-level information about the current system state."""
        return {
            "models_trained": self._clf_sys.is_trained,
            "model_dir": str(Config.MODELS_DIR),
            "training_stats": self._clf_sys.get_training_stats(),
            "feature_count": len(self._feat_ext.get_feature_names()),
            "heading_types": Config.HEADING_TYPES,
        }


def main_cli() -> None:
    """Local-friendly CLI (single file or batch)."""
    parser = argparse.ArgumentParser(
        prog="PDF Heading Detector",
        description="Detect and classify headings in PDF documents",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # primary actions
    parser.add_argument("pdf", nargs="?", help="PDF file to process (single-file mode)")
    parser.add_argument("-o", "--output", help="JSON output path for single-file mode")
    parser.add_argument("--display", action="store_true", help="pretty-print single-file results")

    # training
    parser.add_argument("--train-synthetic", action="store_true", help="train from synthetic data")
    parser.add_argument("--samples", type=int, default=Config.DEFAULT_SAMPLES_PER_CLASS, help="samples per class")
    parser.add_argument("--optimize-params", action="store_true", help="hyper-parameter optimisation")
    parser.add_argument("--force-retrain", action="store_true", help="retrain even if models exist")

    # batch
    parser.add_argument("--batch-dir", default="./input", help="directory of PDFs to process")
    parser.add_argument("--batch-out", default="./output", help="output directory for batch JSON files")
    parser.add_argument("--pattern", default="*.pdf", help="glob pattern for PDFs")

    # misc
    parser.add_argument("--system-info", action="store_true", help="print system info and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose logging")
    parser.add_argument("--log-file", help="write logs to file")

    args = parser.parse_args()
    _setup_logging(args.verbose, args.log_file)
    system = PDFHeadingDetectionSystem()

    # system info
    if args.system_info:
        print(json.dumps(system.info(), indent=2))
        return

    # training
    if args.train_synthetic:
        logging.info("Synthetic-data training requested")
        system.train(
            samples_per_class=args.samples,
            optimise=args.optimize_params,
            force=args.force_retrain,
        )
        return

    # batch mode (default if no single file given)
    if not args.pdf:
        res = system.batch_process(args.batch_dir, args.batch_out, args.pattern)
        print(json.dumps(res, indent=2))
        return

    # single-file mode
    result = system.process_pdf(args.pdf, args.output, auto_train=True)

    if "error" in result:
        logging.error(result["error"])
        sys.exit(1)

    if args.display:
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Outline ({len(result.get('outline', []))} headings):")
        for i, heading in enumerate(result.get('outline', []), 1):
            print(f"  {i}. [{heading['level']}] {heading['text']} (Page {heading['page']})")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main_cli()
