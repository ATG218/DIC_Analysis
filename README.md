# DIC Analysis Pipeline (CHANGES TO README IN PROGRESS)

This repository contains all scripts needed to go from data generated by the thermal-cycling experimental process to a completed DIC analysis with the generation of strain curves and other visualizations.

## Project Structure

The project consists of two main components:
- `DIC_Pipeline`: Scripts for Digital Image Correlation (DIC) analysis and strain visualization
- `notch_parser`: Data sorting and validation

## Prerequisites

- Python 3.8 or higher
- DICe software installed (for DIC analysis)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Analysis_Pipeline
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix or MacOS
source .venv/bin/activate
```

3. Install dependencies:
```bash
# For DIC Pipeline
pip install -r requirements/dic_pipeline.txt

# For Notch Parser
pip install -r requirements/notch_parser.txt
```
