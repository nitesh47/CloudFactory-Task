# German Invoice Field Extraction

An application that pulls image of German Invoices data and provide: company info, amounts, dates, bank details and tells you how confident it is about each extraction. Low confidence? It flags it for human review.

## What It Does

- Extracts 9 fields from invoice images (company name, address, IBAN, total amount, dates, etc.)
- Gives each field a confidence score (0-1)
- Decides if the extraction is good enough to auto-accept or needs someone to check it
- Outputs clean JSON with everything you need

## Getting Started

### Using Docker (easier)

```bash
# build the image
docker build -t invoice-extractor .

# extract from an invoice
docker run -v $(pwd)/sample_invoices:/data invoice-extractor \
    python extract.py /data/invoice_0001.png --pretty
```

### Local Installation

```bash
# install dependencies
poetry install

# quick test
python scripts/test_system.py

# extract from an invoice
python extract.py sample_invoices/invoice_0001.png --pretty
```

First run downloads OCR models (~150MB) from HuggingFace. After that they're cached locally.

## Usage

### Extract a Single Invoice

```bash
python extract.py invoice.png --pretty
```

Options:
- `--pretty, -p` - format the JSON nicely
- `--output, -o FILE` - save to a file
- `--include-ocr` - include raw OCR text

### Run Evaluation on Dataset

```bash
# run on 20 samples
python evaluate_all.py --limit 20

# run on everything
python evaluate_all.py

# run on test data

python evaluate_all.py --data data/test-00000-of-00001-a9d41ee534bb86d0.parquet
```

This outputs precision, recall, F1 for each field, plus saves detailed results to `results/`.

## Output Example

```json
{
  "document_id": "invoice",
  "status": "auto_accept",
  "overall_confidence": 0.85,
  "needs_human_review": false,
  "extracted_fields": {
    "IBAN": {
      "value": "DE89 3704 0044 0532 0130 00",
      "confidence": 0.95,
      "needs_review": false
    },
    "Summe": {
      "value": "1.234,56",
      "confidence": 0.85,
      "needs_review": false
    },
    "Rechnungsnummer": {
      "value": "RE-2024-001",
      "confidence": 0.88,
      "needs_review": false
    }
  },
  "line_items": [
    {
      "description": "Product A",
      "quantity": "2",
      "total": "20,00",
      "confidence": 0.6,
      "needs_review": true
    }
  ]
}
```

## Fields Extracted

| Field | What it is | Example |
|-------|------------|---------|
| Der Name der Firma | Company name | Acme GmbH |
| Die Adresse der Firma | Company address | Musterstr. 1, 12345 Berlin |
| Telefonnummer | Phone number | +49 30 12345678 |
| Rechnungsnummer | Invoice number | RE-2024-001 |
| Rechnungsdatum | Invoice date | 01.02.2024 |
| Falligkeitsdatum | Due date | 15.02.2024 |
| Summe | Total amount | 1.234,56 |
| IBAN | Bank account | DE89 3704 0044 0532 0130 00 |
| Der Name der Bank | Bank name | Sparkasse Berlin |


## Project Structure

```
├── extract.py              # extract from a single invoice
├── evaluate_all.py         # run evaluation on dataset
├── scripts/
│   ├── run_demo.py         # demo with metrics
│   └── test_system.py      # quick test
├── src/invoice_extractor/
│   ├── core/
│   │   ├── pipeline.py     # main extraction logic
│   │   ├── extractor.py    # pattern matching
│   │   ├── doctr_ocr.py    # OCR engine
│   │   └── validators.py   # format validation
│   ├── models/
│   │   └── result.py       # data structures
│   └── evaluation/
│       └── metrics.py      # P/R/F1 calculation
└── data/                   # invoice dataset
```
