"""
LLM-based field extraction for ambiguous cases.
"""

import os
import json
import re
from typing import Dict, Optional, List
from dataclasses import dataclass
from openai import OpenAI
import requests
from .validators import InvoiceValidator


@dataclass
class LLMFieldResult:
    """result for single field from llm"""
    value: Optional[str]
    confidence: float
    validated: bool = False
    source: str = "llm"


@dataclass
class LLMResult:
    """result from llm extraction"""
    fields: Dict[str, LLMFieldResult]
    overall_confidence: float
    raw_response: str
    success: bool = True
    error: Optional[str] = None


class LLMExtractor:
    """
    extracts invoice fields using llm when pattern matching is uncertain
    """

    FIELD_MAP = {
        "Der Name der Bank": "bank_name",
        "Der Name der Firma": "company_name",
        "Die Adresse der Firma": "company_address",
        "Falligkeitsdatum": "due_date",
        "IBAN": "iban",
        "Rechnungsdatum": "invoice_date",
        "Rechnungsnummer": "invoice_number",
        "Summe": "total_amount",
        "Telefonnummer": "phone_number",
    }

    REVERSE_MAP = {v: k for k, v in FIELD_MAP.items()}

    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", base_url: str = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model
        self._base_url = base_url
        self._client = None
        self._validator = InvoiceValidator()

    @property
    def client(self):
        if self._client is None:
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url
            )
        return self._client

    def extract(self, ocr_text: str, target_fields: List[str] = None,
                existing_values: Dict[str, str] = None) -> LLMResult:
        """
        extract fields from ocr text using llm
        """
        if not self._api_key:
            return LLMResult(
                fields={}, overall_confidence=0.0,
                raw_response="", success=False, error="no api key"
            )

        prompt = self._build_prompt(ocr_text, target_fields, existing_values)

        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=800,
            )

            raw = response.choices[0].message.content
            parsed = self._parse_response(raw)

            if not parsed:
                return LLMResult(
                    fields={}, overall_confidence=0.0,
                    raw_response=raw, success=False, error="parse failed"
                )

            # field results with validation
            fields = {}
            valid_count = 0

            for eng_key, value in parsed.items():
                german_key = self.REVERSE_MAP.get(eng_key)
                if not german_key:
                    continue

                # skip if targeting specific fields
                if target_fields and german_key not in target_fields:
                    continue

                # skip empty/null
                if not value or value.lower() in ("null", "none", "n/a", ""):
                    continue

                # validate the extracted value
                validation = self._validator.validate_field(german_key, value)

                if validation.is_valid:
                    valid_count += 1
                    final_value = validation.corrected_value or value
                    conf = 0.85 * validation.confidence_adjustment
                else:
                    final_value = value
                    conf = 0.4 * validation.confidence_adjustment

                fields[german_key] = LLMFieldResult(
                    value=final_value,
                    confidence=conf,
                    validated=validation.is_valid
                )

            # overall confidence based on extraction success
            if fields:
                overall = sum(f.confidence for f in fields.values()) / len(fields)
            else:
                overall = 0.0

            return LLMResult(
                fields=fields,
                overall_confidence=overall,
                raw_response=raw,
                success=True
            )

        except Exception as e:
            return LLMResult(
                fields={}, overall_confidence=0.0,
                raw_response=str(e), success=False, error=str(e)
            )

    def _system_prompt(self) -> str:
        return """You extract data from German invoices.
Return only valid JSON with the exact field names provided.
Use null for fields you cannot find.
For amounts, use German format: 1.234,56
For dates, use format: dd.mm.yyyy
For IBAN, include spaces every 4 characters."""

    def _build_prompt(self, ocr_text: str, target_fields: List[str] = None,
                      existing: Dict[str, str] = None) -> str:
        # field guidance
        if target_fields:
            eng_fields = [self.FIELD_MAP.get(f, f) for f in target_fields]
            focus = f"Focus on extracting these fields: {', '.join(eng_fields)}"
        else:
            focus = "Extract all available invoice fields"

        # context from existing extraction
        context = ""
        if existing:
            context = "\nContext (OCR already found these, please verify or improve):\n"
            for name, value in existing.items():
                eng_name = self.FIELD_MAP.get(name, name)
                if value:
                    context += f"  {eng_name}: {value}\n"

        return f"""Extract invoice information from this OCR text.
{focus}
{context}

OCR Text:
{ocr_text[:3000]}

Return JSON (use null if not found):
{{
  "company_name": "company that issued the invoice",
  "company_address": "full address with postal code",
  "bank_name": "bank name only, no IBAN",
  "iban": "with spaces: DE12 3456 7890 1234 5678 90",
  "invoice_number": "invoice/receipt number",
  "invoice_date": "dd.mm.yyyy",
  "due_date": "dd.mm.yyyy payment deadline",
  "total_amount": "total in German format: 1.234,56",
  "phone_number": "phone with area code"
}}"""

    def _parse_response(self, response: str) -> Optional[dict]:
        """parse llm response to dict"""
        # clean up response
        cleaned = response.strip()

        # remove markdown code blocks
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        # direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # json object in text
        match = re.search(r'\{[^{}]*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # extract key-value pairs
        result = {}
        for line in response.split("\n"):
            kv_match = re.match(r'"?(\w+)"?\s*:\s*"([^"]*)"', line)
            if kv_match:
                result[kv_match.group(1)] = kv_match.group(2)

        return result if result else None

    def is_available(self) -> bool:
        return bool(self._api_key)


class LocalLLMExtractor(LLMExtractor):
    """uses local llm server (ollama, llama.cpp, etc)"""

    def __init__(self, base_url: str = "http://localhost:11434/v1", model: str = "llama3.2"):
        super().__init__(api_key="local", model=model, base_url=base_url)

    def is_available(self) -> bool:
        try:
            resp = requests.get(self._base_url.replace("/v1", "/api/tags"), timeout=2)
            return resp.status_code == 200
        except Exception:
            return False