"""
Field validators for invoice extraction.
Validates extracted values and provides confidence adjustments.
"""

import re
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime


@dataclass
class ValidationResult:
    is_valid: bool
    confidence_adjustment: float
    corrected_value: Optional[str] = None
    error: Optional[str] = None


class FieldValidators:
    """validators for each invoice field type"""

    @staticmethod
    def validate_iban(value: str) -> ValidationResult:
        """validate german iban with checksum"""
        if not value:
            return ValidationResult(False, 0.0, error="empty")

        cleaned = value.replace(" ", "").upper()

        # basic format check
        if not re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$", cleaned):
            return ValidationResult(False, 0.3, error="invalid format")

        # german iban must be 22 chars
        if cleaned.startswith("DE") and len(cleaned) != 22:
            return ValidationResult(False, 0.4, error="wrong length")

        # format with spaces
        formatted = " ".join(cleaned[i:i+4] for i in range(0, len(cleaned), 4))

        # iban checksum validation
        if cleaned.startswith("DE"):
            try:
                rearranged = cleaned[4:] + cleaned[:4]
                numeric = ""
                for char in rearranged:
                    if char.isdigit():
                        numeric += char
                    else:
                        numeric += str(ord(char) - 55)
                if int(numeric) % 97 != 1:
                    # checksum failed but format is correct
                    return ValidationResult(True, 0.85, corrected_value=formatted)
            except (ValueError, OverflowError):
                return ValidationResult(True, 0.7, corrected_value=formatted)

        return ValidationResult(True, 1.0, corrected_value=formatted)

    @staticmethod
    def validate_date(value: str) -> ValidationResult:
        """validate german date format dd.mm.yyyy"""
        if not value:
            return ValidationResult(False, 0.0, error="empty")

        # normalize separators
        normalized = value.replace("/", ".").replace("-", ".")

        # try parsing
        formats = ["%d.%m.%Y", "%d.%m.%y"]
        for fmt in formats:
            try:
                parsed = datetime.strptime(normalized, fmt)
                # sanity check
                if 2000 <= parsed.year <= 2030:
                    formatted = parsed.strftime("%d.%m.%Y")
                    return ValidationResult(True, 1.0, corrected_value=formatted)
                else:
                    return ValidationResult(False, 0.6, error="year out of range")
            except ValueError:
                continue

        return ValidationResult(False, 0.4, error="invalid date format")

    @staticmethod
    def validate_amount(value: str) -> ValidationResult:
        """validate german amount format (comma as decimal)"""
        if not value:
            return ValidationResult(False, 0.0, error="empty")

        cleaned = value.strip().replace("€", "").replace("EUR", "").strip()

        # german format: 1.234,56 or 1234,56
        german_pattern = re.match(r"^(\d{1,3}(?:\.\d{3})*),(\d{2})$", cleaned)
        if german_pattern:
            return ValidationResult(True, 1.0, corrected_value=cleaned)

        # simple format without thousands: 1234,56
        simple_pattern = re.match(r"^(\d+),(\d{2})$", cleaned)
        if simple_pattern:
            return ValidationResult(True, 1.0, corrected_value=cleaned)

        # might be using dot as decimal
        english_pattern = re.match(r"^(\d{1,3}(?:,\d{3})*|\d+)\.(\d{2})$", cleaned)
        if english_pattern:
            main = english_pattern.group(1).replace(",", ".")
            decimal = english_pattern.group(2)
            corrected = f"{main},{decimal}"
            return ValidationResult(True, 0.9, corrected_value=corrected)

        # just digits
        if re.match(r"^\d+$", cleaned):
            return ValidationResult(True, 0.7, corrected_value=cleaned)

        return ValidationResult(False, 0.3, error="invalid amount format")

    @staticmethod
    def validate_phone(value: str) -> ValidationResult:
        """validate german phone number"""
        if not value:
            return ValidationResult(False, 0.0, error="empty")

        # extract digits
        digits = re.sub(r"\D", "", value)

        # german numbers
        if len(digits) < 6:
            return ValidationResult(False, 0.3, error="too short")

        if len(digits) > 15:
            return ValidationResult(False, 0.4, error="too long")

        # clean up formatting
        if digits.startswith("49"):
            formatted = f"+{digits[:2]} {digits[2:5]} {digits[5:]}"
        elif digits.startswith("0"):
            formatted = f"{digits[:4]} {digits[4:]}"
        else:
            formatted = value.strip()

        return ValidationResult(True, 1.0, corrected_value=formatted)

    @staticmethod
    def validate_invoice_number(value: str) -> ValidationResult:
        """validate invoice number format"""
        if not value:
            return ValidationResult(False, 0.0, error="empty")

        cleaned = value.strip()

        # too short
        if len(cleaned) < 3:
            return ValidationResult(False, 0.3, error="too short")

        # common invalid extractions
        invalid_words = ["datum", "summe", "bank", "telefon", "iban", "adresse"]
        if cleaned.lower() in invalid_words:
            return ValidationResult(False, 0.1, error="wrong field")

        # good patterns
        if re.match(r"^(RE|INV|RG|R|Nr)[-.:\s]?\d", cleaned, re.IGNORECASE):
            return ValidationResult(True, 1.0, corrected_value=cleaned)

        # alphanumeric with separators
        if re.match(r"^[A-Z0-9][A-Z0-9\-/]{2,}$", cleaned, re.IGNORECASE):
            return ValidationResult(True, 0.85, corrected_value=cleaned)

        # pure numbers at least 4 digits
        if re.match(r"^\d{4,}$", cleaned):
            return ValidationResult(True, 0.8, corrected_value=cleaned)

        return ValidationResult(False, 0.4, error="unusual format")

    @staticmethod
    def validate_company_name(value: str) -> ValidationResult:
        """validate company name"""
        if not value:
            return ValidationResult(False, 0.0, error="empty")

        cleaned = value.strip()

        if len(cleaned) < 3:
            return ValidationResult(False, 0.3, error="too short")

        # likely wrong field
        invalid_words = ["datum", "summe", "iban", "telefon", "bank:", "rechnung"]
        if cleaned.lower() in invalid_words:
            return ValidationResult(False, 0.1, error="wrong field")

        # has legal suffix - high confidence
        legal_suffixes = ["gmbh", "ag", "e.k.", "ohg", "kg", "gbr", "ug", "mbh", "inc", "ltd"]
        for suffix in legal_suffixes:
            if suffix in cleaned.lower():
                return ValidationResult(True, 1.0, corrected_value=cleaned)

        # reasonable length
        if 3 <= len(cleaned) <= 100:
            return ValidationResult(True, 0.7, corrected_value=cleaned)

        return ValidationResult(False, 0.5, error="unusual format")

    @staticmethod
    def validate_address(value: str) -> ValidationResult:
        """validate german address"""
        if not value:
            return ValidationResult(False, 0.0, error="empty")

        cleaned = value.strip()

        if len(cleaned) < 10:
            return ValidationResult(False, 0.4, error="too short")

        # has postal code
        has_postal = bool(re.search(r"\b\d{5}\b", cleaned))
        if has_postal:
            return ValidationResult(True, 1.0, corrected_value=cleaned)

        # has street number pattern
        has_street = bool(re.search(r"\d+[a-z]?\s", cleaned, re.IGNORECASE))
        if has_street:
            return ValidationResult(True, 0.8, corrected_value=cleaned)

        return ValidationResult(True, 0.6, corrected_value=cleaned)

    @staticmethod
    def validate_bank_name(value: str) -> ValidationResult:
        """validate bank name"""
        if not value:
            return ValidationResult(False, 0.0, error="empty")

        cleaned = value.strip()

        # remove "Bank:" prefix
        if cleaned.lower().startswith("bank:"):
            cleaned = cleaned[5:].strip()

        if len(cleaned) < 3:
            return ValidationResult(False, 0.3, error="too short")

        # known bank keywords
        bank_keywords = ["bank", "sparkasse", "volksbank", "commerzbank", "deutsche", "postbank", "ing", "dkb"]
        for kw in bank_keywords:
            if kw in cleaned.lower():
                return ValidationResult(True, 1.0, corrected_value=cleaned)

        return ValidationResult(True, 0.6, corrected_value=cleaned)


class InvoiceValidator:
    """validates all fields of an extraction result"""

    VALIDATORS = {
        "IBAN": FieldValidators.validate_iban,
        "Rechnungsdatum": FieldValidators.validate_date,
        "Falligkeitsdatum": FieldValidators.validate_date,
        "Rechnungsnummer": FieldValidators.validate_invoice_number,
        "Summe": FieldValidators.validate_amount,
        "Telefonnummer": FieldValidators.validate_phone,
        "Der Name der Firma": FieldValidators.validate_company_name,
        "Die Adresse der Firma": FieldValidators.validate_address,
        "Der Name der Bank": FieldValidators.validate_bank_name,
    }

    def validate_field(self, field_name: str, value: str) -> ValidationResult:
        """validate single field"""
        validator = self.VALIDATORS.get(field_name)
        if validator:
            return validator(value)
        return ValidationResult(True, 1.0, corrected_value=value)

    def validate_all(self, fields: dict) -> Tuple[dict, list]:
        """
        validate all fields
        """
        corrected = {}
        needs_llm = []

        for name, value in fields.items():
            result = self.validate_field(name, value)

            if result.is_valid:
                corrected[name] = result.corrected_value or value
            else:
                corrected[name] = value
                needs_llm.append(name)

        return corrected, needs_llm

    def get_field_quality(self, field_name: str, value: str, confidence: float) -> Tuple[float, bool]:
        """
        get adjusted confidence and whether field needs llm
        """
        result = self.validate_field(field_name, value)
        adjusted = confidence * result.confidence_adjustment

        # needs llm if low confidence or validation failed
        needs_llm = adjusted < 0.5 or not result.is_valid

        return adjusted, needs_llm