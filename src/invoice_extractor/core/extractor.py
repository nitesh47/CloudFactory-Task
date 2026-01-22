"""
German Invoice Field Extractor
Document Structure (DOM)
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from ..models.result import FieldResult, LineItem


@dataclass
class PatternMatch:
    value: str
    confidence: float
    method: str


class InvoiceExtractor:

    RECIPIENT_MARKERS = ["herr ", "frau ", "empfanger", "empfänger", "kundennr", "kundennummer"]
    # Patterns that indicate non-company text
    SKIP_EXACT = ["hier", "logo"]
    SKIP_CONTAINS = ["einfugen", "einfügen", "platzhalter", "[", "]",
                     "datum", "lieferdatum", "ansprechperson", "e-mail", "email",
                     "rechnungsnummer", "rechnungsnr", "rechnung", "angebot", "kundennummer",
                     "telefon:", "telefon-", "tel:", "fax:", "hotline", "www.", "http"]
    BANK_KEYWORDS = ["bank", "sparkasse", "volksbank", "kreditinstitut"]
    BANK_SKIP = ["bankverbindung", "[bankname]", "bankname"]  # labels to skip

    def __init__(self):
        self._phone_pattern = re.compile(r"(?:tel\.?|telefon|fon)[:\s]*([+\d\s\-/]{8,20})", re.I)
        self._date_pattern = re.compile(r"(\d{1,2}[./]\d{1,2}[./]\d{2,4})")
        self._amount_pattern = re.compile(r"(\d{1,3}(?:[.,]\d{3})*[.,]\d{2})")
        self._iban_pattern = re.compile(r"([A-Z]{2}\s?\d{2}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{4}[\s]?\d{0,2})", re.I)

    def extract_all(self, text: str, ocr_boxes: List = None) -> Dict[str, FieldResult]:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        total_lines = len(lines)

        sender_end = self._find_sender_section_end(lines)

        return {
            "IBAN": self._build_result("IBAN", self._extract_iban(text)),
            "Rechnungsdatum": self._build_result("Rechnungsdatum", self._extract_date(lines, "rechnungsdatum")),
            "Falligkeitsdatum": self._build_result("Falligkeitsdatum", self._extract_date(lines, "fallig")),
            "Rechnungsnummer": self._build_result("Rechnungsnummer", self._extract_invoice_number(lines)),
            "Summe": self._build_result("Summe", self._extract_total(lines, total_lines)),
            "Telefonnummer": self._build_result("Telefonnummer", self._extract_phone(lines, sender_end)),
            "Der Name der Firma": self._build_result("Der Name der Firma", self._extract_company_name(lines, sender_end)),
            "Die Adresse der Firma": self._build_result("Die Adresse der Firma", self._extract_company_address(lines, sender_end)),
            "Der Name der Bank": self._build_result("Der Name der Bank", self._extract_bank_name(lines, total_lines)),
        }

    def _build_result(self, field_name: str, match: Optional[PatternMatch]) -> FieldResult:
        if match:
            return FieldResult(field_name=field_name, value=match.value,
                             confidence=match.confidence, extraction_method=match.method)
        return FieldResult(field_name=field_name, value=None, confidence=0.0, extraction_method="not_found")

    def _find_sender_section_end(self, lines: List[str]) -> int:
        for i, line in enumerate(lines[:20]):
            if any(m in line.lower() for m in self.RECIPIENT_MARKERS):
                return i
        return min(10, len(lines))

    def _should_skip_line(self, line: str) -> bool:
        """Check if line should be skipped for company name extraction."""
        lower = line.lower()
        words = lower.split()
        if any(w in words for w in self.SKIP_EXACT):
            return True
        if any(s in lower for s in self.SKIP_CONTAINS):
            return True
        return False

    def _is_valid_company_line(self, line: str) -> bool:
        if len(line) < 3 or len(line) > 60:
            return False
        if self._should_skip_line(line):
            return False
        if any(b in line.lower() for b in self.BANK_KEYWORDS):
            return False
        if re.match(r"^\d{2}[:\.]", line):  # timestamp
            return False
        if re.match(r"^\d+\s+\w", line):  # address number
            return False
        return any(c.isalpha() for c in line)

    def _extract_company_name(self, lines: List[str], sender_end: int) -> Optional[PatternMatch]:
        """
        Extract sender company name from top section of invoice.

        Strategy:
        First check combined header lines (name|address or name - address format)
        Then look for standalone company names with legal suffixes
        Then look for business names with patterns like " & ", "-Bar"
        Finally fall back to multi-word names in top lines
        """
        candidates = []

        # look for combined header lines
        for i, line in enumerate(lines[:min(sender_end, 10)]):
            if self._should_skip_line(line):
                continue

            if "|" in line and re.search(r"\d{5}", line):
                name = line.split("|")[0].strip()
                if name and len(name) > 3:
                    candidates.append((name, 0.93, "pipe_header"))
                    continue

            if "- " in line and re.search(r"\d{5}", line):
                parts = re.split(r"\s*-\s+", line, maxsplit=1)
                name = parts[0].strip()
                if name and len(name) > 3:
                    candidates.append((name, 0.92, "dash_header"))
                    continue

        if candidates:
            best = max(candidates, key=lambda x: (x[1], len(x[0])))
            return PatternMatch(best[0], best[1], best[2])

        address_line_idx = sender_end
        for i, line in enumerate(lines[:sender_end]):
            if re.search(r"\d{5}\s*\w", line) and not self._should_skip_line(line):
                if "|" not in line and "- " not in line:  # Not a combined header
                    address_line_idx = i
                    break

        search_end = min(address_line_idx, sender_end, 8)

        for i, line in enumerate(lines[:search_end]):
            if not self._is_valid_company_line(line):
                continue

            # pure address lines
            if re.search(r"^\d+\s|str\.|straße|weg\s\d|allee\s\d|\d{5}", line.lower()):
                continue

            # incomplete fragments
            if line.rstrip().endswith(("&", "-", "|")):
                continue

            # Company with legal suffix
            if re.search(r"gmbh|ag\b|kg\b|ohg|e\.?v\.?|gbr|ug\b", line.lower()):
                candidates.append((line, 0.95, "company_suffix"))
                continue

            # Complete business name
            if " & " in line or re.search(r"-bar\b|-shop\b|-restaurant", line.lower()):
                candidates.append((line, 0.88, "business_name"))
                continue

            # Multi-word name
            if len(line.split()) >= 2 and len(line) > 10:
                candidates.append((line, 0.8, "multi_word"))
            elif len(line) >= 5:
                candidates.append((line, 0.7, "single_word"))

        if not candidates:
            return None

        best = max(candidates, key=lambda x: (x[1], len(x[0])))
        return PatternMatch(best[0], best[1], best[2])

    def _extract_company_address(self, lines: List[str], sender_end: int) -> Optional[PatternMatch]:
        # Lines with postal code pattern in sender section
        for i, line in enumerate(lines[:sender_end]):
            if self._should_skip_line(line):
                continue

            # Full address line with postal code
            if re.search(r"\d{5}\s*\w+", line):
                if i > 0 and re.search(r"str|weg|allee|platz", lines[i-1].lower()):
                    return PatternMatch(f"{lines[i-1]} {line}", 0.8, "street_city")
                return PatternMatch(line, 0.75, "postal_line")

            if "|" in line or re.search(r"\d{5}", line):
                return PatternMatch(line, 0.7, "combined_header")

        return None

    def _extract_phone(self, lines: List[str], sender_end: int) -> Optional[PatternMatch]:
        for line in lines[:sender_end + 5]:
            match = self._phone_pattern.search(line)
            if match:
                phone = re.sub(r"[^\d\s\-/+]", "", match.group(1)).strip()
                return PatternMatch(phone, 0.85, "keyword_match")

        # search entire document
        for line in lines:
            match = self._phone_pattern.search(line)
            if match:
                phone = re.sub(r"[^\d\s\-/+]", "", match.group(1)).strip()
                return PatternMatch(phone, 0.6, "fallback")
        return None

    def _extract_date(self, lines: List[str], keyword: str) -> Optional[PatternMatch]:
        for line in lines:
            if keyword in line.lower():
                dates = self._date_pattern.findall(line)
                if dates:
                    return PatternMatch(dates[0], 0.9, "keyword_match")

        for line in lines:
            if "datum:" in line.lower() and keyword == "rechnungsdatum":
                dates = self._date_pattern.findall(line)
                if dates:
                    return PatternMatch(dates[0], 0.8, "datum_fallback")
        return None

    def _extract_invoice_number(self, lines: List[str]) -> Optional[PatternMatch]:
        patterns = [
            (r"rechnungs?nummer[.:\s]+([A-Z0-9\-/]+)", 0.95),
            (r"rechnung\s*(?:nr\.?|nummer)[.:\s]*([A-Z0-9\-/]+)", 0.95),
            (r"rechnung\s*#\s*(\d+)", 0.9),
            (r"rechnungs?nr[.:\s]*([A-Z0-9\-/]+)", 0.9),
        ]

        for line in lines:
            for pattern, conf in patterns:
                match = re.search(pattern, line, re.I)
                if match:
                    value = match.group(1).strip()
                    if value and len(value) > 1 and value.lower() not in ["datum", "nummer", "nr"]:
                        return PatternMatch(value, conf, "pattern_match")
        return None

    def _extract_total(self, lines: List[str], total_lines: int) -> Optional[PatternMatch]:
        bottom_start = int(total_lines * 0.4)
        skip_keywords = ["netto", "zwischensumme", "mwst", "ust.", "steuer", "rabatt"]
        priority_keywords = ["gesamtbetrag", "rechnungsbetrag", "endbetrag", "zu zahlen", "zahlbetrag", "gesamtpreis"]

        best = None
        best_idx = -1

        for i, line in enumerate(lines[bottom_start:], bottom_start):
            lower = line.lower()

            if any(s in lower for s in skip_keywords):
                continue

            amounts = self._amount_pattern.findall(line)

            for kw in priority_keywords:
                if kw in lower:
                    if amounts:
                        return PatternMatch(amounts[-1], 0.95, "final_total")
                    if i + 1 < total_lines:
                        next_amounts = self._amount_pattern.findall(lines[i + 1])
                        if next_amounts:
                            return PatternMatch(next_amounts[-1], 0.9, "final_total_next")

            if "summe" in lower or "gesamt" in lower:
                if amounts:
                    if i > best_idx:
                        best_idx = i
                        best = PatternMatch(amounts[-1], 0.85, "total_keyword")
                elif i + 1 < total_lines:
                    next_amounts = self._amount_pattern.findall(lines[i + 1])
                    if next_amounts and i > best_idx:
                        best_idx = i
                        best = PatternMatch(next_amounts[-1], 0.8, "total_next_line")

        return best

    def _extract_bank_name(self, lines: List[str], total_lines: int) -> Optional[PatternMatch]:
        bottom_start = int(total_lines * 0.6)

        for line in lines[bottom_start:]:
            lower = line.lower()
            # skip labels and placeholders
            if any(s in lower for s in self.BANK_SKIP):
                continue
            if any(kw in lower for kw in self.BANK_KEYWORDS):
                if "iban" not in lower and len(line) < 50:
                    return PatternMatch(line.strip(), 0.8, "keyword_match")
        return None

    def _extract_iban(self, text: str) -> Optional[PatternMatch]:
        matches = self._iban_pattern.findall(text)
        if matches:
            raw = matches[0].upper().replace(" ", "")
            if len(raw) >= 18 and raw[:2].isalpha():
                formatted = " ".join(raw[i:i+4] for i in range(0, len(raw), 4))
                return PatternMatch(formatted, 0.95 if len(raw) == 22 else 0.7, "pattern_match")

        # IBAN keyword with OCR errors
        iban_match = re.search(r"IBAN[:\s]*[D\s]*([A-Z]{2}[\d\s!|]{16,26})", text, re.I)
        if iban_match:
            raw = iban_match.group(1).upper()
            cleaned = re.sub(r"[^A-Z0-9]", "", raw)
            if len(cleaned) >= 18:
                formatted = " ".join(cleaned[:22][i:i+4] for i in range(0, min(22, len(cleaned)), 4))
                return PatternMatch(formatted, 0.7, "keyword_fallback")
        return None

    def extract_line_items(self, text: str) -> List[LineItem]:
        items = []
        lines = text.split("\n")

        table_keywords = ["pos", "bezeichnung", "einzelpreis", "anzahl", "menge"]
        end_keywords = ["summe", "gesamt", "total", "netto", "brutto"]

        in_table = False
        for line in lines:
            lower = line.lower()

            if not in_table and sum(1 for k in table_keywords if k in lower) >= 2:
                in_table = True
                continue

            if in_table:
                if any(k in lower for k in end_keywords):
                    break

                item = self._parse_line_item(line)
                if item:
                    items.append(item)

        return items

    def _parse_line_item(self, line: str) -> Optional[LineItem]:
        amounts = self._amount_pattern.findall(line)
        if not amounts:
            return None

        lower = line.lower()
        if any(k in lower for k in ["iban", "bank", "tel", "summe", "gesamt", "rechnung"]):
            return None

        total = amounts[-1]
        unit_price = amounts[-2] if len(amounts) >= 2 else None

        first_amt_pos = line.find(amounts[0])
        desc = re.sub(r"^\d+[\s.)\-]*", "", line[:first_amt_pos]).strip()

        if desc and len(desc) > 2:
            return LineItem(description=desc, quantity=None, unit_price=unit_price,
                          total=total, confidence=0.6, needs_review=True)
        return None