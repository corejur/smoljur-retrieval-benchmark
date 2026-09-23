from __future__ import annotations

import pytest

from scripts.v4.headings import is_heading

REAL_HEADINGS = [
    "CARTÃO DE CRÉDITO",
    "ADVOGADOS",
    "EMPRÉSTIMOS BANCÁRIOS",
    "VALOR TOTAL DE MR DO PERIODO",
    "I - DOS FATOS",
    "III – DO DIREITO",
    "2.1. Da responsabilidade civil",
    "3.1: na apólice do seguro de vida constarão os beneficiarios",
    "DAS ALEGAÇÕES DA PARTE AUTORA",
]

IDENTIFIERS_AND_DATA = [
    "2.99.033",
    "13:59:01 Tribunal de Justiça de Minas Gerais dos e",
    "767.505.313-34 Data de Nascimento: 04/04/1976 Nome",
    "1000521-45.2026.5.02.0612",
    "12.345.678/0001-90",
    "04/04/1976",
    "5005145-88.2025.8.21.0074",
    "1.234,56",
    "",
    "   ",
]


@pytest.mark.parametrize("line", REAL_HEADINGS)
def test_accepts_real_headings(line: str) -> None:
    assert is_heading(line)


@pytest.mark.parametrize("line", IDENTIFIERS_AND_DATA)
def test_rejects_identifiers_and_data_rows(line: str) -> None:
    assert not is_heading(line)


def test_rejects_a_line_that_is_mostly_digits() -> None:
    assert not is_heading("AB 1234567890 / 9876543210 / 1111 2222 3333")


def test_rejects_an_overlong_line() -> None:
    assert not is_heading("DOS FATOS " + "E DO DIREITO " * 40)


def test_numbered_clause_needs_text_after_the_number() -> None:
    assert not is_heading("2.1. 04/04/1976")
    assert is_heading("2.1. Da cobranca indevida")


@pytest.mark.parametrize("line", ["F KO 4MQ33E", "A B C D", "XY ZW 12AB3"])
def test_rejects_ocr_debris_without_a_real_word(line: str) -> None:
    assert not is_heading(line)


def test_accepts_a_single_real_word() -> None:
    assert is_heading("PROCESSO")
