from __future__ import annotations

from scripts.v4.cleaning import clean_text, plain_text_violations


def test_url_correction_shifts_later_spans_but_keeps_them_mappable() -> None:
    text = "Veja www.exemplo.com aqui\nTrecho citado do documento"
    cleaned = clean_text(text)
    assert "https://www.exemplo.com" in cleaned.text
    start = text.index("Trecho citado")
    span = cleaned.map_span(start, start + len("Trecho citado do documento"))
    assert cleaned.text[span.start : span.end] == "Trecho citado do documento"


def test_private_use_bullet_is_normalized_without_moving_offsets() -> None:
    text = " item um\nsegundo trecho"
    cleaned = clean_text(text)
    assert cleaned.text.startswith("•")
    start = text.index("segundo")
    span = cleaned.map_span(start, start + len("segundo trecho"))
    assert cleaned.text[span.start : span.end] == "segundo trecho"


def test_replacement_characters_are_removed() -> None:
    cleaned = clean_text("bom�dia\noutro trecho")
    assert "�" not in cleaned.text
    assert "bomdia" in cleaned.text


def test_leading_prompt_is_stripped() -> None:
    text = "Analise o documento e responda as perguntas:\nConteudo real"
    cleaned = clean_text(text)
    assert cleaned.text == "Conteudo real"


def test_span_inside_removed_region_maps_to_none() -> None:
    text = "Analise o documento e responda as perguntas:\nConteudo real"
    cleaned = clean_text(text)
    assert cleaned.map_span(0, 10) is None


def test_reports_one_event_per_applied_rule() -> None:
    cleaned = clean_text("� texto com www.x.com dentro")
    assert {event.rule for event in cleaned.events} == {
        "replacement_characters",
        "urls",
    }


def test_guarantees_plain_text_for_the_corpus() -> None:
    dirty = (
        "&lt;p&gt;Texto\x03 com​ controle e <b>marcacao</b> "
        "&amp; entidade final"
    )
    cleaned = clean_text(dirty)

    assert plain_text_violations(cleaned.text) == {}
    assert cleaned.text == "Texto com controle• e marcacao & entidade final"


def test_double_escaped_markup_is_decoded_then_stripped() -> None:
    cleaned = clean_text("antes &lt;div class=&quot;x&quot;&gt;dentro&lt;/div&gt; depois")
    assert cleaned.text == "antes dentro depois"


def test_non_html_angle_placeholders_are_preserved() -> None:
    cleaned = clean_text("assinado por <nome> em <data>")
    assert cleaned.text == "assinado por <nome> em <data>"


def test_citation_offsets_survive_markup_removal() -> None:
    source = "cabecalho <b>x</b> &amp; ruido\nTRECHO CITADO DO DOCUMENTO"
    cleaned = clean_text(source)
    start = source.index("TRECHO CITADO DO DOCUMENTO")
    span = cleaned.map_span(start, start + len("TRECHO CITADO DO DOCUMENTO"))
    assert cleaned.text[span.start : span.end] == "TRECHO CITADO DO DOCUMENTO"


def test_plain_text_violations_reports_each_kind() -> None:
    found = plain_text_violations("a\x03b​cd <p>e</p> &amp;")
    assert set(found) == {
        "control_character",
        "invisible_character",
        "private_use_character",
        "html_tag",
        "html_entity",
    }
