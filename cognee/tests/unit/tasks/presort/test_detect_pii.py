import codecs
from unittest.mock import AsyncMock, patch

import pytest

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.tasks.presort.detect_pii import (
    PiiAssessment,
    PiiCategoryAssessment,
    detect_pii,
    iban_valid,
    luhn_valid,
    redact,
    scan_content,
    scan_filename,
)
from cognee.tasks.presort.models import FileRecord
from cognee.tasks.presort.scan_folder import scan_folder


def _text_record(tmp_path, name: str, text: str) -> FileRecord:
    path = tmp_path / name
    path.write_text(text)
    return FileRecord(
        path=str(path), name=name, extension=name.rsplit(".", 1)[-1], is_text=True, size_bytes=1
    )


def test_luhn():
    assert luhn_valid("4111111111111111")  # classic Visa test number
    assert not luhn_valid("4111111111111112")
    assert not luhn_valid("1234")  # too short


def test_iban():
    assert iban_valid("DE89370400440532013000")
    assert not iban_valid("DE89370400440532013001")


def test_redact_email_and_generic():
    assert redact("jane.doe@example.com") == "j***@example.com"
    assert redact("4111111111111111") == "41***11"
    assert redact("ab") == "***"


def test_filename_hints():
    record = FileRecord(path="/x/Passport-Scan (1).pdf", name="Passport-Scan (1).pdf")
    findings = scan_filename(record)
    assert any(f.category == "identity_document" and f.severity == "high" for f in findings)

    boring = FileRecord(path="/x/holiday.jpg", name="holiday.jpg")
    assert scan_filename(boring) == []


def test_content_regexes():
    record = FileRecord(path="/x/notes.txt", name="notes.txt", is_text=True)
    text = (
        "Reach me at jane.doe@example.com or +1 555 123 4567.\n"
        "IBAN: DE89370400440532013000. SSN 123-45-6789.\n"
        "Card: 4111 1111 1111 1111. Token: sk-abcdefabcdefabcdef123456\n"
        "Invalid card: 4111 1111 1111 1112."
    )
    findings = scan_content(record, text)
    categories = {finding.category for finding in findings}
    assert categories == {
        "email_address",
        "phone_number",
        "bank_account",
        "government_id_number",
        "payment_card",
        "credential",
    }
    for finding in findings:
        if finding.redacted_sample:
            assert "jane.doe@example.com" not in finding.redacted_sample
            assert "4111 1111 1111 1111" not in finding.redacted_sample


@pytest.mark.asyncio
async def test_detect_pii_deterministic(tmp_path):
    record = _text_record(tmp_path, "notes.txt", "email me: ada@example.com")
    clean = _text_record(tmp_path, "clean.txt", "nothing personal here")

    findings = await detect_pii([record, clean])

    assert {finding.path for finding in findings} == {record.path}
    assert findings[0].category == "email_address"


@pytest.mark.asyncio
async def test_detect_pii_utf16_from_scan(messy_folder):
    files, _ = await scan_folder(messy_folder)
    record = next(record for record in files if record.name == "contact_utf16.txt")

    findings = await detect_pii(files, use_llm=False)
    contact_findings = [finding for finding in findings if finding.path == record.path]

    assert record.is_text
    assert {finding.category for finding in contact_findings} == {"email_address"}
    assert contact_findings[0].source == "content"
    assert contact_findings[0].redacted_sample == "j***@example.com"
    assert not record.warnings


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bom, encoding",
    [
        (b"", "utf-8"),
        (codecs.BOM_UTF8, "utf-8"),
        (codecs.BOM_UTF16_LE, "utf-16-le"),
        (codecs.BOM_UTF16_BE, "utf-16-be"),
        (codecs.BOM_UTF32_LE, "utf-32-le"),
        (codecs.BOM_UTF32_BE, "utf-32-be"),
    ],
)
async def test_detect_pii_bom_encodings(tmp_path, bom, encoding):
    text = "Contact: jane.doe@example.com or +1 555 123 4567.\n"
    path = tmp_path / "contact.txt"
    path.write_bytes(bom + text.encode(encoding))
    files, _ = await scan_folder(tmp_path)

    findings = await detect_pii(files, use_llm=False)

    assert files[0].is_text
    assert {finding.category for finding in findings} == {"email_address", "phone_number"}
    assert {finding.path for finding in findings} == {str(path)}
    assert all(finding.source == "content" for finding in findings)
    email = next(finding for finding in findings if finding.category == "email_address")
    assert email.redacted_sample == "j***@example.com"
    assert not files[0].warnings


@pytest.mark.asyncio
async def test_detect_pii_skips_null_containing_text_without_bom(tmp_path):
    (tmp_path / "contact.txt").write_bytes(b"Contact: jane.doe@example.com\x00\n")
    files, _ = await scan_folder(tmp_path)

    findings = await detect_pii(files, use_llm=False)

    assert not files[0].is_text
    assert findings == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bom, encoding",
    [(codecs.BOM_UTF16_LE, "utf-16-le"), (codecs.BOM_UTF16_BE, "utf-16-be")],
)
async def test_detect_pii_partial_utf16_sample(tmp_path, bom, encoding):
    text = "Contact: jane.doe@example.com\n"
    prefix = bom + text.encode(encoding)
    path = tmp_path / "contact.txt"
    path.write_bytes(prefix + "more text: later@example.org".encode(encoding))
    files, _ = await scan_folder(tmp_path)

    findings = await detect_pii(files, use_llm=False, max_sample_bytes=len(prefix) + 1)

    assert files[0].is_text
    assert len(findings) == 1
    assert findings[0].path == str(path)
    assert findings[0].category == "email_address"
    assert findings[0].redacted_sample == "j***@example.com"
    assert findings[0].detail == "1 email address(es)"
    assert not files[0].warnings


@pytest.mark.asyncio
async def test_detect_pii_sample_read_error_warns(tmp_path):
    path = tmp_path / "contact.txt"
    path.write_text("Contact: jane.doe@example.com\n", encoding="utf-16")
    files, _ = await scan_folder(tmp_path)
    path.unlink()

    findings = await detect_pii(files, use_llm=False)

    assert files[0].is_text
    assert findings == []
    assert any("could not read sample for PII scan" in warning for warning in files[0].warnings)


@pytest.mark.asyncio
async def test_detect_pii_llm_layer(tmp_path):
    record = _text_record(tmp_path, "medical_notes.txt", "patient history follows")

    assessment = PiiAssessment(
        contains_personal_data=True,
        categories=[
            PiiCategoryAssessment(
                category="medical_document", severity="high", rationale="contains a diagnosis"
            )
        ],
    )
    with patch.object(
        LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(return_value=assessment),
    ) as llm_mock:
        findings = await detect_pii([record], use_llm=True)

    llm_mock.assert_awaited_once()
    llm_findings = [finding for finding in findings if finding.source == "llm"]
    assert len(llm_findings) == 1
    assert llm_findings[0].category == "medical_document"


@pytest.mark.asyncio
async def test_llm_not_called_without_flag_or_findings(tmp_path):
    clean = _text_record(tmp_path, "clean.txt", "nothing personal here")
    with patch.object(LLMGateway, "acreate_structured_output", new=AsyncMock()) as llm_mock:
        await detect_pii([clean], use_llm=True)  # no deterministic findings -> no LLM call
        await detect_pii([clean], use_llm=False)
    llm_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_failure_is_swallowed(tmp_path):
    record = _text_record(tmp_path, "cv.txt", "resume of Ada, ada@example.com")
    with patch.object(
        LLMGateway,
        "acreate_structured_output",
        new=AsyncMock(side_effect=RuntimeError("no llm")),
    ):
        findings = await detect_pii([record], use_llm=True)
    # Deterministic findings survive; the failure lands in record warnings.
    assert any(finding.source in ("filename", "content") for finding in findings)
    assert any("LLM PII assessment failed" in warning for warning in record.warnings)
