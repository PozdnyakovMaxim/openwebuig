# Unified OOXML Source Inventory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make extraction and strict corpus audit consume one canonical raw-OOXML inventory so every visible DOCX number and critical story segment is indexed exactly and consistently.

**Architecture:** A focused `ooxml_source.py` module owns ZIP/XML parsing, relationship selection, visibility rules, segment classification, and filtering. The extractor keeps its structured `python-docx` pass for numbering and layout, then reconciles it against the shared inventory; the audit delegates its source-side inventory to the same module.

**Tech Stack:** Python 3.11, stdlib `zipfile` and `xml.etree.ElementTree`, `python-docx`, `unittest`, existing corpus/chunk pipeline.

---

## File map

- Create `src/document_search/ooxml_source.py`: canonical raw OOXML parser and `SourceTextSegment`.
- Modify `src/document_search/extractor.py`: use the shared inventory for supplemental reconciliation; remove the raw textbox-only fallback.
- Modify `src/document_search/corpus_audit.py`: delegate source inventory and filtering compatibility helpers to the shared module.
- Create `tests/test_ooxml_source.py`: focused parser coverage for stories, textbox forms, `AlternateContent`, visibility and duplicate behavior.
- Modify `tests/test_extraction_pipeline.py`: end-to-end extraction regression for the real missing textbox phrase and exact numbered text.
- Modify `tests/test_corpus_audit.py`: prove audit and extraction share the same inventory contract.

### Task 1: Lock the shared inventory contract with failing tests

**Files:**
- Create: `tests/test_ooxml_source.py`
- Modify: `tests/test_extraction_pipeline.py`

- [ ] **Step 1: Write focused failing tests**

Create tests importing:

```python
from document_search.ooxml_source import SourceTextSegment, source_ooxml_inventory
```

The synthetic DOCX must assert that `source_ooxml_inventory(source_path)` returns
exactly one visible segment for each of:

```python
[
    ("body", "paragraph", "1.4 Основные принципы"),
    ("body", "textbox", "КОММЕРЧЕСКАЯ ТАЙНА"),
    ("footnote", "paragraph", "Текст связанной сноски."),
]
```

Add separate assertions that a DrawingML textbox is detected, an
`mc:AlternateContent` Choice/Fallback pair produces one segment, hidden runs are
excluded, and an unreferenced note is excluded.

- [ ] **Step 2: Add the extraction regression**

Build a DOCX whose `document.xml` contains both numbered body text and:

```xml
<w:txbxContent><w:p><w:r><w:t>КОММЕРЧЕСКАЯ ТАЙНА</w:t></w:r></w:p></w:txbxContent>
```

Assert that `extract_docx` emits one supplemental textbox block and that
`extract_docx_text` contains both `1.4 Основные принципы` and
`КОММЕРЧЕСКАЯ ТАЙНА` exactly once.

- [ ] **Step 3: Run tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest tests.test_ooxml_source tests.test_extraction_pipeline.ExtractionPipelineTest.test_shared_ooxml_inventory_recovers_real_textbox_and_numbers -v
```

Expected: failure because `document_search.ooxml_source` does not exist.

- [ ] **Step 4: Commit the red tests**

```bash
git add tests/test_ooxml_source.py tests/test_extraction_pipeline.py
git commit -m "Test unified OOXML source inventory"
```

### Task 2: Implement the canonical OOXML parser

**Files:**
- Create: `src/document_search/ooxml_source.py`
- Test: `tests/test_ooxml_source.py`

- [ ] **Step 1: Define the public contract**

```python
@dataclass(frozen=True)
class SourceTextSegment:
    part: str
    story: str
    text: str
    style: str = ""
    location: str = "paragraph"
    has_dynamic_page_field: bool = False


def source_ooxml_inventory(source: str | Path | bytes) -> dict[str, Any]:
    ...
```

The returned dictionary has `segments`, `ignored_segments`, `story_counts`, and
`location_counts`.

- [ ] **Step 2: Move the proven raw parser behind the contract**

Move the ZIP/XML relationship traversal and visibility helpers from
`corpus_audit.py` into the new module. Accept bytes using `BytesIO(source)` and
paths using `ZipFile(Path(source))`. Preserve safe target resolution, referenced
story selection, hidden-run filtering, note filtering, TOC filtering, page-field
filtering and `AlternateContent` branch selection.

- [ ] **Step 3: Run focused tests and verify GREEN**

```bash
.venv/bin/python -m unittest tests.test_ooxml_source -v
```

Expected: all shared inventory tests pass.

- [ ] **Step 4: Commit the parser**

```bash
git add src/document_search/ooxml_source.py tests/test_ooxml_source.py
git commit -m "Add canonical OOXML source inventory"
```

### Task 3: Make the audit delegate to the shared parser

**Files:**
- Modify: `src/document_search/corpus_audit.py`
- Modify: `tests/test_corpus_audit.py`

- [ ] **Step 1: Add an identity/contract test**

Patch `document_search.corpus_audit.source_ooxml_inventory`, call
`_source_ooxml_inventory(source_path)`, and assert the patched return value is
returned unchanged. Existing tests continue importing `SourceTextSegment` from
`corpus_audit`, so re-export it from the shared module.

- [ ] **Step 2: Verify the delegation test fails**

```bash
.venv/bin/python -m unittest tests.test_corpus_audit.CorpusAuditTest.test_audit_delegates_to_shared_ooxml_inventory -v
```

Expected: failure because the audit still owns its parser.

- [ ] **Step 3: Replace the independent implementation**

Import:

```python
from .ooxml_source import (
    SourceTextSegment,
    partition_source_segments as _partition_source_segments,
    source_ooxml_inventory,
)
```

Keep only:

```python
def _source_ooxml_inventory(source_path: Path) -> dict[str, Any]:
    return source_ooxml_inventory(source_path)
```

Delete duplicate XML parsing constants/helpers from `corpus_audit.py`.

- [ ] **Step 4: Run all audit tests**

```bash
.venv/bin/python -m unittest tests.test_corpus_audit -v
```

Expected: all audit tests pass, including repeated-note and duplicate-text rules.

- [ ] **Step 5: Commit audit integration**

```bash
git add src/document_search/corpus_audit.py tests/test_corpus_audit.py
git commit -m "Use shared OOXML inventory in corpus audit"
```

### Task 4: Make extraction reconcile against the same inventory

**Files:**
- Modify: `src/document_search/extractor.py`
- Modify: `tests/test_extraction_pipeline.py`

- [ ] **Step 1: Switch supplemental source segments**

Import `SourceTextSegment` and `source_ooxml_inventory`. In
`_supplemental_content_blocks`, require `source_bytes` and obtain:

```python
inventory = source_ooxml_inventory(source_bytes)
segments = inventory["segments"]
```

Group these canonical segments by `(story, normalized_text)`, retain current
occurrence accounting against structured body/table blocks, and preserve
`source_parts`, `source_locations`, and `source_occurrences`.

- [ ] **Step 2: Remove divergent fallback logic**

Delete `_raw_body_textbox_segments` and `_merge_missing_story_segments`.
Retain existing `python-docx` helpers only where they are needed for structured
document extraction, not as the source-completeness authority.

- [ ] **Step 3: Run extraction tests**

```bash
.venv/bin/python -m unittest tests.test_extraction_pipeline -v
```

Expected: all extraction, numbering, story and real textbox regressions pass.

- [ ] **Step 4: Commit extractor integration**

```bash
git add src/document_search/extractor.py tests/test_extraction_pipeline.py
git commit -m "Reconcile extraction with canonical OOXML inventory"
```

### Task 5: Verify the complete atomic rebuild path

**Files:**
- Modify only if a verified defect is found: `reindex`, `scripts/rebuild_corpus_candidate.py`, `scripts/audit_rag_corpus.py`

- [ ] **Step 1: Run targeted suites together**

```bash
.venv/bin/python -m unittest tests.test_ooxml_source tests.test_extraction_pipeline tests.test_corpus_audit -v
```

Expected: all tests pass.

- [ ] **Step 2: Run the full test suite**

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Expected: all tests pass with no failures or errors.

- [ ] **Step 3: Check source quality and repository state**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors; only intended files are modified.

- [ ] **Step 4: Commit any verified rebuild-script correction**

If and only if the rebuild-path tests expose a defect:

```bash
git add reindex scripts/rebuild_corpus_candidate.py scripts/audit_rag_corpus.py
git commit -m "Harden atomic corpus rebuild"
```

Otherwise no rebuild-script commit is created.

### Task 6: Publish and merge to main

**Files:**
- No additional source files.

- [ ] **Step 1: Push the feature branch**

```bash
git push -u origin agent/unify-ooxml-source-inventory
```

- [ ] **Step 2: Open a pull request**

Create a PR titled `Unify DOCX OOXML source inventory` describing the single
parser, exact-number preservation, textbox regression and full test result.

- [ ] **Step 3: Verify checks and merge**

Confirm required GitHub checks pass, then merge the PR into `main`.

- [ ] **Step 4: Provide the only Ubuntu commands**

```bash
git pull
./reindex
```

Expected: strict audit reports zero errors and zero warnings before atomic corpus
replacement; service health returns success.
