"""Plan exhaustive coverage from extracted sections. No model and no I/O."""

MAX_MAP_CALLS = 48
MAX_REDUCTION_CALLS = 16
UNREADABLE_WARNING = "no_native_text_ocr_or_visual_review_required"
NOTES = {
    "native_text_only_layout_tables_images_and_ocr_not_validated": (
        "Native text only; layout, tables, images and OCR were not inspected."
    ),
    "csv_strings_only_no_header_type_or_formula_inference": (
        "CSV cells were read as literal strings without header, type or formula inference."
    ),
    "csv_formula_like_text_not_executed": "Formula-like CSV text was not executed.",
    "csv_inconsistent_column_count": (
        "Inconsistent CSV row widths were retained as text and were not normalized."
    ),
}


def pack_groups(items, fits, max_groups):
    """Pack items without dropping any. Overflow is returned to the caller."""
    groups, current, overflow = [], [], []
    limited = False
    for item in items:
        if limited:
            overflow.append(item)
            continue
        if current and fits([*current, item]):
            current.append(item)
            continue
        if current:
            groups.append(current)
            current = []
            if len(groups) >= max_groups:
                limited = True
                overflow.append(item)
                continue
        if fits([item]):
            current = [item]
        else:
            overflow.append(item)
    if current and len(groups) < max_groups:
        groups.append(current)
    elif current:
        overflow.extend(current)
    return groups, overflow


def _note(notes, code):
    sentence = NOTES.get(code)
    if sentence and sentence not in notes:
        notes.append(sentence)


def _gap(page, code):
    location = dict(page.get("location") or {})
    location["part"] = -1
    return {
        "version_id": page["version_id"],
        "document_id": page["document_id"],
        "filename": page["filename"],
        "sha256": page["sha256"],
        "ordinal": page["ordinal"],
        "part": -1,
        "chunk_id": None,
        "source_id": None,
        "global_alias": None,
        "content": "",
        "location": location,
        "warnings": list(page.get("warnings") or []),
        "index_version": None,
        "char_start": None,
        "char_end": None,
        "preview_url": None,
        "status": "gap",
        "gap_code": code,
    }


def _chunk(page, chunk, workspace_id):
    location = dict(page.get("location") or {})
    location.update(
        part=chunk["part"],
        char_start=chunk["char_start"],
        char_end=chunk["char_end"],
    )
    return {
        "version_id": page["version_id"],
        "document_id": page["document_id"],
        "filename": page["filename"],
        "sha256": page["sha256"],
        "ordinal": page["ordinal"],
        "part": chunk["part"],
        "chunk_id": chunk["id"],
        "source_id": str(chunk["id"]),
        "global_alias": None,
        "content": chunk["content"],
        "location": location,
        "warnings": list(page.get("warnings") or []),
        "index_version": chunk.get("index_version"),
        "char_start": chunk["char_start"],
        "char_end": chunk["char_end"],
        "preview_url": f"/api/workspaces/{workspace_id}/sources/{chunk['id']}",
        "status": "pending",
        "gap_code": None,
    }


def build_sections(pages, workspace_id):
    """One record per indexed chunk, plus a gap for every unreadable page.

    Blank or OCR-required pages are required regions. They are not omitted and
    they cannot be marked covered.
    """
    sections, notes = [], []
    for page in pages:
        warnings = list(page.get("warnings") or [])
        for code in warnings + list(page.get("job_warnings") or []):
            _note(notes, code)
        kind = (page.get("location") or {}).get("kind")
        if kind == "pdf_page":
            _note(notes, "native_text_only_layout_tables_images_and_ocr_not_validated")
        elif kind == "csv_row":
            _note(notes, "csv_strings_only_no_header_type_or_formula_inference")
        chunks = sorted(page.get("chunks") or [], key=lambda item: item["part"])
        blank = not (page.get("content") or "").strip()
        if UNREADABLE_WARNING in warnings or blank:
            sections.append(_gap(page, "unreadable_section"))
            continue
        if not chunks:
            sections.append(_gap(page, "unindexed_section"))
            continue
        sections.extend(_chunk(page, chunk, workspace_id) for chunk in chunks)
    return sections, notes


def _slice(section, start, end):
    content = section["content"][start:end]
    origin = section["char_start"] or 0
    location = dict(section["location"])
    location.update(char_start=origin + start, char_end=origin + start + len(content))
    return {
        **section,
        "content": content,
        "char_start": origin + start,
        "char_end": origin + start + len(content),
        "location": location,
        "global_alias": None,
        "status": "pending",
        "gap_code": None,
    }


def subdivide(section, fits):
    """Break one section into the largest pieces that fit. Never drop characters."""
    if section["status"] != "pending":
        return [section]
    content = section["content"]
    if not content:
        mark_gap(section, "section_exceeds_budget")
        return [section]
    if fits(content):
        return [section]
    pieces, start = [], 0
    while start < len(content):
        low, high, best = start + 1, len(content), None
        while low <= high:
            mid = (low + high) // 2
            if fits(content[start:mid]):
                best = mid
                low = mid + 1
            else:
                high = mid - 1
        if best is None:
            remainder = _slice(section, start, len(content))
            mark_gap(remainder, "section_exceeds_budget")
            pieces.append(remainder)
            break
        pieces.append(_slice(section, start, best))
        start = best
    return pieces


def expand_sections(sections, fits):
    """Required coverage is every character, including pieces of an oversized chunk."""
    expanded = []
    for section in sections:
        expanded.extend(subdivide(section, fits))
    pending, gaps = {}, {}
    for section in expanded:
        key = (str(section["version_id"]), section["ordinal"])
        if section["status"] == "pending":
            section["part"] = pending.get(key, 0)
            pending[key] = section["part"] + 1
        else:
            gaps[key] = gaps.get(key, 0) + 1
            section["part"] = -gaps[key]
        section["location"] = {**section["location"], "part": section["part"]}
    return expanded


def mark_gap(section, code):
    section["status"] = "gap"
    section["gap_code"] = code
    section["global_alias"] = None


def fail_section(section, code):
    """Record a processing gap without renumbering aliases already issued."""
    section["status"] = "gap"
    section["gap_code"] = code


def assign_aliases(batches):
    """Stable aliases follow batch order. Gaps are not aliased."""
    hits = []
    number = 1
    for batch in batches:
        for section in batch:
            section["global_alias"] = f"S{number}"
            hits.append(section)
            number += 1
    return hits


def split_cited(batch, aliases):
    """A section is covered only when the summary cites its alias."""
    wanted = set(aliases)
    cited, missed = [], []
    for section in batch:
        if section.get("global_alias") in wanted:
            cited.append(section)
        else:
            missed.append(section)
    return cited, missed


def verdict(sections, *, reduction_complete):
    """Complete only when every required section was processed and reduced."""
    if not sections or any(section["status"] != "covered" for section in sections):
        return False
    return bool(reduction_complete)


def gap_counts(sections, extra=()):
    counts = {}
    for section in sections:
        if section["status"] != "covered" and section.get("gap_code"):
            counts[section["gap_code"]] = counts.get(section["gap_code"], 0) + 1
    for code in extra:
        counts[code] = counts.get(code, 0) + 1
    return counts


def scope_notice(sections, *, complete, notes, versions, extra_gaps=()):
    total = len(sections)
    covered = sum(section["status"] == "covered" for section in sections)
    version_count = len(versions)
    if complete:
        text = (
            f"Complete coverage of {covered} extracted sections across {version_count} "
            "selected version(s). Every required section was processed. "
            "Ordinary top-k retrieval was not used."
        )
    else:
        rendered = ", ".join(
            f"{code} ({count})"
            for code, count in sorted(gap_counts(sections, extra_gaps).items())
        )
        text = (
            f"Partial coverage: {covered} of {total} sections processed. "
            "The result is not complete."
        )
        if rendered:
            text += f" Gaps: {rendered}."
        if "reduction_incomplete" in extra_gaps and covered == total and total:
            text += " Every extracted section was visited, but the final synthesis did not finish."
        text += " Ordinary top-k retrieval was not used."
    if notes:
        text += " " + " ".join(notes)
    return text + " Not a calculation. Model wording is not a completeness claim."


def public_section(section):
    return {
        "version_id": str(section["version_id"]),
        "ordinal": section["ordinal"],
        "part": section["part"],
        "status": section["status"],
        "gap_code": section["gap_code"],
        "location": section["location"],
        "filename": section["filename"],
        "sha256": section["sha256"],
    }
