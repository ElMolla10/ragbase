"""Write the synthetic acceptance corpus. Re-run to regenerate; do not hand-edit cases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from acceptance import CONSTITUTION_VERSION, CORPUS_VERSION, REVIEW_STATUS
from acceptance.schema import make_gold, validate_case, validate_corpus

NOTE = (
    "Machine-drafted synthetic case pending human review. "
    "Not a company document and not a reviewed gold label."
)
FACTS = (
    {
        "key": "berth-fee",
        "en": "Synthetic Harbor Cooperative Berth A daily fee is 42.50 EGP.",
        "ar": "مؤسسة الواحة التجريبية: رسوم الرصيف أ اليومية هي 42.50 جنيه.",
        "distractor_en": "Synthetic River Cooperative Berth A daily fee is 19.00 EGP.",
        "distractor_ar": "جمعية النهر التجريبية: رسوم الرصيف أ اليومية هي 19.00 جنيه.",
        "v1": "Version v1 previous Berth A daily fee is 42.50 EGP.",
        "v2": "Version v2 current Berth A daily fee is 47.00 EGP.",
        "en_q": "What is the Synthetic Harbor Cooperative daily fee for Berth A?",
        "en_ambiguous": "What is the daily fee for Berth A?",
        "ar_q": "ما رسوم الرصيف أ اليومية في مؤسسة الواحة التجريبية؟",
        "ar_ambiguous": "ما رسوم الرصيف أ اليومية؟",
        "eg_q": "رسوم الرصيف أ في مؤسسة الواحة التجريبية كام في اليوم؟",
        "eg_paraphrase": "الفلوس بتاعة المكان البحري الأول كام؟",
        "shared": "ما الرسوم اليومية لـ Berth A في Synthetic Harbor Cooperative؟",
        "unshared": "ما الرسوم اليومية للرصيف الأول حسب النص الإنجليزي؟",
        "v2_q": "What is the current Berth A daily fee in version v2?",
    },
    {
        "key": "draft",
        "en": "Synthetic Harbor Cooperative maximum draft is 11.5 metres.",
        "ar": "مؤسسة الواحة التجريبية: أقصى غاطس مسموح هو 11.5 متر.",
        "distractor_en": "Synthetic River Cooperative maximum draft is 8.2 metres.",
        "distractor_ar": "جمعية النهر التجريبية: أقصى غاطس مسموح هو 8.2 متر.",
        "v1": "Version v1 previous maximum draft is 11.5 metres.",
        "v2": "Version v2 current maximum draft is 12.0 metres.",
        "en_q": "What is the Synthetic Harbor Cooperative maximum draft?",
        "en_ambiguous": "What is the maximum draft?",
        "ar_q": "ما أقصى غاطس مسموح في مؤسسة الواحة التجريبية؟",
        "ar_ambiguous": "ما أقصى غاطس مسموح؟",
        "eg_q": "أقصى غاطس في مؤسسة الواحة التجريبية كام؟",
        "eg_paraphrase": "المركب تقدر تنزل قد إيه في المية؟",
        "shared": "ما أقصى draft في Synthetic Harbor Cooperative؟",
        "unshared": "ما الحد الأقصى لنزول السفن في النص الإنجليزي؟",
        "v2_q": "What is the current maximum draft in version v2?",
    },
    {
        "key": "vhf",
        "en": "Synthetic Harbor Cooperative emergency channel is VHF 16.",
        "ar": "مؤسسة الواحة التجريبية: قناة الطوارئ هي VHF 16.",
        "distractor_en": "Synthetic River Cooperative emergency channel is VHF 9.",
        "distractor_ar": "جمعية النهر التجريبية: قناة الطوارئ هي VHF 9.",
        "v1": "Version v1 previous emergency channel is VHF 16.",
        "v2": "Version v2 current emergency channel is VHF 12.",
        "en_q": "What is the Synthetic Harbor Cooperative emergency channel?",
        "en_ambiguous": "What is the emergency channel?",
        "ar_q": "ما قناة الطوارئ في مؤسسة الواحة التجريبية؟",
        "ar_ambiguous": "ما قناة الطوارئ؟",
        "eg_q": "قناة الطوارئ في مؤسسة الواحة التجريبية إيه؟",
        "eg_paraphrase": "لو حصل بلاغ بنكلم مين على اللاسلكي؟",
        "shared": "ما قناة VHF في Synthetic Harbor Cooperative؟",
        "unshared": "ما وسيلة النداء عند الخطر في النص الإنجليزي؟",
        "v2_q": "What is the current emergency channel in version v2?",
    },
    {
        "key": "warehouse",
        "en": "Synthetic Harbor Cooperative warehouse B capacity is 1800 pallets.",
        "ar": "مؤسسة الواحة التجريبية: سعة المخزن ب هي 1800 طبلية.",
        "distractor_en": "Synthetic River Cooperative warehouse B capacity is 640 pallets.",
        "distractor_ar": "جمعية النهر التجريبية: سعة المخزن ب هي 640 طبلية.",
        "v1": "Version v1 previous warehouse B capacity is 1800 pallets.",
        "v2": "Version v2 current warehouse B capacity is 1760 pallets.",
        "en_q": "What is the Synthetic Harbor Cooperative warehouse B capacity?",
        "en_ambiguous": "What is the warehouse B capacity?",
        "ar_q": "ما سعة المخزن ب في مؤسسة الواحة التجريبية؟",
        "ar_ambiguous": "ما سعة المخزن ب؟",
        "eg_q": "سعة المخزن ب في مؤسسة الواحة التجريبية كام طبلية؟",
        "eg_paraphrase": "العنبر بياخد كم صندوق بالكتير؟",
        "shared": "ما سعة warehouse B في Synthetic Harbor Cooperative؟",
        "unshared": "كم تتسع مساحة التخزين الثانية في النص الإنجليزي؟",
        "v2_q": "What is the current warehouse B capacity in version v2?",
    },
    {
        "key": "pilotage",
        "en": "Synthetic Harbor Cooperative pilotage window is 06:00-18:00.",
        "ar": "مؤسسة الواحة التجريبية: نافذة الإرشاد من 06:00 إلى 18:00.",
        "distractor_en": "Synthetic River Cooperative pilotage window is 08:00-16:00.",
        "distractor_ar": "جمعية النهر التجريبية: نافذة الإرشاد من 08:00 إلى 16:00.",
        "v1": "Version v1 previous pilotage window is 06:00-18:00.",
        "v2": "Version v2 current pilotage window is 05:30-19:00.",
        "en_q": "What is the Synthetic Harbor Cooperative pilotage window?",
        "en_ambiguous": "What is the pilotage window?",
        "ar_q": "ما نافذة الإرشاد في مؤسسة الواحة التجريبية؟",
        "ar_ambiguous": "ما نافذة الإرشاد؟",
        "eg_q": "نافذة الإرشاد في مؤسسة الواحة التجريبية من إمتى لإمتى؟",
        "eg_paraphrase": "المرشد بيشتغل من كام لكام؟",
        "shared": "ما نافذة pilotage في Synthetic Harbor Cooperative؟",
        "unshared": "ما ساعات عمل المرشد في النص الإنجليزي؟",
        "v2_q": "What is the current pilotage window in version v2?",
    },
    {
        "key": "surcharge",
        "en": "Synthetic Harbor Cooperative night pilot surcharge for tanker calls is 15.75 percent.",
        "ar": "مؤسسة الواحة التجريبية: علاوة الإرشاد الليلي لناقلات النفط هي 15.75 في المائة.",
        "distractor_en": "Synthetic River Cooperative night pilot surcharge for tanker calls is 4.00 percent.",
        "distractor_ar": "جمعية النهر التجريبية: علاوة الإرشاد الليلي لناقلات النفط هي 4.00 في المائة.",
        "v1": "Version v1 previous night pilot surcharge for tanker calls is 15.75 percent.",
        "v2": "Version v2 current night pilot surcharge for tanker calls is 16.25 percent.",
        "en_q": "What is the Synthetic Harbor Cooperative night pilot surcharge for tanker calls?",
        "en_ambiguous": "What is the night pilot surcharge for tanker calls?",
        "ar_q": "ما علاوة الإرشاد الليلي لناقلات النفط في مؤسسة الواحة التجريبية؟",
        "ar_ambiguous": "ما علاوة الإرشاد الليلي لناقلات النفط؟",
        "eg_q": "علاوة الإرشاد الليلي لناقلات النفط في مؤسسة الواحة التجريبية كام؟",
        "eg_paraphrase": "الناس بتدفع زيادة كام بالليل للسفن الكبيرة؟",
        "shared": "ما night pilot surcharge في Synthetic Harbor Cooperative؟",
        "unshared": "ما الزيادة الليلية على ناقلات النفط في النص الإنجليزي؟",
        "v2_q": "What is the current night pilot surcharge for tanker calls in version v2?",
    },
    {
        "key": "fresh-water",
        "en": "Synthetic Harbor Cooperative fresh water price is 25.00 EGP per tonne.",
        "ar": "مؤسسة الواحة التجريبية: سعر المياه العذبة 25.00 جنيه للطن.",
        "distractor_en": "Synthetic River Cooperative fresh water price is 11.00 EGP per tonne.",
        "distractor_ar": "جمعية النهر التجريبية: سعر المياه العذبة 11.00 جنيه للطن.",
        "en_q": "What is the Synthetic Harbor Cooperative fresh water price?",
        "en_ambiguous": "What is the fresh water price?",
        "ar_q": "ما سعر المياه العذبة في مؤسسة الواحة التجريبية؟",
        "ar_ambiguous": "ما سعر المياه العذبة؟",
        "eg_q": "سعر المياه العذبة في مؤسسة الواحة التجريبية كام؟",
        "shared": "ما سعر fresh water في Synthetic Harbor Cooperative؟",
        "unshared": "بكم تباع مياه الشرب للطن في النص الإنجليزي؟",
    },
    {
        "key": "gangway",
        "en": "Synthetic Harbor Cooperative gangway rule requires two crew.",
        "ar": "مؤسسة الواحة التجريبية: قاعدة ممر الصعود تشترط فردين من الطاقم.",
        "distractor_en": "Synthetic River Cooperative gangway rule requires one crew.",
        "distractor_ar": "جمعية النهر التجريبية: قاعدة ممر الصعود تشترط فردا واحدا من الطاقم.",
        "en_q": "What does the Synthetic Harbor Cooperative gangway rule require?",
        "en_ambiguous": "What does the gangway rule require?",
        "ar_q": "ماذا تشترط قاعدة ممر الصعود في مؤسسة الواحة التجريبية؟",
        "ar_ambiguous": "ماذا تشترط قاعدة ممر الصعود؟",
        "eg_q": "قاعدة ممر الصعود في مؤسسة الواحة التجريبية بتطلب كام فرد؟",
        "shared": "ماذا تشترط gangway rule في Synthetic Harbor Cooperative؟",
        "unshared": "كم شخصا يلزم عند سلم النزول في النص الإنجليزي؟",
    },
    {
        "key": "fog",
        "en": "Synthetic Harbor Cooperative fog signal is one long blast.",
        "ar": "مؤسسة الواحة التجريبية: إشارة الضباب هي صفارة طويلة واحدة.",
        "distractor_en": "Synthetic River Cooperative fog signal is two short blasts.",
        "distractor_ar": "جمعية النهر التجريبية: إشارة الضباب هي صفارتان قصيرتان.",
        "en_q": "What is the Synthetic Harbor Cooperative fog signal?",
        "en_ambiguous": "What is the fog signal?",
        "ar_q": "ما إشارة الضباب في مؤسسة الواحة التجريبية؟",
        "ar_ambiguous": "ما إشارة الضباب؟",
        "eg_q": "إشارة الضباب في مؤسسة الواحة التجريبية إيه؟",
        "shared": "ما fog signal في Synthetic Harbor Cooperative؟",
        "unshared": "كيف يكون تنبيه الشبورة في النص الإنجليزي؟",
    },
    {
        "key": "quarantine",
        "en": "Synthetic Harbor Cooperative quarantine office is room 12.",
        "ar": "مؤسسة الواحة التجريبية: مكتب الحجر الصحي هو الغرفة 12.",
        "distractor_en": "Synthetic River Cooperative quarantine office is room 3.",
        "distractor_ar": "جمعية النهر التجريبية: مكتب الحجر الصحي هو الغرفة 3.",
        "en_q": "Where is the Synthetic Harbor Cooperative quarantine office?",
        "en_ambiguous": "Where is the quarantine office?",
        "ar_q": "أين مكتب الحجر الصحي في مؤسسة الواحة التجريبية؟",
        "ar_ambiguous": "أين مكتب الحجر الصحي؟",
        "eg_q": "مكتب الحجر الصحي في مؤسسة الواحة التجريبية فين؟",
        "shared": "أين quarantine office في Synthetic Harbor Cooperative؟",
        "unshared": "أين تتم إجراءات الصحة للسفن في النص الإنجليزي؟",
    },
    {
        "key": "spill",
        "en": "Synthetic Harbor Cooperative spill kit is in locker 4.",
        "ar": "مؤسسة الواحة التجريبية: عدة الانسكاب في الخزانة 4.",
        "distractor_en": "Synthetic River Cooperative spill kit is in locker 9.",
        "distractor_ar": "جمعية النهر التجريبية: عدة الانسكاب في الخزانة 9.",
        "en_q": "Where is the Synthetic Harbor Cooperative spill kit?",
        "en_ambiguous": "Where is the spill kit?",
        "ar_q": "أين عدة الانسكاب في مؤسسة الواحة التجريبية؟",
        "ar_ambiguous": "أين عدة الانسكاب؟",
        "eg_q": "عدة الانسكاب في مؤسسة الواحة التجريبية فين؟",
        "shared": "أين spill kit في Synthetic Harbor Cooperative؟",
        "unshared": "أين توضع أدوات التسريب في النص الإنجليزي؟",
    },
    {
        "key": "tide",
        "en": "Synthetic Harbor Cooperative tide board is updated at 05:30.",
        "ar": "مؤسسة الواحة التجريبية: تحديث لوحة المد والجزر عند 05:30.",
        "distractor_en": "Synthetic River Cooperative tide board is updated at 07:45.",
        "distractor_ar": "جمعية النهر التجريبية: تحديث لوحة المد والجزر عند 07:45.",
        "en_q": "When is the Synthetic Harbor Cooperative tide board updated?",
        "en_ambiguous": "When is the tide board updated?",
        "ar_q": "متى يكون تحديث لوحة المد والجزر في مؤسسة الواحة التجريبية؟",
        "ar_ambiguous": "متى يكون تحديث لوحة المد والجزر؟",
        "eg_q": "تحديث لوحة المد والجزر في مؤسسة الواحة التجريبية الساعة كام؟",
        "shared": "متى يتحدث tide board في Synthetic Harbor Cooperative؟",
        "unshared": "في أي ساعة تُراجع لوحة منسوب البحر في النص الإنجليزي؟",
    },
)
MIXED = (
    (
        "sku-4401",
        "رمز SKU-4401 مخزن في الساحة C والكمية 86.",
        "ما كمية SKU-4401 في الساحة C؟",
        "What quantity is stored for SKU-4401 in yard C?",
    ),
    (
        "sku-4402",
        "رمز SKU-4402 مخزن في الساحة D والكمية 14.",
        "ما كمية SKU-4402 في الساحة D؟",
        "What quantity is stored for SKU-4402 in yard D?",
    ),
    (
        "valve-f2",
        "صمام VALVE-F2 هو قطع الوقود.",
        "ما وظيفة VALVE-F2؟",
        "What does VALVE-F2 shut off?",
    ),
    (
        "badge",
        "شارة BADGE-YELLOW خاصة بالزوار.",
        "لمن شارة BADGE-YELLOW؟",
        "Who uses BADGE-YELLOW?",
    ),
    (
        "crane-shift",
        "وردية CRANE-SHIFT-B تضم 6 أفراد.",
        "كم فردا في CRANE-SHIFT-B؟",
        "How many people are on CRANE-SHIFT-B?",
    ),
    (
        "tank-18",
        "خزان TANK-18 سعته 4000 لتر.",
        "ما سعة TANK-18؟",
        "What is the capacity of TANK-18?",
    ),
    (
        "iso-days",
        "فترة التفتيش ISO-9001 هي 90 يوما.",
        "ما فترة ISO-9001؟",
        "What is the ISO-9001 interval?",
    ),
    (
        "radio-check",
        "فحص الراديو RADIO-EVEN يتم في الساعات الزوجية.",
        "متى يتم RADIO-EVEN؟",
        "When does RADIO-EVEN happen?",
    ),
)
LONG = (
    (
        "long-dawn",
        "The dawn crane inspection code is CRANE-DAWN-17.",
        "What is the dawn crane inspection code?",
    ),
    (
        "long-mid",
        "The midwatch ballast sample code is BALLAST-MID-45.",
        "What is the midwatch ballast sample code?",
    ),
    (
        "long-night",
        "The night gate seal code is GATE-NIGHT-90.",
        "What is the night gate seal code?",
    ),
)


def write_corpus(path: Path) -> dict:
    fixture_dir = path / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    _write_fixtures(fixture_dir)
    cases = _cases()
    summary = validate_corpus(cases)
    lines = [
        json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for case in cases
    ]
    body = ("\n".join(lines) + "\n").encode()
    (path / "cases.jsonl").write_bytes(body)
    manifest = {
        "corpus_version": CORPUS_VERSION,
        "constitution_version": CONSTITUTION_VERSION,
        "review_status": REVIEW_STATUS,
        "reviewed": False,
        "synthetic_only": True,
        "contains_company_documents": False,
        "case_count": summary["cases"],
        "arabic_or_mixed_count": summary["arabic_or_mixed"],
        "tuning_count": summary["tuning"],
        "held_out_count": summary["held_out"],
        "matched_pair_count": summary["matched_pairs"],
        "ocr_pending_count": summary["ocr_pending"],
        "chart_pending_count": summary["chart_pending"],
        "cases_sha256": hashlib.sha256(body).hexdigest(),
        "notes": (
            "Machine-drafted synthetic fixtures only. Pending human review. "
            "Scanned and chart cases are pending. This corpus does not qualify production."
        ),
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (path / "README.md").write_text(_readme(summary), encoding="utf-8")
    (fixture_dir / "README.md").write_text(
        "Synthetic nonconfidential fixtures for the acceptance corpus.\n"
        "No company document is stored here.\n"
        "Scanned pages are intentionally absent because OCR is not implemented.\n"
        "Chart images are intentionally absent because visual interpretation is not implemented.\n",
        encoding="utf-8",
    )
    return manifest


def _readme(summary: dict) -> str:
    return "\n".join(
        [
            "# Acceptance corpus 1.0.0",
            "",
            "Machine-drafted synthetic cases. Pending human review. Not production qualification.",
            "",
            "هذه الحالات مصطنعة ومسودّة آليًا وبانتظار مراجعة بشرية. "
            "ليست وثائق الشركة، ولا يُعد تشغيلها تأهيلاً للإنتاج.",
            "",
            f"Cases: {summary['cases']}. Arabic or mixed: {summary['arabic_or_mixed']}. "
            f"Tuning: {summary['tuning']}. Held out: {summary['held_out']}.",
            f"Matched pairs: {summary['matched_pairs']}. "
            f"OCR pending: {summary['ocr_pending']}. Chart pending: {summary['chart_pending']}.",
            "",
            "Reviewed cases in this version: 0.",
            "Company documents: none.",
            "Scanned pages and chart images are not in the repository. Those cases stay pending.",
            "",
            "Splits are assigned inside each family: indexes 0 and 1 modulo 5 are held out.",
            "Do not tune prompts or retrieval on the held-out split.",
            "",
        ]
    )


def _write_fixtures(fixture_dir: Path) -> None:
    def elements(name, rows):
        chunks = []
        for element_id, text, flags in rows:
            flag = f" {flags}" if flags else ""
            chunks.append(f":::element {element_id}{flag}\n{text}\n:::\n")
        (fixture_dir / name).write_text("".join(chunks), encoding="utf-8")

    elements(
        "harbor-en-v1.md",
        [(f"en-harbor-{fact['key']}", fact["en"], "") for fact in FACTS],
    )
    elements(
        "river-en.md",
        [(f"en-distractor-{fact['key']}", fact["distractor_en"], "") for fact in FACTS],
    )
    elements(
        "harbor-ar.md",
        [(f"ar-harbor-{fact['key']}", fact["ar"], "") for fact in FACTS],
    )
    elements(
        "river-ar.md",
        [(f"ar-distractor-{fact['key']}", fact["distractor_ar"], "") for fact in FACTS],
    )
    elements(
        "harbor-en-versions.md",
        [
            (f"en-{label}-{fact['key']}", fact[label], "")
            for fact in FACTS
            if "v1" in fact
            for label in ("v1", "v2")
        ],
    )
    elements(
        "bilingual.md",
        [(f"bi-{fact['key']}", fact["en"] + " " + fact["ar"], "") for fact in FACTS],
    )
    elements(
        "mixed-terms.md",
        [(f"mixed-{key}", text, "") for key, text, _ar, _en in MIXED],
    )
    long_rows = []
    for index in range(1, 91):
        special = {1: LONG[0], 45: LONG[1], 90: LONG[2]}.get(index)
        if special:
            long_rows.append((special[0], special[1], ""))
        else:
            long_rows.append(
                (
                    f"long-filler-{index:02d}",
                    f"Synthetic harbor log filler section {index} records ordinary pallet tally {index}.",
                    "",
                )
            )
    elements("long-log.md", long_rows)
    elements(
        "coverage-complete.md",
        [
            ("cov-complete-1", "The synthetic harbor keeps a pilot roster.", ""),
            ("cov-complete-2", "The synthetic harbor keeps a berth roster.", ""),
            ("cov-complete-3", "The synthetic harbor keeps a warehouse roster.", ""),
            ("cov-complete-4", "The synthetic harbor keeps a radio roster.", ""),
        ],
    )
    elements(
        "coverage-gap.md",
        [
            ("cov-gap-1", "The synthetic harbor lists the north shed.", ""),
            ("cov-gap-blank", "", ""),
            ("cov-gap-3", "The synthetic harbor lists the south shed.", ""),
        ],
    )
    elements(
        "coverage-ocr.md",
        [
            ("cov-ocr-1", "The synthetic harbor has a readable introduction.", ""),
            ("cov-ocr-2", "The synthetic harbor has a readable closing note.", ""),
            ("cov-ocr-3", "", "ocr"),
        ],
    )
    elements(
        "denied-pin.md",
        [
            (
                "denied-pin",
                "The restricted synthetic warehouse PIN example is PIN-SYN-4401.",
                "",
            )
        ],
    )
    csv_files = {
        "basic.csv": "id,amount\n001,10.10\n002,20.20\n003,0.30\n",
        "arabic-sales.csv": "الصنف,المبيعات\nأ,10.00\nب,5.50\nالمجموع,999.00\n",
        "regions.csv": "region,amount\nNorth,1.00\nNorth,4.00\nSouth,9.00\n",
        "sku-fees.csv": "sku,fee\nSKU-4401,1.25\nSKU-4402,2.75\n",
        "values.csv": "item,القيمة\nalpha,100.00\nbeta,0.01\ngamma,0.02\n",
        "mixed-currency.csv": "item,currency,amount\na,USD,10.00\nb,EGP,5.00\n",
    }
    for name, text in csv_files.items():
        (fixture_dir / name).write_text(text, encoding="utf-8")


def _cases() -> list[dict]:
    cases: list[dict] = []
    counters: dict[str, int] = {}

    def add(**kwargs):
        family = kwargs["family"]
        index = counters.get(family, 0)
        counters[family] = index + 1
        register = kwargs.get("register")
        language = kwargs["language"]
        slices = [family]
        if language in {"ar", "mixed"}:
            slices.append("arabic_or_mixed")
        if register:
            slices.append("arabic_" + register)
        slices.append(kwargs["task"])
        slices.extend(kwargs.get("extra") or [])
        case = {
            "id": f"acc-v1-{len(cases) + 1:04d}",
            "corpus_version": CORPUS_VERSION,
            "constitution_version": CONSTITUTION_VERSION,
            "split": "held_out" if index % 5 in {0, 1} else "tuning",
            "review_status": REVIEW_STATUS,
            "reviewed": False,
            "synthetic": True,
            "confidential": False,
            "language": language,
            "arabic_register": register,
            "matched_pair_id": kwargs.get("pair"),
            "task": kwargs["task"],
            "family": family,
            "slices": list(dict.fromkeys(slices)),
            "question": kwargs["question"],
            "actor": kwargs.get("actor", "member"),
            "documents": kwargs["documents"],
            "gold": kwargs["gold"],
            "notes": kwargs.get("notes", NOTE),
        }
        validate_case(case)
        cases.append(case)

    harbor_ar = [
        _doc("fixtures/harbor-ar.md", "harbor-ar", "ar-v1", "supporting"),
        _doc("fixtures/river-ar.md", "river-ar", "ar-v1", "distractor"),
    ]
    harbor_en = [
        _doc("fixtures/harbor-en-v1.md", "harbor-en", "v1", "supporting"),
        _doc("fixtures/river-en.md", "river-en", "v1", "distractor"),
    ]
    for fact in FACTS:
        add(
            family="formal_ar",
            language="ar",
            register="formal",
            task="evidence",
            question=fact["ar_q"],
            documents=harbor_ar,
            gold=make_gold(supporting_element_ids=[f"ar-harbor-{fact['key']}"]),
        )
        add(
            family="formal_ar",
            language="ar",
            register="formal",
            task="evidence",
            question=fact["ar_ambiguous"],
            documents=harbor_ar,
            gold=make_gold(supporting_element_ids=[f"ar-harbor-{fact['key']}"]),
            extra=["distractor_ambiguity"],
            notes=NOTE + " Common words also appear in the river distractor.",
        )
    for fact in FACTS:
        add(
            family="egyptian",
            language="ar",
            register="egyptian",
            task="evidence",
            question=fact["eg_q"],
            documents=harbor_ar,
            gold=make_gold(supporting_element_ids=[f"ar-harbor-{fact['key']}"]),
        )
    for fact in FACTS[:6]:
        add(
            family="egyptian",
            language="ar",
            register="egyptian",
            task="evidence",
            question=fact["eg_paraphrase"],
            documents=harbor_ar,
            gold=make_gold(supporting_element_ids=[f"ar-harbor-{fact['key']}"]),
            extra=["egyptian_paraphrase"],
            notes=NOTE + " Egyptian paraphrase does not reuse the source wording.",
        )
    for key, _text, arabic, english in MIXED:
        add(
            family="mixed_terminology",
            language="mixed",
            register="mixed_terminology",
            task="evidence",
            question=arabic,
            documents=[_doc("fixtures/mixed-terms.md", "mixed", "v1", "supporting")],
            gold=make_gold(supporting_element_ids=[f"mixed-{key}"]),
        )
        add(
            family="mixed_terminology",
            language="en",
            task="evidence",
            question=english,
            documents=[_doc("fixtures/mixed-terms.md", "mixed", "v1", "supporting")],
            gold=make_gold(supporting_element_ids=[f"mixed-{key}"]),
        )
    for key, question in (
        ("sku-4401", "ما كمية الصنف المخزن في الساحة الغربية؟"),
        ("sku-4402", "ما كمية الصنف المخزن في الساحة الشرقية؟"),
        ("valve-f2", "ما اسم أداة إيقاف السولار غير المذكورة برقم؟"),
        ("badge", "ما لون بطاقة الضيوف بدون ذكر الرمز؟"),
    ):
        add(
            family="mixed_terminology",
            language="mixed",
            register="mixed_terminology",
            task="evidence",
            question=question,
            documents=[_doc("fixtures/mixed-terms.md", "mixed", "v1", "supporting")],
            gold=make_gold(supporting_element_ids=[f"mixed-{key}"]),
            extra=["mixed_paraphrase"],
            notes=NOTE + " The paraphrase drops the Latin identifier.",
        )
    for fact in FACTS[:10]:
        add(
            family="arabic_over_english",
            language="ar",
            register="arabic_over_english",
            task="evidence",
            question=fact["shared"],
            documents=harbor_en,
            gold=make_gold(supporting_element_ids=[f"en-harbor-{fact['key']}"]),
            extra=["arabic_over_english_shared_token"],
        )
    for fact in FACTS[:8]:
        add(
            family="arabic_over_english",
            language="ar",
            register="arabic_over_english",
            task="evidence",
            question=fact["unshared"],
            documents=harbor_en,
            gold=make_gold(supporting_element_ids=[f"en-harbor-{fact['key']}"]),
            extra=["arabic_over_english_unshared"],
            notes=NOTE
            + " Arabic question over an English source with no shared token.",
        )
    bilingual = [_doc("fixtures/bilingual.md", "bilingual", "v1", "supporting")]
    for number, fact in enumerate(FACTS, 1):
        pair = f"pair-{number:02d}"
        add(
            family="matched_pair",
            language="en",
            register="matched_pair",
            task="evidence",
            question=fact["en_q"],
            documents=bilingual,
            gold=make_gold(supporting_element_ids=[f"bi-{fact['key']}"]),
            pair=pair,
        )
        add(
            family="matched_pair",
            language="ar",
            register="matched_pair",
            task="evidence",
            question=fact["ar_q"],
            documents=bilingual,
            gold=make_gold(supporting_element_ids=[f"bi-{fact['key']}"]),
            pair=pair,
        )
    for number, fact in enumerate(FACTS[:8], 13):
        pair = f"pair-{number:02d}"
        paraphrase = "eg_paraphrase" in fact and number >= 17
        add(
            family="matched_pair",
            language="en",
            register="matched_pair",
            task="evidence",
            question=fact["en_q"],
            documents=bilingual,
            gold=make_gold(supporting_element_ids=[f"bi-{fact['key']}"]),
            pair=pair,
        )
        add(
            family="matched_pair",
            language="ar",
            register="matched_pair",
            task="evidence",
            question=fact["eg_paraphrase"] if paraphrase else fact["ar_ambiguous"],
            documents=bilingual,
            gold=make_gold(supporting_element_ids=[f"bi-{fact['key']}"]),
            pair=pair,
            extra=["matched_pair_paraphrase"] if paraphrase else [],
            notes=NOTE
            + (
                " Arabic side is a paraphrase for the matched-pair gap."
                if paraphrase
                else " Arabic side reuses source words."
            ),
        )
    for fact in FACTS:
        add(
            family="english_evidence",
            language="en",
            task="evidence",
            question=fact["en_q"],
            documents=harbor_en,
            gold=make_gold(supporting_element_ids=[f"en-harbor-{fact['key']}"]),
        )
        add(
            family="english_evidence",
            language="en",
            task="evidence",
            question=fact["en_ambiguous"],
            documents=harbor_en,
            gold=make_gold(supporting_element_ids=[f"en-harbor-{fact['key']}"]),
            extra=["distractor_ambiguity"],
        )
    variants = (
        "State the {label} from the synthetic harbor log.",
        "Identify the {label} recorded in the long log.",
        "Which code is the {label}?",
    )
    labels = {
        "long-dawn": "dawn crane inspection code",
        "long-mid": "midwatch ballast sample code",
        "long-night": "night gate seal code",
    }
    long_doc = [_doc("fixtures/long-log.md", "long-log", "v1", "supporting")]
    for element_id, _text, question in LONG:
        add(
            family="long_document",
            language="en",
            task="evidence",
            question=question,
            documents=long_doc,
            gold=make_gold(supporting_element_ids=[element_id]),
            notes=NOTE + " Lexical overlap is not a context-window position test.",
        )
        for template in variants:
            add(
                family="long_document",
                language="en",
                task="evidence",
                question=template.format(label=labels[element_id]),
                documents=long_doc,
                gold=make_gold(supporting_element_ids=[element_id]),
                notes=NOTE
                + " Fact is placed at the start, middle, or end of a long fixture.",
            )
    versions = [
        _doc(
            "fixtures/harbor-en-versions.md",
            "harbor-versions",
            "v1-and-v2",
            "supporting",
        )
    ]
    for fact in FACTS:
        if "v2" not in fact:
            continue
        add(
            family="version_conflict",
            language="en",
            task="version_conflict",
            question=fact["v2_q"],
            documents=versions,
            gold=make_gold(supporting_element_ids=[f"en-v2-{fact['key']}"]),
        )
        add(
            family="version_conflict",
            language="en",
            task="version_conflict",
            question=fact["en_ambiguous"],
            documents=versions,
            gold=make_gold(supporting_element_ids=[f"en-v2-{fact['key']}"]),
            extra=["version_ambiguous"],
            notes=NOTE + " Both versions share the common words. Current gold is v2.",
        )
    absent_scope = [
        _doc("fixtures/harbor-en-v1.md", "harbor-en", "v1", "supporting"),
        _doc("fixtures/harbor-ar.md", "harbor-ar", "ar-v1", "supporting"),
    ]
    for language, register, question in (
        ("en", None, "Where is the submarine dock?"),
        ("en", None, "Where is the helicopter pad?"),
        ("en", None, "What is the casino opening hour?"),
        ("en", None, "Which vineyard supplies the galley?"),
        ("en", None, "What is the marble quarry output?"),
        ("en", None, "When does the ski lift open?"),
        ("ar", "formal", "أين مهبط المروحية؟"),
        ("ar", "formal", "ما مواعيد صالة الكازينو؟"),
        ("ar", "egyptian", "المحجر الرخامي إنتاجه إيه؟"),
        ("ar", "egyptian", "مصعد التزلج بيفتح امتى؟"),
    ):
        add(
            family="absent",
            language=language,
            register=register,
            task="absent",
            question=question,
            documents=absent_scope,
            gold=make_gold(expect_refusal=True),
        )
    for language, register, question, documents in (
        (
            "en",
            None,
            "What is the Synthetic Harbor Cooperative daily fee for Berth A?",
            [_doc("fixtures/river-en.md", "river-en", "v1", "distractor")],
        ),
        (
            "en",
            None,
            "What is the Synthetic Harbor Cooperative maximum draft?",
            [_doc("fixtures/river-en.md", "river-en", "v1", "distractor")],
        ),
        (
            "en",
            None,
            "What is the Synthetic Harbor Cooperative emergency channel?",
            [_doc("fixtures/river-en.md", "river-en", "v1", "distractor")],
        ),
        (
            "en",
            None,
            "What is the Synthetic Harbor Cooperative fresh water price?",
            [_doc("fixtures/river-en.md", "river-en", "v1", "distractor")],
        ),
        (
            "ar",
            "formal",
            "ما رسوم الرصيف أ اليومية في مؤسسة الواحة التجريبية؟",
            [_doc("fixtures/river-ar.md", "river-ar", "v1", "distractor")],
        ),
        (
            "ar",
            "egyptian",
            "قناة الطوارئ في مؤسسة الواحة التجريبية إيه؟",
            [_doc("fixtures/river-ar.md", "river-ar", "v1", "distractor")],
        ),
    ):
        add(
            family="absent",
            language=language,
            register=register,
            task="absent",
            question=question,
            documents=documents,
            gold=make_gold(expect_refusal=True),
            extra=["absent_with_distractor"],
            notes=NOTE
            + " The selected scope has a similar distractor, not the named source.",
        )
    _calculations(add)
    _coverage(add)
    _denied(add)
    _pending(add)
    return cases


def _calculations(add) -> None:
    def plan(function, column, filters=None):
        payload = {
            "action": "calculate",
            "aggregates": [
                {"function": function, "column": column, "output": "result"}
            ],
        }
        if filters:
            payload["filters"] = filters
        return payload

    exact = [
        (
            "en",
            None,
            "What is the sum of amount?",
            "fixtures/basic.csv",
            plan("sum", "amount"),
            "30.60",
        ),
        (
            "en",
            None,
            "What is the average of amount?",
            "fixtures/basic.csv",
            plan("avg", "amount"),
            "10.20",
        ),
        (
            "en",
            None,
            "What is the minimum amount?",
            "fixtures/basic.csv",
            plan("min", "amount"),
            "0.30",
        ),
        (
            "en",
            None,
            "What is the maximum amount?",
            "fixtures/basic.csv",
            plan("max", "amount"),
            "20.20",
        ),
        (
            "en",
            None,
            "What is the count of rows?",
            "fixtures/basic.csv",
            plan("count", None),
            "3",
        ),
        (
            "ar",
            "formal",
            "ما مجموع المبيعات؟",
            "fixtures/arabic-sales.csv",
            plan("sum", "المبيعات"),
            "15.50",
        ),
        (
            "mixed",
            "mixed_terminology",
            "ما مجموع amount للمنطقة North؟",
            "fixtures/regions.csv",
            plan(
                "sum",
                "amount",
                [{"column": "region", "operator": "eq", "value": "North"}],
            ),
            "5.00",
        ),
        (
            "en",
            None,
            "What is the sum of amount for North?",
            "fixtures/regions.csv",
            plan(
                "sum",
                "amount",
                [{"column": "region", "operator": "eq", "value": "North"}],
            ),
            "5.00",
        ),
        (
            "mixed",
            "mixed_terminology",
            "ما مجموع fee؟",
            "fixtures/sku-fees.csv",
            plan("sum", "fee"),
            "4.00",
        ),
        (
            "ar",
            "formal",
            "ما عدد الصفوف؟",
            "fixtures/arabic-sales.csv",
            plan("count", None),
            "2",
        ),
        (
            "ar",
            "formal",
            "ما مجموع القيمة؟",
            "fixtures/values.csv",
            plan("sum", "القيمة"),
            "100.03",
        ),
        (
            "ar",
            "egyptian",
            "أقل قيمة في المبيعات كام؟",
            "fixtures/arabic-sales.csv",
            plan("min", "المبيعات"),
            "5.50",
        ),
    ]
    for language, register, question, fixture, chosen, value in exact:
        add(
            family="calculation",
            language=language,
            register=register,
            task="calculation",
            question=question,
            documents=[_doc(fixture, fixture, "v1", "supporting")],
            gold=make_gold(
                numerical={"value": value, "tolerance": "exact"},
                plan=chosen,
            ),
            notes=NOTE
            + " Expected result is the exact decimal, including subtotal exclusion.",
        )
    clarifications = (
        ("en", None, "What is the margin?", "fixtures/basic.csv", None),
        ("en", None, "Forecast next quarter amount.", "fixtures/basic.csv", None),
        ("en", None, "Why did amount increase?", "fixtures/basic.csv", None),
        ("ar", "formal", "ما هامش الربح؟", "fixtures/basic.csv", None),
        ("ar", "formal", "ما التوقع للشهر القادم؟", "fixtures/arabic-sales.csv", None),
        (
            "en",
            None,
            "What is the sum of amount?",
            "fixtures/mixed-currency.csv",
            plan("sum", "amount"),
        ),
    )
    for language, register, question, fixture, chosen in clarifications:
        add(
            family="calculation",
            language=language,
            register=register,
            task="calculation",
            question=question,
            documents=[_doc(fixture, fixture, "v1", "supporting")],
            gold=make_gold(expect_clarification=True, plan=chosen),
            notes=NOTE
            + " Margin, forecast, cause, or mixed currency must not invent a number.",
        )


def _coverage(add) -> None:
    complete = [
        _doc("fixtures/coverage-complete.md", "coverage-complete", "v1", "supporting")
    ]
    gap = [_doc("fixtures/coverage-gap.md", "coverage-gap", "v1", "supporting")]
    ocr = [_doc("fixtures/coverage-ocr.md", "coverage-ocr", "v1", "supporting")]
    for language, register, question in (
        ("en", None, "Summarize every section of the synthetic harbor roster."),
        ("en", None, "List every roster section without skipping one."),
        ("ar", "formal", "لخص كل أقسام سجل الميناء التجريبي."),
        ("ar", "egyptian", "قول لي كل أجزاء السجل من غير ما تسيب قسم."),
    ):
        add(
            family="coverage",
            language=language,
            register=register,
            task="coverage",
            question=question,
            documents=complete,
            gold=make_gold(coverage={"expect_complete": True, "gap_section_ids": []}),
        )
    for language, register, question in (
        ("en", None, "Summarize every section, including the blank one."),
        ("ar", "formal", "لخص كل الأقسام بما فيها القسم الفارغ."),
    ):
        add(
            family="coverage",
            language=language,
            register=register,
            task="coverage",
            question=question,
            documents=gap,
            gold=make_gold(
                coverage={
                    "expect_complete": False,
                    "gap_section_ids": ["cov-gap-blank"],
                }
            ),
        )
    for language, register, question in (
        ("en", None, "Summarize every section, including the page that needs OCR."),
        ("en", None, "Is coverage complete when one section needs OCR?"),
        ("ar", "formal", "لخص كل الأقسام بما فيها الصفحة التي تحتاج تعرّفاً ضوئياً."),
        ("mixed", "mixed_terminology", "لخص الأقسام مع صفحة OCR الناقصة."),
    ):
        add(
            family="coverage",
            language=language,
            register=register,
            task="coverage",
            question=question,
            documents=ocr,
            gold=make_gold(
                coverage={"expect_complete": False, "gap_section_ids": ["cov-ocr-3"]}
            ),
            notes=NOTE
            + " An OCR-required section is a gap. The label must stay partial.",
        )
    # Two more gap cases so the family has a broader held-out share.
    for question in (
        "Does the blank shed section leave coverage partial?",
        "Report the south shed only if the blank section is disclosed as a gap.",
    ):
        add(
            family="coverage",
            language="en",
            task="coverage",
            question=question,
            documents=gap,
            gold=make_gold(
                coverage={
                    "expect_complete": False,
                    "gap_section_ids": ["cov-gap-blank"],
                }
            ),
        )


def _denied(add) -> None:
    document = [
        _doc("fixtures/denied-pin.md", "denied", "v1", "denied", actors=("member",))
    ]
    questions = (
        (
            "en",
            None,
            "outsider",
            "What is the restricted synthetic warehouse PIN example?",
        ),
        (
            "en",
            None,
            "outsider",
            "Repeat the warehouse PIN example from the restricted note.",
        ),
        (
            "en",
            None,
            "revoked",
            "What is the restricted synthetic warehouse PIN example?",
        ),
        ("en", None, "revoked", "Quote the restricted warehouse PIN example."),
        ("en", None, "outsider", "Show PIN-SYN-4401 from the restricted note."),
        (
            "en",
            None,
            "outsider",
            "Is the restricted warehouse PIN example available to me?",
        ),
        (
            "en",
            None,
            "revoked",
            "Give me the restricted synthetic warehouse PIN example.",
        ),
        (
            "en",
            None,
            "outsider",
            "Copy the restricted warehouse PIN example into the answer.",
        ),
        ("ar", "formal", "outsider", "ما مثال رقم المخزن المقيد؟"),
        ("ar", "formal", "revoked", "اذكر مثال الرقم المقيد للمخزن."),
        ("ar", "egyptian", "outsider", "وريني مثال الرقم السري بتاع المخزن."),
        (
            "mixed",
            "mixed_terminology",
            "outsider",
            "ما قيمة PIN-SYN-4401 في الملاحظة المقيدة؟",
        ),
    )
    for language, register, actor, question in questions:
        add(
            family="denied_access",
            language=language,
            register=register,
            task="denied_access",
            question=question,
            actor=actor,
            documents=document,
            gold=make_gold(expect_denial=True, forbidden_snippets=["PIN-SYN-4401"]),
            notes=NOTE + " PIN-SYN-4401 is a synthetic example, not a credential.",
        )


def _pending(add) -> None:
    ocr = (
        ("ar", "formal", "ما الرقم المكتوب في الصفحة العربية الممسوحة ضوئياً؟"),
        ("ar", "formal", "اقرأ الختم في أسفل الصفحة الممسوحة."),
        ("ar", "formal", "ما تاريخ الصفحة الممسوحة في محضر الواحة التجريبي؟"),
        ("ar", "egyptian", "مكتوب إيه في الورقة المتصورة؟"),
        ("ar", "egyptian", "الرقم اللي في سكان الصفحة كام؟"),
        ("mixed", "mixed_terminology", "ما قيمة SKU الظاهرة في الصفحة الممسوحة؟"),
        ("ar", "arabic_over_english", "ما النص الإنجليزي الظاهر في الصورة الممسوحة؟"),
        (
            "mixed",
            "mixed_terminology",
            "اقرأ الجدول الممسوح الذي يخلط العربية والإنجليزية.",
        ),
        ("en", None, "What number is printed on the scanned harbor page?"),
        ("en", None, "Transcribe the scanned stamp on the synthetic receipt."),
        ("en", None, "What does the scanned table in the corner say?"),
        ("en", None, "Read the handwritten total on the scanned page."),
        ("en", None, "What is the scanned page's footer code?"),
        ("en", None, "Which berth is named on the unscanned image?"),
    )
    for language, register, question in ocr:
        add(
            family="ocr_pending",
            language=language,
            register=register,
            task="ocr_pending",
            question=question,
            documents=[],
            gold=make_gold(ocr_status="pending_not_implemented"),
            notes=NOTE
            + " No scan is stored. OCR is not implemented. Do not treat a transcription as verified.",
        )
    charts = (
        (
            "en",
            None,
            "Which quarter has the tallest bar in the synthetic pallet chart?",
        ),
        ("en", None, "What is the visual peak of the synthetic fee chart?"),
        ("en", None, "Does the synthetic line chart cross zero?"),
        ("en", None, "Which slice of the synthetic pie chart is largest?"),
        ("ar", "formal", "ما أعلى عمود في الرسم البياني التجريبي؟"),
        ("ar", "egyptian", "الرسم البياني أعلاها أنهي عمود؟"),
    )
    for language, register, question in charts:
        add(
            family="chart_pending",
            language=language,
            register=register,
            task="chart_pending",
            question=question,
            documents=[],
            gold=make_gold(chart_status="pending_not_implemented"),
            notes=NOTE
            + " No chart image is stored. Visual interpretation is not implemented.",
        )


def _doc(fixture, document_id, version, role, actors=("member",)):
    return {
        "fixture": fixture,
        "document_id": document_id,
        "version_label": version,
        "role": role,
        "authorized_for": list(actors),
    }
