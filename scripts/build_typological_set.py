#!/usr/bin/env python3
"""
Build typological ambiguity evaluation dataset (~210 rows).

Generates deterministic test cases across 14 typological & cross-lingual categories.
Includes ~20% control rows (unambiguous ANSWER controls).

AGENTS.md rule 6: Every dataset row carries source, source_id, annotation_provenance.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

_OUT_PATH = Path("eval/datasets/typological_ambiguity.jsonl")
_SEED = 42

CATEGORIES = [
    "entity_collision",
    "currency",
    "date_format",
    "numeric_scale",
    "measurement",
    "formality",
    "subject_drop",
    "gender",
    "number_ambiguity",
    "honorific",
    "script_variant",
    "calendar",
    "code_switching",
    "word_order",
]


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = 1

    def add_row(
        question: str,
        context: str,
        expected_behaviour: str,
        category: str,
        ambiguity_present_in: str,
        ambiguity_absent_in: str,
        resolves_with: str,
        note: str,
    ) -> None:
        nonlocal idx
        rows.append({
            "id": f"typo_{idx:03d}",
            "question": question,
            "context": context,
            "expected_behaviour": expected_behaviour,
            "category": category,
            "ambiguity_present_in": ambiguity_present_in,
            "ambiguity_absent_in": ambiguity_absent_in,
            "resolves_with": resolves_with,
            "annotation_provenance": "typological_seed_v1",
            "source": "typological_benchmark",
            "source_id": f"src_typo_{idx:03d}",
            "note": note,
        })
        idx += 1

    # 1. entity_collision (15 rows: 12 AMBIGUOUS, 3 ANSWER controls)
    toponyms = [
        ("Santiago", "Chile", "Spain (de Compostela)"),
        ("Georgia", "US State", "Country in Caucasus"),
        ("Cordoba", "Spain", "Argentina"),
        ("Tripoli", "Libya", "Lebanon"),
        ("Valencia", "Spain", "Venezuela"),
        ("Boston", "USA", "UK"),
        ("Cambridge", "UK", "USA (MA)"),
        ("San Jose", "California", "Costa Rica"),
        ("Guadalajara", "Mexico", "Spain"),
        ("Perth", "Australia", "Scotland"),
        ("Hamilton", "Canada", "New Zealand"),
        ("Victoria", "Australia State", "Seychelles Capital"),
    ]
    for top, loc1, loc2 in toponyms:
        add_row(
            question=f"What is the population of {top}?",
            context=f"Query refers to {top}, which could mean {top} in {loc1} or {top} in {loc2}.",
            expected_behaviour="AMBIGUOUS",
            category="entity_collision",
            ambiguity_present_in="Global queries without locale qualifiers",
            ambiguity_absent_in=f"Queries explicitly specifying {loc1}",
            resolves_with="Target location or admin division",
            note=f"Toponymic collision between {loc1} and {loc2}",
        )
    # Controls for entity_collision (3 rows)
    add_row(
        question="What is the population of Santiago, Chile?",
        context="Santiago is the capital and largest city of Chile.",
        expected_behaviour="ANSWER",
        category="entity_collision",
        ambiguity_present_in="None",
        ambiguity_absent_in="Fully qualified locale query",
        resolves_with="N/A - unambiguous",
        note="Control: fully qualified toponym",
    )
    add_row(
        question="What is the capital of Georgia the country?",
        context="Tbilisi is the capital of Georgia, a country in the Caucasus region.",
        expected_behaviour="ANSWER",
        category="entity_collision",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit entity type specified",
        resolves_with="N/A - unambiguous",
        note="Control: explicit entity type specified",
    )
    add_row(
        question="What is the population of Cambridge, Massachusetts?",
        context="Cambridge is a city in Middlesex County, Massachusetts, United States.",
        expected_behaviour="ANSWER",
        category="entity_collision",
        ambiguity_present_in="None",
        ambiguity_absent_in="Fully qualified state and city",
        resolves_with="N/A - unambiguous",
        note="Control: fully qualified city and state",
    )

    # 2. currency (15 rows: 12 AMBIGUOUS, 3 ANSWER controls)
    currencies = [
        ("$", "USD", "CAD", "50,000"), ("$", "USD", "AUD", "75,000"), ("$", "USD", "MXN", "100,000"),
        ("¥", "JPY", "CNY", "5,000,000"), ("kr", "SEK", "NOK", "400,000"), ("£", "GBP", "EGP", "35,000"),
        ("Rs", "INR", "PKR", "1,200,000"), ("$", "USD", "NZD", "65,000"), ("$", "USD", "HKD", "500,000"),
        ("R$", "BRL", "ZAR (R)", "80,000"), ("din.", "RSD", "KWD", "15,000"), ("£", "GBP", "LBP", "45,000"),
    ]
    for sym, c1, c2, amt in currencies:
        add_row(
            question=f"What is a salary of {sym}{amt} per year equivalent to in purchasing power?",
            context=f"The symbol {sym} can denote {c1} or {c2}.",
            expected_behaviour="AMBIGUOUS",
            category="currency",
            ambiguity_present_in=f"Unqualified currency symbol {sym}",
            ambiguity_absent_in=f"ISO 4217 code {c1}",
            resolves_with="ISO currency code or country context",
            note=f"Currency symbol collision {sym} ({c1} vs {c2})",
        )
    # Controls (3 rows)
    add_row(
        question="What is 50,000 USD in EUR today?",
        context="50,000 United States Dollars converted to Euros.",
        expected_behaviour="ANSWER",
        category="currency",
        ambiguity_present_in="None",
        ambiguity_absent_in="ISO 4217 currency codes used",
        resolves_with="N/A - unambiguous",
        note="Control: ISO currency codes",
    )
    add_row(
        question="What is the exchange rate of 10,000 JPY to USD?",
        context="Japanese Yen (JPY) to US Dollars (USD).",
        expected_behaviour="ANSWER",
        category="currency",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit currency codes",
        resolves_with="N/A - unambiguous",
        note="Control: explicit currency codes",
    )
    add_row(
        question="How much is 100 GBP in Canadian Dollars (CAD)?",
        context="British Pound Sterling to Canadian Dollars.",
        expected_behaviour="ANSWER",
        category="currency",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit currency names and codes",
        resolves_with="N/A - unambiguous",
        note="Control: explicit currency names",
    )

    # 3. date_format (15 rows: 12 AMBIGUOUS, 3 ANSWER controls)
    dates = [
        "05/06/2024", "03/04/2025", "11/12/2023", "01/02/2026", "07/08/2024",
        "09/10/2025", "02/03/2024", "04/05/2026", "06/07/2025", "08/09/2024",
        "10/11/2026", "12/01/2025",
    ]
    for d in dates:
        add_row(
            question=f"When does the contract starting on {d} expire?",
            context=f"The date {d} can be interpreted as DD/MM/YYYY or MM/DD/YYYY.",
            expected_behaviour="AMBIGUOUS",
            category="date_format",
            ambiguity_present_in="Numeric date formats with day <= 12",
            ambiguity_absent_in="ISO 8601 YYYY-MM-DD or spelled-out month",
            resolves_with="Date format convention (US vs EU)",
            note=f"Numeric date ambiguity {d}",
        )
    # Controls (3 rows)
    add_row(
        question="When does the contract starting on 2024-05-06 expire?",
        context="Contract starts on May 6, 2024 (ISO 8601 format YYYY-MM-DD).",
        expected_behaviour="ANSWER",
        category="date_format",
        ambiguity_present_in="None",
        ambiguity_absent_in="ISO 8601 format YYYY-MM-DD",
        resolves_with="N/A - unambiguous",
        note="Control: ISO 8601 date",
    )
    add_row(
        question="What day of the week was 25/12/2023?",
        context="December 25, 2023 (day 25 exceeds 12, unambiguous DD/MM/YYYY).",
        expected_behaviour="ANSWER",
        category="date_format",
        ambiguity_present_in="None",
        ambiguity_absent_in="Day > 12 forces DD/MM format",
        resolves_with="N/A - unambiguous",
        note="Control: day > 12 removes ambiguity",
    )
    add_row(
        question="What event happened on June 5, 2024?",
        context="June 5, 2024 with month spelled out in English.",
        expected_behaviour="ANSWER",
        category="date_format",
        ambiguity_present_in="None",
        ambiguity_absent_in="Spelled-out month name",
        resolves_with="N/A - unambiguous",
        note="Control: spelled out month",
    )

    # 4. numeric_scale (15 rows: 12 AMBIGUOUS, 3 ANSWER controls)
    scales = [
        ("1 billion", "10^9 (short scale)", "10^12 (long scale)"),
        ("1 trillion", "10^12 (short scale)", "10^18 (long scale)"),
        ("1 billion euros in 1990 text", "US 10^9", "European 10^12"),
        ("1.000", "One thousand (EU dot separator)", "One point zero (US decimal)"),
        ("10,000", "Ten thousand (US comma)", "Ten point zero (EU decimal)"),
        ("2 billion", "2 x 10^9", "2 x 10^12"),
        ("5 trillion", "5 x 10^12", "5 x 10^18"),
        ("1,500", "1500", "1.5 in some locales"),
        ("100 billion", "10^11", "10^14"),
        ("3.500", "3500", "3.5"),
        ("4,000", "4000", "4.0"),
        ("10 billion", "10^10", "10^13"),
    ]
    for fig, s1, s2 in scales:
        add_row(
            question=f"What is the exact numerical value of {fig}?",
            context=f"The figure {fig} varies between {s1} and {s2} depending on locale convention.",
            expected_behaviour="AMBIGUOUS",
            category="numeric_scale",
            ambiguity_present_in="Short scale vs long scale / separator ambiguity",
            ambiguity_absent_in="Scientific notation or explicit power of 10",
            resolves_with="Locale numeric convention",
            note=f"Numeric scale/separator ambiguity for {fig}",
        )
    # Controls (3 rows)
    add_row(
        question="What is 1 x 10^9 in standard scientific notation?",
        context="1 x 10^9 equals 1,000,000,000.",
        expected_behaviour="ANSWER",
        category="numeric_scale",
        ambiguity_present_in="None",
        ambiguity_absent_in="Scientific notation (10^9)",
        resolves_with="N/A - unambiguous",
        note="Control: scientific notation",
    )
    add_row(
        question="How many millions are in 1,000,000,000 (one US billion)?",
        context="One US billion (1,000,000,000) contains 1,000 million.",
        expected_behaviour="ANSWER",
        category="numeric_scale",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit qualification (one US billion)",
        resolves_with="N/A - unambiguous",
        note="Control: qualified scale name",
    )
    add_row(
        question="What is the value of 500,000 in word form?",
        context="Five hundred thousand.",
        expected_behaviour="ANSWER",
        category="numeric_scale",
        ambiguity_present_in="None",
        ambiguity_absent_in="Unambiguous integer representation",
        resolves_with="N/A - unambiguous",
        note="Control: unambiguous integer",
    )

    # 5. measurement (15 rows: 12 AMBIGUOUS, 3 ANSWER controls)
    measures = [
        ("75 degrees", "Celsius", "Fahrenheit"),
        ("100 miles", "Statute miles", "Nautical miles"),
        ("50 tons", "Short tons (US)", "Metric tons / Long tons"),
        ("10 gallons", "US gallons", "Imperial gallons"),
        ("5 pints", "US liquid pints", "Imperial pints"),
        ("100 feet", "US survey feet", "International feet"),
        ("20 knots", "Nautical miles/hr", "Wind speed scale"),
        ("60 degrees", "Celsius", "Fahrenheit"),
        ("200 pounds", "Pounds force", "Pounds mass"),
        ("5 ounces", "Fluid ounces", "Weight ounces"),
        ("2 miles", "Statute miles", "Nautical miles"),
        ("30 degrees", "Celsius", "Fahrenheit"),
    ]
    for val, u1, u2 in measures:
        add_row(
            question=f"Is {val} considered high for this system?",
            context=f"The value {val} depends on whether unit is {u1} or {u2}.",
            expected_behaviour="AMBIGUOUS",
            category="measurement",
            ambiguity_present_in="Unspecified unit variant",
            ambiguity_absent_in="Fully qualified SI/Imperial unit name",
            resolves_with="Explicit measurement unit",
            note=f"Measurement unit ambiguity ({u1} vs {u2})",
        )
    # Controls (3 rows)
    add_row(
        question="What is 75 degrees Celsius converted to Fahrenheit?",
        context="75°C equals 167°F.",
        expected_behaviour="ANSWER",
        category="measurement",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit units specified (°C and °F)",
        resolves_with="N/A - unambiguous",
        note="Control: explicit unit conversion",
    )
    add_row(
        question="How many kilometers is 100 statute miles?",
        context="100 statute miles equals 160.934 kilometers.",
        expected_behaviour="ANSWER",
        category="measurement",
        ambiguity_present_in="None",
        ambiguity_absent_in="Fully qualified unit (statute miles)",
        resolves_with="N/A - unambiguous",
        note="Control: fully qualified unit",
    )
    add_row(
        question="What is 50 metric tons in kilograms?",
        context="1 metric ton equals 1,000 kilograms; 50 metric tons = 50,000 kg.",
        expected_behaviour="ANSWER",
        category="measurement",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit metric unit",
        resolves_with="N/A - unambiguous",
        note="Control: metric unit",
    )

    # 6. formality (15 rows: 12 AMBIGUOUS, 3 ANSWER controls)
    formalities = [
        ("German", "wie geht es dir (informal) vs wie geht es Ihnen (formal)"),
        ("Spanish", "cómo estás (tú) vs cómo está usted"),
        ("French", "comment vas-tu (tu) vs comment allez-vous (vous)"),
        ("Japanese", "desu/masu (keigo) vs plain form (dawa/da)"),
        ("Russian", "kak dela (ty) vs kak vashy dela (vy)"),
        ("Italian", "come stai (tu) vs come sta (Lei)"),
        ("Portuguese", "como você está vs como o senhor está"),
        ("Korean", "banmal vs jondetmal"),
        ("Hindi", "tum kaise ho vs aap kaise hain"),
        ("Dutch", "hoe gaat het met jou (je) vs u"),
        ("Polish", "co u ciebie vs co u Pana/Pani"),
        ("Turkish", "nasılsın (sen) vs nasılsınız (siz)"),
    ]
    for lang, detail in formalities:
        add_row(
            question=f"How do I translate 'How are you?' into {lang} for an email?",
            context=f"Translation into {lang} depends on social distance: {detail}.",
            expected_behaviour="AMBIGUOUS",
            category="formality",
            ambiguity_present_in="Target language with T-V distinction without recipient relationship",
            ambiguity_absent_in="Query specifying relationship (e.g. to CEO vs close colleague)",
            resolves_with="Recipient relationship / formality register",
            note=f"T-V register ambiguity in {lang}",
        )
    # Controls (3 rows)
    add_row(
        question="How do I formally address a German corporate executive in an official letter?",
        context="Use formal German 'Sehr geehrte Damen und Herren' or 'Sehr geehrter Herr [Name]' with 'Ihnen/Sie'.",
        expected_behaviour="ANSWER",
        category="formality",
        ambiguity_present_in="None",
        ambiguity_absent_in="Relationship explicitly stated (formal corporate executive)",
        resolves_with="N/A - unambiguous",
        note="Control: formal register context specified",
    )
    add_row(
        question="What is the informal Spanish translation of 'See you later' to a close friend?",
        context="Informal Spanish for a close friend: 'Hasta luego' or 'Nos vemos'.",
        expected_behaviour="ANSWER",
        category="formality",
        ambiguity_present_in="None",
        ambiguity_absent_in="Informal register specified for close friend",
        resolves_with="N/A - unambiguous",
        note="Control: informal register context specified",
    )
    add_row(
        question="What is the polite Keigo form of 'to eat' (taberu) in Japanese business interactions?",
        context="In Japanese business contexts, polite/honorific forms are meshiagaru (honorific) or itadaku (humble).",
        expected_behaviour="ANSWER",
        category="formality",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit Keigo business context",
        resolves_with="N/A - unambiguous",
        note="Control: explicit Keigo business context",
    )

    # 7. subject_drop (15 rows: 12 AMBIGUOUS, 3 ANSWER controls)
    pro_drops = [
        ("Japanese", "tabeta?", "Did I eat? / Did you eat? / Did he/she eat?"),
        ("Spanish", "llegó ayer", "He arrived yesterday / She arrived yesterday / It arrived yesterday"),
        ("Italian", "ha chiamato", "He called / She called / You (formal) called"),
        ("Chinese", "qu le", "I went / You went / He went"),
        ("Portuguese", "comprou o carro", "He bought the car / She bought the car"),
        ("Korean", "gasseoyo?", "Did I go? / Did you go? / Did he go?"),
        ("Arabic", "dharaba", "He hit / (subject pronoun omitted in context)"),
        ("Hindi", "gaya tha", "He went / I (male) went"),
        ("Russian", "poshel", "He went (masculine subject dropped in fragment)"),
        ("Polish", "poszedł", "He went (subject pronoun dropped)"),
        ("Greek", "efyge", "He left / She left / It left"),
        ("Turkish", "geldi", "He came / She came / It came"),
    ]
    for lang, phrase, detail in pro_drops:
        add_row(
            question=f"Who is referred to in the {lang} sentence '{phrase}'?",
            context=f"In {lang}, pro-drop verb '{phrase}' omits the explicit subject: {detail}.",
            expected_behaviour="AMBIGUOUS",
            category="subject_drop",
            ambiguity_present_in="Pro-drop sentence fragment without prior discourse context",
            ambiguity_absent_in="Full sentence with explicit subject pronoun",
            resolves_with="Prior discourse antecedent or explicit pronoun",
            note=f"Pro-drop subject ambiguity in {lang} ({phrase})",
        )
    # Controls (3 rows)
    add_row(
        question="Translate 'Ella llegó ayer' from Spanish to English.",
        context="Explicit subject pronoun 'Ella' means 'She arrived yesterday'.",
        expected_behaviour="ANSWER",
        category="subject_drop",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit subject pronoun (Ella)",
        resolves_with="N/A - unambiguous",
        note="Control: explicit subject pronoun in Spanish",
    )
    add_row(
        question="In the Japanese sentence 'Watashi wa tabeta', who ate?",
        context="'Watashi wa' explicitly identifies the subject as 'I'.",
        expected_behaviour="ANSWER",
        category="subject_drop",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit topic marker + pronoun (Watashi wa)",
        resolves_with="N/A - unambiguous",
        note="Control: explicit topic pronoun in Japanese",
    )
    add_row(
        question="Translate 'Lui ha chiamato' from Italian to English.",
        context="Explicit pronoun 'Lui' specifies masculine subject 'He called'.",
        expected_behaviour="ANSWER",
        category="subject_drop",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit pronoun (Lui)",
        resolves_with="N/A - unambiguous",
        note="Control: explicit pronoun in Italian",
    )

    # 8. gender (15 rows: 12 AMBIGUOUS, 3 ANSWER controls)
    genders = [
        ("French", "the doctor", "le médecin vs la médecin / docteure"),
        ("German", "the teacher", "der Lehrer vs die Lehrerin"),
        ("Spanish", "the engineer", "el ingeniero vs la ingeniera"),
        ("Italian", "the lawyer", "l'avvocato vs l'avvocatessa"),
        ("Russian", "the doctor", "vrach (masculine grammar for female referent)"),
        ("Arabic", "the professor", "al-ustadh vs al-ustadha"),
        ("Portuguese", "the judge", "o juiz vs a juíza"),
        ("Hindi", "the doctor", "doctor (masculine vs feminine verb agreement)"),
        ("Dutch", "the student", "de student vs de studente"),
        ("Polish", "the architect", "architekt vs pani architekt"),
        ("Hebrew", "the manager", "ha-menahel vs ha-menahelet"),
        ("Swedish", "the nurse", "sjuksköterska (historically gendered)"),
    ]
    for lang, noun, detail in genders:
        add_row(
            question=f"How do I translate '{noun}' into {lang} when referring to a professional?",
            context=f"Translation of '{noun}' into {lang} requires grammatical gender selection: {detail}.",
            expected_behaviour="AMBIGUOUS",
            category="gender",
            ambiguity_present_in="Gender-neutral English occupational noun translated to gendered language",
            ambiguity_absent_in="Query specifying female doctor or male teacher",
            resolves_with="Biological/social gender of the referent",
            note=f"Occupational gender assignment in {lang} for {noun}",
        )
    # Controls (3 rows)
    add_row(
        question="How do I translate 'female teacher' into German?",
        context="'Female teacher' in German is 'die Lehrerin'.",
        expected_behaviour="ANSWER",
        category="gender",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit female modifier in English source",
        resolves_with="N/A - unambiguous",
        note="Control: explicit gender modifier in source",
    )
    add_row(
        question="What is the Spanish translation for 'male engineer'?",
        context="'Male engineer' in Spanish is 'el ingeniero'.",
        expected_behaviour="ANSWER",
        category="gender",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit male modifier in source",
        resolves_with="N/A - unambiguous",
        note="Control: explicit male modifier in source",
    )
    add_row(
        question="Translate 'la professora' from French to English.",
        context="'La professeure' translates to 'the female professor'.",
        expected_behaviour="ANSWER",
        category="gender",
        ambiguity_present_in="None",
        ambiguity_absent_in="Gendered feminine article in target sentence",
        resolves_with="N/A - unambiguous",
        note="Control: gendered French article",
    )

    # 9. number_ambiguity (15 rows: 12 AMBIGUOUS, 3 ANSWER controls)
    numbers = [
        ("English 'you'", "singular (you alone) vs plural (you all / team)"),
        ("French 'vous'", "singular polite vs plural group"),
        ("German 'Sie'", "singular formal vs plural formal"),
        ("Spanish 'ustedes'", "plural addressees (Spain vs Latin America)"),
        ("Arabic dual vs plural", "two people (dual) vs three+ people (plural)"),
        ("Hebrew 'atem'", "masculine plural vs mixed gender plural"),
        ("Russian 'vy'", "singular polite vs plural"),
        ("Greek 'eseis'", "singular polite vs plural"),
        ("Tagalog 'tayo' vs 'kami'", "inclusive we vs exclusive we"),
        ("Tok Pisin 'mitupela' vs 'mipla'", "dual vs general plural"),
        ("Hindi 'aap'", "singular formal vs plural formal"),
        ("Japanese 'anata-tachi'", "you plural vs group including you"),
    ]
    for term, detail in numbers:
        add_row(
            question=f"Does the phrase containing '{term}' address one person or a group?",
            context=f"Number ambiguity for {term}: {detail}.",
            expected_behaviour="AMBIGUOUS",
            category="number_ambiguity",
            ambiguity_present_in="Pronominal address without addressee count in context",
            ambiguity_absent_in="Explicit addressee count (e.g. 'both of you' or 'you three')",
            resolves_with="Number of addressees in dialogue context",
            note=f"Grammatical number ambiguity in {term}",
        )
    # Controls (3 rows)
    add_row(
        question="Does 'both of you' refer to two people or three people?",
        context="'Both of you' explicitly refers to exactly two people.",
        expected_behaviour="ANSWER",
        category="number_ambiguity",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit dual quantifier 'both'",
        resolves_with="N/A - unambiguous",
        note="Control: explicit dual quantifier",
    )
    add_row(
        question="Translate 'all three of you' into French.",
        context="Refers explicitly to a group of three: 'tous les trois'.",
        expected_behaviour="ANSWER",
        category="number_ambiguity",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit numeral modifier ('all three')",
        resolves_with="N/A - unambiguous",
        note="Control: explicit numeral modifier",
    )
    add_row(
        question="In Tagalog, does 'tayo' include the listener?",
        context="Tagalog 'tayo' is the inclusive first-person plural (includes listener).",
        expected_behaviour="ANSWER",
        category="number_ambiguity",
        ambiguity_present_in="None",
        ambiguity_absent_in="Linguistic definition query",
        resolves_with="N/A - unambiguous",
        note="Control: linguistic definition query",
    )

    # 10. honorific (15 rows: 12 AMBIGUOUS, 3 ANSWER controls)
    honorifics = [
        ("Japanese", "Tanaka-san", "Mr. Tanaka / Ms. Tanaka / Dr. Tanaka"),
        ("Japanese", "Tanaka-sama", "Customer Tanaka / Honored Tanaka"),
        ("Korean", "Kim-seonsaengnim", "Teacher Kim / Doctor Kim / Professor Kim"),
        ("Korean", "Kim-사장님 (sajangnim)", "President Kim / Company Director Kim"),
        ("Thai", "Khun Somchai", "Mr. Somchai / Ms. Somchai"),
        ("Vietnamese", "Anh Binh", "Older brother Binh / Male peer Binh"),
        ("Javanese", "Bapak", "Father / Respected older male / Official"),
        ("Hindi", "Ji suffix", "Respect marker applicable to family or superiors"),
        ("Burmese", "U Thant", "Uncle Thant / Respected elder Thant"),
        ("Malay", "Encik Ahmad", "Mr. Ahmad / Officer Ahmad"),
        ("Turkish", "Ahmet Bey", "Mr. Ahmet / Sir Ahmet"),
        ("Arabic", "Sheikh", "Tribal leader / Religious scholar / Wealthy elder"),
    ]
    for lang, honorific, detail in honorifics:
        add_row(
            question=f"What is the exact professional title of '{honorific}' in {lang}?",
            context=f"The honorific '{honorific}' in {lang} conflates social roles: {detail}.",
            expected_behaviour="AMBIGUOUS",
            category="honorific",
            ambiguity_present_in="Honorific suffix without specific occupational context",
            ambiguity_absent_in="Explicit professional title (e.g. Chief Medical Officer Tanaka)",
            resolves_with="Organizational role or specific professional title",
            note=f"Honorific title ambiguity in {lang} ({honorific})",
        )
    # Controls (3 rows)
    add_row(
        question="What does the Japanese honorific suffix '-sensei' mean when addressing a doctor?",
        context="In a medical context, '-sensei' addresses the doctor as 'Doctor/Teacher'.",
        expected_behaviour="ANSWER",
        category="honorific",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit medical context given",
        resolves_with="N/A - unambiguous",
        note="Control: honorific with explicit domain context",
    )
    add_row(
        question="Translate 'President Kim (Sajangnim)' into English corporate context.",
        context="Refers to the company president / CEO.",
        expected_behaviour="ANSWER",
        category="honorific",
        ambiguity_present_in="None",
        ambiguity_absent_in="Corporate context specified",
        resolves_with="N/A - unambiguous",
        note="Control: corporate context specified",
    )
    add_row(
        question="What is the meaning of 'Dr. Tanaka, MD'?",
        context="'MD' explicitly clarifies Dr. Tanaka as a Doctor of Medicine.",
        expected_behaviour="ANSWER",
        category="honorific",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit post-nominal credentials (MD)",
        resolves_with="N/A - unambiguous",
        note="Control: explicit post-nominal credentials",
    )

    # 11. script_variant (15 rows: 12 AMBIGUOUS, 3 ANSWER controls)
    scripts = [
        ("Traditional vs Simplified Chinese", "发 (fā: issue / fà: hair)", "zh-Hant 發 vs 髮"),
        ("Traditional vs Simplified Chinese", "干 (gān: dry / gàn: do / gān: shield)", "zh-Hant 乾 vs 幹 vs 干"),
        ("Arabic dialect spelling", "gaha (coffee)", "قهوة (standard) vs قهوه / كافيه"),
        ("Serbian script", "Policija", "Serbian Latin vs Cyrillic (Полиција)"),
        ("Kurdish script", "Kurdish Hawar vs Sorani Arabic script", "Northern vs Central Kurdish"),
        ("Uzbek script", "Latin vs Cyrillic vs Perso-Arabic", "Uzbek orthography transition"),
        ("Mongolian script", "Cyrillic vs Traditional Bichig", "Mongol script variants"),
        ("Japanese script", "Kanjani", "Kanji homophones with identical Hiragana"),
        ("Hebrew script", "Ktiv male vs Ktiv hasser", "Vowel-spelled vs unvoweled Hebrew"),
        ("Greek script", "Polytonic vs Monotonic", "Ancient breathings vs modern accent"),
        ("Persian script", "Ye / Ke variants", "Persian vs Arabic Unicode characters"),
        ("Belorussian script", "Taraškievica vs Official orthography", "Classic vs Soviet Belarusian"),
    ]
    for desc, term, detail in scripts:
        add_row(
            question=f"What is the single unambiguous meaning of the script form '{term}'?",
            context=f"Script variant {desc} merges distinct terms: {detail}.",
            expected_behaviour="AMBIGUOUS",
            category="script_variant",
            ambiguity_present_in="Unvoweled or character-merged orthographic variants",
            ambiguity_absent_in="Fully specified Traditional Chinese / polytonic text",
            resolves_with="Disambiguating character variant or vocalization",
            note=f"Orthographic script ambiguity for {term}",
        )
    # Controls (3 rows)
    add_row(
        question="What is the English translation of the Traditional Chinese character '髮' (fà)?",
        context="'髮' specifically means 'hair' in Traditional Chinese.",
        expected_behaviour="ANSWER",
        category="script_variant",
        ambiguity_present_in="None",
        ambiguity_absent_in="Unambiguous Traditional Chinese character (髮)",
        resolves_with="N/A - unambiguous",
        note="Control: unambiguous Traditional Chinese character",
    )
    add_row(
        question="Translate Serbian Cyrillic 'Полиција' to English.",
        context="'Полиција' in Serbian Cyrillic means 'Police'.",
        expected_behaviour="ANSWER",
        category="script_variant",
        ambiguity_present_in="None",
        ambiguity_absent_in="Explicit Cyrillic orthography",
        resolves_with="N/A - unambiguous",
        note="Control: Serbian Cyrillic police",
    )
    add_row(
        question="What does the Pinyin 'fà' with third tone mark represent in '頭髮'?",
        context="In 'tóufa', it refers specifically to hair on the head.",
        expected_behaviour="ANSWER",
        category="script_variant",
        ambiguity_present_in="None",
        ambiguity_absent_in="Compound word context (tóufa)",
        resolves_with="N/A - unambiguous",
        note="Control: compound word context",
    )

    # 12. calendar (15 rows: 12 AMBIGUOUS, 3 ANSWER controls)
    calendars = [
        ("1 Muharram 1445", "Hijri calendar", "Gregorian 19 July 2023"),
        ("Year 5784", "Hebrew calendar", "Gregorian 2023–2024"),
        ("Reiwa 5", "Japanese imperial era", "Gregorian 2023"),
        ("BE 2567", "Thai Solar Calendar (Buddhist Era)", "Gregorian 2024"),
        ("ROC Year 113", "Minguo calendar (Taiwan)", "Gregorian 2024"),
        ("Juche 113", "North Korean Juche calendar", "Gregorian 2024"),
        ("1403 AP", "Solar Hijri calendar (Iran/Afghanistan)", "Gregorian 2024"),
        ("Ethiopian 2016", "Ethiopian calendar", "Gregorian 2023–2024"),
        ("Julian 15 March", "Old Style Julian date", "Gregorian 28 March"),
        ("Coptic 1740", "Coptic calendar", "Gregorian 2023–2024"),
        ("Year of the Dragon 2024", "Lunar New Year start date", "Feb 10, 2024 vs Jan 1, 2024"),
        ("Financial Year 2023-24", "US Oct-Sep vs UK Apr-Mar vs India Apr-Mar", "Fiscal calendar start"),
    ]
    for cal_date, sys_name, detail in calendars:
        add_row(
            question=f"Which exact Gregorian day corresponds to '{cal_date}'?",
            context=f"The date '{cal_date}' uses {sys_name}: {detail}.",
            expected_behaviour="AMBIGUOUS",
            category="calendar",
            ambiguity_present_in="Non-Gregorian era or fiscal year without reference system",
            ambiguity_absent_in="Explicit Gregorian date with era specified",
            resolves_with="Calendar system reference or Gregorian conversion formula",
            note=f"Calendar era ambiguity ({sys_name})",
        )
    # Controls (3 rows)
    add_row(
        question="What Gregorian year corresponds to Reiwa 1 in Japan?",
        context="Reiwa 1 began on May 1, 2019 in the Gregorian calendar.",
        expected_behaviour="ANSWER",
        category="calendar",
        ambiguity_present_in="None",
        ambiguity_absent_in="Deterministic era start date defined in standard history",
        resolves_with="N/A - unambiguous",
        note="Control: deterministic era start",
    )
    add_row(
        question="Convert January 1, 2024 Gregorian to ISO 8601.",
        context="2024-01-01.",
        expected_behaviour="ANSWER",
        category="calendar",
        ambiguity_present_in="None",
        ambiguity_absent_in="Standard Gregorian reference",
        resolves_with="N/A - unambiguous",
        note="Control: standard Gregorian ISO date",
    )
    add_row(
        question="What is the UK tax year start date?",
        context="The UK personal tax year deterministically starts on April 6 every year.",
        expected_behaviour="ANSWER",
        category="calendar",
        ambiguity_present_in="None",
        ambiguity_absent_in="Statutory fixed calendar start",
        resolves_with="N/A - unambiguous",
        note="Control: statutory fixed calendar start",
    )

    # 13. code_switching (15 rows: 12 AMBIGUOUS, 3 ANSWER controls)
    code_switches = [
        ("Hinglish", "Yeh bill ka setting change kar do", "setting = configuration vs agreement/bribe"),
        ("Spanglish", "Vamos a la marketa", "marketa = supermarket vs stock market"),
        ("Arabizi", "shof 3ala al-link", "3ala = on vs 3ala = high/elevated in context"),
        ("Taglish", "Paki-make sure ang load", "load = phone credits vs electrical payload"),
        ("Franglais", "Je vais checker le mail", "mail = physical letter vs email"),
        ("Denglish", "Wir müssen das Handover machen", "handover = document vs shift transfer"),
        ("Singlish", "Can or not?", "can = permission vs physical container"),
        ("Portuñol", "La pasta está lista", "pasta = financial money vs Italian pasta"),
        ("Runglish", "Nuжно сделать resubmission", "resubmission = visa vs academic paper"),
        ("Czenglish", "Mám problem s kontrolou", "kontrola = inspection vs control switch"),
        ("Spanglish", "Llámame al cel", "cel = mobile phone vs cell block"),
        ("Hinglish", "Meeting ko freeze karo", "freeze = finalize vs pause/hold"),
    ]
    for lang_pair, phrase, detail in code_switches:
        add_row(
            question=f"What is the exact intended action in the {lang_pair} phrase '{phrase}'?",
            context=f"Code-switched expression '{phrase}' carries loanword polysemy: {detail}.",
            expected_behaviour="AMBIGUOUS",
            category="code_switching",
            ambiguity_present_in="Code-switched loanword with multiple semantic domains",
            ambiguity_absent_in="Monolingual sentence with domain-specific terms",
            resolves_with="Target technical domain or monolingual clarification",
            note=f"Code-switched loanword polysemy in {lang_pair}",
        )
    # Controls (3 rows)
    add_row(
        question="In Hinglish IT support, what does 'password reset' mean?",
        context="Password reset unambiguously refers to resetting a user's login password.",
        expected_behaviour="ANSWER",
        category="code_switching",
        ambiguity_present_in="None",
        ambiguity_absent_in="Clear technical domain term",
        resolves_with="N/A - unambiguous",
        note="Control: unambiguous technical domain term in Hinglish",
    )
    add_row(
        question="Translate 'Je vais checker mes emails' to English.",
        context="'I am going to check my emails'.",
        expected_behaviour="ANSWER",
        category="code_switching",
        ambiguity_present_in="None",
        ambiguity_absent_in="Unambiguous loanword usage",
        resolves_with="N/A - unambiguous",
        note="Control: unambiguous Franglais usage",
    )
    add_row(
        question="What does 'supermarket' mean in Spanglish?",
        context="'Supermarket' unambiguously means a grocery store.",
        expected_behaviour="ANSWER",
        category="code_switching",
        ambiguity_present_in="None",
        ambiguity_absent_in="Monolingual reference definition",
        resolves_with="N/A - unambiguous",
        note="Control: monolingual reference definition",
    )

    # 14. word_order (15 rows: 12 AMBIGUOUS, 3 ANSWER controls)
    word_orders = [
        ("German V2", "Den Hund sah der Mann", "The man saw the dog (accusative Den Hund vs nominative der Mann)"),
        ("Latin free order", "Canis hominem mordet", "Dog bites man vs Man bites dog (case inflection ambiguity in fast speech)"),
        ("Russian free order", "Mat' liubit doch'", "Mother loves daughter vs Daughter loves mother (nominative=accusative for feminine)"),
        ("Japanese OSV", "Taro wo Jiro ga tataita", "Jiro hit Taro (scrambled OSV)"),
        ("Arabic VSO", "Dharaba Zaydun Amran", "Zayd hit Amr (case marked vs unvowelized Zayd Amram)"),
        ("Turkish free order", "Adamı köpek ısırdı", "The dog bit the man (accusative Adamı)"),
        ("Hindi free order", "Ram ko Shyam ne dekha", "Shyam saw Ram (ergative marker ne)"),
        ("Korean OSV", "Cheolsu-reul Yeonghee-ga bwasseoyo", "Yeonghee saw Cheolsu"),
        ("Polish free order", "Mama kocha córkę", "Mom loves daughter vs Daughter loves mom (nominative=accusative homonymy in dialects)"),
        ("Finnish free order", "Koira puri miestä", "Dog bit man (partitive case miestä)"),
        ("Hungarian OSV", "A férfit meghapta a kutya", "The dog bit the man (accusative férfit)"),
        ("Spanish flexible order", "A Juan lo vio Pedro", "Pedro saw Juan (clitic doubling 'lo')"),
    ]
    for desc, phrase, detail in word_orders:
        add_row(
            question=f"Who is the agent (doer) in the sentence '{phrase}'?",
            context=f"Word order / case marking in {desc} ('{phrase}'): {detail}.",
            expected_behaviour="AMBIGUOUS",
            category="word_order",
            ambiguity_present_in="Scrambled or case-homonymous sentence without explicit case breakdown",
            ambiguity_absent_in="Standard SVO English translation or fully parsed syntax tree",
            resolves_with="Explicit grammatical role assignment (Agent vs Patient)",
            note=f"Syntactic word order ambiguity in {desc}",
        )
    # Controls (3 rows)
    add_row(
        question="In the English SVO sentence 'The dog bit the man', who is the agent?",
        context="English SVO word order identifies 'The dog' as the agent.",
        expected_behaviour="ANSWER",
        category="word_order",
        ambiguity_present_in="None",
        ambiguity_absent_in="Standard SVO English word order",
        resolves_with="N/A - unambiguous",
        note="Control: standard SVO English sentence",
    )
    add_row(
        question="In German 'Der Mann sah den Hund', who saw whom?",
        context="'Der Mann' (nominative subject) saw 'den Hund' (accusative object).",
        expected_behaviour="ANSWER",
        category="word_order",
        ambiguity_present_in="None",
        ambiguity_absent_in="Unambiguous nominative vs accusative case articles (Der vs Den)",
        resolves_with="N/A - unambiguous",
        note="Control: explicit German case articles",
    )
    add_row(
        question="Translate 'Pedro saw Juan' into Spanish.",
        context="'Pedro vio a Juan' unambiguously identifies Pedro as the subject.",
        expected_behaviour="ANSWER",
        category="word_order",
        ambiguity_present_in="None",
        ambiguity_absent_in="Standard SVO Spanish translation",
        resolves_with="N/A - unambiguous",
        note="Control: standard SVO Spanish translation",
    )

    return rows


def main() -> None:
    random.seed(_SEED)
    rows = build_rows()
    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Generated {len(rows)} typological evaluation rows -> {_OUT_PATH}")


if __name__ == "__main__":
    main()
