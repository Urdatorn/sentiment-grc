from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set

DEFAULT_SYSTEM_PROMPT = (
    "Label this sentence according to its sentimental valence. "
    "Base your response solely on the sentimental value expressed in the text. "
    "Labels include: NEG = Negative; NEU = Neutral; POS = Positive"
)
VALID_LABELS = {"NEG", "NEU", "POS"}
DEFAULT_SOURCE = "acharnians"
ANABASIS_SOURCE = "anabasis"

SENTIMENT_DICT: Dict[str, str] = {
    "Εἶδες, ὦ πᾶσα πόλι, τὸν φρόνιμον ἄνδρα, τὸν ὑπέρσοφον": "POS",
    "Εἶδες τὸν φρόνιμον ἄνδρα, τὸν ὑπέρσοφον": "POS",
    "Εἶδες τὸν φρόνιμον ἄνδρα": "POS",
    "ἃ δ' ὠδυνήθην, ψαμμακοσιογάργαρα.": "NEG",
    "καὶ φιλῶ τοὺς ἱππέας διὰ τοῦτο τοὔργον· ἄξιον γὰρ Ἑλλάδι.": "POS",
    "φιλῶ τοὺς ἱππέας": "POS",
    "ἀποβλέπων εἰς τὸν ἀγρόν, εἰρήνης ἐρῶν,": "POS",
    "Ἀλλ' ἕτερον ἥσθην": "POS",
    "Ἀλλ' ἕτερον ἥσθην, ἡνίκ' ἐπὶ Μόσχῳ ποτὲ Δεξίθεος εἰσῆλθ' ᾀσόμενος Βοιώτιον.": "POS",
    "Ὦ Διονύσια, αὗται μὲν ὄζουσ' ἀμβροσίας καὶ νέκταρος καὶ μὴ 'πιτηρεῖν σιτί' ἡμερῶν τριῶν, κἀν τῷ στόματι λέγουσι· «Βαῖν' ὅπῃ 'θέλεις». Ταύτας δέχομαι καὶ σπένδομαι κἀκπίομαι, χαίρειν κελεύων πολλὰ τοὺς Ἀχαρνέας. Ἐγὼ δὲ πολέμου καὶ κακῶν ἀπαλλαγεὶς ἄξω τὰ κατ' ἀγροὺς εἰσιὼν Διονύσια.": "POS",
    "Ὦ Διονύσια, αὗται μὲν ὄζουσ' ἀμβροσίας καὶ νέκταρος καὶ μὴ 'πιτηρεῖν σιτί' ἡμερῶν τριῶν,": "POS",
    "αὗται ὄζουσ' ἀμβροσίας καὶ νέκταρος ": "POS",
    
    "Παρύσατις μὲν δὴ ἡ μήτηρ ὑπῆρχε τῷ Κύρῳ, φιλοῦσα αὐτὸν μᾶλλον ἢ τὸν βασιλεύοντα Ἀρταξέρξην.": "POS", # Anabasis
    "καὶ τῶν παρ᾽ ἑαυτῷ δὲ βαρβάρων ἐπεμελεῖτο ὡς πολεμεῖν τε ἱκανοὶ εἴησαν καὶ εὐνοϊκῶς ἔχοιεν αὐτῷ.": "POS", # Anabasis
    
    "Ἀλλ' ὠδυνήθην ἕτερον αὖ τραγῳδικόν, ὅτε δὴ 'κεχήνη προσδοκῶν τὸν Αἰσχύλον, ὁ δ' ἀνεῖπεν· Εἴσαγ', ὦ Θέογνι, τὸν χορόν. Πῶς τοῦτ' ἔσεισέ μου δοκεῖς τὴν καρδίαν;": "NEG",
    "ὠδυνήθην ἕτερον αὖ τραγῳδικόν": "NEG",
    "στυγῶν μὲν ἄστυ, τὸν δ' ἐμὸν δῆμον ποθῶν,": "NEG",
    "στυγῶν μὲν ἄστυ": "NEG",
    "Ὦνδρες πρυτάνεις, ἀδικεῖτε τὴν ἐκκλησίαν": "NEG",
    "ἀδικεῖτε τὴν ἐκκλησίαν": "NEG",
    "Ὦ Κραναὰ πόλις, ἆρ' αἰσθάνει τὸν κατάγελων τῶν πρέσβεων;": "NEG",
    "ἆρ' αἰσθάνει τὸν κατάγελων τῶν πρέσβεων;": "NEG",
    "Ἡμεῖς δὲ λαικαστάς τε καὶ καταπύγονας.": "NEG",
    "Ἡμεῖς δὲ καταπύγονας.": "NEG",
    "Ποίας ἀχάνας; Σὺ μὲν ἀλαζὼν εἶ μέγας.": "NEG",
    "Σὺ μὲν ἀλαζὼν εἶ μέγας.": "NEG",
    "Ἕτερος ἀλαζὼν οὗτος εἰσκηρύττεται.": "NEG",
    "Κάκιστ' ἀπολοίμην, εἴ τι τούτων πείθομαι ὧν εἶπας ἐνταυθοῖ σὺ πλὴν τῶν παρνόπων.": "NEG",
    "Κάκιστ' ἀπολοίμην, εἴ τι τούτων πείθομαι": "NEG",
    "Τουτὶ τί ἐστι τὸ κακόν;": "NEG",
    "Οἴμοι τάλας ἀπόλλυμαι, ὑπὸ τῶν Ὀδομάντων τὰ σκόροδα πορθούμενος.": "NEG",
    
    "Τὸν βασιλέως Ὀφθαλμὸν ἡ βουλὴ καλεῖ εἰς τὸ πρυτανεῖον.": "NEU",
    "Πάριτ' εἰς τὸ πρόσθεν, πάριθ'": "NEU",
    "Ἀλλ' Ἀμφίθεός μοι ποῦ 'στιν;": "NEU",
    "Οἱ Θρᾷκες ἴτε δεῦρ', οὓς Θέωρος ἤγαγεν.": "NEU",
    "Ἐγὼ δὲ φευξοῦμαί γε τοὺς Ἀχαρνέας.": "NEU",
}

ANABASIS_TEXTS: Set[str] = {
    "Παρύσατις μὲν δὴ ἡ μήτηρ ὑπῆρχε τῷ Κύρῳ, φιλοῦσα αὐτὸν μᾶλλον ἢ τὸν βασιλεύοντα Ἀρταξέρξην.",
    "καὶ τῶν παρ᾽ ἑαυτῷ δὲ βαρβάρων ἐπεμελεῖτο ὡς πολεμεῖν τε ἱκανοὶ εἴησαν καὶ εὐνοϊκῶς ἔχοιεν αὐτῷ.",
}


def build_chat_record(text: str, label: str, system_prompt: str, source: str = DEFAULT_SOURCE) -> dict:
    normalized_label = label.strip().upper()
    if normalized_label not in VALID_LABELS:
        raise ValueError(f"Invalid label '{label}' for text: {text!r}")

    return {
        "source": source,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
            {"role": "assistant", "content": normalized_label},
        ]
    }


def dict_to_records(data: Dict[str, str], system_prompt: str) -> Iterable[dict]:
    for text, label in data.items():
        source = ANABASIS_SOURCE if text in ANABASIS_TEXTS else DEFAULT_SOURCE
        yield build_chat_record(text=text, label=label, system_prompt=system_prompt, source=source)


def write_jsonl(records: List[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


def write_tsv(records: List[dict], output_path: Path) -> None:
    """Write tab-separated: text \t label \t source (one row per record)."""
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["text", "label", "source"])
        for record in records:
            msgs = record["messages"]
            text = next(m["content"] for m in msgs if m["role"] == "user")
            label = next(m["content"] for m in msgs if m["role"] == "assistant")
            source = record.get("source", DEFAULT_SOURCE)
            writer.writerow([text, label, source])


base = Path("data/sentiment_albin")
records = list(dict_to_records(SENTIMENT_DICT, system_prompt=DEFAULT_SYSTEM_PROMPT))
#write_jsonl(records, base.with_suffix(".jsonl"))
write_tsv(records, base.with_suffix(".tsv"))
print(f"Wrote {len(records)} records to {base}.jsonl and {base}.tsv")