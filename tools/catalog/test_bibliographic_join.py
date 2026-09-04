#!/usr/bin/env python3
"""Unit tests for tools/catalog/bibliographic_join.py.

Covers the failure modes measured on the current matched_goodreads.jsonl.gz:
token_set subset ties (Hunger Games), co-author blocking (The Help), inverted
catalog names (White Oleander), maiden/extra family token (Untamed), CJK name
order (Liu Cixin), initials (Stephen W. Hawking), year-title identity (1984),
and the ISBN-holdout eval protocol.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tools.catalog.bibliographic_join import (  # noqa: E402
    AdversarialPair,
    OLCandidate,
    ResidualLabel,
    TitleAuthorBlockIndex,
    author_score,
    block_keys_for,
    evaluate_adversarial_pairs,
    evaluate_residual_labels,
    evaluate_title_author_against_isbn,
    fold_for,
    fold_split_stats,
    load_adversarial_pairs,
    load_residual_labels,
    match_title_author,
    names_compatible,
    number_to_words,
    split_people,
    strip_series_suffix,
    title_core,
    title_lookup_keys,
    title_score,
    titles_identity,
)
from tools.catalog.ol_common import normalize_for_search  # noqa: E402


def _cand(work_key: str, title: str, author: str, edition_count: int = 1) -> OLCandidate:
    return OLCandidate(
        work_key=work_key,
        title=title,
        author=author,
        title_normalized=normalize_for_search(title),
        author_normalized=normalize_for_search(author),
        edition_count=edition_count,
    )


class TestTitleNormalization(unittest.TestCase):
    def test_strips_series_and_truncated_series(self) -> None:
        self.assertEqual(
            strip_series_suffix("The Hunger Games (The Hunger Games, #1)"),
            "The Hunger Games",
        )
        self.assertEqual(
            strip_series_suffix("A Court of Thorns and Roses (A Court of Thorns and Roses, #1"),
            "A Court of Thorns and Roses",
        )

    def test_strips_free_text_book_annotation_with_no_hash(self) -> None:
        """2j-5d regression: "Women on Top 2 (The Dud Wimpole Saga Book 1)"
        (the 2h acceptance-gate-override case, docs/BOOK_CATALOG.md) has no
        "#" at all, so the comma/hash convention above never stripped it."""
        self.assertEqual(
            strip_series_suffix("Women on Top 2 (The Dud Wimpole Saga Book 1)"),
            "Women on Top 2",
        )
        self.assertEqual(
            strip_series_suffix("Diary of a Wimpy Kid: The Long Haul (Book 9)"),
            "Diary of a Wimpy Kid: The Long Haul",
        )
        self.assertEqual(
            strip_series_suffix("A Study in Scarlet (Sherlock Holmes Book One)"),
            "A Study in Scarlet",
        )

    def test_does_not_strip_non_volume_parenthetical(self) -> None:
        """False-positive guard: a trailing paren containing "book" or
        "volume" as an ordinary word, not a volume keyword + number, must
        survive untouched."""
        self.assertEqual(strip_series_suffix("Steve Jobs (Book of the Year)"), "Steve Jobs (Book of the Year)")
        self.assertEqual(
            strip_series_suffix("The Great Gatsby (Illustrated Edition)"),
            "The Great Gatsby (Illustrated Edition)",
        )
        self.assertEqual(strip_series_suffix("Some Title (2020)"), "Some Title (2020)")

    def test_title_core_drops_article_and_subtitle(self) -> None:
        self.assertEqual(title_core("The Great Gatsby"), "great gatsby")
        self.assertEqual(
            title_core("The Diamond Age: Or, a Young Lady's Illustrated Primer"),
            "diamond age",
        )

    def test_title_core_strips_colon_subtitle_with_no_or_fallback(self) -> None:
        # Regression: normalize_for_search strips ":" as decorative
        # punctuation, so a colon check running *after* normalization never
        # fires. "The Diamond Age" test above happens to also contain
        # " or " and passed anyway, masking this for any subtitle that
        # isn't phrased as "Title: Or, ...". This is the far more common
        # case ("Title: Subtitle") and was previously landing on the *full*
        # title+subtitle as the core, missing OL's short-titled entry
        # entirely (see docs/BOOK_CATALOG.md's title_core subtitle-bug note).
        self.assertEqual(
            title_core("Atomic Habits: An Easy & Proven Way to Build Good Habits & Break Bad Ones"),
            "atomic habits",
        )
        self.assertEqual(
            title_core("Guns, Germs, and Steel: The Fates of Human Societies"),
            "guns germs and steel",
        )

    def test_title_core_keeps_single_word_head_unstripped(self) -> None:
        # A single-word head is too generic to trust as a stripped identity
        # key: "Batman: Knightfall, Part Three: Knightsend" must not reduce
        # to bare "batman" (which would exact-match every other "Batman:
        # ..." story against whatever OL work is titled just "Batman").
        # The cost: genuinely single-word main titles with a subtitle
        # ("Sapiens: A Brief History of Humankind") also don't get
        # stripped -- an accepted, narrower gap. See title_core's docstring.
        self.assertEqual(
            title_core("Batman: Knightfall, Part Three: Knightsend"),
            "batman knightfall part three knightsend",
        )
        self.assertEqual(title_core("Sapiens: A Brief History of Humankind"), "sapiens a brief history of humankind")

    def test_title_core_keeps_volume_marker_in_discarded_tail(self) -> None:
        # "The Chronicles of Amber" has a substantial 4-word head (passes
        # the >= 2 words check above) but "Volume II" vs "Volume I" is the
        # *only* thing distinguishing the two Goodreads entries -- stripping
        # it collapses both onto the same bare series title.
        self.assertEqual(
            title_core("The Chronicles of Amber: Volume II (The Chronicles of Amber #3-5)"),
            "chronicles of amber volume ii",
        )
        self.assertEqual(
            title_core("The Chronicles of Amber: Volume I (The Chronicles of Amber, #1-2)"),
            "chronicles of amber volume i",
        )
        self.assertNotEqual(
            title_core("The Chronicles of Amber: Volume II (The Chronicles of Amber #3-5)"),
            title_core("The Chronicles of Amber: Volume I (The Chronicles of Amber, #1-2)"),
        )

    def test_title_score_rejects_mismatched_volume_numbers(self) -> None:
        # A pre-existing token_sort_ratio weakness: one differing digit
        # among many shared tokens barely moves the ratio, so "Volume 2"
        # vs "Volume 3" scores high enough to pass TITLE_ACCEPT despite
        # being different specific installments.
        self.assertEqual(title_score("ghostbusters volume 2", "ghostbusters volume 3"), 0.0)
        self.assertEqual(title_score("batman knightfall part two", "batman knightfall part three"), 0.0)
        # Shared/no numbers: scored normally, not forced to 0.
        self.assertGreater(title_score("ghostbusters volume 2", "ghostbusters volume 2"), 90.0)
        self.assertGreater(title_score("the great gatsby", "great gatsby"), 50.0)

    def test_title_score_rejects_mismatched_roman_numerals(self) -> None:
        # The Amber case: "chronicles of amber volume i" vs "... volume ii"
        # differ by a single character, so token_sort_ratio alone scores it
        # ~99 -- well above TITLE_ACCEPT -- despite being different volumes.
        self.assertEqual(
            title_score("chronicles of amber volume i", "chronicles of amber volume ii"),
            0.0,
        )

    def test_year_title_identity(self) -> None:
        self.assertEqual(number_to_words(1984), "nineteen eighty four")
        self.assertTrue(titles_identity("1984", "nineteen eighty four"))
        self.assertTrue(titles_identity("nineteen eighty-four", "1984"))
        self.assertFalse(titles_identity("1984", "animal farm"))
        keys = block_keys_for("1984", "George Orwell")
        self.assertIn("orwell|1984", keys)
        self.assertIn("orwell|nineteen", keys)
        lookups = title_lookup_keys("1984")
        self.assertIn("1984", lookups)
        self.assertIn("nineteen eighty-four", lookups)


class TestAuthorParsing(unittest.TestCase):
    def test_inverted_catalog_name(self) -> None:
        people = split_people("Fitch, Janet")
        self.assertEqual(len(people), 1)
        self.assertEqual(people[0].family, "fitch")
        self.assertTrue(people[0].inverted)

    def test_coauthors_split_on_and(self) -> None:
        people = split_people("Kathryn Stockett, Álvaro Abella Villar, and Alvaro Abel")
        families = {p.family for p in people}
        self.assertIn("stockett", families)

    def test_names_compatible_inverted_and_western(self) -> None:
        self.assertTrue(names_compatible("Janet Fitch", "Fitch, Janet"))

    def test_names_compatible_coauthor_and_translators(self) -> None:
        self.assertTrue(
            names_compatible(
                "Kathryn Stockett",
                "Kathryn Stockett, Álvaro Abella Villar, and Alvaro Abel",
            )
        )

    def test_names_compatible_maiden_extra_token(self) -> None:
        self.assertTrue(names_compatible("Glennon Doyle", "Glennon Doyle Melton"))

    def test_names_compatible_cjk_order_flip(self) -> None:
        self.assertTrue(names_compatible("Liu Cixin", "Cixin Liu"))

    def test_names_compatible_initials_vs_full(self) -> None:
        self.assertTrue(names_compatible("Stephen W. Hawking", "Stephen Hawking"))
        self.assertTrue(names_compatible("J.K. Rowling", "J. K. Rowling"))

    def test_names_reject_same_title_different_author(self) -> None:
        self.assertFalse(names_compatible("Stephenie Meyer", "Dean Koontz"))
        self.assertFalse(names_compatible("John Green", "John Boyne"))
        self.assertLess(author_score("Stephenie Meyer", "Dean Koontz"), 100.0)


class TestMatchTitleAuthor(unittest.TestCase):
    def test_hunger_games_does_not_tie_companion(self) -> None:
        index = TitleAuthorBlockIndex(
            [
                _cand("/works/HG", "The Hunger Games", "Suzanne Collins", 142),
                _cand(
                    "/works/GUIDE",
                    "The Hunger Games Official Illustrated Movie Companion",
                    "Suzanne Collins",
                    5,
                ),
            ]
        )
        result = match_title_author("The Hunger Games (The Hunger Games, #1)", "Suzanne Collins", index)
        self.assertEqual(result.method, "title_author")
        self.assertEqual(result.work_key, "/works/HG")

    def test_identity_picks_highest_edition_count(self) -> None:
        index = TitleAuthorBlockIndex(
            [
                _cand("/works/JUNK", "To Kill a Mockingbird", "Harper Lee", 1),
                _cand("/works/CANON", "To Kill a Mockingbird", "Harper Lee", 213),
                _cand("/works/GRAPHIC", "To Kill a Mockingbird", "Fred Fordham and Harper Lee", 2),
            ]
        )
        result = match_title_author("To Kill a Mockingbird", "Harper Lee", index)
        self.assertEqual(result.method, "title_author")
        self.assertEqual(result.work_key, "/works/CANON")

    def test_twilight_does_not_match_other_author(self) -> None:
        index = TitleAuthorBlockIndex(
            [
                _cand("/works/MEYER", "Twilight", "Stephenie Meyer", 131),
                _cand("/works/KOONTZ", "Twilight", "Dean Koontz", 27),
            ]
        )
        result = match_title_author("Twilight (The Twilight Saga, #1)", "Stephenie Meyer", index)
        self.assertEqual(result.work_key, "/works/MEYER")

    def test_help_matches_through_translator_coauthors(self) -> None:
        index = TitleAuthorBlockIndex(
            [
                _cand(
                    "/works/HELP",
                    "The Help",
                    "Kathryn Stockett, Álvaro Abella Villar, and Alvaro Abel",
                    73,
                )
            ]
        )
        result = match_title_author("The Help", "Kathryn Stockett", index)
        self.assertEqual(result.method, "title_author")
        self.assertEqual(result.work_key, "/works/HELP")

    def test_white_oleander_inverted_author(self) -> None:
        index = TitleAuthorBlockIndex([_cand("/works/WO", "White oleander", "Fitch, Janet", 42)])
        result = match_title_author("White Oleander", "Janet Fitch", index)
        self.assertEqual(result.work_key, "/works/WO")

    def test_untamed_maiden_name(self) -> None:
        index = TitleAuthorBlockIndex(
            [
                _cand("/works/BRAND", "Untamed", "Max Brand", 27),
                _cand("/works/DOYLE", "Untamed", "Glennon Doyle Melton", 7),
            ]
        )
        result = match_title_author("Untamed", "Glennon Doyle", index)
        self.assertEqual(result.work_key, "/works/DOYLE")

    def test_hawking_middle_initial(self) -> None:
        index = TitleAuthorBlockIndex(
            [_cand("/works/TIME", "A Brief History of Time", "Stephen Hawking", 120)]
        )
        result = match_title_author("A Brief History of Time", "Stephen W. Hawking", index)
        self.assertEqual(result.work_key, "/works/TIME")

    def test_1984_maps_to_nineteen_eighty_four(self) -> None:
        index = TitleAuthorBlockIndex(
            [
                _cand("/works/YEAR", "1984", "George Orwell and Amélie Audiberti", 8),
                _cand("/works/CANON", "Nineteen Eighty-Four", "George Orwell", 536),
            ]
        )
        result = match_title_author("1984", "George Orwell", index)
        self.assertEqual(result.method, "title_author")
        self.assertEqual(result.work_key, "/works/CANON")

    def test_cjk_name_order(self) -> None:
        index = TitleAuthorBlockIndex(
            [_cand("/works/3BP", "The Three-Body Problem", "Cixin Liu", 40)]
        )
        result = match_title_author(
            "The Three-Body Problem (Remembrance of Earth’s Past, #1)", "Liu Cixin", index
        )
        self.assertEqual(result.work_key, "/works/3BP")

    def test_unknown_author_is_unmatched(self) -> None:
        index = TitleAuthorBlockIndex([_cand("/works/LOTR", "The Lord of the Rings", "J. R. R. Tolkien", 500)])
        result = match_title_author("Some Book", "Someone Nobody Wrote About", index)
        self.assertEqual(result.method, "unmatched")

    def test_pseudonym_without_shared_name_is_unmatched(self) -> None:
        index = TitleAuthorBlockIndex(
            [_cand("/works/FB", "Fantastic Beasts and Where to Find Them", "J. K. Rowling", 20)]
        )
        result = match_title_author("Fantastic Beasts and Where to Find Them", "Newt Scamander", index)
        self.assertEqual(result.method, "unmatched")


class TestIsbnHoldoutEval(unittest.TestCase):
    def test_reports_exact_and_identity_equivalent(self) -> None:
        candidates = [
            _cand("/works/CANON", "Pride and Prejudice", "Jane Austen", 4038),
            _cand("/works/DUP", "Pride and Prejudice", "Jane Austen", 2),
        ]
        index = TitleAuthorBlockIndex(candidates)
        isbn_index = {"9780141439518": ["/works/CANON"]}
        work_ta = {c.work_key: (c.title, c.author) for c in candidates}
        report = evaluate_title_author_against_isbn(
            [(1, "Pride and Prejudice", "Jane Austen", "9780141439518")],
            index,
            isbn_index,
            work_title_author=work_ta,
        )
        self.assertEqual(report["gold_pairs"], 1)
        self.assertEqual(report["exact_work_key"], 1)
        self.assertEqual(report["identity_equivalent"], 1)
        self.assertEqual(report["conflict"], 0)

    def test_identity_equivalent_when_work_keys_differ(self) -> None:
        candidates = [
            _cand("/works/CANON", "The Great Gatsby", "F. Scott Fitzgerald", 1179),
            _cand("/works/ISBN_DUP", "The Great Gatsby", "F. Scott Fitzgerald", 1),
        ]
        index = TitleAuthorBlockIndex(candidates)
        isbn_index = {"9780743273565": ["/works/ISBN_DUP"]}
        work_ta = {c.work_key: (c.title, c.author) for c in candidates}
        report = evaluate_title_author_against_isbn(
            [(1, "The Great Gatsby", "F. Scott Fitzgerald", "9780743273565")],
            index,
            isbn_index,
            work_title_author=work_ta,
        )
        self.assertEqual(report["exact_work_key"], 0)
        self.assertEqual(report["identity_equivalent"], 1)
        self.assertEqual(report["conflict"], 0)

    def test_gold_quality_filter_excludes_placeholder_gold_by_default(self) -> None:
        # A real OL data-quality pattern: an ISBN pre-registered before the
        # book's final title/author was set, so its OL work record is a
        # stub ("Untitled", author "To Be Announced") rather than the real
        # book. That stub is not a usable gold label for *any* matcher,
        # regardless of what it predicts -- filtering it out (default on)
        # keeps the eval measuring matcher correctness, not gold-label noise.
        candidates = [_cand("/works/REAL", "Atomic Habits", "James Clear", 40)]
        index = TitleAuthorBlockIndex(candidates)
        isbn_index = {"9781234567890": ["/works/STUB"]}
        work_ta = {"/works/STUB": ("Untitled", "To Be Announced")}
        report = evaluate_title_author_against_isbn(
            [(1, "Atomic Habits", "James Clear", "9781234567890")],
            index,
            isbn_index,
            work_title_author=work_ta,
        )
        self.assertEqual(report["gold_pairs"], 0)
        self.assertEqual(report["gold_quality_excluded"], 1)

    def test_gold_quality_filter_can_be_disabled(self) -> None:
        candidates = [_cand("/works/REAL", "Atomic Habits", "James Clear", 40)]
        index = TitleAuthorBlockIndex(candidates)
        isbn_index = {"9781234567890": ["/works/STUB"]}
        work_ta = {"/works/STUB": ("Untitled", "To Be Announced")}
        report = evaluate_title_author_against_isbn(
            [(1, "Atomic Habits", "James Clear", "9781234567890")],
            index,
            isbn_index,
            work_title_author=work_ta,
            filter_unreliable_gold=False,
        )
        self.assertEqual(report["gold_pairs"], 1)
        self.assertEqual(report["gold_quality_excluded"], 0)

    def test_gold_quality_filter_keeps_compatible_author_gold(self) -> None:
        candidates = [_cand("/works/CANON", "Pride and Prejudice", "Jane Austen", 4038)]
        index = TitleAuthorBlockIndex(candidates)
        isbn_index = {"9780141439518": ["/works/CANON"]}
        work_ta = {"/works/CANON": ("Pride and Prejudice", "Jane Austen")}
        report = evaluate_title_author_against_isbn(
            [(1, "Pride and Prejudice", "Jane Austen", "9780141439518")],
            index,
            isbn_index,
            work_title_author=work_ta,
        )
        self.assertEqual(report["gold_pairs"], 1)
        self.assertEqual(report["gold_quality_excluded"], 0)

    def test_gold_quality_filter_is_noop_without_work_title_author(self) -> None:
        index = TitleAuthorBlockIndex([_cand("/works/REAL", "Atomic Habits", "James Clear", 40)])
        isbn_index = {"9781234567890": ["/works/STUB"]}
        report = evaluate_title_author_against_isbn(
            [(1, "Atomic Habits", "James Clear", "9781234567890")], index, isbn_index
        )
        self.assertEqual(report["gold_pairs"], 1)
        self.assertEqual(report["gold_quality_excluded"], 0)


class TestFoldFor(unittest.TestCase):
    def test_is_deterministic(self) -> None:
        self.assertEqual(fold_for(33), fold_for(33))
        self.assertIn(fold_for(33), ("tuning", "validation"))

    def test_different_salt_can_change_assignment(self) -> None:
        # Not asserting a specific book_id flips (that would over-specify
        # the hash), just that salt is actually load-bearing, not ignored.
        book_ids = range(1, 500)
        default_folds = {b: fold_for(b) for b in book_ids}
        salted_folds = {b: fold_for(b, salt="a-different-salt") for b in book_ids}
        self.assertNotEqual(default_folds, salted_folds)

    def test_roughly_80_20_split_over_a_large_sample(self) -> None:
        folds = [fold_for(book_id) for book_id in range(1, 20_000)]
        validation_fraction = folds.count("validation") / len(folds)
        self.assertAlmostEqual(validation_fraction, 0.2, delta=0.02)

    def test_split_is_balanced_across_popularity_deciles(self) -> None:
        # Guards the exact pitfall 2b calls out: a *naive* split of a
        # popularity-ordered gold set skews validation toward the tail.
        # A hash-based split should stay close to 80/20 in every decile.
        ratings_by_book_id = {book_id: 1_000_000 - book_id for book_id in range(1, 5_000)}
        stats = fold_split_stats(ratings_by_book_id)
        self.assertEqual(len(stats["deciles"]), 10)
        for decile in stats["deciles"]:
            self.assertAlmostEqual(decile["validation_fraction"], 0.2, delta=0.05)

    def test_fold_split_stats_empty_input(self) -> None:
        self.assertEqual(fold_split_stats({}), {"deciles": [], "overall_validation_fraction": 0.0})


class TestAdversarialPairs(unittest.TestCase):
    def test_load_adversarial_pairs_from_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.yaml"
            path.write_text(
                """
pairs:
  - title_a: "Twilight"
    author_a: "Stephenie Meyer"
    title_b: "Twilight"
    author_b: "Dean Koontz"
    note: "same title, different books"
""",
                encoding="utf-8",
            )
            pairs = load_adversarial_pairs(path)
            self.assertEqual(len(pairs), 1)
            self.assertEqual(pairs[0].title_a, "Twilight")
            self.assertEqual(pairs[0].author_b, "Dean Koontz")
            self.assertEqual(pairs[0].note, "same title, different books")

    def test_load_adversarial_pairs_missing_file_returns_empty(self) -> None:
        self.assertEqual(load_adversarial_pairs(Path("/tmp/does-not-exist-adversarial.yaml")), [])

    def test_detects_false_merge_when_both_sides_resolve_to_same_work(self) -> None:
        # A deliberately bad index that would incorrectly merge Meyer's and
        # Koontz's "Twilight" onto one work_key.
        index = TitleAuthorBlockIndex(
            [_cand("/works/AMBIGUOUS", "Twilight", "Stephenie Meyer and Dean Koontz", 50)]
        )
        pairs = [AdversarialPair(title_a="Twilight", author_a="Stephenie Meyer", title_b="Twilight", author_b="Dean Koontz")]
        report = evaluate_adversarial_pairs(pairs, index)
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["false_merges"], 1)
        self.assertEqual(report["false_merge_pairs"][0]["work_key"], "/works/AMBIGUOUS")

    def test_no_false_merge_when_correctly_disambiguated(self) -> None:
        index = TitleAuthorBlockIndex(
            [
                _cand("/works/MEYER", "Twilight", "Stephenie Meyer", 131),
                _cand("/works/KOONTZ", "Twilight", "Dean Koontz", 27),
            ]
        )
        pairs = [AdversarialPair(title_a="Twilight", author_a="Stephenie Meyer", title_b="Twilight", author_b="Dean Koontz")]
        report = evaluate_adversarial_pairs(pairs, index)
        self.assertEqual(report["false_merges"], 0)
        self.assertEqual(report["false_merge_pairs"], [])

    def test_no_false_merge_when_one_side_is_unmatched(self) -> None:
        index = TitleAuthorBlockIndex([_cand("/works/MEYER", "Twilight", "Stephenie Meyer", 131)])
        pairs = [AdversarialPair(title_a="Twilight", author_a="Stephenie Meyer", title_b="Twilight", author_b="Dean Koontz")]
        report = evaluate_adversarial_pairs(pairs, index)
        self.assertEqual(report["false_merges"], 0)

    def test_the_committed_yaml_loads_and_has_no_self_false_merges(self) -> None:
        # The shipped adversarial set should parse and, trivially, not
        # already contain a pair pointed at literally the same OLCandidate
        # set they were curated against (a real check needs full.sqlite;
        # this just guards the fixture file itself from bit-rotting into
        # unparsable YAML or an empty set).
        committed_path = Path(__file__).resolve().parent / "matcher_adversarial_pairs.yaml"
        pairs = load_adversarial_pairs(committed_path)
        self.assertGreaterEqual(len(pairs), 5)
        for pair in pairs:
            self.assertTrue(pair.title_a and pair.author_a and pair.title_b and pair.author_b)


class TestResidualLabels(unittest.TestCase):
    def _write(self, tmp: Path, text: str) -> Path:
        path = Path(tmp) / "matcher_residual_labels.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(load_residual_labels(Path("/tmp/does-not-exist-residual.yaml")), [])

    def test_unreviewed_entry_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                "entries:\n"
                "  - goodreads_book_id: 1\n"
                "    sample_reason: unmatched_popular\n"
                "    verdict: null\n",
            )
            self.assertEqual(load_residual_labels(path), [])

    def test_reviewed_entry_loads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                "entries:\n"
                "  - goodreads_book_id: 1\n"
                "    sample_reason: low_margin_title_author\n"
                "    candidate_work_key: /works/OL1W\n"
                "    verdict: correct\n"
                "    notes: looks right\n",
            )
            labels = load_residual_labels(path)
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0].goodreads_book_id, 1)
            self.assertEqual(labels[0].verdict, "correct")
            self.assertEqual(labels[0].candidate_work_key, "/works/OL1W")
            self.assertEqual(labels[0].notes, "looks right")

    def test_invalid_verdict_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(
                tmp,
                "entries:\n  - goodreads_book_id: 1\n    verdict: maybe\n",
            )
            with self.assertRaises(ValueError):
                load_residual_labels(path)

    def test_truth_work_key_for_each_verdict(self) -> None:
        correct = ResidualLabel(goodreads_book_id=1, sample_reason="x", verdict="correct", candidate_work_key="/works/A")
        self.assertEqual(correct.truth_work_key, "/works/A")

        wrong_corrected = ResidualLabel(
            goodreads_book_id=2, sample_reason="x", verdict="wrong", candidate_work_key="/works/A", corrected_work_key="/works/B"
        )
        self.assertEqual(wrong_corrected.truth_work_key, "/works/B")

        wrong_uncorrected = ResidualLabel(goodreads_book_id=3, sample_reason="x", verdict="wrong", candidate_work_key="/works/A")
        self.assertIsNone(wrong_uncorrected.truth_work_key)

        no_ol_match = ResidualLabel(goodreads_book_id=4, sample_reason="x", verdict="no_ol_match", candidate_work_key="/works/A")
        self.assertIsNone(no_ol_match.truth_work_key)

        unsure = ResidualLabel(goodreads_book_id=5, sample_reason="x", verdict="unsure", candidate_work_key="/works/A")
        self.assertIsNone(unsure.truth_work_key)

    def test_evaluate_residual_labels_scores_only_checkable_entries(self) -> None:
        labels = [
            ResidualLabel(goodreads_book_id=1, sample_reason="x", verdict="correct", candidate_work_key="/works/A"),
            ResidualLabel(
                goodreads_book_id=2, sample_reason="x", verdict="wrong", candidate_work_key="/works/A", corrected_work_key="/works/B"
            ),
            ResidualLabel(goodreads_book_id=3, sample_reason="x", verdict="no_ol_match", candidate_work_key="/works/C"),
            ResidualLabel(goodreads_book_id=4, sample_reason="x", verdict="unsure", candidate_work_key="/works/D"),
        ]
        predicted = {1: "/works/A", 2: "/works/A", 3: None, 4: "/works/D"}
        report = evaluate_residual_labels(labels, predicted)
        self.assertEqual(report["reviewed"], 4)
        self.assertEqual(report["checkable"], 2)  # only "correct" and "wrong"-with-correction have a truth
        self.assertEqual(report["no_ol_match"], 1)
        self.assertEqual(report["unsure"], 1)
        self.assertEqual(report["correct"], 1)  # book 1 matches; book 2's current prediction is still the old wrong one
        self.assertAlmostEqual(report["accuracy"], 0.5)
        self.assertEqual(len(report["mismatches"]), 1)
        self.assertEqual(report["mismatches"][0]["goodreads_book_id"], 2)

    def test_evaluate_residual_labels_empty_is_zero_not_crash(self) -> None:
        report = evaluate_residual_labels([], {})
        self.assertEqual(report["checkable"], 0)
        self.assertEqual(report["accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
