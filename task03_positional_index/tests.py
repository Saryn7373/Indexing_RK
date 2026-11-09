import unittest

from task03_positional_index.evaluator import Evaluator, difference, intersect, union
from task03_positional_index.index import PositionalIndex
from task03_positional_index.lexer import QuerySyntaxError, tokenize_query
from task03_positional_index.parser import And, Near, Not, Or, Phrase, Term, parse_query
from task03_positional_index.querygen import generate_queries
from task03_positional_index.scan import scan_search
from utils.corpus import load_corpus, load_json
from utils.paths import DATA_DIR

SMALL = load_json(DATA_DIR / "small_corpus.json")


def corpus_docs():
    return {d["doc_id"]: d["text"] for d in load_corpus()}


class SmallCorpusTests(unittest.TestCase):
    def setUp(self):
        self.docs = dict(SMALL["documents"])
        self.index = PositionalIndex.build(self.docs)
        self.evaluator = Evaluator(self.index)

    def test_hand_positions(self):
        for term, expected in SMALL["expected_positions"].items():
            actual = {d: self.index.positions_of(term, d) for d in self.index.docs_with(term)}
            self.assertEqual(actual, expected, term)

    def test_doc_lengths(self):
        self.assertEqual(self.index.doc_lengths, SMALL["doc_lengths"])
        self.assertTrue(self.index.is_sorted())

    def test_queries(self):
        for check in SMALL["queries"]:
            with self.subTest(query=check["query"]):
                if check.get("error"):
                    with self.assertRaises(QuerySyntaxError):
                        self.evaluator.search(check["query"])
                    with self.assertRaises(QuerySyntaxError):
                        scan_search(self.docs, check["query"])
                else:
                    self.assertEqual(self.evaluator.search(check["query"]), check["expected"])
                    self.assertEqual(scan_search(self.docs, check["query"]), check["expected"])

    def test_delete_updates_universe(self):
        self.index.remove_document("S6")
        self.assertEqual(self.evaluator.search("NOT роман"), ["S1", "S2", "S3", "S7"])
        self.assertEqual(self.evaluator.search("обзор"), [])
        self.assertNotIn("обзор", self.index.doc_lists)

    def test_update_removes_old_postings(self):
        self.index.update_document("S5", "роман обзор роман")
        self.assertEqual(self.evaluator.search("анализ"), [])
        self.assertEqual(self.index.positions_of("роман", "S5"), [0, 2])
        self.assertEqual(self.evaluator.search('"роман обзор"'), ["S5", "S6"])

    def test_incremental_equals_rebuild(self):
        self.index.remove_document("S3")
        self.index.update_document("S1", "Анна Каренина и Мёртвые души")
        self.index.add_document("S8", "Мёртвые души. Том второй.")
        self.index.add_document("S0", "Новая книга в библиотеке")
        self.docs.pop("S3")
        self.docs["S1"] = "Анна Каренина и Мёртвые души"
        self.docs["S8"] = "Мёртвые души. Том второй."
        self.docs["S0"] = "Новая книга в библиотеке"
        rebuilt = PositionalIndex.build(self.docs)
        self.assertEqual(self.index.to_dict(), rebuilt.to_dict())
        self.assertTrue(self.index.is_sorted())
        self.assertEqual(self.evaluator.search('"мертвые души"'), ["S1", "S2", "S8"])


class ParserTests(unittest.TestCase):
    def test_precedence(self):
        tree = parse_query("a OR b AND NOT c")
        self.assertEqual(tree, Or(Term("a"), And(Term("b"), Not(Term("c")))))

    def test_brackets_change_order(self):
        tree = parse_query("(a OR b) AND c")
        self.assertEqual(tree, And(Or(Term("a"), Term("b")), Term("c")))

    def test_phrase_and_near(self):
        tree = parse_query('"Мёртвые души" NEAR/5 гоголь')
        self.assertEqual(tree, Near(Phrase(["мертвые", "души"]), Term("гоголь"), 5))

    def test_lowercase_operators_are_words(self):
        self.assertEqual([t.kind for t in tokenize_query("и не and")], ["WORD", "WORD", "WORD"])

    def test_syntax_errors(self):
        for bad in ["AND", "NOT", "()", "a OR", "a )", "a NEAR b", "a NEAR/x b", "a , b", '""', "((a)"]:
            with self.subTest(query=bad), self.assertRaises(QuerySyntaxError):
                parse_query(bad)


class SetOperationTests(unittest.TestCase):
    def test_merge_operations(self):
        a, b = [1, 3, 5, 7], [3, 4, 5, 8]
        self.assertEqual(intersect(a, b), [3, 5])
        self.assertEqual(union(a, b), [1, 3, 4, 5, 7, 8])
        self.assertEqual(difference(a, b), [1, 7])
        self.assertEqual(intersect([], b), [])


class CorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = corpus_docs()
        cls.index = PositionalIndex.build(cls.docs)

    def test_corpus_size(self):
        self.assertGreaterEqual(len(self.docs), 120)
        self.assertTrue(self.index.is_sorted())

    def test_random_queries_match_scan(self):
        evaluator = Evaluator(self.index)
        for query in generate_queries(self.docs, 150, seed=7):
            with self.subTest(query=query):
                self.assertEqual(evaluator.search(query), scan_search(self.docs, query))


if __name__ == "__main__":
    unittest.main()
