from task07_duplicates.generator import check_split, generate, work_key
from task07_duplicates.jaccard import jaccard
from task07_duplicates.minhash import MinHasher, estimate_jaccard, shingle_hash
from task07_duplicates.shingles import text_shingles, word_shingles


def test_pdf_example():
    a, _ = text_shingles("a b c d", 2)
    b, _ = text_shingles("a b c e", 2)
    assert a == {("a", "b"), ("b", "c"), ("c", "d")}
    assert b == {("a", "b"), ("b", "c"), ("c", "e")}
    assert jaccard(a, b) == 0.5


def test_repeated_shingle_counted_once():
    assert word_shingles(["а", "б", "а", "б", "а", "б"], 2) == {("а", "б"), ("б", "а")}


def test_insufficient_tokens():
    shingles, status = text_shingles("Муму", 2)
    assert status == "insufficient_tokens" and shingles == set()
    assert text_shingles("", 3)[1] == "insufficient_tokens"
    assert jaccard(set(), set()) == 0.0


def test_minhash():
    hasher = MinHasher(128, seed=1)
    s = word_shingles("в одной из отдаленных улиц москвы в сером доме".split(), 3)
    assert estimate_jaccard(hasher.signature(s), hasher.signature(s)) == 1.0
    assert list(hasher.signature(s)) == list(MinHasher(128, seed=1).signature(s))
    assert hasher.signature(set()) is None
    assert shingle_hash(("a", "b")) == shingle_hash(("a", "b"))


def test_generator():
    pairs = generate()
    check_split(pairs)
    assert len(pairs) >= 60
    assert len({d for p in pairs for d in p["source_ids"]}) >= 100
    assert work_key("Война и мир. Том 1") == work_key("Война и мир")
    assert generate() == pairs


def main():
    tests = [test_pdf_example, test_repeated_shingle_counted_once, test_insufficient_tokens,
             test_minhash, test_generator]
    for test in tests:
        test()
        print("OK", test.__name__)


if __name__ == "__main__":
    main()
