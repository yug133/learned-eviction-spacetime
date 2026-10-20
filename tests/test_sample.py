import subprocess
import sys
from pathlib import Path

from sample import MASK64, id_hash, sample_path, splitmix64

ROOT = Path(__file__).resolve().parent.parent


def test_splitmix64_known_answer():
    # First output of Vigna's reference splitmix64.c seeded with 0.
    assert splitmix64(0) == 0xE220A8397B1DCDAF


def test_integer_ids_use_top_32_bits_of_splitmix64():
    assert id_hash(b"12345") == splitmix64(12345) >> 32
    assert id_hash(str(2 ** 70).encode()) == splitmix64(2 ** 70 & MASK64) >> 32


def test_id_zero_is_not_always_selected():
    # The old hash had h(0) = 0, so object 0 was in every sample at every rate.
    assert id_hash(b"0") != 0
    for rate in (0.001, 0.01, 0.1, 0.5):
        assert id_hash(b"0") >= rate * 2 ** 32    # id 0 is NOT in these samples


def test_small_consecutive_ids_are_selected_at_the_right_rate():
    # Wikipedia ids are 0, 1, 2, ... For 100,000 ids at rate 1% expect 1,000 selected;
    # binomial sd is about 31, so 850-1150 is a ~5 sd band.
    threshold = 0.01 * 2 ** 32
    selected = sum(1 for i in range(100_000) if id_hash(str(i).encode()) < threshold)
    assert 850 <= selected <= 1150
    # and the very smallest ids are not over-represented
    assert sum(1 for i in range(100) if id_hash(str(i).encode()) < threshold) <= 5


def test_hash_is_32_bit_for_string_ids():
    assert 0 <= id_hash(b"Main_Page") < 2 ** 32


def test_samples_are_nested_and_keep_every_request_of_an_object(tmp_path):
    lines = [f"{1000 + i} {i % 500} 10 extra\n" for i in range(5000)]
    result = subprocess.run(
        [sys.executable, str(ROOT / "sample.py"), "--rates", "0.05", "0.3",
         "--out", str(tmp_path), "--stats", str(tmp_path / "stats.txt")],
        input="".join(lines).encode(), capture_output=True, check=True,
    )
    assert b"total requests" in result.stdout

    small = (tmp_path / "wiki2018_0.05.tr").read_text().splitlines()
    large = (tmp_path / "wiki2018_0.3.tr").read_text().splitlines()
    small_ids = {line.split()[1] for line in small}
    large_ids = {line.split()[1] for line in large}
    assert small_ids <= large_ids
    # each id 0..499 appears exactly 10 times in the input; spatial sampling keeps all 10
    assert len(small) == 10 * len(small_ids)
    assert len(large) == 10 * len(large_ids)
    assert large[0].split() == lines[int(large[0].split()[0]) - 1000].split()[:3]


def test_one_hit_wonder_statistics(tmp_path):
    # ids 1 and 2 once each, id 3 three times, id 4 twice: 2 of 4 objects are one-hit wonders
    lines = "".join(f"{100 + i} {k} 10\n" for i, k in enumerate([3, 1, 3, 4, 2, 4, 3]))
    result = subprocess.run(
        [sys.executable, str(ROOT / "sample.py"), "--rates", "1.0",
         "--out", str(tmp_path), "--stats", str(tmp_path / "stats.txt")],
        input=lines.encode(), capture_output=True, check=True,
    )
    stats = (tmp_path / "stats.txt").read_text()
    assert stats == result.stdout.decode()
    assert "one-hit wonders       : 2 (0.5000 of unique objects)" in stats
    assert "one-hit wonders (est.): 0.5000 of unique objects" in stats


def test_top_object_share(tmp_path):
    # id 7 gets 3 of 5 requests
    lines = "".join(f"{100 + i} {k} 10\n" for i, k in enumerate([7, 1, 7, 2, 7]))
    subprocess.run(
        [sys.executable, str(ROOT / "sample.py"), "--rates", "1.0",
         "--out", str(tmp_path), "--stats", str(tmp_path / "stats.txt")],
        input=lines.encode(), capture_output=True, check=True,
    )
    stats = (tmp_path / "stats.txt").read_text()
    assert "top object share      : 0.6000 of requests (id 7, 3 requests)" in stats


def test_default_seed_is_the_original_hash():
    for key in (b"0", b"295", b"8715", b"123456789"):
        assert id_hash(key) == id_hash(key, 0) == splitmix64(int(key)) >> 32


def offset_partner_shares(hash_fn, offsets=(1, 2, 3, 4, 5), n_objects=200_000, rate=0.01):
    """For each offset d: the fraction of seed-1 ids k for which k + d is selected under
    seed 0. If the seed-1 sample were the seed-0 sample shifted by d ids, this is 100% at
    that d; for unrelated samples it is about the sampling rate."""
    threshold = rate * 2 ** 32
    seed0 = {k for k in range(n_objects + max(offsets)) if hash_fn(k, 0) < threshold}
    seed1 = [k for k in range(n_objects) if hash_fn(k, 1) < threshold]
    return {d: sum(1 for k in seed1 if k + d in seed0) / len(seed1) for d in offsets}


def test_seed_one_is_not_a_shift_of_seed_zero():
    rate = 0.01
    shares = offset_partner_shares(lambda k, seed: id_hash(str(k).encode(), seed), rate=rate)
    for d, share in shares.items():
        assert share < 3 * rate, (d, share)    # ~rate expected: no offset pairs the samples


def test_the_old_shifted_seeding_fails_the_offset_check():
    # splitmix64(id + seed), used until 20 Oct 2026: the seed-1 sample was the seed-0 sample
    # shifted by one id. The check above must catch it.
    def old_hash(k, seed):
        return splitmix64((k + seed) & MASK64) >> 32
    shares = offset_partner_shares(old_hash)
    assert shares[1] == 1.0


def test_nonzero_seed_hashes_the_id_first():
    assert id_hash(b"12345", 7) == splitmix64((splitmix64(12345) + 7) & MASK64) >> 32
    assert id_hash(b"12345", 0) == splitmix64(12345) >> 32    # seed 0 unchanged


def test_seed_goes_into_file_names():
    assert sample_path("data", 0.01, 0).endswith("wiki2018_0.01.tr")
    assert sample_path("data", 0.01, 1).endswith("wiki2018_0.01_seed1.tr")


def run_sample(tmp_path, *extra):
    lines = "".join(f"{100 + i} {i % 50} 10\n" for i in range(500))
    return subprocess.run(
        [sys.executable, str(ROOT / "sample.py"), "--out", str(tmp_path),
         "--stats", str(tmp_path / "stats.txt"), *extra],
        input=lines.encode(), capture_output=True,
    )


def test_single_rate_with_other_seed_and_overwrite_guard(tmp_path):
    first = run_sample(tmp_path, "--rates", "0.5")
    assert first.returncode == 0
    assert (tmp_path / "wiki2018_0.5.tr").exists()
    again = run_sample(tmp_path, "--rates", "0.5")              # stats.txt exists now
    assert again.returncode != 0 and b"would overwrite" in again.stderr
    other = run_sample(tmp_path, "--rates", "0.5", "--seed", "1", "--overwrite")
    assert other.returncode == 0
    assert (tmp_path / "wiki2018_0.5_seed1.tr").exists()
    assert b"hash seed             : 1" in other.stdout
    assert (tmp_path / "wiki2018_0.5.tr").read_text() != (tmp_path / "wiki2018_0.5_seed1.tr").read_text()
