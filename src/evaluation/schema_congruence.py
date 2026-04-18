import random
from dataclasses import dataclass, field


DISTRACTOR_POOLS = {
    "web_security": [
        "HTTPS uses TLS 1.3 for transport encryption.",
        "OAuth 2.0 uses authorization codes for secure delegation.",
        "CSRF tokens are single-use random values tied to session.",
        "Rate limiting prevents brute-force attacks on login endpoints.",
        "Content Security Policy headers mitigate XSS attacks.",
        "SQL parameterized queries prevent injection attacks.",
        "bcrypt work factor should be ≥12 for password hashing.",
        "CORS preflight requests use the OPTIONS HTTP method.",
        "Session cookies should have Secure and HttpOnly flags.",
        "JWTs are base64url-encoded, not encrypted by default.",
    ],
    "unrelated": [
        "The Amazon rainforest covers approximately 5.5 million km².",
        "Piano has 88 keys in standard concert configuration.",
        "Water boils at 100°C at sea level.",
        "Shakespeare wrote 37 plays and 154 sonnets.",
        "The speed of light is approximately 3×10⁸ m/s.",
        "Human genome contains approximately 3 billion base pairs.",
        "The Eiffel Tower was completed in 1889.",
        "Beethoven was deaf when he composed his Ninth Symphony.",
        "The Great Wall of China is not visible from space.",
        "Honey never spoils due to its low moisture content.",
    ],
}


@dataclass
class BenchmarkExample:
    gold_passage: str
    context: str
    question: str
    answer: str
    domain: str
    version: str  # "congruent" or "incongruent"
    distractors: list[str] = field(default_factory=list)
    target_length_k: int = 4


class CongruenceBenchmarkBuilder:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def build_example(
        self,
        gold_passage: str,
        domain: str,
        target_length_k: int,
        version: str,
        question: str = "",
        answer: str = "",
    ) -> BenchmarkExample:
        if version not in ("congruent", "incongruent"):
            raise ValueError(f"version must be 'congruent' or 'incongruent', got {version!r}")
        pool = DISTRACTOR_POOLS[domain] if version == "congruent" else DISTRACTOR_POOLS["unrelated"]
        n_distractors = max(1, target_length_k * 250 // max(len(gold_passage), 1))
        distractors = self.rng.choices(pool, k=min(n_distractors, len(pool)))
        all_passages = distractors.copy()
        insert_pos = self.rng.randint(0, len(all_passages))
        all_passages.insert(insert_pos, gold_passage)
        context = "\n\n".join(all_passages)
        return BenchmarkExample(
            gold_passage=gold_passage,
            context=context,
            question=question or f"What is stated about: {gold_passage[:40]}?",
            answer=answer or gold_passage,
            domain=domain,
            version=version,
            distractors=distractors,
            target_length_k=target_length_k,
        )
