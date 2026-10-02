import re
from pathlib import Path


def load_lexicon(path: Path) -> list[str]:
    skills: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        skill = line.strip().lower()
        if skill and not skill.startswith("#"):
            skills.append(skill)
    return skills


class SkillExtractor:
    """Match a curated lexicon with token boundaries.

    The same resume text always yields the same skills. A missing skill is a
    lexicon update, not a model change.
    """

    def __init__(self, lexicon: list[str]) -> None:
        # Longer phrases first so "machine learning" is recorded as itself
        # even when both tokens also exist as separate skills.
        ordered = sorted(set(lexicon), key=len, reverse=True)
        self.lexicon = ordered
        self._patterns = [
            (
                skill,
                re.compile(rf"(?<![a-z0-9]){re.escape(skill)}(?![a-z0-9])"),
            )
            for skill in ordered
        ]

    def extract(self, text: str) -> list[str]:
        lowered = text.lower()
        return [skill for skill, pattern in self._patterns if pattern.search(lowered)]
