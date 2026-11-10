"""Conservative six-topic strata for UltraChat's first user message.

UltraChat has no native topic labels. These frozen rules are an auditable
sampling instrument, not ground-truth semantic annotations. A prompt with
multiple cues or no cue is withheld from the balanced pilot.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


TOPICS: tuple[str, ...] = (
    "code",
    "creative_writing",
    "knowledge_qa",
    "summarization_transformation",
    "practical_howto",
    "business_marketing",
)

TOPIC_DESCRIPTIONS: dict[str, str] = {
    "code": "Programming, debugging, software APIs or source-code construction",
    "creative_writing": "Original stories, poems, scripts, fictional characters or scenes",
    "knowledge_qa": "Direct factual or conceptual questions, excluding recommendations",
    "summarization_transformation": "Summarization, translation, paraphrase, extraction or editing",
    "practical_howto": "Explicit recipes, household projects or exercise plans",
    "business_marketing": "Requested business plans, branding or marketing deliverables",
}


def _pattern(*expressions: str) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{expression})" for expression in expressions), re.I)


TOPIC_PATTERNS: dict[str, re.Pattern[str]] = {
    "code": _pattern(
        r"```(?:python|javascript|typescript|java|c\+\+|csharp|c#|go|rust|sql|html|css|bash)?",
        r"\b(?:write|create|implement|debug|fix|refactor|optimi[sz]e|develop|explain)\s+(?:a\s+|an\s+|the\s+|this\s+|my\s+)?(?:python|javascript|typescript|java|c\+\+|c#|go|rust|sql|html|css|bash|program|script|source\s+code|code|api|database\s+query)\b",
        r"\b(?:source\s+code|stack\s*trace|compiler\s+error|syntax\s+error|unit\s+test|api\s+endpoint|sql\s+query|javascript\s+function|python\s+function)\b",
        r"\b(?:using|in)\s+(?:python|javascript|typescript|java|c\+\+|c#|go|rust|sql|html|css|bash)\b",
    ),
    "creative_writing": _pattern(
        r"\b(?:write|create|compose|draft|generate)\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?(?:short\s+)?(?:story|poem|sonnet|novel|screenplay|fictional\s+scene|fairy\s+tale|fable|fantasy\s+scene|song\s+lyric|narrative)\b",
        r"\b(?:creative\s+writing|fictional\s+(?:character|story|world)|character\s+backstory|plot\s+outline|story\s+prompt|script\s+for\s+a\s+(?:film|play))\b",
    ),
    "knowledge_qa": _pattern(
        r"\b(?:who\s+(?:is|was|were)|when\s+(?:did|was)|where\s+(?:is|was)|what\s+(?:is|are|was|were)|why\s+(?:is|are|does|do|did)|how\s+(?:does|did|has|have|is|are|was)|which\s+(?:country|city|person|event|year|factor|method))\b",
        r"\b(?:define|definition\s+of|history\s+of|historical\s+background\s+of|facts?\s+about|difference\s+between|compare\s+and\s+contrast)\b",
    ),
    "summarization_transformation": _pattern(
        r"\b(?:summari[sz]e|give\s+(?:me\s+)?a\s+summary|paraphrase|translate|translation\s+of|rephrase|proofread|rewrite|reword)\b",
        r"\b(?:extract\s+(?:the|key|all)|convert\s+(?:this|the)\s+text|edit\s+the\s+following|sentiment\s+analysis\s+of\s+(?:this|the))\b",
    ),
    "practical_howto": _pattern(
        r"\b(?:recipe\s+for|how\s+to\s+(?:cook|bake|prepare|make|clean|repair|grow|plant|organize|decorate)|"
        r"step[- ]by[- ]step\s+(?:instructions|guide)\s+(?:for|on|to)\s+(?:cook|bake|prepar|mak|clean|repair|grow|plant|organiz|decorat)|"
        r"ingredients\s+(?:for|needed\s+to)|meal\s+plan|exercise\s+routine|workout\s+plan|"
        r"diy\s+(?:project|guide|tutorial)|home\s+(?:repair|improvement)\s+(?:guide|instructions))\b",
    ),
    "business_marketing": _pattern(
        r"\b(?:marketing\s+(?:plan|strategy|campaign)|business\s+(?:plan|strategy|proposal)|"
        r"advertising\s+campaign|sales\s+(?:strategy|pitch|plan)|"
        r"customer\s+(?:retention|acquisition)\s+(?:strategy|plan|program)|"
        r"market\s+(?:analysis|research|segmentation)|brand\s+(?:strategy|identity|positioning)|"
        r"social\s+media\s+campaign|business\s+model|revenue\s+model|"
        r"startup\s+(?:pitch|plan|strategy))\b",
    ),
}


# Cited material following these markers must not determine the topic.
CONTEXT_MARKER = re.compile(
    r"\b(?:answer|generate)\s+according\s+to\s*:|"
    r"\b(?:given|based\s+on)\s+the\s+(?:text|passage|article)\s*:|"
    r"\b(?:context|reference\s+text|source\s+text)\s*:", re.I,
)
WRAPPED_QUESTION = re.compile(
    r"^generate\b.{0,140}\bquestion\s*/\s*instruction\s*:\s*", re.I,
)
KNOWLEDGE_START = re.compile(
    r"^(?:(?:what|who|when|where|why|which|how)\b|"
    r"(?:can|could)\s+you\s+(?:explain|describe|define)\b|"
    r"(?:please\s+)?(?:define|explain|describe)\b)", re.I,
)
RECOMMENDATION_START = re.compile(
    r"^(?:what\s+are\s+some\s+(?:good|local|recommended)|"
    r"(?:can|could)\s+you\s+(?:suggest|recommend)|"
    r"which\s+(?:tour|trail|restaurant|hotel))\b", re.I,
)
BUSINESS_START = re.compile(
    r"^(?:(?:please\s+)?(?:write|create|develop|design|prepare|draft|outline|"
    r"propose|provide|generate|formulate|explain|analy[sz]e|include)\b|"
    r"(?:how|what|can|could)\b)", re.I,
)
REQUEST_CUE_CHARS = 240


def request_span(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    wrapper = WRAPPED_QUESTION.search(normalized)
    if wrapper is not None:
        normalized = normalized[wrapper.end():]
    marker = CONTEXT_MARKER.search(normalized)
    if marker is not None:
        normalized = normalized[:marker.start()]
    return normalized[:REQUEST_CUE_CHARS]


@dataclass(frozen=True)
class TopicDecision:
    label: str
    hits: dict[str, list[str]]


def classify_topic(text: str) -> TopicDecision:
    """Return exactly one topic, ``ambiguous``, or ``unclassified``."""
    if not isinstance(text, str):
        raise TypeError("first-user prompt must be text")
    request = request_span(text)
    hits: dict[str, list[str]] = {}
    for topic, pattern in TOPIC_PATTERNS.items():
        if topic == "knowledge_qa":
            if not KNOWLEDGE_START.search(request) or RECOMMENDATION_START.search(request):
                continue
        if topic == "business_marketing" and not BUSINESS_START.search(request):
            continue
        cues = sorted({match.group(0).casefold() for match in pattern.finditer(request)})
        if cues:
            hits[topic] = cues
    if len(hits) > 1:
        return TopicDecision("ambiguous", hits)
    if not hits:
        return TopicDecision("unclassified", {})
    return TopicDecision(next(iter(hits)), hits)
