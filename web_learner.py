import json
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional

KB_PATH = "knowledge_base.json"

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

try:
    import wikipediaapi
    WIKI_AVAILABLE = True
except ImportError:
    WIKI_AVAILABLE = False

try:
    import trafilatura
    TRAFILATURA_AVAILABLE = True
except ImportError:
    TRAFILATURA_AVAILABLE = False


def _load_kb() -> Dict:
    if os.path.exists(KB_PATH):
        with open(KB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"topics": {}}


def _save_kb(kb: Dict):
    with open(KB_PATH, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2, ensure_ascii=False)


def search_duckduckgo(query: str, max_results: int = 5) -> List[Dict]:
    if not DDGS_AVAILABLE:
        return []
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in results]
    except Exception as e:
        print(f"  DuckDuckGo search error: {e}")
        return []


def search_wikipedia(query: str) -> Optional[Dict]:
    if not WIKI_AVAILABLE:
        return None
    try:
        wiki = wikipediaapi.Wikipedia(
            user_agent="ShapeLang/1.0 (shapelang-ai@example.com)",
            language="en",
        )
        page = wiki.page(query)
        if page.exists():
            return {
                "title": page.title,
                "url": page.fullurl,
                "summary": page.summary[:500] if page.summary else "",
                "text": page.text[:2000] if page.text else "",
            }
        for word in query.split():
            page = wiki.page(word.capitalize())
            if page.exists():
                return {
                    "title": page.title,
                    "url": page.fullurl,
                    "summary": page.summary[:500] if page.summary else "",
                    "text": page.text[:2000] if page.text else "",
                }
    except Exception as e:
        print(f"  Wikipedia search error: {e}")
    return None


def extract_content_from_url(url: str) -> str:
    if not TRAFILATURA_AVAILABLE:
        return ""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
            return text[:3000] if text else ""
    except Exception as e:
        print(f"  Content extraction error: {e}")
    return ""


def search_web(query: str, max_results: int = 5) -> List[Dict]:
    results = search_duckduckgo(query, max_results=max_results)
    if not results:
        wiki = search_wikipedia(query)
        if wiki:
            results.append({
                "title": wiki["title"],
                "url": wiki["url"],
                "snippet": wiki["summary"],
            })
    return results


def _parse_into_facts(text: str, topic: str) -> List[str]:
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    topic_lower = topic.lower()
    facts = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 20 or len(sent) > 300:
            continue
        if topic_lower in sent.lower():
            facts.append(sent)
    if not facts:
        for sent in sentences:
            sent = sent.strip()
            if len(sent) >= 20 and len(sent) <= 300:
                facts.append(sent)
    seen = set()
    unique_facts = []
    for fact in facts:
        normalized = fact.lower().strip()
        if normalized not in seen:
            seen.add(normalized)
            unique_facts.append(fact)
    return unique_facts[:20]


def _extract_semantic_tags(text: str, topic: str) -> List[str]:
    tag_keywords = {
        "animal": ["animal", "species", "mammal", "bird", "fish", "reptile"],
        "person": ["person", "human", "man", "woman", "child", "people"],
        "place": ["place", "city", "country", "location", "area", "region"],
        "food": ["food", "eat", "meal", "dish", "cuisine", "recipe"],
        "science": ["science", "research", "study", "experiment", "theory"],
        "technology": ["technology", "computer", "software", "hardware", "digital"],
        "nature": ["nature", "tree", "plant", "flower", "forest", "river"],
        "history": ["history", "ancient", "century", "war", "civilization"],
        "music": ["music", "song", "instrument", "band", "melody"],
        "art": ["art", "painting", "drawing", "sculpture", "artist"],
    }
    text_lower = text.lower()
    topic_lower = topic.lower()
    tags = []
    for tag, keywords in tag_keywords.items():
        for kw in keywords:
            if kw in text_lower or kw in topic_lower:
                tags.append(tag)
                break
    if not tags:
        tags = ["general"]
    return list(set(tags))


def learn_about(topic: str) -> Dict:
    kb = _load_kb()
    topic_lower = topic.lower().strip()

    # Skip queries with no real content words
    meaningful = [w for w in topic_lower.split() if len(w) >= 3 and w.isalpha()]
    if not meaningful:
        print(f"  Skipped (no meaningful content)")
        return {"summary": "", "facts": [], "semantic_tags": ["general"], "sources": []}

    print(f"  Searching for '{topic}'...")
    results = search_web(topic, max_results=3)

    all_text = ""
    sources = []

    for result in results:
        url = result.get("url", "")
        snippet = result.get("snippet", "")
        all_text += snippet + " "
        sources.append(url)

        if url and TRAFILATURA_AVAILABLE:
            print(f"  Extracting content from: {result.get('title', url)[:50]}...")
            content = extract_content_from_url(url)
            if content:
                all_text += content + " "

        time.sleep(0.5)

    wiki_result = search_wikipedia(topic)
    if wiki_result:
        all_text += wiki_result.get("text", "") + " "
        if wiki_result["url"] not in sources:
            sources.append(wiki_result["url"])

    facts = _parse_into_facts(all_text, topic)
    semantic_tags = _extract_semantic_tags(all_text, topic)

    summary = ""
    if wiki_result and wiki_result.get("summary"):
        summary = wiki_result["summary"]
    elif facts:
        summary = facts[0]

    entry = {
        "learned_at": datetime.now().isoformat(),
        "sources": sources[:10],
        "facts": facts,
        "summary": summary,
        "semantic_tags": semantic_tags,
    }

    kb["topics"][topic_lower] = entry
    _save_kb(kb)

    print(f"  Learned {len(facts)} facts about '{topic}'")
    print(f"  Semantic tags: {', '.join(semantic_tags)}")
    print(f"  Sources: {len(sources)}")

    return entry


def get_knowledge(topic: str) -> Optional[Dict]:
    kb = _load_kb()
    return kb["topics"].get(topic.lower().strip())


def answer_with_knowledge(question: str) -> str:
    question_lower = question.lower()

    identity_patterns = [
        "what are you", "who are you", "what is your name",
        "who is this", "what can you do", "how do you work",
    ]
    for pattern in identity_patterns:
        if pattern in question_lower:
            return (
                "I am the Symbol Grounding Network (SGN), an AI that learns language "
                "from the visual shapes of alphabet letters. I use a MobileNetV3-Large "
                "CNN to recognize letters from 7x7 binary grids, extract 19 topological "
                "features, and fuse them with learned CNN features to understand and "
                "generate sentences from a 184-word vocabulary. I can also search the "
                "web via DuckDuckGo and Wikipedia to learn new topics. My architecture "
                "has ~2.05M parameters across 6 inverted-bottleneck blocks with "
                "Squeeze-and-Excitation and h-swish activations."
            )

    kb = _load_kb()

    best_topic = None
    best_score = 0
    for topic in kb["topics"]:
        if topic in question_lower or question_lower in topic:
            score = len(topic)
            if score > best_score:
                best_score = score
                best_topic = topic

    if not best_topic:
        words = question_lower.split()
        for topic in kb["topics"]:
            topic_words = topic.split()
            overlap = len(set(words) & set(topic_words))
            if overlap > best_score:
                best_score = overlap
                best_topic = topic

    if best_topic:
        entry = kb["topics"][best_topic]
        facts = entry.get("facts", [])
        summary = entry.get("summary", "")
        tags = entry.get("semantic_tags", [])

        response_parts = []
        if summary:
            response_parts.append(summary)
        for fact in facts[:3]:
            if fact not in summary:
                response_parts.append(fact)
        if tags:
            response_parts.append(f"Categories: {', '.join(tags)}")

        return " ".join(response_parts) if response_parts else f"I know about {best_topic} but have limited information."

    return "I have not learned about that topic yet. Use 'learn <topic>' to teach me."


def forget_topic(topic: str) -> bool:
    kb = _load_kb()
    key = topic.lower().strip()
    if key in kb["topics"]:
        del kb["topics"][key]
        _save_kb(kb)
        return True
    return False


def list_topics() -> List[str]:
    kb = _load_kb()
    return sorted(kb["topics"].keys())


def get_topic_summary(topic: str) -> str:
    entry = get_knowledge(topic)
    if entry is None:
        return f"No knowledge about '{topic}'"
    lines = [
        f"Topic: {topic}",
        f"Learned: {entry.get('learned_at', 'unknown')}",
        f"Facts: {len(entry.get('facts', []))}",
        f"Tags: {', '.join(entry.get('semantic_tags', []))}",
        f"Sources: {len(entry.get('sources', []))}",
    ]
    summary = entry.get("summary", "")
    if summary:
        lines.append(f"Summary: {summary[:200]}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Web Learner module loaded")
    print(f"  DuckDuckGo: {'available' if DDGS_AVAILABLE else 'not installed'}")
    print(f"  Wikipedia: {'available' if WIKI_AVAILABLE else 'not installed'}")
    print(f"  Trafilatura: {'available' if TRAFILATURA_AVAILABLE else 'not installed'}")
    print(f"  Knowledge base: {KB_PATH}")
    print()

    test_topic = "cats"
    print(f"Learning about {test_topic}...")
    result = learn_about(test_topic)
    print(f"\nResult:")
    print(get_topic_summary(test_topic))

    print(f"\nAnswering: 'what are cats'")
    print(answer_with_knowledge("what are cats"))
