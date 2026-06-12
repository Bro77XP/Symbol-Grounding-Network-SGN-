import sys
import numpy as np
import torch

from letter_grids import get_all_grids, get_grid, display_grid, text_to_grid, grid_to_text
from feature_extractor import extract_features, FEATURE_NAMES
from letter_model import (
    ShapeLangModel, load_letter_model, train_letter_model,
    recognize_letter_from_grid, recognize_word_from_grids,
    LETTERS, MODEL_PATH,
)
from language_engine import (
    VOCAB, WORD_BY_POS, VOCAB_SIZE,
    generate_sentence, parse_sentence, get_meaning,
    format_parse_tree, respond_to_sentence,
    sentence_to_ids, ids_to_sentence,
)
from web_learner import (
    learn_about, get_knowledge, answer_with_knowledge,
    forget_topic, list_topics, get_topic_summary,
)


BANNER = r"""
 Symbol Grounding Network (SGN)
   __  ____     ____   _   _  __  
  / / / ___|   / ___| | \ | | \ \ 
 | |  \___ \  | |  _  |  \| |  | |
 | |   ___) | | |_| | | |\  |  | |
 | |  |____/   \____| |_| \_|  | |
  \_\                         /_/
  learning Language from Letter Shapes — v1.0
  Advanced Deep Learning + Web Learning
"""

COMMANDS = {
    "train":    "Train the model (train [epochs] [variants_per_letter])",
    "show":     "Show grid + features for a letter (show <letter>)",
    "read":     "Enter letters one-by-letter and build a word",
    "word":     "Spell a word directly (word <word>)",
    "recognize":"Show how the AI sees a word (recognize <word>)",
    "speak":    "AI understands and responds (speak <sentence>)",
    "generate": "AI generates a random sentence",
    "parse":    "Parse a sentence into a tree (parse <sentence>)",
    "learn":    "Search the web and learn about a topic (learn <topic>)",
    "ask":      "Answer from learned knowledge (ask <question>)",
    "topics":   "List all learned topics",
    "forget":   "Remove a topic (forget <topic>)",
    "vocab":    "List all vocabulary words",
    "features": "Show feature names",
    "status":   "Show system status",
    "help":     "Show this help message",
    "quit":     "Exit the program",
}


model = None
device = "cpu"


def cmd_train(args=None):
    global model
    epochs = 200
    variants = 200
    if args:
        try:
            epochs = int(args[0])
        except (IndexError, ValueError):
            pass
        try:
            variants = int(args[1])
        except (IndexError, ValueError):
            pass
    print("Training the letter recognition neural network...")
    print(f"Architecture: MobileNetV3-Large (6-block, 2.05M params) + Feature Fusion + Multi-Task Heads")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}, Variants per letter: {variants}\n")
    model = train_letter_model(variants_per_letter=variants, epochs=epochs, device=device)
    print("\nTraining complete! Model ready for recognition.")


def cmd_show(args):
    if not args:
        print("Usage: show <letter>")
        return
    letter = args[0].upper()
    if letter not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        print(f"Invalid letter: {letter}")
        return
    grid = get_grid(letter)
    features = extract_features(grid)

    print(f"\n  Letter: {letter}")
    print(f"  Grid:")
    for line in display_grid(grid).split("\n"):
        print(f"    {line}")
    print(f"\n  Features:")
    for name, val in zip(FEATURE_NAMES, features):
        print(f"    {name:25s}: {val:.4f}")

    if model:
        pred, conf = recognize_letter_from_grid(grid, model, device)
        print(f"\n  Model prediction: {pred} (confidence: {conf:.1%})")


def cmd_read():
    global model
    print("Enter letters one at a time.")
    print("Type a letter name (e.g. 'a') or paste a 7-line binary grid.")
    print("Type 'done' when finished, 'cancel' to abort.\n")

    grids = []
    while True:
        try:
            user_input = input("  Letter: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() == "done":
            break
        if user_input.lower() == "cancel":
            print("  Cancelled.")
            return

        if not user_input:
            continue

        if len(user_input) == 1 and user_input.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            grid = get_grid(user_input.upper())
            grids.append(grid)
            if model:
                pred, conf = recognize_letter_from_grid(grid, model, device)
                print(f"  -> Recognized as: {pred} (confidence: {conf:.1%})")
            else:
                print(f"  -> Added: {user_input.upper()}")
        elif "\n" in user_input or len(user_input) >= 7:
            try:
                grid = text_to_grid(user_input)
                grids.append(grid)
                if model:
                    pred, conf = recognize_letter_from_grid(grid, model, device)
                    print(f"  -> Recognized as: {pred} (confidence: {conf:.1%})")
                else:
                    print(f"  -> Grid added")
            except Exception as e:
                print(f"  Error parsing grid: {e}")
        else:
            for ch in user_input:
                if ch.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                    grid = get_grid(ch.upper())
                    grids.append(grid)
                    if model:
                        pred, conf = recognize_letter_from_grid(grid, model, device)
                        print(f"  -> {ch.upper()} recognized as: {pred} (confidence: {conf:.1%})")
                    else:
                        print(f"  -> Added: {ch.upper()}")

    if grids:
        word = ""
        for grid in grids:
            if model:
                pred, _ = recognize_letter_from_grid(grid, model, device)
                word += pred
            else:
                idx = np.argmax(grid.sum(axis=1))
                word += "?"

        print(f"\n  Word: {word}")
        word_lower = word.lower()
        if word_lower in VOCAB:
            info = VOCAB[word_lower]
            print(f"  Meaning: {info['meaning']}")
            if "semantic" in info:
                print(f"  Tags: {', '.join(info['semantic'])}")
        else:
            print(f"  (Not in vocabulary)")
    else:
        print("  No letters entered.")


def cmd_word(args):
    if not args:
        print("Usage: word <word>")
        return
    word = args[0].upper()
    print(f"\n  Word: {word}")
    for i, ch in enumerate(word):
        if ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            grid = get_grid(ch)
            if model:
                pred, conf = recognize_letter_from_grid(grid, model, device)
                print(f"    Letter {i+1}: {ch} -> {pred} (confidence: {conf:.1%})")
            else:
                print(f"    Letter {i+1}: {ch}")

    word_lower = args[0].lower()
    if word_lower in VOCAB:
        info = VOCAB[word_lower]
        print(f"\n  Meaning: {info['meaning']}")
        print(f"  POS: {info['pos']}")
        if "semantic" in info:
            print(f"  Tags: {', '.join(info['semantic'])}")
    else:
        print(f"  (Not in vocabulary)")


def cmd_recognize(args):
    if not args:
        print("Usage: recognize <word>")
        return
    word = args[0].upper()
    print(f"\n  Recognizing: {word}\n")
    for i, ch in enumerate(word):
        if ch not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            continue
        grid = get_grid(ch)
        print(f"  ─── Letter {i+1}: '{ch}' ───")
        print(display_grid(grid))

        feats = extract_features(grid)
        print(f"  Features ({len(feats)}):")
        for name, val in zip(FEATURE_NAMES, feats):
            print(f"    {name:25s} = {val:8.4f}")

        if model:
            pred, conf = recognize_letter_from_grid(grid, model, device)
            accuracy = "✓" if pred == ch else "✗"
            print(f"  CNN predicts: '{pred}' (confidence: {conf:.1%})  {accuracy}")
        else:
            print(f"  (No model loaded)")
        print()

    word_lower = args[0].lower()
    if word_lower in VOCAB:
        info = VOCAB[word_lower]
        print(f"  Word meaning: {info['meaning']}")
        print(f"  POS: {info['pos']}")
        if "semantic" in info:
            print(f"  Semantic tags: {', '.join(info['semantic'])}")
    else:
        print(f"  Word not in vocabulary.")


def cmd_speak(args):
    if not args:
        print("Usage: speak <sentence>")
        return
    sentence = " ".join(args)
    print(f"\n  Input: {sentence}")

    tree = parse_sentence(sentence)
    if tree:
        print(f"\n  Parse tree:")
        for line in format_parse_tree(tree).split("\n"):
            print(f"    {line}")

        meaning = get_meaning(sentence)
        if meaning:
            subj = meaning.get("subject", {})
            action = meaning.get("action", {})
            obj = meaning.get("object", {})

            if subj:
                print(f"\n  Subject: {' '.join(subj.get('words', []))}")
                if subj.get("semantic"):
                    print(f"    Tags: {', '.join(subj['semantic'])}")
            if action:
                print(f"  Action: {action.get('word', '')} ({action.get('meaning', '')})")
            if obj:
                print(f"  Object: {' '.join(obj.get('words', []))}")
                if obj.get("semantic"):
                    print(f"    Tags: {', '.join(obj['semantic'])}")

        # Auto-learn topics from subject/object nouns (skip pronouns)
        PRONOUNS = {"it", "you", "they", "he", "she", "we", "me", "him", "her", "us", "them"}
        topics_to_learn = set()
        for part in [subj, obj]:
            words = part.get("words", []) if part else []
            semantic = part.get("semantic", []) if part else []
            if words and "question" not in semantic and "number" not in semantic:
                topic = " ".join(words).lower()
                if topic not in PRONOUNS and not get_knowledge(topic):
                    topics_to_learn.add(topic)

        for topic in topics_to_learn:
            print(f"\n  (Auto-learning about '{topic}'...)")
            result = learn_about(topic)
            if result.get("facts") or result.get("summary"):
                print(f"    Learned {len(result.get('facts', []))} facts")

        # Build combined KB from all recognized topics in the sentence
        all_nouns = []
        for part in [subj, obj]:
            words = part.get("words", []) if part else []
            if words:
                all_nouns.append(" ".join(words).lower())

        kb_parts = {}
        for noun in all_nouns:
            entry = get_knowledge(noun)
            if entry:
                kb_parts[noun] = entry
        if kb_parts:
            kb = {"topics": kb_parts}
        else:
            kb = {}

        response = respond_to_sentence(sentence, kb)
        print(f"\n  Response: {response}")
    else:
        print("  Could not parse that sentence.")

        # Extract meaningful keywords instead of learning the full sentence
        STOPWORDS = {"i", "a", "an", "the", "is", "are", "was", "were", "be", "being",
                     "been", "have", "has", "had", "do", "does", "did", "will", "would",
                     "could", "should", "may", "might", "can", "shall", "to", "of", "in",
                     "for", "on", "with", "at", "by", "from", "as", "into", "through",
                     "during", "before", "after", "above", "below", "between", "and",
                     "but", "or", "nor", "not", "so", "yet", "if", "because", "then",
                     "else", "when", "where", "why", "how", "what", "who", "which",
                     "this", "that", "these", "those", "it", "you", "they", "he", "she",
                     "we", "me", "him", "her", "us", "them", "my", "your", "his", "its",
                     "our", "their", "no", "yes", "all", "each", "every", "both", "some",
                     "any", "none", "one", "two", "other", "another", "much", "many",
                     "more", "most", "few", "less", "own", "same", "just", "also", "very",
                     "too", "really", "quite", "then", "here", "there", "about", "up",
                     "out", "off", "over", "down", "only", "like", "well", "even"}

        raw = sentence.lower()
        PUNCT = " \"'()?!.,;:-"
        # Clean each word: strip surrounding punctuation, keep internal hyphens/apostrophes
        words = [w.strip(PUNCT) for w in raw.split()]
        words = [w for w in words if w and w.strip("'").isalpha()]
        # Filter: non-stopwords, length >= 3
        keywords = [w for w in words if w not in STOPWORDS and len(w) >= 3]

        topic_words = []
        if keywords:
            # Use known nouns first, then other meaningful keywords
            known_nouns = [w for w in keywords if w in VOCAB and VOCAB[w]["pos"] == "noun"]
            # Remove duplicates while preserving order
            seen = set()
            candidates = known_nouns if known_nouns else keywords
            topic_words = []
            for w in candidates:
                if w not in seen:
                    seen.add(w)
                    topic_words.append(w)
                if len(topic_words) >= 3:
                    break
        elif len(raw) >= 3 and len(raw) <= 40:
            topic_words = [raw]

        for topic in topic_words:
            if not get_knowledge(topic):
                print(f"  (Auto-learning about '{topic}'...)")
                result = learn_about(topic)
                if result.get("facts") or result.get("summary"):
                    print(f"    Learned {len(result.get('facts', []))} facts")

        kb_parts = {}
        for topic in topic_words:
            entry = get_knowledge(topic)
            if entry:
                kb_parts[topic] = entry
        kb = {"topics": kb_parts} if kb_parts else {}
        response = respond_to_sentence(sentence, kb)
        print(f"\n  Response: {response}")


def cmd_generate():
    print()
    for i in range(3):
        sentence = generate_sentence()
        print(f"  {i+1}. {sentence}")
    print()


def cmd_parse(args):
    if not args:
        print("Usage: parse <sentence>")
        return
    sentence = " ".join(args)
    tree = parse_sentence(sentence)
    if tree:
        print(f"\n  Parse tree:")
        for line in format_parse_tree(tree).split("\n"):
            print(f"    {line}")

        meaning = get_meaning(sentence)
        if meaning:
            print(f"\n  Meaning:")
            if meaning.get("subject"):
                print(f"    Subject: {meaning['subject']}")
            if meaning.get("action"):
                print(f"    Action: {meaning['action']}")
            if meaning.get("object"):
                print(f"    Object: {meaning['object']}")
    else:
        print("  Could not parse that sentence.")


def cmd_learn(args):
    if not args:
        print("Usage: learn <topic>")
        return
    topic = " ".join(args)
    print()
    result = learn_about(topic)
    if result.get("facts"):
        print(f"\n  Facts learned:")
        for i, fact in enumerate(result["facts"][:5], 1):
            print(f"    {i}. {fact}")
    if result.get("summary"):
        print(f"\n  Summary: {result['summary'][:200]}")


def cmd_ask(args):
    if not args:
        print("Usage: ask <question>")
        return
    question = " ".join(args)
    print()
    answer = answer_with_knowledge(question)
    print(f"  {answer}")


def cmd_topics():
    topics = list_topics()
    if topics:
        print(f"\n  Learned topics ({len(topics)}):")
        for t in topics:
            entry = get_knowledge(t)
            n_facts = len(entry.get("facts", [])) if entry else 0
            print(f"    - {t} ({n_facts} facts)")
    else:
        print("  No topics learned yet. Use 'learn <topic>' to start.")


def cmd_forget(args):
    if not args:
        print("Usage: forget <topic>")
        return
    topic = " ".join(args)
    if forget_topic(topic):
        print(f"  Forgot: {topic}")
    else:
        print(f"  Topic not found: {topic}")


def cmd_vocab():
    print(f"\n  Vocabulary: {len(VOCAB)} words")
    for pos in ["noun", "verb", "adj", "det", "pronoun", "prep", "conj"]:
        words = WORD_BY_POS.get(pos, [])
        if words:
            print(f"\n  {pos.upper()}S ({len(words)}):")
            for i in range(0, len(words), 6):
                chunk = words[i:i+6]
                print(f"    {', '.join(chunk)}")


def cmd_features():
    print(f"\n  Feature names ({len(FEATURE_NAMES)}):")
    for i, name in enumerate(FEATURE_NAMES, 1):
        print(f"    {i:2d}. {name}")


def cmd_status():
    print(f"\n  System Status:")
    print(f"    Model loaded: {'yes' if model else 'no'}")
    print(f"    Model path: {MODEL_PATH}")
    print(f"    Device: {device}")
    print(f"    Vocabulary: {len(VOCAB)} words")
    print(f"    Token vocab: {VOCAB_SIZE} tokens")
    print(f"    Letters: {len(LETTERS)} (A-Z)")
    topics = list_topics()
    print(f"    Learned topics: {len(topics)}")
    if topics:
        print(f"      {', '.join(topics[:10])}")
    print(f"    Letter grids: 26 (7x7)")
    print(f"    Features: {len(FEATURE_NAMES)}")


def cmd_help():
    print(f"\n  Commands:")
    for cmd, desc in COMMANDS.items():
        print(f"    {cmd:12s} — {desc}")
    print()


def main():
    global model, device

    print(BANNER)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    import os
    if os.path.exists(MODEL_PATH):
        model = load_letter_model(device)
    else:
        print(f"  No trained model found. Run 'train' to create one.\n")

    print(f"  Type 'help' for commands.\n")

    while True:
        try:
            user_input = input("shapelang> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye!")
            break

        if not user_input:
            continue

        parts = user_input.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "quit" or cmd == "exit" or cmd == "q":
            print("  Goodbye!")
            break
        elif cmd == "train":
            cmd_train(args)
        elif cmd == "show":
            cmd_show(args)
        elif cmd == "read":
            cmd_read()
        elif cmd == "word":
            cmd_word(args)
        elif cmd == "recognize":
            cmd_recognize(args)
        elif cmd == "speak":
            cmd_speak(args)
        elif cmd == "generate":
            cmd_generate()
        elif cmd == "parse":
            cmd_parse(args)
        elif cmd == "learn":
            cmd_learn(args)
        elif cmd == "ask":
            cmd_ask(args)
        elif cmd == "topics":
            cmd_topics()
        elif cmd == "forget":
            cmd_forget(args)
        elif cmd == "vocab":
            cmd_vocab()
        elif cmd == "features":
            cmd_features()
        elif cmd == "status":
            cmd_status()
        elif cmd == "help":
            cmd_help()
        else:
            print(f"  Unknown command: {cmd}. Type 'help' for commands.")


if __name__ == "__main__":
    main()
