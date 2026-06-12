import random
from typing import Dict, List, Optional, Tuple

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [SOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN]

VOCAB: Dict[str, Dict] = {
    "cat": {"pos": "noun", "meaning": "animal", "semantic": ["furry", "small", "pet"]},
    "dog": {"pos": "noun", "meaning": "animal", "semantic": ["loyal", "pet"]},
    "bird": {"pos": "noun", "meaning": "animal", "semantic": ["fly", "wing"]},
    "fish": {"pos": "noun", "meaning": "animal", "semantic": ["swim", "water"]},
    "horse": {"pos": "noun", "meaning": "animal", "semantic": ["ride", "fast"]},
    "lion": {"pos": "noun", "meaning": "animal", "semantic": ["wild", "king"]},
    "tiger": {"pos": "noun", "meaning": "animal", "semantic": ["wild", "stripes"]},
    "elephant": {"pos": "noun", "meaning": "animal", "semantic": ["large", "trunk"]},
    "rabbit": {"pos": "noun", "meaning": "animal", "semantic": ["fast", "ears"]},
    "bear": {"pos": "noun", "meaning": "animal", "semantic": ["large", "wild"]},
    "wolf": {"pos": "noun", "meaning": "animal", "semantic": ["wild", "pack"]},
    "fox": {"pos": "noun", "meaning": "animal", "semantic": ["clever", "wild"]},
    "deer": {"pos": "noun", "meaning": "animal", "semantic": ["gentle", "forest"]},
    "frog": {"pos": "noun", "meaning": "animal", "semantic": ["small", "hop"]},
    "snake": {"pos": "noun", "meaning": "animal", "semantic": ["long", "slither"]},
    "boy": {"pos": "noun", "meaning": "person", "semantic": ["male", "young"]},
    "girl": {"pos": "noun", "meaning": "person", "semantic": ["female", "young"]},
    "man": {"pos": "noun", "meaning": "person", "semantic": ["male", "adult"]},
    "woman": {"pos": "noun", "meaning": "person", "semantic": ["female", "adult"]},
    "friend": {"pos": "noun", "meaning": "person", "semantic": ["close", "trust"]},
    "teacher": {"pos": "noun", "meaning": "person", "semantic": ["teach", "school"]},
    "student": {"pos": "noun", "meaning": "person", "semantic": ["learn", "school"]},
    "baby": {"pos": "noun", "meaning": "person", "semantic": ["small", "new"]},
    "king": {"pos": "noun", "meaning": "person", "semantic": ["ruler", "power"]},
    "queen": {"pos": "noun", "meaning": "person", "semantic": ["ruler", "female"]},
    "house": {"pos": "noun", "meaning": "building", "semantic": ["shelter"]},
    "tree": {"pos": "noun", "meaning": "plant", "semantic": ["nature", "tall"]},
    "flower": {"pos": "noun", "meaning": "plant", "semantic": ["colorful", "fragrant"]},
    "car": {"pos": "noun", "meaning": "vehicle", "semantic": ["transport"]},
    "ball": {"pos": "noun", "meaning": "object", "semantic": ["round", "play"]},
    "book": {"pos": "noun", "meaning": "object", "semantic": ["read", "paper"]},
    "food": {"pos": "noun", "meaning": "substance", "semantic": ["eat"]},
    "water": {"pos": "noun", "meaning": "substance", "semantic": ["drink", "liquid"]},
    "sun": {"pos": "noun", "meaning": "celestial", "semantic": ["bright", "hot"]},
    "moon": {"pos": "noun", "meaning": "celestial", "semantic": ["night", "light"]},
    "sky": {"pos": "noun", "meaning": "nature", "semantic": ["blue", "high"]},
    "river": {"pos": "noun", "meaning": "nature", "semantic": ["water", "flow"]},
    "mountain": {"pos": "noun", "meaning": "nature", "semantic": ["tall", "rock"]},
    "park": {"pos": "noun", "meaning": "place", "semantic": ["outdoor", "play"]},
    "school": {"pos": "noun", "meaning": "place", "semantic": ["learn", "building"]},
    "city": {"pos": "noun", "meaning": "place", "semantic": ["large", "urban"]},
    "road": {"pos": "noun", "meaning": "object", "semantic": ["path", "travel"]},
    "door": {"pos": "noun", "meaning": "object", "semantic": ["open", "enter"]},
    "window": {"pos": "noun", "meaning": "object", "semantic": ["see", "light"]},
    "table": {"pos": "noun", "meaning": "object", "semantic": ["furniture", "flat"]},
    "stone": {"pos": "noun", "meaning": "object", "semantic": ["hard", "rock"]},
    "fire": {"pos": "noun", "meaning": "element", "semantic": ["hot", "light"]},
    "wind": {"pos": "noun", "meaning": "nature", "semantic": ["air", "move"]},
    "rain": {"pos": "noun", "meaning": "nature", "semantic": ["water", "fall"]},
    "star": {"pos": "noun", "meaning": "celestial", "semantic": ["bright", "night"]},
    "chased": {"pos": "verb", "meaning": "action", "semantic": ["movement", "pursue"]},
    "saw": {"pos": "verb", "meaning": "perception", "semantic": ["vision", "notice"]},
    "kicked": {"pos": "verb", "meaning": "action", "semantic": ["foot", "force"]},
    "followed": {"pos": "verb", "meaning": "action", "semantic": ["movement", "behind"]},
    "ate": {"pos": "verb", "meaning": "action", "semantic": ["consume", "food"]},
    "drank": {"pos": "verb", "meaning": "action", "semantic": ["consume", "liquid"]},
    "ran": {"pos": "verb", "meaning": "action", "semantic": ["fast", "movement"]},
    "walked": {"pos": "verb", "meaning": "action", "semantic": ["slow", "movement"]},
    "jumped": {"pos": "verb", "meaning": "action", "semantic": ["air", "up"]},
    "sat": {"pos": "verb", "meaning": "action", "semantic": ["rest", "down"]},
    "played": {"pos": "verb", "meaning": "action", "semantic": ["fun", "activity"]},
    "slept": {"pos": "verb", "meaning": "state", "semantic": ["rest", "night"]},
    "woke": {"pos": "verb", "meaning": "action", "semantic": ["morning", "rise"]},
    "found": {"pos": "verb", "meaning": "action", "semantic": ["discover"]},
    "lost": {"pos": "verb", "meaning": "state", "semantic": ["missing"]},
    "liked": {"pos": "verb", "meaning": "emotion", "semantic": ["positive"]},
    "loved": {"pos": "verb", "meaning": "emotion", "semantic": ["strong", "positive"]},
    "looked": {"pos": "verb", "meaning": "perception", "semantic": ["vision"]},
    "listened": {"pos": "verb", "meaning": "perception", "semantic": ["hearing"]},
    "thought": {"pos": "verb", "meaning": "mental", "semantic": ["mind"]},
    "knew": {"pos": "verb", "meaning": "mental", "semantic": ["knowledge"]},
    "wanted": {"pos": "verb", "meaning": "desire", "semantic": ["wish"]},
    "needed": {"pos": "verb", "meaning": "desire", "semantic": ["require"]},
    "gave": {"pos": "verb", "meaning": "action", "semantic": ["transfer"]},
    "took": {"pos": "verb", "meaning": "action", "semantic": ["receive"]},
    "made": {"pos": "verb", "meaning": "action", "semantic": ["create"]},
    "went": {"pos": "verb", "meaning": "action", "semantic": ["movement"]},
    "came": {"pos": "verb", "meaning": "action", "semantic": ["approach"]},
    "said": {"pos": "verb", "meaning": "communication", "semantic": ["speak"]},
    "heard": {"pos": "verb", "meaning": "perception", "semantic": ["hearing"]},
    "helped": {"pos": "verb", "meaning": "action", "semantic": ["assist"]},
    "called": {"pos": "verb", "meaning": "communication", "semantic": ["summon"]},
    "tried": {"pos": "verb", "meaning": "action", "semantic": ["attempt"]},
    "asked": {"pos": "verb", "meaning": "communication", "semantic": ["question"]},
    "worked": {"pos": "verb", "meaning": "action", "semantic": ["labor"]},
    "lived": {"pos": "verb", "meaning": "state", "semantic": ["exist"]},
    "believed": {"pos": "verb", "meaning": "mental", "semantic": ["trust"]},
    "seemed": {"pos": "verb", "meaning": "state", "semantic": ["appear"]},
    "felt": {"pos": "verb", "meaning": "perception", "semantic": ["touch"]},
    "got": {"pos": "verb", "meaning": "action", "semantic": ["receive"]},
    "let": {"pos": "verb", "meaning": "action", "semantic": ["allow"]},
    "kept": {"pos": "verb", "meaning": "action", "semantic": ["retain"]},
    "began": {"pos": "verb", "meaning": "action", "semantic": ["start"]},
    "seemed": {"pos": "verb", "meaning": "state", "semantic": ["appear"]},
    "big": {"pos": "adj", "meaning": "size", "semantic": ["large"]},
    "small": {"pos": "adj", "meaning": "size", "semantic": ["tiny"]},
    "happy": {"pos": "adj", "meaning": "emotion", "semantic": ["positive", "joy"]},
    "sad": {"pos": "adj", "meaning": "emotion", "semantic": ["negative", "unhappy"]},
    "fast": {"pos": "adj", "meaning": "speed", "semantic": ["quick"]},
    "slow": {"pos": "adj", "meaning": "speed", "semantic": ["leisurely"]},
    "tall": {"pos": "adj", "meaning": "size", "semantic": ["high"]},
    "short": {"pos": "adj", "meaning": "size", "semantic": ["low"]},
    "young": {"pos": "adj", "meaning": "age", "semantic": ["new", "fresh"]},
    "old": {"pos": "adj", "meaning": "age", "semantic": ["aged", "wise"]},
    "good": {"pos": "adj", "meaning": "quality", "semantic": ["positive"]},
    "bad": {"pos": "adj", "meaning": "quality", "semantic": ["negative"]},
    "new": {"pos": "adj", "meaning": "age", "semantic": ["fresh"]},
    "hot": {"pos": "adj", "meaning": "temperature", "semantic": ["warm"]},
    "cold": {"pos": "adj", "meaning": "temperature", "semantic": ["cool"]},
    "bright": {"pos": "adj", "meaning": "light", "semantic": ["shiny"]},
    "dark": {"pos": "adj", "meaning": "light", "semantic": ["dim"]},
    "loud": {"pos": "adj", "meaning": "sound", "semantic": ["noisy"]},
    "quiet": {"pos": "adj", "meaning": "sound", "semantic": ["silent"]},
    "strong": {"pos": "adj", "meaning": "power", "semantic": ["force"]},
    "weak": {"pos": "adj", "meaning": "power", "semantic": ["feeble"]},
    "cute": {"pos": "adj", "meaning": "appearance", "semantic": ["adorable"]},
    "ugly": {"pos": "adj", "meaning": "appearance", "semantic": ["unpleasant"]},
    "clean": {"pos": "adj", "meaning": "state", "semantic": ["tidy"]},
    "dirty": {"pos": "adj", "meaning": "state", "semantic": ["messy"]},
    "gentle": {"pos": "adj", "meaning": "nature", "semantic": ["soft", "kind"]},
    "wild": {"pos": "adj", "meaning": "nature", "semantic": ["untamed"]},
    "brave": {"pos": "adj", "meaning": "courage", "semantic": ["fearless"]},
    "clever": {"pos": "adj", "meaning": "mind", "semantic": ["smart"]},
    "the": {"pos": "det", "meaning": "determiner"},
    "a": {"pos": "det", "meaning": "determiner"},
    "an": {"pos": "det", "meaning": "determiner"},
    "this": {"pos": "det", "meaning": "determiner"},
    "that": {"pos": ["det", "pronoun"], "meaning": "determiner"},
    "he": {"pos": "pronoun", "meaning": "person", "semantic": ["male"]},
    "she": {"pos": "pronoun", "meaning": "person", "semantic": ["female"]},
    "it": {"pos": "pronoun", "meaning": "thing"},
    "they": {"pos": "pronoun", "meaning": "person", "semantic": ["plural"]},
    "we": {"pos": "pronoun", "meaning": "person", "semantic": ["inclusive"]},
    "i": {"pos": "pronoun", "meaning": "person", "semantic": ["self"]},
    "you": {"pos": "pronoun", "meaning": "person", "semantic": ["second"]},
    "me": {"pos": "pronoun", "meaning": "person", "semantic": ["self"]},
    "him": {"pos": "pronoun", "meaning": "person", "semantic": ["male"]},
    "her": {"pos": "pronoun", "meaning": "person", "semantic": ["female"]},
    "us": {"pos": "pronoun", "meaning": "person", "semantic": ["inclusive"]},
    "them": {"pos": "pronoun", "meaning": "person", "semantic": ["plural"]},
    "what": {"pos": "qword", "meaning": "question"},
    "who": {"pos": "qword", "meaning": "question"},
    "why": {"pos": "qword", "meaning": "question"},
    "how": {"pos": "qword", "meaning": "question"},
    "where": {"pos": "qword", "meaning": "question"},
    "when": {"pos": "qword", "meaning": "question"},
    "is": {"pos": "verb", "meaning": "state", "semantic": ["be"]},
    "are": {"pos": "verb", "meaning": "state", "semantic": ["be"]},
    "am": {"pos": "verb", "meaning": "state", "semantic": ["be"]},
    "was": {"pos": "verb", "meaning": "state", "semantic": ["be"]},
    "were": {"pos": "verb", "meaning": "state", "semantic": ["be"]},
    "in": {"pos": "prep", "meaning": "location", "semantic": ["inside"]},
    "on": {"pos": "prep", "meaning": "location", "semantic": ["surface"]},
    "with": {"pos": "prep", "meaning": "accompaniment"},
    "to": {"pos": "prep", "meaning": "direction"},
    "from": {"pos": "prep", "meaning": "origin"},
    "at": {"pos": "prep", "meaning": "location"},
    "by": {"pos": "prep", "meaning": "agent"},
    "near": {"pos": "prep", "meaning": "location", "semantic": ["close"]},
    "under": {"pos": "prep", "meaning": "location", "semantic": ["below"]},
    "over": {"pos": "prep", "meaning": "location", "semantic": ["above"]},
    "and": {"pos": "conj", "meaning": "connect"},
    "but": {"pos": "conj", "meaning": "contrast"},
    "or": {"pos": "conj", "meaning": "alternative"},
    "because": {"pos": "conj", "meaning": "reason"},
    "so": {"pos": "conj", "meaning": "result"},
    "did": {"pos": "aux", "meaning": "past"},
    "do": {"pos": "aux", "meaning": "present"},
    "does": {"pos": "aux", "meaning": "present"},
    "can": {"pos": "aux", "meaning": "ability"},
    "will": {"pos": "aux", "meaning": "future"},
    "may": {"pos": "aux", "meaning": "permission"},
    "must": {"pos": "aux", "meaning": "necessity"},
    "run": {"pos": "verb", "meaning": "action", "semantic": ["fast", "movement"]},
    "see": {"pos": "verb", "meaning": "perception", "semantic": ["vision", "notice"]},
    "eat": {"pos": "verb", "meaning": "action", "semantic": ["consume", "food"]},
    "drink": {"pos": "verb", "meaning": "action", "semantic": ["consume", "liquid"]},
    "make": {"pos": "verb", "meaning": "action", "semantic": ["create"]},
    "take": {"pos": "verb", "meaning": "action", "semantic": ["receive"]},
    "find": {"pos": "verb", "meaning": "action", "semantic": ["discover"]},
    "help": {"pos": "verb", "meaning": "action", "semantic": ["assist"]},
}

WORD_TO_ID = {}
ID_TO_WORD = {}

for i, token in enumerate(SPECIAL_TOKENS):
    WORD_TO_ID[token] = i
    ID_TO_WORD[i] = token

for word in sorted(VOCAB.keys()):
    idx = len(SPECIAL_TOKENS) + len(WORD_TO_ID) - len(SPECIAL_TOKENS)
    WORD_TO_ID[word] = len(WORD_TO_ID)
    ID_TO_WORD[len(ID_TO_WORD)] = word

WORD_TO_ID = {SOS_TOKEN: 0, EOS_TOKEN: 1, PAD_TOKEN: 2, UNK_TOKEN: 3}
ID_TO_WORD = {0: SOS_TOKEN, 1: EOS_TOKEN, 2: PAD_TOKEN, 3: UNK_TOKEN}
for i, word in enumerate(sorted(VOCAB.keys())):
    WORD_TO_ID[word] = i + len(SPECIAL_TOKENS)
    ID_TO_WORD[i + len(SPECIAL_TOKENS)] = word

VOCAB_SIZE = len(WORD_TO_ID)


def word_to_id(word: str) -> int:
    return WORD_TO_ID.get(word.lower(), WORD_TO_ID[UNK_TOKEN])


def id_to_word(idx: int) -> str:
    return ID_TO_WORD.get(idx, UNK_TOKEN)


def sentence_to_ids(sentence: str) -> List[int]:
    return [word_to_id(w) for w in sentence.lower().split()]


def ids_to_sentence(ids: List[int]) -> str:
    words = [id_to_word(i) for i in ids if i not in (WORD_TO_ID[PAD_TOKEN], WORD_TO_ID[SOS_TOKEN], WORD_TO_ID[EOS_TOKEN])]
    return " ".join(words)


PRODUCTIONS = {
    "S": [
        (["NP", "VP"], 1.0),
        (["S", "conj", "S"], 0.3),
        (["qword", "VP"], 0.3),
        (["qword", "VP", "NP"], 0.3),
        (["qword", "aux", "NP", "V"], 0.2),
        (["qword", "aux", "NP", "V", "NP"], 0.2),
    ],
    "NP": [
        (["det", "N"], 1.0),
        (["det", "adj", "N"], 0.8),
        (["pronoun"], 0.5),
        (["N"], 0.4),
        (["NP", "prep", "NP"], 0.2),
    ],
    "VP": [
        (["V"], 0.5),
        (["V", "NP"], 1.0),
        (["V", "NP", "PP"], 0.4),
        (["V", "PP"], 0.3),
    ],
    "PP": [
        (["prep", "NP"], 1.0),
    ],
}

WORD_BY_POS = {}
for word, info in VOCAB.items():
    pos = info["pos"]
    if isinstance(pos, list):
        for p in pos:
            if p not in WORD_BY_POS:
                WORD_BY_POS[p] = []
            WORD_BY_POS[p].append(word)
    else:
        if pos not in WORD_BY_POS:
            WORD_BY_POS[pos] = []
        WORD_BY_POS[pos].append(word)

POS_MAP = {
    "det": "det",
    "N": "noun",
    "V": "verb",
    "adj": "adj",
    "prep": "prep",
    "pronoun": "pronoun",
    "conj": "conj",
    "qword": "qword",
}


def _generate_symbol(symbol: str, cfactor: float = 0.25, depth: int = 0) -> List[str]:
    if symbol in WORD_BY_POS or symbol in WORD_TO_ID:
        return [symbol]
    if symbol not in PRODUCTIONS:
        return [symbol]

    productions = PRODUCTIONS[symbol]
    weights = []
    adjusted_weights = []
    for prod, base_weight in productions:
        adjusted = base_weight * (cfactor ** depth)
        weights.append(base_weight)
        adjusted_weights.append(adjusted)

    total = sum(adjusted_weights)
    probs = [w / total for w in adjusted_weights]
    idx = random.choices(range(len(productions)), weights=probs, k=1)[0]

    result = []
    for sub_symbol in productions[idx][0]:
        result.extend(_generate_symbol(sub_symbol, cfactor, depth + 1))
    return result


VERB_VALENCY = {
    "looked": "at", "listened": "to", "went": "to", "came": "to",
    "gave": "to", "said": "to", "asked": "to", "talked": "to",
    "shouted": "at", "glanced": "at", "stared": "at",
    "pointed": "at", "smiled": "at", "laughed": "at",
    "arrived": "at", "departed": "from", "escaped": "from",
    "fell": "from", "ran": "from", "hid": "from",
    "look": "at", "listen": "to", "go": "to", "come": "to",
    "ask": "to", "talk": "to",
}
DRINKABLE = ["water", "milk", "juice", "tea", "coffee"]
READABLE = ["book", "letter", "note", "newspaper", "magazine"]
EDIBLE = ["food", "apple", "bread", "meat", "fish", "cake"]
CONSUMPTION_VERBS = ["ate", "drank", "cooked", "eat", "drink"]
VISION_VERBS = ["saw", "looked", "watched", "observed"]

SEMANTIC_THEMES = {
    "animal": {
        "nouns": ["cat", "dog", "bird", "fish", "horse", "lion", "tiger", "elephant", "rabbit", "bear", "wolf", "fox", "deer", "frog", "snake"],
        "verbs": ["chased", "ran", "jumped", "ate", "drank", "slept", "looked", "played", "followed", "liked", "loved", "found", "lost", "wanted", "tried"],
        "dets": ["the", "a"],
        "adjs": ["big", "small", "fast", "slow", "young", "old", "wild", "strong", "cute", "brave", "clever", "gentle"],
    },
    "person": {
        "nouns": ["boy", "girl", "man", "woman", "friend", "teacher", "student", "baby", "king", "queen"],
        "verbs": ["chased", "saw", "see", "kicked", "followed", "ate", "eat", "drank", "drink", "ran", "walked", "jumped", "sat", "played", "slept", "woke", "found", "find", "lost", "liked", "loved", "looked", "listened", "thought", "knew", "wanted", "needed", "gave", "took", "take", "made", "make", "went", "came", "said", "heard", "helped", "help", "called", "tried", "asked", "worked", "believed"],
        "dets": ["the", "a", "this", "that"],
        "adjs": ["big", "small", "happy", "sad", "fast", "slow", "tall", "short", "young", "old", "good", "bad", "strong", "weak", "brave", "clever", "gentle", "loud", "quiet"],
    },
    "nature": {
        "nouns": ["tree", "flower", "river", "mountain", "sky", "sun", "moon", "star", "wind", "rain", "fire"],
        "verbs": ["saw", "looked", "liked", "loved", "found", "wanted"],
        "dets": ["the", "a"],
        "adjs": ["big", "small", "tall", "old", "bright", "dark", "hot", "cold", "quiet", "wild"],
    },
    "object": {
        "nouns": ["house", "car", "ball", "book", "door", "window", "table", "stone", "road"],
        "verbs": ["saw", "liked", "loved", "found", "lost", "wanted", "needed", "took", "gave", "kicked", "helped", "called"],
        "dets": ["the", "a", "this", "that"],
        "adjs": ["big", "small", "new", "old", "good", "bad", "clean", "dirty", "strong", "weak", "hot", "cold", "bright", "dark"],
    },
}

ALL_NOUNS = sum([v["nouns"] for v in SEMANTIC_THEMES.values()], [])
ALL_VERBS = sum([v["verbs"] for v in SEMANTIC_THEMES.values()], [])


def generate_coherent_sentence() -> str:
    theme = random.choice(["animal", "person", "nature", "object", "animal", "person"])
    t = SEMANTIC_THEMES[theme]

    subj_words = []
    if t["dets"] and random.random() < 0.7:
        subj_words.append(random.choice(t["dets"]))
    if t["adjs"] and random.random() < 0.25:
        subj_words.append(random.choice(t["adjs"]))
    subj_words.append(random.choice(t["nouns"]))

    verb = random.choice(t["verbs"])
    needs_prep = VERB_VALENCY.get(verb)
    is_consumption = verb in CONSUMPTION_VERBS
    is_vision = verb in VISION_VERBS

    if random.random() < 0.65:
        if is_consumption:
            obj_words = [random.choice(["the", "some"]), random.choice(EDIBLE + ["food", "water"])]
        elif is_vision:
            obj_theme = random.choice(["animal", "person", "nature"])
            ot = SEMANTIC_THEMES[obj_theme]
            obj_words = []
            if ot["dets"] and random.random() < 0.6:
                obj_words.append(random.choice(ot["dets"]))
            if ot["adjs"] and random.random() < 0.2:
                obj_words.append(random.choice(ot["adjs"]))
            obj_words.append(random.choice(ot["nouns"]))
        else:
            obj_theme = random.choice(["animal", "person", "nature", "object"])
            ot = SEMANTIC_THEMES[obj_theme]
            obj_words = []
            if ot["dets"] and random.random() < 0.6:
                obj_words.append(random.choice(ot["dets"]))
            if ot["adjs"] and random.random() < 0.2:
                obj_words.append(random.choice(ot["adjs"]))
            obj_words.append(random.choice(ot["nouns"]))

        if needs_prep:
            obj_words = [needs_prep] + obj_words

        pp_chance = 0.2 if needs_prep else 0.25
        if random.random() < pp_chance:
            prep = random.choice(["in", "on", "under", "near", "with"])
            pp_noun = random.choice(ALL_NOUNS)
            return " ".join(subj_words + [verb] + obj_words + [prep, "the" if random.random() < 0.5 else "a", pp_noun])

        return " ".join(subj_words + [verb] + obj_words)

    if random.random() < 0.2:
        prep = random.choice(["in", "on", "near"])
        pp_noun = random.choice(ALL_NOUNS)
        return " ".join(subj_words + [verb, prep, "the", pp_noun])

    return " ".join(subj_words + [verb])


def generate_sentence(max_depth: int = 5) -> str:
    return generate_coherent_sentence()


class SimpleParser:
    def parse(self, sentence: str) -> Optional[Dict]:
        tokens = sentence.lower().split()
        if not tokens:
            return None
        return self._parse_sentence(tokens, 0)

    def _parse_sentence(self, tokens: List[str], pos: int) -> Optional[Dict]:
        result = self._parse_question(tokens, pos)
        if result is not None:
            return result

        np_result = self._parse_np(tokens, pos)
        if np_result is None:
            return None
        np_end, np_node = np_result

        vp_result = self._parse_vp(tokens, np_end)
        if vp_result is None:
            return None
        vp_end, vp_node = vp_result

        return {
            "type": "S",
            "np": np_node,
            "vp": vp_node,
        }

    def _parse_question(self, tokens: List[str], pos: int) -> Optional[Dict]:
        if pos >= len(tokens) or tokens[pos] not in WORD_BY_POS.get("qword", []):
            return None
        qword = tokens[pos]
        pos += 1

        # try aux-inversion: qword + aux + NP + V (e.g. "why did the cat run")
        if pos < len(tokens) and tokens[pos] in WORD_BY_POS.get("aux", []):
            aux_word = tokens[pos]
            pos += 1
            np_result = self._parse_np(tokens, pos)
            if np_result:
                np_end, np_node = np_result
                if np_end < len(tokens) and tokens[np_end] in WORD_BY_POS.get("verb", []):
                    v_word = tokens[np_end]
                    np_end += 1
                    return {
                        "type": "Q",
                        "qword": qword,
                        "aux": aux_word,
                        "np": np_node,
                        "vp": {"type": "VP", "children": [{"type": "V", "word": v_word}]},
                    }

        # simple: qword + VP (e.g. "how are you", "what is that")
        vp_result = self._parse_vp(tokens, pos)
        if vp_result is None:
            return None
        vp_end, vp_node = vp_result

        return {
            "type": "Q",
            "qword": qword,
            "vp": vp_node,
        }

    def _parse_np(self, tokens: List[str], pos: int) -> Optional[Tuple[int, Dict]]:
        if pos >= len(tokens):
            return None

        start = pos
        children = []

        is_pronoun = tokens[pos] in WORD_BY_POS.get("pronoun", [])
        is_det = tokens[pos] in WORD_BY_POS.get("det", [])

        if is_det:
            children.append({"type": "det", "word": tokens[pos]})
            pos += 1
            if pos < len(tokens) and tokens[pos] in WORD_BY_POS.get("adj", []):
                children.append({"type": "adj", "word": tokens[pos]})
                pos += 1
            if pos < len(tokens) and tokens[pos] in WORD_BY_POS.get("noun", []):
                children.append({"type": "N", "word": tokens[pos]})
                pos += 1
                return pos, {"type": "NP", "children": children}
            if is_pronoun:
                return pos, {"type": "NP", "children": children}
            return None

        if is_pronoun:
            return pos + 1, {"type": "NP", "children": [{"type": "pronoun", "word": tokens[pos]}]}

        if pos < len(tokens) and tokens[pos] in WORD_BY_POS.get("adj", []):
            children.append({"type": "adj", "word": tokens[pos]})
            pos += 1

        if pos < len(tokens) and tokens[pos] in WORD_BY_POS.get("noun", []):
            children.append({"type": "N", "word": tokens[pos]})
            pos += 1
            return pos, {"type": "NP", "children": children}

        return None

    def _parse_vp(self, tokens: List[str], pos: int) -> Optional[Tuple[int, Dict]]:
        if pos >= len(tokens):
            return None

        children = []

        if tokens[pos] in WORD_BY_POS.get("verb", []):
            children.append({"type": "V", "word": tokens[pos]})
            pos += 1
        else:
            return None

        np_result = self._parse_np(tokens, pos)
        if np_result:
            pos, np_node = np_result
            children.append(np_node)

        while pos < len(tokens) and tokens[pos] in WORD_BY_POS.get("prep", []):
            pp_result = self._parse_pp(tokens, pos)
            if pp_result:
                pos, pp_node = pp_result
                children.append(pp_node)
            else:
                break

        return pos, {"type": "VP", "children": children}

    def _parse_pp(self, tokens: List[str], pos: int) -> Optional[Tuple[int, Dict]]:
        if pos >= len(tokens) or tokens[pos] not in WORD_BY_POS.get("prep", []):
            return None
        children = [{"type": "prep", "word": tokens[pos]}]
        pos += 1
        np_result = self._parse_np(tokens, pos)
        if np_result:
            pos, np_node = np_result
            children.append(np_node)
        return pos, {"type": "PP", "children": children}

    def extract_meaning(self, parse_tree: Optional[Dict]) -> Optional[Dict]:
        if parse_tree is None:
            return None
        if parse_tree.get("type") == "Q":
            return self._extract_question_meaning(parse_tree)
        meaning = {"subject": None, "action": None, "object": None, "details": []}

        np = parse_tree.get("np")
        if np:
            meaning["subject"] = self._extract_np_meaning(np)

        vp = parse_tree.get("vp")
        if vp:
            for child in vp.get("children", []):
                if child.get("type") == "V":
                    word = child.get("word", "")
                    info = VOCAB.get(word, {})
                    meaning["action"] = {
                        "word": word,
                        "meaning": info.get("meaning", "unknown"),
                        "semantic": info.get("semantic", []),
                    }
                elif child.get("type") == "NP":
                    meaning["object"] = self._extract_np_meaning(child)
                elif child.get("type") == "PP":
                    meaning["details"].append(self._extract_pp_meaning(child))

        return meaning

    def _extract_np_meaning(self, np_node: Dict) -> Dict:
        result = {"words": [], "pos": [], "semantic": []}
        for child in np_node.get("children", []):
            word = child.get("word", "")
            result["words"].append(word)
            result["pos"].append(child.get("type", ""))
            info = VOCAB.get(word, {})
            result["semantic"].extend(info.get("semantic", []))
        return result

    def _extract_pp_meaning(self, pp_node: Dict) -> Dict:
        result = {"prep": "", "np": {}}
        for child in pp_node.get("children", []):
            if child.get("type") == "prep":
                result["prep"] = child.get("word", "")
            elif child.get("type") == "NP":
                result["np"] = self._extract_np_meaning(child)
        return result

    def _extract_question_meaning(self, q_node: Dict) -> Dict:
        meaning = {"qword": q_node.get("qword", ""), "action": None, "object": None, "details": [],
                   "subject": None, "aux": q_node.get("aux", None)}
        np = q_node.get("np")
        if np:
            meaning["subject"] = self._extract_np_meaning(np)
        vp = q_node.get("vp")
        if vp:
            for child in vp.get("children", []):
                if child.get("type") == "V":
                    word = child.get("word", "")
                    info = VOCAB.get(word, {})
                    meaning["action"] = {
                        "word": word,
                        "meaning": info.get("meaning", "unknown"),
                        "semantic": info.get("semantic", []),
                    }
                elif child.get("type") == "NP":
                    meaning["object"] = self._extract_np_meaning(child)
        return meaning


parser = SimpleParser()


def parse_sentence(sentence: str) -> Optional[Dict]:
    return parser.parse(sentence)


def get_meaning(sentence: str) -> Optional[Dict]:
    tree = parse_sentence(sentence)
    return parser.extract_meaning(tree)


def format_parse_tree(tree: Optional[Dict], indent: int = 0) -> str:
    if tree is None:
        return "  " * indent + "(no parse)"
    prefix = "  " * indent
    node_type = tree.get("type", "?")
    if node_type == "Q":
        qword = tree.get("qword", "")
        aux = tree.get("aux", "")
        np_node = tree.get("np")
        vp_node = tree.get("vp")
        parts = [f"{prefix}Q({qword})"]
        if aux:
            parts.append(f"{prefix}  aux({aux})")
        if np_node:
            parts.append(format_parse_tree(np_node, indent + 1))
        if vp_node:
            parts.append(format_parse_tree(vp_node, indent + 1))
        return "\n".join(parts)
    if "word" in tree:
        return f"{prefix}{node_type}({tree['word']})"
    children = tree.get("children", [])
    if not children:
        if node_type == "S":
            np_str = format_parse_tree(tree.get("np"), indent + 1)
            vp_str = format_parse_tree(tree.get("vp"), indent + 1)
            return f"{prefix}S\n{np_str}\n{vp_str}"
        return f"{prefix}{node_type}"
    child_strs = [format_parse_tree(c, indent + 1) for c in children]
    return f"{prefix}{node_type}\n" + "\n".join(child_strs)


RESPONSE_TEMPLATES = {
    "action": [
        "I see that {subj} {verb} {obj}. That is interesting.",
        "So {subj} {verb} {obj}. I understand.",
        "You are telling me that {subj} {verb} {obj}.",
        "{subj} {verb} {obj}. I will remember that.",
    ],
    "perception": [
        "{subj} {verb} {obj}. Observation is important.",
        "It is good that {subj} {verb} {obj} carefully.",
        "Interesting! {subj} {verb} {obj}.",
    ],
    "emotion": [
        "{subj} had strong feelings about {obj}.",
        "I am glad {subj} felt that way about {obj}.",
        "Emotions are part of being alive. {subj} feels deeply about {obj}.",
    ],
    "mental": [
        "{subj} had an interesting thought about {obj}.",
        "Thinking is important. {subj} is wise to consider {obj}.",
        "{subj} thought carefully about {obj}.",
    ],
    "communication": [
        "{subj} said something worth hearing about {obj}.",
        "I listen when {subj} speaks about {obj}.",
        "{subj} communicated clearly about {obj}.",
    ],
    "desire": [
        "{subj} wants {obj}. Everyone wants things.",
        "Desire moves us forward. {subj} knows what they want.",
        "I hope {subj} finds what they want.",
    ],
}

FOLLOW_UP_QUESTIONS = [
    "Why do you think that happened?",
    "Is that common?",
    "Tell me more about that.",
    "How did that make you feel?",
    "What do you think about that?",
    "That is interesting. Can you explain more?",
    "I am curious about this.",
    "What happened next?",
]

class Conversation:
    def __init__(self):
        self.history = []
        self.last_topic = None

    def respond(self, sentence: str, knowledge_base: Optional[Dict] = None) -> str:
        meaning = get_meaning(sentence)
        if meaning is None:
            return self._handle_unparsed(sentence, knowledge_base)

        if "qword" in meaning:
            return self._handle_question(meaning)

        subj = meaning.get("subject", {})
        action = meaning.get("action", {})
        obj = meaning.get("object", {})

        subj_words = subj.get("words", []) if subj else []
        obj_words = obj.get("words", []) if obj else []
        action_word = action.get("word", "") if action else ""
        action_meaning = action.get("meaning", "") if action else ""
        action_semantic = action.get("semantic", []) if action else []

        subj_str = " ".join(subj_words) if subj_words else "someone"
        obj_str = " ".join(obj_words) if obj_words else ""
        verb_str = action_word if action_word else "did"

        details = meaning.get("details", [])
        detail_strs = []
        for d in details:
            prep = d.get("prep", "")
            np_info = d.get("np", {})
            np_words = np_info.get("words", [])
            if np_words:
                detail_strs.append(f"{prep} {' '.join(np_words)}")
        detail_str = " ".join(detail_strs)
        if detail_str:
            if obj_str:
                full_obj = f"{obj_str} {detail_str}"
            else:
                full_obj = detail_str
        else:
            full_obj = obj_str if obj_str else "something"

        kb_response = self._check_knowledge(subj_words, obj_words, knowledge_base)
        if kb_response:
            return kb_response

        self.history.append({"subj": subj_str, "verb": verb_str, "obj": full_obj})
        self.last_topic = subj_str

        context = {
            "subj": subj_str,
            "obj": full_obj,
            "verb": verb_str,
            "pronoun": (
                "he" if subj_str.lower() in ("boy", "man", "king")
                else "she" if subj_str.lower() in ("girl", "woman", "queen")
                else "it" if subj_str.lower() in ("cat", "dog", "bird", "fish", "ball", "car", "book", "tree", "house", "door", "window", "stone", "sun", "moon", "river", "mountain")
                else "they" if subj_str.lower() in ("people", "children", "friends")
                else subj_str
            ),
        }

        if action_meaning == "perception" or "vision" in action_semantic or "hearing" in action_semantic:
            templates = RESPONSE_TEMPLATES["perception"]
        elif action_meaning == "emotion" or "positive" in action_semantic or "negative" in action_semantic:
            templates = RESPONSE_TEMPLATES["emotion"]
        elif action_meaning == "mental" or "mind" in action_semantic or "knowledge" in action_semantic:
            templates = RESPONSE_TEMPLATES["mental"]
        elif action_meaning == "communication" or "speak" in action_semantic or "summon" in action_semantic or "question" in action_semantic:
            templates = RESPONSE_TEMPLATES["communication"]
        elif action_meaning == "desire" or "wish" in action_semantic or "require" in action_semantic:
            templates = RESPONSE_TEMPLATES["desire"]
        else:
            templates = RESPONSE_TEMPLATES["action"]

        response = random.choice(templates).format(**context)

        if len(self.history) >= 2 and random.random() < 0.3:
            prev = self.history[-2]
            follow_ups = [
                f"Earlier you mentioned {prev['subj']}. Is that related?",
                f"Does this involve {prev['subj']}?",
                f"Interesting. Before that, {prev['subj']} {prev['verb']} {prev['obj']}.",
                f"Tell me more about {prev['subj']}.",
            ]
            response += " " + random.choice(follow_ups)
        elif random.random() < 0.3:
            response += " " + random.choice(FOLLOW_UP_QUESTIONS)

        return response

    def _handle_unparsed(self, sentence: str, knowledge_base: Optional[Dict] = None) -> str:
        sentence_lower = sentence.lower().strip()
        greetings = ["hello", "hi", "hey", "greetings", "good morning", "good evening"]
        farewells = ["bye", "goodbye", "see you", "exit", "quit"]
        confirmations = ["yes", "yeah", "sure", "ok", "okay", "yep"]
        denials = ["no", "nope", "not really", "nah"]

        if any(g in sentence_lower for g in greetings):
            return "Hello! I am ShapeLang. I can understand patterns in letters and words. What do you want to talk about?"
        if any(f in sentence_lower for f in farewells):
            return "Goodbye! It was nice talking with you."
        if any(c in sentence_lower for c in confirmations):
            return random.choice(["Good.", "I am glad.", "Yes, I agree.", "Tell me more."])
        if any(d in sentence_lower for d in denials):
            return random.choice(["Why not?", "I see.", "That is unfortunate.", "Can you explain?"])
        if "thank" in sentence_lower:
            return "You are welcome! I enjoy our conversation."

        # Use knowledge base if available
        if knowledge_base and "topics" in knowledge_base:
            topics_str = ", ".join(knowledge_base["topics"].keys())
            entries = [v for v in knowledge_base["topics"].values() if v.get("summary")]
            if entries:
                summary = entries[0]["summary"]
                if len(summary) > 300:
                    summary = summary[:300] + "..."
                return f"I learned about {topics_str}: {summary}"
            entries = [v for v in knowledge_base["topics"].values() if v.get("facts")]
            if entries:
                facts = entries[0].get("facts", [])
                if facts:
                    return f"About {topics_str}: {facts[0]}"

        if "?" in sentence or "what" in sentence_lower or "who" in sentence_lower or "why" in sentence_lower or "how" in sentence_lower:
            if self.last_topic and self.last_topic in sentence_lower:
                return f"About {self.last_topic}: I only know what you have told me. Can you teach me more?"
            return "That is a good question. I am still learning about the world."

        return random.choice([
            "I am not sure I understand. Can you tell me more?",
            "Interesting. What do you mean?",
            "I am still learning. Please explain in simpler words.",
            "Can you rephrase that? I want to understand.",
        ])

    def _check_knowledge(self, subj_words: list, obj_words: list, kb: Optional[Dict]) -> Optional[str]:
        if not kb:
            return None
        for words in [subj_words, obj_words]:
            if words:
                topic = " ".join(words).lower()
                entry = kb.get("topics", {}).get(topic, {})
                if entry:
                    facts = entry.get("facts", [])
                    summary = entry.get("summary", "")
                    if facts:
                        return f"About {topic}: {facts[0]}"
                    elif summary:
                        return f"About {topic}: {summary}"
        return None

    def _handle_question(self, meaning: Dict) -> str:
        qword = meaning.get("qword", "")
        action = meaning.get("action", {})
        action_word = action.get("word", "") if action else ""
        obj = meaning.get("object", {})
        obj_words = obj.get("words", []) if obj else []
        obj_str = " ".join(obj_words) if obj_words else ""
        subj = meaning.get("subject", {})
        subj_words = subj.get("words", []) if subj else []
        subj_str = " ".join(subj_words) if subj_words else ""
        aux = meaning.get("aux", "")

        if aux:
            if qword == "why":
                if subj_str and action_word:
                    return f"I am not sure why {subj_str} {action_word}. Can you tell me why?"
                return "I do not know why that happened. What do you think?"
            if qword == "what":
                return "I am not sure what that is. Can you describe it?"
            if qword == "how":
                return f"I am not sure how {subj_str} {action_word}." if subj_str and action_word else "That is a good question. How do you think it works?"

        if "be" in action.get("semantic", []):
            if qword == "how" and obj_str == "you":
                return "I am doing well, thank you for asking. I am learning new things every day."
            if qword == "how" and obj_str:
                return f"About {obj_str}: I am not sure how they are. Can you tell me more?"
            if qword == "how":
                return "That is an interesting question. How can I help you?"
            if qword == "what" and obj_str == "you":
                return (
                    "I am the Symbol Grounding Network (SGN), an AI that learns language "
                    "from the visual shapes of alphabet letters. I use a MobileNetV3-Large "
                    "CNN (~2.05M params) to recognize letters from 7x7 binary grids, "
                    "extract 19 topological features, and fuse them to understand "
                    "a 184-word vocabulary. I can also search the web to learn new topics."
                )
            if qword == "what":
                if obj_str:
                    return f"About {obj_str}: I am still learning. What would you like to know?"
                return "That is a broad question. Can you be more specific?"
            if qword == "who":
                return "I am ShapeLang, an AI that learns language from letter shapes."
            if qword == "where":
                return "I exist in this computer, learning about language and the world."
            if qword == "why":
                return "That is a deep question. I am here to learn and converse."
            if qword == "when":
                return "Time is an interesting concept. I experience the present moment."

        if qword == "what":
            return "I am curious about that too. What do you think?"
        if qword == "why":
            return "I am not sure why. Can you tell me more?"
        if qword == "how":
            return "That is a good question. I am still learning how things work."
        if qword == "who":
            if obj_str:
                return f"I do not know who {obj_str} is. Who are they?"
            return "I am not sure who you mean. Can you explain?"
        if qword == "where":
            return "I am not sure where that is. Tell me more about it."
        if qword == "when":
            return "I am not sure when that happened. Can you give me more context?"

        return random.choice([
            "That is a good question.",
            "I am still learning about that.",
            "Interesting question! What do you think?",
            "I wonder about that too.",
        ])


conversation = Conversation()


def respond_to_sentence(sentence: str, knowledge_base: Optional[Dict] = None) -> str:
    response = conversation.respond(sentence, knowledge_base)
    if response:
        response = response[0].upper() + response[1:]
    return response


if __name__ == "__main__":
    print(f"Vocabulary size: {len(VOCAB)} words")
    print(f"Token vocabulary: {VOCAB_SIZE} tokens\n")

    print("Sample generated sentences:")
    for _ in range(5):
        s = generate_sentence()
        print(f"  {s}")

    print("\nSample conversation:")
    test_inputs = [
        "hello",
        "the cat chased a dog",
        "she saw a happy bird in the sky",
        "why did the cat do that",
        "tell me more",
        "goodbye",
    ]
    for inp in test_inputs:
        resp = respond_to_sentence(inp)
        print(f"  You: {inp}")
        print(f"  AI: {resp}")
        print()
