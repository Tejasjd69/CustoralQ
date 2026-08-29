from pathlib import Path
import re
import pandas as pd
from gensim.models import Word2Vec


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "cx"
    / "cfpb_complaints.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "cx_nlp"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_and_tokenize(text):

    if pd.isna(text):
        return []

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove CFPB anonymization artifacts
    text = re.sub(
        r"\bx+\b",
        " ",
        text
    )

    # Remove words made almost entirely of x's
    text = re.sub(
        r"\b[x]{2,}\b",
        " ",
        text
    )

    # Keep alphabetic characters
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    tokens = text.split()

    # Remove very short tokens
    tokens = [
        word
        for word in tokens
        if len(word) >= 3
    ]

    return tokens


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print("WORD2VEC CUSTOMER VOICE ANALYSIS")
    print("=" * 70)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False
    )

    print(
        f"\nLoaded complaints: {len(df):,}"
    )

    df["complaint_what_happened"] = (
        df["complaint_what_happened"]
        .fillna("")
        .astype(str)
    )

    return df


# ============================================================
# PREPARE SENTENCES
# ============================================================

def prepare_sentences(df):

    print("\nPreparing complaint text...")

    sentences = []

    for text in df[
        "complaint_what_happened"
    ]:

        tokens = clean_and_tokenize(text)

        if len(tokens) >= 5:
            sentences.append(tokens)

    print(
        f"Usable complaint documents: "
        f"{len(sentences):,}"
    )

    return sentences


# ============================================================
# TRAIN WORD2VEC
# ============================================================

def train_word2vec(sentences):

    print("\nTraining Word2Vec...")

    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=5,
        workers=4,
        sg=1,
        epochs=10,
        seed=42
    )

    print(
        f"Vocabulary size: "
        f"{len(model.wv):,}"
    )

    print(
        f"Vector dimensions: "
        f"{model.vector_size}"
    )

    return model


# ============================================================
# SEMANTIC SIMILARITY
# ============================================================

def semantic_similarity(model):

    print("\n" + "=" * 70)
    print("SEMANTIC SIMILARITY")
    print("=" * 70)

    target_words = [
        "fraud",
        "payment",
        "account",
        "credit",
        "loan",
        "refund",
        "charge",
        "transaction",
        "debt",
        "mortgage",
    ]

    all_results = []

    for word in target_words:

        if word not in model.wv:

            print(
                f"\n{word}: not in vocabulary"
            )

            continue

        similar = model.wv.most_similar(
            word,
            topn=10
        )

        print(
            f"\n{word.upper()}"
        )

        for similar_word, score in similar:

            print(
                f"  {similar_word:25s} "
                f"{score:.4f}"
            )

            all_results.append({
                "target_word": word,
                "similar_word": similar_word,
                "similarity": score
            })

    result = pd.DataFrame(
        all_results
    )

    result.to_csv(
        OUTPUT_DIR
        / "word2vec_semantic_similarity.csv",
        index=False
    )

    return result


# ============================================================
# IMPORTANT CX CONCEPTS
# ============================================================

def cx_concept_analysis(model):

    print("\n" + "=" * 70)
    print("CX CONCEPT ANALYSIS")
    print("=" * 70)

    concepts = {
        "fraud": [
            "fraud",
            "unauthorized",
            "identity"
        ],

        "payments": [
            "payment",
            "payments",
            "transaction"
        ],

        "credit": [
            "credit",
            "reporting",
            "score"
        ],

        "debt": [
            "debt",
            "collection",
            "collector"
        ],

        "service": [
            "service",
            "customer",
            "support"
        ],

        "refund": [
            "refund",
            "charge",
            "charged"
        ],
    }

    rows = []

    for concept, words in concepts.items():

        valid_words = [
            word
            for word in words
            if word in model.wv
        ]

        if not valid_words:
            continue

        vector = sum(
            model.wv[word]
            for word in valid_words
        ) / len(valid_words)

        similarities = (
            model.wv.similar_by_vector(
                vector,
                topn=15
            )
        )

        print(
            f"\n{concept.upper()}"
        )

        for word, score in similarities:

            print(
                f"  {word:25s} "
                f"{score:.4f}"
            )

            rows.append({
                "concept": concept,
                "similar_word": word,
                "similarity": score
            })

    result = pd.DataFrame(rows)

    result.to_csv(
        OUTPUT_DIR
        / "word2vec_cx_concepts.csv",
        index=False
    )

    return result


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    model_path = (
        OUTPUT_DIR
        / "cfpb_word2vec.model"
    )

    model.save(str(model_path))

    print(
        f"\nModel saved to:\n{model_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    df = load_data()

    sentences = prepare_sentences(df)

    if len(sentences) < 100:

        raise RuntimeError(
            "Not enough usable complaint narratives."
        )

    model = train_word2vec(
        sentences
    )

    semantic_similarity(
        model
    )

    cx_concept_analysis(
        model
    )

    save_model(
        model
    )

    print("\n" + "=" * 70)
    print("WORD2VEC ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()