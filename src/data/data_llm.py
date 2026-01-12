from openai import OpenAI

LM_CLIENT = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="none",
)


def improve_synopsis_with_llm(omdb_plot, orig_synopsis, title, year):
    prompt = f"""
Rewrite the synopsis of the following film in clean, factual English.
Rules:
- About 5 sentences long
- No hallucinations
- Use the OMDb plot as the factual base
- Preserve meaning
- If the OMDb plot is good as-is, expand it to about 5 sentences.

Film: {title} ({year})
OMDb Plot: {omdb_plot}
Original Synopsis: {orig_synopsis}

Rewrite now.
"""
    res = LM_CLIENT.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return res.choices[0].message["content"].strip()


def improve_review(review_text):
    prompt = f"""
You are a review normalizer.
Rewrite the following user review in clean, grammatically correct English.

Rules:
- Preserve meaning, tone, humor and sentiment  
- Do NOT add new content  
- Remove emojis and non-Latin symbols  
- Keep similar length  

Original review:
{review_text}

Rewrite it now.
"""
    res = LM_CLIENT.chat.completions.create(
        model="local-model",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return res.choices[0].message["content"].strip()


def process_entry(entry):
    improved_reviews = [improve_review(r) for r in entry["review_texts"]]

    return {
        "title": entry["title"],
        "year": entry["year"],
        "review_texts": improved_reviews,
    }
