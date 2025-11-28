import gradio as gr

from pathlib import Path
from src.generation.generate_transformer import load_model, generate


BASE_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "transformer_6.8M.pt"
print("BASE_DIR:", BASE_DIR)
print("Tokenizer file exists?", (BASE_DIR / "tokenizer" / "bpe_hugging_face_tokenizer.json").exists())
model, tokenizer, tokenizer_type, device = load_model(MODEL_PATH)


def review_from_synopsis(synopsis):
    prompt = f"Synopsis: {synopsis}\nReview:"
    review = generate(
        model,
        tokenizer,
        tokenizer_type,
        device,
        prompt=prompt,
        length=200
    )
    return review


if __name__ == "__main__":
    custom_css = """
    body, .gradio-container {
        background: #14181C !important;
    }
    #main-card, .gr-textbox textarea, .star-row {
        background: #445566 !important;
        color: #FFF !important;
        border: none;
        border-radius: 12px;
    }
    .gr-textbox textarea {
        color: #FFF !important;
    }
    #review-output {
        background: #445566 !important;
        color: #EEE !important;
        border: none;
        border-radius: 12px;
        font-size: 1.1em;
    }
    .star-row {
        display: flex;
        align-items: center;
        gap: 2px;
        padding: 8px 0;
        justify-content: center;
    }
    .star {
        font-size: 2.5em;
        padding: 0 2px;
        cursor: pointer;
        transition: color 0.1s;
        color: #334455;
        user-select: none;
    }
    .star.filled {
        color: #00C030;
        text-shadow: 0 0 4px #00C03088;
    }
    """

    with gr.Blocks(css=custom_css) as demo:
        gr.Markdown(
            "# <span style='color:#00C030;'>★</span> Letterboxd-style Film Log Demo",
            elem_id="main-card"
        )

        with gr.Row():
            synopsis_input = gr.Textbox(
                label="Paste the film's synopsis",
                lines=5,
                interactive=True
            )

        gr.Markdown("Rate it:")
        # Stars UI + hidden input for current rating
        stars_hidden = gr.Textbox(value="3", visible=False)

        stars_html = gr.HTML("""
    <div id='star-slider-row' class="star-row" style="text-align:center;">
      <span class="star" data-value="1">&#9733;</span>
      <span class="star" data-value="2">&#9733;</span>
      <span class="star" data-value="3">&#9733;</span>
      <span class="star" data-value="4">&#9733;</span>
      <span class="star" data-value="5">&#9733;</span>
    </div>
    <script>
    function setStarRating(val) {
        let stars = document.querySelectorAll('#star-slider-row .star');
        for (let i=0; i<5; i++) {
            stars[i].classList.toggle('filled', i < val);
        }
        // Set value in the Gradio hidden textbox
        document.querySelector('input[aria-label="stars_hidden"]').value = val;
    }
    // Add interactivity:
    setTimeout(()=>{
        let stars = document.querySelectorAll('#star-slider-row .star');
        let curr = parseInt(document.querySelector('input[aria-label="stars_hidden"]').value);
        setStarRating(curr);
        stars.forEach((el, idx) => {
            el.onclick = () => setStarRating(idx+1);
        });
    }, 200);
    </script>
    """)
        with gr.Row():
            generate_button = gr.Button("Generate Review", elem_id="main-card")

        gr.Markdown("Generated Review")
        generated_review = gr.Textbox(
            lines=6,
            label="",
            elem_id="review-output",
            interactive=False
        )

        generate_button.click(
            review_from_synopsis,
            inputs=[synopsis_input, stars_hidden],
            outputs=generated_review
        )


    '''
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# FilmCriticLM")
        with gr.Row():
            synopsis_input = gr.Textbox(
                label="Film Synopsis",
                placeholder="Paste the synopsis here...",
                lines=6
            )
        star_rating = gr.Radio(
            choices=["★☆☆☆☆", "★★☆☆☆", "★★★☆☆", "★★★★☆", "★★★★★"],
            label="Your Star Rating",
            value="★★★☆☆"
        )
        generate_button = gr.Button("Generate Review")
    
        gr.Markdown("Generated Review")
        generated_review = gr.Textbox(
            label="",
            interactive=False,
            lines=7
        )
        generate_button.click(
            review_from_synopsis,
            inputs=[synopsis_input, star_rating],
            outputs=generated_review,
        )
    
        gr.Markdown(
            "<sub>Note: Your star rating is only shown, not used as input. <br/>\
            For demo: Paste a synopsis and see how the model reviews the movie!</sub>"
        )
    '''

    demo.launch()