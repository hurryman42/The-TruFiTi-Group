// STAR + HEART RATING (Letterboxd-style)
const halves = [...document.querySelectorAll(".half")];
const stars = [...document.querySelectorAll(".star")];
const heart = document.getElementById("heart");

let currentRating = 0;
let liked = false;

// --- Draw selected stars (0, 0.5, 1.0, ... 5.0) ---
function render(rating) {
    stars.forEach((star, idx) => {
        const full = idx + 1;       // 1, 2, 3, 4, 5
        const half = full - 0.5;    // 0.5, 1.5, 2.5, 3.5, 4.5

        star.classList.remove("full-selected", "half-selected");

        if (rating >= full) {
            star.classList.add("full-selected");
        } else if (rating === half) {
            star.classList.add("half-selected");
        }
    });
}

// --- Hover preview (only when not yet rated) ---
halves.forEach(h => {
    h.addEventListener("mouseenter", () => {
        if (currentRating !== 0) return;

        const hoverVal = parseFloat(h.dataset.value);
        stars.forEach((star, idx) => {
            const full = idx + 1;
            const half = full - 0.5;

            star.classList.remove("hovered");

            if (hoverVal >= full) {
                star.classList.add("hovered");
            } else if (hoverVal === half) {
                // for half hover, show full hover overlay (LB effect)
                star.classList.add("hovered");
            }
        });
    });
});

halves.forEach(h => {
    h.addEventListener("mouseleave", () => {
        if (currentRating !== 0) return;
        stars.forEach(star => star.classList.remove("hovered"));
    });
});

// --- Click to select/reset rating ---
halves.forEach(h => {
    h.addEventListener("click", () => {
        const val = parseFloat(h.dataset.value);
        currentRating = (currentRating === val ? 0 : val);
        stars.forEach(star => star.classList.remove("hovered"));
        render(currentRating);
    });
});

// --- HEART toggle ---
heart.onclick = () => {
    liked = !liked;
    heart.textContent = liked ? "♥" : "♡";
    heart.classList.toggle("liked", liked);
};

// --- API SUBMISSION ---
document.getElementById("generate-btn").onclick = async () => {
    const synopsis = document.getElementById("synopsis-input").value;

    const payload = {
        synopsis,
        rating: currentRating,
        liked
    };

    const response = await fetch("/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await response.json();
    document.getElementById("output").value = data.review;
};
