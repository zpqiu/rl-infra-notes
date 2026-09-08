const slides = [...document.querySelectorAll(".slide")];
const viewport = document.querySelector("#viewport");
const progress = document.querySelector(".progress");
let current = 0;

document.querySelectorAll("[data-tex]").forEach((node) => {
  katex.render(node.dataset.tex, node, {
    displayMode: node.dataset.display === "true",
    throwOnError: false,
    strict: false,
  });
});

function fit() {
  const scale = Math.min(window.innerWidth / 1280, window.innerHeight / 720);
  viewport.style.transform = `scale(${scale})`;
}

function show(index, updateHash = true) {
  current = (index + slides.length) % slides.length;
  slides.forEach((slide, i) => slide.classList.toggle("active", i === current));
  progress.style.width = `${((current + 1) / slides.length) * 100}%`;
  if (updateHash) history.replaceState(null, "", `#${current + 1}`);
}

function fromHash() {
  const n = Number(location.hash.slice(1));
  show(Number.isFinite(n) && n > 0 ? n - 1 : 0, false);
}

document.querySelector("#prev").addEventListener("click", () => show(current - 1));
document.querySelector("#next").addEventListener("click", () => show(current + 1));

document.addEventListener("keydown", (event) => {
  if (["ArrowRight", "ArrowDown", "PageDown", " "].includes(event.key)) {
    event.preventDefault();
    show(current + 1);
  } else if (["ArrowLeft", "ArrowUp", "PageUp"].includes(event.key)) {
    event.preventDefault();
    show(current - 1);
  } else if (event.key === "Home") {
    show(0);
  } else if (event.key === "End") {
    show(slides.length - 1);
  } else if (event.key.toLowerCase() === "f") {
    document.documentElement.requestFullscreen?.();
  }
});

window.addEventListener("resize", fit);
window.addEventListener("hashchange", fromHash);
fit();
fromHash();
