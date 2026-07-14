const landing = document.querySelector(".landing");
const projectBar = document.querySelector(".project-bar");
const scrollCue = document.querySelector(".scroll-cue");
let enteredContent = false;
let previousScroll = window.scrollY;

function setProjectChrome(visible) {
  projectBar?.classList.toggle("is-visible", visible);
  document.body.classList.toggle("has-entered-content", visible);
}

function updateProjectChrome() {
  if (!landing) return;

  const transitionPoint = landing.offsetHeight - 100;
  const scrollingUp = window.scrollY < previousScroll;
  if (window.scrollY >= transitionPoint) enteredContent = true;
  if (scrollingUp && window.scrollY < transitionPoint) enteredContent = false;
  setProjectChrome(enteredContent);
  previousScroll = window.scrollY;
}

scrollCue?.addEventListener("click", () => {
  enteredContent = true;
  setProjectChrome(true);
});

window.addEventListener("scroll", updateProjectChrome, { passive: true });
updateProjectChrome();

const videos = document.querySelectorAll(".video-player video");

function closeInlineControls() {
  videos.forEach((video) => {
    if (document.fullscreenElement !== video) video.controls = false;
  });
}

document.querySelectorAll(".video-fullscreen").forEach((button) => {
  button.addEventListener("click", async () => {
    const video = button.closest(".video-player")?.querySelector("video");
    if (!video) return;

    video.controls = true;
    try {
      if (video.requestFullscreen) {
        await video.requestFullscreen();
      } else if (video.webkitEnterFullscreen) {
        video.webkitEnterFullscreen();
      }
    } catch {
      video.controls = false;
    }
  });
});

document.addEventListener("fullscreenchange", closeInlineControls);
videos.forEach((video) => video.addEventListener("webkitendfullscreen", closeInlineControls));

const researchDrawer = document.querySelector(".research-drawer");
const researchToggles = document.querySelectorAll(".research-toggle");
const researchClosers = document.querySelectorAll("[data-research-close]");

function setResearchDrawer(open) {
  document.body.classList.toggle("research-open", open);
  researchDrawer?.setAttribute("aria-hidden", String(!open));
  researchToggles.forEach((toggle) => toggle.setAttribute("aria-expanded", String(open)));
}

researchToggles.forEach((toggle) => {
  toggle.addEventListener("click", () => setResearchDrawer(true));
});

researchClosers.forEach((closer) => {
  closer.addEventListener("click", () => setResearchDrawer(false));
});

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") setResearchDrawer(false);
});
