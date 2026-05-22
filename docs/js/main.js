/* PSCDL 2026 — minor interactivity (smooth scroll, hover lift, header shrink) */

// Smooth-scroll for in-page anchors
document.querySelectorAll('a[href^="#"]').forEach(a => {
  a.addEventListener('click', e => {
    const target = document.querySelector(a.getAttribute('href'));
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  });
});

// Reveal-on-scroll: progressively show .card and .download-card
const obs = new IntersectionObserver(entries => {
  entries.forEach(en => {
    if (en.isIntersecting) {
      en.target.style.opacity = 1;
      en.target.style.transform = 'translateY(0)';
      obs.unobserve(en.target);
    }
  });
}, { threshold: 0.15 });

document.querySelectorAll('.card, .download-card').forEach(el => {
  el.style.opacity = 0;
  el.style.transform = 'translateY(12px)';
  el.style.transition = 'opacity 0.45s ease, transform 0.45s ease';
  obs.observe(el);
});
