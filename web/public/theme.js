(() => {
  const root = document.documentElement;
  const system = matchMedia("(prefers-color-scheme: dark)");
  const read = () => {
    try {
      const value = localStorage.getItem("theme");
      return value === "light" || value === "dark" ? value : null;
    } catch {
      return null;
    }
  };
  let selected = read();
  const apply = () => {
    root.dataset.theme = selected || (system.matches ? "dark" : "light");
  };
  apply();

  document.addEventListener("click", (event) => {
    if (!(event.target instanceof Element) || !event.target.closest("[data-theme-toggle]")) return;
    selected = root.dataset.theme === "dark" ? "light" : "dark";
    apply();
    try {
      localStorage.setItem("theme", selected);
    } catch {
      // The toggle still works when browser storage is unavailable.
    }
  });
  system.addEventListener("change", apply);
  const restore = () => {
    selected = read();
    apply();
  };
  window.addEventListener("pageshow", restore);
  window.addEventListener("storage", (event) => {
    if (event.key === "theme" || event.key === null) restore();
  });
})();
