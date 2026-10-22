import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";

/** Keep desktop table scrollbars within the viewport and support mouse and keyboard scrolling. */
export function ScrollArea({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const [edges, setEdges] = useState({ left: false, right: false, inset: 0, bar: 0, foot: 0 });

  // Resize the container without rerendering table rows.
  const fit = () => {
    const el = ref.current;
    if (!el) return;
    const top = el.getBoundingClientRect().top + scrollY;
    const foot = el.closest(".table-card")?.querySelector(".footnote")?.getBoundingClientRect().height ?? 0;
    // Phones scroll the page; a nested scroll box there would show only a few rows.
    el.style.maxHeight = innerWidth <= 760 ? "none" : `${Math.max(320, innerHeight - top - foot - 12)}px`;
  };
  const edgesNow = () => {
    const el = ref.current;
    if (!el) return;
    const inset = el.querySelector<HTMLElement>("thead .sticky")?.offsetWidth ?? 0;
    const left = el.scrollLeft > 1;
    const right = el.scrollLeft + el.clientWidth < el.scrollWidth - 1;
    // Keep arrow buttons clear of native scrollbars.
    const bar = el.offsetWidth - el.clientWidth, foot = el.offsetHeight - el.clientHeight;
    setEdges((e) => (e.left === left && e.right === right && e.inset === inset && e.bar === bar && e.foot === foot
      ? e : { left, right, inset, bar, foot }));
  };
  const update = () => {
    fit();
    edgesNow();
  };
  useLayoutEffect(fit);
  useEffect(() => {
    const frame = requestAnimationFrame(edgesNow);
    return () => cancelAnimationFrame(frame);
  });

  const page = (dir: 1 | -1) => {
    const el = ref.current;
    if (el) el.scrollBy({ left: dir * 0.7 * (el.clientWidth - edges.inset), behavior: "smooth" });
  };

  useEffect(() => {
    const el = ref.current!;
    const controls = document.querySelector(".controls");
    const observer = new ResizeObserver(update);
    if (controls) observer.observe(controls);
    addEventListener("resize", update);

    // A vertical wheel over the header row scrolls sideways; elsewhere it scrolls rows as usual.
    const wheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey || e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY)) return;
      if (!(e.target as Element).closest("thead")) return;
      if (el.scrollWidth <= el.clientWidth) return;
      el.scrollLeft += e.deltaY;
      e.preventDefault();
    };

    // Mouse drag pans the table; a drag does not count as a click on the cell under the pointer.
    let drag: { x: number; y: number; left: number; top: number; moved: boolean } | null = null;
    const down = (e: PointerEvent) => {
      if (e.pointerType !== "mouse" || e.button !== 0) return;
      if ((e.target as Element).closest("input, select, a, .pick")) return;
      drag = { x: e.clientX, y: e.clientY, left: el.scrollLeft, top: el.scrollTop, moved: false };
    };
    const move = (e: PointerEvent) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) < 6) return;
      if (!drag.moved) {
        drag.moved = true;
        el.classList.add("panning");
        getSelection()?.removeAllRanges();
      }
      el.scrollLeft = drag.left - dx;
      el.scrollTop = drag.top - dy;
    };
    const up = () => {
      if (drag?.moved) {
        el.classList.remove("panning");
        // Prevent the drag's final click from opening a cell.
        const swallow = (c: Event) => {
          c.stopPropagation();
          c.preventDefault();
        };
        addEventListener("click", swallow, { capture: true, once: true });
        setTimeout(() => removeEventListener("click", swallow, { capture: true }), 0);
      }
      drag = null;
    };
    const cancel = () => {
      el.classList.remove("panning");
      drag = null;
    };

    let over = false;
    const enter = () => (over = true);
    const leave = () => (over = false);
    const key = (e: KeyboardEvent) => {
      if ((!over && !el.contains(document.activeElement)) || e.altKey || e.ctrlKey || e.metaKey) return;
      if ((document.activeElement as Element | null)?.closest("input, select, textarea, .panel")) return;
      if (document.querySelector(".panel")) return;
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        el.scrollBy({ left: (e.key === "ArrowRight" ? 1 : -1) * 240, behavior: "smooth" });
        e.preventDefault();
      }
    };

    el.addEventListener("scroll", edgesNow, { passive: true });
    el.addEventListener("wheel", wheel, { passive: false });
    el.addEventListener("pointerdown", down);
    addEventListener("pointermove", move);
    addEventListener("pointerup", up);
    addEventListener("pointercancel", cancel);
    el.addEventListener("mouseenter", enter);
    el.addEventListener("mouseleave", leave);
    addEventListener("keydown", key);
    return () => {
      observer.disconnect();
      removeEventListener("resize", update);
      el.removeEventListener("scroll", edgesNow);
      el.removeEventListener("wheel", wheel);
      el.removeEventListener("pointerdown", down);
      removeEventListener("pointermove", move);
      removeEventListener("pointerup", up);
      removeEventListener("pointercancel", cancel);
      el.removeEventListener("mouseenter", enter);
      el.removeEventListener("mouseleave", leave);
      removeEventListener("keydown", key);
    };
  }, []);

  return (
    <div className="scroll-area">
      <div className="table-scroll" ref={ref} tabIndex={0} role="region" aria-label="Scrollable model table">{children}</div>
      {edges.left && (
        <button className="edge left" style={{ left: edges.inset + 1, bottom: edges.foot }} aria-label="Scroll left" onClick={() => page(-1)}>‹</button>
      )}
      {edges.right && (
        <button className="edge right" style={{ right: edges.bar, bottom: edges.foot }} aria-label="Scroll right" onClick={() => page(1)}>›</button>
      )}
    </div>
  );
}
