import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { ErrorBoundary } from "./ErrorBoundary";
import "./styles.css";

function showBrowserError(title: string, detail: string) {
  let element = document.getElementById("browser-error-overlay");
  if (!element) {
    element = document.createElement("pre");
    element.id = "browser-error-overlay";
    document.body.appendChild(element);
  }
  element.textContent = `${title}\n\n${detail}`;
}

window.addEventListener("error", (event) => {
  const detail = event.error?.stack || event.message || "Unknown browser error.";
  showBrowserError("Uncaught browser error", detail);
});

window.addEventListener("unhandledrejection", (event) => {
  const reason = event.reason;
  const detail = reason instanceof Error ? reason.stack || reason.message : String(reason);
  showBrowserError("Unhandled promise rejection", detail);
});

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>
);
