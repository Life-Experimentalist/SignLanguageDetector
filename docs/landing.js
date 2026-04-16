const counterBaseUrl = "https://counter.vkrishna04.me";
const projectName = "sign-language-detector";
const themeStorageKey = "sld-theme";

function applyTheme(theme) {
  document.body.setAttribute("data-theme", theme);

  const label = document.getElementById("themeToggleLabel");
  if (label) {
    label.textContent = theme === "dark" ? "Light" : "Dark";
  }
}

function getInitialTheme() {
  const savedTheme = localStorage.getItem(themeStorageKey);
  if (savedTheme === "dark" || savedTheme === "light") {
    return savedTheme;
  }

  const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  return prefersDark ? "dark" : "light";
}

function setupThemeToggle() {
  const toggle = document.getElementById("themeToggle");
  if (!toggle) {
    return;
  }

  let currentTheme = getInitialTheme();
  applyTheme(currentTheme);

  toggle.addEventListener("click", () => {
    currentTheme = currentTheme === "dark" ? "light" : "dark";
    localStorage.setItem(themeStorageKey, currentTheme);
    applyTheme(currentTheme);
  });
}

async function incrementAndFetchViews() {
  const viewsEl = document.getElementById("viewsCount");
  const healthEl = document.getElementById("counterHealth");

  try {
    const incrementResponse = await fetch(
      `${counterBaseUrl}/api/views/${projectName}`,
      {
        method: "POST",
      },
    );

    if (!incrementResponse.ok) {
      throw new Error(`Increment failed: ${incrementResponse.status}`);
    }

    const viewsResponse = await fetch(
      `${counterBaseUrl}/api/views/${projectName}`,
    );
    if (!viewsResponse.ok) {
      throw new Error(`View fetch failed: ${viewsResponse.status}`);
    }

    const viewsData = await viewsResponse.json();
    const viewCount =
      viewsData.totalViews ??
      viewsData.views ??
      viewsData.count ??
      viewsData.total ??
      "Unavailable";

    viewsEl.textContent = String(viewCount);
    healthEl.textContent = "Live";
  } catch (error) {
    viewsEl.textContent = "Unavailable";
    healthEl.textContent = "Counter offline";
  }
}

setupThemeToggle();
incrementAndFetchViews();
