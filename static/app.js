const state = {
  map: null,
  heatLayer: null,
  markers: [],
  pendingMapPoints: [],
  demoText: document.getElementById("singleText").value
};

const riskScore = document.getElementById("riskScore");
const riskLevel = document.getElementById("riskLevel");
const flagCount = document.getElementById("flagCount");
const cityCount = document.getElementById("cityCount");
const resultOutput = document.getElementById("resultOutput");
const entityOutput = document.getElementById("entityOutput");
const recordsTable = document.getElementById("recordsTable");

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("is-active"));
    document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("is-active"));
    button.classList.add("is-active");
    document.getElementById(button.dataset.tab).classList.add("is-active");
    if (button.dataset.tab === "map") {
      setTimeout(() => {
        initMap();
        state.map.invalidateSize();
        if (state.pendingMapPoints.length) {
          updateMap(state.pendingMapPoints);
        } else {
          refreshMap();
        }
      }, 50);
    }
  });
});

document.getElementById("loadDemo").addEventListener("click", () => {
  document.getElementById("singleText").value = state.demoText;
});

document.getElementById("analyzeText").addEventListener("click", async () => {
  const text = document.getElementById("singleText").value.trim();
  const threshold = Number(document.getElementById("singleThreshold").value);
  if (!text) {
    showError("Enter text before running analysis.");
    return;
  }
  setBusy("Analyzing text...");
  const data = await postJson("/api/analyze", { text, threshold });
  if (data) {
    renderSingle(data.result);
    renderTable([data.result]);
    updateMap(data.map);
  }
});

document.getElementById("runBatch").addEventListener("click", async () => {
  const file = document.getElementById("batchFile").files[0];
  const text = document.getElementById("batchText").value.trim();
  const threshold = Number(document.getElementById("batchThreshold").value);
  setBusy("Scanning records...");

  let data;
  if (file) {
    const form = new FormData();
    form.append("file", file);
    form.append("threshold", threshold);
    data = await postForm("/api/batch", form);
  } else {
    data = await postJson("/api/batch", { text, threshold });
  }

  if (data) {
    renderSummary(data.summary);
    renderTable(data.results);
    updateMap(data.map);
  }
});

document.getElementById("runScrape").addEventListener("click", async () => {
  const urls = document.getElementById("scrapeUrls").value.trim();
  const threshold = Number(document.getElementById("scrapeThreshold").value);
  if (!urls) {
    showError("Enter at least one public URL.");
    return;
  }
  setBusy("Fetching public pages...");
  const data = await postJson("/api/scrape", { urls, threshold });
  if (data) {
    renderSummary(data.summary, data.errors);
    renderTable(data.results);
    updateMap(data.map);
  }
});

document.getElementById("resetMap").addEventListener("click", async () => {
  const data = await postJson("/api/reset-map", {});
  if (data) updateMap(data.map);
});

document.getElementById("loadMapDemo").addEventListener("click", async () => {
  setBusy("Loading demo hotspot data...");
  const data = await postJson("/api/demo-map", { threshold: 0.5 });
  if (data) {
    renderSummary(data.summary);
    renderTable(data.results);
    updateMap(data.map);
  }
});

async function postJson(url, body) {
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Request failed.");
    return data;
  } catch (error) {
    showError(error.message);
    return null;
  }
}

async function postForm(url, form) {
  try {
    const response = await fetch(url, { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Request failed.");
    return data;
  } catch (error) {
    showError(error.message);
    return null;
  }
}

function renderSingle(result) {
  riskScore.textContent = `${result.risk_percent}%`;
  riskLevel.textContent = result.level;
  flagCount.textContent = result.flag_count;
  cityCount.textContent = result.cities.length;

  const badgeClass = result.risk >= 0.65 ? "high" : result.risk >= 0.5 ? "mid" : "low";
  resultOutput.innerHTML = `
    <div class="result-title">
      <strong>${escapeHtml(result.label)}</strong>
      <span class="badge ${badgeClass}">${escapeHtml(result.level)}</span>
    </div>
    <div class="bar"><span style="width:${result.risk_percent}%"></span></div>
    <p style="margin:14px 0 0">Risk ${result.risk} and safe ${result.safe}</p>
    <p style="margin:8px 0 0">Method: ${escapeHtml(result.method || "indicator model")}</p>
  `;
  renderEntities(result);
}

function renderSummary(summary, errors = []) {
  riskScore.textContent = `${summary.average_risk || 0}%`;
  riskLevel.textContent = `${summary.flagged || 0} flagged`;
  flagCount.textContent = summary.total || 0;
  cityCount.textContent = (summary.top_cities || []).length;

  const cityText = (summary.top_cities || []).map(([city, count]) => `${city} (${count})`).join(", ") || "No mapped cities";
  const errorText = errors.length ? `<p>${errors.length} URL request failed.</p>` : "";
  resultOutput.innerHTML = `
    <div class="result-title">
      <strong>${summary.total || 0} records scanned</strong>
      <span class="badge">${summary.flagged || 0} flagged</span>
    </div>
    <p>Average risk ${summary.average_risk || 0}%.</p>
    <p>Top cities: ${escapeHtml(cityText)}</p>
    ${errorText}
  `;
  entityOutput.innerHTML = "";
}

function renderEntities(result) {
  const chips = [
    ...result.cities.map((value) => ["City", value]),
    ...result.phones.map((value) => ["Phone", value]),
    ...result.persons.map((value) => ["Person", value]),
    ...Object.entries(result.flags).flatMap(([category, values]) => values.map((value) => [category, value]))
  ];
  entityOutput.innerHTML = chips.length
    ? chips.map(([label, value]) => `<span class="chip">${escapeHtml(label)}: ${escapeHtml(value)}</span>`).join("")
    : `<span class="empty-state">No entities detected.</span>`;
}

function renderTable(results) {
  if (!results || !results.length) {
    recordsTable.innerHTML = `<tr><td colspan="6" class="empty-table">No records analyzed yet.</td></tr>`;
    return;
  }

  recordsTable.innerHTML = results.map((item, index) => {
    const source = item.source_url || `Record ${index + 1}`;
    const flags = Object.values(item.flags || {}).flat().length;
    return `
      <tr>
        <td>${escapeHtml(source)}</td>
        <td>${item.risk_percent}%</td>
        <td>${escapeHtml(item.level)}</td>
        <td>${escapeHtml((item.cities || []).join(", ") || "None")}</td>
        <td>${flags}</td>
        <td class="preview">${escapeHtml((item.text || "").slice(0, 220))}</td>
      </tr>
    `;
  }).join("");
}

function initMap() {
  if (state.map) return;
  state.map = L.map("mapCanvas", { center: [22.5, 82.0], zoom: 5 });
  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: "OpenStreetMap and CARTO",
    subdomains: "abcd",
    maxZoom: 19
  }).addTo(state.map);
}

async function refreshMap() {
  const response = await fetch("/api/map");
  const data = await response.json();
  updateMap(data.map);
}

function updateMap(points = []) {
  state.pendingMapPoints = points;
  const canvas = document.getElementById("mapCanvas");
  const notice = document.getElementById("mapNotice");
  if (!canvas.offsetParent) {
    return;
  }

  initMap();
  state.map.invalidateSize();
  if (state.heatLayer) state.map.removeLayer(state.heatLayer);
  state.markers.forEach((marker) => state.map.removeLayer(marker));
  state.markers = [];

  if (!points.length) {
    state.map.setView([22.5, 82.0], 5);
    if (notice) notice.classList.remove("is-hidden");
    return;
  }

  if (notice) notice.classList.add("is-hidden");
  state.heatLayer = L.heatLayer(
    points.map((point) => [point.lat, point.lng, point.intensity]),
    { radius: 42, blur: 32, maxZoom: 10 }
  ).addTo(state.map);

  points.forEach((point) => {
    const marker = L.circleMarker([point.lat, point.lng], {
      radius: Math.min(22, 8 + point.count * 3),
      color: "#c93636",
      fillColor: "#c93636",
      fillOpacity: 0.72,
      weight: 2
    }).addTo(state.map);
    marker.bindPopup(`${escapeHtml(point.city)}<br>Flagged mentions: ${point.count}`);
    state.markers.push(marker);
  });

  const bounds = L.latLngBounds(points.map((point) => [point.lat, point.lng]));
  state.map.fitBounds(bounds.pad(0.28), { maxZoom: 7 });
}

function setBusy(message) {
  resultOutput.innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
}

function showError(message) {
  resultOutput.innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
