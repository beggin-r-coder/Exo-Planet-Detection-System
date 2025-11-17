async function fetchVisualization() {
  const res = await fetch("/visualize");
  const data = await res.json();
  return data;
}

async function fetchSummary() {
  const res = await fetch("/summary");
  return await res.json();
}

let chartInstance = null;

function renderChart(ctx, time, flux) {
  const avg = flux.reduce((a, b) => a + b, 0) / (flux.length || 1);
  const avgLine = new Array(flux.length).fill(avg);
  
  // Destroy existing chart if it exists
  if (chartInstance) {
    chartInstance.destroy();
  }
  
  chartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: time,
      datasets: [
        {
          label: "Flux",
          data: flux,
          borderColor: "rgba(96,165,250,1)",
          backgroundColor: "rgba(96,165,250,0.2)",
          tension: 0.25,
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
        },
        {
          label: "Average",
          data: avgLine,
          borderColor: "rgba(148,163,184,0.8)",
          borderDash: [6, 6],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      animation: { duration: 600 },
      plugins: {
        legend: { labels: { color: "#e2e8f0" } },
        tooltip: { enabled: true },
        zoom: {
          zoom: {
            wheel: {
              enabled: true,
              speed: 0.1,
            },
            pinch: {
              enabled: true,
            },
            mode: "xy",
          },
          pan: {
            enabled: true,
            mode: "xy",
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Time", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148,163,184,0.12)" },
        },
        y: {
          title: { display: true, text: "Flux", color: "#94a3b8" },
          ticks: { color: "#94a3b8" },
          grid: { color: "rgba(148,163,184,0.12)" },
        },
      },
    },
  });
  
  return chartInstance;
}

async function init() {
  const chartEl = document.getElementById("fluxChart");
  if (chartEl) {
    const data = await fetchVisualization();
    if (!data || data.error) {
      console.error("Visualization error", data && data.error);
      return;
    }
    renderChart(chartEl.getContext("2d"), data.time, data.flux);
    
    // Add reset zoom button functionality
    const resetZoomBtn = document.getElementById("resetZoom");
    if (resetZoomBtn && chartInstance) {
      resetZoomBtn.addEventListener("click", () => {
        if (chartInstance) {
          chartInstance.resetZoom();
        }
      });
    }
  }

  const summary = await fetchSummary();
  if (summary) {
    const conf = document.getElementById("conf");
    const minFlux = document.getElementById("minFlux");
    const avgFlux = document.getElementById("avgFlux");
    const dataPoints = document.getElementById("dataPoints");
    const interpretation = document.getElementById("interpretation");
    const period = document.getElementById("period");
    const depth = document.getElementById("depth");
    const duration = document.getElementById("duration");
    if (conf)
      conf.textContent =
        summary.detection_confidence != null
          ? summary.detection_confidence + "%"
          : "—";
    if (minFlux)
      minFlux.textContent = summary.min_flux != null ? summary.min_flux : "—";
    if (avgFlux)
      avgFlux.textContent = summary.avg_flux != null ? summary.avg_flux : "—";
    if (dataPoints)
      dataPoints.textContent = summary.count != null ? summary.count : "—";
    if (interpretation) {
      const confVal = summary.detection_confidence || 0;
      interpretation.textContent =
        confVal >= 50
          ? "High detection probability. Periodic dips suggest a likely transit."
          : "Low detection probability. Flux variations may be due to stellar activity.";
    }
    if (period)
      period.textContent =
        summary.estimated_period != null ? summary.estimated_period : "—";
    if (depth)
      depth.textContent =
        summary.transit_depth != null ? summary.transit_depth : "—";
    if (duration)
      duration.textContent =
        summary.transit_duration != null ? summary.transit_duration : "—";
  }
}

document.addEventListener("DOMContentLoaded", init);

// Dark mode functionality
document.addEventListener("DOMContentLoaded", () => {
  const themeToggle = document.getElementById("themeToggle");
  const body = document.body;

  if (themeToggle) {
    // Load saved theme
    const savedTheme = localStorage.getItem("theme") || "dark";
    body.setAttribute("data-theme", savedTheme);
    updateThemeIcon(savedTheme);

    themeToggle.addEventListener("click", () => {
      const currentTheme = body.getAttribute("data-theme");
      const newTheme = currentTheme === "dark" ? "light" : "dark";
      body.setAttribute("data-theme", newTheme);
      localStorage.setItem("theme", newTheme);
      updateThemeIcon(newTheme);
    });

    function updateThemeIcon(theme) {
      themeToggle.textContent = theme === "dark" ? "🌙" : "☀️";
    }
  }
});
