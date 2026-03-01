async function fetchJson(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed ${url}: ${res.status}`);
    return await res.json();
  }
  
  function parseCellId(cell_id) {
    // must be "lat_lon"
    const [latStr, lonStr] = cell_id.split("_");
    return { lat: parseFloat(latStr), lon: parseFloat(lonStr) };
  }
  
  function classLabel(c) {
    if (c === 0) return "6.0–6.9";
    if (c === 1) return "7.0–7.9";
    if (c === 2) return "8.0+";
    return "N/A";
  }
  
  async function render() {
    const gd = document.getElementById("dashboard-container");
  
    // Load predictions + live quakes
    const preds = await fetchJson("/api/predictions/latest");
    const live = await fetchJson("/api/live");
  
    // Build prediction arrays
    const lat = [];
    const lon = [];
    const risk = [];
    const hover = [];
    const isAlert = [];
    const alertLat = [];
    const alertLon = [];
    const alertText = [];
  
    preds.forEach((r) => {
      const { lat: la, lon: lo } = parseCellId(r.cell_id);
      lat.push(la);
      lon.push(lo);
      risk.push(r.risk_prob);
  
      hover.push(
        `<b>Cell:</b> ${r.cell_id}` +
        `<br><b>Risk:</b> ${Number(r.risk_prob).toFixed(2)}` +
        `<br><b>Predicted quake:</b> ${r.predicted_quake}` +
        `<br><b>Magnitude:</b> ${classLabel(r.predicted_class)}`
      );
  
      if (Number(r.predicted_quake) === 1) {
        alertLat.push(la);
        alertLon.push(lo);
        alertText.push(`⚠️ ${classLabel(r.predicted_class)}`);
        isAlert.push(true);
      } else {
        isAlert.push(false);
      }
    });
  
    // Prediction layer
    const tracePred = {
      type: "scattermapbox",
      mode: "markers",
      name: "Predicted Risk Zones",
      lat,
      lon,
      text: hover,
      hoverinfo: "text",
      marker: {
        size: 14,
        opacity: 0.75,
        color: risk,
        colorscale: "Viridis",
        showscale: true,
        colorbar: {
          title: { text: "Risk Level" },
          thickness: 20,
          x: 0.9
        }
      }
    };
  
    // Alert overlay
    const traceAlert = {
      type: "scattermapbox",
      mode: "markers+text",
      name: "Alert Zones",
      lat: alertLat,
      lon: alertLon,
      text: alertText,
      textposition: "top center",
      hoverinfo: "none",
      marker: { size: 12, color: "white" },
      textfont: { size: 12, color: "white" }
    };
  
    // Live quake overlay
    const liveLat = live.map((q) => q.lat);
    const liveLon = live.map((q) => q.lon);
    const liveHover = live.map((q) =>
      `<b>LIVE QUAKE</b><br>Mag: ${q.mag}<br>Time: ${q.time}`
    );
  
    const traceLive = {
      type: "scattermapbox",
      mode: "markers",
      name: "Live Quakes (24h)",
      lat: liveLat,
      lon: liveLon,
      text: liveHover,
      hoverinfo: "text",
      marker: { size: 9, color: "cyan", opacity: 0.9 }
    };
  
    const layout = {
      template: "plotly_dark",
      mapbox: {
        style: "carto-darkmatter",
        center: { lat: 20, lon: 0 },
        zoom: 1.5
      },
      margin: { r: 0, t: 60, l: 0, b: 0 },
      title: { text: "🌍 Global Earthquake Risk & Live Monitoring", x: 0.05 },
      legend: {
        yanchor: "top",
        y: 0.99,
        xanchor: "left",
        x: 0.01,
        bgcolor: "rgba(0,0,0,0.5)"
      },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)"
    };
  
    Plotly.newPlot(gd, [tracePred, traceAlert, traceLive], layout, { responsive: true });
    window.onresize = () => Plotly.Plots.resize(gd);
  }
  
  document.addEventListener("DOMContentLoaded", () => {
    render().catch((err) => {
      console.error(err);
      const gd = document.getElementById("dashboard-container");
      gd.innerHTML = `<div style="padding:16px;color:white;">Failed to load dashboard data. Check the server console.</div>`;
    });
  });