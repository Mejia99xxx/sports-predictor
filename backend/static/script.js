/**
 * Frontend script — all API-Football calls go through the Flask proxy
 * at /proxy/api-football?path=... to avoid CORS and IP restrictions.
 */

// ── Proxy helper ─────────────────────────────────────────────────────────────
async function apiFootball(path) {
  const res = await fetch("/proxy/api-football?path=" + encodeURIComponent(path));
  if (!res.ok) throw new Error("Proxy error: " + res.status);
  return res.json();
}

document.addEventListener("DOMContentLoaded", async function () {
  const ligaSelect        = document.getElementById("liga_id");
  const localSelect       = document.getElementById("local_team_id");
  const visitanteSelect   = document.getElementById("visitante_team_id");
  const localNameInput    = document.getElementById("local_team_name");
  const visitanteNameInput = document.getElementById("visitante_team_name");
  const teamsError        = document.getElementById("team-error");
  const ligaError         = document.getElementById("liga-error");
  const infoMsg           = document.getElementById("info-msg");
  const form              = document.getElementById("predict-form");

  if (!ligaSelect) return;

  // ── Helpers ─────────────────────────────────────────────────────────────
  function setPlaceholder(select, text) {
    select.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = text;
    opt.disabled = true; opt.selected = true;
    select.appendChild(opt);
    select.disabled = true;
  }

  function populateTeams(select, teams) {
    select.innerHTML = "";
    if (!teams || teams.length === 0) {
      setPlaceholder(select, "No se encontraron equipos para esta liga");
      return;
    }
    const ph = document.createElement("option");
    ph.value = ""; ph.textContent = "-- Seleccionar equipo --";
    ph.disabled = true; ph.selected = true;
    select.appendChild(ph);
    teams.forEach(item => {
      const t = item.team;
      const opt = document.createElement("option");
      opt.value = t.id; opt.textContent = t.name;
      select.appendChild(opt);
    });
    select.disabled = false;
  }

  function showError(el, msg) { if (el) { el.textContent = msg; el.classList.add("visible"); } }
  function hideError(el)      { if (el) { el.textContent = ""; el.classList.remove("visible"); } }
  function showInfo(msg)      { if (infoMsg) { infoMsg.textContent = msg; infoMsg.classList.add("visible"); } }
  function hideInfo()         { if (infoMsg) { infoMsg.textContent = ""; infoMsg.classList.remove("visible"); } }

  // ── Load leagues from hardcoded server endpoint ─────────────────────────
  async function cargarLigas() {
    try {
      const res = await fetch("/ligas");
      const data = await res.json();
      const ligas = data.response || [];

      ligaSelect.innerHTML = "";
      if (ligas.length === 0) {
        setPlaceholder(ligaSelect, "No se encontraron ligas");
        showError(ligaError, "No se pudieron cargar las ligas.");
        return;
      }
      const ph = document.createElement("option");
      ph.value = ""; ph.textContent = "-- Selecciona una liga --";
      ph.disabled = true; ph.selected = true;
      ligaSelect.appendChild(ph);

      ligas.forEach(item => {
        const opt = document.createElement("option");
        opt.value = item.league.id;
        opt.textContent = item.league.name + " (" + item.country.name + ")";
        ligaSelect.appendChild(opt);
      });
      ligaSelect.disabled = false;
      hideError(ligaError);
    } catch (err) {
      setPlaceholder(ligaSelect, "Error al cargar ligas");
      showError(ligaError, "No se pudieron cargar las ligas.");
    }
  }

  // ── League change → load teams via proxy or hardcoded ───────────────────
  ligaSelect.addEventListener("change", async function () {
    const leagueId = this.value;
    if (!leagueId) return;

    setPlaceholder(localSelect, "Cargando equipos…");
    setPlaceholder(visitanteSelect, "Cargando equipos…");
    if (localNameInput) localNameInput.value = "";
    if (visitanteNameInput) visitanteNameInput.value = "";
    hideError(teamsError);

    try {
      // Try hardcoded endpoint first (fastest)
      const res = await fetch("/equipos?league_id=" + encodeURIComponent(leagueId));
      const data = await res.json();
      let teams = data.response || [];

      // If empty, try via proxy (slower but more complete)
      if (teams.length === 0) {
        const seasons = [2025, 2024, 2023, 2022];
        for (const season of seasons) {
          const apiData = await apiFootball(`/teams?league=${leagueId}&season=${season}`);
          if (apiData.response && apiData.response.length > 0) {
            teams = apiData.response;
            break;
          }
        }
      }

      populateTeams(localSelect, teams);
      populateTeams(visitanteSelect, teams);

      if (teams.length === 0) {
        showError(teamsError, "No se encontraron equipos para esta liga.");
      }
    } catch (err) {
      setPlaceholder(localSelect, "Error al cargar equipos");
      setPlaceholder(visitanteSelect, "Error al cargar equipos");
      showError(teamsError, "No se pudieron cargar los equipos.");
    }
  });

  // ── Team selection → update hidden name inputs ───────────────────────────
  if (localSelect) {
    localSelect.addEventListener("change", function () {
      const sel = this.options[this.selectedIndex];
      if (localNameInput && sel && sel.value) localNameInput.value = sel.textContent;
    });
  }
  if (visitanteSelect) {
    visitanteSelect.addEventListener("change", function () {
      const sel = this.options[this.selectedIndex];
      if (visitanteNameInput && sel && sel.value) visitanteNameInput.value = sel.textContent;
    });
  }

  // ── Form submit → fetch stats via proxy, send vector to server ───────────
  if (form) {
    form.addEventListener("submit", async function (e) {
      e.preventDefault();

      const localId       = localSelect ? localSelect.value : "";
      const visitanteId   = visitanteSelect ? visitanteSelect.value : "";
      const localName     = localNameInput ? localNameInput.value : localId;
      const visitanteName = visitanteNameInput ? visitanteNameInput.value : visitanteId;

      if (!localId || !visitanteId) {
        showError(teamsError, "Selecciona ambos equipos antes de predecir.");
        return;
      }
      if (localId === visitanteId) {
        showError(teamsError, "Los equipos local y visitante deben ser diferentes.");
        return;
      }

      showInfo("⏳ Obteniendo estadísticas... puede tardar unos segundos.");
      hideError(teamsError);

      try {
        const [statsLocal, statsVisitante] = await Promise.all([
          fetchTeamStats(parseInt(localId)),
          fetchTeamStats(parseInt(visitanteId))
        ]);

        if (!statsLocal) {
          showError(teamsError, `No hay partidos recientes disponibles para ${localName}.`);
          return;
        }
        if (!statsVisitante) {
          showError(teamsError, `No hay partidos recientes disponibles para ${visitanteName}.`);
          return;
        }

        const vector = buildVector(statsLocal, statsVisitante);
        hideError(teamsError);
        hideInfo();

        const res = await fetch("/predecir/vector", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            vector: vector,
            equipo_local: localName,
            equipo_visitante: visitanteName
          })
        });

        const result = await res.json();
        if (result.error) { showError(teamsError, result.error); return; }

        // Show result with new design
        const overlay = document.getElementById("result-overlay");
        const content = document.getElementById("result-content");
        if (overlay && content) {
          const r = result.resultado;
          let cssClass = "draw";
          let trophy = "🤝";
          if (r === "Gana equipo local")    { cssClass = "win-home"; trophy = "🏆"; }
          if (r === "Gana equipo visitante") { cssClass = "win-away"; trophy = "🏆"; }

          content.innerHTML = `
            <div class="trophy">${trophy}</div>
            <div class="teams">
              <span class="team-name">${result.equipo_local}</span>
              <span style="color:#aaa;margin:0 0.5rem;">vs</span>
              <span class="team-name">${result.equipo_visitante}</span>
            </div>
            <div class="prediction-box">
              <div class="prediction-label">El modelo predice</div>
              <div class="prediction-value ${cssClass}">${r}</div>
            </div>
            <a href="/inicio" class="btn-back">← Nueva predicción</a>`;

          overlay.style.display = "flex";
        } else {
          document.body.innerHTML = `
            <div class="container">
              <div class="card-header"><div class="logo">⚽</div><h1>Pronosticador de Fútbol</h1></div>
              <div class="result-card">
                <div class="trophy">🏆</div>
                <div class="teams"><span class="team-name">${result.equipo_local}</span> vs <span class="team-name">${result.equipo_visitante}</span></div>
                <div class="prediction-box">
                  <div class="prediction-label">El modelo predice</div>
                  <div class="prediction-value">${result.resultado}</div>
                </div>
                <a href="/inicio" class="btn-back">← Nueva predicción</a>
              </div>
            </div>`;
        }

      } catch (err) {
        showError(teamsError, "Error al procesar la predicción: " + err.message);
      }
    });
  }

  await cargarLigas();
});

// ── Fetch most recent match stats via proxy ──────────────────────────────────
async function fetchTeamStats(teamId) {
  const seasons = [2025, 2024, 2023, 2022];
  for (const season of seasons) {
    try {
      const data = await apiFootball(`/fixtures?team=${teamId}&season=${season}`);
      const fixtures = (data.response || []).filter(
        f => f.fixture && f.fixture.status && f.fixture.status.short === "FT"
      );
      if (fixtures.length === 0) continue;

      fixtures.sort((a, b) => b.fixture.date.localeCompare(a.fixture.date));

      for (const fixture of fixtures.slice(0, 5)) {
        const fid = fixture.fixture.id;
        const statsData = await apiFootball(`/fixtures/statistics?fixture=${fid}`);
        if (statsData.response && statsData.response.length >= 2) {
          return { stats: statsData.response, fixture };
        }
      }
    } catch (e) { continue; }
  }
  return null;
}

// ── Build 9-value prediction vector ─────────────────────────────────────────
function buildVector(localData, visitanteData) {
  function getStat(stats, idx, type) {
    const team = stats[idx];
    if (!team) return 0;
    const stat = team.statistics.find(s => s.type === type);
    if (!stat || stat.value === null || stat.value === undefined) return 0;
    return parseFloat(String(stat.value).replace("%", "").trim()) || 0;
  }

  const ls = localData.stats;
  const vs = visitanteData.stats;

  const golesLocal = localData.fixture.goals ? (localData.fixture.goals.home || 0) : 0;
  const golesVis   = visitanteData.fixture.goals ? (visitanteData.fixture.goals.away || 0) : 0;

  return [
    getStat(ls, 0, "Ball Possession"),
    getStat(vs, 0, "Ball Possession"),
    getStat(ls, 0, "Total Shots"),
    getStat(vs, 0, "Total Shots"),
    getStat(ls, 0, "Fouls"),
    getStat(vs, 0, "Fouls"),
    getStat(ls, 0, "Yellow Cards"),
    getStat(vs, 0, "Yellow Cards"),
    golesLocal - golesVis
  ];
}
