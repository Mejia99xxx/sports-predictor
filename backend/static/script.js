/**
 * Frontend-driven API calls.
 * All calls to API-Football are made directly from the browser
 * to bypass cloud server IP restrictions on the free plan.
 */

let API_KEY = "";
let API_BASE = "https://v3.football.api-sports.io";

// ── Load API config from server ───────────────────────────────────────────
async function loadApiConfig() {
  try {
    const res = await fetch("/api-config");
    const cfg = await res.json();
    API_KEY = cfg.api_key;
    API_BASE = cfg.base_url;
  } catch (e) {
    console.error("No se pudo cargar la configuración de la API:", e);
  }
}

function apiHeaders() {
  return { "x-apisports-key": API_KEY };
}

document.addEventListener("DOMContentLoaded", async function () {
  const ligaSelect       = document.getElementById("liga_id");
  const localSelect      = document.getElementById("local_team_id");
  const visitanteSelect  = document.getElementById("visitante_team_id");
  const localNameInput   = document.getElementById("local_team_name");
  const visitanteNameInput = document.getElementById("visitante_team_name");
  const teamsError       = document.getElementById("team-error");
  const ligaError        = document.getElementById("liga-error");
  const form             = document.querySelector("form[action='/predecir/equipos']");

  if (!ligaSelect) return;

  // Load API config first
  await loadApiConfig();

  // ── Helpers ──────────────────────────────────────────────────────────────
  function setPlaceholder(select, text, disabled = true) {
    select.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = text;
    opt.disabled = true; opt.selected = true;
    select.appendChild(opt);
    select.disabled = disabled;
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

  function showError(el, msg) { if (el) { el.textContent = msg; el.style.display = "block"; } }
  function hideError(el)      { if (el) { el.textContent = ""; el.style.display = "none"; } }

  // ── Load leagues from hardcoded /ligas endpoint ───────────────────────────
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

  // ── League change → load teams directly from API-Football ────────────────
  ligaSelect.addEventListener("change", async function () {
    const leagueId = this.value;
    if (!leagueId) return;

    setPlaceholder(localSelect, "Cargando equipos…");
    setPlaceholder(visitanteSelect, "Cargando equipos…");
    if (localNameInput) localNameInput.value = "";
    if (visitanteNameInput) visitanteNameInput.value = "";
    hideError(teamsError);

    try {
      // First try hardcoded endpoint
      let res = await fetch("/equipos?league_id=" + encodeURIComponent(leagueId));
      let data = await res.json();
      let teams = data.response || [];

      // If empty, try directly from API-Football
      if (teams.length === 0 && API_KEY) {
        // Determine best season to try
        const seasons = [2025, 2024, 2023, 2022];
        for (const season of seasons) {
          const apiRes = await fetch(
            `${API_BASE}/teams?league=${leagueId}&season=${season}`,
            { headers: apiHeaders() }
          );
          const apiData = await apiRes.json();
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

  // ── Team selection → update hidden name inputs ────────────────────────────
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

  // ── Form submit → fetch stats client-side, send vector to server ──────────
  if (form) {
    form.addEventListener("submit", async function (e) {
      e.preventDefault();

      const localId      = localSelect ? localSelect.value : "";
      const visitanteId  = visitanteSelect ? visitanteSelect.value : "";
      const localName    = localNameInput ? localNameInput.value : localId;
      const visitanteName = visitanteNameInput ? visitanteNameInput.value : visitanteId;

      if (!localId || !visitanteId) {
        showError(teamsError, "Selecciona ambos equipos antes de predecir.");
        return;
      }
      if (localId === visitanteId) {
        showError(teamsError, "Los equipos local y visitante deben ser diferentes.");
        return;
      }

      showError(teamsError, "Obteniendo estadísticas... puede tardar unos segundos.");

      try {
        // Fetch stats for both teams directly from API-Football
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

        // Build prediction vector
        const vector = buildVector(statsLocal, statsVisitante);

        // Send to server for ML prediction
        hideError(teamsError);
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

        if (result.error) {
          showError(teamsError, result.error);
          return;
        }

        // Show result
        document.body.innerHTML = `
          <div class="container" style="text-align:center;padding:2rem;">
            <h1>Resultado del Pronóstico</h1>
            <p style="font-size:1.1rem;">${result.equipo_local} vs ${result.equipo_visitante}</p>
            <p>El modelo predice: <strong style="font-size:1.3rem;">${result.resultado}</strong></p>
            <a href="/inicio" style="display:inline-block;margin-top:1rem;color:#007bff;">← Volver</a>
          </div>`;

      } catch (err) {
        showError(teamsError, "Error al procesar la predicción: " + err.message);
      }
    });
  }

  await cargarLigas();
});

// ── Fetch most recent match stats for a team directly from API-Football ──────
async function fetchTeamStats(teamId) {
  const seasons = [2025, 2024, 2023, 2022];
  for (const season of seasons) {
    try {
      const res = await fetch(
        `${API_BASE}/fixtures?team=${teamId}&season=${season}`,
        { headers: apiHeaders() }
      );
      const data = await res.json();
      const fixtures = (data.response || []).filter(
        f => f.fixture && f.fixture.status && f.fixture.status.short === "FT"
      );
      if (fixtures.length === 0) continue;

      // Sort by date desc, try most recent with stats
      fixtures.sort((a, b) => b.fixture.date.localeCompare(a.fixture.date));

      for (const fixture of fixtures.slice(0, 5)) {
        const fid = fixture.fixture.id;
        const statsRes = await fetch(
          `${API_BASE}/fixtures/statistics?fixture=${fid}`,
          { headers: apiHeaders() }
        );
        const statsData = await statsRes.json();
        if (statsData.response && statsData.response.length >= 2) {
          return { stats: statsData.response, fixture };
        }
      }
    } catch (e) {
      continue;
    }
  }
  return null;
}

// ── Build the 9-value prediction vector from two teams' stats ────────────────
function buildVector(localData, visitanteData) {
  function getStat(stats, teamIndex, type) {
    const team = stats[teamIndex];
    if (!team) return 0;
    const stat = team.statistics.find(s => s.type === type);
    if (!stat || stat.value === null || stat.value === undefined) return 0;
    const val = String(stat.value).replace("%", "").trim();
    return parseFloat(val) || 0;
  }

  function getGoals(fixtureData, teamId) {
    const f = fixtureData.fixture;
    const teams = fixtureData.fixture_teams || fixtureData.teams;
    if (!teams) return 0;
    if (teams.home && teams.home.id === teamId) return fixtureData.goals ? (fixtureData.goals.home || 0) : 0;
    if (teams.away && teams.away.id === teamId) return fixtureData.goals ? (fixtureData.goals.away || 0) : 0;
    return 0;
  }

  const ls = localData.stats;
  const vs = visitanteData.stats;

  const posLocal     = getStat(ls, 0, "Ball Possession");
  const posVisitante = getStat(vs, 0, "Ball Possession");
  const tirosLocal   = getStat(ls, 0, "Total Shots");
  const tirosVis     = getStat(vs, 0, "Total Shots");
  const faltasLocal  = getStat(ls, 0, "Fouls");
  const faltasVis    = getStat(vs, 0, "Fouls");
  const tarjLocal    = getStat(ls, 0, "Yellow Cards");
  const tarjVis      = getStat(vs, 0, "Yellow Cards");

  // Goal difference from their respective last matches
  const golesLocal = (localData.fixture.goals && localData.fixture.goals.home !== null)
    ? (localData.fixture.goals.home || 0) : 0;
  const golesVis   = (visitanteData.fixture.goals && visitanteData.fixture.goals.home !== null)
    ? (visitanteData.fixture.goals.away || 0) : 0;
  const difGoles   = golesLocal - golesVis;

  return [posLocal, posVisitante, tirosLocal, tirosVis, faltasLocal, faltasVis, tarjLocal, tarjVis, difGoles];
}
