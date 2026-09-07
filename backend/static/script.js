document.addEventListener("DOMContentLoaded", function () {
  const ligaSelect = document.getElementById("liga_id");
  const localSelect = document.getElementById("local_team_id");
  const visitanteSelect = document.getElementById("visitante_team_id");
  const localNameInput = document.getElementById("local_team_name");
  const visitanteNameInput = document.getElementById("visitante_team_name");
  const teamsError = document.getElementById("team-error");
  const ligaError = document.getElementById("liga-error");

  if (!ligaSelect) return;

  function setPlaceholder(select, text) {
    select.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = text;
    opt.disabled = true;
    opt.selected = true;
    select.appendChild(opt);
  }

  function populateTeams(select, teams) {
    select.innerHTML = "";
    if (teams.length === 0) {
      setPlaceholder(select, "No se encontraron equipos para esta liga");
      return;
    }
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = "-- Seleccionar equipo --";
    placeholder.disabled = true;
    placeholder.selected = true;
    select.appendChild(placeholder);
    teams.forEach(function (item) {
      const team = item.team;
      const opt = document.createElement("option");
      opt.value = team.id;
      opt.textContent = team.name;
      select.appendChild(opt);
    });
  }

  function showTeamError(msg) {
    if (teamsError) { teamsError.textContent = msg; teamsError.style.display = "block"; }
  }
  function clearTeamError() {
    if (teamsError) { teamsError.textContent = ""; teamsError.style.display = "none"; }
  }

  // ── Load leagues on page load ─────────────────────────────────────────────
  async function cargarLigas() {
    try {
      const res = await fetch("/ligas");
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();
      const ligas = data.response || [];

      ligaSelect.innerHTML = "";
      if (ligas.length === 0) {
        setPlaceholder(ligaSelect, "No se encontraron ligas");
        if (ligaError) { ligaError.textContent = "No se pudieron cargar las ligas. Intenta de nuevo más tarde."; ligaError.style.display = "block"; }
        return;
      }

      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "-- Selecciona una liga --";
      placeholder.disabled = true;
      placeholder.selected = true;
      ligaSelect.appendChild(placeholder);

      ligas.forEach(function (item) {
        const opt = document.createElement("option");
        opt.value = item.league.id;
        opt.textContent = item.league.name + " (" + item.country.name + ")";
        ligaSelect.appendChild(opt);
      });

      ligaSelect.disabled = false;
      if (ligaError) ligaError.style.display = "none";
    } catch (err) {
      setPlaceholder(ligaSelect, "Error al cargar ligas");
      if (ligaError) { ligaError.textContent = "No se pudieron cargar las ligas. Intenta de nuevo más tarde."; ligaError.style.display = "block"; }
    }
  }

  cargarLigas();

  // ── League change → load teams ────────────────────────────────────────────
  ligaSelect.addEventListener("change", async function () {
    const leagueId = this.value;
    if (!leagueId) return;

    localSelect.disabled = true;
    visitanteSelect.disabled = true;
    setPlaceholder(localSelect, "Cargando equipos…");
    setPlaceholder(visitanteSelect, "Cargando equipos…");
    if (localNameInput) localNameInput.value = "";
    if (visitanteNameInput) visitanteNameInput.value = "";
    clearTeamError();

    try {
      const response = await fetch("/equipos?league_id=" + encodeURIComponent(leagueId));
      if (!response.ok) throw new Error("HTTP " + response.status);
      const data = await response.json();
      const teams = data.response || [];
      populateTeams(localSelect, teams);
      populateTeams(visitanteSelect, teams);
      if (teams.length > 0) {
        localSelect.disabled = false;
        visitanteSelect.disabled = false;
      }
    } catch (err) {
      setPlaceholder(localSelect, "Error al cargar equipos");
      setPlaceholder(visitanteSelect, "Error al cargar equipos");
      showTeamError("No se pudieron cargar los equipos. Por favor, inténtelo de nuevo.");
    }
  });

  // ── Team selection → update hidden name inputs ────────────────────────────
  if (localSelect) {
    localSelect.addEventListener("change", function () {
      const selected = this.options[this.selectedIndex];
      if (localNameInput && selected && selected.value) localNameInput.value = selected.textContent;
    });
  }
  if (visitanteSelect) {
    visitanteSelect.addEventListener("change", function () {
      const selected = this.options[this.selectedIndex];
      if (visitanteNameInput && selected && selected.value) visitanteNameInput.value = selected.textContent;
    });
  }
});
