/**
 * Async league-change handler.
 * When the user selects a league, fetches the teams for that league
 * and populates both the local and visitante team dropdowns.
 * Also updates the hidden team-name inputs when a team is selected.
 */

document.addEventListener("DOMContentLoaded", function () {
  const ligaSelect = document.getElementById("liga_id");
  const localSelect = document.getElementById("local_team_id");
  const visitanteSelect = document.getElementById("visitante_team_id");
  const localNameInput = document.getElementById("local_team_name");
  const visitanteNameInput = document.getElementById("visitante_team_name");
  // Error element defined in index.html
  const teamsError = document.getElementById("team-error");

  // Guard: only wire up if the league selector exists in the page
  if (!ligaSelect) return;

  // Helper: clear a select and insert a single placeholder option
  function setPlaceholder(select, text) {
    select.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = text;
    opt.disabled = true;
    opt.selected = true;
    select.appendChild(opt);
  }

  // Helper: populate a select with a list of teams
  function populateTeams(select, teams) {
    select.innerHTML = "";

    if (teams.length === 0) {
      setPlaceholder(select, "No se encontraron equipos para esta liga");
      return;
    }

    // Leading placeholder
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

  // Helper: show/hide the error message element
  function showError(message) {
    if (teamsError) {
      teamsError.textContent = message;
      teamsError.style.display = "block";
    }
  }

  function clearError() {
    if (teamsError) {
      teamsError.textContent = "";
      teamsError.style.display = "none";
    }
  }

  // Liga change → fetch teams asynchronously
  ligaSelect.addEventListener("change", async function () {
    const leagueId = this.value;
    if (!leagueId) return;

    // Disable dropdowns and show loading state
    localSelect.disabled = true;
    visitanteSelect.disabled = true;
    setPlaceholder(localSelect, "Cargando equipos…");
    setPlaceholder(visitanteSelect, "Cargando equipos…");

    // Clear hidden name inputs
    if (localNameInput) localNameInput.value = "";
    if (visitanteNameInput) visitanteNameInput.value = "";

    // Clear any previous error message
    clearError();

    try {
      const response = await fetch(`/equipos?league_id=${encodeURIComponent(leagueId)}`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: Error al cargar los equipos`);
      }

      const data = await response.json();
      const teams = data.response || [];

      populateTeams(localSelect, teams);
      populateTeams(visitanteSelect, teams);

      // Re-enable dropdowns only when there are teams to choose from
      if (teams.length > 0) {
        localSelect.disabled = false;
        visitanteSelect.disabled = false;
      }
    } catch (err) {
      // Keep dropdowns disabled; show an error message in #team-error
      setPlaceholder(localSelect, "Error al cargar equipos");
      setPlaceholder(visitanteSelect, "Error al cargar equipos");

      showError("No se pudieron cargar los equipos. Por favor, inténtelo de nuevo.");
    }
  });

  // Local team selection → update hidden input with team name
  if (localSelect) {
    localSelect.addEventListener("change", function () {
      const selected = this.options[this.selectedIndex];
      if (localNameInput && selected && selected.value) {
        localNameInput.value = selected.textContent;
      }
    });
  }

  // Visitante team selection → update hidden input with team name
  if (visitanteSelect) {
    visitanteSelect.addEventListener("change", function () {
      const selected = this.options[this.selectedIndex];
      if (visitanteNameInput && selected && selected.value) {
        visitanteNameInput.value = selected.textContent;
      }
    });
  }
});
