// Prophit dashboard — vanilla JS, zero frameworks (SRD 7.1).
// Renders docs/daily_slate.json produced by src/publication/.

const STAT_LABELS = {
  pts: "PTS",
  reb: "REB",
  ast: "AST",
  fg3m: "3PM",
  pra: "PRA",
  pr: "P+R",
  pa: "P+A",
};

function evChipClass(ev) {
  if (ev == null) return "ev-neutral";
  if (ev >= 5.0) return "ev-pos";
  if (ev >= 0) return "ev-neutral";
  return "ev-caution";
}

function fmtSigned(v, digits = 1) {
  if (v == null) return "—";
  return `${v >= 0 ? "+" : ""}${Number(v).toFixed(digits)}`;
}

function fmtOdds(american) {
  if (american == null) return "—";
  return american > 0 ? `+${american}` : `${american}`;
}

function cardHtml(play) {
  const stat = STAT_LABELS[play.stat] || play.stat;
  const loc = play.is_home ? "vs" : "@";
  const ev =
    play.recommended_ev_pct ??
    Math.max(play.ev_over_pct ?? -999, play.ev_under_pct ?? -999);
  const evShown = ev === -999 ? null : ev;
  const side =
    play.recommended_side === "under"
      ? "Under"
      : play.recommended_side === "over"
        ? "Over"
        : null;
  const chipText = side
    ? `${side} ${fmtSigned(evShown)}% EV`
    : `${fmtSigned(evShown)}% EV`;

  const f = play.factors || {};
  const facts = [
    ["Proj. Min", f.projected_minutes],
    ["DvP", f.dvp_label],
    ["Rest", f.rest_label],
    ["Win Prob", `${(play.p_over * 100).toFixed(1)}%`],
    ["Fair Over", fmtOdds(play.fair_over_american)],
    [
      "Book",
      `O ${fmtOdds(play.over_price)} / U ${fmtOdds(play.under_price)}`,
    ],
  ]
    .filter(([, v]) => v != null)
    .map(([k, v]) => `<div>${k}: <b>${v}</b></div>`)
    .join("");

  const m = play.milestone || {};
  const ladder = m.triggered
    ? `<div class="ladder">
         ${m.milestones
           .map(
             (r) => `<div class="ladder-rung">
               <span class="ladder-k">${r.k}+ ${stat}</span>
               <span>Fair <b class="ladder-fair">${fmtOdds(r.fair_american)}</b>
                 (${(r.p_over * 100).toFixed(1)}%)</span>
               <span class="ladder-target">Target ${fmtOdds(r.alt_target_american)}+</span>
             </div>`,
           )
           .join("")}
       </div>`
    : "";

  return `<article class="play-card" data-stat="${play.stat}"
      data-value="${play.recommended_side ? 1 : 0}"
      data-milestone="${m.triggered ? 1 : 0}">
    <div class="card-top">
      <div>
        <div class="player-name">${play.player_name}</div>
        <div class="player-meta">${play.position || ""} ${play.team_abbr || ""} ${loc} ${play.opponent_abbr || ""}</div>
      </div>
      <span class="chip ${evChipClass(evShown)}">${chipText}</span>
    </div>
    <div class="line-row">
      <span>Line: <b>${play.line} ${stat}</b></span>
      <span class="model">Model: ${play.model_mean}</span>
    </div>
    <div class="prob-row">
      <span>Over ${(play.p_over * 100).toFixed(1)}%</span>
      <span>Under ${(play.p_under * 100).toFixed(1)}%</span>
      <span>Edge ${fmtSigned(play.model_edge)}</span>
    </div>
    ${m.triggered ? '<div class="milestone-flag">Milestone alert</div>' : ""}
    <details class="tray"><summary>Factors</summary>
      <div class="fact-grid">${facts}</div>
    </details>
    ${m.triggered ? `<details class="tray"><summary>Milestone ladder</summary>${ladder}</details>` : ""}
  </article>`;
}

function applyFilter(filter) {
  document.querySelectorAll(".play-card").forEach((card) => {
    const show =
      filter === "all" ||
      (filter === "value" && card.dataset.value === "1") ||
      (filter === "milestones" && card.dataset.milestone === "1") ||
      card.dataset.stat === filter;
    card.style.display = show ? "" : "none";
  });
}

function renderSlate(slate) {
  const playsEl = document.getElementById("plays");
  document.getElementById("slate-date").textContent = slate.date || "";
  document.getElementById("slate-summary").textContent =
    `${slate.total_plays} plays · ${slate.recommended_count} recommended · ` +
    `${slate.milestone_count} milestone alerts`;
  document.getElementById("generated-at").textContent =
    `Generated ${slate.generated_at || ""}`;

  playsEl.innerHTML = slate.plays.length
    ? slate.plays.map(cardHtml).join("")
    : '<div class="empty-state">No plays evaluated for this slate.</div>';
  applyFilter(
    document.querySelector(".tab.active")?.dataset.filter || "value",
  );
}

async function loadSlate() {
  try {
    const res = await fetch("daily_slate.json", { cache: "no-store" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderSlate(await res.json());
  } catch (err) {
    document.getElementById("slate-summary").textContent =
      "No slate published yet.";
    document.getElementById("plays").innerHTML =
      '<div class="empty-state">The morning pipeline has not run yet.</div>';
  }
}

document.getElementById("filter-tabs").addEventListener("click", (e) => {
  const btn = e.target.closest(".tab");
  if (!btn) return;
  document
    .querySelectorAll(".tab")
    .forEach((t) => t.classList.toggle("active", t === btn));
  applyFilter(btn.dataset.filter);
});

document.addEventListener("DOMContentLoaded", loadSlate);
