// Prophit dashboard — vanilla JS, zero frameworks (SRD 7.1).
// Renders daily_slate.json produced by src/publication/.

// NBA CDN ids + accent colors (tuned for dark charcoal UI)
const TEAM_META = {
  ATL: { id: 1610612737, color: "#e03a3e" },
  BOS: { id: 1610612738, color: "#2ecc71" },
  BKN: { id: 1610612751, color: "#b8bec4" },
  CHA: { id: 1610612766, color: "#1fb6c9" },
  CHI: { id: 1610612741, color: "#e5484d" },
  CLE: { id: 1610612739, color: "#c98a5e" },
  DAL: { id: 1610612742, color: "#5b9bd5" },
  DEN: { id: 1610612743, color: "#fec524" },
  DET: { id: 1610612765, color: "#5b9bd5" },
  GSW: { id: 1610612744, color: "#ffc72c" },
  HOU: { id: 1610612745, color: "#e5484d" },
  IND: { id: 1610612754, color: "#fdbb30" },
  LAC: { id: 1610612746, color: "#5b9bd5" },
  LAL: { id: 1610612747, color: "#fdb927" },
  MEM: { id: 1610612763, color: "#6a8eae" },
  MIA: { id: 1610612748, color: "#f9a01b" },
  MIL: { id: 1610612749, color: "#3ddc97" },
  MIN: { id: 1610612750, color: "#5b9bd5" },
  NOP: { id: 1610612740, color: "#e5484d" },
  NYK: { id: 1610612752, color: "#f58426" },
  OKC: { id: 1610612760, color: "#5b9bd5" },
  ORL: { id: 1610612753, color: "#5b9bd5" },
  PHI: { id: 1610612755, color: "#5b9bd5" },
  PHX: { id: 1610612756, color: "#e56020" },
  POR: { id: 1610612757, color: "#e5484d" },
  SAC: { id: 1610612758, color: "#8b7fd4" },
  SAS: { id: 1610612759, color: "#b8bec4" },
  TOR: { id: 1610612761, color: "#e5484d" },
  UTA: { id: 1610612762, color: "#f9a01b" },
  WAS: { id: 1610612764, color: "#5b9bd5" },
};

const STAT_LABELS = {
  pts: "PTS",
  reb: "REB",
  ast: "AST",
  fg3m: "3PM",
  pra: "PRA",
  pr: "P+R",
  pa: "P+A",
  ra: "R+A",
  blk: "BLK",
  stl: "STL",
  tov: "TOV",
};

const teamColor = (abbr) => TEAM_META[abbr]?.color || "#8b7fd4";

const teamLogoUrl = (abbr) =>
  TEAM_META[abbr]
    ? `https://cdn.nba.com/logos/nba/${TEAM_META[abbr].id}/primary/L/logo.svg`
    : null;

const headshotUrl = (playerId) =>
  `https://cdn.nba.com/headshots/nba/latest/260x190/${playerId}.png`;

const initials = (name) =>
  (name || "?")
    .split(/\s+/)
    .filter(Boolean)
    .map((w) => w[0])
    .slice(0, 2)
    .join("")
    .toUpperCase();

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

function logoImg(abbr) {
  const url = teamLogoUrl(abbr);
  return url
    ? `<img src="${url}" alt="${abbr}" loading="lazy" onerror="this.remove()"/>`
    : "";
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
  const chipText = play.market_suppressed
    ? "MKT OFF"
    : play.low_confidence
      ? "LOW CONF"
      : side
        ? `${side} ${fmtSigned(evShown)}%`
        : `${fmtSigned(evShown)}% EV`;
  const edge = play.model_edge ?? 0;
  const arrow = edge >= 0 ? "up" : "down";
  const pOver = Math.max(0, Math.min(100, (play.p_over || 0) * 100));
  const pUnder = 100 - pOver;
  const underFavored = pUnder > pOver;
  const favoredPct = Math.max(pOver, pUnder);

  const f = play.factors || {};
  const facts = [
    ["Proj. Min", f.projected_minutes],
    ["Sample", play.history_games != null ? `${play.history_games} g` : null],
    [
      "Confidence",
      play.market_suppressed
        ? "Market auto-suppressed (losing calibration ROI)"
        : play.low_confidence
          ? `Low — ${(play.confidence_reasons || []).join(", ")}`
          : null,
    ],
    ["DvP", f.dvp_label],
    ["Rest", f.rest_label],
    ["Altitude", f.altitude ? "Yes" : null],
    [
      "Book",
      `O ${fmtOdds(play.over_price)} / U ${fmtOdds(play.under_price)}`,
    ],
    ["Fair Over", fmtOdds(play.fair_over_american)],
  ]
    .filter(([, v]) => v != null)
    .map(([k, v]) => `<div>${k}: <b>${v}</b></div>`)
    .join("");

  const m = play.milestone || {};
  const ladder = m.triggered
    ? `<div class="ladder">
         ${m.milestones
           .map((r) => {
             const pct = Math.max(0, Math.min(100, r.p_over * 100));
             return `<div class="ladder-rung">
               <span class="ladder-k">${r.k}+</span>
               <div>
                 <div class="ladder-bar"><div class="ladder-fill" style="width:${pct}%"></div></div>
               </div>
               <span class="ladder-odds"><b>${fmtOdds(r.fair_american)}</b>
                 (${pct.toFixed(0)}%) · <span class="target">${fmtOdds(r.alt_target_american)}+</span></span>
             </div>`;
           })
           .join("")}
       </div>`
    : "";

  return `<article class="play-card" data-stat="${play.stat}"
      data-value="${play.recommended_side ? 1 : 0}"
      data-milestone="${m.triggered ? 1 : 0}"
      style="--team:${teamColor(play.team_abbr)}">
    <div class="card-top">
      <div class="avatar"><span class="initials">${initials(play.player_name)}</span><img
        src="${headshotUrl(play.player_id)}" alt=""
        loading="lazy"
        onload="this.closest('.avatar').classList.add('loaded')"
        onerror="this.remove()"/></div>
      <div class="player-block">
        <div class="player-name">${play.player_name}</div>
        <div class="player-meta">${logoImg(play.team_abbr)}
          <span>${play.position || ""} ${play.team_abbr || ""} ${loc}</span>
          ${logoImg(play.opponent_abbr)}
          <span>${play.opponent_abbr || ""}</span>
        </div>
      </div>
      <span class="chip ${play.market_suppressed || play.low_confidence ? "ev-caution" : evChipClass(evShown)}">${chipText}</span>
    </div>
    <div class="stat-row">
      <span class="stat-badge ${play.stat}">${stat}</span>
      <div class="line-model">Line <b>${play.line}</b> → Model <b>${play.model_mean}</b>
        <span class="edge-arrow ${arrow}">${edge >= 0 ? "▲" : "▼"} ${fmtSigned(edge)}</span>
      </div>
    </div>
    <div class="prob-bar"><div class="prob-fill${underFavored ? " under" : ""}"
      style="width:${favoredPct}%"></div></div>
    <div class="prob-labels">
      <span class="${underFavored ? "" : "favored"}">Over ${pOver.toFixed(1)}%</span>
      <span class="${underFavored ? "favored" : ""}">Under ${pUnder.toFixed(1)}%</span>
    </div>
    ${m.triggered ? '<div class="milestone-flag">MILESTONE ALERT</div>' : ""}
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
  if (slate.demo) {
    document.getElementById("demo-badge").hidden = false;
  }
  renderModelRecord(slate.model_stats, slate.calibration);

  playsEl.innerHTML = slate.plays.length
    ? slate.plays.map(cardHtml).join("")
    : '<div class="empty-state">No plays evaluated for this slate.</div>';
  applyFilter(
    document.querySelector(".tab.active")?.dataset.filter || "value",
  );
}

function renderModelRecord(stats, calibration) {
  const el = document.getElementById("model-record");
  if (!el) return;
  const r = (stats && stats.record) || {};
  const a = (stats && stats.accuracy) || {};
  const parts = [];
  if (calibration && calibration.min_ev_pct != null) {
    const sup = (calibration.suppressed_markets || []).join(", ");
    parts.push(
      `<span class="rec-chip">EV ≥ ${calibration.min_ev_pct}%` +
        `${calibration.ran ? " · calibrated today" : ""}</span>` +
        (sup ? `<span class="rec-chip neg">off: ${sup}</span>` : ""),
    );
  }
  if (r.bets > 0) {
    const roi = fmtSigned(r.roi_pct);
    const cls = r.units >= 0 ? "pos" : "neg";
    parts.push(
      `<span class="rec-chip ${cls}">${r.wins}-${r.losses}` +
        `${r.pushes ? `-${r.pushes}` : ""} · ${fmtSigned(r.units)}u ` +
        `(${roi}% ROI)</span>`,
    );
  }
  if (a.favored_hit_rate != null) {
    parts.push(
      `<span class="rec-chip">${a.favored_hit_rate}% favored accuracy ` +
        `<i>n=${a.sample}</i></span>`,
    );
  }
  if (parts.length) {
    el.innerHTML = `<span class="rec-label">MODEL RECORD</span>${parts.join("")}`;
    el.hidden = false;
  }
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
