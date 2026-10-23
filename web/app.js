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

const SMART_FILTERS = [
  ["value", "Top Value"],
  ["milestones", "Milestones"],
  ["overs", "Overs"],
  ["unders", "Unders"],
  ["all", "All"],
];

const PAGE_SIZE = 40;

const state = {
  filter: "value",
  stat: null, // null = all stats
  search: "",
  sort: "ev",
  limit: PAGE_SIZE,
  plays: [],
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

function bestEv(play) {
  const ev =
    play.recommended_ev_pct ??
    Math.max(play.ev_over_pct ?? -999, play.ev_under_pct ?? -999);
  return ev === -999 ? null : ev;
}

function favoredSide(play) {
  if (play.recommended_side) return play.recommended_side;
  return (play.p_over || 0) >= 0.5 ? "over" : "under";
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

function cardHtml(play, featured) {
  const stat = STAT_LABELS[play.stat] || play.stat;
  const loc = play.is_home ? "vs" : "@";
  const evShown = bestEv(play);
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

  return `<article class="play-card${featured ? " featured" : ""}" data-stat="${play.stat}"
      style="--team:${teamColor(play.team_abbr)}">
    ${featured ? '<div class="featured-tag">TOP PICK</div>' : ""}
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

// --- filtering / sorting pipeline -------------------------------------------

function matchesFilter(play) {
  switch (state.filter) {
    case "value":
      return !!play.recommended_side;
    case "milestones":
      return !!(play.milestone && play.milestone.triggered);
    case "overs":
      return play.recommended_side === "over";
    case "unders":
      return play.recommended_side === "under";
    default:
      return true;
  }
}

function matchesSearch(play) {
  if (!state.search) return true;
  const q = state.search;
  return (
    (play.player_name || "").toLowerCase().includes(q) ||
    (play.team_abbr || "").toLowerCase().includes(q) ||
    (play.opponent_abbr || "").toLowerCase().includes(q)
  );
}

const SORTERS = {
  ev: (a, b) => (bestEv(b) ?? -999) - (bestEv(a) ?? -999),
  prob: (a, b) =>
    Math.max(b.p_over || 0, b.p_under || 0) -
    Math.max(a.p_over || 0, a.p_under || 0),
  edge: (a, b) => Math.abs(b.model_edge || 0) - Math.abs(a.model_edge || 0),
  name: (a, b) => (a.player_name || "").localeCompare(b.player_name || ""),
};

function visiblePlays() {
  return state.plays
    .filter(
      (p) =>
        matchesFilter(p) &&
        (state.stat === null || p.stat === state.stat) &&
        matchesSearch(p),
    )
    .sort(SORTERS[state.sort] || SORTERS.ev);
}

// --- rendering ---------------------------------------------------------------

function renderTabs() {
  const counts = {
    value: 0,
    milestones: 0,
    overs: 0,
    unders: 0,
    all: state.plays.length,
  };
  for (const p of state.plays) {
    if (p.recommended_side) counts.value++;
    if (p.milestone && p.milestone.triggered) counts.milestones++;
    if (p.recommended_side === "over") counts.overs++;
    if (p.recommended_side === "under") counts.unders++;
  }
  document.querySelectorAll("#filter-tabs .tab").forEach((btn) => {
    const f = btn.dataset.filter;
    const label = SMART_FILTERS.find(([k]) => k === f)[1];
    btn.innerHTML = `${label} <span class="tab-n">${counts[f]}</span>`;
  });

  const statCounts = {};
  for (const p of state.plays) {
    statCounts[p.stat] = (statCounts[p.stat] || 0) + 1;
  }
  const order = Object.keys(STAT_LABELS).filter((s) => statCounts[s]);
  const tabsEl = document.getElementById("stat-tabs");
  tabsEl.innerHTML =
    `<button class="tab stat-tab${state.stat === null ? " active" : ""}" data-stat="">All stats</button>` +
    order
      .map(
        (s) =>
          `<button class="tab stat-tab${state.stat === s ? " active" : ""}" data-stat="${s}">${STAT_LABELS[s]} <span class="tab-n">${statCounts[s]}</span></button>`,
      )
      .join("");
}

function renderPlays() {
  const visible = visiblePlays();
  const slice = visible.slice(0, state.limit);
  const featured =
    state.filter === "value" && state.sort === "ev" && !state.search
      ? slice[0]
      : null;
  const playsEl = document.getElementById("plays");
  playsEl.innerHTML = slice.length
    ? slice.map((p) => cardHtml(p, p === featured)).join("")
    : '<div class="empty-state">No plays match these filters.</div>';

  document.getElementById("results-count").textContent =
    `${Math.min(state.limit, visible.length)} of ${visible.length}`;
  document.getElementById("load-more").hidden =
    visible.length <= state.limit;
}

function renderSlate(slate) {
  state.plays = slate.plays || [];
  document.getElementById("slate-date").textContent = slate.date || "";
  document.getElementById("generated-at").textContent =
    `Generated ${slate.generated_at || ""}`;
  if (slate.demo) {
    document.getElementById("demo-badge").hidden = false;
  }
  renderModelRecord(slate.model_stats, slate.calibration);
  renderTabs();
  renderPlays();
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
    document.getElementById("results-count").textContent =
      "No slate published yet.";
    document.getElementById("plays").innerHTML =
      '<div class="empty-state">The morning pipeline has not run yet.</div>';
  }
}

// --- events ------------------------------------------------------------------

document.getElementById("filter-tabs").addEventListener("click", (e) => {
  const btn = e.target.closest(".tab");
  if (!btn) return;
  document
    .querySelectorAll("#filter-tabs .tab")
    .forEach((t) => t.classList.toggle("active", t === btn));
  state.filter = btn.dataset.filter;
  state.limit = PAGE_SIZE;
  renderPlays();
});

document.getElementById("stat-tabs").addEventListener("click", (e) => {
  const btn = e.target.closest(".stat-tab");
  if (!btn) return;
  document
    .querySelectorAll(".stat-tab")
    .forEach((t) => t.classList.toggle("active", t === btn));
  state.stat = btn.dataset.stat || null;
  state.limit = PAGE_SIZE;
  renderPlays();
});

document.getElementById("search").addEventListener("input", (e) => {
  state.search = e.target.value.trim().toLowerCase();
  state.limit = PAGE_SIZE;
  renderPlays();
});

document.getElementById("sort-select").addEventListener("change", (e) => {
  state.sort = e.target.value;
  renderPlays();
});

document.getElementById("load-more").addEventListener("click", () => {
  state.limit += PAGE_SIZE;
  renderPlays();
});

document.addEventListener("DOMContentLoaded", loadSlate);
