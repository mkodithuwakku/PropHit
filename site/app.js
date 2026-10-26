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

const PAGE_SIZE = 60;

const state = {
  view: "browse", // browse | digest | games
  game: null, // matchup key scoping browse to one game
  filter: "value",
  stat: null,
  search: "",
  sort: "ev",
  dsort: "ev", // best-picks sort: ev | prob
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

function matchup(play) {
  const home = play.home_team_abbr;
  const away = play.away_team_abbr;
  return { key: `${away}@${home}`, away, home };
}

// LOW CONF plays are hidden app-wide — the calibration + confidence
// gates already decided they are not actionable tonight.
function basePlays() {
  return state.plays.filter((p) => !p.low_confidence);
}

function sideText(play) {
  return play.recommended_side === "under"
    ? "Under"
    : play.recommended_side === "over"
      ? "Over"
      : null;
}

function chipText(play) {
  if (play.market_suppressed) return "MKT OFF";
  const side = sideText(play);
  const ev = bestEv(play);
  return side ? `${side} · ${fmtSigned(ev)}% EV` : `${fmtSigned(ev)}% EV`;
}

function probNums(play) {
  const pct = (p) => Math.max(0, Math.min(100, p * 100));
  const pOver = pct(play.p_over ?? 0);
  const pUnder = pct(
    play.p_under ?? 1 - (play.p_over ?? 0) - (play.p_push ?? 0),
  );
  return { pOver, pUnder, pPush: Math.max(0, 100 - pOver - pUnder) };
}

// EV can favor the less likely side at a sufficiently generous price.
// Keep every recommendation label, bar and sort tied to that side's chance.
function displayedOutcome(play) {
  const { pOver, pUnder } = probNums(play);
  const side = play.recommended_side || (pUnder > pOver ? "under" : "over");
  return { side, probability: side === "under" ? pUnder : pOver };
}

function factGrid(play) {
  const f = play.factors || {};
  return [
    ["Proj. Min", f.projected_minutes],
    ["Sample", play.history_games != null ? `${play.history_games} g` : null],
    [
      "Confidence",
      play.market_suppressed
        ? "Market auto-suppressed (losing calibration ROI)"
        : null,
    ],
    ["DvP", f.dvp_label],
    ["Rest", f.rest_label],
    ["Altitude", f.altitude ? "Yes" : null],
    ["Book", `O ${fmtOdds(play.over_price)} / U ${fmtOdds(play.under_price)}`],
    ["Fair Over", fmtOdds(play.fair_over_american)],
  ]
    .filter(([, v]) => v != null)
    .map(([k, v]) => `<div>${k}: <b>${v}</b></div>`)
    .join("");
}

function ladderHtml(m) {
  return `<div class="ladder">
    ${m.milestones
      .map((r) => {
        const pct = Math.max(0, Math.min(100, r.p_over * 100));
        return `<div class="ladder-rung">
          <span class="ladder-k">${r.k}+</span>
          <div><div class="ladder-bar"><div class="ladder-fill" style="width:${pct}%"></div></div></div>
          <span class="ladder-odds"><b>${fmtOdds(r.fair_american)}</b>
            (${pct.toFixed(0)}%) · <span class="target">${fmtOdds(r.alt_target_american)}+</span></span>
        </div>`;
      })
      .join("")}
  </div>`;
}

// Full-width Browse cards on phones; two columns when there is room.
function cardHtml(play, { featured = false } = {}) {
  const { pOver, pUnder, pPush } = probNums(play);
  const { side, probability } = displayedOutcome(play);
  const isUnder = side === "under";
  const edge = play.model_edge ?? 0;
  const m = play.milestone || {};
  const loc = play.is_home ? "vs" : "@";

  return `<article class="play-card${featured ? " featured" : ""}"
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
        <div class="player-meta">${logoImg(play.team_abbr)}<span>${play.team_abbr || ""} ${loc}</span>${logoImg(play.opponent_abbr)}<span>${play.opponent_abbr || ""}</span></div>
      </div>
    </div>
    <div class="card-recommendation">
      <span>${play.recommended_side ? "Pick" : "Market"}</span>
      <span class="chip ${play.market_suppressed ? "ev-caution" : evChipClass(bestEv(play))}">${chipText(play)}</span>
    </div>
    <div class="stat-row">
      <span class="stat-badge ${play.stat}">${STAT_LABELS[play.stat] || play.stat}</span>
      <div class="line-model"><span>Line <b>${play.line}</b></span><span>Projection <b>${play.model_mean}</b></span>
        <span class="edge-arrow ${edge >= 0 ? "up" : "down"}">${edge >= 0 ? "▲" : "▼"}${fmtSigned(edge)}</span>
      </div>
      ${m.triggered ? '<span class="ms-badge">▲ ALT</span>' : ""}
    </div>
    <div class="prob-bar"><div class="prob-fill${isUnder ? " under" : ""}"
      style="width:${probability}%"></div></div>
    <div class="prob-labels">
      <span class="${isUnder ? "" : "favored"}">Over ${pOver.toFixed(1)}%</span>
      ${pPush >= 0.05 ? `<span>Push ${pPush.toFixed(1)}%</span>` : ""}
      <span class="${isUnder ? "favored" : ""}">Under ${pUnder.toFixed(1)}%</span>
    </div>
    <details class="tray"><summary>Factors${m.triggered ? " + Ladder" : ""}</summary>
      <div class="fact-grid">${factGrid(play)}</div>
      ${m.triggered ? `<div class="tray-label">MILESTONE LADDER</div>${ladderHtml(m)}` : ""}
    </details>
  </article>`;
}

// Ultra-compact tile — three per row in Best Picks.
function miniHtml(play) {
  const { side, probability } = displayedOutcome(play);
  const ev = bestEv(play);
  const evCls = play.market_suppressed ? "neg" : ev >= 5 ? "pos" : ev >= 0 ? "neu" : "neg";

  return `<article class="mini-card" style="--team:${teamColor(play.team_abbr)}">
    <div class="mc-top">
      <div class="avatar"><span class="initials">${initials(play.player_name)}</span><img
        src="${headshotUrl(play.player_id)}" alt=""
        loading="lazy"
        onload="this.closest('.avatar').classList.add('loaded')"
        onerror="this.remove()"/></div>
      <span class="mc-ev ${evCls}">${side === "over" ? "O" : "U"} ${fmtSigned(ev, 0)}%</span>
    </div>
    <div class="mc-name">${play.player_name}</div>
    <div class="mc-meta">${play.team_abbr || ""}${play.is_home ? " vs" : " @"} ${play.opponent_abbr || ""} · <b class="mc-stat">${STAT_LABELS[play.stat] || play.stat}</b></div>
    <div class="mc-line">${play.line} → <b>${play.model_mean}</b></div>
    <div class="prob-bar"><div class="prob-fill${side === "under" ? " under" : ""}"
      style="width:${probability}%"></div></div>
    <div class="mc-fav">${side === "over" ? "Over" : "Under"} ${probability.toFixed(0)}%</div>
    ${play.milestone && play.milestone.triggered ? '<div class="mc-flag">▲ MILESTONE</div>' : ""}
  </article>`;
}

// --- pipelines ---------------------------------------------------------------

const SORTERS = {
  ev: (a, b) => (bestEv(b) ?? -999) - (bestEv(a) ?? -999),
  prob: (a, b) =>
    displayedOutcome(b).probability - displayedOutcome(a).probability,
  edge: (a, b) => Math.abs(b.model_edge || 0) - Math.abs(a.model_edge || 0),
  name: (a, b) => (a.player_name || "").localeCompare(b.player_name || ""),
};

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

function inGame(play) {
  return !state.game || matchup(play).key === state.game;
}

function browsePlays() {
  return basePlays()
    .filter(
      (p) =>
        inGame(p) &&
        matchesFilter(p) &&
        (state.stat === null || p.stat === state.stat) &&
        matchesSearch(p),
    )
    .sort(SORTERS[state.sort] || SORTERS.ev);
}

function gameList() {
  const games = {};
  for (const p of basePlays()) {
    const mu = matchup(p);
    const g = (games[mu.key] = games[mu.key] || {
      key: mu.key,
      away: mu.away,
      home: mu.home,
      plays: 0,
      rec: 0,
    });
    g.plays++;
    if (p.recommended_side) g.rec++;
  }
  return Object.values(games).sort(
    (a, b) => b.rec - a.rec || a.key.localeCompare(b.key),
  );
}

// --- render --------------------------------------------------------------------

function renderPlays() {
  const playsEl = document.getElementById("plays");
  const loadMore = document.getElementById("load-more");
  const meta = document.getElementById("results-count");
  playsEl.classList.remove("cols2", "cols3");

  if (state.view === "digest") {
    playsEl.classList.add("cols3");
    const rec = basePlays()
      .filter((p) => p.recommended_side && Number(p.line) !== 0.5)
      .sort(
        state.dsort === "prob"
          ? SORTERS.prob
          : SORTERS.ev,
      );
    playsEl.innerHTML = rec.length
      ? rec.map(miniHtml).join("")
      : '<div class="empty-state">No eligible Best Picks on this slate. Browse includes 0.5 lines.</div>';
    loadMore.hidden = true;
    return;
  }

  if (state.view === "games") {
    const games = gameList();
    playsEl.innerHTML = games.length
      ? games
          .map(
            (g) => `<button class="game-card" data-game="${g.key}">
              <span class="game-teams">
                ${logoImg(g.away)}<b>${g.away}</b>
                <span class="game-at">@</span>
                ${logoImg(g.home)}<b>${g.home}</b>
              </span>
              <span class="game-counts">
                ${g.rec ? `<span class="rec-n">${g.rec} picks</span>` : ""}
                <span class="prop-n">${g.plays} props</span>
              </span>
            </button>`,
          )
          .join("")
      : '<div class="empty-state">No games on this slate.</div>';
    loadMore.hidden = true;
    return;
  }

  // browse (optionally scoped to one game)
  playsEl.classList.add("cols2");
  const visible = browsePlays();
  const slice = visible.slice(0, state.limit);
  const featured =
    state.filter === "value" && state.sort === "ev" && !state.search
      ? slice[0]
      : null;
  playsEl.innerHTML = slice.length
    ? slice.map((p) => cardHtml(p, { featured: p === featured })).join("")
    : '<div class="empty-state">No plays match these filters.</div>';
  meta.textContent = `${Math.min(state.limit, visible.length)} of ${visible.length}`;
  loadMore.hidden = visible.length <= state.limit;
}

function syncChrome() {
  document.getElementById("browse-controls").hidden = state.view !== "browse";
  document.getElementById("digest-sort").hidden = state.view !== "digest";
  document.getElementById("digest-note").hidden = state.view !== "digest";
  document.getElementById("game-picker-label").textContent = state.game
    ? state.game.replace("@", " @ ")
    : "All games";
  document.getElementById("game-picker-clear").hidden = !state.game;
  document
    .querySelectorAll("#view-tabs .view-tab")
    .forEach((t) => t.classList.toggle("active", t.dataset.view === state.view));
}

function renderGamePicker() {
  const list = document.getElementById("game-picker-list");
  const games = gameList();
  list.innerHTML =
    `<button class="game-option${state.game === null ? " active" : ""}" data-game="">
      <span class="game-teams"><b>All games</b></span>
    </button>` +
    games
      .map(
        (g) => `<button class="game-option${state.game === g.key ? " active" : ""}" data-game="${g.key}">
          <span class="game-teams">
            ${logoImg(g.away)}<b>${g.away}</b>
            <span class="game-at">@</span>
            ${logoImg(g.home)}<b>${g.home}</b>
          </span>
          ${g.rec ? `<span class="rec-n">${g.rec}</span>` : ""}
        </button>`,
      )
      .join("");
}

function setView(view) {
  // Switching views clears the single-game scope (fixes stale game bar).
  state.view = view;
  if (view !== "browse") state.game = null;
  state.limit = PAGE_SIZE;
  syncChrome();
  renderTabs();
  renderGamePicker();
  renderPlays();
  window.scrollTo({ top: 0 });
}

function openGame(key) {
  // Games drill down into the regular Browse view, scoped to that game.
  state.game = key;
  state.view = "browse";
  state.limit = PAGE_SIZE;
  document.getElementById("game-picker-list").hidden = true;
  syncChrome();
  renderTabs();
  renderGamePicker();
  renderPlays();
  window.scrollTo({ top: 0 });
}

function clearGame() {
  state.game = null;
  state.limit = PAGE_SIZE;
  syncChrome();
  renderTabs();
  renderGamePicker();
  renderPlays();
}

function renderTabs() {
  const counts = { value: 0, milestones: 0, overs: 0, unders: 0, all: 0 };
  for (const p of basePlays()) {
    if (!inGame(p)) continue;
    counts.all++;
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
  for (const p of basePlays()) {
    if (!inGame(p)) continue;
    statCounts[p.stat] = (statCounts[p.stat] || 0) + 1;
  }
  const order = Object.keys(STAT_LABELS).filter((s) => statCounts[s]);
  const tabsEl = document.getElementById("stat-tabs");
  tabsEl.innerHTML =
    `<button class="tab stat-tab${state.stat === null ? " active" : ""}" data-stat="">All</button>` +
    order
      .map(
        (s) =>
          `<button class="tab stat-tab${state.stat === s ? " active" : ""}" data-stat="${s}">${STAT_LABELS[s]} <span class="tab-n">${statCounts[s]}</span></button>`,
      )
      .join("");
}

function renderSlate(slate) {
  state.plays = slate.plays || [];
  document.getElementById("slate-date").textContent = slate.date || "";
  document.getElementById("generated-at").textContent =
    `Generated ${slate.generated_at || ""}`;
  document.getElementById("demo-badge").hidden = !slate.demo;
  renderModelRecord(slate.model_stats, slate.calibration);
  renderTabs();
  renderGamePicker();
  syncChrome();
  renderPlays();
}

function renderModelRecord(stats, calibration) {
  const el = document.getElementById("model-record");
  if (!el) return;
  const r = (stats && stats.record) || {};
  const a = (stats && stats.accuracy) || {};
  const parts = [];
  if (calibration && calibration.min_ev_pct != null) {
    parts.push(
      `<span class="rec-chip">EV ≥ ${calibration.min_ev_pct}%` +
        `${calibration.ran ? " · calibrated today" : ""}</span>`,
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
    // Request a fresh snapshot rather than reusing a cached response.
    const res = await fetch(`daily_slate.json?t=${Date.now()}`, {
      cache: "no-store",
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    renderSlate(await res.json());
  } catch (err) {
    document.getElementById("results-count").textContent =
      "No slate published yet.";
    document.getElementById("plays").innerHTML =
      '<div class="empty-state">The morning pipeline has not run yet.</div>';
  }
}

// --- events --------------------------------------------------------------------

document.getElementById("view-tabs").addEventListener("click", (e) => {
  const btn = e.target.closest(".view-tab");
  if (!btn) return;
  setView(btn.dataset.view);
});

document.getElementById("plays").addEventListener("click", (e) => {
  const gc = e.target.closest(".game-card");
  if (gc) openGame(gc.dataset.game);
});

// game-picker dropdown: toggle, select, clear, outside-click close
document.getElementById("game-picker-btn").addEventListener("click", (e) => {
  if (e.target.closest("#game-picker-clear")) return;
  const list = document.getElementById("game-picker-list");
  list.hidden = !list.hidden;
});

document.getElementById("game-picker-clear").addEventListener("click", (e) => {
  e.stopPropagation();
  document.getElementById("game-picker-list").hidden = true;
  clearGame();
});

document.getElementById("game-picker-list").addEventListener("click", (e) => {
  const opt = e.target.closest(".game-option");
  if (!opt) return;
  document.getElementById("game-picker-list").hidden = true;
  if (opt.dataset.game) {
    openGame(opt.dataset.game);
  } else {
    clearGame();
  }
});

document.addEventListener("click", (e) => {
  if (!e.target.closest("#game-picker")) {
    document.getElementById("game-picker-list").hidden = true;
  }
});

document.getElementById("digest-sort").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-dsort]");
  if (!btn) return;
  state.dsort = btn.dataset.dsort;
  document
    .querySelectorAll("#digest-sort .tab")
    .forEach((t) => t.classList.toggle("active", t === btn));
  renderPlays();
});

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
