// Offline dashboard regressions using Node's built-in runner (no dependencies).
const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const path = require("node:path");
const { test } = require("node:test");
const vm = require("node:vm");

const plays = JSON.parse(
  readFileSync(path.join(__dirname, "fixtures/dashboard_plays.json"), "utf8"),
);

function dashboard() {
  const elements = new Map();
  const document = {
    getElementById(id) {
      if (!elements.has(id)) {
        elements.set(id, {
          innerHTML: "",
          classList: { add() {}, remove() {}, toggle() {} },
          addEventListener() {},
        });
      }
      return elements.get(id);
    },
    querySelectorAll: () => [],
    addEventListener() {},
  };
  const ctx = vm.createContext({
    document,
    // Leave initial loading pending: tests supply fixture data, never network.
    fetch: () => new Promise(() => {}),
    plays: structuredClone(plays),
  });
  vm.runInContext(
    readFileSync(path.join(__dirname, "../web/app.js"), "utf8"),
    ctx,
  );
  vm.runInContext("state.plays = plays", ctx);
  return {
    run: (code) => vm.runInContext(code, ctx),
    html: () => document.getElementById("plays").innerHTML,
  };
}

test("cards show the recommended side's probability even for a long shot", () => {
  const app = dashboard();
  const mini = app.run("miniHtml(plays[0])");
  assert.match(mini, /O \+150%/);
  assert.match(mini, /Over 13%/);
  assert.doesNotMatch(mini, /Over 87%/);
  assert.match(mini, /style="width:13\.3%"/);
  const card = app.run("cardHtml(plays[0])");
  assert.match(card, /Over · \+150\.3% EV/);
  assert.match(card, /class="favored">Over 13\.3%/);
  assert.match(card, /Under 86\.7%/);
  // Exercise the symmetric case: a less likely under recommended on price.
  assert.match(
    app.run("miniHtml({...plays[3], recommended_side: 'under'})"),
    /Under 35%/,
  );
  assert.match(app.run("miniHtml(plays[2])"), /Under 78%/);
});

test("Best Picks excludes 0.5 lines in both sorts but Browse retains them", () => {
  const app = dashboard();
  for (const sort of ["ev", "prob"]) {
    app.run(`state.view = 'digest'; state.dsort = '${sort}'; renderPlays()`);
    assert.doesNotMatch(app.html(), /Isaiah Hartenstein|Andrew Nembhard/);
    assert.match(app.html(), /Jamal Shead/);
    assert.match(app.html(), /Fixture PRA Player/);
  }
  app.run("state.view = 'browse'; renderPlays()");
  assert.match(app.html(), /Isaiah Hartenstein/);
  assert.match(app.html(), /Andrew Nembhard/);
  // A 1.5 line remains eligible; low-confidence gating still applies.
  app.run(`
    state.plays = [
      {...plays[0], line: 1.5},
      {...plays[2], low_confidence: true}
    ];
    state.view = 'digest'; renderPlays();
  `);
  assert.match(app.html(), /Isaiah Hartenstein/);
  assert.doesNotMatch(app.html(), /Jamal Shead/);
  app.run("state.plays = plays.slice(0, 2); renderPlays()");
  assert.match(app.html(), /No eligible Best Picks/);
});

test("probability sorting ranks the recommended chance rather than its opposite", () => {
  const app = dashboard();
  // Make the long shot eligible to isolate sorting from the 0.5 exclusion.
  app.run(`
    state.plays = plays.map(p => ({...p, line: 2.5}));
    state.view = 'digest'; state.dsort = 'prob'; renderPlays();
  `);
  assert.ok(app.html().indexOf("Jamal Shead") < app.html().indexOf("Isaiah Hartenstein"));
  assert.ok(app.html().indexOf("Fixture PRA Player") < app.html().indexOf("Andrew Nembhard"));
});

test("integer-line pushes are not displayed as under wins", () => {
  const app = dashboard();
  const html = app.run(`cardHtml({
    ...plays[2], line: 10, p_over: 0.3, p_under: 0.5
  })`);
  assert.match(html, /Under 50\.0%/);
  assert.match(html, /Push 20\.0%/);
  assert.doesNotMatch(html, /Under 70/);
});
