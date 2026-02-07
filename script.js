const seedValue = document.getElementById("seedValue");
const remixButton = document.getElementById("remix");
const runButton = document.getElementById("run");
const shareButton = document.getElementById("share");
const promptInput = document.getElementById("prompt");
const modeSelect = document.getElementById("mode");
const layerSelect = document.getElementById("layer");
const headSelect = document.getElementById("head");
const temperatureInput = document.getElementById("temperature");
const energyToggle = document.getElementById("energyToggle");
const tokenRow = document.getElementById("tokenRow");
const attentionCanvas = document.getElementById("attention");
const mapLabel = document.getElementById("mapLabel");
const nextTokens = document.getElementById("nextTokens");
const energyPanel = document.getElementById("energyPanel");
const energyBars = document.getElementById("energyBars");
const toast = document.getElementById("toast");

const ctx = attentionCanvas.getContext("2d");

const config = {
  layers: 2,
  heads: 4,
  modelDim: 48,
  headDim: 12,
  maxTokens: 24,
};

const coreVocab = [
  "the",
  "a",
  "and",
  "to",
  "of",
  "in",
  "is",
  "that",
  "for",
  "with",
  "on",
  "as",
  "by",
  "attention",
  "token",
  "model",
  "transformer",
  "learn",
  "flow",
  "context",
  "matrix",
  "probability",
  ".",
  ",",
  "!",
  "?",
  "<eos>",
];

const archetypes = [
  "previous",
  "next",
  "punctuation",
  "repeat",
];

const state = {
  seed: "",
  rng: null,
  tokens: [],
  selectedToken: 0,
  attentions: [],
  energies: [],
  vocab: [...coreVocab],
};

function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function hashString(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i += 1) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function rngFromString(str) {
  return mulberry32(hashString(str));
}

function setSeed(seed) {
  state.seed = seed;
  seedValue.textContent = seed;
  state.rng = mulberry32(hashString(seed));
  const url = new URL(window.location.href);
  url.searchParams.set("seed", seed);
  window.history.replaceState({}, "", url.toString());
}

function showToast(message) {
  toast.textContent = message;
  toast.classList.remove("hidden");
  setTimeout(() => toast.classList.add("hidden"), 1600);
}

function tokenize(text) {
  return text
    .trim()
    .split(/(\s+|[.,!?])/)
    .filter((token) => token.trim().length)
    .slice(0, config.maxTokens);
}

function embedToken(token) {
  const rand = rngFromString(`${state.seed}:${token}`);
  const vector = new Array(config.modelDim).fill(0).map(() => rand() * 2 - 1);
  return vector;
}

function dot(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += a[i] * b[i];
  return sum;
}

function add(a, b) {
  return a.map((value, i) => value + b[i]);
}

function scale(a, scalar) {
  return a.map((value) => value * scalar);
}

function softmax(logits, temperature = 1) {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp((x - max) / temperature));
  const sum = exps.reduce((acc, val) => acc + val, 0) || 1;
  return exps.map((val) => val / sum);
}

function layerNorm(vector) {
  const mean = vector.reduce((acc, val) => acc + val, 0) / vector.length;
  const variance =
    vector.reduce((acc, val) => acc + (val - mean) ** 2, 0) / vector.length;
  const denom = Math.sqrt(variance + 1e-5);
  return vector.map((val) => (val - mean) / denom);
}

function createMatrix(rows, cols, rand) {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => rand() * 2 - 1)
  );
}

function linear(vector, matrix) {
  const out = new Array(matrix[0].length).fill(0);
  for (let i = 0; i < matrix[0].length; i += 1) {
    let sum = 0;
    for (let j = 0; j < vector.length; j += 1) {
      sum += vector[j] * matrix[j][i];
    }
    out[i] = sum;
  }
  return out;
}

function initWeights() {
  const rand = state.rng;
  const weights = [];
  for (let l = 0; l < config.layers; l += 1) {
    weights.push({
      Wq: createMatrix(config.modelDim, config.modelDim, rand),
      Wk: createMatrix(config.modelDim, config.modelDim, rand),
      Wv: createMatrix(config.modelDim, config.modelDim, rand),
      Wo: createMatrix(config.modelDim, config.modelDim, rand),
      W1: createMatrix(config.modelDim, config.modelDim * 2, rand),
      W2: createMatrix(config.modelDim * 2, config.modelDim, rand),
    });
  }
  return weights;
}

function attentionBias(tokens, headIndex, i, j) {
  if (modeSelect.value !== "archetype") return 0;
  const type = archetypes[headIndex % archetypes.length];
  if (type === "previous") return j === i - 1 ? 1.2 : 0;
  if (type === "next") return j === i + 1 ? 1.2 : 0;
  if (type === "punctuation") {
    return /[.,!?]/.test(tokens[j]) ? 1.1 : 0;
  }
  if (type === "repeat") {
    return tokens[j] === tokens[i] ? 1.1 : 0;
  }
  return 0;
}

function runModel(tokens) {
  const weights = initWeights();
  let X = tokens.map((token) => layerNorm(embedToken(token)));
  const attentions = [];
  const energies = [];

  for (let l = 0; l < config.layers; l += 1) {
    const { Wq, Wk, Wv, Wo, W1, W2 } = weights[l];
    const Q = X.map((x) => linear(x, Wq));
    const K = X.map((x) => linear(x, Wk));
    const V = X.map((x) => linear(x, Wv));

    const headOutputs = Array.from({ length: config.heads }, () => []);
    const layerAttn = Array.from({ length: config.heads }, () => []);

    for (let h = 0; h < config.heads; h += 1) {
      const start = h * config.headDim;
      const end = start + config.headDim;

      const scores = [];
      for (let i = 0; i < tokens.length; i += 1) {
        const row = [];
        for (let j = 0; j < tokens.length; j += 1) {
          const q = Q[i].slice(start, end);
          const k = K[j].slice(start, end);
          const raw = dot(q, k) / Math.sqrt(config.headDim);
          row.push(raw + attentionBias(tokens, h, i, j));
        }
        scores.push(row);
      }

      const weightsRow = scores.map((row) => softmax(row, temperatureInput.value));
      layerAttn[h] = weightsRow;

      for (let i = 0; i < tokens.length; i += 1) {
        const weighted = new Array(config.headDim).fill(0);
        for (let j = 0; j < tokens.length; j += 1) {
          const weight = weightsRow[i][j];
          const v = V[j].slice(start, end);
          for (let k = 0; k < config.headDim; k += 1) {
            weighted[k] += v[k] * weight;
          }
        }
        headOutputs[h][i] = weighted;
      }
    }

    const concat = headOutputs[0].map((_, i) => {
      const joined = [];
      for (let h = 0; h < config.heads; h += 1) {
        joined.push(...headOutputs[h][i]);
      }
      return joined;
    });

    const attnOut = concat.map((row) => linear(row, Wo));
    X = X.map((row, i) => layerNorm(add(row, attnOut[i])));

    const ff = X.map((row) => {
      const hidden = linear(row, W1).map((val) => Math.max(0, val));
      return linear(hidden, W2);
    });
    X = X.map((row, i) => layerNorm(add(row, ff[i])));

    const energy = Math.sqrt(
      X.reduce((acc, row) => acc + dot(row, row), 0) / X.length
    );
    energies.push(energy);
    attentions.push(layerAttn);
  }

  return { X, attentions, energies };
}

function buildVocab(tokens) {
  const vocab = new Set(coreVocab);
  tokens.forEach((token) => vocab.add(token));
  state.vocab = Array.from(vocab).slice(0, 80);
}

function nextTokenProbs(vector) {
  const logits = state.vocab.map((token) => dot(vector, embedToken(token)));
  const probs = softmax(logits, 0.8);
  const ranked = state.vocab
    .map((token, i) => ({ token, prob: probs[i] }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 8);
  return ranked;
}

function renderTokens(tokens) {
  tokenRow.innerHTML = "";
  tokens.forEach((token, i) => {
    const span = document.createElement("button");
    span.type = "button";
    span.className = "token";
    span.textContent = token;
    if (i === state.selectedToken) span.classList.add("active");
    span.addEventListener("click", () => {
      state.selectedToken = i;
      renderTokens(state.tokens);
      drawAttention();
    });
    tokenRow.appendChild(span);
  });
}

function drawAttention() {
  const layerIndex = Number(layerSelect.value);
  const headIndex = Number(headSelect.value);
  const attention = state.attentions[layerIndex][headIndex];
  const tokens = state.tokens;
  const size = tokens.length;
  const cell = attentionCanvas.width / Math.max(size, 1);

  ctx.clearRect(0, 0, attentionCanvas.width, attentionCanvas.height);
  ctx.fillStyle = "#06070d";
  ctx.fillRect(0, 0, attentionCanvas.width, attentionCanvas.height);

  for (let i = 0; i < size; i += 1) {
    for (let j = 0; j < size; j += 1) {
      const value = attention[i][j];
      const intensity = Math.min(1, value * 3);
      const alpha = i === state.selectedToken ? 0.3 + intensity * 0.7 : 0.15 + intensity * 0.6;
      ctx.fillStyle = `rgba(122, 240, 255, ${alpha})`;
      ctx.fillRect(j * cell, i * cell, cell, cell);
    }
  }

  ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= size; i += 1) {
    ctx.beginPath();
    ctx.moveTo(0, i * cell);
    ctx.lineTo(size * cell, i * cell);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(i * cell, 0);
    ctx.lineTo(i * cell, size * cell);
    ctx.stroke();
  }

  mapLabel.textContent = `Layer ${layerIndex + 1} · Head ${headIndex + 1}`;
}

function renderNextToken(vector) {
  nextTokens.innerHTML = "";
  nextTokenProbs(vector).forEach(({ token, prob }) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${token}</span><strong>${(prob * 100).toFixed(1)}%</strong>`;
    nextTokens.appendChild(li);
  });
}

function renderEnergy(energies) {
  energyBars.innerHTML = "";
  const max = Math.max(...energies);
  energies.forEach((value, index) => {
    const bar = document.createElement("div");
    bar.className = "energy-bar";
    const fill = document.createElement("span");
    fill.style.width = `${(value / max) * 100}%`;
    bar.appendChild(fill);
    bar.title = `Layer ${index + 1}: ${value.toFixed(2)}`;
    energyBars.appendChild(bar);
  });
}

function run() {
  const tokens = tokenize(promptInput.value);
  if (!tokens.length) {
    showToast("Enter a prompt first");
    return;
  }
  state.tokens = tokens;
  state.selectedToken = 0;
  buildVocab(tokens);
  const { X, attentions, energies } = runModel(tokens);
  state.attentions = attentions;
  state.energies = energies;

  renderTokens(tokens);
  drawAttention();
  renderNextToken(X[X.length - 1]);
  renderEnergy(energies);
}

function initSelectors() {
  layerSelect.innerHTML = "";
  headSelect.innerHTML = "";
  for (let i = 0; i < config.layers; i += 1) {
    const option = document.createElement("option");
    option.value = i;
    option.textContent = `Layer ${i + 1}`;
    layerSelect.appendChild(option);
  }
  for (let i = 0; i < config.heads; i += 1) {
    const option = document.createElement("option");
    option.value = i;
    option.textContent = `Head ${i + 1} (${archetypes[i % archetypes.length]})`;
    headSelect.appendChild(option);
  }
}

function copyLink() {
  const url = new URL(window.location.href);
  url.searchParams.set("seed", state.seed);
  url.searchParams.set("prompt", promptInput.value.trim());
  navigator.clipboard.writeText(url.toString()).then(() => {
    showToast("Share link copied");
  });
}

function loadFromURL() {
  const params = new URLSearchParams(window.location.search);
  const seed = params.get("seed") || Math.random().toString(36).slice(2, 10);
  const prompt = params.get("prompt");
  if (prompt) promptInput.value = prompt;
  setSeed(seed);
}

remixButton.addEventListener("click", () => {
  setSeed(Math.random().toString(36).slice(2, 10));
  run();
});

runButton.addEventListener("click", run);
shareButton.addEventListener("click", copyLink);

layerSelect.addEventListener("change", drawAttention);
headSelect.addEventListener("change", drawAttention);
modeSelect.addEventListener("change", run);
temperatureInput.addEventListener("input", drawAttention);

energyToggle.addEventListener("click", () => {
  energyPanel.classList.toggle("hidden");
  energyToggle.textContent = energyPanel.classList.contains("hidden")
    ? "Show"
    : "Hide";
});

initSelectors();
loadFromURL();
run();
