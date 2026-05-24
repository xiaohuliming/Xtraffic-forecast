import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def rfft_split(x, k_low):
    T = x.shape[0]
    F = np.fft.rfft(x)
    F_low = F.copy(); F_low[k_low:] = 0
    F_high = F.copy(); F_high[:k_low] = 0
    return np.fft.irfft(F_low, n=T), np.fft.irfft(F_high, n=T), np.abs(F)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--region", default="Alameda")
    p.add_argument("--data_dir", default="outputs/dist_net/region_data")
    p.add_argument("--out", default="outputs/fourier_test/fourier_viz.html")
    p.add_argument("--n_examples", type=int, default=4)
    p.add_argument("--n_agg", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    samples_h5 = data_dir / f"{args.region}_samples.h5"
    traffic_h5 = data_dir / f"{args.region}_traffic.h5"

    with h5py.File(samples_h5, "r") as f:
        split = f["split"][:]
        aff_mask = f["affected_mask"][:]
        sample_start = f["sample_start"][:]
        T_h = int(f.attrs["T_h"])
        T_p = int(f.attrs["T_p"])
        N = int(f.attrs["N"])
    with h5py.File(traffic_h5, "r") as f:
        flow = f["flow_series_imputed"][:]

    rng = np.random.default_rng(args.seed)
    test_idx = np.where(split == 2)[0]
    aff_count = aff_mask.sum(axis=1)
    aff_pool = test_idx[aff_count[test_idx] >= 5]
    nor_pool = test_idx[aff_count[test_idx] == 0]

    aff_pick = rng.choice(aff_pool, size=args.n_examples, replace=False)
    nor_pick = rng.choice(nor_pool, size=args.n_examples, replace=False)

    channel_names = ["flow", "occupancy", "speed"]

    examples = []
    for idx in list(aff_pick) + list(nor_pick):
        is_aff = idx in aff_pick
        s0 = int(sample_start[idx])
        aff_nodes = np.where(aff_mask[idx])[0]
        node = int(aff_nodes[0]) if (is_aff and aff_nodes.size > 0) else int(rng.integers(0, N))

        per_channel = {}
        for ch in range(3):
            hist = flow[s0:s0 + T_h, node, ch].astype(float)
            fut = flow[s0 + T_h:s0 + T_h + T_p, node, ch].astype(float)
            spec_full = []
            for K in range(1, T_h // 2 + 2):
                low, high, mag = rfft_split(hist, K)
                spec_full.append({
                    "K": K,
                    "low": low.tolist(),
                    "high": high.tolist(),
                    "mag": mag.tolist(),
                })
            per_channel[channel_names[ch]] = {
                "hist": hist.tolist(),
                "fut": fut.tolist(),
                "decomp": spec_full,
            }
        examples.append({
            "idx": int(idx),
            "node": node,
            "is_affected": bool(is_aff),
            "n_affected_in_sample": int(aff_count[idx]),
            "channels": per_channel,
        })

    # Aggregate energy stats for each (channel, K)
    n_agg = args.n_agg
    aff_test = test_idx[aff_count[test_idx] >= 5]
    nor_test = test_idx[aff_count[test_idx] == 0]
    aff_samp = rng.choice(aff_test, size=min(n_agg, aff_test.size), replace=False)
    nor_samp = rng.choice(nor_test, size=min(n_agg, nor_test.size), replace=False)

    agg = {}
    for ch in range(3):
        per_k = {}
        for K in range(1, T_h // 2 + 2):
            def energy(idx_set, only_aff_node):
                vals = []
                for s_idx in idx_set:
                    s0_ = int(sample_start[s_idx])
                    if only_aff_node and aff_mask[s_idx].any():
                        nodes = np.where(aff_mask[s_idx])[0][:3]
                    else:
                        nodes = rng.choice(N, size=3, replace=False)
                    for nd in nodes:
                        x = flow[s0_:s0_ + T_h, int(nd), ch].astype(float)
                        F = np.fft.rfft(x)
                        e_total = (np.abs(F) ** 2).sum()
                        if e_total < 1e-9:
                            continue
                        e_high = (np.abs(F[K:]) ** 2).sum()
                        vals.append(e_high / e_total)
                return vals

            r_aff = energy(aff_samp, True)
            r_nor = energy(nor_samp, False)
            per_k[K] = {"affected": r_aff, "normal": r_nor}
        agg[channel_names[ch]] = per_k

    payload = {
        "region": args.region,
        "T_h": T_h,
        "T_p": T_p,
        "examples": examples,
        "agg": agg,
    }

    json_blob = json.dumps(payload, separators=(",", ":"))

    html = HTML_TEMPLATE.replace("__PAYLOAD__", json_blob)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"saved: {out_path}")
    print(f"  examples: {len(examples)}  agg size per (ch,K): "
          f"aff~{len(agg['flow'][3]['affected'])}, nor~{len(agg['flow'][3]['normal'])}")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Fourier 分解测试 — 事故 vs 正常</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", sans-serif;
         margin: 24px; max-width: 1280px; background: #fafafa; color: #222; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  h2 { font-size: 17px; margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
  .panel { background: white; border: 1px solid #e0e0e0; border-radius: 6px;
           padding: 14px 16px; margin: 10px 0; }
  .ctrl { display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
          padding: 10px; background: #f3f3f3; border-radius: 4px; margin-bottom: 10px; }
  .ctrl label { font-size: 13px; }
  .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; }
  .summary { font-size: 13px; padding: 8px 12px; background: #eef5ff;
             border-left: 3px solid #4a8; border-radius: 3px; }
  .badge { display: inline-block; padding: 1px 6px; border-radius: 3px;
           font-size: 11px; font-weight: 600; margin-left: 6px; }
  .badge-aff { background: #fde2e2; color: #b03030; }
  .badge-nor { background: #e2e8fd; color: #2050b0; }
  table { border-collapse: collapse; margin-top: 10px; font-size: 13px; }
  th, td { border: 1px solid #ddd; padding: 4px 10px; text-align: right; }
  th { background: #f0f0f0; }
  .note { font-size: 12px; color: #666; margin-top: 6px; }
</style>
</head>
<body>

<h1>Fourier 分解测试 — 能否区分"正常流量"和"事故残差"</h1>
<div class="note">数据：XTraffic / <span id="region-name"></span> 区 · history 窗口 T_h=<span id="th"></span> 步 (5min/步, 共 1 小时)</div>

<div class="panel">
  <h2>① 控制面板</h2>
  <div class="ctrl">
    <label>Channel:
      <select id="sel-ch">
        <option value="flow">flow (车流量) ← 主要预测目标</option>
        <option value="occupancy">occupancy (占道率)</option>
        <option value="speed">speed (车速)</option>
      </select>
    </label>
    <label>Cutoff K (低频保留的 bins 数):
      <input id="sel-k" type="range" min="1" max="7" value="3" step="1">
      <span id="k-val">3</span>
    </label>
    <label>样本对:
      <select id="sel-pair"></select>
    </label>
  </div>
  <div class="summary" id="agg-summary"></div>
</div>

<div class="panel">
  <h2>② 时域分解：原始信号 → 低频(送 NormalBranch) + 高频(送 IncidentBranch)</h2>
  <div class="grid">
    <div id="plot-aff-time"></div>
    <div id="plot-nor-time"></div>
  </div>
  <div class="note">蓝色 = 1 小时 history；绿色虚线 = 未来 1 小时；红色 = 低频重构；紫色 = 高频残差</div>
</div>

<div class="panel">
  <h2>③ 频谱：每个 bin 的能量幅值</h2>
  <div class="grid">
    <div id="plot-aff-spec"></div>
    <div id="plot-nor-spec"></div>
  </div>
  <div class="note">竖虚线 = 当前 K cutoff。bin 0 是 DC (均值)；bin 越大 → 频率越高</div>
</div>

<div class="panel">
  <h2>④ 总体统计：高频能量占比分布（affected 节点 vs normal 节点）</h2>
  <div id="plot-hist"></div>
  <div id="stat-table"></div>
</div>

<script>
const DATA = __PAYLOAD__;

document.getElementById("region-name").textContent = DATA.region;
document.getElementById("th").textContent = DATA.T_h;

// Populate sample-pair selector
const selPair = document.getElementById("sel-pair");
const affEx = DATA.examples.filter(e => e.is_affected);
const norEx = DATA.examples.filter(e => !e.is_affected);
const nPairs = Math.min(affEx.length, norEx.length);
for (let i = 0; i < nPairs; i++) {
  const o = document.createElement("option");
  o.value = i;
  o.textContent = `Pair ${i+1}  (aff idx=${affEx[i].idx} / nor idx=${norEx[i].idx})`;
  selPair.appendChild(o);
}

function welchT(a, b) {
  const ma = a.reduce((s,x)=>s+x,0)/a.length;
  const mb = b.reduce((s,x)=>s+x,0)/b.length;
  const va = a.reduce((s,x)=>s+(x-ma)**2,0)/(a.length-1);
  const vb = b.reduce((s,x)=>s+(x-mb)**2,0)/(b.length-1);
  const t = (ma-mb)/Math.sqrt(va/a.length + vb/b.length);
  return { ma, mb, t };
}

function median(arr) {
  const s = [...arr].sort((a,b)=>a-b);
  return s[Math.floor(s.length/2)];
}

function plotExample(ex, K, ch, divTime, divSpec, color) {
  const channel = ex.channels[ch];
  const T_h = DATA.T_h;
  const T_p = DATA.T_p;
  const decomp = channel.decomp.find(d => d.K === K);
  const tHist = Array.from({length: T_h}, (_,i)=>i);
  const tFut = Array.from({length: T_p}, (_,i)=>i+T_h);

  const title = (ex.is_affected ? "AFFECTED" : "NORMAL")
    + ` · idx=${ex.idx} node=${ex.node} aff_nodes=${ex.n_affected_in_sample}`;

  Plotly.newPlot(divTime, [
    { x: tHist, y: channel.hist, mode: "lines+markers", name: "history",
      line: {color: color, width: 2} },
    { x: tFut, y: channel.fut, mode: "lines", name: "future",
      line: {color: "green", width: 1.5, dash: "dash"} },
    { x: tHist, y: decomp.low, mode: "lines", name: `low-freq (K=${K})`,
      line: {color: "red", width: 2.5} },
    { x: tHist, y: decomp.high, mode: "lines", name: "high-freq",
      line: {color: "purple", width: 2}, yaxis: "y2" },
  ], {
    title: {text: title, font: {size: 12}},
    height: 320, margin: {l: 50, r: 50, t: 35, b: 35},
    xaxis: {title: "step (5 min/step)"},
    yaxis: {title: ch},
    yaxis2: {title: "high-freq", overlaying: "y", side: "right",
             showgrid: false, zeroline: true, zerolinecolor: "#999"},
    shapes: [{type: "line", x0: T_h-0.5, x1: T_h-0.5, yref: "paper", y0: 0, y1: 1,
              line: {color: "#aaa", dash: "dot"}}],
    legend: {orientation: "h", y: -0.18},
  }, {displayModeBar: false});

  const mag = decomp.mag;
  const freqs = mag.map((_,i)=>i);
  const colors = freqs.map(f => f < K ? "#3a7" : "#a48");
  Plotly.newPlot(divSpec, [{
    x: freqs, y: mag, type: "bar", marker: {color: colors},
    text: mag.map(v=>v.toFixed(1)), textposition: "outside",
  }], {
    title: {text: `|FFT bins|  · 绿=低频→Normal 紫=高频→Incident · K=${K}`, font: {size: 11}},
    height: 260, margin: {l: 50, r: 20, t: 30, b: 35},
    xaxis: {title: "frequency bin", dtick: 1},
    yaxis: {title: "|F[k]|"},
    shapes: [{type: "line", x0: K-0.5, x1: K-0.5, yref: "paper", y0: 0, y1: 1,
              line: {color: "red", dash: "dash", width: 2}}],
  }, {displayModeBar: false});
}

function plotHist(ch, K) {
  const data = DATA.agg[ch][K];
  const a = data.affected;
  const n = data.normal;
  const { ma, mb, t } = welchT(a, n);
  const medA = median(a), medN = median(n);

  const maxV = Math.max(...a, ...n);
  const nBins = 40;
  const binWidth = maxV / nBins;
  Plotly.newPlot("plot-hist", [
    { x: n, type: "histogram", name: `normal (μ=${mb.toFixed(4)}, med=${medN.toFixed(4)}, n=${n.length})`,
      marker: {color: "steelblue"}, opacity: 0.55, histnorm: "probability",
      xbins: {start: 0, end: maxV, size: binWidth} },
    { x: a, type: "histogram", name: `affected (μ=${ma.toFixed(4)}, med=${medA.toFixed(4)}, n=${a.length})`,
      marker: {color: "firebrick"}, opacity: 0.55, histnorm: "probability",
      xbins: {start: 0, end: maxV, size: binWidth} },
  ], {
    title: {text: `高频能量占比分布  · channel=${ch} · K=${K}`, font: {size: 13}},
    barmode: "overlay", height: 360, margin: {l: 60, r: 30, t: 40, b: 50},
    xaxis: {title: `high-freq energy ratio  (bins≥${K} / total)`},
    yaxis: {title: "probability"},
    shapes: [
      {type: "line", x0: mb, x1: mb, yref: "paper", y0: 0, y1: 1,
       line: {color: "steelblue", dash: "dash", width: 2}},
      {type: "line", x0: ma, x1: ma, yref: "paper", y0: 0, y1: 1,
       line: {color: "firebrick", dash: "dash", width: 2}},
    ],
  }, {displayModeBar: false});

  const ratio = ma / Math.max(mb, 1e-9);
  document.getElementById("stat-table").innerHTML = `
    <table>
      <tr><th></th><th>n</th><th>mean</th><th>median</th><th>ratio (aff/nor)</th><th>Welch t</th></tr>
      <tr><td>normal</td><td>${n.length}</td><td>${mb.toFixed(4)}</td><td>${medN.toFixed(4)}</td><td rowspan="2" style="text-align:center;font-weight:700;font-size:15px;">${ratio.toFixed(2)}x</td><td rowspan="2" style="text-align:center;">${t.toFixed(3)}</td></tr>
      <tr><td>affected</td><td>${a.length}</td><td>${ma.toFixed(4)}</td><td>${medA.toFixed(4)}</td></tr>
    </table>
  `;

  let verdict = "";
  if (Math.abs(ratio - 1) < 0.15) verdict = "<b>结论：两组分布几乎完全重叠 — Fourier 分解无法分离正常/事故信号</b>";
  else if (ratio > 1.5) verdict = "<b>结论：affected 组高频能量显著更高 — Fourier 可能有用</b>";
  else verdict = "<b>结论：有微弱差异但区分能力弱</b>";
  document.getElementById("agg-summary").innerHTML =
    `当前选项下，affected 节点高频能量占比 = ${ma.toFixed(4)} | normal = ${mb.toFixed(4)} | 比值 = ${ratio.toFixed(2)}x · Welch t=${t.toFixed(2)}<br>${verdict}`;
}

function refresh() {
  const ch = document.getElementById("sel-ch").value;
  const K = parseInt(document.getElementById("sel-k").value);
  const pairIdx = parseInt(document.getElementById("sel-pair").value);
  document.getElementById("k-val").textContent = K;
  plotExample(affEx[pairIdx], K, ch, "plot-aff-time", "plot-aff-spec", "#b03030");
  plotExample(norEx[pairIdx], K, ch, "plot-nor-time", "plot-nor-spec", "#2050b0");
  plotHist(ch, K);
}

document.getElementById("sel-ch").addEventListener("change", refresh);
document.getElementById("sel-k").addEventListener("input", refresh);
document.getElementById("sel-pair").addEventListener("change", refresh);
refresh();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
