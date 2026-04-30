// Minimal chart helper — one place so we can swap libraries later.
// Uses Chart.js via ESM CDN. For the scale we're working at (100-1000
// updates per run), Chart.js performance is fine.

import Chart from 'https://esm.sh/chart.js@4/auto';

Chart.defaults.color = '#8a93a6';
Chart.defaults.borderColor = '#262a36';
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';

const PALETTE = {
  policy_loss:        '#f87171',  // red
  value_loss:         '#60a5fa',  // blue
  entropy_loss:       '#34d399',  // green
  approx_kl:          '#fbbf24',  // amber
  mean_reward:        '#c084fc',  // purple
  win_rate:                '#10b981',  // green (W of WDL — secondary, vs training opponent)
  draw_rate:               '#9ca3af',  // gray  (D of WDL — turtling / mutual stall)
  loss_rate:               '#ef4444',  // red   (L of WDL — actually losing)
  win_rate_vs_leaderboard: '#f472b6',  // pink (primary — vs top-N champion archive)
  episodes_completed:      '#22d3ee',  // cyan
  pool_size:               '#a78bfa',  // violet
  clip_fraction:           '#fb923c',  // orange (PPO step-too-large signal)
  grad_norm:               '#facc15',  // yellow (instability spikes)
  explained_variance:      '#a3e635',  // lime (critic honesty)
  mean_episode_return:     '#c084fc',  // purple
  episode_return_p10:      '#7c3aed',
  episode_return_p50:      '#a78bfa',
  episode_return_p90:      '#7c3aed',
  episode_return_min:      '#3f3f46',
  episode_return_max:      '#3f3f46',
};

/** Episode-return band chart: faint min/max envelope, shaded p10–p90 band,
 * mean line on top. Renders only the keys present in `points`. Designed
 * to match what AlphaStar / OpenAI Five published — mean alone hides
 * variance collapse.
 */
export function bandChart(canvas, points, opts = {}) {
  const { yLabel = 'episode return', xLabel = 'update' } = opts;
  const labels = points.map((_, i) => i + 1);
  const has = k => points.some(p => p[k] != null);
  const series = (k) => points.map(p => (p[k] ?? null));

  const datasets = [];
  if (has('episode_return_min') && has('episode_return_max')) {
    datasets.push({
      label: 'min',
      data: series('episode_return_min'),
      borderColor: 'rgba(63,63,70,0.6)',
      borderWidth: 1,
      borderDash: [3, 3],
      pointRadius: 0,
      fill: false,
      spanGaps: true,
    });
    datasets.push({
      label: 'max',
      data: series('episode_return_max'),
      borderColor: 'rgba(63,63,70,0.6)',
      borderWidth: 1,
      borderDash: [3, 3],
      pointRadius: 0,
      fill: false,
      spanGaps: true,
    });
  }
  if (has('episode_return_p10') && has('episode_return_p90')) {
    // Stack the p10 + p90 lines with a fill from p10 → p90 so the band reads.
    datasets.push({
      label: 'p10',
      data: series('episode_return_p10'),
      borderColor: 'rgba(124,58,237,0.6)',
      backgroundColor: 'rgba(124,58,237,0.15)',
      borderWidth: 1,
      pointRadius: 0,
      fill: false,
      spanGaps: true,
    });
    datasets.push({
      label: 'p10–p90',
      data: series('episode_return_p90'),
      borderColor: 'rgba(124,58,237,0.6)',
      backgroundColor: 'rgba(124,58,237,0.15)',
      borderWidth: 1,
      pointRadius: 0,
      fill: '-1',  // fill to previous dataset (p10) to shade the band
      spanGaps: true,
    });
  }
  if (has('mean_episode_return')) {
    datasets.push({
      label: 'mean',
      data: series('mean_episode_return'),
      borderColor: PALETTE.mean_episode_return,
      backgroundColor: PALETTE.mean_episode_return,
      borderWidth: 2,
      pointRadius: 0,
      fill: false,
      spanGaps: true,
      tension: 0.2,
    });
  }
  return new Chart(canvas, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { title: { display: true, text: xLabel }, grid: { color: 'rgba(96,165,250,0.05)' } },
        y: { title: { display: true, text: yLabel }, grid: { color: 'rgba(96,165,250,0.05)' } },
      },
      plugins: {
        legend: { position: 'bottom', labels: { padding: 10, boxWidth: 10, boxHeight: 10 } },
      },
    },
  });
}

/** Render a multi-series line chart on a <canvas>.
 *
 * points: [{update, policy_loss, value_loss, entropy_loss, approx_kl, ...}]
 * keys:   which metrics to plot. Keys missing from a point become gaps.
 * opts:   { yLabel, yMin, yMax, xLabel, label } — optional axis tweaks
 */
export function lineChart(canvas, points, keys, opts = {}) {
  const { yLabel = 'value', yMin, yMax, xLabel = 'update', yPct = false } = opts;
  const labels = points.map((_, i) => i + 1);
  const datasets = keys.map(k => ({
    label: k,
    data: points.map(p => (p[k] ?? null)),
    borderColor: PALETTE[k] ?? '#e6e8ef',
    backgroundColor: PALETTE[k] ?? '#e6e8ef',
    borderWidth: 1.5,
    pointRadius: 0,
    spanGaps: true,
    tension: 0.2,
  }));
  const yAxis = {
    title: { display: true, text: yLabel },
    grid: { color: 'rgba(96,165,250,0.05)' },
  };
  if (yMin !== undefined) yAxis.min = yMin;
  if (yMax !== undefined) yAxis.max = yMax;
  if (yPct) {
    yAxis.ticks = {
      callback: (v) => `${(v * 100).toFixed(0)}%`,
    };
  }
  return new Chart(canvas, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { title: { display: true, text: xLabel }, grid: { color: 'rgba(96,165,250,0.05)' } },
        y: yAxis,
      },
      plugins: {
        legend: { position: 'bottom', labels: { padding: 10, boxWidth: 10, boxHeight: 10 } },
        tooltip: yPct ? {
          callbacks: {
            label: (ctx) => {
              const v = ctx.parsed.y;
              return v == null ? `${ctx.dataset.label}: —`
                               : `${ctx.dataset.label}: ${(v * 100).toFixed(1)}%`;
            },
          },
        } : undefined,
      },
    },
  });
}
