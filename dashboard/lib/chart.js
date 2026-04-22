// Minimal chart helper — one place so we can swap libraries later.
// Uses Chart.js via ESM CDN. For the scale we're working at (100-1000
// updates per run), Chart.js performance is fine.

import Chart from 'https://esm.sh/chart.js@4/auto';

Chart.defaults.color = '#8a93a6';
Chart.defaults.borderColor = '#262a36';
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';

const PALETTE = {
  policy_loss:   '#f87171',  // red
  value_loss:    '#60a5fa',  // blue
  entropy_loss:  '#34d399',  // green
  approx_kl:     '#fbbf24',  // amber
  mean_reward:   '#c084fc',  // purple
  win_rate:      '#f472b6',  // pink
};

/** Render a multi-series line chart on a <canvas>.
 *
 * points: [{update, policy_loss, value_loss, entropy_loss, approx_kl, ...}]
 * keys:   which metrics to plot. Keys missing from a point become gaps.
 */
export function lineChart(canvas, points, keys, { yLabel = 'value' } = {}) {
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
  return new Chart(canvas, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { title: { display: true, text: 'update' }, grid: { color: 'rgba(96,165,250,0.05)' } },
        y: { title: { display: true, text: yLabel }, grid: { color: 'rgba(96,165,250,0.05)' } },
      },
      plugins: {
        legend: { position: 'bottom', labels: { padding: 10, boxWidth: 10, boxHeight: 10 } },
      },
    },
  });
}
