export const DEFAULT_ENDPOINT = 'http://10.8.0.27:31188/workers';
export const PALETTE = ['#76dce1', '#b59af8', '#f0ac84', '#8cd9ad', '#88b4fa', '#ec94c4'];
const modelColors = new Map();
function modelColor(id) { if (!modelColors.has(id)) modelColors.set(id, PALETTE[modelColors.size % PALETTE.length]); return modelColors.get(id); }
export const numeric = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : null;
export function hash(value) { let n = 2166136261; for (const c of String(value)) n = Math.imul(n ^ c.charCodeAt(0), 16777619); return n >>> 0; }
export function shortAddress(value) { try { return new URL(value).host; } catch { return String(value || 'unknown'); } }
export function escapeHTML(value) { return String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
export function validateEndpoint(value) { const url = new URL(value); if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) throw new Error('请填写不含账号密码的 HTTP 或 HTTPS 接口地址。'); return url.href; }
function sumMeasured(ranks, key) { const values = ranks.map(r => numeric(r[key])); return values.length && values.every(v => v !== null) ? values.reduce((a,b) => a+b, 0) : null; }
export function normalizePayload(payload) {
  if (!payload || !Array.isArray(payload.workers)) throw new Error('接口响应缺少 workers 数组。');
  const seen = new Set();
  const workers = payload.workers.map((raw, index) => {
    if (!raw || typeof raw !== 'object' || typeof raw.url !== 'string') throw new Error(`worker ${index + 1} 缺少有效的 url。`);
    const id = String(raw.id || raw.url);
    if (seen.has(id)) return null;
    seen.add(id);
    const ranks = Array.isArray(raw.engine_load?.loads) ? raw.engine_load.loads.filter(r => r && typeof r === 'object') : [];
    const usages = ranks.map(r => numeric(r.token_usage)).filter(v => v !== null && v <= 1);
    const modelIds = [...new Set((Array.isArray(raw.models) ? raw.models : []).map(m => typeof m?.id === 'string' ? m.id : null).filter(Boolean))];
    if (!modelIds.length) modelIds.push(String(raw.model_id || 'Unknown model'));
    const running = sumMeasured(ranks, 'num_running_reqs');
    const waiting = sumMeasured(ranks, 'num_waiting_reqs');
    const healthy = raw.is_healthy === true && (!raw.status || raw.status === 'ready');
    const capacity = sumMeasured(ranks, 'max_total_num_tokens');
    const kv = usages.length === ranks.length && usages.length ? usages.reduce((a,b) => a+b, 0) / usages.length : null;
    return { id, url: raw.url, address: shortAddress(raw.url), modelIds, runtime: String(raw.runtime_type || 'unknown'), version: String(raw.labels?.version || raw.engine_load?.version || '—'), status: !healthy || raw.stalled ? 'offline' : waiting > 0 ? 'waiting' : 'healthy', healthy, load: numeric(raw.load), running, waiting, kv, waitingTokens: sumMeasured(ranks, 'num_waiting_uncached_tokens'), usedTokens: capacity > 0 ? sumMeasured(ranks, 'num_used_tokens') : null, capacity: capacity > 0 ? capacity : null, maxRunning: sumMeasured(ranks, 'max_running_requests') || null, engineTimestamp: typeof raw.engine_load?.timestamp === 'string' && Number.isFinite(Date.parse(raw.engine_load.timestamp)) ? raw.engine_load.timestamp : null, labels: raw.labels || {}, capabilities: [...new Set((Array.isArray(raw.models) ? raw.models : []).flatMap(m => Array.isArray(m.model_type) ? m.model_type : []).filter(x => typeof x === 'string'))], raw };
  }).filter(Boolean).sort((a,b) => a.address.localeCompare(b.address, undefined, {numeric:true}));
  const groups = new Map();
  for (const worker of workers) for (const id of worker.modelIds) { if (!groups.has(id)) groups.set(id, { id, workers: [], color: modelColor(id) }); groups.get(id).workers.push(worker); }
  return { workers, groups: [...groups.values()].sort((a,b) => a.id.localeCompare(b.id)), receivedAt: Date.now() };
}
export function summarize(workers) {
  const sum = key => { const known = workers.filter(w => w[key] !== null); return { value: known.length ? known.reduce((n,w) => n+w[key], 0) : workers.length ? null : 0, partial: known.length < workers.length }; };
  return { total: workers.length, healthy: workers.filter(w => w.healthy).length, inflight: sum('load'), running: sum('running'), waiting: sum('waiting') };
}
export function relativeLoads(workers) {
  const measured = workers.filter(w => w.status !== 'offline' && w.load !== null).map(w => w.load);
  const min = measured.length ? Math.min(...measured) : 0;
  const max = measured.length ? Math.max(...measured) : 0;
  return new Map(workers.map(w => [w.id, { heat: w.status === 'offline' || w.load === null ? null : max === min ? 0 : (w.load-min)/(max-min), min, max }]));
}
export function heatColor(heat) {
  if (heat === null) return '#687387';
  const stops = [[112,220,230], [246,192,105], [255,101,108]];
  const t = Math.max(0, Math.min(1, heat)) * 2;
  const a = stops[Math.min(1,Math.floor(t))], b = stops[Math.min(2,Math.floor(t)+1)], f = t >= 2 ? 1 : t % 1;
  return '#' + a.map((v,i) => Math.round(v+(b[i]-v)*f).toString(16).padStart(2,'0')).join('');
}

// Explicit demo only: plausible shapes based on the user's SMG payloads, not live observations.
export function demoPayload(t = 0) {
  const models = [['Qwen3.8-27B', 4], ['DeepSeek-V3.2', 3], ['Llama-3.3-70B', 3], ['Qwen3-VL-32B', 2]];
  const workers = models.flatMap(([model, count], g) => Array.from({length:count}, (_, i) => {
    const wave = Math.sin(t * .00035 + i * 2.4 + g);
    const running = Math.round(10 + wave * 7 + g * 3);
    const waiting = g === 1 && i === 1 ? 8 + Math.round(3 * wave) : i === 2 ? Math.max(0, Math.round(wave * 3)) : 0;
    const offline = g === 2 && i === 2;
    const kv = Math.min(.99, .22 + (wave + 1) * .16 + g * .11);
    return { id:`demo-${g}-${i}`, model_id:model, url:`http://10.244.${g}.${40+i}:9000`, models:[{id:model,model_type:['chat','completions','responses', ...(g === 3 ? ['vision'] : []),'tools']}], runtime_type:i%2?'vllm':'sglang', worker_type:'regular', connection_mode:'http', labels:{version:i%2?'0.29.0':'0.5.21', 'smg.ai/pod-name':`inference-${g}-worker-${i+1}`,max_running_requests:'48'}, is_healthy:!offline, status:offline?'not_ready':'ready', load:offline?0:running+waiting+1, engine_load:offline?undefined:{timestamp:new Date().toISOString(),loads:[{dp_rank:0,num_running_reqs:running,num_waiting_reqs:waiting,token_usage:kv,num_waiting_uncached_tokens:waiting*8064,num_used_tokens:Math.round(kv*156107),max_total_num_tokens:156107,max_running_requests:48}]}};
  }));
  return {workers};
}
