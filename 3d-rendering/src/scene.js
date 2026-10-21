import * as THREE from '../vendor/three.module.js';
import { hash, relativeLoads, heatColor, escapeHTML } from './data.js';

const TAU = Math.PI * 2;
const overviewDistance = count => count > 2 ? 42 : 34;
function random(seed) {
  return () => {
    seed |= 0; seed = seed + 0x6d2b79f5 | 0;
    let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
function circle(radius, color, opacity = .18, segments = 160) {
  const points = Array.from({ length: segments }, (_, i) => new THREE.Vector3(Math.cos(i / segments * TAU) * radius, 0, Math.sin(i / segments * TAU) * radius));
  return new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false, blending: THREE.AdditiveBlending }));
}
function glowTexture() {
  const canvas = document.createElement('canvas'); canvas.width = canvas.height = 128;
  const ctx = canvas.getContext('2d'), gradient = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  for (const [at, alpha] of [[0, 1], [.12, .8], [.32, .22], [.65, .045], [1, 0]]) gradient.addColorStop(at, `rgba(255,255,255,${alpha})`);
  ctx.fillStyle = gradient; ctx.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(canvas);
}
function glow(texture, color, size, opacity) {
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false }));
  sprite.scale.setScalar(size); return sprite;
}
const planetVertex = `
varying vec3 vNormal; varying vec3 vPosition; varying vec3 vView;
void main() {
  vNormal = normalize(normalMatrix * normal); vPosition = position;
  vec4 mv = modelViewMatrix * vec4(position, 1.); vView = -mv.xyz;
  gl_Position = projectionMatrix * mv;
}`;
const planetFragment = `
uniform vec3 tint; uniform float time; uniform float hot;
varying vec3 vNormal; varying vec3 vPosition; varying vec3 vView;
float pattern(vec3 p) { return sin(p.x*8.+sin(p.z*11.+time*.12)*1.8)*sin(p.y*12.+sin(p.x*6.)*2.)*.5+.5; }
void main() {
  vec3 n = normalize(vNormal);
  float rim = pow(1.-max(0.,dot(n,normalize(vView))),2.5);
  float p = pattern(vPosition);
  float bands = sin(vPosition.y*37.+sin(vPosition.x*15.)*2.+time*.08)*.5+.5;
  float light = .3+.7*max(0.,dot(n,normalize(vec3(-.4,.7,1.))));
  vec3 base = tint*(.13+p*.42+bands*.13)*light;
  gl_FragColor = vec4(base+tint*rim*(1.7+hot*.6)+tint*pow(p,10.)*.4,1.);
}`;
function planet(radius, color) {
  return new THREE.Mesh(new THREE.SphereGeometry(radius, 40, 28), new THREE.ShaderMaterial({
    vertexShader: planetVertex, fragmentShader: planetFragment,
    uniforms: { tint: { value: new THREE.Color(color) }, time: { value: 0 }, hot: { value: 0 } },
  }));
}
function disposeObject(object) {
  object.traverse(o => {
    o.geometry?.dispose();
    if (o.material) for (const material of Array.isArray(o.material) ? o.material : [o.material]) material.dispose();
  });
  object.removeFromParent();
}

export class Universe {
  constructor(container, callbacks = {}) {
    this.container = container; this.callbacks = callbacks;
    this.groups = new Map(); this.nodes = new Map(); this.time = 0;
    this.paused = matchMedia('(prefers-reduced-motion: reduce)').matches;
    this.selected = null; this.focus = null; this.lastFrame = performance.now();
    this.width = 1; this.height = 1;
    this.scene = new THREE.Scene(); this.scene.fog = new THREE.FogExp2('#070c16', .0065);
    this.camera = new THREE.PerspectiveCamera(43, 1, .1, 250);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
    this.renderer.setClearColor('#060b13', 0);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping; this.renderer.toneMappingExposure = 1.25;
    this.renderer.domElement.setAttribute('aria-label', '拖拽旋转星系，滚轮缩放，点击星球选择 worker');
    this.renderer.domElement.setAttribute('role', 'img'); container.append(this.renderer.domElement);
    this.labelLayer = document.createElement('div'); this.labelLayer.className = 'world-labels'; container.append(this.labelLayer);
    this.texture = glowTexture();
    this.theta = 0; this.phi = .86; this.radius = 34; this.desiredRadius = 34;
    this.target = new THREE.Vector3(); this.raycaster = new THREE.Raycaster(); this.pointer = new THREE.Vector2();
    this.tmp = new THREE.Vector3(); this.origin = new THREE.Vector3(); this.side = new THREE.Vector3();
    this.cameraDir = new THREE.Vector3(); this.labelPosition = new THREE.Vector3(); this.tangent = new THREE.Vector3();
    this.makeBackground(); this.makeGateway(); this.bindInput();
    this.resizeObserver = new ResizeObserver(() => this.resize()); this.resizeObserver.observe(container); this.resize();
    this.frame = this.frame.bind(this); this.raf = requestAnimationFrame(this.frame);
  }
  makeLabel(className, html, color) {
    const el = document.createElement('div'); el.className = `world-label ${className}`; el.innerHTML = html;
    if (color) el.style.setProperty('--label-color', color);
    this.labelLayer.append(el); return el;
  }
  makeBackground() {
    const rng = random(717), positions = [], colors = [], color = new THREE.Color();
    for (let i = 0; i < 3400; i++) {
      const radius = 38 + rng() * 75, theta = rng() * TAU, y = rng() * 2 - 1, ring = Math.sqrt(1 - y * y);
      positions.push(radius * ring * Math.cos(theta), radius * y, radius * ring * Math.sin(theta));
      color.setHSL(.54 + rng() * .14, .12 + rng() * .2, .23 + rng() * .45); colors.push(color.r, color.g, color.b);
    }
    const geom = new THREE.BufferGeometry(); geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3)); geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    this.scene.add(new THREE.Points(geom, new THREE.PointsMaterial({ size: .075, vertexColors: true, transparent: true, opacity: .75, sizeAttenuation: true, depthWrite: false })));
    const dustPositions = [];
    for (let i = 0; i < 900; i++) { const angle = rng() * TAU, r = 3 + rng() * 22; dustPositions.push(Math.cos(angle) * r, (rng() - .5) * .7 - 1.2, Math.sin(angle) * r); }
    const dustGeom = new THREE.BufferGeometry(); dustGeom.setAttribute('position', new THREE.Float32BufferAttribute(dustPositions, 3));
    this.scene.add(new THREE.Points(dustGeom, new THREE.PointsMaterial({ color: '#52779d', size: .028, transparent: true, opacity: .28, depthWrite: false })));
    const referenceRing = circle(15.8, '#446383', .12, 256); referenceRing.position.y = -.8; this.scene.add(referenceRing);
    const marks = [];
    for (let i = 0; i < 100; i++) { const a = i / 100 * TAU, r = 15.8; marks.push(Math.cos(a) * r, -.8, Math.sin(a) * r, Math.cos(a) * (r + (i % 5 === 0 ? .25 : .1)), -.8, Math.sin(a) * (r + (i % 5 === 0 ? .25 : .1))); }
    const geomMarks = new THREE.BufferGeometry(); geomMarks.setAttribute('position', new THREE.Float32BufferAttribute(marks, 3));
    this.scene.add(new THREE.LineSegments(geomMarks, new THREE.LineBasicMaterial({ color: '#5c7d9d', transparent: true, opacity: .22 })));
    for (const [x, z, size, tint] of [[-9, -10, 30, '#263f70'], [12, 1, 25, '#184954'], [0, 12, 28, '#39315e']]) {
      const cloud = glow(this.texture, tint, size, .11); cloud.position.set(x, -3, z); this.scene.add(cloud);
    }
  }
  makeGateway() {
    this.gateway = new THREE.Group(); this.scene.add(this.gateway);
    this.core = planet(1.15, '#f6cc86'); this.core.material.uniforms.hot.value = 1; this.core.userData.gateway = true; this.gateway.add(this.core);
    this.gateway.add(glow(this.texture, '#efaa53', 7, .47)); this.gateway.add(glow(this.texture, '#ffdca5', 3.3, .26));
    this.coreRings = [];
    for (let i = 0; i < 3; i++) { const ring = circle(1.5 + i * .23, '#efc58e', .35 - i * .07); ring.rotation.x = .1 + i * .55; ring.rotation.z = .28 + i * .44; this.gateway.add(ring); this.coreRings.push(ring); }
    this.gatewayLabel = this.makeLabel('gateway-label', 'ai-gateway<small>XHIVE-PRO · CENTRAL CORE</small>');
    const halo = circle(2.15, '#d9b06d', .13); halo.rotation.x = .05; this.gateway.add(halo);
  }
  createGalaxy(group, index, total) {
    const root = new THREE.Group(); this.scene.add(root); const rng = random(hash(group.id));
    const galaxy = { id: group.id, root, color: group.color, index, angle: index / total * TAU + .25, radius: total === 1 ? 7.8 : 8.1 + (index % 2) * 2.8, speed: .018 / (1 + index * .09), nodes: [], size: 2.5 + Math.min(group.workers.length, 14) * .065 };
    const positions = [], colors = [], c = new THREE.Color(group.color);
    for (let i = 0; i < 850; i++) {
      const r = Math.pow(rng(), .7) * galaxy.size, a = r * 1.95 + (i % 3) * TAU / 3 + (rng() - .5) * .7;
      positions.push(Math.cos(a) * r, (rng() - .5) * .22, Math.sin(a) * r);
      const brightness = .2 + rng() * .6; colors.push(c.r * brightness, c.g * brightness, c.b * brightness);
    }
    const geom = new THREE.BufferGeometry(); geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3)); geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    galaxy.dust = new THREE.Points(geom, new THREE.PointsMaterial({ size: .034, vertexColors: true, transparent: true, opacity: .75, blending: THREE.AdditiveBlending, depthWrite: false })); root.add(galaxy.dust);
    const disk = glow(this.texture, group.color, galaxy.size * 3, .12); disk.position.y = -.05; root.add(disk);
    root.add(circle(galaxy.size + .2, group.color, .18)); root.add(circle(1.4, group.color, .09)); root.add(glow(this.texture, group.color, .65, .7));
    galaxy.orbit = circle(galaxy.radius, group.color, .12); galaxy.orbit.position.y = -.3; this.scene.add(galaxy.orbit);
    galaxy.label = this.makeLabel('galaxy-label', '', group.color); this.groups.set(group.id, galaxy); return galaxy;
  }
  createBeam() {
    const segments = 48, positions = new Float32Array((segments + 1) * 6), uv = new Float32Array((segments + 1) * 4), indices = [];
    for (let i = 0; i <= segments; i++) { uv[i * 4] = uv[i * 4 + 2] = i / segments; uv[i * 4 + 1] = 0; uv[i * 4 + 3] = 1; if (i < segments) { const a = i * 2; indices.push(a, a + 1, a + 2, a + 1, a + 3, a + 2); } }
    const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage)); geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2)); geometry.setIndex(indices);
    const material = new THREE.ShaderMaterial({
      transparent: true, depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
      uniforms: { color: { value: new THREE.Color('#70dce6') }, time: { value: 0 }, activity: { value: 1 }, opacity: { value: 1 } },
      vertexShader: 'varying vec2 vUv;void main(){vUv=uv;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
      fragmentShader: 'varying vec2 vUv;uniform vec3 color;uniform float time;uniform float activity;uniform float opacity;void main(){float edge=pow(1.-abs(vUv.y*2.-1.),3.);float pulse=pow(max(0.,sin(vUv.x*31.-time*(1.5+activity))),22.);float alpha=(.24+pulse*.8)*edge*opacity;gl_FragColor=vec4(color*(1.+pulse*.9),alpha);}',
    });
    const beam = new THREE.Mesh(geometry, material); beam.frustumCulled = false; this.scene.add(beam); return beam;
  }
  createNode(worker, galaxy, index, count) {
    const key = `${galaxy.id}\0${worker.id}`, body = new THREE.Group(); galaxy.root.add(body);
    const mesh = planet(.36 + Math.min(index, 4) * .015, '#70dce6'); mesh.userData.nodeKey = key; body.add(mesh);
    const hit = new THREE.Mesh(new THREE.SphereGeometry(.58, 12, 8), new THREE.MeshBasicMaterial({ visible: false })); hit.userData.nodeKey = key; body.add(hit);
    const aura = glow(this.texture, '#70dce6', 2.2, .33); body.add(aura);
    const ring = circle(.55, '#70dce6', .6, 72); ring.rotation.z = .36; body.add(ring);
    const kvGeometry = new THREE.BufferGeometry(); kvGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(65 * 3), 3));
    const kvRing = new THREE.Line(kvGeometry, new THREE.LineBasicMaterial({ color: '#70dce6', transparent: true, opacity: .85, blending: THREE.AdditiveBlending, depthWrite: false })); kvRing.rotation.copy(ring.rotation); body.add(kvRing);
    const selection = circle(.72, '#e7f8ff', .75, 90); selection.rotation.x = Math.PI / 2; selection.visible = false; body.add(selection);
    const node = { key, worker, galaxy, body, mesh, hit, aura, ring, kvRing, selection, angle: index / count * TAU + hash(worker.id) % 50 / 100, radius: 1.65 + (index % 3) * .53 + Math.floor(index / 6) * .5, speed: .13 / (1 + index * .08), world: new THREE.Vector3(), beam: this.createBeam(), heat: 0, targetHeat: 0, tint: new THREE.Color() };
    node.label = this.makeLabel('worker-label', '', '#70dce6'); this.nodes.set(key, node); return node;
  }
  setData(data) {
    const previousCount = this.groups.size, nextGroups = new Set(data.groups.map(g => g.id));
    for (const [key, node] of this.nodes) if (!data.groups.some(g => g.id === node.galaxy.id && g.workers.some(w => w.id === node.worker.id))) {
      disposeObject(node.body); disposeObject(node.beam); node.label.remove(); this.nodes.delete(key);
    }
    for (const [id, group] of this.groups) if (!nextGroups.has(id)) { disposeObject(group.root); disposeObject(group.orbit); group.label.remove(); this.groups.delete(id); }
    data.groups.forEach((group, i) => {
      const galaxy = this.groups.get(group.id) || this.createGalaxy(group, i, data.groups.length), heats = relativeLoads(group.workers); galaxy.nodes = [];
      galaxy.label.innerHTML = `${escapeHTML(group.id)}<small>MODEL CONSTELLATION · ${group.workers.length} NODES</small>`;
      group.workers.forEach((worker, index) => {
        const key = `${group.id}\0${worker.id}`, node = this.nodes.get(key) || this.createNode(worker, galaxy, index, group.workers.length);
        node.worker = worker; node.targetHeat = heats.get(worker.id).heat; node.color = heatColor(node.targetHeat);
        node.label.innerHTML = `${escapeHTML(worker.address)}<span class="label-load">${worker.load ?? '—'}</span>`;
        node.label.style.setProperty('--label-color', node.color); node.label.classList.toggle('selected', this.selected === worker.id);
        const attr = node.kvRing.geometry.attributes.position, fraction = worker.kv ?? 0;
        for (let j = 0; j < 65; j++) { const a = j / 64 * fraction * TAU; attr.setXYZ(j, Math.cos(a) * .61, 0, Math.sin(a) * .61); }
        attr.needsUpdate = true; node.kvRing.visible = worker.kv !== null; galaxy.nodes.push(node);
      });
    });
    if (this.focus?.type === 'worker' && !data.workers.some(w => w.id === this.focus.id) || this.focus?.type === 'group' && !this.groups.has(this.focus.id)) this.resetView();
    if (!this.hasData || previousCount !== data.groups.length) {
      if (!this.focus) this.desiredRadius = overviewDistance(data.groups.length);
      if (!this.hasData) this.radius = this.desiredRadius;
      this.hasData = data.groups.length > 0;
      [...this.groups.values()].forEach((group, i) => {
        group.index = i; group.angle = i / Math.max(1, this.groups.size) * TAU + .25 + this.time * .018;
        const radius = this.groups.size === 1 ? 7.8 : 8.1 + (i % 2) * 2.8;
        if (group.radius !== radius) { disposeObject(group.orbit); group.radius = radius; group.orbit = circle(radius, group.color, .12); group.orbit.position.y = -.3; this.scene.add(group.orbit); }
      });
    }
    this.renderScene(0);
  }
  focusWorker(id, groupId) {
    const node = [...this.nodes.values()].find(n => n.worker.id === id && (!groupId || n.galaxy.id === groupId)); if (!node) return;
    this.selected = id; this.focus = { type: 'worker', id, groupId: node.galaxy.id }; this.desiredRadius = 10;
    for (const n of this.nodes.values()) n.label.classList.toggle('selected', n.worker.id === id);
  }
  focusGroup(id) { if (!this.groups.has(id)) return; this.focus = { type: 'group', id }; this.selected = null; this.desiredRadius = 14; for (const n of this.nodes.values()) n.label.classList.remove('selected'); }
  resetView() { this.focus = null; this.selected = null; this.theta = 0; this.phi = .86; this.desiredRadius = overviewDistance(this.groups.size); for (const n of this.nodes.values()) n.label.classList.remove('selected'); }
  zoom(amount) { this.desiredRadius = THREE.MathUtils.clamp(this.desiredRadius * amount, 5, 85); }
  setPaused(paused) { this.paused = paused; }
  setLabels(visible) { this.labelLayer.style.display = visible ? '' : 'none'; }
  bindInput() {
    const canvas = this.renderer.domElement, pointers = new Map(); let drag = null, pinchDistance = null;
    canvas.addEventListener('pointerdown', e => {
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY }); canvas.setPointerCapture(e.pointerId);
      if (pointers.size === 1) drag = { x: e.clientX, y: e.clientY, startX: e.clientX, startY: e.clientY, moved: false }; else if (drag) drag.moved = true;
    });
    canvas.addEventListener('pointermove', e => {
      if (pointers.has(e.pointerId)) {
        pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
        if (pointers.size === 2) { const [a, b] = [...pointers.values()], d = Math.hypot(a.x - b.x, a.y - b.y); if (pinchDistance) this.zoom(pinchDistance / Math.max(1, d)); pinchDistance = d; return; }
        if (drag) {
          const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
          if (Math.hypot(e.clientX - drag.startX, e.clientY - drag.startY) > 4) drag.moved = true;
          this.theta -= dx * .005; this.phi = THREE.MathUtils.clamp(this.phi + dy * .005, .12, Math.PI - .12); drag.x = e.clientX; drag.y = e.clientY; return;
        }
      }
      const picked = this.pick(e.clientX, e.clientY); canvas.style.cursor = picked ? 'pointer' : 'grab'; this.callbacks.onHover?.(picked?.worker || null, e.clientX, e.clientY, picked?.galaxy?.id);
    });
    canvas.addEventListener('pointerup', e => {
      if (!pointers.has(e.pointerId)) return;
      const click = drag && !drag.moved && pointers.size === 1; pointers.delete(e.pointerId); pinchDistance = null;
      if (click) { const picked = this.pick(e.clientX, e.clientY); if (picked?.worker) this.callbacks.onSelect?.(picked.worker.id, picked.galaxy.id); else if (picked?.gateway) this.callbacks.onGateway?.(); }
      drag = null;
    });
    canvas.addEventListener('pointercancel', e => { pointers.delete(e.pointerId); drag = null; pinchDistance = null; });
    canvas.addEventListener('pointerleave', () => this.callbacks.onHover?.(null));
    canvas.addEventListener('wheel', e => { e.preventDefault(); this.zoom(Math.exp(e.deltaY * .001)); }, { passive: false });
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('dblclick', () => { this.resetView(); this.callbacks.onGateway?.(); });
    canvas.addEventListener('webglcontextlost', e => { e.preventDefault(); this.contextLost = true; this.callbacks.onError?.('3D 渲染上下文暂时丢失，请刷新页面恢复。'); });
  }
  pick(x, y) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.pointer.set((x - rect.left) / rect.width * 2 - 1, -(y - rect.top) / rect.height * 2 + 1); this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObjects([this.core, ...[...this.nodes.values()].map(n => n.hit)], false);
    if (!hits.length) return null; return hits[0].object.userData.gateway ? { gateway: true } : this.nodes.get(hits[0].object.userData.nodeKey);
  }
  resize() { this.width = this.container.clientWidth; this.height = this.container.clientHeight; this.camera.aspect = this.width / Math.max(1, this.height); this.camera.updateProjectionMatrix(); this.renderer.setSize(this.width, this.height); }
  projectLabel(element, position, offset = 0) {
    this.tmp.copy(position).project(this.camera);
    const shown = this.tmp.z < 1 && this.tmp.z > -1 && Math.abs(this.tmp.x) < 1.15 && Math.abs(this.tmp.y) < 1.15;
    element.style.visibility = shown ? 'visible' : 'hidden';
    if (shown) element.style.transform = `translate(${(this.tmp.x * .5 + .5) * this.width}px,${(-this.tmp.y * .5 + .5) * this.height + offset}px) translate(-50%,-50%)`;
  }
  updateBeam(node) {
    const end = node.world, positions = node.beam.geometry.attributes.position, arc = Math.min(2.3, end.length() * .17);
    this.camera.getWorldDirection(this.cameraDir); this.tangent.copy(end); this.side.crossVectors(this.tangent, this.cameraDir).normalize().multiplyScalar(.05 + (node.heat ?? 0) * .04);
    for (let i = 0; i <= 48; i++) { const t = i / 48, x = end.x * t, y = end.y * t + Math.sin(t * Math.PI) * arc, z = end.z * t; positions.setXYZ(i * 2, x + this.side.x, y + this.side.y, z + this.side.z); positions.setXYZ(i * 2 + 1, x - this.side.x, y - this.side.y, z - this.side.z); }
    positions.needsUpdate = true; node.beam.material.uniforms.time.value = this.time;
    node.beam.material.uniforms.activity.value = Math.min(4, (node.worker.load || 0) / 12);
    node.beam.material.uniforms.opacity.value = node.worker.status === 'offline' ? .16 : node.worker.load === 0 ? .55 : 1;
  }
  renderScene(dt) {
    for (const galaxy of this.groups.values()) {
      if (!this.paused) { galaxy.angle += dt * galaxy.speed; galaxy.dust.rotation.y += dt * .025; }
      galaxy.root.position.set(Math.cos(galaxy.angle) * galaxy.radius, Math.sin(galaxy.angle * .7) * .45, Math.sin(galaxy.angle) * galaxy.radius); galaxy.root.rotation.z = Math.sin(galaxy.index * 1.7) * .12;
      for (const node of galaxy.nodes) {
        if (!this.paused) node.angle += dt * node.speed;
        node.body.position.set(Math.cos(node.angle) * node.radius, Math.sin(node.angle * 1.3) * .22, Math.sin(node.angle) * node.radius);
        node.mesh.rotation.y = this.time * .1; node.mesh.material.uniforms.time.value = this.time;
        node.heat = node.targetHeat === null ? null : THREE.MathUtils.lerp(node.heat ?? 0, node.targetHeat, Math.min(1, dt * 4 || 1));
        node.tint.set(heatColor(node.heat)); node.mesh.material.uniforms.tint.value.copy(node.tint).multiplyScalar(node.worker.status === 'offline' ? .3 : 1); node.mesh.material.uniforms.hot.value = node.heat ?? 0;
        node.aura.material.color.copy(node.tint); node.aura.material.opacity = node.worker.status === 'offline' ? .12 : .25 + (node.heat ?? 0) * .18 + Math.sin(this.time * (1 + (node.heat ?? 0) * 2)) * .025;
        node.aura.scale.setScalar(1.9 + (node.heat ?? 0) * .8); node.ring.material.color.copy(node.tint); node.kvRing.material.color.copy(node.tint); node.beam.material.uniforms.color.value.copy(node.tint); node.selection.visible = node.worker.id === this.selected;
      }
    }
    this.scene.updateMatrixWorld(true); for (const node of this.nodes.values()) node.body.getWorldPosition(node.world);
    let target = this.origin;
    if (this.focus?.type === 'group') target = this.groups.get(this.focus.id)?.root.position || target;
    else if (this.focus?.type === 'worker') target = [...this.nodes.values()].find(n => n.worker.id === this.focus.id && n.galaxy.id === this.focus.groupId)?.world || target;
    const lerp = 1 - Math.exp(-dt * 4); this.target.lerp(target, lerp); this.radius = THREE.MathUtils.lerp(this.radius, this.desiredRadius, lerp);
    const mobileScale = this.width < 640 ? (this.focus ? 1.4 : 1.9) : 1, r = this.radius * mobileScale;
    this.camera.position.set(this.target.x + r * Math.sin(this.phi) * Math.sin(this.theta), this.target.y + r * Math.cos(this.phi), this.target.z + r * Math.sin(this.phi) * Math.cos(this.theta)); this.camera.lookAt(this.target); this.camera.updateMatrixWorld();
    this.core.material.uniforms.time.value = this.time; this.core.rotation.y = this.time * .07; this.coreRings.forEach((ring, i) => ring.rotation.y = this.time * (.03 + i * .01));
    this.projectLabel(this.gatewayLabel, this.labelPosition.set(0, -1.7, 0), 17);
    for (const galaxy of this.groups.values()) {
      this.labelPosition.copy(galaxy.root.position); this.labelPosition.z -= galaxy.size + .65; this.projectLabel(galaxy.label, this.labelPosition, -10);
      for (const node of galaxy.nodes) { this.updateBeam(node); this.projectLabel(node.label, node.world, 28); node.label.style.opacity = this.selected && this.selected !== node.worker.id ? '.45' : '1'; }
    }
    if (!this.contextLost) this.renderer.render(this.scene, this.camera);
  }
  frame(now) { this.raf = requestAnimationFrame(this.frame); const dt = Math.min((now - this.lastFrame) / 1000, .06); this.lastFrame = now; if (document.hidden) return; if (!this.paused) this.time += dt; this.renderScene(dt); }
  dispose() { cancelAnimationFrame(this.raf); this.resizeObserver.disconnect(); disposeObject(this.scene); this.texture.dispose(); this.renderer.dispose(); this.container.replaceChildren(); }
}
