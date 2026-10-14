/**
 * SO-ARM101 cell rendered from the real MJCF through the MuJoCo WebAssembly bindings and
 * three.js. MuJoCo compiles the model and meshes and runs forward kinematics for the joint
 * targets that cellMotion.ts derives from the replay; three.js draws every visual geom at the
 * pose MuJoCo computes. No dynamics are stepped: the trace already fixes what happened.
 */
import loadMujoco, { type MainModule, type MjData, type MjModel } from '@mujoco/mujoco';
import wasmUrl from '@mujoco/mujoco/mujoco.wasm?url';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GRIPPER_JOINT, poseFor, spot, type GripperRange, type Waypoints } from '@/lib/cell-motion';
import { cssVar } from '@/lib/svg-dom';
import type { Job, ReplayState } from '@/lib/replay';
import type { CellRenderer } from '@/lib/schematic-cell';

const MESH_FILES = [
  'base_motor_holder_so101_v1', 'base_so101_v2', 'motor_holder_so101_base_v1', 'motor_holder_so101_wrist_v1',
  'moving_jaw_so101_v1', 'rotation_pitch_so101_v1', 'sts3215_03a_no_horn_v1', 'sts3215_03a_v1',
  'under_arm_so101_v1', 'upper_arm_so101_v1', 'waveshare_mounting_plate_so101_v2', 'wrist_roll_follower_so101_v1',
  'wrist_roll_pitch_so101_v2',
];
const PRODUCT_COLOURS: Record<string, string> = { P1: '#2a78d6', P2: '#eb6834', P3: '#1baf7a', P4: '#d9a21b', P5: '#8e5bd6', P6: '#d03b3b' };
const PART = 0.02;
const COLLISION_GROUP = 3;
/** Layers of 3×3 parts an output spot holds before its oldest slots are reused. */
const OUTPUT_LAYERS = 4;

interface Layout { radius: number; station_azimuth: number; spot_offset: number; gripper_open: number; gripper_closed: number }
interface Timing { layout: Layout; waypoints: Waypoints }

/** Live typed-array views the bindings expose as `any`. */
interface ModelArrays {
  ngeom: number;
  geom_type: Int32Array; geom_dataid: Int32Array; geom_group: Int32Array; geom_matid: Int32Array; geom_rgba: Float32Array;
  mat_rgba: Float32Array;
  mesh_vertadr: Int32Array; mesh_vertnum: Int32Array; mesh_faceadr: Int32Array; mesh_facenum: Int32Array;
  mesh_vert: Float32Array; mesh_face: Int32Array;
}
interface DataArrays { qpos: Float64Array; geom_xpos: Float64Array; geom_xmat: Float64Array; site_xpos: Float64Array; site_xmat: Float64Array }

/**
 * Where a held part sits in the gripperframe site, measured from the vendored meshes: the fixed
 * finger's inner face is the plane z = 0, the moving jaw's inner face closes to z ≈ 0.023 at the
 * calibrated "closed" angle, and the finger pads run from x ≈ -0.015 to +0.008 along the approach
 * axis. A 20 mm part therefore sits centred in that gap, slightly up the pads.
 */
const HELD_OFFSET = { x: -0.0035, y: 0, z: 0.0115 };
/** Gripper angle while holding a part: the calibrated closed angle leaves a 23 mm gap, which fits the 20 mm part. */
const GRIP_HOLD = 0.1;

export class MujocoCell implements CellRenderer {
  private readonly scene = new THREE.Scene();
  private readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private readonly geomMeshes: { index: number; mesh: THREE.Mesh }[] = [];
  private readonly parts = new THREE.Group();
  private readonly held: THREE.Mesh;
  private readonly gripSite: number;
  private readonly grip: GripperRange;
  private readonly partGeometry = new THREE.BoxGeometry(PART, PART, PART);
  private readonly materials = new Map<string, THREE.MeshStandardMaterial>();
  private disposed = false;
  private readonly resize: ResizeObserver;

  private constructor(
    private readonly container: HTMLElement,
    private readonly mujoco: MainModule,
    private readonly model: MjModel,
    private readonly data: MjData,
    private readonly timing: Timing,
  ) {
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1); // MuJoCo is z-up
    this.grip = { open: timing.layout.gripper_open, closed: timing.layout.gripper_closed, hold: GRIP_HOLD };
    this.gripSite = model.site('gripperframe').id;

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, devicePixelRatio));
    container.appendChild(this.renderer.domElement);
    this.camera = new THREE.PerspectiveCamera(30, 3 / 2, 0.01, 5);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(-0.72, 0.0, 0.50);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0.05, 0, 0.03);
    this.controls.enableDamping = true;
    this.controls.maxPolarAngle = Math.PI / 2 - 0.05;
    this.controls.minDistance = 0.25;
    this.controls.maxDistance = 1.2;
    this.controls.update();

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8a80, 1.1));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(-0.3, 0.4, 0.8);
    this.scene.add(sun);

    this.buildFloorAndStations();
    this.buildGeoms();
    this.scene.add(this.parts);
    this.held = new THREE.Mesh(this.partGeometry, this.material('#2a78d6'));
    this.held.visible = false;
    this.scene.add(this.held);

    this.resize = new ResizeObserver(() => this.fit());
    this.resize.observe(container);
    this.fit();
    this.loop();
  }

  /** Loads the WASM module, the MJCF, the meshes and the calibration, then builds the view. */
  static async create(container: HTMLElement, assetBase: string, onProgress?: (msg: string) => void): Promise<MujocoCell> {
    onProgress?.('loading MuJoCo');
    const mujoco = await loadMujoco({ locateFile: (p: string) => (p.endsWith('.wasm') ? wasmUrl : p) });
    onProgress?.('loading SO-101 model and meshes');
    const text = async (f: string) => { const r = await fetch(`${assetBase}${f}`); if (!r.ok) throw new Error(`${f}: ${r.status}`); return r.text(); };
    const bytes = async (f: string) => { const r = await fetch(`${assetBase}${f}`); if (!r.ok) throw new Error(`${f}: ${r.status}`); return new Uint8Array(await r.arrayBuffer()); };
    const [scene, robot, timingText, ...meshes] = await Promise.all([
      text('scene.xml'), text('so101_new_calib.xml'), text('cell_timing.json'), ...MESH_FILES.map((m) => bytes(`assets/${m}.stl`)),
    ]);
    const FS = (mujoco as unknown as { FS: { mkdir(p: string): void; writeFile(p: string, d: string | Uint8Array): void } }).FS;
    try { FS.mkdir('/so101'); FS.mkdir('/so101/assets'); } catch { /* already there on a second load */ }
    FS.writeFile('/so101/scene.xml', scene);
    FS.writeFile('/so101/so101_new_calib.xml', robot);
    MESH_FILES.forEach((m, i) => FS.writeFile(`/so101/assets/${m}.stl`, meshes[i]));
    onProgress?.('compiling the model');
    const model = mujoco.MjModel.from_xml_path('/so101/scene.xml');
    const data = new mujoco.MjData(model);
    const timing = JSON.parse(timingText) as Timing;
    return new MujocoCell(container, mujoco, model, data, timing);
  }

  private material(hex: string, opts: Partial<THREE.MeshStandardMaterialParameters> = {}): THREE.MeshStandardMaterial {
    const key = hex + JSON.stringify(opts);
    let m = this.materials.get(key);
    if (!m) { m = new THREE.MeshStandardMaterial({ color: new THREE.Color(hex), roughness: 0.6, metalness: 0.05, ...opts }); this.materials.set(key, m); }
    return m;
  }

  private floor: THREE.Mesh | null = null;
  private grid: THREE.PolarGridHelper | null = null;
  private readonly themeQuery = matchMedia('(prefers-color-scheme: dark)');
  private readonly onTheme = (): void => this.applyTheme();

  /** Floor and grid follow the page palette, so they swap with the colour scheme like the rest of the UI. */
  private applyTheme(): void {
    const floorMat = new THREE.MeshStandardMaterial({ color: new THREE.Color(cssVar('--tray') || '#e8e7e1'), roughness: 0.95, metalness: 0 });
    if (this.floor) { (this.floor.material as THREE.Material).dispose(); this.floor.material = floorMat; }
    if (this.grid) {
      const edge = new THREE.Color(cssVar('--tray-edge') || '#b9b8ae');
      const line = new THREE.Color(cssVar('--grid') || '#d5d4cb');
      this.grid.removeFromParent();
      this.grid.dispose();
      this.grid = new THREE.PolarGridHelper(0.45, 12, 6, 48, edge, line);
      this.grid.rotation.x = Math.PI / 2;
      this.scene.add(this.grid);
    }
  }

  private buildFloorAndStations(): void {
    this.floor = new THREE.Mesh(new THREE.CircleGeometry(0.45, 64), this.material('#e8e7e1', { roughness: 0.95 }));
    this.floor.position.z = -0.001;
    this.scene.add(this.floor);
    this.grid = new THREE.PolarGridHelper(0.45, 12, 6, 48, 0xb9b8ae, 0xd5d4cb);
    this.grid.rotation.x = Math.PI / 2;
    this.scene.add(this.grid);
    this.applyTheme();
    this.themeQuery.addEventListener('change', this.onTheme);
    const lay = this.timing.layout;
    for (const s of [0, 1]) {
      const [tx, ty] = spot(lay, s, 'tray', 0);
      const tray = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.062, 0.005), this.material('#c3c2b7'));
      tray.position.set(tx, ty, 0.0025);
      tray.rotation.z = Math.atan2(ty, tx);
      this.scene.add(tray);
      const [ox, oy] = spot(lay, s, 'out', 0);
      const out = new THREE.Mesh(new THREE.BoxGeometry(0.072, 0.072, 0.004), this.material('#d9d8d0'));
      out.position.set(ox, oy, 0.002);
      this.scene.add(out);
    }
  }

  private buildGeoms(): void {
    const m = this.model as unknown as ModelArrays;
    const meshType = (this.mujoco as unknown as { mjtGeom?: { mjGEOM_MESH: { value: number } } }).mjtGeom?.mjGEOM_MESH.value ?? 7;
    for (let g = 0; g < m.ngeom; g++) {
      if (m.geom_type[g] !== meshType || m.geom_group[g] === COLLISION_GROUP) continue;
      const id = m.geom_dataid[g];
      const va = m.mesh_vertadr[id], vn = m.mesh_vertnum[id], fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute('position', new THREE.BufferAttribute(m.mesh_vert.slice(va * 3, (va + vn) * 3), 3));
      geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(m.mesh_face.slice(fa * 3, (fa + fn) * 3)), 1));
      geometry.computeVertexNormals();
      const mat = m.geom_matid[g];
      const rgba = mat >= 0 ? m.mat_rgba.slice(mat * 4, mat * 4 + 4) : m.geom_rgba.slice(g * 4, g * 4 + 4);
      const colour = new THREE.Color(rgba[0], rgba[1], rgba[2]);
      // The vendored model is uniformly dark; tint the printed parts orange like the real SO-101.
      const printed = colour.r + colour.g + colour.b > 1.2 || mat < 0;
      const hex = printed ? '#eb6834' : `#${colour.getHexString()}`;
      const mesh = new THREE.Mesh(geometry, this.material(hex, { roughness: printed ? 0.7 : 0.4, metalness: printed ? 0 : 0.2 }));
      mesh.matrixAutoUpdate = false;
      this.scene.add(mesh);
      this.geomMeshes.push({ index: g, mesh });
    }
  }

  private fit(): void {
    const w = Math.max(200, this.container.clientWidth);
    const h = Math.round((w * 2) / 3);
    this.renderer.setSize(w, h, false);
    this.renderer.domElement.style.width = '100%';
    this.renderer.domElement.style.height = 'auto';
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  private loop = (): void => {
    if (this.disposed) return;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(this.loop);
  };

  update(state: ReplayState, chosenJob: number | null): void {
    const pose = poseFor(state, this.timing.waypoints, this.grip);
    const d = this.data as unknown as DataArrays;
    for (let i = 0; i < 6; i++) d.qpos[i] = pose.q[i];
    this.mujoco.mj_kinematics(this.model, this.data);

    const mat = new THREE.Matrix4();
    for (const { index: g, mesh } of this.geomMeshes) {
      const p = g * 3, r = g * 9, X = d.geom_xmat, P = d.geom_xpos;
      mat.set(X[r], X[r + 1], X[r + 2], P[p], X[r + 3], X[r + 4], X[r + 5], P[p + 1], X[r + 6], X[r + 7], X[r + 8], P[p + 2], 0, 0, 0, 1);
      mesh.matrix.copy(mat);
    }

    // waiting parts in the trays, placed parts on the output spots
    this.parts.clear();
    const lay = this.timing.layout;
    let fromPocket: [number, number, number] | null = null;
    for (const e of state.queue) {
      const s = e.job.tray;
      const k = e.pocket;
      if (k == null) continue; // every pocket taken: the part waits off-tray
      if (pose.lifted && state.inHand?.id === e.job.id) {
        fromPocket = this.pocketWorld(s, k);
        if (pose.dropped) this.addPart(this.dropSpot(s), e.job, false, true);
        continue;
      }
      this.addPart(this.pocketWorld(s, k), e.job, e.job.id === chosenJob, false);
    }
    // parts already placed on each station's output spot (the one being released counts too)
    // 3×3 per layer, stacked up to OUTPUT_LAYERS; beyond that the oldest slots are reused so the
    // part just released is always visible.
    for (const s of [0, 1]) {
      const placed = state.stats.completedByTray[s] + (pose.placed && state.armTo === s ? 1 : 0);
      const [ox, oy] = spot(lay, s, 'out', 0);
      const capacity = 9 * OUTPUT_LAYERS;
      const first = Math.max(0, placed - capacity);
      for (let i = first; i < placed; i++) {
        const slot = i % capacity, layer = Math.floor(slot / 9), k = slot % 9;
        this.addPart([ox + ((k % 3) - 1) * 0.022, oy + (Math.floor(k / 3) - 1) * 0.022, 0.004 + PART / 2 + layer * PART], null, false, false);
      }
    }

    this.held.visible = pose.carrying && state.inHand != null;
    if (this.held.visible && state.inHand) {
      // Seat the part between the fingers in the gripper frame, aligned with the gripper; while the
      // fingers close in the pocket it slides from where it lay to where it is held.
      const sp = this.gripSite * 3, sm = this.gripSite * 9;
      const X = d.site_xmat;
      const frame = new THREE.Matrix4().set(X[sm], X[sm + 1], X[sm + 2], 0, X[sm + 3], X[sm + 4], X[sm + 5], 0, X[sm + 6], X[sm + 7], X[sm + 8], 0, 0, 0, 0, 1);
      const seat = new THREE.Vector3(HELD_OFFSET.x, HELD_OFFSET.y, HELD_OFFSET.z).applyMatrix4(frame)
        .add(new THREE.Vector3(d.site_xpos[sp], d.site_xpos[sp + 1], d.site_xpos[sp + 2]));
      if (fromPocket && pose.settle < 1) seat.lerpVectors(new THREE.Vector3(...fromPocket), seat, pose.settle);
      this.held.position.copy(seat);
      this.held.quaternion.setFromRotationMatrix(frame);
      this.held.material = this.material(PRODUCT_COLOURS[state.inHand.product] ?? '#898781');
    }
  }

  /** World position of a tray pocket: a 4×2 grid on the tray plate, aligned with the station. */
  private pocketWorld(tray: number, k: number): [number, number, number] {
    const [tx, ty] = spot(this.timing.layout, tray, 'tray', 0);
    const az = Math.atan2(ty, tx);
    const u = ((k % 4) - 1.5) * 0.023, v = (Math.floor(k / 4) - 0.5) * 0.026;
    return [tx + u * Math.cos(az) - v * Math.sin(az), ty + u * Math.sin(az) + v * Math.cos(az), 0.005 + PART / 2];
  }

  private dropSpot(station: number): [number, number, number] {
    const lay = this.timing.layout;
    const [tx, ty] = spot(lay, station, 'tray', 0);
    const [ox, oy] = spot(lay, station, 'out', 0);
    return [(tx + ox) / 2, (ty + oy) / 2, PART / 2];
  }

  private addPart(pos: [number, number, number], job: Job | null, chosen: boolean, dropped: boolean): void {
    const hex = dropped ? '#d03b3b' : job ? PRODUCT_COLOURS[job.product] ?? '#898781' : '#b9b8ae';
    const mesh = new THREE.Mesh(this.partGeometry, this.material(hex, chosen ? { emissive: new THREE.Color('#2a78d6'), emissiveIntensity: 0.6 } : {}));
    mesh.position.set(pos[0], pos[1], pos[2] + (chosen ? 0.004 : 0));
    if (dropped) mesh.rotation.set(0.6, 0.4, 0.2);
    this.parts.add(mesh);
  }

  dispose(): void {
    this.disposed = true;
    this.themeQuery.removeEventListener('change', this.onTheme);
    this.resize.disconnect();
    this.controls.dispose();
    this.renderer.dispose();
    this.renderer.domElement.remove();
    this.data.delete();
    this.model.delete();
  }
}

export const JOINTS = GRIPPER_JOINT + 1;
