# Camera-based work-cell experiment design

Status: Implemented in the [workcell notebook](notebooks/vision_workcell_experiments.ipynb); local core checks pass, while Isaac/PhysX runtime verification remains. See the [animated comparison with Phase 2](docs/phase2-vs-vision-workcell.html).

## 1. Objective

Measure how job sequencing and handling mode affect on-time delivery, lateness, station switching, and picking failures when a camera-based executor handles the selected job.

## 2. Physical cell

One WOWROBO SO-ARM101 follower, two input trays, separate output areas, the supplied camera, and repeatable object presentation. The manufacturer's SO-ARM100/101 camera specification lists 2MP, 30 FPS, and a 3m USB cable; the camera-included kit lists one camera. Use one RGB image stream without simulated depth. Mount the RGB camera under the fixed gripper body after wrist_roll, following the SO101 wrist mount. Its transform changes with the arm; the supplied optical extrinsics are simulation settings requiring real hand-eye calibration. Validate reachability and visibility for every pocket before fixing tray dimensions.

## 3. Observations

Images, camera calibration, and camera poses calculated from measured joint angles drive picking. Scan occupied trays sequentially and return to the start station. Detection confidence controls the action mask. Order metadata supplies deadlines, weight, and handling risk. Ground-truth object poses and segmentation are confined to annotation, oracle controls, and evaluation. The scheduler receives 79 structured values: nine values for each of eight pockets, plus seven global values.

## 4. Actions

The scheduler chooses one of eight pockets and normal/careful handling. A shared executor receives the target pocket and performs camera-based localization, approach re-observation and correction, grasp, transfer, release, and output-view verification.

## 5. Outcomes and reward

Replace analytic timing and sampled drops in the coupled environment with elapsed time and observed outcomes. Keep the five Phase 2 score terms. Record non-drop failures and scrapped jobs as separate diagnostics; their time and completion effects still affect the score. Charge scan motion, settling and capture against the shift clock and scan station switches against the movement penalty. Store pick and scan measurements separately; clip partial scan switch costs using measured motion intervals.

## 6. Learning boundaries

Freeze perception and picking across scheduler comparisons. All scheduler policies use the same geometry-based pick executor. It calculates joint angles for the target gripper pose. Train schedulers on measured pick/scan samples and select checkpoints with camera and physics validation.

## 7. Experimental controls

Preserve familiar products, P5/P6 feature holdouts, busy arrivals, and changed damage penalties. Add oracle-versus-camera comparisons and separate appearance shifts. Use identical exogenous shift and scene seeds across schedulers.

## 8. Runtime and migration

Use LeHome's SO101 assets and camera/object patterns in the workcell notebook. Run the notebook kernel and Isaac Sim 6.0.0.1 worker with Python 3.12. Keep SimulationApp in a separate process; the notebook uses Isaac Sim Core and RTX sensor APIs. Generate the worker and shared perception module in `%%writefile` cells; both processes import that module. Use calibrated event simulation for training and coupled physics for validation and evaluation. Cache reuse depends on the explicit experiment version, settings, and measured data hashes; increment the version when changing experiment logic.

After evaluation, replay the selected PPO/DQN checkpoints on matched familiar orders and scenes. Record a workcell overview beside the policy wrist RGB stream, annotate simulation time and the selected pocket/mode, and embed the H.264 MP4 in the notebook. Store demonstration orders, checkpoint identity, settings, metrics and events separately from test results. Frame recording preserves the physics and wrist-camera render cadence; encoding failures invalidate the video without changing motion outcomes.

## 9. Agent implementation guide

Read the workcell notebook and animated comparison before implementation. Preserve target identity through asynchronous execution. Use the WOWROBO camera as a single RGB input. Enumerate its actual capture modes and calibrate intrinsics, distortion and camera-to-gripper transforms; do not infer 1920×1080 or an FOV from the 2MP label. Preserve the image aspect ratio when resizing model inputs. Declare coordinate frames, action units, failure semantics, clock policy, observation schema, feeder capacity, and train/test provenance in configuration. Do not expose privileged simulator state to the camera policy. Require camera attachment and pose calculations, moving-view localization, scan visibility and physical picking gates before training. The camera and bracket mass and collision geometry are outside the current model.
