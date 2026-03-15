#!/usr/bin/env python3
"""
Browser-based monitoring test server.

Run:  python -m monitoring.server
Open:  http://localhost:8501

If Python crashes (SIGABRT) when you click "Start exam & camera", the cause is
often MediaPipe/TFLite loading libarrow on macOS (Apple Python or Cursor env).
Workarounds:
  - Run the server in a normal Terminal (not Cursor): python3 -m monitoring.server
  - Use Homebrew Python: brew install python@3.11 && python3.11 -m venv .venv
    && source .venv/bin/activate && pip install -r requirements.txt && python -m monitoring.server
"""
from __future__ import annotations

import base64
import json
import os
import sys
import uuid as uuid_lib
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from concurrent.futures import ThreadPoolExecutor

# Optional ultralytics YOLO for phone detection. If not installed, phone detection is disabled.
try:
  from ultralytics import YOLO
except Exception:
  YOLO = None

# Do NOT import BehaviorMonitor or MonitoringConfig here. Loading them pulls in
# MediaPipe/TFLite and libarrow, which can crash (SIGABRT/mutex) on macOS with
# Apple Python or when run inside Cursor. We import them only when the first
# client actually starts monitoring.
from monitoring.models import (
    BehaviorFlag,
    EventSeverity,
    EVENT_MESSAGES,
    FLAG_SEVERITY,
    ProctoringEventType,
)

# Map proctoring event_type (mouse/keyboard) to BehaviorFlag for event_log
EVENT_TYPE_TO_FLAG = {
    "mouse_leave_window": BehaviorFlag.MOUSE_LEAVE,
    "mouse_erratic": BehaviorFlag.MOUSE_ERRATIC,
    "mouse_inactivity": BehaviorFlag.MOUSE_INACTIVE,
    "mouse_unusual_clicks": BehaviorFlag.MOUSE_UNUSUAL_CLICKS,
    "forbidden_shortcut": BehaviorFlag.FORBIDDEN_SHORTCUT,
    "copy_paste_suspected": BehaviorFlag.COPY_PASTE_SUSPECTED,
    "paste_used": BehaviorFlag.PASTE_USED,
    "keyboard_inactivity": BehaviorFlag.KEYBOARD_INACTIVE,
}

app = FastAPI(title="Monitoring Test")
_monitor = None
_monitor_error: str | None = None

# Module-level YOLO model and executor (lazy-initialized)
_PHONE_YOLO = None
_PHONE_EXECUTOR = None


def get_monitor():
    """Lazy-init: load MediaPipe/monitor only on first use so the server can start without crashing."""
    global _monitor, _monitor_error
    if _monitor is not None:
        return _monitor, None
    if _monitor_error is not None:
        return None, _monitor_error
    try:
        from monitoring.behavior_monitor import BehaviorMonitor
        from monitoring.config import MonitoringConfig
        _monitor = BehaviorMonitor(MonitoringConfig())
        return _monitor, None
    except Exception as e:
        _monitor_error = str(e)
        return None, _monitor_error


_mouse_tracker = None
_keyboard_tracker = None


def get_mouse_tracker():
    global _mouse_tracker
    if _mouse_tracker is None:
        from monitoring.mouse_tracker import MouseTracker
        from monitoring.config import MonitoringConfig
        _mouse_tracker = MouseTracker(MonitoringConfig())
    return _mouse_tracker


def get_keyboard_tracker():
    global _keyboard_tracker
    if _keyboard_tracker is None:
        from monitoring.keyboard_tracker import KeyboardTracker
        from monitoring.config import MonitoringConfig
        _keyboard_tracker = KeyboardTracker(MonitoringConfig())
    return _keyboard_tracker


def _phone_detection_worker(frame_bgr, conf_thresh=0.35):
  """Run a YOLO model on the provided BGR frame and return True if a phone-like
  object is detected with confidence >= conf_thresh. This runs in a background
  thread to avoid blocking the WebSocket loop.
  """
  global _PHONE_YOLO
  if YOLO is None:
    return False
  try:
    if _PHONE_YOLO is None:
      # lazy-load the small YOLOv8 model (downloads if needed)
      _PHONE_YOLO = YOLO('yolov8n.pt')
  except Exception:
    return False
  try:
    # convert to RGB for the model
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = _PHONE_YOLO(rgb, imgsz=320, conf=conf_thresh, verbose=False)
    if not results:
      return False
    r = results[0]
    boxes = getattr(r, 'boxes', None)
    if boxes is None:
      return False
    cls_ids = getattr(boxes, 'cls', None)
    confs = getattr(boxes, 'conf', None)
    if cls_ids is None:
      return False
    for i, cid in enumerate(cls_ids):
      try:
        idx = int(cid.item()) if hasattr(cid, 'item') else int(cid)
      except Exception:
        try:
          idx = int(cid)
        except Exception:
          continue
      name = ''
      try:
        name = _PHONE_YOLO.names.get(idx, '') if hasattr(_PHONE_YOLO, 'names') else ''
      except Exception:
        name = ''
      conf = 0.0
      try:
        conf = float(confs[i].item()) if hasattr(confs[i], 'item') else float(confs[i])
      except Exception:
        conf = 0.0
      if name and ('phone' in name.lower() or 'cell' in name.lower()):
        if conf >= conf_thresh:
          return True
    return False
  except Exception:
    return False

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Monitoring Test — Eye & Behavior Tracking</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #080b0f;
  color: #c8d8e8;
  font-family: 'Segoe UI', system-ui, sans-serif;
  display: flex; flex-direction: column; align-items: center;
  min-height: 100vh; padding: 20px;
}
h1 { font-size: 20px; color: #00d4ff; margin-bottom: 16px; letter-spacing: 1px; }
.container { display: flex; gap: 24px; flex-wrap: wrap; justify-content: center; }
.video-box {
  position: relative; border: 2px solid #1e2d3d; border-radius: 12px; overflow: hidden;
  background: #0e1318;
}
video { width: 640px; height: 480px; display: block; transform: scaleX(-1); }
.overlay {
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  pointer-events: none;
}
.panel {
  background: #111820; border: 1px solid #1e2d3d; border-radius: 12px;
  padding: 20px; min-width: 300px; max-width: 360px;
}
.panel h2 { font-size: 14px; color: #7a9ab5; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 14px; }
.metric { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1a2530; }
.metric-label { color: #7a9ab5; font-size: 13px; }
.metric-value { font-weight: 600; font-size: 13px; font-family: 'Courier New', monospace; }
.flag-list { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.flag {
  padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.5px;
}
.flag-normal { background: rgba(0,255,136,0.15); color: #00ff88; border: 1px solid rgba(0,255,136,0.3); }
.flag-warning { background: rgba(255,61,90,0.15); color: #ff3d5a; border: 1px solid rgba(255,61,90,0.3); }
.status-dot {
  display: inline-block; width: 8px; height: 8px; border-radius: 50%;
  margin-right: 6px; vertical-align: middle;
}
.dot-green { background: #00ff88; box-shadow: 0 0 6px rgba(0,255,136,0.5); }
.dot-red { background: #ff3d5a; box-shadow: 0 0 6px rgba(255,61,90,0.5); }
.dot-amber { background: #ffaa00; box-shadow: 0 0 6px rgba(255,170,0,0.5); }
.main { display: flex; gap: 20px; align-items: flex-start; flex-wrap: wrap; justify-content: center; max-width: 1400px; }
.event-sidebar {
  background: #0e1318; border: 1px solid #1e2d3d; border-radius: 12px;
  padding: 16px; min-width: 300px; max-width: 320px; max-height: 70vh; overflow: hidden; display: flex; flex-direction: column;
}
.event-sidebar h2 {
  font-size: 12px; color: #7a9ab5; text-transform: uppercase; letter-spacing: 1px;
  margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px solid #1a2530;
}
#event-log {
  flex: 1; overflow-y: auto; font-family: 'Courier New', monospace; font-size: 11px;
}
#event-log .event { padding: 8px 10px; margin-bottom: 6px; border-radius: 8px; border-left: 3px solid; }
#event-log .event.critical { background: rgba(255,61,90,0.12); border-color: #ff3d5a; color: #ff8a9e; }
#event-log .event.warning { background: rgba(255,170,0,0.12); border-color: #ffaa00; color: #ffc952; }
#event-log .event.resolved { background: rgba(0,255,136,0.08); border-color: #00ff88; color: #7dd4a8; }
#event-log .event .event-time { opacity: 0.85; font-size: 10px; margin-bottom: 2px; }
#event-log .event .event-msg { font-weight: 500; }
#event-log .event .event-badge { font-size: 9px; text-transform: uppercase; margin-top: 4px; opacity: 0.9; }
#log {
  margin-top: 16px; background: #0a0e13; border: 1px solid #1e2d3d; border-radius: 8px;
  padding: 12px; font-family: 'Courier New', monospace; font-size: 11px;
  max-height: 120px; overflow-y: auto; color: #7a9ab5; width: 100%; max-width: 1000px;
}
#log div { padding: 2px 0; }
.log-flag { color: #ff3d5a; }
.log-ok { color: #00ff88; }
.btn {
  margin-top: 16px; padding: 10px 28px; border: 1px solid #00d4ff; background: transparent;
  color: #00d4ff; border-radius: 8px; cursor: pointer; font-size: 14px; font-weight: 600;
  transition: 0.2s;
}
.btn:hover { background: rgba(0,212,255,0.1); }
.btn.active { background: #00d4ff; color: #080b0f; }
/* L1 banner */
.l1-banner {
  position: fixed; top: 0; left: 0; right: 0; z-index: 40;
  background: rgba(255,170,0,0.95); color: #1a1a1a; padding: 10px 20px; text-align: center;
  font-weight: 600; font-size: 14px; box-shadow: 0 2px 10px rgba(0,0,0,0.2);
}
/* L2 modal */
.l2-modal {
  position: fixed; inset: 0; z-index: 50; background: rgba(0,0,0,0.6);
  display: none; align-items: center; justify-content: center;
}
.l2-modal.show { display: flex; }
.l2-box {
  background: #1a1a1a; border: 2px solid #ffaa00; border-radius: 12px; padding: 24px; max-width: 400px; text-align: center;
}
.l2-box h3 { color: #ffaa00; margin-bottom: 12px; }
.l2-box button { margin-top: 16px; padding: 10px 24px; background: #ffaa00; color: #000; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; }
/* L3 overlay */
.l3-overlay {
  position: fixed; inset: 0; z-index: 60; border: 6px solid #ff3d5a; background: rgba(0,0,0,0.5);
  display: none; align-items: center; justify-content: center;
}
.l3-overlay.show { display: flex; }
.l3-box { background: #2a0a0a; color: #ff8a9e; padding: 32px; border-radius: 12px; text-align: center; max-width: 420px; }
.l3-box h2 { color: #ff3d5a; margin-bottom: 12px; }
.l3-box button { margin-top: 20px; padding: 12px 28px; background: #ff3d5a; color: #fff; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; }
/* Student layout: exam area + webcam corner */
.student-layout { display: flex; gap: 24px; flex: 1; max-width: 1200px; width: 100%; }
.exam-area { flex: 1; background: #111820; border: 1px solid #1e2d3d; border-radius: 12px; padding: 24px; }
.exam-area h2 { font-size: 12px; color: #7a9ab5; margin-bottom: 16px; }
.webcam-corner { position: relative; width: 240px; flex-shrink: 0; }
.webcam-corner video { width: 100%; border-radius: 8px; border: 2px solid #1e2d3d; transform: scaleX(-1); }
.timer { font-family: 'Courier New', monospace; font-size: 18px; color: #00d4ff; margin-bottom: 12px; }
/* Proctor panel */
.proctor-panel { min-width: 320px; max-width: 360px; display: flex; flex-direction: column; background: #0e1318; border: 1px solid #1e2d3d; border-radius: 12px; overflow: hidden; max-height: 85vh; }
.proctor-panel h2 { font-size: 12px; color: #7a9ab5; padding: 12px; border-bottom: 1px solid #1a2530; }
.proctor-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; padding: 12px; font-size: 11px; }
.proctor-stats span { color: #7a9ab5; }
.proctor-stats .val { color: #c8d8e8; font-weight: 600; }
.risk-low { color: #00ff88; }
.risk-med { color: #ffaa00; }
.risk-high { color: #ff3d5a; }
#proctor-events { flex: 1; overflow-y: auto; padding: 8px; }
.proctor-event { padding: 10px; margin-bottom: 8px; border-radius: 8px; border-left: 4px solid; font-size: 11px; }
.proctor-event.sev1 { background: rgba(255,170,0,0.1); border-color: #ffaa00; }
.proctor-event.sev2 { background: rgba(255,140,0,0.15); border-color: #ff8c00; }
.proctor-event.sev3 { background: rgba(255,61,90,0.15); border-color: #ff3d5a; }
.proctor-event img { max-width: 100%; height: auto; border-radius: 4px; margin-top: 6px; max-height: 80px; object-fit: cover; }
.export-btn { margin: 12px; padding: 10px; background: #1e2d3d; color: #00d4ff; border: 1px solid #00d4ff; border-radius: 8px; cursor: pointer; font-weight: 600; }
.export-btn:hover { background: rgba(0,212,255,0.1); }
/* Mouse & keyboard logs */
.input-logs-panel { background: #0e1318; border: 1px solid #1e2d3d; border-radius: 12px; padding: 16px; min-width: 280px; max-width: 320px; max-height: 85vh; display: flex; flex-direction: column; }
.input-logs-panel h2 { font-size: 12px; color: #7a9ab5; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid #1a2530; }
.input-log-list { font-family: 'Courier New', monospace; font-size: 11px; overflow-y: auto; flex: 1; min-height: 120px; max-height: 200px; }
.input-log-list .entry { padding: 4px 8px; margin-bottom: 2px; border-radius: 4px; background: #0a0e13; color: #8a9aaa; border-left: 3px solid #1e2d3d; }
.input-log-list .entry.mouse-move { border-left-color: #00d4ff; }
.input-log-list .entry.mouse-click { border-left-color: #00ff88; }
.input-log-list .entry.mouse-leave { border-left-color: #ffaa00; }
.input-log-list .entry.key-down { border-left-color: #7a9ab5; }
.input-log-list .entry.key-up { border-left-color: #5a7a95; }
.input-log-list .entry .time { opacity: 0.8; font-size: 10px; margin-right: 8px; }
.input-log-list.empty { color: #5a7a95; padding: 8px; }
</style>
</head>
<body>
<div id="l1-banner" class="l1-banner" style="display:none;">⚠️ Please look at your screen</div>
<div id="l2-modal" class="l2-modal"><div class="l2-box"><h3>⚠️ Proctoring Alert</h3><p id="l2-message">Please look at your screen.</p><button onclick="dismissL2()">I understand — continue exam</button></div></div>
<div id="l3-overlay" class="l3-overlay"><div class="l3-box"><h2>🚨 Critical — Proctor Notified</h2><p id="l3-message">A critical event has been recorded.</p><button onclick="dismissL3()">Acknowledge</button></div></div>

<h1>Exam Proctoring — Real-time Monitoring</h1>
<div class="main">
<div class="student-layout">
  <div class="exam-area">
    <div class="timer">Time: <span id="exam-timer">45:00</span></div>
    <h2>Exam content</h2>
    <p style="color:#7a9ab5;">[Placeholder] Question 1: Answer the following. Keep your face visible and look at the screen.</p>
  </div>
  <div class="webcam-corner" style="position:relative;">
    <video id="video" autoplay playsinline muted style="width:100%; max-width:240px;"></video>
    <canvas id="overlay" class="overlay" width="640" height="480" style="position:absolute; top:0; left:0; width:100%; max-width:240px; height:auto; pointer-events:none;"></canvas>
    <p style="margin-top:8px; font-size:11px; color:#7a9ab5;"><span id="status-dot" class="status-dot dot-green"></span> <span id="status-text">Tracking off</span></p>
    <div style="display:flex; gap:8px; align-items:center; margin-top:8px;">
      <input id="monitor-student-id" placeholder="Student ID (for reverify)" style="padding:8px; border-radius:6px; border:1px solid #1e2d3d; background:#0e1318; color:#c8d8e8;" />
      <input id="reverify-interval" type="number" min="1" value="5" style="width:72px; padding:8px; border-radius:6px; border:1px solid #1e2d3d; background:#0e1318; color:#c8d8e8;" />
      <span style="color:#7a9ab5; font-size:11px;">min</span>
    </div>
    <button class="btn" id="startBtn" onclick="toggleStream()" style="margin-top:8px;">Start exam & camera</button>
  </div>
</div>
<div class="main">
<div class="container">
  <div class="panel">
    <h2>Live Metrics</h2>
    <div class="metric">
      <span class="metric-label">Gaze Direction</span>
      <span class="metric-value" id="m-gaze">—</span>
    </div>
    <div class="metric">
      <span class="metric-label">Horizontal Ratio</span>
      <span class="metric-value" id="m-hratio">—</span>
    </div>
    <div class="metric">
      <span class="metric-label">Vertical Ratio</span>
      <span class="metric-value" id="m-vratio">—</span>
    </div>
    <div class="metric">
      <span class="metric-label">Head Yaw</span>
      <span class="metric-value" id="m-yaw">—</span>
    </div>
    <div class="metric">
      <span class="metric-label">Head Pitch</span>
      <span class="metric-value" id="m-pitch">—</span>
    </div>
    <div class="metric">
      <span class="metric-label">Head Roll</span>
      <span class="metric-value" id="m-roll">—</span>
    </div>
    <div class="metric">
      <span class="metric-label">Faces Detected</span>
      <span class="metric-value" id="m-faces">—</span>
    </div>
    <h2 style="margin-top:18px;">Active Flags</h2>
    <div class="flag-list" id="flags">
      <span class="flag flag-normal">WAITING</span>
    </div>
    <p style="margin-top:12px; font-size:11px; color:#5a7a95;">Mouse &amp; keyboard monitored (movement, leave window, shortcuts, typing pauses)</p>
  </div>
  <div class="event-sidebar">
    <h2>Audit trail</h2>
    <div id="event-log"></div>
  </div>
  <div class="proctor-panel">
    <h2>Proctor dashboard</h2>
    <div class="proctor-stats">
      <span>L1 warnings</span><span class="val" id="stat-l1">0</span>
      <span>L2 alerts</span><span class="val" id="stat-l2">0</span>
      <span>L3 critical</span><span class="val" id="stat-l3">0</span>
      <span>Risk</span><span class="val" id="stat-risk">Low</span>
    </div>
    <div id="proctor-events"></div>
    <button class="export-btn" onclick="exportLog()">Export log (JSON)</button>
  </div>
  <div class="input-logs-panel">
    <h2>Mouse log</h2>
    <div id="mouse-log" class="input-log-list empty">No mouse activity yet. Move and click to see entries.</div>
    <h2 style="margin-top:14px;">Keyboard log</h2>
    <div id="keyboard-log" class="input-log-list empty">No key activity yet. Type to see entries.</div>
  </div>
</div>
<div id="log"></div>

<script>
const video = document.getElementById('video');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
let ws = null;
let streaming = false;
let sendInterval = null;
let proctoringEvents = [];
let timerStarted = false;
let l1DismissTimer = null;
// Periodic re-verification (client-side): every N minutes capture a probe image and POST to identity API
let reverifyIntervalId = null;
let reverifyFailures = 0;
const REVERIFY_API = 'http://localhost:8000/api/v1/verify';
// Mouse/keyboard batches sent with each frame
let mouseMovements = [];
let mouseClicks = [];
let keyEvents = [];
let lastMouseSnapshot = null;
let mouseMoveThrottle = null;
let pendingMove = null;
const MOUSE_THROTTLE_MS = 50;
const MAX_MOUSE_LOG = 80;
const MAX_KEYBOARD_LOG = 80;
let mouseLog = [];
let keyboardLog = [];
let pasteOccurred = false;
// Audio/VAD state
// (audio VAD removed in this build)

function handleAlerts(alerts) {
  alerts.forEach(a => {
    if (a.level === 1) {
      document.getElementById('l1-banner').style.display = 'block';
      document.getElementById('l1-banner').textContent = '⚠️ ' + (a.message || 'Please look at your screen');
      if (l1DismissTimer) clearTimeout(l1DismissTimer);
      l1DismissTimer = setTimeout(() => { document.getElementById('l1-banner').style.display = 'none'; }, 4000);
    } else if (a.level === 2) {
      document.getElementById('l2-message').textContent = a.message || 'Please look at your screen.';
      document.getElementById('l2-modal').classList.add('show');
    } else if (a.level === 3) {
      document.getElementById('l3-message').textContent = a.message || 'A critical event has been recorded. Proctor notified.';
      document.getElementById('l3-overlay').classList.add('show');
    }
  });
}
function dismissL2() { document.getElementById('l2-modal').classList.remove('show'); }
function dismissL3() { document.getElementById('l3-overlay').classList.remove('show'); }

function updateStatus(d) {
  const dot = document.getElementById('status-dot');
  const text = document.getElementById('status-text');
  if (!streaming) { dot.className = 'status-dot dot-amber'; text.textContent = 'Tracking off'; return; }
  const fc = d.face_count || 0;
  const lowLight = d.low_light;
  const conf = d.confidence || 0;
  if (fc === 0) { dot.className = 'status-dot dot-red'; text.textContent = 'No face'; return; }
  if (lowLight || conf < 0.85) { dot.className = 'status-dot dot-amber'; text.textContent = 'Degraded'; return; }
  dot.className = 'status-dot dot-green'; text.textContent = 'Tracking active';
}

function startExamTimer(sessionStart) {
  timerStarted = true;
  const totalSec = 45 * 60;
  function tick() {
    const elapsed = Math.floor(Date.now() / 1000 - sessionStart);
    const left = Math.max(0, totalSec - elapsed);
    const m = Math.floor(left / 60);
    const s = left % 60;
    document.getElementById('exam-timer').textContent = m + ':' + (s < 10 ? '0' : '') + s;
    if (streaming) requestAnimationFrame(tick);
  }
  tick();
}

function refreshProctorPanel(events) {
  const el = document.getElementById('proctor-events');
  el.innerHTML = '';
  (events || []).slice().reverse().forEach(ev => {
    const div = document.createElement('div');
    div.className = 'proctor-event sev' + ev.severity;
    let html = '<div><strong>' + (ev.timestamp ? new Date(ev.timestamp).toLocaleTimeString() : '') + '</strong> L' + ev.severity + ' · ' + (ev.event_type || '').replace(/_/g, ' ') + '</div><div style="color:#7a9ab5; margin-top:4px;">' + (ev.message || '') + '</div>';
    if (ev.screenshot_ref) html += '<img src="data:image/jpeg;base64,' + ev.screenshot_ref + '" alt="" />';
    div.innerHTML = html;
    el.appendChild(div);
  });
  const l1 = (events || []).filter(e => e.severity === 1).length;
  const l2 = (events || []).filter(e => e.severity === 2).length;
  const l3 = (events || []).filter(e => e.severity === 3).length;
  document.getElementById('stat-l1').textContent = l1;
  document.getElementById('stat-l2').textContent = l2;
  document.getElementById('stat-l3').textContent = l3;
  let risk = 'Low';
  if (l3 > 0 || l2 >= 2) risk = 'High';
  else if (l2 >= 1 || l1 >= 3) risk = 'Medium';
  const riskEl = document.getElementById('stat-risk');
  riskEl.textContent = risk;
  riskEl.className = 'val risk-' + risk.toLowerCase();
}

function exportLog() {
  const data = JSON.stringify(proctoringEvents, null, 2);
  const blob = new Blob([data], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'proctoring-events-' + new Date().toISOString().slice(0, 10) + '.json';
  a.click();
  URL.revokeObjectURL(a.href);
}

async function toggleStream() {
  const btn = document.getElementById('startBtn');
  if (streaming) {
    stopStream();
    btn.textContent = 'Start Monitoring';
    btn.classList.remove('active');
    return;
  }
  try {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
    video.srcObject = stream;
    await video.play();
    btn.textContent = 'Stop Monitoring';
    btn.classList.add('active');
    streaming = true;
    connectWS();
  } catch (e) {
    addLog('Camera error: ' + e.message, true);
  }
}

function stopStream() {
  streaming = false;
  if (sendInterval) { clearInterval(sendInterval); sendInterval = null; }
  stopReverifyTimer();
  if (ws) { ws.close(); ws = null; }
  const stream = video.srcObject;
  if (stream) stream.getTracks().forEach(t => t.stop());
  video.srcObject = null;
  ctx.clearRect(0, 0, 640, 480);
  // no audio teardown (audio VAD removed)
}

function connectWS() {
  ws = new WebSocket('ws://' + location.host + '/ws/monitor');
  ws.onopen = () => {
    addLog('Connected to monitoring server', false);
    // Send frames at ~5 fps
    sendInterval = setInterval(sendFrame, 200);
    // Start periodic re-verification if configured
    startReverifyTimerFromInputs();
  };
  ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.error) {
      addLog(data.error + (data.detail ? ': ' + data.detail : ''), true);
      if (data.hint) addLog(data.hint, false);
      return;
    }
    updateMetrics(data);
    drawOverlay(data);
    appendNewEvents(data.new_events);
    refreshEventLog(data.event_log);
    if (data.alerts && data.alerts.length) handleAlerts(data.alerts);
    if (data.proctoring_events) { proctoringEvents = data.proctoring_events; refreshProctorPanel(data.proctoring_events); }
    if (data.session_start && !timerStarted) startExamTimer(data.session_start);
    updateStatus(data);
  };
  ws.onclose = () => { addLog('WebSocket closed', true); };
  ws.onerror = () => { addLog('WebSocket error', true); };
}

function sendFrame() {
  if (!ws || ws.readyState !== WebSocket.OPEN || !streaming) return;
  const c = document.createElement('canvas');
  c.width = 640; c.height = 480;
  c.getContext('2d').drawImage(video, 0, 0, 640, 480);
  const dataUrl = c.toDataURL('image/jpeg', 0.7);
  const base64 = dataUrl.split(',')[1];
  // Append to display logs before clearing
  const now = Date.now() / 1000;
  mouseMovements.forEach(function(m) {
    mouseLog.push({ t: m.t, type: 'move', x: m.x, y: m.y, inside: m.inside });
  });
  mouseClicks.forEach(function(c) {
    mouseLog.push({ t: c.t, type: 'click', button: c.button, x: c.x, y: c.y });
  });
  if (lastMouseSnapshot && lastMouseSnapshot.inside_window === false) {
    if (!mouseLog.length || mouseLog[mouseLog.length - 1].type !== 'leave') {
      mouseLog.push({ t: lastMouseSnapshot.timestamp, type: 'leave' });
    }
  }
  keyEvents.forEach(function(k) {
    keyboardLog.push({ t: k.timestamp, key: k.key, code: k.code, keydown: k.keydown, mods: [k.ctrl && 'Ctrl', k.alt && 'Alt', k.meta && 'Meta', k.shift && 'Shift'].filter(Boolean).join('+') });
  });
  while (mouseLog.length > MAX_MOUSE_LOG) mouseLog.shift();
  while (keyboardLog.length > MAX_KEYBOARD_LOG) keyboardLog.shift();
  refreshInputLogs();

  const payload = {
    frame: base64,
    mouse: (mouseMovements.length || mouseClicks.length || lastMouseSnapshot) ? {
      movements: mouseMovements.slice(),
      clicks: mouseClicks.slice(),
      last_snapshot: lastMouseSnapshot
    } : null,
    keys: keyEvents.length ? keyEvents.slice() : null,
    paste_event: pasteOccurred,
    // no audio speaking flag (audio VAD removed)
  };
  pasteOccurred = false;
  ws.send(JSON.stringify(payload));
  mouseMovements = [];
  mouseClicks = [];
  keyEvents = [];
}

// --- Periodic re-verification functions ---------------------------------
function startReverifyTimerFromInputs() {
  const studentId = (document.getElementById('monitor-student-id') || {}).value || '';
  const mins = parseInt((document.getElementById('reverify-interval') || {}).value || '5', 10);
  if (!studentId || !mins || mins <= 0) return;
  startReverifyTimer(studentId, mins);
}

function startReverifyTimer(studentId, minutes) {
  stopReverifyTimer();
  // run immediately then every minutes
  doReverify(studentId);
  reverifyIntervalId = setInterval(() => doReverify(studentId), minutes * 60 * 1000);
  addLog('Started periodic re-verification every ' + minutes + ' min for ' + studentId, false);
}

function stopReverifyTimer() {
  if (reverifyIntervalId) { clearInterval(reverifyIntervalId); reverifyIntervalId = null; addLog('Stopped periodic re-verification', false); }
  reverifyFailures = 0;
}

function doReverify(studentId) {
  if (!streaming || !studentId) return;
  try {
    const c = document.createElement('canvas');
    c.width = 640; c.height = 480;
    c.getContext('2d').drawImage(video, 0, 0, 640, 480);
    c.toBlob(async function(blob) {
      if (!blob) return;
      const form = new FormData();
      form.append('student_id', studentId);
      form.append('photo', blob, 'probe.jpg');
      try {
        const res = await fetch(REVERIFY_API, { method: 'POST', body: form });
        const json = await res.json();
        handleReverifyResult(json);
      } catch (err) {
        addLog('Reverify request failed: ' + err.message, true);
      }
    }, 'image/jpeg', 0.7);
  } catch (err) {
    addLog('Reverify capture failed: ' + err.message, true);
  }
}

function handleReverifyResult(json) {
  if (!json) return;
  if (json.success) {
    reverifyFailures = 0;
    // If we had prior failures, resolve them in the UI
    if (reverifyFailures > 0) {
      addLog('Reverify recovered for ' + (json.student_id || '') + ' — identity confirmed', false);
      handleAlerts([{ level: 1, message: 'Identity re-verified — tracking OK' }]);
    } else {
      addLog('Reverify OK (' + json.student_id + ') conf=' + (json.confidence || 0).toFixed(3), false);
    }
    return;
  }
  reverifyFailures += 1;
  addLog('Reverify FAILED for ' + (json.student_id || '') + ' · ' + (json.message || 'no message'), true);
  // Treat any single reverify failure as a potential identity change and show L2 modal
  if (reverifyFailures === 1) {
    handleAlerts([{ level: 2, message: 'Identity mismatch detected — please re-align and look at the camera' }]);
  } else if (reverifyFailures === 2) {
    handleAlerts([{ level: 2, message: 'Repeated identity mismatch — please re-authenticate' }]);
  } else {
    handleAlerts([{ level: 3, message: 'Multiple identity mismatches — proctor alerted' }]);
  }
}

function refreshInputLogs() {
  const mouseEl = document.getElementById('mouse-log');
  const keyEl = document.getElementById('keyboard-log');
  if (!mouseEl || !keyEl) return;
  mouseEl.classList.remove('empty');
  keyEl.classList.remove('empty');
  mouseEl.innerHTML = '';
  keyEl.innerHTML = '';
  if (mouseLog.length === 0) {
    mouseEl.classList.add('empty');
    mouseEl.textContent = 'No mouse activity yet. Move and click to see entries.';
  } else {
    mouseLog.slice().reverse().slice(0, 50).forEach(function(entry) {
      const div = document.createElement('div');
      const time = new Date(entry.t * 1000).toLocaleTimeString();
      var msg = '';
      if (entry.type === 'move') {
        div.className = 'entry mouse-move';
        msg = 'move (' + Math.round(entry.x) + ', ' + Math.round(entry.y) + ') ' + (entry.inside ? 'in' : 'out');
      } else if (entry.type === 'click') {
        div.className = 'entry mouse-click';
        var btn = entry.button === 0 ? 'left' : (entry.button === 2 ? 'right' : 'middle');
        msg = 'click ' + btn + ' (' + Math.round(entry.x) + ', ' + Math.round(entry.y) + ')';
      } else {
        div.className = 'entry mouse-leave';
        msg = 'leave window';
      }
      div.innerHTML = '<span class="time">' + time + '</span>' + msg;
      mouseEl.appendChild(div);
    });
  }
  if (keyboardLog.length === 0) {
    keyEl.classList.add('empty');
    keyEl.textContent = 'No key activity yet. Type to see entries.';
  } else {
    keyboardLog.slice().reverse().slice(0, 50).forEach(function(entry) {
      const div = document.createElement('div');
      const time = new Date(entry.t * 1000).toLocaleTimeString();
      div.className = 'entry ' + (entry.keydown ? 'key-down' : 'key-up');
      var keyLabel = entry.code || entry.key || '?';
      if (entry.mods) keyLabel = entry.mods + (entry.mods && keyLabel ? '+' : '') + keyLabel;
      div.innerHTML = '<span class="time">' + time + '</span>' + (entry.keydown ? 'keydown' : 'keyup') + ' ' + keyLabel;
      keyEl.appendChild(div);
    });
  }
}

function recordMouseMove(x, y, inside) {
  const t = Date.now() / 1000;
  mouseMovements.push({ t, x, y, inside });
  lastMouseSnapshot = { timestamp: t, x, y, inside_window: inside, buttons: 0 };
}
function recordMouseLeave() {
  lastMouseSnapshot = lastMouseSnapshot || {};
  lastMouseSnapshot.inside_window = false;
  lastMouseSnapshot.timestamp = Date.now() / 1000;
}
function recordClick(button, x, y) {
  mouseClicks.push({ t: Date.now() / 1000, button, x, y });
}
function recordKey(ev, keydown) {
  keyEvents.push({
    timestamp: Date.now() / 1000,
    key: ev.key || '',
    code: ev.code || '',
    keydown,
    ctrl: ev.ctrlKey || false,
    alt: ev.altKey || false,
    meta: ev.metaKey || false,
    shift: ev.shiftKey || false
  });
}

function setupMouseKeyboardCapture() {
  document.addEventListener('mousemove', function(ev) {
    const inside = document.hasFocus() && ev.clientX >= 0 && ev.clientY >= 0 &&
      ev.clientX <= window.innerWidth && ev.clientY <= window.innerHeight;
    pendingMove = { x: ev.clientX, y: ev.clientY, inside };
    if (!mouseMoveThrottle) {
      mouseMoveThrottle = setTimeout(function() {
        mouseMoveThrottle = null;
        if (pendingMove) {
          recordMouseMove(pendingMove.x, pendingMove.y, pendingMove.inside);
          pendingMove = null;
        }
      }, MOUSE_THROTTLE_MS);
    }
  });
  document.addEventListener('mouseleave', recordMouseLeave);
  document.addEventListener('mouseout', function(ev) {
    if (ev.relatedTarget === null) recordMouseLeave();
  });
  document.addEventListener('click', function(ev) {
    recordClick(ev.button || 0, ev.clientX, ev.clientY);
  });
  document.addEventListener('keydown', function(ev) { recordKey(ev, true); });
  document.addEventListener('keyup', function(ev) { recordKey(ev, false); });
  document.addEventListener('paste', function() { pasteOccurred = true; });
}
document.addEventListener('DOMContentLoaded', setupMouseKeyboardCapture);

// audio capture and VAD disabled in this build (reverted)

function updateMetrics(d) {
  if (d.gaze) {
    const dirColor = d.gaze.direction === 'center' ? '#00ff88' : '#ffaa00';
    document.getElementById('m-gaze').innerHTML = '<span style="color:' + dirColor + '">' + d.gaze.direction.toUpperCase() + '</span>';
    document.getElementById('m-hratio').textContent = d.gaze.horizontal_ratio.toFixed(3);
    document.getElementById('m-vratio').textContent = d.gaze.vertical_ratio.toFixed(3);
  } else {
    document.getElementById('m-gaze').textContent = '—';
    document.getElementById('m-hratio').textContent = '—';
    document.getElementById('m-vratio').textContent = '—';
  }
  if (d.head_pose) {
    document.getElementById('m-yaw').textContent = d.head_pose.yaw.toFixed(1) + '°';
    document.getElementById('m-pitch').textContent = d.head_pose.pitch.toFixed(1) + '°';
    document.getElementById('m-roll').textContent = d.head_pose.roll.toFixed(1) + '°';
  }
  const fc = d.face_count || 0;
  const fcColor = fc === 1 ? '#00ff88' : (fc === 0 ? '#ff3d5a' : '#ffaa00');
  document.getElementById('m-faces').innerHTML = '<span style="color:' + fcColor + '">' + fc + '</span>';

  // Flags
  const flagsEl = document.getElementById('flags');
  flagsEl.innerHTML = '';
  (d.flags || []).forEach(f => {
    const cls = f === 'normal' ? 'flag-normal' : 'flag-warning';
    flagsEl.innerHTML += '<span class="flag ' + cls + '">' + f.toUpperCase() + '</span>';
  });
}


function drawOverlay(d) {
  ctx.clearRect(0, 0, 640, 480);
  if (!d.gaze) return;

  // Draw gaze direction indicator
  const cx = 580, cy = 60, r = 30;
  ctx.strokeStyle = 'rgba(0,212,255,0.5)';
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();

  // Iris dot position based on ratios
  const dotX = cx + (d.gaze.horizontal_ratio - 0.5) * r * 2;
  const dotY = cy + (d.gaze.vertical_ratio - 0.5) * r * 2;
  const dotColor = d.gaze.direction === 'center' ? '#00ff88' : '#ff3d5a';
  ctx.fillStyle = dotColor;
  ctx.beginPath(); ctx.arc(dotX, dotY, 5, 0, Math.PI * 2); ctx.fill();

  // Status text
  const isOk = d.flags.length === 1 && d.flags[0] === 'normal';
  ctx.font = '14px monospace';
  ctx.fillStyle = isOk ? '#00ff88' : '#ff3d5a';
  ctx.fillText(isOk ? 'STATUS: OK' : 'STATUS: ALERT', 10, 470);
}

let lastEventLogLength = 0;
function appendNewEvents(newEvents) {
  if (!newEvents || !newEvents.length) return;
  newEvents.forEach(ev => {
    if (ev.severity !== 'resolved') addLog(ev.message || ('FLAG: ' + (ev.flag || '').toUpperCase()), true);
  });
}
function refreshEventLog(eventLog) {
  if (!eventLog) return;
  if (eventLog.length === lastEventLogLength) return;
  lastEventLogLength = eventLog.length;
  const el = document.getElementById('event-log');
  el.innerHTML = '';
  eventLog.slice().reverse().forEach(ev => addEventToSidebar(ev, el));
}
function addEventToSidebar(ev, el) {
  const time = new Date(ev.timestamp * 1000).toLocaleTimeString();
  const severity = ev.severity || 'info';
  const div = document.createElement('div');
  div.className = 'event ' + severity;
  div.innerHTML = '<div class="event-time">' + time + '</div><div class="event-msg">' + (ev.message || ev.flag) + '</div><div class="event-badge">' + (ev.flag ? ev.flag.toUpperCase().replace(/_/g, ' ') : '') + ' · ' + severity.toUpperCase() + '</div>';
  el.appendChild(div);
}

let logCount = 0;
function addLog(msg, isFlag) {
  const el = document.getElementById('log');
  const cls = isFlag ? 'log-flag' : 'log-ok';
  const ts = new Date().toLocaleTimeString();
  el.innerHTML = '<div class="' + cls + '">[' + ts + '] ' + msg + '</div>' + el.innerHTML;
  logCount++;
  if (logCount > 100) { el.removeChild(el.lastChild); }
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.get("/health")
async def health():
    """Quick health check so you can confirm the server is up before opening the app."""
    return {"status": "ok", "message": "Open http://localhost:8501 in your browser"}


def _event_entry(timestamp: float, flag: BehaviorFlag, severity: EventSeverity, message: str) -> dict:
    return {"timestamp": timestamp, "flag": flag.value, "severity": severity.value, "message": message}


def _capture_screenshot(frame) -> str | None:
    if frame is None or frame.size == 0:
        return None
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _parse_incoming(raw: str):
  """Parse incoming message: JSON with frame + optional mouse/keys/paste_event + student_id,
  or legacy base64-only string (frame only).

  Returns: frame_b64, mouse, keys, paste_event, student_id
  """
  raw = raw.strip()
  if raw.startswith("{"):
    try:
      obj = json.loads(raw)
      frame_b64 = obj.get("frame")
      mouse = obj.get("mouse")
      keys = obj.get("keys")
      paste_event = bool(obj.get("paste_event"))
      student_id = obj.get("student_id") if "student_id" in obj else None
      return frame_b64, mouse, keys, paste_event, student_id
    except json.JSONDecodeError:
      pass
  return raw, None, None, False, None


def _find_enrolled_image(student_id: str) -> str | None:
    """Return path to enrolled image for student_id if it exists (tries common extensions)."""
    try:
        from app.core.config import settings
        base = Path(settings.ENROLLED_FACES_DIR)
    except Exception:
        return None
    for ext in ('.jpg', '.jpeg', '.png', '.webp'):
        p = base / f"{student_id}{ext}"
        if p.exists():
            return str(p)
    return None


@app.websocket("/ws/monitor")
async def ws_monitor(websocket: WebSocket):
    await websocket.accept()
    event_log: list[dict] = []
    prev_flags: set[str] = set()
    max_log_entries = 500
    proctoring_events: list[dict] = []
    session_start: float | None = None
    last_l2_time: float = 0.0
    L2_COOLDOWN_S = 30.0
    # Server-side re-verification state
    reverify_last_time: float = 0.0
    reverify_failures: int = 0
    REVERIFY_INTERVAL_S: float = 5.0
    # Phone detection state (uses background YOLO worker)
    phone_future = None
    phone_detect_start: float | None = None
    phone_flagged = False
    PHONE_CHECK_INTERVAL_S = 0.5
    PHONE_DEBOUNCE_S = 2.0
    PHONE_CONF_THRESH = 0.35
    last_phone_check_time: float = 0.0
  # Audio/background-voice detection state removed (VAD reverted)

    try:
        while True:
            raw = await websocket.receive_text()
            frame_b64, mouse_data, key_events, paste_event, student_id = _parse_incoming(raw)
            if not frame_b64:
                await websocket.send_json({"error": "Missing frame data"})
                continue
            img_bytes = base64.b64decode(frame_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                await websocket.send_json({"error": "Could not decode frame"})
                continue

            frame = cv2.flip(frame, 1)
            import time as time_mod
            now = time_mod.time()
            if session_start is None:
                session_start = now

            monitor, err = get_monitor()
            if err is not None:
                await websocket.send_json({
                    "error": "Monitoring failed to start",
                    "detail": err,
                    "hint": "Try running the server in a normal Terminal (not Cursor) or use a Python from Homebrew: brew install python@3.11 && python3.11 -m venv .venv && source .venv/bin/activate",
                })
                continue
            result = monitor.process_frame(frame)
            current_flags = {f.value for f in result.flags if f != BehaviorFlag.NORMAL}
            new_events: list[dict] = []

            # Mouse and keyboard tracking (client-sent data)
            extra_alerts: list[dict] = []
            if mouse_data:
                mt = get_mouse_tracker()
                movements = mouse_data.get("movements") or []
                clicks = mouse_data.get("clicks") or []
                last_snapshot = mouse_data.get("last_snapshot")
                extra_alerts.extend(mt.process_batch(movements, clicks, last_snapshot, now))
            if key_events:
                kt = get_keyboard_tracker()
                kt_alerts, _ = kt.process_batch(key_events, now)
                extra_alerts.extend(kt_alerts)
            if paste_event:
                extra_alerts.append({
                    "level": 2,
                    "event_type": ProctoringEventType.paste_used.value,
                    "message": "Paste used (clipboard)",
                    "duration_ms": 0,
                    "confidence_score": 1.0,
                })

            for f in result.flags:
                if f == BehaviorFlag.NORMAL:
                    continue
                if f.value not in prev_flags:
                    severity = FLAG_SEVERITY.get(f, EventSeverity.WARNING)
                    msg = EVENT_MESSAGES.get(f, f.value.replace("_", " ").title())
                    entry = _event_entry(result.timestamp, f, severity, msg)
                    event_log.append(entry)
                    new_events.append(entry)
                    if len(event_log) > max_log_entries:
                        event_log.pop(0)

            for prev in prev_flags:
                if prev not in current_flags:
                    entry = _event_entry(
                        result.timestamp,
                        BehaviorFlag.NORMAL,
                        EventSeverity.RESOLVED,
                        EVENT_MESSAGES[BehaviorFlag.NORMAL],
                    )
                    event_log.append(entry)
                    new_events.append(entry)
                    if len(event_log) > max_log_entries:
                        event_log.pop(0)
                    break
            prev_flags = current_flags

            # Proctoring alerts: create ProctoringEvent, capture screenshot for L2/L3
            alerts_to_send: list[dict] = []
            # Detect multiple people (or another human) visible in the camera -> immediate L2
            try:
                if getattr(result, 'face_count', 0) and result.face_count > 1:
                    if (now - last_l2_time) >= L2_COOLDOWN_S:
                        last_l2_time = now
                        screenshot_ref = _capture_screenshot(frame)
                        pe = {
                            "id": str(uuid_lib.uuid4()),
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "event_type": "multiple_faces",
                            "severity": 2,
                            "duration_ms": 0,
                            "head_yaw": None,
                            "head_pitch": None,
                            "gaze_vector": None,
                            "confidence_score": 0.0,
                            "screenshot_ref": screenshot_ref,
                            "dismissed": False,
                            "flagged_by_proctor": False,
                            "message": f"Multiple faces detected ({result.face_count})",
                        }
                        proctoring_events.append(pe)
                        if len(proctoring_events) > 500:
                            proctoring_events.pop(0)
                        alerts_to_send.append({"level": 2, "event_type": "multiple_faces", "message": pe["message"], "id": pe["id"]})
            except Exception:
                # don't let this detection crash the WS loop
                pass
            all_alerts = list(getattr(result, "alerts", []) or []) + extra_alerts
            for al in all_alerts:
                level = al.get("level", 1)
                if level == 2 and (now - last_l2_time) < L2_COOLDOWN_S:
                    continue
                if level == 2:
                    last_l2_time = now
                screenshot_ref = _capture_screenshot(frame) if level >= 2 else None
                pe = {
                    "id": str(uuid_lib.uuid4()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "event_type": al.get("event_type", "gaze_deviation"),
                    "severity": level,
                    "duration_ms": al.get("duration_ms", 0),
                    "head_yaw": al.get("head_yaw"),
                    "head_pitch": al.get("head_pitch"),
                    "gaze_vector": al.get("gaze_vector"),
                    "confidence_score": al.get("confidence_score"),
                    "screenshot_ref": screenshot_ref,
                    "dismissed": False,
                    "flagged_by_proctor": False,
                    "message": al.get("message", ""),
                }
                proctoring_events.append(pe)
                if len(proctoring_events) > 500:
                    proctoring_events.pop(0)
                alerts_to_send.append({**al, "id": pe["id"]})

            # Audit trail for mouse/keyboard alerts
            for al in extra_alerts:
                flag = EVENT_TYPE_TO_FLAG.get(al.get("event_type"))
                if flag is not None:
                    severity = FLAG_SEVERITY.get(flag, EventSeverity.WARNING)
                    msg = al.get("message", EVENT_MESSAGES.get(flag, al.get("event_type", "")))
                    entry = _event_entry(result.timestamp, flag, severity, msg)
                    event_log.append(entry)
                    new_events.append(entry)
                    if len(event_log) > max_log_entries:
                        event_log.pop(0)

            # Server-side periodic re-verification (every REVERIFY_INTERVAL_S seconds)
            if student_id:
                try:
                    if (now - reverify_last_time) >= REVERIFY_INTERVAL_S:
                        reverify_last_time = now
                        enrolled_path = _find_enrolled_image(student_id)
                        if enrolled_path:
                            try:
                                from app.core.config import settings
                                # save probe to temp file
                                probe_path = Path(settings.UPLOAD_DIR) / f"probe_ws_{uuid_lib.uuid4().hex}.jpg"
                                cv2.imwrite(str(probe_path), frame)
                                # run DeepFace.verify
                                from deepface import DeepFace
                                v = DeepFace.verify(
                                    img1_path=str(probe_path),
                                    img2_path=str(enrolled_path),
                                    model_name=settings.DEEPFACE_MODEL,
                                    detector_backend=settings.DEEPFACE_DETECTOR,
                                    distance_metric=settings.DEEPFACE_DISTANCE_METRIC,
                                    enforce_detection=False,
                                    silent=True,
                                )
                                verified = bool(v.get("verified", False))
                                distance = float(v.get("distance", 1.0))
                                threshold = float(v.get("threshold", settings.VERIFICATION_THRESHOLD))
                                # mismatch -> create proctoring event + alert
                                if not verified:
                                    reverify_failures += 1
                                    screenshot_ref = _capture_screenshot(frame)
                                    pe = {
                                        "id": str(uuid_lib.uuid4()),
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "event_type": "identity_mismatch",
                                        "severity": 2,
                                        "duration_ms": 0,
                                        "head_yaw": None,
                                        "head_pitch": None,
                                        "gaze_vector": None,
                                        "confidence_score": 1.0 - min(max(distance / max(threshold * 2, 0.001), 0.0), 1.0),
                                        "screenshot_ref": screenshot_ref,
                                        "dismissed": False,
                                        "flagged_by_proctor": False,
                                        "message": f"Identity mismatch detected (distance={distance:.4f})",
                                    }
                                    proctoring_events.append(pe)
                                    if len(proctoring_events) > 500:
                                        proctoring_events.pop(0)
                                    alerts_to_send.append({"level": 2, "event_type": "identity_mismatch", "message": pe["message"], "id": pe["id"]})
                                else:
                                    if reverify_failures > 0:
                                        # resolved entry
                                        entry = _event_entry(now, BehaviorFlag.NORMAL, EventSeverity.RESOLVED, "Identity re-verified")
                                        event_log.append(entry)
                                        new_events.append(entry)
                                    reverify_failures = 0
                            finally:
                                try:
                                    probe_path.unlink(missing_ok=True)
                                except Exception:
                                    pass
                        else:
                            # no enrolled image found for this student
                            pass
                except Exception as exc:  # don't let verification crash the WS loop
                    # record a lightweight event
                    try:
                        entry = _event_entry(now, BehaviorFlag.NORMAL, EventSeverity.WARNING, f"Reverify error: {exc}")
                        event_log.append(entry)
                        new_events.append(entry)
                    except Exception:
                        pass

            # Phone detection (background YOLO). We check results from a background
            # future and schedule new work every PHONE_CHECK_INTERVAL_S seconds.
            try:
                global _PHONE_EXECUTOR
                if _PHONE_EXECUTOR is None:
                    _PHONE_EXECUTOR = ThreadPoolExecutor(max_workers=1)

                # collect finished future
                if phone_future is not None and phone_future.done():
                    detected = False
                    try:
                        detected = bool(phone_future.result(timeout=0))
                    except Exception:
                        detected = False
                    phone_future = None
                    if detected:
                        if phone_detect_start is None:
                            phone_detect_start = now
                        if phone_detect_start is not None and (now - phone_detect_start) >= PHONE_DEBOUNCE_S and not phone_flagged:
                            phone_flagged = True
                            screenshot_ref = _capture_screenshot(frame)
                            pe = {
                                "id": str(uuid_lib.uuid4()),
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "event_type": ProctoringEventType.possible_phone.value,
                                "severity": 2,
                                "duration_ms": 0,
                                "head_yaw": None,
                                "head_pitch": None,
                                "gaze_vector": None,
                                "confidence_score": None,
                                "screenshot_ref": screenshot_ref,
                                "dismissed": False,
                                "flagged_by_proctor": False,
                                "message": "Phone detected (sustained)",
                            }
                            proctoring_events.append(pe)
                            if len(proctoring_events) > 500:
                                proctoring_events.pop(0)
                            alerts_to_send.append({"level": 2, "event_type": ProctoringEventType.possible_phone.value, "message": pe["message"], "id": pe["id"]})
                    else:
                        phone_detect_start = None
                        phone_flagged = False

                # schedule detection work if none pending
                if phone_future is None and (now - last_phone_check_time) >= PHONE_CHECK_INTERVAL_S and YOLO is not None:
                    last_phone_check_time = now
                    try:
                        phone_future = _PHONE_EXECUTOR.submit(_phone_detection_worker, frame.copy(), PHONE_CONF_THRESH)
                    except Exception:
                        phone_future = None
            except Exception:
                # keep WS loop resilient to detection failures
                pass

      # Audio VAD logic removed (reverted to pre-audio state)

            response = {
                "face_count": result.face_count,
                "flags": [f.value for f in result.flags],
                "gaze": None,
                "head_pose": None,
                "new_events": new_events,
                "event_log": event_log[-100:],
                "alerts": alerts_to_send,
                "proctoring_events": proctoring_events[-200:],
                "confidence": getattr(result, "confidence", 0),
                "low_light": getattr(result, "low_light", False),
                "session_start": session_start,
            }
            if result.gaze:
                response["gaze"] = {
                    "direction": result.gaze.direction.value,
                    "horizontal_ratio": result.gaze.horizontal_ratio,
                    "vertical_ratio": result.gaze.vertical_ratio,
                }
            if result.head_pose:
                response["head_pose"] = {
                    "yaw": result.head_pose.yaw,
                    "pitch": result.head_pose.pitch,
                    "roll": result.head_pose.roll,
                }

            await websocket.send_json(response)

    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import sys
    port = 8501
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    print(f"Starting monitoring server at http://127.0.0.1:{port}")
    print(f"  Open in browser: http://localhost:{port}")
    print(f"  Health check:    http://localhost:{port}/health")
    uvicorn.run(app, host="0.0.0.0", port=port)
