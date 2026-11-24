(function () {
  "use strict";

  var VIEW_META = {
    monitor: {
      index: 0,
      title: "实时监测台",
      subtitle: "单目单应投影测速 · CTRV 运动学 · OBB-SSM 冲突评估"
    },
    alerts: {
      index: 1,
      title: "双端预警镜像",
      subtitle: "L1 车端超速提示与 L2/L3 车端-路侧协同避险分发镜像"
    },
    history: {
      index: 2,
      title: "历史档案与时间轴",
      subtitle: "运行摘要、车速风险曲线与结构化冲突明细检索"
    }
  };

  var SCENARIO_LABELS = {
    crosswalk_braking: "斑马线急刹",
    cyclist_vru_cutin: "骑行者切入",
    ctrv_curve_yield: "弯道 CTRV"
  };

  var PIPELINE_STAGE_INFO = {
    detect: "YOLOv8: 识别机动车、行人及非机动车骑行者，输出像素检测框与类别",
    track: "ByteTrack+HSV: 跨帧轨迹关联与遮挡碎片外观直方图缝合",
    motion: "单应 / CTRV: 地面逆透视映射、车辆锚点几何中心偏置补偿与弯道速度滤波",
    conflict: "OBB-SSM: 旋转有向包围盒分离轴测距，计算 TTC、MTTC、DRAC 与 PET",
    alert: "双端预警: L1 仅下发车端超速提示，L2/L3 同步驱动车端驾驶舱与路侧警示屏"
  };

  var RISK_LABELS = {
    0: "L0 正常通行",
    1: "L1 超速提示",
    2: "L2 注意礼让",
    3: "L3 紧急避险"
  };

  var SAMPLE_RUN_FALLBACK = "run-20261020_201655";

  var state = {
    theme: "dark",
    activeView: "monitor",
    mode: "simulation", // "simulation" | "replay" | "live"
    scenario: "crosswalk_braking",
    paused: false,
    pipelineConfig: {
      speedModel: "ctrv",
      conflictModel: "obb_ssm",
      stitching: true,
      cyclistVru: true
    },
    runs: [],
    presets: null,
    selectedRunId: null,
    historyData: null,
    historyTableTab: "conflicts",
    replay: {
      runId: null,
      framesCount: 0,
      fps: 10,
      currentIndex: 0,
      playing: false,
      timerId: null,
      useJpegFallback: false,
      data: null
    },
    liveJob: {
      jobId: null,
      status: null,
      sinceSeq: 0,
      pollTimerId: null,
      lastMaxSpeed: null,
      lastMinTtc: null,
      activeAlertsCount: 0
    },
    alertLog: [],
    alertLogFilter: "all",
    lastLogSignature: "",
    lastLogTimestamp: -1
  };

  var dom = {};
  var widgets = {
    kpiMaxSpeed: null,
    kpiMinTtc: null,
    kpiActiveAlerts: null,
    kpiFrames: null,
    pipeline: null,
    roadTwin: null,
    speedChart: null
  };

  function $(id) {
    return document.getElementById(id);
  }

  function formatNumber(val, digits, fallback) {
    if (typeof val !== "number" || !Number.isFinite(val)) {
      return fallback !== undefined ? fallback : "--";
    }
    return val.toFixed(digits);
  }

  function parseRiskLevel(raw) {
    if (typeof raw === "number" && Number.isFinite(raw)) {
      return Math.max(0, Math.min(3, Math.round(raw)));
    }
    if (typeof raw === "string") {
      var up = raw.trim().toUpperCase();
      if (up.indexOf("CRITICAL") >= 0 || up === "3" || up.indexOf("LEVEL_3") >= 0) return 3;
      if (up.indexOf("CAUTION") >= 0 || up === "2" || up.indexOf("LEVEL_2") >= 0) return 2;
      if (up.indexOf("SPEEDING") >= 0 || up === "1" || up.indexOf("LEVEL_1") >= 0) return 1;
    }
    return 0;
  }

  function riskText(level) {
    return RISK_LABELS[level] || RISK_LABELS[0];
  }

  function categoryLabel(cat) {
    if (cat === "pedestrian") return "行人";
    if (cat === "bicycle") return "自行车";
    if (cat === "motorcycle") return "电动/摩托车";
    if (cat === "bus") return "客车";
    if (cat === "truck") return "货车";
    if (cat === "car" || cat === "vehicle" || cat === "motor_vehicle") return "机动车";
    return cat || "--";
  }

  function isVruCategory(cat) {
    return cat === "pedestrian" || cat === "bicycle" || cat === "motorcycle";
  }

  function fetchJson(url, options) {
    return fetch(url, options).then(function (resp) {
      return resp.text().then(function (text) {
        var payload = null;
        if (text) {
          try {
            payload = JSON.parse(text);
          } catch (_err) {
            payload = null;
          }
        }
        if (!resp.ok) {
          var msg = (payload && (payload.error || payload.detail)) || ("HTTP " + resp.status);
          var err = new Error(msg);
          err.status = resp.status;
          err.payload = payload;
          throw err;
        }
        return payload;
      });
    });
  }

  function initDomRefs() {
    dom.sidebarNav = document.querySelector(".sidebar-nav");
    dom.navItems = Array.prototype.slice.call(document.querySelectorAll(".nav-item"));
    dom.mobileNavItems = Array.prototype.slice.call(document.querySelectorAll(".mobile-nav-item"));
    dom.views = {
      monitor: $("view-monitor"),
      alerts: $("view-alerts"),
      history: $("view-history")
    };
    dom.viewTitle = $("view-title");
    dom.viewSubtitle = $("view-subtitle");

    dom.serviceHealthDot = $("service-health-dot");
    dom.serviceHealthText = $("service-health-text");
    dom.sidebarModeText = $("sidebar-mode-text");
    dom.sidebarTargetText = $("sidebar-target-text");
    dom.themeToggleLabel = $("theme-toggle-label");
    dom.topbarThemeLabel = $("topbar-theme-label");

    dom.topbarRiskPill = $("topbar-risk-pill");
    dom.topbarRiskText = $("topbar-risk-text");

    dom.kpiMaxSpeed = $("kpi-max-speed");
    dom.kpiSpeedMeta = $("kpi-speed-meta");
    dom.kpiMinTtc = $("kpi-min-ttc");
    dom.kpiTtcMeta = $("kpi-ttc-meta");
    dom.kpiActiveAlerts = $("kpi-active-alerts");
    dom.kpiAlertsMeta = $("kpi-alerts-meta");
    dom.kpiFrames = $("kpi-frames");
    dom.kpiFramesMeta = $("kpi-frames-meta");

    dom.viewportModeBadge = $("viewport-mode-badge");
    dom.badgeSpeedModel = $("badge-speed-model");
    dom.badgeConflictModel = $("badge-conflict-model");
    dom.badgeVruModel = $("badge-vru-model");

    dom.viewportSimPanel = $("viewport-sim-panel");
    dom.viewportReplayPanel = $("viewport-replay-panel");
    dom.viewportLivePanel = $("viewport-live-panel");
    dom.simSceneName = $("sim-scene-name");
    dom.simObjectsSummary = $("sim-objects-summary");
    dom.simRiskSummary = $("sim-risk-summary");
    dom.btnReplaySample = $("btn-replay-sample");

    dom.replayVideo = $("replay-video");
    dom.replayFrameImg = $("replay-frame-img");
    dom.replayMediaCaption = $("replay-media-caption");
    dom.replayControls = $("replay-controls");
    dom.btnReplayToggle = $("btn-replay-toggle");
    dom.replayScrubber = $("replay-scrubber");
    dom.replayFrameLabel = $("replay-frame-label");
    dom.btnBackSim = $("btn-back-sim");

    dom.liveMjpegImg = $("live-mjpeg-img");
    dom.liveMediaCaption = $("live-media-caption");
    dom.liveControls = $("live-controls");
    dom.liveJobStatusText = $("live-job-status-text");
    dom.btnStopJob = $("btn-stop-job");
    dom.btnLiveToSim = $("btn-live-to-sim");

    dom.twinModeLabel = $("twin-mode-label");
    dom.scenarioSelect = $("scenario-select");
    dom.btnTwinPause = $("btn-twin-pause");
    dom.roadTwinCanvas = $("road-twin-canvas");
    dom.twinCounts = $("twin-counts");

    dom.pipelineSvg = $("pipeline-svg");
    dom.pipelineStageDetail = $("pipeline-stage-detail");
    dom.toggleStitching = $("toggle-stitching");
    dom.toggleCyclistVru = $("toggle-cyclist-vru");

    dom.driverTerminal = $("driver-terminal");
    dom.driverRiskBadge = $("driver-risk-badge");
    dom.driverAlertMessage = $("driver-alert-message");
    dom.driverAlertAdvice = $("driver-alert-advice");
    dom.driverVehId = $("driver-veh-id");
    dom.driverSpeed = $("driver-speed");
    dom.driverTtc = $("driver-ttc");
    dom.driverDist = $("driver-dist");
    dom.driverQueueList = $("driver-queue-list");

    dom.pedestrianTerminal = $("pedestrian-terminal");
    dom.pedestrianRiskBadge = $("pedestrian-risk-badge");
    dom.pedestrianAlertMessage = $("pedestrian-alert-message");
    dom.pedestrianAlertAdvice = $("pedestrian-alert-advice");
    dom.pedTargetId = $("ped-target-id");
    dom.pedVehSpeed = $("ped-veh-speed");
    dom.pedTtc = $("ped-ttc");
    dom.pedDist = $("ped-dist");
    dom.pedestrianQueueList = $("pedestrian-queue-list");

    dom.alertLogCount = $("alert-log-count");
    dom.alertLogList = $("alert-log-list");
    dom.btnClearAlertLog = $("btn-clear-alert-log");

    dom.historyRunSelect = $("history-run-select");
    dom.btnRefreshRuns = $("btn-refresh-runs");
    dom.btnHistoryReplay = $("btn-history-replay");
    dom.histFrames = $("hist-frames");
    dom.histFps = $("hist-fps");
    dom.histMaxSpeed = $("hist-max-speed");
    dom.histSpeedLimit = $("hist-speed-limit");
    dom.histCautionCount = $("hist-caution-count");
    dom.histSpeedingCount = $("hist-speeding-count");
    dom.histCriticalCount = $("hist-critical-count");
    dom.histElapsed = $("hist-elapsed");
    dom.timelineHoverInfo = $("timeline-hover-info");
    dom.historySourceLabel = $("history-source-label");
    dom.speedTimelineCanvas = $("speed-timeline-canvas");
    dom.historyTableMeta = $("history-table-meta");

    dom.conflictsTableRegion = $("conflicts-table-region");
    dom.eventsTableRegion = $("events-table-region");
    dom.speedsTableRegion = $("speeds-table-region");
    dom.conflictsTbody = $("conflicts-tbody");
    dom.eventsTbody = $("events-tbody");
    dom.speedsTbody = $("speeds-tbody");

    dom.jobDialog = $("job-dialog");
    dom.jobForm = $("job-form");
    dom.jobDialogError = $("job-dialog-error");
    dom.presetVideosList = $("preset-videos-list");
    dom.jobSource = $("job-source");
    dom.jobCalib = $("job-calib");
    dom.jobSpeedLimit = $("job-speed-limit");
    dom.jobSpeedModel = $("job-speed-model");
    dom.jobConflictModel = $("job-conflict-model");
    dom.jobMaxFrames = $("job-max-frames");
    dom.jobDemo = $("job-demo");
    dom.jobStitching = $("job-stitching");
    dom.jobCyclistVru = $("job-cyclist-vru");
    dom.jobAnchorBias = $("job-anchor-bias");
    dom.jobDropStale = $("job-drop-stale");
    dom.jobDialogSubmit = $("job-dialog-submit");

    dom.mobileNavDialog = $("mobile-nav-dialog");
  }

  function applyTheme(theme) {
    state.theme = theme === "light" ? "light" : "dark";
    if (state.theme === "light") {
      document.documentElement.setAttribute("data-theme", "light");
    } else {
      document.documentElement.setAttribute("data-theme", "dark");
    }
    try {
      window.localStorage.setItem("campusmotion_theme", state.theme);
    } catch (_err) {
      // ignore storage failure
    }
    if (dom.themeToggleLabel) {
      dom.themeToggleLabel.textContent = state.theme === "light" ? "切换深色石墨" : "切换浅色纸张";
    }
    if (dom.topbarThemeLabel) {
      dom.topbarThemeLabel.textContent = state.theme === "light" ? "深色" : "浅色";
    }
    if (widgets.roadTwin && typeof widgets.roadTwin.refreshTheme === "function") {
      widgets.roadTwin.refreshTheme();
    }
    if (widgets.speedChart && typeof widgets.speedChart.refreshTheme === "function") {
      widgets.speedChart.refreshTheme();
    }
  }

  function initTheme() {
    var saved = "dark";
    try {
      var stored = window.localStorage.getItem("campusmotion_theme");
      if (stored === "light" || stored === "dark") {
        saved = stored;
      }
    } catch (_err) {
      saved = "dark";
    }
    applyTheme(saved);
  }

  function toggleTheme() {
    applyTheme(state.theme === "light" ? "dark" : "light");
  }

  function setupDialog(dialog) {
    if (!dialog) return;
    dialog.addEventListener("close", function () {
      dialog.returnValue = "";
    });
    if (!("closedBy" in HTMLDialogElement.prototype)) {
      var pressedOutside = false;
      dialog.addEventListener("pointerdown", function (ev) {
        if (ev.target !== dialog) {
          pressedOutside = false;
          return;
        }
        var rect = dialog.getBoundingClientRect();
        pressedOutside =
          ev.clientX < rect.left ||
          ev.clientX > rect.right ||
          ev.clientY < rect.top ||
          ev.clientY > rect.bottom;
      });
      dialog.addEventListener("pointerup", function (ev) {
        if (!pressedOutside || ev.target !== dialog) {
          pressedOutside = false;
          return;
        }
        pressedOutside = false;
        var rect = dialog.getBoundingClientRect();
        var releasedOutside =
          ev.clientX < rect.left ||
          ev.clientX > rect.right ||
          ev.clientY < rect.top ||
          ev.clientY > rect.bottom;
        if (releasedOutside && dialog.open) {
          dialog.close();
        }
      });
    }
  }

  function switchView(viewId) {
    if (!VIEW_META[viewId]) return;
    state.activeView = viewId;
    var meta = VIEW_META[viewId];

    if (dom.sidebarNav) {
      dom.sidebarNav.style.setProperty("--active-index", String(meta.index));
    }

    dom.navItems.forEach(function (btn) {
      var active = btn.dataset.view === viewId;
      btn.classList.toggle("is-active", active);
      if (active) {
        btn.setAttribute("aria-current", "page");
      } else {
        btn.removeAttribute("aria-current");
      }
    });

    dom.mobileNavItems.forEach(function (btn) {
      btn.classList.toggle("is-active", btn.dataset.view === viewId);
    });

    Object.keys(dom.views).forEach(function (key) {
      dom.views[key].hidden = key !== viewId;
    });

    dom.viewTitle.textContent = meta.title;
    dom.viewSubtitle.textContent = meta.subtitle;

    if (viewId === "history") {
      if (!state.runs.length) {
        loadRunsList();
      } else if (widgets.speedChart && typeof widgets.speedChart.refreshTheme === "function") {
        window.requestAnimationFrame(function () {
          widgets.speedChart.refreshTheme();
        });
      }
    }
  }

  function createFallbackCounter(el, opts) {
    var fallback = (opts && opts.fallback) || "--";
    var decimals = opts && typeof opts.decimals === "number" ? opts.decimals : 1;
    var unit = (opts && opts.unit) || "";
    return {
      setValue: function (val) {
        if (typeof val !== "number" || !Number.isFinite(val)) {
          el.textContent = fallback;
          return;
        }
        el.textContent = val.toFixed(decimals) + (unit ? " " + unit : "");
      },
      destroy: function () {}
    };
  }

  function initWidgets() {
    var CM = window.CampusMotion || {};
    var CounterCtor = CM.RollingCounter;
    if (typeof CounterCtor === "function") {
      widgets.kpiMaxSpeed = new CounterCtor(dom.kpiMaxSpeed, { decimals: 1, unit: "km/h", fallback: "--" });
      widgets.kpiMinTtc = new CounterCtor(dom.kpiMinTtc, { decimals: 2, unit: "s", fallback: "--" });
      widgets.kpiActiveAlerts = new CounterCtor(dom.kpiActiveAlerts, { decimals: 0, unit: "条", fallback: "0 条" });
      widgets.kpiFrames = new CounterCtor(dom.kpiFrames, { decimals: 0, unit: "帧", fallback: "0 帧" });
    } else {
      widgets.kpiMaxSpeed = createFallbackCounter(dom.kpiMaxSpeed, { decimals: 1, unit: "km/h", fallback: "--" });
      widgets.kpiMinTtc = createFallbackCounter(dom.kpiMinTtc, { decimals: 2, unit: "s", fallback: "--" });
      widgets.kpiActiveAlerts = createFallbackCounter(dom.kpiActiveAlerts, { decimals: 0, unit: "条", fallback: "0 条" });
      widgets.kpiFrames = createFallbackCounter(dom.kpiFrames, { decimals: 0, unit: "帧", fallback: "0 帧" });
    }

    if (typeof CM.AnimatedSignalPipeline === "function") {
      widgets.pipeline = new CM.AnimatedSignalPipeline(dom.pipelineSvg);
      widgets.pipeline.setConfig(state.pipelineConfig);
    }

    if (typeof CM.CampusRoadTwinCanvas === "function") {
      widgets.roadTwin = new CM.CampusRoadTwinCanvas(dom.roadTwinCanvas, {
        onTelemetry: handleTwinTelemetry
      });
    }

    if (typeof CM.SpeedTimelineChart === "function") {
      widgets.speedChart = new CM.SpeedTimelineChart(dom.speedTimelineCanvas, {
        speedLimitKmh: 30,
        onHover: function (pt) {
          if (!pt) {
            dom.timelineHoverInfo.textContent = "点击曲线节点可跳转至监测台对应帧";
            return;
          }
          var ttcPart = typeof pt.ttcS === "number" ? (" · TTC " + pt.ttcS.toFixed(2) + "s") : "";
          dom.timelineHoverInfo.textContent =
            "帧 #" + pt.frameIndex + " (" + formatNumber(pt.t, 1, "0.0") + "s) · " +
            formatNumber(pt.speedKmh, 1, "0.0") + " km/h · " + riskText(pt.riskLevel || 0) + ttcPart;
        },
        onSelectFrame: function (frameIndex) {
          if (state.selectedRunId) {
            enterReplayMode(state.selectedRunId, frameIndex);
            switchView("monitor");
          }
        }
      });
    }
  }

  function updateRiskPill(riskLevel) {
    var lvl = parseRiskLevel(riskLevel);
    dom.topbarRiskPill.setAttribute("data-risk", String(lvl));
    dom.topbarRiskText.textContent = riskText(lvl);
  }

  function updateAlgorithmBadges() {
    dom.badgeSpeedModel.textContent = state.pipelineConfig.speedModel.toUpperCase();
    dom.badgeConflictModel.textContent = state.pipelineConfig.conflictModel === "obb_ssm" ? "OBB-SSM" : "CVM-CIRCLE";
    var flags = [];
    if (state.pipelineConfig.stitching) flags.push("HSV缝合");
    if (state.pipelineConfig.cyclistVru) flags.push("VRU");
    dom.badgeVruModel.textContent = flags.length ? flags.join("+") : "基础检测";
  }

  function handleTwinTelemetry(snap) {
    if (!snap || state.mode !== "simulation") return;

    widgets.kpiMaxSpeed.setValue(snap.maxSpeedKmh);
    widgets.kpiMinTtc.setValue(snap.minTtcS);
    widgets.kpiActiveAlerts.setValue(snap.activeConflicts || 0);
    widgets.kpiFrames.setValue(snap.frameCount || 0);
    dom.kpiFramesMeta.textContent = "累计时间 " + formatNumber(snap.timestampS, 1, "0.0") + " s";

    var risk = parseRiskLevel(snap.riskLevel);
    updateRiskPill(risk);

    if (widgets.pipeline) {
      widgets.pipeline.setRiskLevel(risk);
    }

    dom.simSceneName.textContent = SCENARIO_LABELS[state.scenario] || state.scenario;
    dom.simObjectsSummary.textContent =
      (snap.vehicleCount || 0) + " 车辆 / " + (snap.vruCount || 0) + " 行人与骑行者";
    dom.simRiskSummary.textContent = riskText(risk) + (typeof snap.minTtcS === "number" ? (" · TTC " + snap.minTtcS.toFixed(2) + "s") : "");
    dom.twinCounts.textContent = "车辆 " + (snap.vehicleCount || 0) + " · 行人/骑行 " + (snap.vruCount || 0);

    updateDualEndMirrors({
      riskLevel: risk,
      timestampS: snap.timestampS,
      driverAlerts: snap.driverAlerts || [],
      pedestrianAlerts: snap.pedestrianAlerts || []
    });
  }

  function updateDualEndMirrors(payload) {
    var driverAlerts = Array.isArray(payload.driverAlerts) ? payload.driverAlerts : [];
    var pedAlerts = Array.isArray(payload.pedestrianAlerts) ? payload.pedestrianAlerts : [];

    var driverRisk = driverAlerts.reduce(function (acc, a) {
      return Math.max(acc, parseRiskLevel(a.risk_level));
    }, 0);
    var pedRisk = pedAlerts.reduce(function (acc, a) {
      return Math.max(acc, parseRiskLevel(a.risk_level));
    }, 0);

    dom.driverTerminal.setAttribute("data-risk", String(driverRisk));
    dom.driverRiskBadge.setAttribute("data-risk", String(driverRisk));
    dom.driverRiskBadge.textContent = riskText(driverRisk);

    if (driverAlerts.length > 0) {
      var topD = driverAlerts[0];
      dom.driverAlertMessage.textContent = topD.message || "请注意前方交通冲突风险";
      dom.driverAlertAdvice.textContent = topD.advice || "建议减速并做好制动准备";
      dom.driverVehId.textContent = topD.veh_track_id != null ? ("#" + topD.veh_track_id) : "--";
      dom.driverSpeed.textContent = topD.veh_speed_kmh != null ? (Number(topD.veh_speed_kmh).toFixed(1) + " km/h") : "--";
      dom.driverTtc.textContent = topD.ttc_seconds != null ? (Number(topD.ttc_seconds).toFixed(2) + " s") : "--";
      dom.driverDist.textContent =
        (topD.current_distance_m != null ? Number(topD.current_distance_m).toFixed(2) + "m" : "--") +
        " / " +
        (topD.min_distance_m != null ? Number(topD.min_distance_m).toFixed(2) + "m" : "--");
      renderQueueChips(dom.driverQueueList, driverAlerts.slice(1, 4));
    } else {
      dom.driverAlertMessage.textContent = "当前道路通行平稳，无车端干预指令";
      dom.driverAlertAdvice.textContent = "保持校园限速行驶并留意斑马线行人动态";
      dom.driverVehId.textContent = "--";
      dom.driverSpeed.textContent = "--";
      dom.driverTtc.textContent = "--";
      dom.driverDist.textContent = "--";
      dom.driverQueueList.textContent = "";
    }

    dom.pedestrianTerminal.setAttribute("data-risk", String(pedRisk));
    dom.pedestrianRiskBadge.setAttribute("data-risk", String(pedRisk));
    dom.pedestrianRiskBadge.textContent = pedRisk === 0 ? "L0 静默" : riskText(pedRisk);

    if (pedAlerts.length > 0) {
      var topP = pedAlerts[0];
      dom.pedestrianAlertMessage.textContent = topP.message || "请注意来车快速逼近";
      dom.pedestrianAlertAdvice.textContent = topP.advice || "请止步观望并向路侧安全区退让";
      dom.pedTargetId.textContent =
        (topP.target_kind === "bicycle" || topP.target_kind === "motorcycle" ? "骑行者 #" : "行人 #") +
        (topP.ped_track_id != null ? topP.ped_track_id : "--");
      dom.pedVehSpeed.textContent = topP.veh_speed_kmh != null ? (Number(topP.veh_speed_kmh).toFixed(1) + " km/h") : "--";
      dom.pedTtc.textContent = topP.ttc_seconds != null ? (Number(topP.ttc_seconds).toFixed(2) + " s") : "--";
      dom.pedDist.textContent = topP.current_distance_m != null ? (Number(topP.current_distance_m).toFixed(2) + " m") : "--";
      renderQueueChips(dom.pedestrianQueueList, pedAlerts.slice(1, 4));
    } else {
      dom.pedestrianAlertMessage.textContent = "当前无路侧协同避险触发";
      dom.pedestrianAlertAdvice.textContent = "L1 单车超速仅提示驾驶员端，L2/L3 轨迹冲突时同步点亮路侧屏";
      dom.pedTargetId.textContent = "--";
      dom.pedVehSpeed.textContent = "--";
      dom.pedTtc.textContent = "--";
      dom.pedDist.textContent = "--";
      dom.pedestrianQueueList.textContent = "";
    }

    appendAlertLogBatch(driverAlerts, pedAlerts, payload.timestampS);
  }

  function renderQueueChips(container, items) {
    container.textContent = "";
    items.forEach(function (item) {
      var div = document.createElement("div");
      div.className = "queue-chip";
      div.textContent = item.message || "";
      container.appendChild(div);
    });
  }

  function appendAlertLogBatch(driverAlerts, pedAlerts, timestampS) {
    var combined = driverAlerts.concat(pedAlerts);
    if (!combined.length) return;

    var ts = typeof timestampS === "number" && Number.isFinite(timestampS) ? timestampS : 0;
    var sig = combined
      .map(function (a) {
        return [a.role || a.target || "", a.risk_level, a.veh_track_id, a.ped_track_id].join(":");
      })
      .join("|");

    if (sig === state.lastLogSignature && Math.abs(ts - state.lastLogTimestamp) < 0.75) {
      return;
    }
    state.lastLogSignature = sig;
    state.lastLogTimestamp = ts;

    combined.slice(0, 4).forEach(function (item) {
      var roleVal = item.role || item.target || "driver";
      state.alertLog.unshift({
        riskLevel: parseRiskLevel(item.risk_level),
        roleLabel: roleVal === "pedestrian" ? "路侧端" : "车端",
        timestampS: ts,
        vehId: item.veh_track_id,
        pedId: item.ped_track_id,
        speedKmh: item.veh_speed_kmh,
        ttcS: item.ttc_seconds,
        distM: item.current_distance_m,
        message: item.message || ""
      });
    });

    if (state.alertLog.length > 120) {
      state.alertLog.length = 120;
    }
    renderAlertLog();
  }

  function renderAlertLog() {
    var filter = state.alertLogFilter;
    var filtered = state.alertLog.filter(function (entry) {
      if (filter === "all") return true;
      return String(entry.riskLevel) === filter;
    });

    dom.alertLogCount.textContent = filtered.length + " 条记录";
    dom.alertLogList.textContent = "";

    if (!filtered.length) {
      var empty = document.createElement("p");
      empty.className = "empty-state";
      empty.textContent = "当前筛选条件下暂无双端预警记录";
      dom.alertLogList.appendChild(empty);
      return;
    }

    var frag = document.createDocumentFragment();
    filtered.slice(0, 60).forEach(function (entry) {
      var row = document.createElement("div");
      row.className = "alert-log-item";

      var badge = document.createElement("span");
      badge.className = "risk-pill tabular";
      badge.setAttribute("data-risk", String(entry.riskLevel));
      badge.textContent = "L" + entry.riskLevel + " · " + entry.roleLabel;

      var body = document.createElement("div");
      body.className = "alert-log-body";

      var msg = document.createElement("span");
      msg.className = "alert-log-msg";
      msg.textContent = entry.message;

      var meta = document.createElement("span");
      meta.className = "alert-log-meta mono tabular";
      var metaParts = [
        "t=" + formatNumber(entry.timestampS, 1, "0.0") + "s",
        "车辆#" + (entry.vehId != null ? entry.vehId : "--")
      ];
      if (entry.pedId != null) metaParts.push("目标#" + entry.pedId);
      if (entry.speedKmh != null) metaParts.push(formatNumber(Number(entry.speedKmh), 1, "--") + " km/h");
      if (entry.ttcS != null) metaParts.push("TTC " + formatNumber(Number(entry.ttcS), 2, "--") + "s");
      if (entry.distM != null) metaParts.push("间距 " + formatNumber(Number(entry.distM), 2, "--") + "m");
      meta.textContent = metaParts.join(" · ");

      body.appendChild(msg);
      body.appendChild(meta);
      row.appendChild(badge);
      row.appendChild(body);
      frag.appendChild(row);
    });
    dom.alertLogList.appendChild(frag);
  }

  function setSidebarTarget(label) {
    var text = label || "--";
    if (dom.sidebarTargetText) {
      dom.sidebarTargetText.textContent = text;
      dom.sidebarTargetText.title = text;
    }
  }

  function setMonitorMode(mode) {
    state.mode = mode;

    if (mode !== "replay") {
      stopReplayPlayback();
    }
    if (mode !== "live") {
      stopLiveJobPolling();
    }

    dom.viewportSimPanel.hidden = mode !== "simulation";
    dom.viewportReplayPanel.hidden = mode !== "replay";
    dom.viewportLivePanel.hidden = mode !== "live";
    dom.replayControls.hidden = mode !== "replay";
    dom.liveControls.hidden = mode !== "live";
    dom.scenarioSelect.hidden = mode !== "simulation";

    if (mode === "simulation") {
      dom.viewportModeBadge.textContent = "仿真模式";
      dom.sidebarModeText.textContent = "仿真演示";
      setSidebarTarget(SCENARIO_LABELS[state.scenario] || state.scenario);
      dom.twinModeLabel.textContent = "6.0m x 15.0m";
      dom.btnTwinPause.hidden = false;
      if (widgets.roadTwin) {
        widgets.roadTwin.setMode("simulation");
        widgets.roadTwin.setScenario(state.scenario);
        widgets.roadTwin.setPaused(state.paused);
      }
      if (widgets.pipeline) {
        widgets.pipeline.setPaused(state.paused);
      }
    } else if (mode === "replay") {
      dom.viewportModeBadge.textContent = "样本回放";
      dom.sidebarModeText.textContent = "历史回放";
      setSidebarTarget(state.replay.runId || "--");
      dom.btnTwinPause.hidden = true;
      if (widgets.roadTwin) {
        widgets.roadTwin.setMode("replay");
      }
    } else if (mode === "live") {
      dom.viewportModeBadge.textContent = "实时检测";
      dom.sidebarModeText.textContent = "实时作业";
      setSidebarTarget(state.liveJob.jobId || "--");
      dom.btnTwinPause.hidden = true;
      if (widgets.roadTwin) {
        widgets.roadTwin.setMode("replay");
      }
    }
  }

  function stopReplayPlayback() {
    state.replay.playing = false;
    if (state.replay.timerId !== null) {
      window.clearInterval(state.replay.timerId);
      state.replay.timerId = null;
    }
    if (dom.replayVideo && !dom.replayVideo.paused) {
      try {
        dom.replayVideo.pause();
      } catch (_err) {
        // ignore
      }
    }
    if (dom.btnReplayToggle) {
      dom.btnReplayToggle.textContent = "播放回放";
    }
  }

  function buildRunDataset(runId, summary, eventsRes, conflictsRes, speedsRes) {
    var conflicts = (conflictsRes && conflictsRes.conflicts) || [];
    var events = (eventsRes && eventsRes.events) || [];
    var speedRows = (speedsRes && speedsRes.rows) || [];

    var speedsByFrame = {};
    var conflictsByFrame = {};
    var minX = Infinity;
    var maxX = -Infinity;
    var minY = Infinity;
    var maxY = -Infinity;
    var maxFrameIdx = 0;

    speedRows.forEach(function (row) {
      var fIdx = parseInt(row.frame_idx, 10);
      if (!Number.isFinite(fIdx)) return;
      if (fIdx > maxFrameIdx) maxFrameIdx = fIdx;
      if (!speedsByFrame[fIdx]) speedsByFrame[fIdx] = [];
      speedsByFrame[fIdx].push(row);

      var x = parseFloat(row.x_m);
      var y = parseFloat(row.y_m);
      if (Number.isFinite(x)) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
      }
      if (Number.isFinite(y)) {
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    });

    conflicts.forEach(function (rec) {
      var fIdx = typeof rec.frame_index === "number" ? rec.frame_index : parseInt(rec.frame_index, 10);
      if (!Number.isFinite(fIdx)) return;
      if (fIdx > maxFrameIdx) maxFrameIdx = fIdx;
      if (!conflictsByFrame[fIdx]) conflictsByFrame[fIdx] = [];
      conflictsByFrame[fIdx].push(rec);
    });

    if (!Number.isFinite(minX) || !Number.isFinite(maxX)) {
      minX = -3.2;
      maxX = 3.2;
    } else {
      minX = Math.floor((minX - 1.2) * 2) / 2;
      maxX = Math.ceil((maxX + 1.2) * 2) / 2;
    }
    if (!Number.isFinite(minY) || !Number.isFinite(maxY)) {
      minY = 0;
      maxY = 15.0;
    } else {
      minY = Math.max(0, Math.floor((minY - 1.8) * 2) / 2);
      maxY = Math.ceil((maxY + 2.0) * 2) / 2;
    }

    var totalFrames = Math.max(
      (summary && typeof summary.frames_processed === "number" ? summary.frames_processed : 0),
      maxFrameIdx + 1
    );

    return {
      runId: runId,
      summary: summary || {},
      events: events,
      conflicts: conflicts,
      speedRows: speedRows,
      speedsByFrame: speedsByFrame,
      conflictsByFrame: conflictsByFrame,
      totalFrames: totalFrames,
      bounds: { xMin: minX, xMax: maxX, yMin: minY, yMax: maxY }
    };
  }

  function synthesizeDualAlertsFromConflicts(conflictList, timestampS) {
    var driverAlerts = [];
    var pedAlerts = [];
    var highest = 0;

    conflictList.forEach(function (a) {
      var lvl = parseRiskLevel(a.risk_level);
      if (lvl <= 0) return;
      if (lvl > highest) highest = lvl;

      var base = {
        veh_track_id: a.veh_track_id,
        ped_track_id: a.ped_track_id,
        vehicle_kind: a.vehicle_kind || "motor_vehicle",
        target_kind: a.target_kind || "pedestrian",
        risk_level: lvl,
        veh_speed_kmh: a.veh_speed_kmh,
        ttc_seconds: a.ttc_seconds,
        min_distance_m: a.min_distance_m,
        current_distance_m: a.current_distance_m
      };

      if (lvl === 1) {
        driverAlerts.push(Object.assign({}, base, {
          role: "driver",
          message: a.risk_desc || ("超速提示：车辆 #" + a.veh_track_id + " 超过限速"),
          advice: "请立即减速慢行，控制车速在限速以内"
        }));
      } else if (lvl === 2) {
        driverAlerts.push(Object.assign({}, base, {
          role: "driver",
          message: a.risk_desc || ("注意预警：车辆 #" + a.veh_track_id + " 与目标 #" + a.ped_track_id + " 轨迹汇聚"),
          advice: "请提前备刹减速并留意行人或骑行者动态"
        }));
        if (a.ped_track_id != null) {
          pedAlerts.push(Object.assign({}, base, {
            role: "pedestrian",
            message: "路旁提示：目标 #" + a.ped_track_id + " 请注意车辆 #" + a.veh_track_id + " 正在接近",
            advice: "请停步观望并沿安全区域通行"
          }));
        }
      } else if (lvl === 3) {
        driverAlerts.push(Object.assign({}, base, {
          role: "driver",
          message: a.risk_desc || ("紧急碰撞预警：车辆 #" + a.veh_track_id + " 与目标 #" + a.ped_track_id + " 距离过近"),
          advice: "请立即紧急制动停车避让"
        }));
        if (a.ped_track_id != null) {
          pedAlerts.push(Object.assign({}, base, {
            role: "pedestrian",
            message: "紧急避险预警：目标 #" + a.ped_track_id + " 请注意车辆 #" + a.veh_track_id + " 快速逼近",
            advice: "请立即止步并向路侧安全区退让"
          }));
        }
      }
    });

    return {
      riskLevel: highest,
      timestampS: timestampS,
      driverAlerts: driverAlerts,
      pedestrianAlerts: pedAlerts
    };
  }

  function buildReplayFramePayload(dataset, frameIndex) {
    var rows = dataset.speedsByFrame[frameIndex] || [];
    var prevRows = frameIndex > 0 ? (dataset.speedsByFrame[frameIndex - 1] || []) : [];
    var prevById = {};
    prevRows.forEach(function (pr) {
      prevById[String(pr.track_id)] = pr;
    });

    var conflictRecords = dataset.conflictsByFrame[frameIndex] || [];
    var assessments = conflictRecords.map(function (r) {
      return r.assessment || r;
    });

    var timestampS = 0;
    if (rows.length && rows[0].timestamp_s !== undefined) {
      timestampS = parseFloat(rows[0].timestamp_s) || 0;
    } else if (conflictRecords.length && typeof conflictRecords[0].timestamp_s === "number") {
      timestampS = conflictRecords[0].timestamp_s;
    } else {
      timestampS = frameIndex * 0.1;
    }

    var vehicles = [];
    var vrus = [];
    var maxSpeed = 0;

    rows.forEach(function (row) {
      var tid = parseInt(row.track_id, 10);
      var x = parseFloat(row.x_m);
      var y = parseFloat(row.y_m);
      var speedKmh = parseFloat(row.speed_kmh);
      if (!Number.isFinite(speedKmh)) speedKmh = 0;
      if (speedKmh > maxSpeed) maxSpeed = speedKmh;

      var prev = prevById[String(row.track_id)];
      var vx = 0;
      var vy = 0;
      var headingDeg = 90;
      if (prev) {
        var px = parseFloat(prev.x_m);
        var py = parseFloat(prev.y_m);
        var dt = Math.max(0.03, (parseFloat(row.timestamp_s) - parseFloat(prev.timestamp_s)) || 0.1);
        if (Number.isFinite(px) && Number.isFinite(py) && Number.isFinite(x) && Number.isFinite(y)) {
          vx = (x - px) / dt;
          vy = (y - py) / dt;
          if (Math.hypot(vx, vy) > 0.08) {
            headingDeg = (Math.atan2(vy, vx) * 180) / Math.PI;
          }
        }
      }

      var riskLvl = parseRiskLevel(row.risk_level);
      var cat = row.category || "vehicle";
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;

      if (isVruCategory(cat)) {
        vrus.push({
          id: tid,
          trackId: tid,
          x: x,
          y: y,
          vx: vx,
          vy: vy,
          speedMs: speedKmh / 3.6,
          category: cat,
          riskLevel: riskLvl
        });
      } else {
        vehicles.push({
          id: tid,
          trackId: tid,
          x: x,
          y: y,
          vx: vx,
          vy: vy,
          speedKmh: speedKmh,
          headingDeg: headingDeg,
          lengthM: cat === "bus" ? 8.0 : (cat === "truck" ? 6.2 : 4.2),
          widthM: cat === "bus" || cat === "truck" ? 2.2 : 1.85,
          category: cat,
          riskLevel: riskLvl
        });
      }
    });

    var minTtc = null;
    var activeCount = 0;
    var highestRisk = 0;

    assessments.forEach(function (a) {
      var r = parseRiskLevel(a.risk_level);
      if (r > highestRisk) highestRisk = r;
      if (r > 0) activeCount += 1;
      if (typeof a.ttc_seconds === "number" && Number.isFinite(a.ttc_seconds)) {
        if (minTtc === null || a.ttc_seconds < minTtc) {
          minTtc = a.ttc_seconds;
        }
      }
      if (typeof a.veh_speed_kmh === "number" && a.veh_speed_kmh > maxSpeed) {
        maxSpeed = a.veh_speed_kmh;
      }
    });

    return {
      frameIndex: frameIndex,
      timestampS: timestampS,
      bounds: dataset.bounds,
      vehicles: vehicles,
      vrus: vrus,
      conflicts: assessments,
      label: dataset.runId + " · 帧 #" + frameIndex,
      maxSpeedKmh: maxSpeed,
      minTtcS: minTtc,
      activeConflicts: activeCount,
      riskLevel: highestRisk
    };
  }

  function renderReplayFrame(frameIndex) {
    var dataset = state.replay.data;
    if (!dataset) return;

    var maxIdx = Math.max(0, dataset.totalFrames - 1);
    var idx = Math.max(0, Math.min(maxIdx, Math.round(frameIndex)));
    state.replay.currentIndex = idx;

    var framePayload = buildReplayFramePayload(dataset, idx);

    dom.replayScrubber.max = String(maxIdx);
    dom.replayScrubber.value = String(idx);
    dom.replayFrameLabel.textContent =
      idx + " / " + maxIdx + " (" + formatNumber(framePayload.timestampS, 1, "0.0") + "s)";
    dom.replayMediaCaption.textContent =
      dataset.runId + " · 帧 #" + idx + " · " + riskText(framePayload.riskLevel);

    if (dom.replayFrameImg && (state.replay.useJpegFallback || !state.replay.playing)) {
      dom.replayFrameImg.hidden = false;
      dom.replayVideo.hidden = true;
      dom.replayFrameImg.src =
        "/api/runs/" + encodeURIComponent(dataset.runId) + "/frame?index=" + idx;
    }

    if (widgets.roadTwin) {
      widgets.roadTwin.setReplayFrame(framePayload);
    }

    widgets.kpiMaxSpeed.setValue(framePayload.maxSpeedKmh);
    widgets.kpiMinTtc.setValue(framePayload.minTtcS);
    widgets.kpiActiveAlerts.setValue(framePayload.activeConflicts);
    widgets.kpiFrames.setValue(idx + 1);
    dom.kpiFramesMeta.textContent =
      "回放总帧数 " + dataset.totalFrames + " · " + formatNumber(framePayload.timestampS, 1, "0.0") + " s";
    dom.twinCounts.textContent =
      "车辆 " + framePayload.vehicles.length + " · 行人/骑行 " + framePayload.vrus.length;

    updateRiskPill(framePayload.riskLevel);
    if (widgets.pipeline) {
      widgets.pipeline.setRiskLevel(framePayload.riskLevel);
      widgets.pipeline.setPaused(!state.replay.playing);
    }

    if (widgets.speedChart && state.selectedRunId === dataset.runId) {
      widgets.speedChart.setActiveFrame(idx);
    }

    var dualAlerts = synthesizeDualAlertsFromConflicts(framePayload.conflicts, framePayload.timestampS);
    updateDualEndMirrors(dualAlerts);
  }

  function startReplayPlayback() {
    if (!state.replay.data) return;
    stopReplayPlayback();
    state.replay.playing = true;
    dom.btnReplayToggle.textContent = "暂停回放";

    if (!state.replay.useJpegFallback && dom.replayVideo && dom.replayVideo.readyState >= 2 && dom.replayVideo.videoWidth > 0) {
      dom.replayVideo.hidden = false;
      dom.replayFrameImg.hidden = true;
      dom.replayVideo.play().catch(function () {
        state.replay.useJpegFallback = true;
        startJpegReplayLoop();
      });
      return;
    }
    startJpegReplayLoop();
  }

  function startJpegReplayLoop() {
    state.replay.useJpegFallback = true;
    dom.replayVideo.hidden = true;
    dom.replayFrameImg.hidden = false;

    state.replay.timerId = window.setInterval(function () {
      if (!state.replay.playing || !state.replay.data) return;
      if (dom.replayFrameImg && !dom.replayFrameImg.complete) return;
      var maxIdx = Math.max(0, state.replay.data.totalFrames - 1);
      var nextIdx = state.replay.currentIndex + 1;
      if (nextIdx > maxIdx) nextIdx = 0;
      renderReplayFrame(nextIdx);
    }, 180);
  }

  function enterReplayMode(runId, initialFrame) {
    setMonitorMode("replay");
    state.replay.runId = runId;
    state.replay.useJpegFallback = true;
    setSidebarTarget(runId);
    dom.replayMediaCaption.textContent = "正在加载历史样本 " + runId;

    return Promise.all([
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/summary"),
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/events"),
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/conflicts?limit=500"),
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/speeds?limit=500")
    ])
      .then(function (results) {
        var dataset = buildRunDataset(runId, results[0], results[1], results[2], results[3]);
        state.replay.data = dataset;
        state.replay.framesCount = dataset.totalFrames;
        dom.twinModeLabel.textContent =
          "x[" + dataset.bounds.xMin + "," + dataset.bounds.xMax + "] y[" + dataset.bounds.yMin + "," + dataset.bounds.yMax + "]";

        // Probe MP4 codec support; fallback immediately to JPEG frame endpoint if unsupported.
        if (dom.replayVideo) {
          dom.replayVideo.src = "/api/runs/" + encodeURIComponent(runId) + "/video";
          dom.replayVideo.onerror = function () {
            state.replay.useJpegFallback = true;
          };
        }

        renderReplayFrame(typeof initialFrame === "number" ? initialFrame : 0);
        if (typeof initialFrame !== "number") {
          startReplayPlayback();
        }
      })
      .catch(function (err) {
        dom.replayMediaCaption.textContent = "加载失败: " + err.message;
      });
  }

  function stopLiveJobPolling() {
    if (state.liveJob.pollTimerId !== null) {
      window.clearInterval(state.liveJob.pollTimerId);
      state.liveJob.pollTimerId = null;
    }
    if (dom.liveMjpegImg) {
      dom.liveMjpegImg.removeAttribute("src");
    }
  }

  function enterLiveJobMode(job) {
    if (!job || !job.job_id) return;
    stopLiveJobPolling();
    state.liveJob.jobId = job.job_id;
    state.liveJob.status = job.status || "queued";
    state.liveJob.sinceSeq = 0;
    state.liveJob.lastMaxSpeed = null;
    state.liveJob.lastMinTtc = null;
    state.liveJob.activeAlertsCount = 0;

    setMonitorMode("live");
    dom.liveMjpegImg.src = "/api/jobs/" + encodeURIComponent(job.job_id) + "/mjpeg";
    dom.liveMediaCaption.textContent = "实时流 · " + job.job_id;
    dom.liveJobStatusText.textContent = "状态: " + job.status + " · 0 帧";
    dom.btnStopJob.disabled = false;

    state.liveJob.pollTimerId = window.setInterval(pollLiveJobState, 700);
    pollLiveJobState();
  }

  function pollLiveJobState() {
    var jobId = state.liveJob.jobId;
    if (!jobId || state.mode !== "live") return;

    Promise.all([
      fetchJson("/api/jobs/" + encodeURIComponent(jobId)),
      fetchJson("/api/jobs/" + encodeURIComponent(jobId) + "/alerts?since=" + state.liveJob.sinceSeq + "&limit=100")
    ])
      .then(function (results) {
        var statusObj = results[0];
        var alertsObj = results[1];
        if (!statusObj || state.mode !== "live") return;

        state.liveJob.status = statusObj.status;
        widgets.kpiFrames.setValue(statusObj.frames_processed || 0);
        dom.kpiFramesMeta.textContent =
          "最新时间戳 " + formatNumber(statusObj.latest_timestamp_s, 1, "0.0") + " s";

        dom.liveJobStatusText.textContent =
          "作业 " + statusObj.job_id + " · " + statusObj.status + " · " + (statusObj.frames_processed || 0) + " 帧";

        if (alertsObj && Array.isArray(alertsObj.alerts) && alertsObj.alerts.length > 0) {
          alertsObj.alerts.forEach(function (batch) {
            if (typeof batch.seq === "number" && batch.seq > state.liveJob.sinceSeq) {
              state.liveJob.sinceSeq = batch.seq;
            }
            var highest = parseRiskLevel(batch.highest_risk_level);
            updateRiskPill(highest);
            if (widgets.pipeline) widgets.pipeline.setRiskLevel(highest);

            var all = (batch.driver_alerts || []).concat(batch.pedestrian_alerts || []);
            all.forEach(function (a) {
              if (typeof a.veh_speed_kmh === "number") {
                if (state.liveJob.lastMaxSpeed === null || a.veh_speed_kmh > state.liveJob.lastMaxSpeed) {
                  state.liveJob.lastMaxSpeed = a.veh_speed_kmh;
                }
              }
              if (typeof a.ttc_seconds === "number") {
                if (state.liveJob.lastMinTtc === null || a.ttc_seconds < state.liveJob.lastMinTtc) {
                  state.liveJob.lastMinTtc = a.ttc_seconds;
                }
              }
            });

            state.liveJob.activeAlertsCount = batch.alert_count || all.length;
            widgets.kpiMaxSpeed.setValue(state.liveJob.lastMaxSpeed);
            widgets.kpiMinTtc.setValue(state.liveJob.lastMinTtc);
            widgets.kpiActiveAlerts.setValue(state.liveJob.activeAlertsCount);

            updateDualEndMirrors({
              riskLevel: highest,
              timestampS: batch.timestamp_s,
              driverAlerts: batch.driver_alerts || [],
              pedestrianAlerts: batch.pedestrian_alerts || []
            });
          });
        }

        if (statusObj.status === "completed" || statusObj.status === "stopped" || statusObj.status === "failed") {
          dom.btnStopJob.disabled = true;
          if (statusObj.error) {
            dom.liveMediaCaption.textContent = "作业失败: " + statusObj.error;
          } else {
            dom.liveMediaCaption.textContent = "作业已结束 (" + statusObj.status + ")";
          }
          if (state.liveJob.pollTimerId !== null) {
            window.clearInterval(state.liveJob.pollTimerId);
            state.liveJob.pollTimerId = null;
          }
          loadRunsList();
        }
      })
      .catch(function (_err) {
        // Keep polling resiliently unless stopped
      });
  }

  function loadHealth() {
    return fetchJson("/api/health")
      .then(function (data) {
        dom.serviceHealthDot.setAttribute("data-state", "online");
        dom.serviceHealthText.textContent = "已连接 (" + (data.backend || "stdlib") + ")";
      })
      .catch(function () {
        dom.serviceHealthDot.setAttribute("data-state", "offline");
        dom.serviceHealthText.textContent = "离线仿真";
      });
  }

  function loadRunsList() {
    return fetchJson("/api/runs")
      .then(function (data) {
        var runs = (data && data.runs) || [];
        state.runs = runs;
        dom.historyRunSelect.textContent = "";

        if (!runs.length) {
          var emptyOpt = document.createElement("option");
          emptyOpt.value = "";
          emptyOpt.textContent = "暂无历史分析记录";
          dom.historyRunSelect.appendChild(emptyOpt);
          dom.btnHistoryReplay.disabled = true;
          return;
        }

        dom.btnHistoryReplay.disabled = false;
        runs.forEach(function (run) {
          var opt = document.createElement("option");
          opt.value = run.run_id;
          opt.textContent =
            run.run_id + " (" + (run.frames_processed || 0) + "帧 · 峰值 " +
            formatNumber(run.max_observed_speed_kmh, 1, "--") + " km/h)";
          dom.historyRunSelect.appendChild(opt);
        });

        var preferred = state.selectedRunId && runs.some(function (r) { return r.run_id === state.selectedRunId; })
          ? state.selectedRunId
          : runs[0].run_id;
        dom.historyRunSelect.value = preferred;
        loadHistoryRunDetail(preferred);
      })
      .catch(function () {
        // Safe offline degradation
      });
  }

  function loadHistoryRunDetail(runId) {
    if (!runId) return;
    state.selectedRunId = runId;

    return Promise.all([
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/summary"),
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/events"),
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/conflicts?limit=500"),
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/speeds?limit=500")
    ])
      .then(function (results) {
        var dataset = buildRunDataset(runId, results[0], results[1], results[2], results[3]);
        state.historyData = dataset;
        renderHistorySummary(dataset);
        renderHistoryTimeline(dataset);
        renderHistoryTables(dataset);
      })
      .catch(function (err) {
        dom.historyTableMeta.textContent = "读取记录失败: " + err.message;
      });
  }

  function renderHistorySummary(dataset) {
    var s = dataset.summary || {};
    dom.histFrames.textContent = (s.frames_processed != null ? s.frames_processed : dataset.totalFrames) + " 帧";
    dom.histFps.textContent = (s.average_fps != null ? formatNumber(s.average_fps, 1, "--") : "--") + " FPS";
    dom.histMaxSpeed.textContent = formatNumber(s.max_observed_speed_kmh, 1, "--") + " km/h";
    dom.histSpeedLimit.textContent = "限速 " + formatNumber(s.speed_limit_kmh, 1, "30.0") + " km/h";
    dom.histCautionCount.textContent = String(s.total_caution_events != null ? s.total_caution_events : 0) + " 次";
    dom.histSpeedingCount.textContent = "超速告警 " + (s.total_speeding_alerts != null ? s.total_speeding_alerts : 0) + " 次";
    dom.histCriticalCount.textContent = String(s.total_critical_conflicts != null ? s.total_critical_conflicts : 0) + " 次";
    dom.histElapsed.textContent = "耗时 " + formatNumber(s.elapsed_seconds, 2, "--") + " s";
    dom.historySourceLabel.textContent = s.source_video || dataset.runId;
  }

  function renderHistoryTimeline(dataset) {
    if (!widgets.speedChart) return;
    var byFrame = {};

    dataset.speedRows.forEach(function (row) {
      var fIdx = parseInt(row.frame_idx, 10);
      if (!Number.isFinite(fIdx)) return;
      var spd = parseFloat(row.speed_kmh) || 0;
      var t = parseFloat(row.timestamp_s) || fIdx * 0.1;
      var ttc = row.ttc_s !== "" && row.ttc_s != null ? parseFloat(row.ttc_s) : null;
      var risk = parseRiskLevel(row.risk_level);

      if (!byFrame[fIdx]) {
        byFrame[fIdx] = { frameIndex: fIdx, t: t, speedKmh: spd, ttcS: ttc, riskLevel: risk };
      } else {
        if (spd > byFrame[fIdx].speedKmh) byFrame[fIdx].speedKmh = spd;
        if (risk > byFrame[fIdx].riskLevel) byFrame[fIdx].riskLevel = risk;
        if (ttc !== null && Number.isFinite(ttc)) {
          if (byFrame[fIdx].ttcS === null || ttc < byFrame[fIdx].ttcS) {
            byFrame[fIdx].ttcS = ttc;
          }
        }
      }
    });

    dataset.conflicts.forEach(function (rec) {
      var fIdx = typeof rec.frame_index === "number" ? rec.frame_index : parseInt(rec.frame_index, 10);
      if (!Number.isFinite(fIdx)) return;
      var a = rec.assessment || rec;
      var spd = typeof a.veh_speed_kmh === "number" ? a.veh_speed_kmh : 0;
      var ttc = typeof a.ttc_seconds === "number" ? a.ttc_seconds : null;
      var risk = parseRiskLevel(a.risk_level);
      var t = typeof rec.timestamp_s === "number" ? rec.timestamp_s : fIdx * 0.1;

      if (!byFrame[fIdx]) {
        byFrame[fIdx] = { frameIndex: fIdx, t: t, speedKmh: spd, ttcS: ttc, riskLevel: risk };
      } else {
        if (spd > byFrame[fIdx].speedKmh) byFrame[fIdx].speedKmh = spd;
        if (risk > byFrame[fIdx].riskLevel) byFrame[fIdx].riskLevel = risk;
        if (ttc !== null && (byFrame[fIdx].ttcS === null || ttc < byFrame[fIdx].ttcS)) {
          byFrame[fIdx].ttcS = ttc;
        }
      }
    });

    var points = Object.keys(byFrame)
      .map(function (k) { return byFrame[k]; })
      .sort(function (a, b) { return a.frameIndex - b.frameIndex; });

    var limit = (dataset.summary && typeof dataset.summary.speed_limit_kmh === "number")
      ? dataset.summary.speed_limit_kmh
      : 30;
    widgets.speedChart.setData(points, limit);
  }

  function makeCell(tagName, text, className, scope) {
    var el = document.createElement(tagName);
    if (className) el.className = className;
    if (scope) el.setAttribute("scope", scope);
    el.textContent = text;
    return el;
  }

  function makeEmptyTableRow(message) {
    var tr = document.createElement("tr");
    tr.className = "empty-row";
    var th = document.createElement("th");
    th.setAttribute("scope", "row");
    th.textContent = "--";
    var td = document.createElement("td");
    td.colSpan = 6;
    td.textContent = message;
    tr.appendChild(th);
    tr.appendChild(td);
    return tr;
  }

  function renderHistoryTables(dataset) {
    // 1. Conflicts table
    dom.conflictsTbody.textContent = "";
    if (!dataset.conflicts.length) {
      dom.conflictsTbody.appendChild(makeEmptyTableRow("该记录暂无逐帧冲突评估数据"));
    } else {
      var cFrag = document.createDocumentFragment();
      dataset.conflicts.forEach(function (rec) {
        var a = rec.assessment || rec;
        var fIdx = rec.frame_index != null ? rec.frame_index : 0;
        var lvl = parseRiskLevel(a.risk_level);

        var tr = document.createElement("tr");
        tr.addEventListener("click", function () {
          enterReplayMode(dataset.runId, fIdx);
          switchView("monitor");
        });

        tr.appendChild(makeCell("th", "#" + fIdx + " · " + formatNumber(rec.timestamp_s, 1, "0.0") + "s", "mono tabular", "row"));

        var targetPart = a.ped_track_id != null
          ? ("车辆 #" + a.veh_track_id + " / 目标 #" + a.ped_track_id)
          : ("车辆 #" + a.veh_track_id);
        tr.appendChild(makeCell("td", targetPart, "mono"));

        var riskCell = document.createElement("td");
        var pill = document.createElement("span");
        pill.className = "risk-pill tabular";
        pill.setAttribute("data-risk", String(lvl));
        pill.textContent = riskText(lvl);
        riskCell.appendChild(pill);
        tr.appendChild(riskCell);

        tr.appendChild(makeCell("td", formatNumber(a.veh_speed_kmh, 1, "--"), "mono tabular"));

        var ttcStr = formatNumber(a.ttc_seconds, 2, "--") + "s";
        if (typeof a.pet_seconds === "number") {
          ttcStr += " / PET " + a.pet_seconds.toFixed(2) + "s";
        }
        tr.appendChild(makeCell("td", ttcStr, "mono tabular"));

        var distStr = formatNumber(a.current_distance_m, 2, "--") + " / " + formatNumber(a.min_distance_m, 2, "--") + " m";
        tr.appendChild(makeCell("td", distStr, "mono tabular"));

        tr.appendChild(makeCell("td", a.risk_desc || "--", "cell-wrap"));
        cFrag.appendChild(tr);
      });
      dom.conflictsTbody.appendChild(cFrag);
    }

    // 2. Events table
    dom.eventsTbody.textContent = "";
    if (!dataset.events.length) {
      dom.eventsTbody.appendChild(makeEmptyTableRow("该记录未生成 events.jsonl 聚合事件（可在冲突风险明细查看逐帧记录）"));
    } else {
      var eFrag = document.createDocumentFragment();
      dataset.events.forEach(function (ev) {
        var lvl = parseRiskLevel(ev.peak_risk_level);
        var tr = document.createElement("tr");
        tr.addEventListener("click", function () {
          enterReplayMode(dataset.runId, ev.start_frame || 0);
          switchView("monitor");
        });

        tr.appendChild(makeCell("th", "EV-" + ev.event_id, "mono tabular", "row"));

        var actors = (ev.kind === "speeding" ? "超速" : "冲突") +
          " · V#" + ev.veh_track_id +
          (ev.ped_track_id != null ? (" / " + categoryLabel(ev.target_kind || "pedestrian") + "#" + ev.ped_track_id) : "");
        tr.appendChild(makeCell("td", actors, "mono"));

        var spanTxt = "#" + ev.start_frame + "-" + ev.end_frame +
          " (" + formatNumber(ev.start_timestamp_s, 1, "0.0") + "-" + formatNumber(ev.end_timestamp_s, 1, "0.0") + "s)";
        tr.appendChild(makeCell("td", spanTxt, "mono tabular"));

        var riskCell = document.createElement("td");
        var pill = document.createElement("span");
        pill.className = "risk-pill tabular";
        pill.setAttribute("data-risk", String(lvl));
        pill.textContent = riskText(lvl);
        riskCell.appendChild(pill);
        tr.appendChild(riskCell);

        tr.appendChild(makeCell("td", formatNumber(ev.min_ttc_seconds, 2, "--"), "mono tabular"));

        var ssmTxt = formatNumber(ev.min_mttc_seconds, 2, "--") + "s / " + formatNumber(ev.max_drac_ms2, 2, "--") + " m/s^2";
        tr.appendChild(makeCell("td", ssmTxt, "mono tabular"));

        tr.appendChild(makeCell("td", ev.end_reason || "--", "mono"));
        eFrag.appendChild(tr);
      });
      dom.eventsTbody.appendChild(eFrag);
    }

    // 3. Speeds table
    dom.speedsTbody.textContent = "";
    if (!dataset.speedRows.length) {
      dom.speedsTbody.appendChild(makeEmptyTableRow("该记录暂无轨迹测速明细"));
    } else {
      var sFrag = document.createDocumentFragment();
      dataset.speedRows.forEach(function (row) {
        var fIdx = parseInt(row.frame_idx, 10) || 0;
        var lvl = parseRiskLevel(row.risk_level);
        var tr = document.createElement("tr");
        tr.addEventListener("click", function () {
          enterReplayMode(dataset.runId, fIdx);
          switchView("monitor");
        });

        tr.appendChild(makeCell("th", "#" + fIdx + " · T#" + (row.track_id || "--"), "mono tabular", "row"));
        tr.appendChild(makeCell("td", categoryLabel(row.category)));
        tr.appendChild(makeCell("td", formatNumber(parseFloat(row.timestamp_s), 2, "--"), "mono tabular"));

        var xyTxt = "(" + formatNumber(parseFloat(row.x_m), 2, "--") + ", " + formatNumber(parseFloat(row.y_m), 2, "--") + ")";
        tr.appendChild(makeCell("td", xyTxt, "mono tabular"));

        tr.appendChild(makeCell("td", formatNumber(parseFloat(row.speed_kmh), 1, "--"), "mono tabular"));
        tr.appendChild(makeCell("td", String(row.is_speeding) === "1" ? "超速" : "正常"));

        var rTxt = riskText(lvl) + (row.ttc_s ? (" · " + row.ttc_s + "s") : "");
        tr.appendChild(makeCell("td", rTxt, "mono tabular"));
        sFrag.appendChild(tr);
      });
      dom.speedsTbody.appendChild(sFrag);
    }

    updateHistoryTableVisibility();
  }

  function updateHistoryTableVisibility() {
    var tab = state.historyTableTab;
    dom.conflictsTableRegion.hidden = tab !== "conflicts";
    dom.eventsTableRegion.hidden = tab !== "events";
    dom.speedsTableRegion.hidden = tab !== "speeds";

    document.querySelectorAll("[data-table-tab]").forEach(function (btn) {
      var active = btn.dataset.tableTab === tab;
      btn.classList.toggle("is-active", active);
      btn.setAttribute("aria-selected", active ? "true" : "false");
    });

    if (state.historyData) {
      var count = 0;
      if (tab === "conflicts") count = state.historyData.conflicts.length;
      else if (tab === "events") count = state.historyData.events.length;
      else count = state.historyData.speedRows.length;
      dom.historyTableMeta.textContent = "共 " + count + " 条记录（点击行跳转回放）";
    }
  }

  function openJobDialog() {
    dom.jobDialogError.hidden = true;
    dom.jobDialogError.textContent = "";
    if (!dom.jobDialog.open) {
      dom.jobDialog.showModal();
    }
    loadPresets();
  }

  function loadPresets() {
    return fetchJson("/api/presets")
      .then(function (data) {
        state.presets = data;
        dom.presetVideosList.textContent = "";
        var videos = (data && data.videos) || [];
        videos.forEach(function (v) {
          var btn = document.createElement("button");
          btn.type = "button";
          btn.className = "preset-chip";
          btn.textContent = v.label || v.path;
          btn.addEventListener("click", function () {
            dom.jobSource.value = v.path;
            dom.jobSource.dispatchEvent(new Event("input", { bubbles: true }));
            dom.jobSource.focus();
          });
          dom.presetVideosList.appendChild(btn);
        });

        dom.jobCalib.textContent = "";
        var calibs = (data && data.calibs) || [];
        calibs.forEach(function (c) {
          var opt = document.createElement("option");
          opt.value = c.path;
          opt.textContent = c.label || c.path;
          if (data.default_calib && c.path === data.default_calib) {
            opt.selected = true;
          }
          dom.jobCalib.appendChild(opt);
        });
        if (!dom.jobSource.value && videos.length > 0) {
          dom.jobSource.value = videos[0].path;
        }
      })
      .catch(function () {
        // Keep form editable manually even offline
      });
  }

  function handleJobSubmit(ev) {
    ev.preventDefault();
    dom.jobDialogError.hidden = true;
    dom.jobDialogError.textContent = "";

    var source = dom.jobSource.value.trim();
    if (!source) {
      dom.jobDialogError.textContent = "请输入视频路径、RTSP 地址或摄像头编号";
      dom.jobDialogError.hidden = false;
      dom.jobSource.focus();
      return;
    }

    var speedLimit = parseFloat(dom.jobSpeedLimit.value);
    if (!Number.isFinite(speedLimit) || speedLimit <= 0) {
      dom.jobDialogError.textContent = "限速阈值必须大于 0";
      dom.jobDialogError.hidden = false;
      return;
    }

    var payload = {
      source: source,
      demo: Boolean(dom.jobDemo.checked),
      speed_limit_kmh: speedLimit,
      speed_model: dom.jobSpeedModel.value,
      conflict_model: dom.jobConflictModel.value,
      enable_stitching: Boolean(dom.jobStitching.checked),
      include_cyclist_vru: Boolean(dom.jobCyclistVru.checked),
      compensate_anchor_bias: Boolean(dom.jobAnchorBias.checked),
      live_drop_stale_frames: Boolean(dom.jobDropStale.checked)
    };

    if (dom.jobCalib.value) {
      payload.calib_path = dom.jobCalib.value;
    }

    var maxFramesRaw = dom.jobMaxFrames.value.trim();
    if (maxFramesRaw !== "") {
      var parsedMax = parseInt(maxFramesRaw, 10);
      if (!Number.isFinite(parsedMax) || parsedMax < 1) {
        dom.jobDialogError.textContent = "最大处理帧数必须是正整数";
        dom.jobDialogError.hidden = false;
        return;
      }
      payload.max_frames = parsedMax;
    }

    state.pipelineConfig.speedModel = payload.speed_model;
    state.pipelineConfig.conflictModel = payload.conflict_model;
    state.pipelineConfig.stitching = payload.enable_stitching;
    state.pipelineConfig.cyclistVru = payload.include_cyclist_vru;
    updateAlgorithmBadges();
    if (widgets.pipeline) {
      widgets.pipeline.setConfig(state.pipelineConfig);
    }

    dom.jobDialogSubmit.disabled = true;
    dom.jobDialogSubmit.textContent = "启动中...";

    fetchJson("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify(payload)
    })
      .then(function (job) {
        dom.jobDialog.close();
        switchView("monitor");
        enterLiveJobMode(job);
      })
      .catch(function (err) {
        dom.jobDialogError.textContent = "启动失败: " + err.message;
        dom.jobDialogError.hidden = false;
      })
      .finally(function () {
        dom.jobDialogSubmit.disabled = false;
        dom.jobDialogSubmit.textContent = "启动后台检测";
      });
  }

  function bindEvents() {
    dom.navItems.forEach(function (btn) {
      btn.addEventListener("click", function () {
        switchView(btn.dataset.view);
      });
    });

    dom.mobileNavItems.forEach(function (btn) {
      btn.addEventListener("click", function () {
        switchView(btn.dataset.view);
        if (dom.mobileNavDialog.open) {
          dom.mobileNavDialog.close();
        }
      });
    });

    $("mobile-nav-open-btn").addEventListener("click", function () {
      if (!dom.mobileNavDialog.open) {
        dom.mobileNavDialog.showModal();
      }
    });

    $("mobile-nav-close").addEventListener("click", function () {
      if (dom.mobileNavDialog.open) {
        dom.mobileNavDialog.close();
      }
    });

    $("sidebar-theme-btn").addEventListener("click", toggleTheme);
    $("topbar-theme-btn").addEventListener("click", toggleTheme);
    $("mobile-theme-btn").addEventListener("click", toggleTheme);

    $("open-job-dialog-btn").addEventListener("click", openJobDialog);
    $("mobile-new-job-btn").addEventListener("click", function () {
      if (dom.mobileNavDialog.open) dom.mobileNavDialog.close();
      openJobDialog();
    });

    $("job-dialog-close").addEventListener("click", function () {
      if (dom.jobDialog.open) dom.jobDialog.close();
    });
    $("job-dialog-cancel").addEventListener("click", function () {
      if (dom.jobDialog.open) dom.jobDialog.close();
    });
    dom.jobForm.addEventListener("submit", handleJobSubmit);

    dom.scenarioSelect.addEventListener("change", function () {
      state.scenario = dom.scenarioSelect.value;
      setSidebarTarget(SCENARIO_LABELS[state.scenario] || state.scenario);
      if (widgets.roadTwin && state.mode === "simulation") {
        widgets.roadTwin.setScenario(state.scenario);
      }
    });

    dom.btnTwinPause.addEventListener("click", function () {
      state.paused = !state.paused;
      dom.btnTwinPause.textContent = state.paused ? "继续演示" : "暂停演示";
      if (widgets.roadTwin) {
        widgets.roadTwin.setPaused(state.paused);
      }
      if (widgets.pipeline) {
        widgets.pipeline.setPaused(state.paused);
      }
    });

    dom.btnReplaySample.addEventListener("click", function () {
      var targetId = (state.runs.length && state.runs[0].run_id) || SAMPLE_RUN_FALLBACK;
      enterReplayMode(targetId, 0);
    });

    dom.btnBackSim.addEventListener("click", function () {
      setMonitorMode("simulation");
    });

    dom.btnLiveToSim.addEventListener("click", function () {
      setMonitorMode("simulation");
    });

    dom.btnReplayToggle.addEventListener("click", function () {
      if (state.replay.playing) {
        stopReplayPlayback();
      } else {
        startReplayPlayback();
      }
      if (widgets.pipeline) {
        widgets.pipeline.setPaused(!state.replay.playing);
      }
    });

    dom.replayScrubber.addEventListener("input", function () {
      stopReplayPlayback();
      var idx = parseInt(dom.replayScrubber.value, 10) || 0;
      renderReplayFrame(idx);
    });

    dom.btnStopJob.addEventListener("click", function () {
      var jobId = state.liveJob.jobId;
      if (!jobId) return;
      dom.btnStopJob.disabled = true;
      fetchJson("/api/jobs/" + encodeURIComponent(jobId) + "/stop", { method: "POST" })
        .then(function (job) {
          dom.liveJobStatusText.textContent = "作业 " + job.job_id + " · " + job.status;
          stopLiveJobPolling();
          loadRunsList();
        })
        .catch(function (err) {
          dom.liveMediaCaption.textContent = "停止失败: " + err.message;
        });
    });

    dom.toggleStitching.addEventListener("change", function () {
      state.pipelineConfig.stitching = Boolean(dom.toggleStitching.checked);
      updateAlgorithmBadges();
      if (widgets.pipeline) widgets.pipeline.setConfig(state.pipelineConfig);
    });

    dom.toggleCyclistVru.addEventListener("change", function () {
      state.pipelineConfig.cyclistVru = Boolean(dom.toggleCyclistVru.checked);
      updateAlgorithmBadges();
      if (widgets.pipeline) widgets.pipeline.setConfig(state.pipelineConfig);
    });

    document.querySelectorAll(".pipeline-node").forEach(function (node) {
      function activateNode() {
        document.querySelectorAll(".pipeline-node").forEach(function (n) {
          n.classList.toggle("is-selected", n === node);
        });
        var stage = node.getAttribute("data-stage");
        dom.pipelineStageDetail.textContent = PIPELINE_STAGE_INFO[stage] || "模块运行中";
      }
      node.addEventListener("click", activateNode);
      node.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          activateNode();
        }
      });
    });

    document.querySelectorAll("[data-log-filter]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        state.alertLogFilter = btn.dataset.logFilter || "all";
        document.querySelectorAll("[data-log-filter]").forEach(function (b) {
          b.classList.toggle("is-active", b === btn);
        });
        renderAlertLog();
      });
    });

    dom.btnClearAlertLog.addEventListener("click", function () {
      state.alertLog = [];
      state.lastLogSignature = "";
      renderAlertLog();
    });

    dom.historyRunSelect.addEventListener("change", function () {
      loadHistoryRunDetail(dom.historyRunSelect.value);
    });

    dom.btnRefreshRuns.addEventListener("click", function () {
      loadRunsList();
    });

    dom.btnHistoryReplay.addEventListener("click", function () {
      if (state.selectedRunId) {
        enterReplayMode(state.selectedRunId, 0);
        switchView("monitor");
      }
    });

    document.querySelectorAll("[data-table-tab]").forEach(function (btn) {
      btn.addEventListener("click", function () {
        state.historyTableTab = btn.dataset.tableTab || "conflicts";
        updateHistoryTableVisibility();
      });
    });
  }

  function init() {
    initDomRefs();
    initTheme();
    setupDialog(dom.jobDialog);
    setupDialog(dom.mobileNavDialog);
    initWidgets();
    updateAlgorithmBadges();
    bindEvents();
    setMonitorMode("simulation");
    dom.pipelineStageDetail.textContent = PIPELINE_STAGE_INFO.detect;
    loadHealth();
    loadRunsList();
    loadPresets();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
