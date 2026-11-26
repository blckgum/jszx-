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
    },
    studio: {
      index: 3,
      title: "标定与实验中心",
      subtitle: "可视化道路单应标定 · 禁停区掩膜 · 算法场景回归与消融实验室"
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

  function getActivePipelineConfig() {
    if (state.mode === "live" && state.liveJob && state.liveJob.config) {
      return state.liveJob.config;
    }
    if (
      state.mode === "replay" &&
      state.replay &&
      state.replay.data &&
      state.replay.data.summary &&
      state.replay.data.summary.configuration
    ) {
      var rCfg = state.replay.data.summary.configuration;
      return {
        speedModel: String(rCfg.speed_model || "ctrv"),
        conflictModel: String(rCfg.conflict_model || "obb_ssm"),
        stitching: Boolean(rCfg.enable_stitching),
        cyclistVru: Boolean(rCfg.include_cyclist_vru)
      };
    }
    return state && state.pipelineConfig ? state.pipelineConfig : {};
  }

  function getPipelineStageDetail(stage) {
    var cfg = getActivePipelineConfig();
    if (stage === "detect") {
      return PIPELINE_STAGE_INFO.detect;
    }
    if (stage === "track") {
      return cfg.stitching === false
        ? "ByteTrack: 仅启用跨帧运动与 IoU 轨迹关联（未启用 HSV 外观缝合）"
        : PIPELINE_STAGE_INFO.track;
    }
    if (stage === "motion") {
      var sm = String(cfg.speedModel || "ctrv").toLowerCase();
      if (sm === "kalman") {
        return "单应 / Kalman: 地面逆透视映射、车辆锚点偏置补偿与 2D 匀加速 (CA) 卡尔曼滤波";
      }
      if (sm === "ema") {
        return "单应 / EMA: 地面逆透视映射、车辆锚点偏置补偿与滑动窗口指数平滑";
      }
      return PIPELINE_STAGE_INFO.motion;
    }
    if (stage === "conflict") {
      var cm = String(cfg.conflictModel || "obb_ssm").toLowerCase();
      var vruSuffix = cfg.cyclistVru === false ? "（已关闭骑行者 VRU 评估）" : "（含非机动车骑行者 VRU）";
      if (cm === "cvm_circle") {
        return "CVM-Circle: 恒速外推与圆形包络近似测距，计算 TTC 与最小间距 " + vruSuffix;
      }
      return "OBB-SSM: 旋转有向包围盒分离轴测距，计算 TTC、MTTC、DRAC 与 PET " + vruSuffix;
    }
    if (stage === "alert") {
      return PIPELINE_STAGE_INFO.alert;
    }
    return "模块运行中";
  }

  function updatePipelineStageDetail() {
    if (!dom.pipelineStageDetail) return;
    dom.pipelineStageDetail.textContent = getPipelineStageDetail(state.selectedPipelineStage || "detect");
  }

  var state = {
    theme: "dark",
    activeView: "monitor",
    mode: "simulation", // "simulation" | "replay" | "live"
    scenario: "crosswalk_braking",
    paused: false,
    selectedPipelineStage: "detect",
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
    replayReqToken: 0,
    historyReqToken: 0,
    replay: {
      runId: null,
      framesCount: 0,
      fps: 25,
      currentIndex: 0,
      playing: false,
      timerId: null,
      playRafId: null,
      playAnchorPerfMs: null,
      playAnchorMediaMs: null,
      videoRafId: null,
      useJpegFallback: false,
      noVideo: false,
      hasVideoFrameMap: false,
      videoFrameSet: null,
      isFixture: false,
      data: null
    },
    liveJob: {
      jobId: null,
      status: null,
      sinceSeq: 0,
      pollTimerId: null,
      pollToken: 0,
      pollSeq: 0,
      lastAppliedPollSeq: 0,
      lastFramesProcessed: -1,
      config: null,
      lastMaxSpeed: null,
      lastMinTtc: null,
      speedLimitKmh: 30,
      activeAlertsCount: 0,
      quietPolls: 0,
      notifiedDone: false
    },
    systemInfo: {
      desktop_mode: false,
      allow_shutdown: false,
      can_browse_files: false,
      cuda_available: false,
      weights: []
    },
    soundEnabled: false,
    studioTab: "calib",
    calibTool: "points",
    calibState: {
      imagePoints: [],
      parkingZonesImage: [],
      currentParkingDraft: [],
      imageWidth: 0,
      imageHeight: 0
    },
    calibGroundPoints: [],
    calibGroundEdited: false,
    calibLastRectDims: { W: 6.0, L: 15.0 },
    calibImageDims: { w: 0, h: 0 },
    calibVersion: 0,
    calibPreviewReqId: 0,
    calibExtractReqId: 0,
    calibLoadedSource: "",
    calibPreview: null,
    chartsByRun: {},
    pendingDeleteRunId: null,
    scenariosResult: null,
    ablationResult: null,
    alertLog: [],
    alertLogFilter: "all",
    lastLogSignature: "",
    lastLogCycle: -1,
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
    simCamera: null,
    replayCamera: null,
    speedChart: null,
    calibCanvas: null,
    audioSynth: null
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
      history: $("view-history"),
      studio: $("view-studio")
    };
    dom.viewTitle = $("view-title");
    dom.viewSubtitle = $("view-subtitle");

    dom.serviceHealthDot = $("service-health-dot");
    dom.serviceHealthText = $("service-health-text");
    dom.sidebarModeText = $("sidebar-mode-text");
    dom.sidebarTargetText = $("sidebar-target-text");
    dom.sidebarComputeText = $("sidebar-compute-text");
    dom.btnDesktopShutdown = $("btn-desktop-shutdown");
    dom.mobileDesktopShutdown = $("mobile-desktop-shutdown");
    dom.themeToggleLabel = $("theme-toggle-label");
    dom.topbarThemeLabel = $("topbar-theme-label");
    dom.mobileThemeBtn = $("mobile-theme-btn");

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
    dom.simCameraCanvas = $("sim-camera-canvas");
    dom.simControls = $("sim-controls");
    dom.simSceneTelemetry = $("sim-scene-telemetry");
    dom.btnImportVideo = $("btn-import-video");
    dom.btnReplaySample = $("btn-replay-sample");

    dom.replayVideo = $("replay-video");
    dom.replayFrameImg = $("replay-frame-img");
    dom.replaySimCanvas = $("replay-sim-canvas");
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
    dom.simSpeedModel = $("sim-speed-model");
    dom.simConflictModel = $("sim-conflict-model");
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
    dom.btnAlertSoundToggle = $("btn-alert-sound-toggle");
    dom.alertSoundLabel = $("alert-sound-label");

    dom.historyRunSelect = $("history-run-select");
    dom.btnRefreshRuns = $("btn-refresh-runs");
    dom.btnHistoryReplay = $("btn-history-replay");
    dom.btnHistoryCharts = $("btn-history-charts");
    dom.btnExportCsv = $("btn-export-csv");
    dom.btnExportConflicts = $("btn-export-conflicts");
    dom.btnExportSummary = $("btn-export-summary");
    dom.btnHistoryOpenFolder = $("btn-history-open-folder");
    dom.btnHistoryDelete = $("btn-history-delete");
    dom.historyChartsCard = $("history-charts-card");
    dom.historyChartsMeta = $("history-charts-meta");
    dom.historyChartsGrid = $("history-charts-grid");
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

    dom.studioTabCalib = $("studio-tab-calib");
    dom.studioTabLab = $("studio-tab-lab");
    dom.studioStatusPill = $("studio-status-pill");
    dom.studioPanelCalib = $("studio-panel-calib");
    dom.studioPanelLab = $("studio-panel-lab");

    dom.calibPresetVideos = $("calib-preset-videos");
    dom.calibSource = $("calib-source");
    dom.btnCalibBrowseVideo = $("btn-calib-browse-video");
    dom.calibFrameIndex = $("calib-frame-index");
    dom.btnCalibExtract = $("btn-calib-extract");
    dom.calibFrameMeta = $("calib-frame-meta");
    dom.calibPointCount = $("calib-point-count");
    dom.btnCalibToolPoints = $("btn-calib-tool-points");
    dom.btnCalibToolParking = $("btn-calib-tool-parking");
    dom.btnCalibFinishParking = $("btn-calib-finish-parking");
    dom.btnCalibUndo = $("btn-calib-undo");
    dom.btnCalibClear = $("btn-calib-clear");
    dom.calibEditorCanvas = $("calib-editor-canvas");
    dom.calibCoordsPill = $("calib-coords-pill");
    dom.calibForm = $("calib-form");
    dom.calibWidthM = $("calib-width-m");
    dom.calibLengthM = $("calib-length-m");
    dom.calibName = $("calib-name");
    dom.calibFilename = $("calib-filename");
    dom.calibUseRansac = $("calib-use-ransac");
    dom.calibCorrespWrap = $("calib-corresp-wrap");
    dom.calibCorrespList = $("calib-corresp-list");
    dom.calibError = $("calib-error");
    dom.calibValidBadge = $("calib-valid-badge");
    dom.btnCalibPreview = $("btn-calib-preview");
    dom.btnCalibSave = $("btn-calib-save");
    dom.calibRmseVal = $("calib-rmse-val");
    dom.calibInliersVal = $("calib-inliers-val");
    dom.calibSavedVal = $("calib-saved-val");

    dom.btnRunScenarios = $("btn-run-scenarios");
    dom.scenariosStatusBadge = $("scenarios-status-badge");
    dom.scenariosPassRatio = $("scenarios-pass-ratio");
    dom.scenariosSummaryText = $("scenarios-summary-text");
    dom.scenariosTbody = $("scenarios-tbody");
    dom.btnRunAblation = $("btn-run-ablation");
    dom.ablationMeta = $("ablation-meta");
    dom.ablationTurnTbody = $("ablation-turn-tbody");
    dom.ablationCoreTbody = $("ablation-core-tbody");

    dom.jobDialog = $("job-dialog");
    dom.jobForm = $("job-form");
    dom.jobDialogError = $("job-dialog-error");
    dom.presetVideosList = $("preset-videos-list");
    dom.jobSource = $("job-source");
    dom.btnJobBrowseSource = $("btn-job-browse-source");
    dom.jobCalib = $("job-calib");
    dom.btnJobBrowseCalib = $("btn-job-browse-calib");
    dom.jobWeights = $("job-weights");
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

    dom.confirmDeleteDialog = $("confirm-delete-dialog");
    dom.deleteDialogRunId = $("delete-dialog-run-id");
    dom.confirmDeleteError = $("confirm-delete-error");
    dom.confirmDeleteCancel = $("confirm-delete-cancel");
    dom.confirmDeleteSubmit = $("confirm-delete-submit");

    dom.chartLightboxDialog = $("chart-lightbox-dialog");
    dom.lightboxTitle = $("lightbox-title");
    dom.lightboxSubtitle = $("lightbox-subtitle");
    dom.lightboxImg = $("lightbox-img");
    dom.lightboxDownload = $("lightbox-download");

    dom.mobileNavDialog = $("mobile-nav-dialog");
    dom.toastRegion = $("toast-region");
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
    if (dom.mobileThemeBtn) {
      dom.mobileThemeBtn.textContent = state.theme === "light" ? "切换深色石墨" : "切换浅色纸张";
    }
    if (widgets.replayCamera && typeof widgets.replayCamera.refreshTheme === "function") {
      widgets.replayCamera.refreshTheme();
    }
    if (widgets.roadTwin && typeof widgets.roadTwin.refreshTheme === "function") {
      widgets.roadTwin.refreshTheme();
    }
    if (widgets.simCamera && typeof widgets.simCamera.refreshTheme === "function") {
      widgets.simCamera.refreshTheme();
    }
    if (widgets.speedChart && typeof widgets.speedChart.refreshTheme === "function") {
      widgets.speedChart.refreshTheme();
    }
    if (widgets.calibCanvas && typeof widgets.calibCanvas.refreshTheme === "function") {
      widgets.calibCanvas.refreshTheme();
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
      var active = btn.dataset.view === viewId;
      btn.classList.toggle("is-active", active);
      if (active) {
        btn.setAttribute("aria-current", "page");
      } else {
        btn.removeAttribute("aria-current");
      }
    });

    Object.keys(dom.views).forEach(function (key) {
      dom.views[key].hidden = key !== viewId;
    });

    dom.viewTitle.textContent = meta.title;
    dom.viewSubtitle.textContent = meta.subtitle;
    document.title = meta.title + " | CampusMotion";

    if (viewId === "monitor") {
      window.requestAnimationFrame(function () {
        if (widgets.roadTwin && typeof widgets.roadTwin.refreshTheme === "function") {
          widgets.roadTwin.refreshTheme();
        }
        if (state.mode === "simulation" && widgets.simCamera && typeof widgets.simCamera.refreshTheme === "function") {
          widgets.simCamera.refreshTheme();
        } else if (state.mode === "replay" && widgets.replayCamera && typeof widgets.replayCamera.refreshTheme === "function") {
          widgets.replayCamera.refreshTheme();
        }
      });
    } else if (viewId === "history") {
      if (state.mode === "replay" && state.replay.playing) {
        stopReplayPlayback();
        if (widgets.pipeline && typeof widgets.pipeline.setPaused === "function") {
          widgets.pipeline.setPaused(true);
        }
      }
      if (!state.runs.length) {
        loadRunsList();
      } else if (widgets.speedChart && typeof widgets.speedChart.refreshTheme === "function") {
        window.requestAnimationFrame(function () {
          widgets.speedChart.refreshTheme();
        });
      }
    } else if (viewId === "studio") {
      if (state.mode === "replay" && state.replay.playing) {
        stopReplayPlayback();
        if (widgets.pipeline && typeof widgets.pipeline.setPaused === "function") {
          widgets.pipeline.setPaused(true);
        }
      }
      if (state.studioTab === "calib" && widgets.calibCanvas && typeof widgets.calibCanvas.refreshTheme === "function") {
        window.requestAnimationFrame(function () {
          widgets.calibCanvas.refreshTheme();
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

    if (typeof CM.SimulatedCameraCanvas === "function" && dom.simCameraCanvas) {
      widgets.simCamera = new CM.SimulatedCameraCanvas(dom.simCameraCanvas);
    }

    if (typeof CM.SimulatedCameraCanvas === "function" && dom.replaySimCanvas) {
      widgets.replayCamera = new CM.SimulatedCameraCanvas(dom.replaySimCanvas);
    }

    if (typeof CM.CampusRoadTwinCanvas === "function") {
      widgets.roadTwin = new CM.CampusRoadTwinCanvas(dom.roadTwinCanvas, {
        config: state.pipelineConfig,
        onTelemetry: handleTwinTelemetry,
        onRenderState: function (renderState) {
          if (state.mode === "simulation" && widgets.simCamera) {
            widgets.simCamera.renderState(renderState);
          }
        }
      });
      if (typeof widgets.roadTwin.setConfig === "function") {
        widgets.roadTwin.setConfig(state.pipelineConfig);
      }
      if (widgets.simCamera && typeof widgets.roadTwin.getCurrentRenderState === "function") {
        widgets.simCamera.renderState(widgets.roadTwin.getCurrentRenderState());
      }
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
            var runObj = state.runs.find(function (r) { return r.run_id === state.selectedRunId; });
            if (runObj && !isPlayableRun(runObj)) {
              showToast("该历史记录无有效视频帧，无法回放");
              return;
            }
            enterReplayMode(state.selectedRunId, frameIndex);
            switchView("monitor");
          }
        }
      });
    }

    if (typeof CM.CalibrationCanvas === "function" && dom.calibEditorCanvas) {
      widgets.calibCanvas = new CM.CalibrationCanvas(dom.calibEditorCanvas, {
        onChange: handleCalibCanvasChange
      });
    }

    if (typeof CM.AlertAudioSynth === "function") {
      widgets.audioSynth = new CM.AlertAudioSynth();
    }
  }

  function updateRiskPill(riskLevel) {
    var lvl = parseRiskLevel(riskLevel);
    dom.topbarRiskPill.setAttribute("data-risk", String(lvl));
    dom.topbarRiskText.textContent = riskText(lvl);
  }

  function updateAlgorithmBadges() {
    var cfg = getActivePipelineConfig();
    if (dom.badgeSpeedModel) {
      dom.badgeSpeedModel.textContent = String(cfg.speedModel || "ctrv").toUpperCase();
    }
    if (dom.badgeConflictModel) {
      dom.badgeConflictModel.textContent = cfg.conflictModel === "cvm_circle" ? "CVM-CIRCLE" : "OBB-SSM";
    }
    if (dom.badgeVruModel) {
      var flags = [];
      if (cfg.stitching) flags.push("HSV缝合");
      if (cfg.cyclistVru) flags.push("VRU");
      dom.badgeVruModel.textContent = flags.length ? flags.join("+") : "基础检测";
    }
  }

  function syncPipelineControls() {
    var simEditable = state.mode === "simulation";
    if (dom.simSpeedModel) {
      dom.simSpeedModel.value = state.pipelineConfig.speedModel;
      dom.simSpeedModel.disabled = !simEditable;
    }
    if (dom.simConflictModel) {
      dom.simConflictModel.value = state.pipelineConfig.conflictModel;
      dom.simConflictModel.disabled = !simEditable;
    }
    if (dom.toggleStitching) {
      dom.toggleStitching.checked = Boolean(state.pipelineConfig.stitching);
      dom.toggleStitching.disabled = !simEditable;
    }
    if (dom.toggleCyclistVru) {
      dom.toggleCyclistVru.checked = Boolean(state.pipelineConfig.cyclistVru);
      dom.toggleCyclistVru.disabled = !simEditable;
    }
  }

  function applyPipelineConfigChange() {
    var activeCfg = getActivePipelineConfig();
    updateAlgorithmBadges();
    updatePipelineStageDetail();
    if (widgets.pipeline && typeof widgets.pipeline.setConfig === "function") {
      widgets.pipeline.setConfig(activeCfg);
    }
    if (state.mode === "simulation" && widgets.roadTwin && typeof widgets.roadTwin.setConfig === "function") {
      widgets.roadTwin.setConfig(state.pipelineConfig);
    }
  }

  function resetAlertLogDedup() {
    state.lastLogSignature = "";
    state.lastLogCycle = -1;
    state.lastLogTimestamp = -1;
  }

  function handleTwinTelemetry(snap) {
    if (!snap || state.mode !== "simulation") return;

    widgets.kpiMaxSpeed.setValue(snap.maxSpeedKmh);
    widgets.kpiMinTtc.setValue(snap.minTtcS);
    widgets.kpiActiveAlerts.setValue(snap.activeConflicts || 0);
    widgets.kpiFrames.setValue(snap.frameCount || 0);
    dom.kpiFramesMeta.textContent =
      "周期 " + formatNumber(snap.timestampS, 1, "0.0") + " / " +
      formatNumber(snap.cycleDurationS || 8, 1, "8.0") + " s · " +
      (snap.fps || 25) + " FPS";
    if (dom.kpiSpeedMeta) {
      dom.kpiSpeedMeta.textContent = "限速基准 30.0 km/h";
    }

    var risk = parseRiskLevel(snap.riskLevel);
    updateRiskPill(risk);

    if (widgets.pipeline) {
      widgets.pipeline.setRiskLevel(risk);
    }

    if (dom.simSceneTelemetry) {
      var ttcNote = typeof snap.minTtcS === "number" ? (" · TTC " + snap.minTtcS.toFixed(2) + "s") : "";
      dom.simSceneTelemetry.textContent =
        (SCENARIO_LABELS[state.scenario] || state.scenario) + " · " +
        (snap.vehicleCount || 0) + " 车辆 · " + (snap.vruCount || 0) + " VRU · " +
        riskText(risk) + ttcNote;
    }
    dom.twinCounts.textContent = "车辆 " + (snap.vehicleCount || 0) + " · 行人/骑行 " + (snap.vruCount || 0);

    updateDualEndMirrors({
      riskLevel: risk,
      timestampS: snap.timestampS,
      cycleIndex: snap.cycleIndex,
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

    appendAlertLogBatch(driverAlerts, pedAlerts, payload.timestampS, payload.cycleIndex);

    var peakRisk = Math.max(driverRisk, pedRisk, parseRiskLevel(payload.riskLevel));
    if (state.soundEnabled && widgets.audioSynth && typeof widgets.audioSynth.triggerRisk === "function") {
      widgets.audioSynth.triggerRisk(peakRisk);
    }
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

  function appendAlertLogBatch(driverAlerts, pedAlerts, timestampS, cycleIndex) {
    var combined = driverAlerts.concat(pedAlerts);
    if (!combined.length) return;

    var ts = typeof timestampS === "number" && Number.isFinite(timestampS) ? timestampS : 0;
    var hasCycle = typeof cycleIndex === "number" && Number.isFinite(cycleIndex);
    var cyc = hasCycle ? cycleIndex : -1;
    var sig = combined
      .map(function (a) {
        return [a.role || a.target || "", parseRiskLevel(a.risk_level), a.veh_track_id, a.ped_track_id].join(":");
      })
      .join("|");

    if (sig === state.lastLogSignature) {
      if (state.mode === "simulation" && hasCycle) {
        if (cyc === state.lastLogCycle) {
          return;
        }
      } else if (Math.abs(ts - state.lastLogTimestamp) < 1.2) {
        return;
      }
    }
    state.lastLogSignature = sig;
    if (hasCycle) {
      state.lastLogCycle = cyc;
    }
    state.lastLogTimestamp = ts;

    var sourceLabel = state.mode === "simulation"
      ? ("仿真 · " + (SCENARIO_LABELS[state.scenario] || state.scenario))
      : (state.mode === "replay" ? "回放" : "实时");

    combined.slice(0, 4).forEach(function (item) {
      var roleVal = item.role || item.target || "driver";
      state.alertLog.unshift({
        riskLevel: parseRiskLevel(item.risk_level),
        roleLabel: roleVal === "pedestrian" ? "路侧端" : "车端",
        sourceLabel: sourceLabel,
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
      var metaParts = [];
      if (entry.sourceLabel) metaParts.push(entry.sourceLabel);
      metaParts.push("t=" + formatNumber(entry.timestampS, 1, "0.0") + "s");
      metaParts.push("车辆#" + (entry.vehId != null ? entry.vehId : "--"));
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
    resetAlertLogDedup();

    if (mode !== "replay") {
      state.replayReqToken += 1;
      stopReplayPlayback();
    }
    if (mode !== "live") {
      stopLiveJobPolling();
    }

    dom.viewportSimPanel.hidden = mode !== "simulation";
    dom.viewportReplayPanel.hidden = mode !== "replay";
    dom.viewportLivePanel.hidden = mode !== "live";
    if (dom.simControls) dom.simControls.hidden = mode !== "simulation";
    dom.replayControls.hidden = mode !== "replay";
    dom.liveControls.hidden = mode !== "live";
    dom.scenarioSelect.hidden = mode !== "simulation";

    if (mode === "simulation") {
      dom.viewportModeBadge.textContent = "仿真模式";
      dom.sidebarModeText.textContent = "仿真演示";
      setSidebarTarget(SCENARIO_LABELS[state.scenario] || state.scenario);
      dom.twinModeLabel.textContent = "6.0m x 15.0m";
      dom.btnTwinPause.hidden = false;
      if (dom.kpiSpeedMeta) {
        dom.kpiSpeedMeta.textContent = "限速基准 30.0 km/h";
      }
      if (widgets.roadTwin) {
        widgets.roadTwin.setMode("simulation");
        widgets.roadTwin.setScenario(state.scenario);
        widgets.roadTwin.setPaused(state.paused);
        if (widgets.simCamera && typeof widgets.roadTwin.getCurrentRenderState === "function") {
          widgets.simCamera.renderState(widgets.roadTwin.getCurrentRenderState());
        }
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
      dom.twinModeLabel.textContent = "实时检测追踪";
      dom.btnTwinPause.hidden = true;
      if (widgets.roadTwin) {
        widgets.roadTwin.setMode("replay");
      }
    }
    syncPipelineControls();
    applyPipelineConfigChange();
  }

  function nowPerfMs() {
    if (typeof performance !== "undefined" && typeof performance.now === "function") {
      return performance.now();
    }
    return Date.now();
  }

  function replayFrameHasVideo(idx) {
    if (state.replay.noVideo) return false;
    if (state.replay.videoFrameSet) {
      return state.replay.videoFrameSet.has(idx);
    }
    return true;
  }

  function getReplayFrameTimeMs(dataset, idx) {
    if (!dataset) return idx * 40;
    var rows = dataset.speedsByFrame && dataset.speedsByFrame[idx];
    if (rows && rows.length) {
      var ts = parseFloat(rows[0].timestamp_s);
      if (Number.isFinite(ts)) return ts * 1000;
    }
    var cfs = dataset.conflictsByFrame && dataset.conflictsByFrame[idx];
    if (cfs && cfs.length && typeof cfs[0].timestamp_s === "number" && Number.isFinite(cfs[0].timestamp_s)) {
      return cfs[0].timestamp_s * 1000;
    }
    return idx * (dataset.nominalFrameMs || 40);
  }

  function resetReplayClockAnchor(idx) {
    state.replay.playAnchorPerfMs = nowPerfMs();
    state.replay.playAnchorMediaMs = getReplayFrameTimeMs(state.replay.data, idx);
  }

  function stopReplayPlayback() {
    state.replay.playing = false;
    state.replay.playAnchorPerfMs = null;
    if (state.replay.timerId !== null) {
      window.clearTimeout(state.replay.timerId);
      state.replay.timerId = null;
    }
    if (state.replay.playRafId !== null) {
      if (typeof window.cancelAnimationFrame === "function") {
        window.cancelAnimationFrame(state.replay.playRafId);
      }
      state.replay.playRafId = null;
    }
    if (state.replay.videoRafId !== null) {
      if (typeof window.cancelAnimationFrame === "function") {
        window.cancelAnimationFrame(state.replay.videoRafId);
      }
      state.replay.videoRafId = null;
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

  var REPLAY_CHUNK_SIZE = 250;

  function mergeLoadedRange(ranges, startIdx, endIdx) {
    if (startIdx > endIdx) return ranges;
    var next = ranges.concat([[startIdx, endIdx]]);
    next.sort(function (a, b) { return a[0] - b[0]; });
    var merged = [];
    for (var i = 0; i < next.length; i += 1) {
      var cur = next[i];
      if (!merged.length || cur[0] > merged[merged.length - 1][1] + 1) {
        merged.push([cur[0], cur[1]]);
      } else if (cur[1] > merged[merged.length - 1][1]) {
        merged[merged.length - 1][1] = cur[1];
      }
    }
    return merged;
  }

  function isReplayFrameLoaded(dataset, frameIdx) {
    if (!dataset) return false;
    if (dataset.allFramesLoaded) return true;
    var ranges = dataset.loadedRanges || [];
    for (var i = 0; i < ranges.length; i += 1) {
      if (frameIdx >= ranges[i][0] && frameIdx <= ranges[i][1]) {
        return true;
      }
    }
    return false;
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
    var maxSpeedFrameIdx = -1;
    var maxConflictFrameIdx = -1;
    var maxFrameIdx = 0;
    var frameTimes = {};

    speedRows.forEach(function (row) {
      var fIdx = parseInt(row.frame_idx, 10);
      if (!Number.isFinite(fIdx)) return;
      if (fIdx > maxSpeedFrameIdx) maxSpeedFrameIdx = fIdx;
      if (fIdx > maxFrameIdx) maxFrameIdx = fIdx;
      if (!speedsByFrame[fIdx]) speedsByFrame[fIdx] = [];
      speedsByFrame[fIdx].push(row);
      var tVal = parseFloat(row.timestamp_s);
      if (Number.isFinite(tVal) && frameTimes[fIdx] === undefined) {
        frameTimes[fIdx] = tVal;
      }

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
      if (fIdx > maxConflictFrameIdx) maxConflictFrameIdx = fIdx;
      if (fIdx > maxFrameIdx) maxFrameIdx = fIdx;
      if (!conflictsByFrame[fIdx]) conflictsByFrame[fIdx] = [];
      conflictsByFrame[fIdx].push(rec);
      if (typeof rec.timestamp_s === "number" && Number.isFinite(rec.timestamp_s) && frameTimes[fIdx] === undefined) {
        frameTimes[fIdx] = rec.timestamp_s;
      }
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
      minY = Math.floor((minY - 1.8) * 2) / 2;
      maxY = Math.ceil((maxY + 2.0) * 2) / 2;
    }

    var sortedTimeFrames = Object.keys(frameTimes).map(Number).sort(function (a, b) { return a - b; });
    var dtSamples = [];
    for (var ti = 1; ti < sortedTimeFrames.length; ti += 1) {
      var fA = sortedTimeFrames[ti - 1];
      var fB = sortedTimeFrames[ti];
      if (fB > fA) {
        var stepDt = (frameTimes[fB] - frameTimes[fA]) / (fB - fA);
        if (Number.isFinite(stepDt) && stepDt >= 0.005 && stepDt <= 1.0) {
          dtSamples.push(stepDt);
        }
      }
    }
    var nominalFrameMs = 40;
    if (dtSamples.length > 0) {
      dtSamples.sort(function (a, b) { return a - b; });
      nominalFrameMs = dtSamples[Math.floor(dtSamples.length / 2)] * 1000;
    } else if (summary && summary.configuration) {
      var cfgFps = summary.configuration.fps_used !== undefined && summary.configuration.fps_used !== null
        ? summary.configuration.fps_used
        : summary.configuration.fps;
      if (typeof cfgFps === "number" && Number.isFinite(cfgFps) && cfgFps > 0) {
        nominalFrameMs = 1000 / cfgFps;
      }
    }
    nominalFrameMs = Math.max(5, Math.min(500, nominalFrameMs));

    var totalFrames = Math.max(
      (summary && typeof summary.frames_processed === "number" ? summary.frames_processed : 0),
      maxFrameIdx + 1
    );
    var speedsTotalRows = speedsRes && typeof speedsRes.total_rows === "number" ? speedsRes.total_rows : speedRows.length;
    var speedsTruncated = Boolean(speedsRes && speedsRes.truncated);
    var conflictsTotalCount = conflictsRes && typeof conflictsRes.total_count === "number"
      ? conflictsRes.total_count
      : conflicts.length;
    var conflictsTruncated = Boolean(
      (conflictsRes && conflictsRes.truncated) ||
      (conflictsRes && typeof conflictsRes.total_count === "number" ? conflictsRes.total_count > conflicts.length : conflicts.length >= 5000)
    );
    var allFramesLoaded = !speedsTruncated && !conflictsTruncated;
    var loadedRanges = [];
    if (allFramesLoaded) {
      loadedRanges = [[0, Math.max(0, totalFrames - 1)]];
    } else {
      // 截断预览末帧可能只含前半部分目标，保守退一帧作为已完整装载上界
      var speedCompleteMax = speedsTruncated ? (maxSpeedFrameIdx - 1) : (totalFrames - 1);
      var conflictCompleteMax = conflictsTruncated ? (maxConflictFrameIdx - 1) : (totalFrames - 1);
      var initMax = Math.min(speedCompleteMax, conflictCompleteMax);
      if (initMax >= 0) {
        loadedRanges = [[0, initMax]];
      }
    }

    return {
      runId: runId,
      summary: summary || {},
      events: events,
      conflicts: conflicts,
      speedRows: speedRows,
      speedsTotalRows: speedsTotalRows,
      speedsTruncated: speedsTruncated,
      conflictsTotalCount: conflictsTotalCount,
      conflictsTruncated: conflictsTruncated,
      allFramesLoaded: allFramesLoaded,
      loadedRanges: loadedRanges,
      cappedFrames: {},
      loadingChunks: {},
      nominalFrameMs: nominalFrameMs,
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
    var prevById = {};
    for (var back = 1; back <= 3; back += 1) {
      var prevIdx = frameIndex - back;
      if (prevIdx < 0) break;
      var prevRows = dataset.speedsByFrame[prevIdx] || [];
      prevRows.forEach(function (pr) {
        var key = String(pr.track_id);
        if (!prevById[key]) {
          prevById[key] = pr;
        }
      });
    }

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
      var nomSec = (dataset && dataset.nominalFrameMs ? dataset.nominalFrameMs : 40) / 1000;
      timestampS = frameIndex * nomSec;
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
        var prevFrameNum = parseInt(prev.frame_idx, 10);
        var frameGap = Number.isFinite(prevFrameNum) ? Math.max(1, frameIndex - prevFrameNum) : 1;
        var rawDt = parseFloat(row.timestamp_s) - parseFloat(prev.timestamp_s);
        var fallbackStep = (dataset && dataset.nominalFrameMs ? dataset.nominalFrameMs : 40) / 1000;
        var dt = (Number.isFinite(rawDt) && rawDt > 0.015) ? rawDt : Math.max(0.02, frameGap * fallbackStep);
        if (Number.isFinite(px) && Number.isFinite(py) && Number.isFinite(x) && Number.isFinite(y)) {
          vx = (x - px) / dt;
          vy = (y - py) / dt;
          if (Math.hypot(vx, vy) > 0.08) {
            headingDeg = (Math.atan2(vy, vx) * 180) / Math.PI;
          }
        }
      }

      var riskLvl = parseRiskLevel(row.risk_level);
      var coarseCat = String(row.category || "vehicle").trim().toLowerCase();
      var fineKind = String(row.object_kind || row.class_name || coarseCat || "vehicle").trim().toLowerCase();
      if (!Number.isFinite(x) || !Number.isFinite(y)) return;

      if (isVruCategory(fineKind) || isVruCategory(coarseCat)) {
        var vruKind = isVruCategory(fineKind) ? fineKind : coarseCat;
        vrus.push({
          id: tid,
          trackId: tid,
          x: x,
          y: y,
          vx: vx,
          vy: vy,
          speedMs: speedKmh / 3.6,
          category: vruKind,
          kind: vruKind,
          radiusM: vruKind === "motorcycle" ? 0.65 : (vruKind === "bicycle" ? 0.50 : 0.40),
          lengthM: vruKind === "motorcycle" ? 2.1 : (vruKind === "bicycle" ? 1.8 : 0.8),
          widthM: vruKind === "motorcycle" ? 0.8 : (vruKind === "bicycle" ? 0.7 : 0.8),
          riskLevel: riskLvl
        });
      } else {
        var parsedLen = parseFloat(row.length_m);
        var parsedWid = parseFloat(row.width_m);
        var vehLen = Number.isFinite(parsedLen) && parsedLen > 0
          ? parsedLen
          : (fineKind === "bus" ? 10.0 : (fineKind === "truck" ? 8.0 : 4.5));
        var vehWid = Number.isFinite(parsedWid) && parsedWid > 0
          ? parsedWid
          : (fineKind === "bus" ? 2.55 : (fineKind === "truck" ? 2.5 : 1.9));
        vehicles.push({
          id: tid,
          trackId: tid,
          x: x,
          y: y,
          vx: vx,
          vy: vy,
          speedKmh: speedKmh,
          headingDeg: headingDeg,
          lengthM: vehLen,
          widthM: vehWid,
          kind: fineKind,
          category: fineKind,
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

  function ensureReplayFrameWindow(dataset, frameIdx, reqToken) {
    if (!dataset || dataset.allFramesLoaded) return Promise.resolve();
    var totalF = Math.max(1, dataset.totalFrames || 1);
    var clamped = Math.max(0, Math.min(totalF - 1, Math.round(frameIdx)));
    if (isReplayFrameLoaded(dataset, clamped)) return Promise.resolve();

    var chunkStart = Math.floor(clamped / REPLAY_CHUNK_SIZE) * REPLAY_CHUNK_SIZE;
    var chunkEnd = Math.min(totalF - 1, chunkStart + REPLAY_CHUNK_SIZE - 1);
    while (chunkStart < clamped && isReplayFrameLoaded(dataset, chunkStart)) {
      chunkStart += 1;
    }
    while (chunkEnd > clamped && isReplayFrameLoaded(dataset, chunkEnd)) {
      chunkEnd -= 1;
    }
    var chunkKey = chunkStart + ":" + chunkEnd;
    if (dataset.loadingChunks[chunkKey]) {
      return dataset.loadingChunks[chunkKey];
    }

    var base = "/api/runs/" + encodeURIComponent(dataset.runId);
    var promise = (function fetchSubRange(reqStart, reqEnd) {
      return Promise.all([
        fetchJson(base + "/speeds?frame_start=" + reqStart + "&frame_end=" + reqEnd + "&limit=20000"),
        fetchJson(base + "/conflicts?frame_start=" + reqStart + "&frame_end=" + reqEnd + "&limit=20000")
      ]).then(function (results) {
        if (state.replay.data !== dataset) return;
        var spRes = results[0] || {};
        var cfRes = results[1] || {};
        var spRows = spRes.rows || [];
        var cfRows = cfRes.conflicts || [];
        var spTrunc = Boolean(spRes.truncated);
        var cfTrunc = Boolean(cfRes.truncated);

        var maxSpFrame = reqStart - 1;
        spRows.forEach(function (row) {
          var idx = parseInt(row.frame_idx, 10);
          if (Number.isFinite(idx) && idx > maxSpFrame) maxSpFrame = idx;
        });
        var maxCfFrame = reqStart - 1;
        cfRows.forEach(function (rec) {
          var idx = typeof rec.frame_index === "number" ? rec.frame_index : parseInt(rec.frame_index, 10);
          if (Number.isFinite(idx) && idx > maxCfFrame) maxCfFrame = idx;
        });

        var completeEnd = reqEnd;
        if (spTrunc) completeEnd = Math.min(completeEnd, maxSpFrame - 1);
        if (cfTrunc) completeEnd = Math.min(completeEnd, maxCfFrame - 1);
        // 单帧即超出单页上限时保留该帧已返回片段并记录标记，同时继续向后请求同区间其余帧
        var hitSingleFrameCap = (spTrunc || cfTrunc) && completeEnd < reqStart;
        if (hitSingleFrameCap) {
          completeEnd = reqStart;
          if (!dataset.cappedFrames) dataset.cappedFrames = {};
          dataset.cappedFrames[reqStart] = true;
        }

        for (var fi = reqStart; fi <= completeEnd; fi += 1) {
          delete dataset.speedsByFrame[fi];
          delete dataset.conflictsByFrame[fi];
        }
        spRows.forEach(function (row) {
          var idx = parseInt(row.frame_idx, 10);
          if (!Number.isFinite(idx) || idx < reqStart || idx > completeEnd) return;
          if (!dataset.speedsByFrame[idx]) dataset.speedsByFrame[idx] = [];
          dataset.speedsByFrame[idx].push(row);
          var rx = parseFloat(row.x_m);
          var ry = parseFloat(row.y_m);
          if (Number.isFinite(rx)) {
            dataset.bounds.xMin = Math.min(dataset.bounds.xMin, Math.floor((rx - 1.2) * 2) / 2);
            dataset.bounds.xMax = Math.max(dataset.bounds.xMax, Math.ceil((rx + 1.2) * 2) / 2);
          }
          if (Number.isFinite(ry)) {
            dataset.bounds.yMin = Math.min(dataset.bounds.yMin, Math.floor((ry - 1.8) * 2) / 2);
            dataset.bounds.yMax = Math.max(dataset.bounds.yMax, Math.ceil((ry + 2.0) * 2) / 2);
          }
        });
        cfRows.forEach(function (rec) {
          var idx = typeof rec.frame_index === "number" ? rec.frame_index : parseInt(rec.frame_index, 10);
          if (!Number.isFinite(idx) || idx < reqStart || idx > completeEnd) return;
          if (!dataset.conflictsByFrame[idx]) dataset.conflictsByFrame[idx] = [];
          dataset.conflictsByFrame[idx].push(rec);
        });
        dataset.loadedRanges = mergeLoadedRange(dataset.loadedRanges, reqStart, completeEnd);
        dom.twinModeLabel.textContent =
          "x[" + dataset.bounds.xMin + "," + dataset.bounds.xMax + "] y[" + dataset.bounds.yMin + "," + dataset.bounds.yMax + "]";

        if (reqToken === state.replayReqToken && state.mode === "replay" && state.replay.runId === dataset.runId) {
          if (state.replay.currentIndex >= reqStart && state.replay.currentIndex <= completeEnd) {
            renderReplayFrame(state.replay.currentIndex);
          }
          if (
            state.replay.playing &&
            state.replay.useJpegFallback &&
            state.replay.playRafId === null &&
            state.replay.timerId === null &&
            isReplayFrameLoaded(dataset, state.replay.currentIndex)
          ) {
            resetReplayClockAnchor(state.replay.currentIndex);
            scheduleNextReplayTick();
          }
        }

        if ((spTrunc || cfTrunc) && completeEnd < reqEnd) {
          return fetchSubRange(completeEnd + 1, reqEnd);
        }
        return null;
      });
    })(chunkStart, chunkEnd)
      .catch(function (err) {
        if (reqToken === state.replayReqToken && state.mode === "replay" && state.replay.runId === dataset.runId) {
          dom.replayMediaCaption.textContent = "加载帧区间失败: " + err.message;
        }
      })
      .finally(function () {
        delete dataset.loadingChunks[chunkKey];
      });

    dataset.loadingChunks[chunkKey] = promise;
    return promise;
  }

  function nextFrameDeltaMs(dataset, curIdx) {
    var fallbackMs = (dataset && dataset.nominalFrameMs) ? dataset.nominalFrameMs : 40;
    if (!dataset) return fallbackMs;
    var maxIdx = Math.max(0, dataset.totalFrames - 1);
    if (curIdx >= maxIdx) return fallbackMs;
    var rCur = dataset.speedsByFrame[curIdx] || [];
    var rNext = dataset.speedsByFrame[curIdx + 1] || [];
    var tCur = rCur.length ? parseFloat(rCur[0].timestamp_s) : NaN;
    var tNext = rNext.length ? parseFloat(rNext[0].timestamp_s) : NaN;
    if (!Number.isFinite(tCur)) {
      var cCur = dataset.conflictsByFrame[curIdx] || [];
      if (cCur.length && typeof cCur[0].timestamp_s === "number") tCur = cCur[0].timestamp_s;
    }
    if (!Number.isFinite(tNext)) {
      var cNext = dataset.conflictsByFrame[curIdx + 1] || [];
      if (cNext.length && typeof cNext[0].timestamp_s === "number") tNext = cNext[0].timestamp_s;
    }
    if (Number.isFinite(tCur) && Number.isFinite(tNext) && tNext > tCur) {
      return Math.max(5, Math.min(1000, (tNext - tCur) * 1000));
    }
    return fallbackMs;
  }

  function renderReplayFrame(frameIndex) {
    var dataset = state.replay.data;
    if (!dataset) return;

    var maxIdx = Math.max(0, dataset.totalFrames - 1);
    var idx = Math.max(0, Math.min(maxIdx, Math.round(frameIndex)));
    state.replay.currentIndex = idx;

    dom.replayScrubber.max = String(maxIdx);
    dom.replayScrubber.value = String(idx);

    if (!isReplayFrameLoaded(dataset, idx)) {
      var loadingTs = idx * ((dataset.nominalFrameMs || 40) / 1000);
      dom.replayFrameLabel.textContent = idx + " / " + maxIdx + " (加载中...)";
      dom.replayMediaCaption.textContent =
        dataset.runId + " · 帧 #" + idx + " · 正在按帧区间加载轨迹与风险数据...";
      dom.twinCounts.textContent = "帧 #" + idx + " 轨迹区间加载中...";
      dom.kpiFramesMeta.textContent = "回放总帧数 " + dataset.totalFrames + " · " + formatNumber(loadingTs, 1, "0.0") + " s (区间加载)";
      ensureReplayFrameWindow(dataset, idx, state.replayReqToken);
      return;
    }

    if (state.replay.playing && !dataset.allFramesLoaded && idx + 40 <= maxIdx && !isReplayFrameLoaded(dataset, idx + 40)) {
      ensureReplayFrameWindow(dataset, idx + 40, state.replayReqToken);
    }

    var framePayload = buildReplayFramePayload(dataset, idx);
    var frameHasVideo = replayFrameHasVideo(idx);

    dom.replayFrameLabel.textContent =
      idx + " / " + maxIdx + " (" + formatNumber(framePayload.timestampS, 1, "0.0") + "s)";
    dom.replayMediaCaption.textContent =
      dataset.runId + " · 帧 #" + idx + " · " + riskText(framePayload.riskLevel) +
      (state.replay.isFixture ? " · 测试固件(非实拍)" : "") +
      (state.replay.noVideo ? " · 纯轨迹重构（无标注 MP4）" : (!frameHasVideo ? " · 当前帧无视频画面（轨迹重构）" : ""));

    if (dom.replayFrameImg && (state.replay.useJpegFallback || !state.replay.playing)) {
      if (!frameHasVideo) {
        dom.replayFrameImg.hidden = true;
        if (dom.replayFrameImg.dataset) delete dom.replayFrameImg.dataset.frameIndex;
        dom.replayVideo.hidden = true;
        if (dom.replaySimCanvas) dom.replaySimCanvas.hidden = false;
      } else {
        if (dom.replaySimCanvas) dom.replaySimCanvas.hidden = true;
        dom.replayFrameImg.hidden = false;
        dom.replayVideo.hidden = true;
        if (dom.replayFrameImg.dataset) dom.replayFrameImg.dataset.frameIndex = String(idx);
        dom.replayFrameImg.src =
          "/api/runs/" + encodeURIComponent(dataset.runId) + "/frame?index=" + idx;
      }
    }

    if (widgets.roadTwin) {
      widgets.roadTwin.setReplayFrame(framePayload);
    }
    if (!frameHasVideo && widgets.replayCamera && widgets.roadTwin && typeof widgets.roadTwin.getCurrentRenderState === "function") {
      var camState = Object.assign({}, widgets.roadTwin.getCurrentRenderState(), {
        cameraLabel: state.replay.noVideo ? "TRACKS-ONLY · 3D 轨迹重构（无标注 MP4）" : "SPARSE FRAME · 3D 轨迹重构",
        cameraShortLabel: state.replay.noVideo ? "TRACKS-ONLY · 3D 重构" : "SPARSE · 3D 重构"
      });
      widgets.replayCamera.refreshTheme();
      widgets.replayCamera.renderState(camState);
    }

    widgets.kpiMaxSpeed.setValue(framePayload.maxSpeedKmh);
    widgets.kpiMinTtc.setValue(framePayload.minTtcS);
    widgets.kpiActiveAlerts.setValue(framePayload.activeConflicts);
    widgets.kpiFrames.setValue(idx + 1);
    dom.kpiFramesMeta.textContent =
      "回放总帧数 " + dataset.totalFrames + " · " + formatNumber(framePayload.timestampS, 1, "0.0") + " s";
    if (dom.kpiSpeedMeta) {
      var replayLimit = dataset.summary && typeof dataset.summary.speed_limit_kmh === "number"
        ? dataset.summary.speed_limit_kmh
        : 30;
      dom.kpiSpeedMeta.textContent = "限速基准 " + formatNumber(replayLimit, 1, "30.0") + " km/h";
    }
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

  function scheduleNextReplayTick() {
    if (!state.replay.playing || !state.replay.data) {
      state.replay.timerId = null;
      state.replay.playRafId = null;
      return;
    }
    if (state.replay.playAnchorPerfMs === null) {
      resetReplayClockAnchor(state.replay.currentIndex);
    }

    function onClockTick() {
      state.replay.playRafId = null;
      state.replay.timerId = null;
      if (!state.replay.playing || !state.replay.data) return;
      var dataset = state.replay.data;

      if (
        replayFrameHasVideo(state.replay.currentIndex) &&
        dom.replayFrameImg &&
        !dom.replayFrameImg.hidden &&
        !dom.replayFrameImg.complete
      ) {
        resetReplayClockAnchor(state.replay.currentIndex);
        scheduleNextReplayTick();
        return;
      }

      var now = nowPerfMs();
      var maxIdx = Math.max(0, dataset.totalFrames - 1);
      var targetMediaMs = state.replay.playAnchorMediaMs + Math.max(0, now - state.replay.playAnchorPerfMs);
      var curIdx = state.replay.currentIndex;
      var nextIdx = curIdx;

      if (curIdx >= maxIdx) {
        var wrapThreshold = getReplayFrameTimeMs(dataset, maxIdx) + nextFrameDeltaMs(dataset, maxIdx);
        if (targetMediaMs >= wrapThreshold) {
          nextIdx = 0;
          resetReplayClockAnchor(0);
        }
      } else {
        while (nextIdx < maxIdx) {
          if (!isReplayFrameLoaded(dataset, nextIdx + 1)) {
            nextIdx += 1;
            break;
          }
          if (getReplayFrameTimeMs(dataset, nextIdx + 1) <= targetMediaMs + 0.5) {
            nextIdx += 1;
          } else {
            break;
          }
        }
      }

      if (nextIdx !== curIdx) {
        if (!isReplayFrameLoaded(dataset, nextIdx)) {
          state.replay.playAnchorPerfMs = null;
          renderReplayFrame(nextIdx);
          return;
        }
        renderReplayFrame(nextIdx);
      }
      if (state.replay.playing) {
        scheduleNextReplayTick();
      }
    }

    if (typeof window.requestAnimationFrame === "function") {
      state.replay.playRafId = window.requestAnimationFrame(onClockTick);
    } else {
      var delayMs = nextFrameDeltaMs(state.replay.data, state.replay.currentIndex);
      state.replay.timerId = window.setTimeout(onClockTick, Math.max(4, Math.round(delayMs)));
    }
  }

  function syncVideoReplayFrameLoop() {
    if (!state.replay.playing || state.replay.useJpegFallback || !state.replay.data || !dom.replayVideo) {
      state.replay.videoRafId = null;
      return;
    }
    var dataset = state.replay.data;
    var stepSec = (dataset.nominalFrameMs || 40) / 1000;
    var estIdx = Math.min(
      Math.max(0, dataset.totalFrames - 1),
      Math.max(0, Math.round((dom.replayVideo.currentTime || 0) / stepSec))
    );
    if (estIdx !== state.replay.currentIndex) {
      renderReplayFrame(estIdx);
    }
    state.replay.videoRafId = window.requestAnimationFrame(syncVideoReplayFrameLoop);
  }

  function startReplayPlayback() {
    if (!state.replay.data) return;
    stopReplayPlayback();
    state.replay.playing = true;
    dom.btnReplayToggle.textContent = "暂停回放";

    if (
      !state.replay.useJpegFallback &&
      !state.replay.noVideo &&
      !state.replay.hasVideoFrameMap &&
      dom.replayVideo &&
      dom.replayVideo.readyState >= 2 &&
      dom.replayVideo.videoWidth > 0
    ) {
      dom.replayVideo.hidden = false;
      dom.replayFrameImg.hidden = true;
      var stepSec = ((state.replay.data.nominalFrameMs || 40) / 1000);
      try {
        dom.replayVideo.currentTime = state.replay.currentIndex * stepSec;
      } catch (_err) {
        // ignore seek error before play
      }
      dom.replayVideo.play()
        .then(function () {
          if (state.replay.playing && !state.replay.useJpegFallback) {
            state.replay.videoRafId = window.requestAnimationFrame(syncVideoReplayFrameLoop);
          }
        })
        .catch(function () {
          state.replay.useJpegFallback = true;
          startJpegReplayLoop();
        });
      return;
    }
    startJpegReplayLoop();
  }

  function startJpegReplayLoop() {
    state.replay.useJpegFallback = true;
    var frameHasVideo = replayFrameHasVideo(state.replay.currentIndex);
    if (dom.replayVideo) dom.replayVideo.hidden = true;
    if (dom.replayFrameImg) dom.replayFrameImg.hidden = !frameHasVideo;
    if (dom.replaySimCanvas) dom.replaySimCanvas.hidden = frameHasVideo;
    resetReplayClockAnchor(state.replay.currentIndex);
    scheduleNextReplayTick();
  }

  function enterReplayMode(runId, initialFrame) {
    setMonitorMode("replay");
    stopReplayPlayback();
    if (widgets.pipeline && typeof widgets.pipeline.setPaused === "function") {
      widgets.pipeline.setPaused(true);
    }
    var reqToken = ++state.replayReqToken;
    state.replay.runId = runId;
    state.replay.data = null;
    state.replay.framesCount = 0;
    state.replay.currentIndex = 0;
    state.replay.useJpegFallback = true;
    state.replay.noVideo = false;
    state.replay.hasVideoFrameMap = false;
    state.replay.videoFrameSet = null;
    state.replay.isFixture = false;
    if (dom.replaySimCanvas) {
      dom.replaySimCanvas.hidden = true;
    }
    if (dom.replayVideo) {
      dom.replayVideo.onerror = null;
      dom.replayVideo.onloadedmetadata = null;
      dom.replayVideo.hidden = true;
      dom.replayVideo.removeAttribute("src");
    }
    if (dom.replayFrameImg) {
      dom.replayFrameImg.hidden = false;
      if (dom.replayFrameImg.dataset) delete dom.replayFrameImg.dataset.frameIndex;
      dom.replayFrameImg.onerror = function () {
        if (state.mode !== "replay" || state.replay.runId !== runId || state.replay.noVideo) return;
        var failedIdx = dom.replayFrameImg.dataset ? dom.replayFrameImg.dataset.frameIndex : "";
        if (failedIdx && failedIdx !== String(state.replay.currentIndex)) return;
        if (!state.replay.hasVideoFrameMap) {
          state.replay.noVideo = true;
        }
        dom.replayFrameImg.hidden = true;
        dom.replayFrameImg.removeAttribute("src");
        if (dom.replaySimCanvas) {
          dom.replaySimCanvas.hidden = false;
          if (widgets.replayCamera && widgets.roadTwin && typeof widgets.roadTwin.getCurrentRenderState === "function") {
            var camState = Object.assign({}, widgets.roadTwin.getCurrentRenderState(), {
              cameraLabel: state.replay.hasVideoFrameMap ? "SPARSE FRAME · 3D 轨迹重构" : "TRACKS-ONLY · 3D 轨迹重构（无标注 MP4）",
              cameraShortLabel: state.replay.hasVideoFrameMap ? "SPARSE · 3D 重构" : "TRACKS-ONLY · 3D 重构"
            });
            widgets.replayCamera.refreshTheme();
            widgets.replayCamera.renderState(camState);
          }
        }
        if (state.replay.data) {
          dom.replayMediaCaption.textContent = state.replay.hasVideoFrameMap
            ? runId + " · 帧 #" + state.replay.currentIndex + " · 当前帧无视频画面（轨迹重构）"
            : runId + " · 纯轨迹重构（无标注 MP4）";
        }
      };
    }
    setSidebarTarget(runId);
    dom.replayMediaCaption.textContent = "正在加载历史记录 " + runId;

    return Promise.all([
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/summary"),
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/events"),
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/conflicts?limit=5000"),
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/speeds?limit=5000")
    ])
      .then(function (results) {
        if (reqToken !== state.replayReqToken || state.mode !== "replay" || state.replay.runId !== runId) {
          return;
        }
        var summaryObj = results[0] || {};
        var dataset = buildRunDataset(runId, summaryObj, results[1], results[2], results[3]);
        var srcVid = String(summaryObj.source_video || "");
        state.replay.isFixture = srcVid.indexOf("tracking-fixtures") >= 0 || /sample_[ab]\.mp4/i.test(srcVid);
        state.replay.noVideo = summaryObj.has_video === false;
        state.replay.hasVideoFrameMap = Boolean(summaryObj.has_video_frame_map);
        state.replay.videoFrameSet = Array.isArray(summaryObj.video_frame_indices)
          ? new Set(summaryObj.video_frame_indices)
          : null;
        state.replay.data = dataset;
        state.replay.framesCount = dataset.totalFrames;
        applyPipelineConfigChange();
        dom.twinModeLabel.textContent =
          "x[" + dataset.bounds.xMin + "," + dataset.bounds.xMax + "] y[" + dataset.bounds.yMin + "," + dataset.bounds.yMax + "]";

        if (dom.replayVideo && !state.replay.noVideo && !state.replay.hasVideoFrameMap) {
          dom.replayVideo.hidden = true;
          dom.replayVideo.onloadedmetadata = function () {
            if (reqToken !== state.replayReqToken || state.mode !== "replay" || state.replay.runId !== runId) return;
            if (dom.replayVideo.videoWidth > 0) {
              state.replay.useJpegFallback = false;
            }
          };
          dom.replayVideo.onerror = function () {
            state.replay.useJpegFallback = true;
          };
          dom.replayVideo.src = "/api/runs/" + encodeURIComponent(runId) + "/video";
        } else if (dom.replayVideo) {
          dom.replayVideo.hidden = true;
        }

        renderReplayFrame(typeof initialFrame === "number" ? initialFrame : 0);
        if (typeof initialFrame !== "number") {
          startReplayPlayback();
        }
      })
      .catch(function (err) {
        if (reqToken !== state.replayReqToken || state.mode !== "replay" || state.replay.runId !== runId) {
          return;
        }
        dom.replayMediaCaption.textContent = "加载失败: " + err.message;
      });
  }

  function stopLiveJobPolling(options) {
    state.liveJob.pollToken += 1;
    if (state.liveJob.pollTimerId !== null) {
      window.clearTimeout(state.liveJob.pollTimerId);
      state.liveJob.pollTimerId = null;
    }
    if (dom.liveMjpegImg && !(options && options.preserveFrame)) {
      dom.liveMjpegImg.onerror = null;
      dom.liveMjpegImg.removeAttribute("src");
    }
  }

  function enterLiveJobMode(job, submittedConfig) {
    if (!job || !job.job_id) return;
    stopLiveJobPolling();
    var fallbackLimit = parseFloat(dom.jobSpeedLimit ? dom.jobSpeedLimit.value : "");
    var jobLimit = (typeof job.speed_limit_kmh === "number" && Number.isFinite(job.speed_limit_kmh))
      ? job.speed_limit_kmh
      : (Number.isFinite(fallbackLimit) && fallbackLimit > 0 ? fallbackLimit : 30);
    var cfgBase = submittedConfig || state.pipelineConfig || {};
    state.liveJob.config = {
      speedModel: String(job.speed_model || cfgBase.speedModel || "ema"),
      conflictModel: String(job.conflict_model || cfgBase.conflictModel || "cvm_circle"),
      stitching: job.enable_stitching !== undefined ? Boolean(job.enable_stitching) : Boolean(cfgBase.stitching),
      cyclistVru: job.include_cyclist_vru !== undefined ? Boolean(job.include_cyclist_vru) : Boolean(cfgBase.cyclistVru)
    };
    state.liveJob.jobId = job.job_id;
    state.liveJob.status = job.status || "queued";
    state.liveJob.sinceSeq = 0;
    state.liveJob.pollSeq = 0;
    state.liveJob.lastAppliedPollSeq = 0;
    state.liveJob.lastFramesProcessed = -1;
    state.liveJob.lastMaxSpeed = null;
    state.liveJob.lastMinTtc = null;
    state.liveJob.speedLimitKmh = jobLimit;
    state.liveJob.activeAlertsCount = 0;
    state.liveJob.quietPolls = 0;
    state.liveJob.notifiedDone = false;

    setMonitorMode("live");
    applyPipelineConfigChange();
    if (dom.liveMjpegImg) {
      dom.liveMjpegImg.hidden = false;
      dom.liveMjpegImg.onerror = function () {
        if (state.mode !== "live" || state.liveJob.jobId !== job.job_id) return;
        dom.liveMjpegImg.hidden = true;
      };
      dom.liveMjpegImg.src = "/api/jobs/" + encodeURIComponent(job.job_id) + "/mjpeg";
    }
    dom.liveMediaCaption.textContent = "实时流 · " + job.job_id;
    dom.liveJobStatusText.textContent = "状态: " + job.status + " · 0 帧";
    dom.btnStopJob.disabled = false;

    if (widgets.roadTwin) {
      widgets.roadTwin.setReplayFrame({
        frameIndex: 0,
        timestampS: 0,
        vehicles: [],
        vrus: [],
        conflicts: [],
        label: "LIVE · " + job.job_id
      });
    }
    dom.twinCounts.textContent = "车辆 0 · 行人/骑行 0";

    widgets.kpiMaxSpeed.setValue(null);
    widgets.kpiMinTtc.setValue(null);
    widgets.kpiActiveAlerts.setValue(0);
    widgets.kpiFrames.setValue(0);
    dom.kpiFramesMeta.textContent = "等待视频帧解码...";
    if (dom.kpiSpeedMeta) {
      dom.kpiSpeedMeta.textContent = "限速基准 " + formatNumber(jobLimit, 1, "30.0") + " km/h";
    }
    updateRiskPill(0);
    if (widgets.pipeline) {
      widgets.pipeline.setRiskLevel(0);
    }
    updateDualEndMirrors({
      riskLevel: 0,
      timestampS: 0,
      driverAlerts: [],
      pedestrianAlerts: []
    });

    pollLiveJobState();
  }

  function pollLiveJobState() {
    var jobId = state.liveJob.jobId;
    if (!jobId || state.mode !== "live") return Promise.resolve();
    var token = state.liveJob.pollToken;
    var seq = ++state.liveJob.pollSeq;
    var finished = false;

    return Promise.all([
      fetchJson("/api/jobs/" + encodeURIComponent(jobId)),
      fetchJson("/api/jobs/" + encodeURIComponent(jobId) + "/alerts?since=" + state.liveJob.sinceSeq + "&limit=100")
    ])
      .then(function (results) {
        var statusObj = results[0];
        var alertsObj = results[1];
        if (
          !statusObj ||
          token !== state.liveJob.pollToken ||
          seq < state.liveJob.lastAppliedPollSeq ||
          state.mode !== "live" ||
          state.liveJob.jobId !== jobId
        ) {
          return;
        }
        var isDone =
          statusObj.status === "completed" ||
          statusObj.status === "stopped" ||
          statusObj.status === "failed";
        var curFrames = typeof statusObj.frames_processed === "number" ? statusObj.frames_processed : 0;
        if (!isDone && curFrames < state.liveJob.lastFramesProcessed) {
          return;
        }
        state.liveJob.lastAppliedPollSeq = seq;
        state.liveJob.lastFramesProcessed = curFrames;

        state.liveJob.status = statusObj.status;
        widgets.kpiFrames.setValue(curFrames);
        dom.kpiFramesMeta.textContent =
          "最新时间戳 " + formatNumber(statusObj.latest_timestamp_s, 1, "0.0") + " s";
        if (dom.kpiSpeedMeta) {
          var liveLimit = (statusObj.summary && typeof statusObj.summary.speed_limit_kmh === "number")
            ? statusObj.summary.speed_limit_kmh
            : (state.liveJob.speedLimitKmh || 30);
          dom.kpiSpeedMeta.textContent = "限速基准 " + formatNumber(liveLimit, 1, "30.0") + " km/h";
        }

        dom.liveJobStatusText.textContent =
          "作业 " + statusObj.job_id + " · " + statusObj.status + " · " + curFrames + " 帧";

        var rawAlerts = alertsObj && Array.isArray(alertsObj.alerts) ? alertsObj.alerts : [];
        var receivedAlerts = rawAlerts.filter(function (batch) {
          if (typeof batch.seq === "number") {
            return batch.seq > state.liveJob.sinceSeq;
          }
          return true;
        });

        var hasTwinSnap = Boolean(statusObj.latest_twin);
        if (hasTwinSnap) {
          var lv = statusObj.latest_twin;
          if (widgets.roadTwin) {
            widgets.roadTwin.setReplayFrame(lv);
          }
          var vehCnt = Array.isArray(lv.vehicles) ? lv.vehicles.length : 0;
          var vruCnt = Array.isArray(lv.vrus) ? lv.vrus.length : 0;
          dom.twinCounts.textContent = "车辆 " + vehCnt + " · 行人/骑行 " + vruCnt;

          var twinSpeed = (typeof lv.maxSpeedKmh === "number" && Number.isFinite(lv.maxSpeedKmh))
            ? lv.maxSpeedKmh
            : null;
          var twinTtc = (typeof lv.minTtcS === "number" && Number.isFinite(lv.minTtcS))
            ? lv.minTtcS
            : null;
          widgets.kpiMaxSpeed.setValue(twinSpeed);
          widgets.kpiMinTtc.setValue(twinTtc);

          if (receivedAlerts.length === 0) {
            var twinRisk = parseRiskLevel(lv.riskLevel);
            var twinActiveConflicts = Array.isArray(lv.conflicts)
              ? lv.conflicts.filter(function (c) { return parseRiskLevel(c.riskLevel) > 0; }).length
              : 0;
            if (twinRisk === 0) {
              state.liveJob.activeAlertsCount = 0;
              widgets.kpiActiveAlerts.setValue(0);
              updateRiskPill(0);
              if (widgets.pipeline) widgets.pipeline.setRiskLevel(0);
            } else {
              state.liveJob.activeAlertsCount = twinActiveConflicts || state.liveJob.activeAlertsCount;
              widgets.kpiActiveAlerts.setValue(state.liveJob.activeAlertsCount);
              updateRiskPill(twinRisk);
              if (widgets.pipeline) widgets.pipeline.setRiskLevel(twinRisk);
            }
          }
        }

        if (dom.liveMjpegImg && statusObj.has_latest_frame) {
          dom.liveMjpegImg.hidden = false;
        }

        if (receivedAlerts.length > 0) {
          state.liveJob.quietPolls = 0;
          receivedAlerts.forEach(function (batch) {
            if (typeof batch.seq === "number" && batch.seq > state.liveJob.sinceSeq) {
              state.liveJob.sinceSeq = batch.seq;
            }
            var highest = parseRiskLevel(batch.highest_risk_level);
            updateRiskPill(highest);
            if (widgets.pipeline) widgets.pipeline.setRiskLevel(highest);

            var all = (batch.driver_alerts || []).concat(batch.pedestrian_alerts || []);
            var batchMaxSpeed = null;
            var batchMinTtc = null;
            all.forEach(function (a) {
              if (typeof a.veh_speed_kmh === "number" && Number.isFinite(a.veh_speed_kmh)) {
                if (batchMaxSpeed === null || a.veh_speed_kmh > batchMaxSpeed) {
                  batchMaxSpeed = a.veh_speed_kmh;
                }
              }
              if (typeof a.ttc_seconds === "number" && Number.isFinite(a.ttc_seconds)) {
                if (batchMinTtc === null || a.ttc_seconds < batchMinTtc) {
                  batchMinTtc = a.ttc_seconds;
                }
              }
            });

            state.liveJob.lastMaxSpeed = batchMaxSpeed;
            state.liveJob.lastMinTtc = batchMinTtc;
            state.liveJob.activeAlertsCount = batch.alert_count || all.length;
            widgets.kpiActiveAlerts.setValue(state.liveJob.activeAlertsCount);
            if (!hasTwinSnap) {
              widgets.kpiMaxSpeed.setValue(batchMaxSpeed);
              widgets.kpiMinTtc.setValue(batchMinTtc);
            }

            updateDualEndMirrors({
              riskLevel: highest,
              timestampS: batch.timestamp_s,
              driverAlerts: batch.driver_alerts || [],
              pedestrianAlerts: batch.pedestrian_alerts || []
            });
          });
        } else {
          state.liveJob.quietPolls += 1;
          if ((isDone || state.liveJob.quietPolls >= 3) && state.liveJob.activeAlertsCount > 0) {
            state.liveJob.activeAlertsCount = 0;
            widgets.kpiActiveAlerts.setValue(0);
            if (!hasTwinSnap) {
              widgets.kpiMinTtc.setValue(null);
            }
            updateRiskPill(0);
            if (widgets.pipeline) widgets.pipeline.setRiskLevel(0);
            updateDualEndMirrors({
              riskLevel: 0,
              timestampS: statusObj.latest_timestamp_s || 0,
              driverAlerts: [],
              pedestrianAlerts: []
            });
          }
        }

        if (isDone) {
          finished = true;
          if (!hasTwinSnap && statusObj.summary && typeof statusObj.summary.max_observed_speed_kmh === "number") {
            widgets.kpiMaxSpeed.setValue(statusObj.summary.max_observed_speed_kmh);
          }
          if (dom.liveMjpegImg && !statusObj.has_latest_frame) {
            dom.liveMjpegImg.hidden = true;
          }
          dom.btnStopJob.disabled = true;
          if (statusObj.error) {
            dom.liveMediaCaption.textContent = "作业失败: " + statusObj.error;
          } else {
            dom.liveMediaCaption.textContent = "作业已结束 (" + statusObj.status + " · " + curFrames + " 帧)";
          }
          stopLiveJobPolling({ preserveFrame: true });
          if (!state.liveJob.notifiedDone) {
            state.liveJob.notifiedDone = true;
            if (statusObj.status === "completed") {
              showToast(
                "检测任务已完成 (" + curFrames + " 帧)",
                {
                  label: "查看档案",
                  onClick: function () {
                    switchView("history");
                    loadRunsList();
                  }
                }
              );
            } else if (statusObj.status === "failed") {
              showToast("检测任务失败: " + (statusObj.error || "未知错误"));
            } else {
              showToast("检测任务已停止");
            }
          }
          loadRunsList();
        }
      })
      .catch(function (_err) {
        // Keep polling resiliently unless stopped
      })
      .finally(function () {
        if (!finished && token === state.liveJob.pollToken && state.mode === "live" && state.liveJob.jobId === jobId) {
          state.liveJob.pollTimerId = window.setTimeout(pollLiveJobState, 700);
        }
      });
  }

  function showToast(text, action) {
    if (!dom.toastRegion || !text) return;
    while (dom.toastRegion.children.length >= 4) {
      dom.toastRegion.removeChild(dom.toastRegion.firstChild);
    }

    var item = document.createElement("div");
    item.className = "toast-item";

    var msg = document.createElement("span");
    msg.className = "toast-text";
    msg.textContent = text;
    item.appendChild(msg);

    var actions = document.createElement("div");
    actions.className = "toast-actions";

    var removed = false;
    function dismiss() {
      if (removed) return;
      removed = true;
      if (item.parentNode) {
        item.parentNode.removeChild(item);
      }
    }

    if (action && action.label && typeof action.onClick === "function") {
      var actBtn = document.createElement("button");
      actBtn.type = "button";
      actBtn.className = "toast-action-btn";
      actBtn.textContent = action.label;
      actBtn.addEventListener("click", function () {
        dismiss();
        action.onClick();
      });
      actions.appendChild(actBtn);
    }

    var closeBtn = document.createElement("button");
    closeBtn.type = "button";
    closeBtn.className = "toast-close-btn";
    closeBtn.textContent = "关闭";
    closeBtn.addEventListener("click", dismiss);
    actions.appendChild(closeBtn);

    item.appendChild(actions);
    dom.toastRegion.appendChild(item);

    window.setTimeout(dismiss, 4800);
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

  function applySystemInfo(info) {
    var sys = info || {};
    state.systemInfo = sys;

    if (dom.sidebarComputeText) {
      dom.sidebarComputeText.textContent = sys.cuda_available ? "CUDA GPU" : "CPU";
    }

    var showShutdown = Boolean(sys.desktop_mode && sys.allow_shutdown);
    if (dom.btnDesktopShutdown) dom.btnDesktopShutdown.hidden = !showShutdown;
    if (dom.mobileDesktopShutdown) dom.mobileDesktopShutdown.hidden = !showShutdown;

    var canBrowse = Boolean(sys.can_browse_files);
    if (dom.btnCalibBrowseVideo) dom.btnCalibBrowseVideo.hidden = !canBrowse;
    if (dom.btnJobBrowseSource) dom.btnJobBrowseSource.hidden = !canBrowse;
    if (dom.btnJobBrowseCalib) dom.btnJobBrowseCalib.hidden = !canBrowse;

    if (dom.jobWeights) {
      var prev = dom.jobWeights.value;
      dom.jobWeights.textContent = "";
      var defOpt = document.createElement("option");
      defOpt.value = "";
      defOpt.textContent = "默认权重 (自动选择)";
      dom.jobWeights.appendChild(defOpt);

      var weights = Array.isArray(sys.weights) ? sys.weights : [];
      weights.forEach(function (w) {
        var opt = document.createElement("option");
        if (typeof w === "string") {
          opt.value = w;
          opt.textContent = w;
        } else if (w && typeof w === "object") {
          opt.value = w.path || w.label || "";
          opt.textContent = w.label || w.path || "";
        }
        if (opt.value) {
          dom.jobWeights.appendChild(opt);
        }
      });
      if (prev) {
        dom.jobWeights.value = prev;
      }
    }
  }

  function loadSystemInfo() {
    return fetchJson("/api/system/info")
      .then(function (data) {
        applySystemInfo(data);
      })
      .catch(function () {
        applySystemInfo({
          desktop_mode: false,
          allow_shutdown: false,
          can_browse_files: false,
          cuda_available: false,
          weights: []
        });
      });
  }

  function handleDesktopShutdown() {
    if (dom.btnDesktopShutdown) dom.btnDesktopShutdown.disabled = true;
    if (dom.mobileDesktopShutdown) dom.mobileDesktopShutdown.disabled = true;

    fetchJson("/api/system/shutdown", { method: "POST" })
      .then(function () {
        if (dom.serviceHealthDot) dom.serviceHealthDot.setAttribute("data-state", "offline");
        if (dom.serviceHealthText) dom.serviceHealthText.textContent = "服务已停止";
        showToast("桌面服务正在安全关闭...");
        window.setTimeout(function () {
          try {
            window.close();
          } catch (_err) {
            // ignore
          }
        }, 240);
      })
      .catch(function (err) {
        if (dom.btnDesktopShutdown) dom.btnDesktopShutdown.disabled = false;
        if (dom.mobileDesktopShutdown) dom.mobileDesktopShutdown.disabled = false;
        showToast("关闭桌面程序失败: " + err.message);
      });
  }

  function browseNativeFile(kind, onPicked) {
    return fetchJson("/api/dialog/browse", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({ kind: kind })
    })
      .then(function (res) {
        if (res && !res.cancelled && res.path && typeof onPicked === "function") {
          onPicked(res.path);
        }
      })
      .catch(function (err) {
        showToast("打开文件选择器失败: " + err.message);
      });
  }

  function setHistoryActionsDisabled(disabled) {
    var flag = Boolean(disabled);
    [
      dom.btnHistoryReplay,
      dom.btnHistoryCharts,
      dom.btnExportCsv,
      dom.btnExportConflicts,
      dom.btnExportSummary,
      dom.btnHistoryOpenFolder,
      dom.btnHistoryDelete
    ].forEach(function (btn) {
      if (btn) btn.disabled = flag;
    });
  }

  function isPlayableRun(run) {
    return Boolean(
      run &&
      run.status !== "failed" &&
      run.status !== "invalid" &&
      Number(run.frames_processed) > 0
    );
  }

  function syncHistoryReplayButton(runId, summaryObj) {
    if (!dom.btnHistoryReplay) return;
    if (!runId) {
      dom.btnHistoryReplay.disabled = true;
      return;
    }
    var listedRun = state.runs.find(function (r) { return r.run_id === runId; }) || null;
    var candidate = summaryObj ? Object.assign({}, listedRun || {}, summaryObj) : listedRun;
    dom.btnHistoryReplay.disabled = !isPlayableRun(candidate);
  }

  function clearHistoryDisplay() {
    if (dom.histFrames) dom.histFrames.textContent = "-- 帧";
    if (dom.histFps) dom.histFps.textContent = "-- FPS";
    if (dom.histMaxSpeed) dom.histMaxSpeed.textContent = "-- km/h";
    if (dom.histSpeedLimit) dom.histSpeedLimit.textContent = "限速 30.0 km/h";
    if (dom.histCautionCount) dom.histCautionCount.textContent = "0 次";
    if (dom.histSpeedingCount) dom.histSpeedingCount.textContent = "超速告警 0 次";
    if (dom.histCriticalCount) dom.histCriticalCount.textContent = "0 次";
    if (dom.histElapsed) dom.histElapsed.textContent = "耗时 -- s";
    if (dom.historySourceLabel) dom.historySourceLabel.textContent = "";
    if (widgets.speedChart && typeof widgets.speedChart.setData === "function") {
      widgets.speedChart.setData([], 30);
    }
    if (dom.conflictsTbody) {
      dom.conflictsTbody.textContent = "";
      dom.conflictsTbody.appendChild(makeEmptyTableRow("暂无历史分析记录"));
    }
    if (dom.eventsTbody) {
      dom.eventsTbody.textContent = "";
      dom.eventsTbody.appendChild(makeEmptyTableRow("暂无历史分析记录"));
    }
    if (dom.speedsTbody) {
      dom.speedsTbody.textContent = "";
      dom.speedsTbody.appendChild(makeEmptyTableRow("暂无历史分析记录"));
    }
    if (dom.historyTableMeta) {
      dom.historyTableMeta.textContent = "暂无历史分析记录";
    }
  }

  function loadRunsList() {
    return fetchJson("/api/runs")
      .then(function (data) {
        var runs = (data && data.runs) || [];
        state.runs = runs;
        dom.historyRunSelect.textContent = "";

        if (!runs.length) {
          state.historyReqToken += 1;
          var emptyOpt = document.createElement("option");
          emptyOpt.value = "";
          emptyOpt.textContent = "暂无历史分析记录";
          dom.historyRunSelect.appendChild(emptyOpt);
          state.selectedRunId = null;
          state.historyData = null;
          setHistoryActionsDisabled(true);
          if (dom.btnReplaySample) dom.btnReplaySample.hidden = true;
          if (dom.historyChartsCard) dom.historyChartsCard.hidden = true;
          clearHistoryDisplay();
          return;
        }

        var topPlayable = runs.find(isPlayableRun) || null;
        if (dom.btnReplaySample) {
          if (!topPlayable) {
            dom.btnReplaySample.hidden = true;
          } else {
            var topSrc = String(topPlayable.source_video || "");
            var topIsFixture = topSrc.indexOf("tracking-fixtures") >= 0 || /sample_[ab]\.mp4/i.test(topSrc);
            dom.btnReplaySample.hidden = false;
            dom.btnReplaySample.textContent = topIsFixture ? "回放测试固件" : "回放最新记录";
          }
        }

        setHistoryActionsDisabled(false);
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
        syncHistoryReplayButton(preferred, null);
        loadHistoryRunDetail(preferred);
      })
      .catch(function () {
        // Safe offline degradation
      });
  }

  function loadHistoryRunDetail(runId) {
    if (!runId) return;
    state.selectedRunId = runId;
    var reqToken = ++state.historyReqToken;
    syncHistoryReplayButton(runId, null);
    renderHistoryChartsGallery(runId, state.chartsByRun[runId] || []);

    return Promise.all([
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/summary"),
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/events"),
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/conflicts?limit=5000"),
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/speeds?limit=5000"),
      fetchJson("/api/runs/" + encodeURIComponent(runId) + "/timeline?max_points=2000").catch(function () { return null; })
    ])
      .then(function (results) {
        if (reqToken !== state.historyReqToken || state.selectedRunId !== runId) {
          return;
        }
        var dataset = buildRunDataset(runId, results[0], results[1], results[2], results[3]);
        var tl = results[4];
        dataset.timelinePoints = (tl && Array.isArray(tl.points) && tl.points.length > 0) ? tl.points : null;
        state.historyData = dataset;
        syncHistoryReplayButton(runId, results[0]);
        renderHistorySummary(dataset);
        renderHistoryTimeline(dataset);
        renderHistoryTables(dataset);
      })
      .catch(function (err) {
        if (reqToken !== state.historyReqToken || state.selectedRunId !== runId) {
          return;
        }
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
    var limit = (dataset.summary && typeof dataset.summary.speed_limit_kmh === "number")
      ? dataset.summary.speed_limit_kmh
      : 30;
    if (Array.isArray(dataset.timelinePoints) && dataset.timelinePoints.length > 0) {
      var fullSeries = dataset.timelinePoints.map(function (pt) {
        return {
          frameIndex: pt.frame_index,
          t: typeof pt.timestamp_s === "number" ? pt.timestamp_s : pt.frame_index * 0.1,
          speedKmh: typeof pt.speed_kmh === "number" ? pt.speed_kmh : 0,
          ttcS: typeof pt.ttc_s === "number" ? pt.ttc_s : null,
          riskLevel: parseRiskLevel(pt.risk_level)
        };
      });
      widgets.speedChart.setData(fullSeries, limit);
      return;
    }

    var byFrame = {};
    var hasVehicleRows = dataset.speedRows.some(function (row) {
      var catLow = String(row.category || "").toLowerCase();
      return catLow && catLow !== "pedestrian" && catLow !== "person";
    });

    dataset.speedRows.forEach(function (row) {
      var fIdx = parseInt(row.frame_idx, 10);
      if (!Number.isFinite(fIdx)) return;
      var spd = parseFloat(row.speed_kmh) || 0;
      var catLow = String(row.category || "").toLowerCase();
      var isPedOnly = catLow === "pedestrian" || catLow === "person";
      var chartSpeed = (!hasVehicleRows || !isPedOnly) ? spd : 0;
      var t = parseFloat(row.timestamp_s) || fIdx * 0.1;
      var ttc = row.ttc_s !== "" && row.ttc_s != null ? parseFloat(row.ttc_s) : null;
      var risk = parseRiskLevel(row.risk_level);

      if (!byFrame[fIdx]) {
        byFrame[fIdx] = { frameIndex: fIdx, t: t, speedKmh: chartSpeed, ttcS: ttc, riskLevel: risk };
      } else {
        if (chartSpeed > byFrame[fIdx].speedKmh) byFrame[fIdx].speedKmh = chartSpeed;
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

  function makeJumpRowHeader(text, ariaLabel, onJump) {
    var th = document.createElement("th");
    th.setAttribute("scope", "row");
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "table-jump-btn mono tabular";
    btn.textContent = text;
    if (ariaLabel) btn.setAttribute("aria-label", ariaLabel);
    btn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      onJump();
    });
    th.appendChild(btn);
    return th;
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

        function jumpConflict() {
          enterReplayMode(dataset.runId, fIdx);
          switchView("monitor");
        }

        var tr = document.createElement("tr");
        tr.addEventListener("click", jumpConflict);

        tr.appendChild(
          makeJumpRowHeader(
            "#" + fIdx + " · " + formatNumber(rec.timestamp_s, 1, "0.0") + "s",
            "跳转到第 " + fIdx + " 帧冲突回放",
            jumpConflict
          )
        );

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

        var ttcStr = a.ttc_seconds != null ? (formatNumber(a.ttc_seconds, 2, "--") + "s") : "--";
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
        var startF = ev.start_frame || 0;
        function jumpEvent() {
          enterReplayMode(dataset.runId, startF);
          switchView("monitor");
        }

        var tr = document.createElement("tr");
        tr.addEventListener("click", jumpEvent);

        tr.appendChild(
          makeJumpRowHeader(
            "EV-" + ev.event_id,
            "跳转到事件 EV-" + ev.event_id + " 起始帧 #" + startF + " 回放",
            jumpEvent
          )
        );

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

    // 3. Speeds table (cap DOM rows at 500 for smooth table rendering)
    dom.speedsTbody.textContent = "";
    if (!dataset.speedRows.length) {
      dom.speedsTbody.appendChild(makeEmptyTableRow("该记录暂无轨迹测速明细"));
    } else {
      var sFrag = document.createDocumentFragment();
      var displaySpeedRows = dataset.speedRows.slice(0, 500);
      displaySpeedRows.forEach(function (row) {
        var fIdx = parseInt(row.frame_idx, 10) || 0;
        var lvl = parseRiskLevel(row.risk_level);
        function jumpSpeed() {
          enterReplayMode(dataset.runId, fIdx);
          switchView("monitor");
        }

        var tr = document.createElement("tr");
        tr.addEventListener("click", jumpSpeed);

        tr.appendChild(
          makeJumpRowHeader(
            "#" + fIdx + " · T#" + (row.track_id || "--"),
            "跳转到第 " + fIdx + " 帧目标 #" + (row.track_id || "--") + " 回放",
            jumpSpeed
          )
        );
        tr.appendChild(makeCell("td", categoryLabel(row.object_kind || row.category)));
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
      btn.setAttribute("tabindex", active ? "0" : "-1");
    });

    if (state.historyData) {
      var hd = state.historyData;
      var metaMsg = "";
      if (tab === "conflicts") {
        metaMsg = hd.conflictsTruncated
          ? ("已加载前 " + hd.conflicts.length + " 条冲突评估 / 共 " + hd.conflictsTotalCount + " 条（回放滑块按帧区间自动续载）")
          : ("共 " + hd.conflicts.length + " 条记录（可点击或 Tab 聚焦跳转回放）");
      } else if (tab === "events") {
        metaMsg = "共 " + hd.events.length + " 条记录（可点击或 Tab 聚焦跳转回放）";
      } else {
        var shownSpeed = Math.min(hd.speedRows.length, 500);
        if (hd.speedsTruncated) {
          metaMsg = "展示前 " + shownSpeed + " 条 / CSV 总计 " + hd.speedsTotalRows + " 行（回放滑块支持按帧区间按需续载完整轨迹）";
        } else if (hd.speedRows.length > shownSpeed) {
          metaMsg = "表内展示前 " + shownSpeed + " / " + hd.speedRows.length + " 条（可点击或 Tab 聚焦跳转回放）";
        } else {
          metaMsg = "共 " + hd.speedRows.length + " 条记录（可点击或 Tab 聚焦跳转回放）";
        }
      }
      dom.historyTableMeta.textContent = metaMsg;
    }
  }

  function renderHistoryChartsGallery(runId, charts) {
    if (!dom.historyChartsCard || !dom.historyChartsGrid) return;
    var list = Array.isArray(charts) ? charts : [];
    dom.historyChartsGrid.textContent = "";
    if (!list.length) {
      dom.historyChartsCard.hidden = true;
      if (dom.historyChartsMeta) dom.historyChartsMeta.textContent = "0 张图表";
      return;
    }

    dom.historyChartsCard.hidden = false;
    if (dom.historyChartsMeta) dom.historyChartsMeta.textContent = list.length + " 张图表";

    var frag = document.createDocumentFragment();
    list.forEach(function (chart) {
      var name = chart.name || "chart.png";
      var url = chart.url || ("/api/runs/" + encodeURIComponent(runId) + "/charts/" + encodeURIComponent(name));

      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chart-thumb-btn";
      btn.addEventListener("click", function () {
        openChartLightbox(name, url, runId);
      });

      var img = document.createElement("img");
      img.className = "chart-thumb-img";
      img.src = url;
      img.alt = name;

      var cap = document.createElement("span");
      cap.className = "chart-thumb-caption mono";
      cap.textContent = name;

      btn.appendChild(img);
      btn.appendChild(cap);
      frag.appendChild(btn);
    });
    dom.historyChartsGrid.appendChild(frag);
  }

  function openChartLightbox(name, url, runId) {
    if (!dom.chartLightboxDialog) return;
    if (dom.lightboxTitle) dom.lightboxTitle.textContent = name || "实验图表预览";
    if (dom.lightboxSubtitle) dom.lightboxSubtitle.textContent = (runId || "") + " / " + (name || "");
    if (dom.lightboxImg) {
      dom.lightboxImg.src = url;
      dom.lightboxImg.alt = name || "论文图表大图预览";
    }
    if (dom.lightboxDownload) {
      dom.lightboxDownload.href = url;
      dom.lightboxDownload.setAttribute("download", (runId ? runId + "_" : "") + (name || "chart.png"));
    }
    if (!dom.chartLightboxDialog.open) {
      dom.chartLightboxDialog.showModal();
    }
  }

  function generateHistoryCharts() {
    var runId = state.selectedRunId;
    if (!runId || !dom.btnHistoryCharts) return;

    dom.btnHistoryCharts.disabled = true;
    var origLabel = dom.btnHistoryCharts.textContent;
    dom.btnHistoryCharts.textContent = "生成中...";

    fetchJson("/api/runs/" + encodeURIComponent(runId) + "/charts", { method: "POST" })
      .then(function (data) {
        var charts = (data && data.charts) || [];
        state.chartsByRun[runId] = charts;
        renderHistoryChartsGallery(runId, charts);
        showToast("已生成 " + charts.length + " 张论文图表");
      })
      .catch(function (err) {
        showToast("生成图表失败: " + err.message);
      })
      .finally(function () {
        dom.btnHistoryCharts.disabled = !state.selectedRunId;
        dom.btnHistoryCharts.textContent = origLabel;
      });
  }

  function triggerRunExport(filename) {
    var runId = state.selectedRunId;
    if (!runId) return;
    var url = "/api/runs/" + encodeURIComponent(runId) + "/export/" + encodeURIComponent(filename);
    var link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", runId + "_" + filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  function openRunFolder() {
    var runId = state.selectedRunId;
    if (!runId) return;
    fetchJson("/api/runs/" + encodeURIComponent(runId) + "/open-folder", { method: "POST" })
      .then(function (res) {
        showToast("已打开目录: " + ((res && res.path) || runId));
      })
      .catch(function (err) {
        showToast("打开目录失败: " + err.message);
      });
  }

  function openDeleteRunDialog() {
    var runId = state.selectedRunId;
    if (!runId || !dom.confirmDeleteDialog) return;
    state.pendingDeleteRunId = runId;
    if (dom.deleteDialogRunId) dom.deleteDialogRunId.textContent = runId;
    if (dom.confirmDeleteError) {
      dom.confirmDeleteError.hidden = true;
      dom.confirmDeleteError.textContent = "";
    }
    if (!dom.confirmDeleteDialog.open) {
      dom.confirmDeleteDialog.showModal();
    }
  }

  function confirmDeleteRun() {
    var runId = state.pendingDeleteRunId;
    if (!runId || !dom.confirmDeleteSubmit) return;

    dom.confirmDeleteSubmit.disabled = true;
    dom.confirmDeleteSubmit.textContent = "删除中...";
    if (dom.confirmDeleteError) {
      dom.confirmDeleteError.hidden = true;
      dom.confirmDeleteError.textContent = "";
    }

    fetchJson("/api/runs/" + encodeURIComponent(runId), { method: "DELETE" })
      .then(function () {
        if (dom.confirmDeleteDialog && dom.confirmDeleteDialog.open) {
          dom.confirmDeleteDialog.close();
        }
        delete state.chartsByRun[runId];
        if (state.mode === "replay" && state.replay.runId === runId) {
          setMonitorMode("simulation");
        }
        if (state.selectedRunId === runId) {
          state.selectedRunId = null;
          state.historyData = null;
        }
        if (dom.historyChartsCard) dom.historyChartsCard.hidden = true;
        showToast("已删除运行记录 " + runId);
        loadRunsList();
      })
      .catch(function (err) {
        if (dom.confirmDeleteError) {
          dom.confirmDeleteError.textContent = "删除失败: " + err.message;
          dom.confirmDeleteError.hidden = false;
        }
      })
      .finally(function () {
        dom.confirmDeleteSubmit.disabled = false;
        dom.confirmDeleteSubmit.textContent = "确认删除";
      });
  }

  /* ======================================================================
   * View 4: Studio (Visual Road Calibration + Algorithm Lab)
   * ====================================================================== */
  function switchStudioTab(tab) {
    state.studioTab = tab === "lab" ? "lab" : "calib";
    if (dom.studioPanelCalib) dom.studioPanelCalib.hidden = state.studioTab !== "calib";
    if (dom.studioPanelLab) dom.studioPanelLab.hidden = state.studioTab !== "lab";

    if (dom.studioTabCalib) {
      var isCalib = state.studioTab === "calib";
      dom.studioTabCalib.classList.toggle("is-active", isCalib);
      dom.studioTabCalib.setAttribute("aria-selected", isCalib ? "true" : "false");
      dom.studioTabCalib.setAttribute("tabindex", isCalib ? "0" : "-1");
    }
    if (dom.studioTabLab) {
      var isLab = state.studioTab === "lab";
      dom.studioTabLab.classList.toggle("is-active", isLab);
      dom.studioTabLab.setAttribute("aria-selected", isLab ? "true" : "false");
      dom.studioTabLab.setAttribute("tabindex", isLab ? "0" : "-1");
    }

    if (state.studioTab === "calib" && widgets.calibCanvas && typeof widgets.calibCanvas.refreshTheme === "function") {
      window.requestAnimationFrame(function () {
        widgets.calibCanvas.refreshTheme();
      });
    }
  }

  function setCalibTool(tool) {
    state.calibTool = tool === "parking" ? "parking" : "points";
    if (dom.btnCalibToolPoints) dom.btnCalibToolPoints.classList.toggle("is-active", state.calibTool === "points");
    if (dom.btnCalibToolParking) dom.btnCalibToolParking.classList.toggle("is-active", state.calibTool === "parking");
    if (widgets.calibCanvas && typeof widgets.calibCanvas.setTool === "function") {
      widgets.calibCanvas.setTool(state.calibTool);
    }
  }

  function formatCoordNum(val) {
    if (typeof val !== "number" || !Number.isFinite(val)) return "0";
    var rounded = Math.round(val * 10) / 10;
    return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
  }

  function resetCalibValidationBadge() {
    state.calibVersion += 1;
    state.calibPreview = null;
    if (dom.calibValidBadge) {
      dom.calibValidBadge.textContent = "待校验";
      dom.calibValidBadge.className = "meta-tag mono";
    }
  }

  function getCalibDimensions() {
    var rawW = parseFloat(dom.calibWidthM ? dom.calibWidthM.value : "6.0");
    var rawL = parseFloat(dom.calibLengthM ? dom.calibLengthM.value : "15.0");
    return {
      W: Number.isFinite(rawW) && rawW > 0 ? rawW : 6.0,
      L: Number.isFinite(rawL) && rawL > 0 ? rawL : 15.0
    };
  }

  function defaultRectGroundPoints(W, L) {
    return [
      [0, 0],
      [W, 0],
      [W, L],
      [0, L]
    ];
  }

  function guessGroundPointForImagePt(pt, allPts, W, L) {
    var ref = allPts.length >= 4 ? allPts.slice(0, 4) : allPts;
    var minU = Infinity;
    var maxU = -Infinity;
    var minV = Infinity;
    var maxV = -Infinity;
    ref.forEach(function (p) {
      var u = Number(p[0]);
      var v = Number(p[1]);
      if (Number.isFinite(u) && Number.isFinite(v)) {
        if (u < minU) minU = u;
        if (u > maxU) maxU = u;
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    });
    var spanU = maxU - minU;
    var spanV = maxV - minV;
    var gx = spanU > 1
      ? Math.max(0, Math.min(1, (Number(pt[0]) - minU) / spanU)) * W
      : W * 0.5;
    var gy = spanV > 1
      ? Math.max(0, Math.min(1, (Number(pt[1]) - minV) / spanV)) * L
      : L * 0.5;
    if (!Number.isFinite(gx)) gx = W * 0.5;
    if (!Number.isFinite(gy)) gy = L * 0.5;
    return [Math.round(gx * 10) / 10, Math.round(gy * 10) / 10];
  }

  function renderCalibCorrespRows() {
    if (!dom.calibCorrespList) return;
    dom.calibCorrespList.textContent = "";
    var frag = document.createDocumentFragment();
    state.calibGroundPoints.forEach(function (gp, i) {
      var row = document.createElement("div");
      row.className = "calib-corresp-row";

      var tag = document.createElement("span");
      tag.className = "calib-corresp-tag mono tabular";
      tag.textContent = "P" + (i + 1);
      row.appendChild(tag);

      function makeGpField(labelText, axis, initialVal) {
        var lbl = document.createElement("label");
        lbl.className = "calib-corresp-field";
        var spanEl = document.createElement("span");
        spanEl.textContent = labelText;
        var inp = document.createElement("input");
        inp.type = "number";
        inp.step = "0.1";
        inp.className = "form-control tabular";
        inp.setAttribute("data-gp-idx", String(i));
        inp.setAttribute("data-gp-axis", axis);
        inp.value = Number.isFinite(initialVal) ? formatCoordNum(initialVal) : "";
        inp.addEventListener("input", function () {
          state.calibGroundEdited = true;
          if (!Array.isArray(state.calibGroundPoints[i])) {
            state.calibGroundPoints[i] = [0, 0];
          }
          var raw = String(inp.value || "").trim();
          var parsed = raw === "" ? NaN : Number(raw);
          state.calibGroundPoints[i][axis === "y" ? 1 : 0] = Number.isFinite(parsed) ? parsed : NaN;
          resetCalibValidationBadge();
        });
        lbl.appendChild(spanEl);
        lbl.appendChild(inp);
        return lbl;
      }

      row.appendChild(makeGpField("X(m)", "x", gp[0]));
      row.appendChild(makeGpField("Y(m)", "y", gp[1]));
      frag.appendChild(row);
    });
    dom.calibCorrespList.appendChild(frag);
  }

  function syncCalibGroundPoints(pts) {
    var list = Array.isArray(pts) ? pts : [];
    var dims = getCalibDimensions();
    var defaults = defaultRectGroundPoints(dims.W, dims.L);
    var prev = Array.isArray(state.calibGroundPoints) ? state.calibGroundPoints : [];
    var prevLen = prev.length;

    if (list.length === 0) {
      state.calibGroundPoints = [];
      state.calibGroundEdited = false;
      if (dom.calibCorrespWrap) dom.calibCorrespWrap.hidden = true;
      if (dom.calibCorrespList) dom.calibCorrespList.textContent = "";
      return;
    }

    if (list.length <= 4) {
      if (prevLen <= 4 || !state.calibGroundEdited) {
        state.calibGroundPoints = defaults.slice(0, list.length);
        state.calibGroundEdited = false;
      } else {
        state.calibGroundPoints = prev.slice(0, list.length).map(function (p) {
          return [Number(p[0]), Number(p[1])];
        });
      }
      if (dom.calibCorrespWrap) dom.calibCorrespWrap.hidden = true;
      if (dom.calibCorrespList) dom.calibCorrespList.textContent = "";
      return;
    }

    var next = [];
    for (var i = 0; i < list.length; i += 1) {
      if (i < prevLen && Array.isArray(prev[i])) {
        next.push([prev[i][0], prev[i][1]]);
      } else if (i < 4) {
        next.push(defaults[i].slice());
      } else {
        next.push(guessGroundPointForImagePt(list[i], list, dims.W, dims.L));
      }
    }
    state.calibGroundPoints = next;
    if (dom.calibCorrespWrap) dom.calibCorrespWrap.hidden = false;
    var existingRows = dom.calibCorrespList ? dom.calibCorrespList.children.length : 0;
    if (existingRows !== next.length) {
      renderCalibCorrespRows();
    }
  }

  function handleCalibDimChange() {
    resetCalibValidationBadge();
    var dims = getCalibDimensions();
    var oldDefs = defaultRectGroundPoints(state.calibLastRectDims.W, state.calibLastRectDims.L);
    var newDefs = defaultRectGroundPoints(dims.W, dims.L);
    var pts = (state.calibState && Array.isArray(state.calibState.imagePoints)) ? state.calibState.imagePoints : [];
    if (pts.length <= 4) {
      if (!state.calibGroundEdited) {
        state.calibGroundPoints = newDefs.slice(0, pts.length);
      }
    } else {
      for (var i = 0; i < Math.min(4, state.calibGroundPoints.length); i += 1) {
        var cur = state.calibGroundPoints[i];
        if (!cur || (Math.abs(cur[0] - oldDefs[i][0]) < 1e-6 && Math.abs(cur[1] - oldDefs[i][1]) < 1e-6)) {
          state.calibGroundPoints[i] = newDefs[i].slice();
        }
      }
      renderCalibCorrespRows();
    }
    state.calibLastRectDims = { W: dims.W, L: dims.L };
  }

  function handleCalibCanvasChange(snap) {
    if (!snap) return;
    state.calibState = snap;
    var pts = Array.isArray(snap.imagePoints) ? snap.imagePoints : [];
    var zones = Array.isArray(snap.parkingZonesImage) ? snap.parkingZonesImage : [];
    var draft = Array.isArray(snap.currentParkingDraft) ? snap.currentParkingDraft : [];

    syncCalibGroundPoints(pts);
    resetCalibValidationBadge();

    if (dom.calibPointCount) {
      var draftNote = draft.length > 0 ? (" · 禁停区草稿 " + draft.length + " 点") : "";
      dom.calibPointCount.textContent =
        "已选 P" + pts.length + (pts.length >= 4 ? " (满足4+点)" : " · 至少 4 点") +
        " · 禁停区 " + zones.length + " 处" + draftNote;
    }

    if (dom.calibCoordsPill) {
      if (!pts.length && !draft.length) {
        dom.calibCoordsPill.textContent = "在画布点击标定参考点 P1..PN（支持拖拽微调）";
      } else {
        var parts = pts.map(function (pt, i) {
          return "P" + (i + 1) + "(" + formatCoordNum(pt[0]) + ", " + formatCoordNum(pt[1]) + ")";
        });
        if (draft.length > 0) {
          parts.push("禁停草稿(" + draft.length + "点)");
        }
        dom.calibCoordsPill.textContent = parts.join(" · ");
      }
    }
  }

  function showCalibError(message) {
    if (!dom.calibError) return;
    if (!message) {
      dom.calibError.hidden = true;
      dom.calibError.textContent = "";
      return;
    }
    dom.calibError.textContent = message;
    dom.calibError.hidden = false;
  }

  function extractCalibrationFrame() {
    var source = dom.calibSource ? dom.calibSource.value.trim() : "";
    if (!source) {
      showCalibError("请输入或浏览本地视频路径后再提取标定帧");
      if (dom.calibSource) dom.calibSource.focus();
      return Promise.resolve(null);
    }
    var frameIdx = parseInt(dom.calibFrameIndex ? dom.calibFrameIndex.value : "0", 10);
    if (!Number.isFinite(frameIdx) || frameIdx < 0) {
      showCalibError("帧号必须是非负整数");
      return Promise.resolve(null);
    }

    var reqId = ++state.calibExtractReqId;
    showCalibError("");
    if (dom.btnCalibExtract) {
      dom.btnCalibExtract.disabled = true;
      dom.btnCalibExtract.textContent = "提取中...";
    }

    return fetchJson("/api/calibration/frame", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({ source: source, frame_index: frameIdx })
    })
      .then(function (res) {
        var curSource = dom.calibSource ? dom.calibSource.value.trim() : "";
        var curFrame = parseInt(dom.calibFrameIndex ? dom.calibFrameIndex.value : "0", 10);
        if (reqId !== state.calibExtractReqId || curSource !== source || curFrame !== frameIdx) {
          return null;
        }
        if (dom.calibFrameMeta) {
          dom.calibFrameMeta.textContent =
            (res.width || 0) + "x" + (res.height || 0) +
            " · 帧 #" + (res.frame_index != null ? res.frame_index : frameIdx) +
            " / " + (res.frame_count != null ? res.frame_count : "--") +
            " (" + formatNumber(res.fps, 1, "--") + " FPS)";
        }
        var nextW = Number(res.width) || 0;
        var nextH = Number(res.height) || 0;
        var prevW = state.calibImageDims ? state.calibImageDims.w : 0;
        var prevH = state.calibImageDims ? state.calibImageDims.h : 0;
        var currentSnap = (widgets.calibCanvas && typeof widgets.calibCanvas.getState === "function")
          ? widgets.calibCanvas.getState()
          : state.calibState;
        var hasOutOfBounds = false;
        if (nextW > 0 && nextH > 0 && currentSnap) {
          var existingPts = Array.isArray(currentSnap.imagePoints) ? currentSnap.imagePoints : [];
          hasOutOfBounds = existingPts.some(function (pt) {
            return Number(pt[0]) > nextW || Number(pt[1]) > nextH || Number(pt[0]) < 0 || Number(pt[1]) < 0;
          });
        }
        var dimsChanged = (prevW > 0 && prevH > 0 && (prevW !== nextW || prevH !== nextH)) || hasOutOfBounds;
        var sourceChanged = Boolean(state.calibLoadedSource && state.calibLoadedSource !== source);
        state.calibLoadedSource = source;
        state.calibImageDims = { w: nextW, h: nextH };
        if (widgets.calibCanvas && typeof widgets.calibCanvas.loadImage === "function") {
          widgets.calibCanvas.loadImage(res.image_data_url, res.width, res.height);
          if (dimsChanged || sourceChanged) {
            if (typeof widgets.calibCanvas.clearPoints === "function") {
              widgets.calibCanvas.clearPoints();
            }
            if (typeof widgets.calibCanvas.clearParkingZones === "function") {
              widgets.calibCanvas.clearParkingZones();
            }
          }
          resetCalibValidationBadge();
        }
        showToast("已提取标定帧 #" + (res.frame_index != null ? res.frame_index : frameIdx));
        return res;
      })
      .catch(function (err) {
        if (reqId !== state.calibExtractReqId) return null;
        showCalibError("提取视频帧失败: " + err.message);
        return null;
      })
      .finally(function () {
        if (reqId === state.calibExtractReqId && dom.btnCalibExtract) {
          dom.btnCalibExtract.disabled = false;
          dom.btnCalibExtract.textContent = "提取标定帧";
        }
      });
  }

  function buildCalibrationPayload(requireFilename) {
    var snap = (widgets.calibCanvas && typeof widgets.calibCanvas.getState === "function")
      ? widgets.calibCanvas.getState()
      : state.calibState;
    var pts = (snap && Array.isArray(snap.imagePoints)) ? snap.imagePoints : [];
    if (pts.length < 4) {
      throw new Error("请至少在画布上标定 4 个参考点 (当前 " + pts.length + " 个)");
    }

    var widthM = parseFloat(dom.calibWidthM ? dom.calibWidthM.value : "");
    var lengthM = parseFloat(dom.calibLengthM ? dom.calibLengthM.value : "");
    if (!Number.isFinite(widthM) || widthM <= 0 || !Number.isFinite(lengthM) || lengthM <= 0) {
      throw new Error("标定宽度 width_m 与长度 length_m 必须为正数");
    }

    var name = dom.calibName ? dom.calibName.value.trim() : "campus_road";
    if (!name) {
      throw new Error("场景名称不能为空");
    }

    var payload = {
      image_points: pts.map(function (pt) { return [Number(pt[0]), Number(pt[1])]; }),
      width_m: widthM,
      length_m: lengthM,
      name: name,
      use_ransac: Boolean(dom.calibUseRansac && dom.calibUseRansac.checked),
      parking_zones_image: (snap && Array.isArray(snap.parkingZonesImage)) ? snap.parkingZonesImage : []
    };

    if (pts.length >= 5) {
      if (dom.calibCorrespList) {
        var gpInputs = dom.calibCorrespList.querySelectorAll("input[data-gp-idx]");
        Array.prototype.forEach.call(gpInputs, function (inp) {
          var idx = parseInt(inp.getAttribute("data-gp-idx"), 10);
          var axis = inp.getAttribute("data-gp-axis");
          if (Number.isFinite(idx) && idx >= 0 && idx < pts.length) {
            if (!Array.isArray(state.calibGroundPoints[idx])) {
              state.calibGroundPoints[idx] = [NaN, NaN];
            }
            var rawVal = String(inp.value || "").trim();
            var numVal = rawVal === "" ? NaN : Number(rawVal);
            state.calibGroundPoints[idx][axis === "y" ? 1 : 0] = numVal;
          }
        });
      }
      if (!Array.isArray(state.calibGroundPoints) || state.calibGroundPoints.length !== pts.length) {
        syncCalibGroundPoints(pts);
      }
      var groundPoints = [];
      for (var gi = 0; gi < pts.length; gi += 1) {
        var gp = state.calibGroundPoints[gi];
        var gx = gp ? Number(gp[0]) : NaN;
        var gy = gp ? Number(gp[1]) : NaN;
        if (!gp || !Number.isFinite(gx) || !Number.isFinite(gy)) {
          throw new Error("第 P" + (gi + 1) + " 点的地面坐标 (X, Y) 必须为有效数值");
        }
        groundPoints.push([gx, gy]);
      }
      payload.ground_points = groundPoints;
    }

    if (requireFilename) {
      var filename = dom.calibFilename ? dom.calibFilename.value.trim() : "";
      if (!filename || filename.indexOf("/") >= 0 || filename.indexOf("\\") >= 0 || filename.indexOf("..") >= 0) {
        throw new Error("保存文件名必须为单层合法文件名 (如 campus_calibrated.yaml)");
      }
      if (!/\.ya?ml$/i.test(filename)) {
        filename += ".yaml";
        if (dom.calibFilename) dom.calibFilename.value = filename;
      }
      payload.filename = filename;
    }

    return payload;
  }

  function previewCalibration() {
    var payload;
    try {
      payload = buildCalibrationPayload(false);
    } catch (err) {
      showCalibError(err.message);
      return;
    }

    var ver = state.calibVersion;
    var reqId = ++state.calibPreviewReqId;
    showCalibError("");
    if (dom.btnCalibPreview) {
      dom.btnCalibPreview.disabled = true;
      dom.btnCalibPreview.textContent = "求解中...";
    }

    return fetchJson("/api/calibration/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify(payload)
    })
      .then(function (res) {
        if (ver !== state.calibVersion || reqId !== state.calibPreviewReqId) {
          return;
        }
        state.calibPreview = res;
        if (dom.calibValidBadge) {
          dom.calibValidBadge.textContent = res.valid ? "单应矩阵有效" : "校验异常";
          dom.calibValidBadge.className = res.valid ? "status-badge-pass mono" : "status-badge-fail mono";
        }
        if (dom.calibRmseVal) {
          dom.calibRmseVal.textContent = formatNumber(res.reprojection_rmse_m, 4, "--") + " m";
        }
        if (dom.calibInliersVal) {
          var inliers = res.inlier_count != null ? res.inlier_count : payload.image_points.length;
          var zoneCount = Array.isArray(res.parking_zones) ? res.parking_zones.length : 0;
          dom.calibInliersVal.textContent = inliers + " / " + payload.image_points.length + " · 禁停区 " + zoneCount;
        }
      })
      .catch(function (err) {
        if (ver !== state.calibVersion || reqId !== state.calibPreviewReqId) {
          return;
        }
        if (dom.calibValidBadge) {
          dom.calibValidBadge.textContent = "校验失败";
          dom.calibValidBadge.className = "status-badge-fail mono";
        }
        showCalibError("单应矩阵校验失败: " + err.message);
      })
      .finally(function () {
        if (reqId === state.calibPreviewReqId && dom.btnCalibPreview) {
          dom.btnCalibPreview.disabled = false;
          dom.btnCalibPreview.textContent = "校验单应矩阵";
        }
      });
  }

  function saveCalibration() {
    var payload;
    try {
      payload = buildCalibrationPayload(true);
    } catch (err) {
      showCalibError(err.message);
      return;
    }

    showCalibError("");
    if (dom.btnCalibSave) {
      dom.btnCalibSave.disabled = true;
      dom.btnCalibSave.textContent = "保存中...";
    }

    fetchJson("/api/calibration/save", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify(payload)
    })
      .then(function (res) {
        if (dom.calibValidBadge) {
          dom.calibValidBadge.textContent = "已保存实测标定";
          dom.calibValidBadge.className = "status-badge-pass mono";
        }
        if (dom.calibRmseVal && res.reprojection_rmse_m != null) {
          dom.calibRmseVal.textContent = formatNumber(res.reprojection_rmse_m, 4, "--") + " m";
        }
        if (dom.calibSavedVal) {
          dom.calibSavedVal.textContent = res.path || res.label || payload.filename;
        }
        loadPresets().then(function () {
          if (res.path && dom.jobCalib) {
            dom.jobCalib.value = res.path;
          }
        });
        showToast("已保存标定配置: " + (res.label || payload.filename), {
          label: "新建检测任务",
          onClick: openJobDialog
        });
      })
      .catch(function (err) {
        showCalibError("保存标定文件失败: " + err.message);
      })
      .finally(function () {
        if (dom.btnCalibSave) {
          dom.btnCalibSave.disabled = false;
          dom.btnCalibSave.textContent = "保存标定文件";
        }
      });
  }

  function runScenariosRegression() {
    if (!dom.btnRunScenarios) return;
    dom.btnRunScenarios.disabled = true;
    dom.btnRunScenarios.textContent = "运行中...";
    if (dom.scenariosStatusBadge) dom.scenariosStatusBadge.textContent = "执行中";

    fetchJson("/api/tools/scenarios", { method: "POST" })
      .then(function (res) {
        state.scenariosResult = res;
        var total = res && res.total != null ? res.total : 0;
        var passed = res && res.passed != null ? res.passed : 0;
        var allPassed = Boolean(res && res.all_passed);

        if (dom.scenariosPassRatio) {
          dom.scenariosPassRatio.textContent = passed + " / " + total;
        }
        if (dom.scenariosStatusBadge) {
          dom.scenariosStatusBadge.textContent = allPassed ? "全部通过" : "存在失败";
          dom.scenariosStatusBadge.className = allPassed ? "status-badge-pass mono" : "status-badge-fail mono";
        }
        if (dom.scenariosSummaryText) {
          dom.scenariosSummaryText.textContent = allPassed
            ? "全部 " + total + " 项确定性冲突场景回归通过"
            : "共有 " + (total - passed) + " 项场景与期望风险等级不一致";
        }

        if (dom.scenariosTbody) {
          dom.scenariosTbody.textContent = "";
          var rows = (res && Array.isArray(res.results)) ? res.results : [];
          if (!rows.length) {
            var emptyTr = document.createElement("tr");
            emptyTr.className = "empty-row";
            var eth = makeCell("th", "--", "mono", "row");
            var etd = makeCell("td", "未返回场景回归明细");
            etd.colSpan = 4;
            emptyTr.appendChild(eth);
            emptyTr.appendChild(etd);
            dom.scenariosTbody.appendChild(emptyTr);
          } else {
            var frag = document.createDocumentFragment();
            rows.forEach(function (item) {
              var tr = document.createElement("tr");
              tr.appendChild(makeCell("th", item.name || "--", "mono", "row"));
              tr.appendChild(makeCell("td", item.description || "--", "cell-wrap"));
              tr.appendChild(makeCell("td", (item.expected_levels || []).join(", ") || "--", "mono"));
              tr.appendChild(makeCell("td", (item.actual_levels || []).join(", ") || "--", "mono"));

              var statusTd = document.createElement("td");
              var badge = document.createElement("span");
              badge.className = item.passed ? "status-badge-pass mono" : "status-badge-fail mono";
              badge.textContent = item.passed ? "PASS" : "FAIL";
              statusTd.appendChild(badge);
              tr.appendChild(statusTd);
              frag.appendChild(tr);
            });
            dom.scenariosTbody.appendChild(frag);
          }
        }
        showToast("场景回归完成: " + passed + " / " + total + " 通过");
      })
      .catch(function (err) {
        if (dom.scenariosStatusBadge) {
          dom.scenariosStatusBadge.textContent = "运行失败";
          dom.scenariosStatusBadge.className = "status-badge-fail mono";
        }
        showToast("场景回归失败: " + err.message);
      })
      .finally(function () {
        dom.btnRunScenarios.disabled = false;
        dom.btnRunScenarios.textContent = "运行场景回归";
      });
  }

  function runAblationExperiment() {
    if (!dom.btnRunAblation) return;
    dom.btnRunAblation.disabled = true;
    dom.btnRunAblation.textContent = "实验运行中...";
    if (dom.ablationMeta) dom.ablationMeta.textContent = "计算中";

    fetchJson("/api/tools/ablation", { method: "POST" })
      .then(function (res) {
        state.ablationResult = res;
        var turnRows = (res && Array.isArray(res.turn_speed_ablation)) ? res.turn_speed_ablation : [];
        var cmp = (res && res.algorithm_comparison) || {};

        if (dom.ablationMeta) {
          dom.ablationMeta.textContent = "已完成 · 弯道 " + turnRows.length + " 组";
        }

        if (dom.ablationTurnTbody) {
          dom.ablationTurnTbody.textContent = "";
          if (!turnRows.length) {
            dom.ablationTurnTbody.appendChild(makeEmptyTableRow("未返回弯道误差消融数据"));
          } else {
            var tFrag = document.createDocumentFragment();
            turnRows.forEach(function (row) {
              var tr = document.createElement("tr");
              tr.appendChild(makeCell("th", String(row.estimator_mode || "--").toUpperCase(), "mono", "row"));
              tr.appendChild(makeCell("td", formatNumber(row.target_speed_kmh, 1, "--"), "mono tabular"));
              tr.appendChild(makeCell("td", formatNumber(row.turn_rate_rad_s, 2, "--"), "mono tabular"));
              tr.appendChild(makeCell("td", formatNumber(row.mean_abs_error_kmh, 3, "--"), "mono tabular"));
              tr.appendChild(makeCell("td", formatNumber(row.max_abs_error_kmh, 3, "--"), "mono tabular"));
              tr.appendChild(makeCell("td", formatNumber(row.mean_heading_error_deg, 2, "--"), "mono tabular"));
              tr.appendChild(makeCell("td", formatNumber(row.max_heading_error_deg, 2, "--"), "mono tabular"));
              tFrag.appendChild(tr);
            });
            dom.ablationTurnTbody.appendChild(tFrag);
          }
        }

        if (dom.ablationCoreTbody) {
          dom.ablationCoreTbody.textContent = "";
          var coreRows = [];

          if (cmp.speed_estimation) {
            var spd = cmp.speed_estimation;
            var ema = spd.ema || {};
            var kal = spd.kalman || {};
            coreRows.push({
              dim: "透视测速滤波",
              base: "EMA: MAE " + formatNumber(ema.mae_ms, 3, "--") + " m/s",
              prop: "Kalman: MAE " + formatNumber(kal.mae_ms, 3, "--") + " m/s",
              delta: "滞后 " + formatNumber(ema.mean_lag_s, 2, "--") + "s -> " + formatNumber(kal.mean_lag_s, 2, "--") + "s",
              ok: Boolean(spd.kalman_improves_mae),
              badge: spd.kalman_improves_mae ? "误差降低" : "无提升"
            });
          }

          if (Array.isArray(cmp.conflict_detection)) {
            cmp.conflict_detection.forEach(function (c) {
              var cvm = c.cvm_circle || {};
              var obb = c.obb_ssm || {};
              var scen = c.scenario || "--";
              var ok = false;
              var badge = "未达预期";
              if (scen === "parallel_pass") {
                ok = cvm.risk_level !== obb.risk_level && obb.risk_level === "LEVEL_0_SAFE";
                badge = ok ? "误报消除" : "存在误报";
              } else if (scen === "braking_yield") {
                ok = obb.risk_level === "LEVEL_2_CAUTION" || (obb.mttc_s != null && obb.mttc_s !== obb.ttc_s);
                badge = ok ? "礼让降级" : "未识别礼让";
              } else if (scen === "perpendicular_crossing") {
                ok = obb.risk_level === "LEVEL_3_CRITICAL";
                badge = ok ? "准确检出" : "漏报风险";
              } else {
                ok = obb.risk_level !== "LEVEL_0_SAFE";
                badge = ok ? "检出一致" : "异常";
              }
              coreRows.push({
                dim: "OBB-SSM: " + scen,
                base: "CVM: " + (cvm.risk_level || "--") + " (" + formatNumber(cvm.ttc_s, 2, "--") + "s)",
                prop: "OBB: " + (obb.risk_level || "--") + " (" + formatNumber(obb.ttc_s, 2, "--") + "s)",
                delta: "MTTC " + formatNumber(obb.mttc_s, 2, "--") + "s · DRAC " + formatNumber(obb.drac_ms2, 2, "--"),
                ok: ok,
                badge: badge
              });
            });
          }

          if (cmp.tracklet_stitching) {
            var st = cmp.tracklet_stitching;
            var un = st.unstitched || {};
            var sd = st.stitched || {};
            coreRows.push({
              dim: "遮挡断轨缝合",
              base: "未缝合: " + (un.unique_track_ids != null ? un.unique_track_ids : "--") + " ID / 切换 " + (un.id_switches != null ? un.id_switches : "--"),
              prop: "缝合: " + (sd.unique_track_ids != null ? sd.unique_track_ids : "--") + " ID / 缝合 " + (sd.stitched_count != null ? sd.stitched_count : "--"),
              delta: "MAE " + formatNumber(un.speed_mae_ms, 3, "--") + " -> " + formatNumber(sd.speed_mae_ms, 3, "--") + " m/s",
              ok: Boolean(st.stitching_preserved_continuity),
              badge: st.stitching_preserved_continuity ? "连续保持" : "未保持"
            });
          }

          if (!coreRows.length) {
            var emptyCore = document.createElement("tr");
            emptyCore.className = "empty-row";
            var cth = makeCell("th", "--", "mono", "row");
            var ctd = makeCell("td", "未返回核心模块对比数据");
            ctd.colSpan = 4;
            emptyCore.appendChild(cth);
            emptyCore.appendChild(ctd);
            dom.ablationCoreTbody.appendChild(emptyCore);
          } else {
            var cFrag = document.createDocumentFragment();
            coreRows.forEach(function (r) {
              var tr = document.createElement("tr");
              tr.appendChild(makeCell("th", r.dim, "mono", "row"));
              tr.appendChild(makeCell("td", r.base, "mono tabular"));
              tr.appendChild(makeCell("td", r.prop, "mono tabular"));
              tr.appendChild(makeCell("td", r.delta, "mono tabular"));
              var resTd = document.createElement("td");
              var b = document.createElement("span");
              b.className = r.ok ? "status-badge-pass mono" : "status-badge-fail mono";
              b.textContent = r.badge;
              resTd.appendChild(b);
              tr.appendChild(resTd);
              cFrag.appendChild(tr);
            });
            dom.ablationCoreTbody.appendChild(cFrag);
          }
        }
        showToast("全量算法消融实验完成");
      })
      .catch(function (err) {
        if (dom.ablationMeta) dom.ablationMeta.textContent = "运行失败";
        showToast("消融实验失败: " + err.message);
      })
      .finally(function () {
        dom.btnRunAblation.disabled = false;
        dom.btnRunAblation.textContent = "运行全量消融实验";
      });
  }

  function toggleAlertSound() {
    state.soundEnabled = !state.soundEnabled;
    if (widgets.audioSynth && typeof widgets.audioSynth.setEnabled === "function") {
      widgets.audioSynth.setEnabled(state.soundEnabled);
    }
    if (dom.alertSoundLabel) {
      dom.alertSoundLabel.textContent = state.soundEnabled ? "声音提示: 开启" : "声音提示: 关闭";
    }
    if (dom.btnAlertSoundToggle) {
      dom.btnAlertSoundToggle.classList.toggle("btn-sound-active", state.soundEnabled);
      dom.btnAlertSoundToggle.setAttribute("aria-pressed", state.soundEnabled ? "true" : "false");
    }
  }

  function openJobDialog() {
    dom.jobDialogError.hidden = true;
    dom.jobDialogError.textContent = "";
    if (!dom.jobDialog.open) {
      dom.jobDialog.showModal();
    }
    loadPresets();
    loadSystemInfo();
  }

  function selectOrAppendCalibOption(path) {
    if (!dom.jobCalib || !path) return;
    var found = Array.prototype.some.call(dom.jobCalib.options, function (opt) {
      return opt.value === path;
    });
    if (!found) {
      var opt = document.createElement("option");
      opt.value = path;
      opt.textContent = path;
      dom.jobCalib.appendChild(opt);
    }
    dom.jobCalib.value = path;
  }

  function renderVideoPresetChips(container, videos, onSelect) {
    if (!container) return;
    container.textContent = "";
    videos.forEach(function (v) {
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "preset-chip";
      btn.textContent = v.label || v.path;
      btn.addEventListener("click", function () {
        onSelect(v.path);
      });
      container.appendChild(btn);
    });
  }

  function loadPresets() {
    return fetchJson("/api/presets")
      .then(function (data) {
        state.presets = data;
        var videos = (data && data.videos) || [];

        renderVideoPresetChips(dom.presetVideosList, videos, function (path) {
          dom.jobSource.value = path;
          dom.jobSource.dispatchEvent(new Event("input", { bubbles: true }));
          dom.jobSource.focus();
        });

        renderVideoPresetChips(dom.calibPresetVideos, videos, function (path) {
          if (dom.calibSource) {
            dom.calibSource.value = path;
            dom.calibSource.dispatchEvent(new Event("input", { bubbles: true }));
            dom.calibSource.focus();
          }
        });

        var prevCalib = dom.jobCalib.value;
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
        if (prevCalib) {
          selectOrAppendCalibOption(prevCalib);
        }
        if (!dom.jobSource.value && videos.length > 0) {
          dom.jobSource.value = videos[0].path;
        }
        if (dom.calibSource && !dom.calibSource.value && videos.length > 0) {
          dom.calibSource.value = videos[0].path;
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

    if (dom.jobWeights && dom.jobWeights.value) {
      payload.weights = dom.jobWeights.value;
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

    var submittedConfig = {
      speedModel: payload.speed_model,
      conflictModel: payload.conflict_model,
      stitching: payload.enable_stitching,
      cyclistVru: payload.include_cyclist_vru
    };

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
        if (job && typeof job === "object") {
          job.speed_limit_kmh = payload.speed_limit_kmh;
        }
        enterLiveJobMode(job, submittedConfig);
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

  function handleGlobalKeydown(ev) {
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    if (document.querySelector("dialog[open]")) return;

    var target = ev.target;
    var tag = target && target.tagName ? target.tagName.toUpperCase() : "";
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || (target && target.isContentEditable)) {
      return;
    }

    if (ev.key === "1") {
      switchView("monitor");
    } else if (ev.key === "2") {
      switchView("alerts");
    } else if (ev.key === "3") {
      switchView("history");
    } else if (ev.key === "4") {
      switchView("studio");
    } else if (ev.key === "n" || ev.key === "N") {
      ev.preventDefault();
      openJobDialog();
    } else if (ev.code === "Space" || ev.key === " ") {
      if (tag === "BUTTON" || tag === "A" || (target && target.getAttribute && target.getAttribute("role") === "button")) {
        return;
      }
      if (state.activeView === "monitor") {
        ev.preventDefault();
        if (state.mode === "simulation" && dom.btnTwinPause) {
          dom.btnTwinPause.click();
        } else if (state.mode === "replay" && dom.btnReplayToggle) {
          dom.btnReplayToggle.click();
        }
      }
    }
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

    if (dom.btnDesktopShutdown) {
      dom.btnDesktopShutdown.addEventListener("click", handleDesktopShutdown);
    }
    if (dom.mobileDesktopShutdown) {
      dom.mobileDesktopShutdown.addEventListener("click", function () {
        if (dom.mobileNavDialog.open) dom.mobileNavDialog.close();
        handleDesktopShutdown();
      });
    }

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

    if (dom.btnJobBrowseSource) {
      dom.btnJobBrowseSource.addEventListener("click", function () {
        browseNativeFile("video", function (path) {
          dom.jobSource.value = path;
          dom.jobSource.dispatchEvent(new Event("input", { bubbles: true }));
        });
      });
    }
    if (dom.btnJobBrowseCalib) {
      dom.btnJobBrowseCalib.addEventListener("click", function () {
        browseNativeFile("calib", function (path) {
          selectOrAppendCalibOption(path);
        });
      });
    }

    dom.scenarioSelect.addEventListener("change", function () {
      state.scenario = dom.scenarioSelect.value;
      resetAlertLogDedup();
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

    if (dom.btnImportVideo) {
      dom.btnImportVideo.addEventListener("click", openJobDialog);
    }

    dom.btnReplaySample.addEventListener("click", function () {
      var playable = state.runs.find(isPlayableRun);
      if (!playable || !playable.run_id) {
        showToast("暂无可回放的有效分析记录，请先导入视频分析");
        return;
      }
      enterReplayMode(playable.run_id, 0);
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
          dom.liveJobStatusText.textContent =
            "作业 " + job.job_id + " · " + job.status + " · " + (job.frames_processed || 0) + " 帧";
          dom.liveMediaCaption.textContent = "作业已停止 (" + (job.frames_processed || 0) + " 帧)";
          if (dom.liveMjpegImg && !job.has_latest_frame) {
            dom.liveMjpegImg.hidden = true;
          }
          stopLiveJobPolling({ preserveFrame: true });
          loadRunsList();
        })
        .catch(function (err) {
          dom.btnStopJob.disabled = false;
          dom.liveMediaCaption.textContent = "停止失败: " + err.message;
        });
    });

    if (dom.simSpeedModel) {
      dom.simSpeedModel.addEventListener("change", function () {
        if (state.mode !== "simulation") return;
        state.pipelineConfig.speedModel = dom.simSpeedModel.value || "ctrv";
        applyPipelineConfigChange();
      });
    }

    if (dom.simConflictModel) {
      dom.simConflictModel.addEventListener("change", function () {
        if (state.mode !== "simulation") return;
        state.pipelineConfig.conflictModel = dom.simConflictModel.value || "obb_ssm";
        applyPipelineConfigChange();
      });
    }

    dom.toggleStitching.addEventListener("change", function () {
      if (state.mode !== "simulation") return;
      state.pipelineConfig.stitching = Boolean(dom.toggleStitching.checked);
      applyPipelineConfigChange();
    });

    dom.toggleCyclistVru.addEventListener("change", function () {
      if (state.mode !== "simulation") return;
      state.pipelineConfig.cyclistVru = Boolean(dom.toggleCyclistVru.checked);
      applyPipelineConfigChange();
    });

    document.querySelectorAll(".pipeline-node").forEach(function (node) {
      function activateNode() {
        document.querySelectorAll(".pipeline-node").forEach(function (n) {
          var selected = n === node;
          n.classList.toggle("is-selected", selected);
          n.setAttribute("aria-pressed", selected ? "true" : "false");
        });
        state.selectedPipelineStage = node.getAttribute("data-stage") || "detect";
        updatePipelineStageDetail();
      }
      node.addEventListener("click", activateNode);
      node.addEventListener("keydown", function (ev) {
        if (ev.key === "Enter" || ev.key === " ") {
          ev.preventDefault();
          activateNode();
        }
      });
    });

    if (dom.btnAlertSoundToggle) {
      dom.btnAlertSoundToggle.addEventListener("click", toggleAlertSound);
    }

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
      resetAlertLogDedup();
      renderAlertLog();
    });

    dom.historyRunSelect.addEventListener("change", function () {
      loadHistoryRunDetail(dom.historyRunSelect.value);
    });

    dom.btnRefreshRuns.addEventListener("click", function () {
      loadRunsList();
    });

    dom.btnHistoryReplay.addEventListener("click", function () {
      if (!state.selectedRunId) return;
      var selectedRun = state.runs.find(function (r) { return r.run_id === state.selectedRunId; });
      if (selectedRun && !isPlayableRun(selectedRun)) {
        showToast("该历史记录无有效视频帧，无法回放");
        return;
      }
      enterReplayMode(state.selectedRunId, 0);
      switchView("monitor");
    });

    if (dom.btnHistoryCharts) {
      dom.btnHistoryCharts.addEventListener("click", generateHistoryCharts);
    }
    if (dom.btnExportCsv) {
      dom.btnExportCsv.addEventListener("click", function () {
        triggerRunExport("speeds.csv");
      });
    }
    if (dom.btnExportConflicts) {
      dom.btnExportConflicts.addEventListener("click", function () {
        triggerRunExport("conflicts.jsonl");
      });
    }
    if (dom.btnExportSummary) {
      dom.btnExportSummary.addEventListener("click", function () {
        triggerRunExport("summary.json");
      });
    }
    if (dom.btnHistoryOpenFolder) {
      dom.btnHistoryOpenFolder.addEventListener("click", openRunFolder);
    }
    if (dom.btnHistoryDelete) {
      dom.btnHistoryDelete.addEventListener("click", openDeleteRunDialog);
    }

    $("confirm-delete-close").addEventListener("click", function () {
      if (dom.confirmDeleteDialog.open) dom.confirmDeleteDialog.close();
    });
    dom.confirmDeleteCancel.addEventListener("click", function () {
      if (dom.confirmDeleteDialog.open) dom.confirmDeleteDialog.close();
    });
    dom.confirmDeleteSubmit.addEventListener("click", confirmDeleteRun);

    $("lightbox-close").addEventListener("click", function () {
      if (dom.chartLightboxDialog.open) dom.chartLightboxDialog.close();
    });
    $("lightbox-done").addEventListener("click", function () {
      if (dom.chartLightboxDialog.open) dom.chartLightboxDialog.close();
    });

    function bindTablistKeyboard(selector, onSelectBtn) {
      var btns = Array.prototype.slice.call(document.querySelectorAll(selector));
      btns.forEach(function (btn, idx) {
        btn.addEventListener("click", function () {
          onSelectBtn(btn);
        });
        btn.addEventListener("keydown", function (ev) {
          var nextIdx = idx;
          if (ev.key === "ArrowRight") {
            nextIdx = (idx + 1) % btns.length;
          } else if (ev.key === "ArrowLeft") {
            nextIdx = (idx - 1 + btns.length) % btns.length;
          } else if (ev.key === "Home") {
            nextIdx = 0;
          } else if (ev.key === "End") {
            nextIdx = btns.length - 1;
          } else {
            return;
          }
          ev.preventDefault();
          var targetBtn = btns[nextIdx];
          if (targetBtn) {
            onSelectBtn(targetBtn);
            targetBtn.focus();
          }
        });
      });
    }

    bindTablistKeyboard("[data-table-tab]", function (btn) {
      state.historyTableTab = btn.dataset.tableTab || "conflicts";
      updateHistoryTableVisibility();
    });

    bindTablistKeyboard("[data-studio-tab]", function (btn) {
      switchStudioTab(btn.dataset.studioTab || "calib");
    });

    if (dom.btnCalibBrowseVideo) {
      dom.btnCalibBrowseVideo.addEventListener("click", function () {
        browseNativeFile("video", function (path) {
          dom.calibSource.value = path;
          dom.calibSource.dispatchEvent(new Event("input", { bubbles: true }));
        });
      });
    }
    if (dom.calibSource) {
      var onCalibSourceInput = function () {
        state.calibExtractReqId += 1;
        if (dom.btnCalibExtract) {
          dom.btnCalibExtract.disabled = false;
          dom.btnCalibExtract.textContent = "提取标定帧";
        }
        resetCalibValidationBadge();
        var nextSrc = dom.calibSource.value.trim();
        if (state.calibLoadedSource && state.calibLoadedSource !== nextSrc && widgets.calibCanvas) {
          if (typeof widgets.calibCanvas.clearPoints === "function") {
            widgets.calibCanvas.clearPoints();
          }
          if (typeof widgets.calibCanvas.clearParkingZones === "function") {
            widgets.calibCanvas.clearParkingZones();
          }
        }
      };
      dom.calibSource.addEventListener("input", onCalibSourceInput);
      dom.calibSource.addEventListener("change", onCalibSourceInput);
    }
    if (dom.calibFrameIndex) {
      var onCalibFrameChange = function () {
        state.calibExtractReqId += 1;
        if (dom.btnCalibExtract) {
          dom.btnCalibExtract.disabled = false;
          dom.btnCalibExtract.textContent = "提取标定帧";
        }
      };
      dom.calibFrameIndex.addEventListener("input", onCalibFrameChange);
      dom.calibFrameIndex.addEventListener("change", onCalibFrameChange);
    }
    if (dom.btnCalibExtract) {
      dom.btnCalibExtract.addEventListener("click", extractCalibrationFrame);
    }
    if (dom.btnCalibToolPoints) {
      dom.btnCalibToolPoints.addEventListener("click", function () {
        setCalibTool("points");
      });
    }
    if (dom.btnCalibToolParking) {
      dom.btnCalibToolParking.addEventListener("click", function () {
        setCalibTool("parking");
      });
    }
    if (dom.btnCalibFinishParking) {
      dom.btnCalibFinishParking.addEventListener("click", function () {
        if (widgets.calibCanvas && typeof widgets.calibCanvas.finishParkingZone === "function") {
          var ok = widgets.calibCanvas.finishParkingZone();
          if (!ok) {
            showCalibError("请先在“绘制禁停区”模式下至少点击 3 个顶点后再完成闭合");
          } else {
            showCalibError("");
            showToast("已完成当前禁停标定区闭合");
          }
        }
      });
    }
    if (dom.btnCalibUndo) {
      dom.btnCalibUndo.addEventListener("click", function () {
        if (widgets.calibCanvas && typeof widgets.calibCanvas.undo === "function") {
          widgets.calibCanvas.undo();
        }
      });
    }
    if (dom.btnCalibClear) {
      dom.btnCalibClear.addEventListener("click", function () {
        if (!widgets.calibCanvas) return;
        if (state.calibTool === "parking" && typeof widgets.calibCanvas.clearParkingZones === "function") {
          widgets.calibCanvas.clearParkingZones();
        } else if (typeof widgets.calibCanvas.clearPoints === "function") {
          widgets.calibCanvas.clearPoints();
        }
      });
    }
    if (dom.calibWidthM) {
      dom.calibWidthM.addEventListener("input", handleCalibDimChange);
      dom.calibWidthM.addEventListener("change", handleCalibDimChange);
    }
    if (dom.calibLengthM) {
      dom.calibLengthM.addEventListener("input", handleCalibDimChange);
      dom.calibLengthM.addEventListener("change", handleCalibDimChange);
    }
    if (dom.calibUseRansac) {
      dom.calibUseRansac.addEventListener("change", resetCalibValidationBadge);
    }
    if (dom.btnCalibPreview) {
      dom.btnCalibPreview.addEventListener("click", previewCalibration);
    }
    if (dom.btnCalibSave) {
      dom.btnCalibSave.addEventListener("click", saveCalibration);
    }

    if (dom.btnRunScenarios) {
      dom.btnRunScenarios.addEventListener("click", runScenariosRegression);
    }
    if (dom.btnRunAblation) {
      dom.btnRunAblation.addEventListener("click", runAblationExperiment);
    }

    document.addEventListener("keydown", handleGlobalKeydown);
    document.addEventListener("visibilitychange", function () {
      if (document.visibilityState === "visible" && state.replay.playing && state.replay.data) {
        resetReplayClockAnchor(state.replay.currentIndex);
      }
    });
  }

  function init() {
    initDomRefs();
    initTheme();
    document.querySelectorAll('dialog[closedby="any"]').forEach(setupDialog);
    initWidgets();
    syncPipelineControls();
    updateAlgorithmBadges();
    if (dom.btnAlertSoundToggle) {
      dom.btnAlertSoundToggle.setAttribute("aria-pressed", state.soundEnabled ? "true" : "false");
    }
    bindEvents();
    switchView("monitor");
    setMonitorMode("simulation");
    switchStudioTab("calib");
    setCalibTool("points");
    updatePipelineStageDetail();
    loadHealth();
    loadSystemInfo();
    loadRunsList();
    loadPresets();
  }

  if (typeof window !== "undefined") {
    window.__CampusMotionAppTestHook = {
      state: state,
      dom: dom,
      widgets: widgets,
      initDomRefs: initDomRefs,
      getActivePipelineConfig: getActivePipelineConfig,
      updateAlgorithmBadges: updateAlgorithmBadges,
      setMonitorMode: setMonitorMode,
      buildRunDataset: buildRunDataset,
      isReplayFrameLoaded: isReplayFrameLoaded,
      ensureReplayFrameWindow: ensureReplayFrameWindow,
      buildReplayFramePayload: buildReplayFramePayload,
      renderReplayFrame: renderReplayFrame,
      enterReplayMode: enterReplayMode,
      startReplayPlayback: startReplayPlayback,
      stopReplayPlayback: stopReplayPlayback,
      nextFrameDeltaMs: nextFrameDeltaMs,
      replayFrameHasVideo: replayFrameHasVideo,
      loadHistoryRunDetail: loadHistoryRunDetail,
      renderHistoryTimeline: renderHistoryTimeline,
      enterLiveJobMode: enterLiveJobMode,
      pollLiveJobState: pollLiveJobState,
      stopLiveJobPolling: stopLiveJobPolling,
      resetCalibValidationBadge: resetCalibValidationBadge,
      extractCalibrationFrame: extractCalibrationFrame,
      previewCalibration: previewCalibration,
      renderHistoryTables: renderHistoryTables
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
