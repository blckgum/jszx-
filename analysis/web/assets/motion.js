/* Campus Conflict Perception Console - Motion & Visualization Engine
 * Pure classic browser script (no build step, no external dependencies).
 * Exports window.CampusMotion = {
 *   RollingCounter,
 *   AnimatedSignalPipeline,
 *   CampusRoadTwinCanvas,
 *   SpeedTimelineChart,
 *   CalibrationCanvas,
 *   AlertAudioSynth
 * }
 */
(function (global) {
    "use strict";

    var RISK_NAMES = [
        "LEVEL_0_SAFE",
        "LEVEL_1_SPEEDING",
        "LEVEL_2_CAUTION",
        "LEVEL_3_CRITICAL"
    ];

    var VEHICLE_DIMS = {
        car: { lengthM: 4.5, widthM: 1.9 },
        suv: { lengthM: 4.7, widthM: 1.95 },
        motor_vehicle: { lengthM: 4.5, widthM: 1.9 },
        vehicle: { lengthM: 4.5, widthM: 1.9 },
        bus: { lengthM: 9.6, widthM: 2.5 },
        truck: { lengthM: 7.6, widthM: 2.45 },
        motorcycle: { lengthM: 2.1, widthM: 0.8 },
        bicycle: { lengthM: 1.8, widthM: 0.7 },
        pedestrian: { lengthM: 0.7, widthM: 0.7 },
        person: { lengthM: 0.7, widthM: 0.7 }
    };

    function clamp(val, min, max) {
        return val < min ? min : (val > max ? max : val);
    }

    function isFiniteNum(v) {
        return typeof v === "number" && Number.isFinite(v);
    }

    function toNum(v, fallback) {
        if (v === null || v === undefined || v === "" || typeof v === "boolean") {
            return fallback;
        }
        var n = Number(v);
        return Number.isFinite(n) ? n : fallback;
    }

    function parseRiskLevel(raw) {
        if (typeof raw === "number" && Number.isFinite(raw)) {
            return clamp(Math.round(raw), 0, 3);
        }
        if (typeof raw === "string") {
            var s = raw.trim().toUpperCase();
            if (s === "LEVEL_3_CRITICAL" || s === "CRITICAL" || s === "3") return 3;
            if (s === "LEVEL_2_CAUTION" || s === "CAUTION" || s === "2") return 2;
            if (s === "LEVEL_1_SPEEDING" || s === "SPEEDING" || s === "1") return 1;
            if (s === "LEVEL_0_SAFE" || s === "SAFE" || s === "0") return 0;
            var n = Number(s);
            if (Number.isFinite(n)) return clamp(Math.round(n), 0, 3);
        }
        return 0;
    }

    /* ----------------------------------------------------------------------
     * Reduced-motion media tracker (WCAG 2.2.2 / 2.3.1)
     * ---------------------------------------------------------------------- */
    function createReducedMotionTracker(onChange) {
        var mql = (typeof global.matchMedia === "function")
            ? global.matchMedia("(prefers-reduced-motion: reduce)")
            : null;
        var reduced = mql ? Boolean(mql.matches) : false;

        function handleChange(ev) {
            reduced = Boolean(ev && ev.matches);
            if (typeof onChange === "function") {
                onChange(reduced);
            }
        }

        if (mql) {
            if (typeof mql.addEventListener === "function") {
                mql.addEventListener("change", handleChange);
            } else if (typeof mql.addListener === "function") {
                mql.addListener(handleChange);
            }
        }

        return {
            isReduced: function () {
                return reduced;
            },
            destroy: function () {
                if (!mql) return;
                if (typeof mql.removeEventListener === "function") {
                    mql.removeEventListener("change", handleChange);
                } else if (typeof mql.removeListener === "function") {
                    mql.removeListener(handleChange);
                }
            }
        };
    }

    /* ----------------------------------------------------------------------
     * Dynamic CSS custom property color reader & light-dark() evaluator
     * ---------------------------------------------------------------------- */
    function createColorProbe() {
        if (typeof document === "undefined" || !document.documentElement) {
            return null;
        }
        var el = document.createElement("span");
        el.setAttribute("aria-hidden", "true");
        el.style.position = "absolute";
        el.style.width = "0";
        el.style.height = "0";
        el.style.overflow = "hidden";
        el.style.pointerEvents = "none";
        el.style.visibility = "hidden";
        (document.body || document.documentElement).appendChild(el);
        return el;
    }

    function resolveCssColor(probe, rootStyle, varNames, fallback) {
        if (!rootStyle) return fallback;
        for (var i = 0; i < varNames.length; i += 1) {
            var raw = rootStyle.getPropertyValue(varNames[i]);
            if (!raw) continue;
            var trimmed = raw.trim();
            if (!trimmed) continue;
            if (probe && typeof global.getComputedStyle === "function") {
                // Evaluate CSS expressions such as light-dark(...) or color-mix(...)
                probe.style.color = "";
                probe.style.color = "var(" + varNames[i] + ")";
                if (probe.style.color) {
                    var computed = global.getComputedStyle(probe).color;
                    if (computed) return computed;
                }
            }
            if (trimmed.indexOf("light-dark(") === -1 && trimmed.indexOf("var(") === -1) {
                return trimmed;
            }
        }
        return fallback;
    }

    function parseRgbChannels(colorStr) {
        if (typeof colorStr !== "string") return null;
        var s = colorStr.trim();
        if (s.charAt(0) === "#") {
            var hex = s.slice(1);
            if (hex.length === 3 || hex.length === 4) {
                var r3 = parseInt(hex.charAt(0) + hex.charAt(0), 16);
                var g3 = parseInt(hex.charAt(1) + hex.charAt(1), 16);
                var b3 = parseInt(hex.charAt(2) + hex.charAt(2), 16);
                return Number.isFinite(r3) ? [r3, g3, b3] : null;
            }
            if (hex.length === 6 || hex.length === 8) {
                var r6 = parseInt(hex.slice(0, 2), 16);
                var g6 = parseInt(hex.slice(2, 4), 16);
                var b6 = parseInt(hex.slice(4, 6), 16);
                return Number.isFinite(r6) ? [r6, g6, b6] : null;
            }
        }
        var match = s.match(/rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/i);
        if (match) {
            return [Number(match[1]), Number(match[2]), Number(match[3])];
        }
        return null;
    }

    function withAlpha(colorStr, alpha) {
        var a = clamp(alpha, 0, 1);
        var ch = parseRgbChannels(colorStr);
        if (!ch) return colorStr;
        return "rgba(" + Math.round(ch[0]) + ", " + Math.round(ch[1]) + ", " + Math.round(ch[2]) + ", " + a.toFixed(3) + ")";
    }

    function detectLightTheme(rootStyle) {
        if (typeof document === "undefined" || !document.documentElement) {
            return false;
        }
        var root = document.documentElement;
        var attr = (root.getAttribute("data-theme") || "").toLowerCase();
        if (attr === "light") return true;
        if (attr === "dark") return false;
        if (root.classList && root.classList.contains("light")) return true;
        if (rootStyle && rootStyle.colorScheme) {
            var cs = rootStyle.colorScheme.trim().toLowerCase();
            if (cs === "light") return true;
            if (cs === "dark") return false;
        }
        return false;
    }

    function readThemePalette(probe) {
        var rootStyle = (typeof document !== "undefined" && document.documentElement && typeof global.getComputedStyle === "function")
            ? global.getComputedStyle(document.documentElement)
            : null;
        var isLight = detectLightTheme(rootStyle);

        var d = isLight
            ? {
                bgPage: "#fafafb",
                bgCard: "#ffffff",
                bgRaised: "#f4f5f7",
                roadShoulder: "#eff1f5",
                roadSurface: "#e6e9ef",
                roadMark: "rgba(20, 21, 24, 0.22)",
                roadCurb: "rgba(20, 21, 24, 0.30)",
                gridLine: "rgba(20, 21, 24, 0.08)",
                hairline: "rgba(20, 21, 24, 0.12)",
                textPrimary: "#141518",
                textSecondary: "#4b4e57",
                textMuted: "#6f737d",
                accent: "#b47400",
                safe: "#1d8f5a",
                critical: "#c93136",
                neutralPipe: "#8c929c",
                vehicleNeutral: "#2b3340",
                parkingZone: "#8f6716"
            }
            : {
                bgPage: "#141415",
                bgCard: "#19191b",
                bgRaised: "#232325",
                roadShoulder: "#111214",
                roadSurface: "#17181c",
                roadMark: "rgba(237, 237, 237, 0.20)",
                roadCurb: "rgba(237, 237, 237, 0.26)",
                gridLine: "rgba(237, 237, 237, 0.065)",
                hairline: "rgba(237, 237, 237, 0.10)",
                textPrimary: "#ededed",
                textSecondary: "#a1a1a6",
                textMuted: "#85858b",
                accent: "#e5a00d",
                safe: "#4cc38a",
                critical: "#e5484d",
                neutralPipe: "#5d626d",
                vehicleNeutral: "#8a94a6",
                parkingZone: "#c8962e"
            };

        return {
            isLight: isLight,
            bgPage: resolveCssColor(probe, rootStyle, ["--bg-page", "--bg-canvas", "--surface-page", "--bg"], d.bgPage),
            bgCard: resolveCssColor(probe, rootStyle, ["--bg-card", "--surface-card", "--card-bg", "--bg-surface"], d.bgCard),
            bgRaised: resolveCssColor(probe, rootStyle, ["--bg-raised", "--surface-raised", "--bg-elevated", "--bg-subtle"], d.bgRaised),
            roadShoulder: resolveCssColor(probe, rootStyle, ["--road-shoulder", "--bg-sidebar"], d.roadShoulder),
            roadSurface: resolveCssColor(probe, rootStyle, ["--road-surface", "--bg-card"], d.roadSurface),
            roadMark: d.roadMark,
            roadCurb: d.roadCurb,
            gridLine: d.gridLine,
            hairline: resolveCssColor(probe, rootStyle, ["--border-hairline", "--hairline", "--border-subtle", "--border"], d.hairline),
            textPrimary: resolveCssColor(probe, rootStyle, ["--text-primary", "--text", "--fg-primary", "--ink"], d.textPrimary),
            textSecondary: resolveCssColor(probe, rootStyle, ["--text-secondary", "--fg-secondary"], d.textSecondary),
            textMuted: resolveCssColor(probe, rootStyle, ["--text-muted", "--fg-muted", "--muted"], d.textMuted),
            accent: resolveCssColor(probe, rootStyle, ["--accent-amber", "--accent", "--color-accent", "--amber"], d.accent),
            safe: resolveCssColor(probe, rootStyle, ["--semantic-safe", "--color-safe", "--safe", "--status-safe"], d.safe),
            critical: resolveCssColor(probe, rootStyle, ["--semantic-critical", "--color-critical", "--danger", "--critical", "--status-critical"], d.critical),
            neutralPipe: d.neutralPipe,
            vehicleNeutral: d.vehicleNeutral,
            parkingZone: d.parkingZone
        };
    }

    function observeThemeChanges(onThemeChange) {
        if (typeof MutationObserver !== "function" || typeof document === "undefined" || !document.documentElement) {
            return function () {};
        }
        var obs = new MutationObserver(function () {
            onThemeChange();
        });
        obs.observe(document.documentElement, {
            attributes: true,
            attributeFilter: ["data-theme", "class", "style"]
        });
        return function () {
            obs.disconnect();
        };
    }

    function riskColor(palette, riskLevel) {
        var lvl = parseRiskLevel(riskLevel);
        if (lvl === 3) return palette.critical;
        if (lvl === 2 || lvl === 1) return palette.accent;
        return palette.safe;
    }

    /* ======================================================================
     * 1. RollingCounter (ported from number-flow)
     * ====================================================================== */
    function RollingCounter(el, options) {
        if (!(this instanceof RollingCounter)) {
            return new RollingCounter(el, options);
        }
        if (!el) {
            throw new Error("RollingCounter requires a valid DOM element");
        }
        var opts = options || {};
        this._el = el;
        this._decimals = clamp(toNum(opts.decimals, 1), 0, 6);
        this._unit = typeof opts.unit === "string" ? opts.unit : "";
        this._fallback = typeof opts.fallback === "string" ? opts.fallback : "--";
        this._lastDigits = [];
        this._lastNumText = null;
        this._lastValid = null;
        this._lastUnit = null;
        this._lastUpdateMs = 0;
        this._lastDurationMs = 320;
        this._rafId = null;
        this._destroyed = false;

        var self = this;
        this._motionTracker = createReducedMotionTracker(function (isReduced) {
            self._applyReducedState(isReduced);
        });

        this._el.classList.add("nf-counter");
        this._el.style.display = "inline-flex";
        this._el.style.alignItems = "baseline";
        this._el.style.fontVariantNumeric = "tabular-nums";
        this._el.style.whiteSpace = "nowrap";

        this.setValue(null);
    }

    RollingCounter.prototype._applyReducedState = function (isReduced) {
        var strips = this._el.querySelectorAll(".nf-strip");
        var dur = this._lastDurationMs || 320;
        for (var i = 0; i < strips.length; i += 1) {
            strips[i].style.transition = isReduced
                ? "none"
                : "transform " + dur + "ms cubic-bezier(0.22, 1, 0.36, 1)";
        }
    };

    RollingCounter.prototype._createDigitSlot = function (initialDigit, targetDigit, isReduced) {
        var slot = document.createElement("span");
        slot.className = "nf-slot";
        slot.setAttribute("aria-hidden", "true");
        slot.style.display = "inline-block";
        slot.style.overflow = "hidden";
        slot.style.height = "1.28em";
        slot.style.lineHeight = "1.28";
        slot.style.verticalAlign = "text-bottom";
        slot.style.width = "1ch";
        slot.style.textAlign = "center";

        var strip = document.createElement("span");
        strip.className = "nf-strip";
        strip.style.display = "flex";
        strip.style.flexDirection = "column";
        strip.style.lineHeight = "1.28";
        strip.style.fontVariantNumeric = "tabular-nums";
        strip.style.willChange = isReduced ? "auto" : "transform";

        for (var d = 0; d <= 9; d += 1) {
            var cell = document.createElement("span");
            cell.className = "nf-digit";
            cell.textContent = String(d);
            cell.style.display = "flex";
            cell.style.alignItems = "center";
            cell.style.justifyContent = "center";
            cell.style.height = "1.28em";
            cell.style.lineHeight = "1.28";
            strip.appendChild(cell);
        }

        if (isReduced) {
            strip.style.transition = "none";
            strip.style.transform = "translateY(-" + (targetDigit * 10) + "%)";
        } else {
            strip.style.transition = "none";
            strip.style.transform = "translateY(-" + (initialDigit * 10) + "%)";
        }

        slot.appendChild(strip);
        return { slot: slot, strip: strip, targetDigit: targetDigit };
    };

    RollingCounter.prototype.setValue = function (numOrNull, opts) {
        if (this._destroyed) return;
        var o = opts || {};
        var decimals = o.decimals !== undefined ? clamp(toNum(o.decimals, this._decimals), 0, 6) : this._decimals;
        var unit = o.unit !== undefined ? String(o.unit) : this._unit;
        var fallback = o.fallback !== undefined ? String(o.fallback) : this._fallback;

        this._decimals = decimals;
        this._unit = unit;
        this._fallback = fallback;

        var isValid = numOrNull !== null &&
            numOrNull !== undefined &&
            numOrNull !== "" &&
            typeof numOrNull !== "boolean" &&
            Number.isFinite(Number(numOrNull));

        var numText = isValid ? Number(numOrNull).toFixed(decimals) : fallback;
        if (numText === this._lastNumText && isValid === this._lastValid && unit === this._lastUnit) {
            return;
        }

        var nowMs = typeof performance !== "undefined" ? performance.now() : Date.now();
        var elapsedMs = this._lastUpdateMs > 0 ? (nowMs - this._lastUpdateMs) : 9999;
        this._lastUpdateMs = nowMs;
        var animDurationMs = (decimals === 0 && elapsedMs < 350) ? 180 : 320;
        this._lastDurationMs = animDurationMs;
        this._lastNumText = numText;
        this._lastValid = isValid;
        this._lastUnit = unit;

        var fullAria = isValid && unit ? (numText + " " + unit) : numText;
        this._el.setAttribute("aria-label", fullAria);
        this._el.setAttribute("data-value", numText);

        var isReduced = this._motionTracker.isReduced();
        var prevDigits = this._lastDigits;
        var newDigits = [];
        var digitSlots = [];

        if (this._rafId !== null && typeof global.cancelAnimationFrame === "function") {
            global.cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }

        while (this._el.firstChild) {
            this._el.removeChild(this._el.firstChild);
        }

        if (!isValid) {
            var fbSpan = document.createElement("span");
            fbSpan.className = "nf-char nf-fallback";
            fbSpan.setAttribute("aria-hidden", "true");
            fbSpan.textContent = fallback;
            this._el.appendChild(fbSpan);
            this._lastDigits = [];
            return;
        }

        // Count digit characters first to index right-to-left for smooth carry continuity
        var totalDigits = 0;
        for (var c = 0; c < numText.length; c += 1) {
            var ch = numText.charCodeAt(c);
            if (ch >= 48 && ch <= 57) totalDigits += 1;
        }

        var digitCursor = 0;
        for (var i = 0; i < numText.length; i += 1) {
            var chStr = numText.charAt(i);
            var code = numText.charCodeAt(i);
            if (code >= 48 && code <= 57) {
                var digit = code - 48;
                var fromRight = totalDigits - 1 - digitCursor;
                var prevIdx = prevDigits.length - 1 - fromRight;
                var startDigit = (prevIdx >= 0 && prevIdx < prevDigits.length) ? prevDigits[prevIdx] : digit;
                var created = this._createDigitSlot(startDigit, digit, isReduced);
                this._el.appendChild(created.slot);
                digitSlots.push(created);
                newDigits.push(digit);
                digitCursor += 1;
            } else {
                var sep = document.createElement("span");
                sep.className = "nf-char";
                sep.setAttribute("aria-hidden", "true");
                sep.textContent = chStr;
                sep.style.display = "inline-block";
                sep.style.lineHeight = "1.28";
                sep.style.verticalAlign = "text-bottom";
                this._el.appendChild(sep);
            }
        }

        if (unit) {
            var unitSpan = document.createElement("span");
            unitSpan.className = "nf-unit";
            unitSpan.setAttribute("aria-hidden", "true");
            unitSpan.textContent = unit;
            unitSpan.style.lineHeight = "1.28";
            this._el.appendChild(unitSpan);
        }

        this._lastDigits = newDigits;

        if (!isReduced && digitSlots.length > 0 && typeof global.requestAnimationFrame === "function") {
            var self = this;
            this._rafId = global.requestAnimationFrame(function () {
                self._rafId = null;
                if (self._destroyed) return;
                for (var k = 0; k < digitSlots.length; k += 1) {
                    var item = digitSlots[k];
                    item.strip.style.transition = "transform " + animDurationMs + "ms cubic-bezier(0.22, 1, 0.36, 1)";
                    item.strip.style.transform = "translateY(-" + (item.targetDigit * 10) + "%)";
                }
            });
        }
    };

    RollingCounter.prototype.destroy = function () {
        if (this._destroyed) return;
        this._destroyed = true;
        if (this._rafId !== null && typeof global.cancelAnimationFrame === "function") {
            global.cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }
        this._motionTracker.destroy();
    };

    /* ======================================================================
     * 2. AnimatedSignalPipeline (ported from Magic UI AnimatedBeam)
     * ====================================================================== */
    function AnimatedSignalPipeline(svgEl, options) {
        if (!(this instanceof AnimatedSignalPipeline)) {
            return new AnimatedSignalPipeline(svgEl, options);
        }
        if (!svgEl) {
            throw new Error("AnimatedSignalPipeline requires a valid <svg> element");
        }
        var opts = options || {};
        this._svg = svgEl;
        this._onPulse = typeof opts.onPulse === "function" ? opts.onPulse : null;
        this._paused = false;
        this._riskLevel = 0;
        this._config = {
            speedModel: "ctrv",
            conflictModel: "obb_ssm",
            stitching: true,
            cyclistVru: true
        };
        this._destroyed = false;
        this._rafId = null;
        this._lastFrameTs = null;
        this._phase = 0;
        this._lastCycleIdx = -1;

        this._probe = createColorProbe();
        this._palette = readThemePalette(this._probe);

        var self = this;
        this._motionTracker = createReducedMotionTracker(function () {
            self._syncAnimationLoop();
        });
        this._unobserveTheme = observeThemeChanges(function () {
            self._palette = readThemePalette(self._probe);
            self._applyColors();
        });

        this._ensureSvgStructure();
        this._measurePaths();

        if (typeof ResizeObserver === "function") {
            this._resizeObserver = new ResizeObserver(function () {
                self._measurePaths();
            });
            this._resizeObserver.observe(this._svg);
        } else {
            this._resizeObserver = null;
        }

        this.setRiskLevel(0);
        this.setConfig(this._config);
        this._syncAnimationLoop();
    }

    AnimatedSignalPipeline.prototype._ensureSvgStructure = function () {
        var existingPaths = this._svg.querySelectorAll(".beam-path");
        var ns = "http://www.w3.org/2000/svg";

        if (existingPaths.length === 0) {
            // Synthesize clean responsive 5-node signal flow if host SVG is empty
            this._svg.setAttribute("viewBox", "0 0 920 120");
            var nodes = [
                { id: "detect", x: 92, label: "YOLOv8", sub: "Target Det" },
                { id: "track", x: 276, label: "ByteTrack+HSV", sub: "Appearance" },
                { id: "motion", x: 460, label: "Homography/CTRV", sub: "Kinematics" },
                { id: "conflict", x: 644, label: "OBB-SSM", sub: "TTC / PET" },
                { id: "alert", x: 828, label: "Dual-End Alert", sub: "Driver / VRU" }
            ];
            for (var s = 0; s < nodes.length - 1; s += 1) {
                var x1 = nodes[s].x + 68;
                var x2 = nodes[s + 1].x - 68;
                var dStr = "M " + x1 + " 60 L " + x2 + " 60";
                var basePath = document.createElementNS(ns, "path");
                basePath.setAttribute("class", "beam-path");
                basePath.setAttribute("data-segment", String(s));
                basePath.setAttribute("d", dStr);
                basePath.setAttribute("fill", "none");
                basePath.setAttribute("stroke-width", "2");
                this._svg.appendChild(basePath);

                var pulsePath = document.createElementNS(ns, "path");
                pulsePath.setAttribute("class", "beam-pulse");
                pulsePath.setAttribute("data-segment", String(s));
                pulsePath.setAttribute("d", dStr);
                pulsePath.setAttribute("fill", "none");
                pulsePath.setAttribute("stroke-width", "2.5");
                pulsePath.setAttribute("stroke-linecap", "round");
                this._svg.appendChild(pulsePath);
            }
            for (var n = 0; n < nodes.length; n += 1) {
                var g = document.createElementNS(ns, "g");
                g.setAttribute("data-stage", nodes[n].id);
                var rect = document.createElementNS(ns, "rect");
                rect.setAttribute("x", String(nodes[n].x - 68));
                rect.setAttribute("y", "32");
                rect.setAttribute("width", "136");
                rect.setAttribute("height", "56");
                rect.setAttribute("rx", "8");
                rect.setAttribute("fill", "var(--bg-raised, #232325)");
                rect.setAttribute("stroke", "var(--border-hairline, rgba(237,237,237,0.12))");
                g.appendChild(rect);

                var tMain = document.createElementNS(ns, "text");
                tMain.setAttribute("x", String(nodes[n].x));
                tMain.setAttribute("y", "56");
                tMain.setAttribute("text-anchor", "middle");
                tMain.setAttribute("fill", "var(--text-primary, #ededed)");
                tMain.setAttribute("font-size", "12");
                tMain.setAttribute("data-stage-label", nodes[n].id);
                tMain.textContent = nodes[n].label;
                g.appendChild(tMain);

                var tSub = document.createElementNS(ns, "text");
                tSub.setAttribute("x", String(nodes[n].x));
                tSub.setAttribute("y", "74");
                tSub.setAttribute("text-anchor", "middle");
                tSub.setAttribute("fill", "var(--text-muted, #85858b)");
                tSub.setAttribute("font-size", "10");
                tSub.setAttribute("data-stage-sub", nodes[n].id);
                tSub.textContent = nodes[n].sub;
                g.appendChild(tSub);

                this._svg.appendChild(g);
            }
        } else if (this._svg.querySelectorAll(".beam-pulse").length === 0) {
            // Auto-create overlay .beam-pulse paths matching existing .beam-path tracks
            for (var i = 0; i < existingPaths.length; i += 1) {
                var src = existingPaths[i];
                var clone = document.createElementNS(ns, "path");
                clone.setAttribute("class", "beam-pulse");
                clone.setAttribute("d", src.getAttribute("d") || "");
                clone.setAttribute("fill", "none");
                clone.setAttribute("stroke-linecap", "round");
                clone.setAttribute("stroke-width", src.getAttribute("stroke-width") || "2.5");
                if (src.hasAttribute("data-segment")) {
                    clone.setAttribute("data-segment", src.getAttribute("data-segment"));
                }
                if (src.parentNode) {
                    src.parentNode.insertBefore(clone, src.nextSibling);
                }
            }
        }

        this._basePaths = Array.prototype.slice.call(this._svg.querySelectorAll(".beam-path"));
        this._pulsePaths = Array.prototype.slice.call(this._svg.querySelectorAll(".beam-pulse"));
    };

    AnimatedSignalPipeline.prototype._measurePaths = function () {
        this._segments = [];
        for (var i = 0; i < this._pulsePaths.length; i += 1) {
            var p = this._pulsePaths[i];
            var len = 120;
            try {
                var measured = typeof p.getTotalLength === "function" ? p.getTotalLength() : 0;
                if (Number.isFinite(measured) && measured > 4) {
                    len = measured;
                }
            } catch (err) {
                len = 120;
            }
            var pulseLen = clamp(len * 0.30, 16, 46);
            this._segments.push({
                pulseEl: p,
                baseEl: this._basePaths[i] || null,
                length: len,
                pulseLen: pulseLen,
                index: i
            });
            p.style.fill = "none";
            p.style.strokeLinecap = "round";
        }
        this._applyStaticOrAnimatedPattern();
    };

    AnimatedSignalPipeline.prototype._applyColors = function () {
        var pal = this._palette;
        var alertColor = pal.neutralPipe;
        if (this._riskLevel === 3) alertColor = pal.critical;
        else if (this._riskLevel === 2 || this._riskLevel === 1) alertColor = pal.accent;

        var lastIdx = this._segments.length - 1;
        for (var i = 0; i < this._segments.length; i += 1) {
            var seg = this._segments[i];
            var isAlertSeg = (i === lastIdx) ||
                (seg.pulseEl.getAttribute("data-segment") === "alert") ||
                (seg.pulseEl.getAttribute("data-role") === "alert");
            var pulseColor = isAlertSeg ? alertColor : pal.accent;
            var baseColor = isAlertSeg && this._riskLevel >= 2 ? withAlpha(alertColor, 0.35) : pal.hairline;

            seg.pulseEl.style.stroke = pulseColor;
            if (seg.baseEl) {
                seg.baseEl.style.stroke = baseColor;
            }
        }

        // Update optional SVG alert indicator elements if present without overriding .is-selected CSS
        var alertBadge = this._svg.querySelector("[data-stage='alert'] rect, [data-node='alert'] rect");
        if (alertBadge) {
            alertBadge.style.stroke = this._riskLevel >= 2 ? alertColor : "";
        }
        this._svg.setAttribute("data-risk", String(this._riskLevel));
    };

    AnimatedSignalPipeline.prototype._applyStaticOrAnimatedPattern = function () {
        var isStatic = this._paused || this._motionTracker.isReduced();
        for (var i = 0; i < this._segments.length; i += 1) {
            var seg = this._segments[i];
            if (isStatic) {
                // Static crisp beam with subtle accent opacity and zero dash animation
                seg.pulseEl.style.strokeDasharray = "none";
                seg.pulseEl.style.strokeDashoffset = "0";
                seg.pulseEl.style.opacity = i === this._segments.length - 1
                    ? (this._riskLevel >= 2 ? "0.75" : "0.35")
                    : "0.32";
            } else {
                seg.pulseEl.style.strokeDasharray = seg.pulseLen.toFixed(1) + " " + (seg.length + seg.pulseLen).toFixed(1);
                seg.pulseEl.style.opacity = i === this._segments.length - 1 && this._riskLevel >= 2 ? "0.95" : "0.78";
            }
        }
        this._applyColors();
    };

    AnimatedSignalPipeline.prototype._syncAnimationLoop = function () {
        var shouldRun = !this._destroyed && !this._paused && !this._motionTracker.isReduced();
        this._applyStaticOrAnimatedPattern();

        if (!shouldRun) {
            if (this._rafId !== null && typeof global.cancelAnimationFrame === "function") {
                global.cancelAnimationFrame(this._rafId);
                this._rafId = null;
            }
            this._lastFrameTs = null;
            return;
        }

        if (this._rafId !== null || typeof global.requestAnimationFrame !== "function") {
            return;
        }

        var self = this;
        function tick(now) {
            if (self._destroyed || self._paused || self._motionTracker.isReduced()) {
                self._rafId = null;
                return;
            }
            if (self._lastFrameTs === null) {
                self._lastFrameTs = now;
            }
            var dt = Math.min(0.08, Math.max(0, (now - self._lastFrameTs) / 1000));
            self._lastFrameTs = now;

            // Smooth 0.45 cycles/sec directional signal flow (well below 3 Hz threshold)
            var speedCps = self._riskLevel >= 3 ? 0.58 : 0.42;
            self._phase += dt * speedCps;

            var segCount = Math.max(1, self._segments.length);
            var activeIdx = Math.floor(self._phase * segCount) % segCount;
            if (activeIdx !== self._lastCycleIdx) {
                self._lastCycleIdx = activeIdx;
                if (self._onPulse) {
                    self._onPulse(activeIdx, self._riskLevel);
                }
            }

            for (var i = 0; i < self._segments.length; i += 1) {
                var seg = self._segments[i];
                var localT = (self._phase - i * 0.16) % 1;
                if (localT < 0) localT += 1;
                var totalTravel = seg.length + seg.pulseLen;
                var offset = seg.pulseLen - localT * totalTravel;
                seg.pulseEl.style.strokeDashoffset = offset.toFixed(2);
            }

            self._rafId = global.requestAnimationFrame(tick);
        }

        this._rafId = global.requestAnimationFrame(tick);
    };

    AnimatedSignalPipeline.prototype.setRiskLevel = function (level) {
        if (this._destroyed) return;
        this._riskLevel = parseRiskLevel(level);
        this._applyStaticOrAnimatedPattern();
    };

    AnimatedSignalPipeline.prototype.setConfig = function (cfg) {
        if (this._destroyed) return;
        var c = cfg || {};
        if (typeof c.speedModel === "string") this._config.speedModel = c.speedModel;
        if (typeof c.conflictModel === "string") this._config.conflictModel = c.conflictModel;
        if (typeof c.stitching === "boolean") this._config.stitching = c.stitching;
        if (typeof c.cyclistVru === "boolean") this._config.cyclistVru = c.cyclistVru;

        var trackLabel = this._svg.querySelector("[data-stage-label='track'], [data-node-title='track']");
        if (trackLabel) {
            trackLabel.textContent = this._config.stitching ? "ByteTrack+HSV" : "ByteTrack";
        }
        var motionLabel = this._svg.querySelector("[data-stage-label='motion'], [data-node-title='motion']");
        if (motionLabel) {
            var sm = this._config.speedModel.toLowerCase();
            motionLabel.textContent = sm === "kalman" ? "Homography/Kalman" : (sm === "ema" ? "Homography/EMA" : "Homography/CTRV");
        }
        var conflictLabel = this._svg.querySelector("[data-stage-label='conflict'], [data-node-title='conflict']");
        if (conflictLabel) {
            var cm = this._config.conflictModel.toLowerCase();
            conflictLabel.textContent = cm === "cvm_circle" ? "CVM-Circle" : "OBB-SSM";
        }
        var conflictSub = this._svg.querySelector("[data-stage-sub='conflict'], [data-node-sub='conflict']");
        if (conflictSub) {
            conflictSub.textContent = this._config.cyclistVru ? "TTC / PET + VRU" : "TTC / PET";
        }
    };

    AnimatedSignalPipeline.prototype.setPaused = function (paused) {
        if (this._destroyed) return;
        this._paused = Boolean(paused);
        this._syncAnimationLoop();
    };

    AnimatedSignalPipeline.prototype.destroy = function () {
        if (this._destroyed) return;
        this._destroyed = true;
        if (this._rafId !== null && typeof global.cancelAnimationFrame === "function") {
            global.cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }
        this._unobserveTheme();
        this._motionTracker.destroy();
        if (this._probe && this._probe.parentNode) {
            this._probe.parentNode.removeChild(this._probe);
        }
    };

    /* ======================================================================
     * 3. CampusRoadTwinCanvas (Bird's-Eye Metric Road Twin + Physics Engine)
     * ====================================================================== */
    var DEFAULT_BOUNDS = { xMin: -3.2, xMax: 3.2, yMin: 0.0, yMax: 15.0 };

    function normalizeBounds(b) {
        if (!b) return { xMin: DEFAULT_BOUNDS.xMin, xMax: DEFAULT_BOUNDS.xMax, yMin: DEFAULT_BOUNDS.yMin, yMax: DEFAULT_BOUNDS.yMax };
        if (Array.isArray(b) && b.length >= 4) {
            return {
                xMin: toNum(b[0], DEFAULT_BOUNDS.xMin),
                xMax: toNum(b[1], DEFAULT_BOUNDS.xMax),
                yMin: toNum(b[2], DEFAULT_BOUNDS.yMin),
                yMax: toNum(b[3], DEFAULT_BOUNDS.yMax)
            };
        }
        return {
            xMin: toNum(b.xMin !== undefined ? b.xMin : b.x_min, DEFAULT_BOUNDS.xMin),
            xMax: toNum(b.xMax !== undefined ? b.xMax : b.x_max, DEFAULT_BOUNDS.xMax),
            yMin: toNum(b.yMin !== undefined ? b.yMin : b.y_min, DEFAULT_BOUNDS.yMin),
            yMax: toNum(b.yMax !== undefined ? b.yMax : b.y_max, DEFAULT_BOUNDS.yMax)
        };
    }

    function computeCpaAndTtc(veh, vru, horizonS) {
        var horizon = horizonS || 3.5;
        var dt = 0.05;
        var steps = Math.round(horizon / dt);

        var psi = veh.headingRad;
        var x = veh.x;
        var y = veh.y;
        var v = Math.hypot(veh.vx, veh.vy);
        var a = veh.accelMs2 || 0;
        var omega = veh.yawRate || 0;

        var px = vru.x;
        var py = vru.y;
        var pvx = vru.vx;
        var pvy = vru.vy;

        var halfL = (veh.lengthM || 4.5) * 0.5;
        var halfW = (veh.widthM || 1.9) * 0.5;
        var safeR = vru.radiusM || 0.42;

        var currentDist = Math.hypot(px - x, py - y);
        var minClearance = Infinity;
        var bestX = x;
        var bestY = y;
        var ttc = null;

        // If vehicle is effectively stationary and not overlapping right now, no TTC conflict
        if (v < 0.25 && a <= 0.1) {
            var dx0 = px - x;
            var dy0 = py - y;
            var c0 = Math.cos(psi);
            var s0 = Math.sin(psi);
            var lx0 = dx0 * c0 + dy0 * s0;
            var ly0 = -dx0 * s0 + dy0 * c0;
            var cx0 = clamp(lx0, -halfL, halfL);
            var cy0 = clamp(ly0, -halfW, halfW);
            var clear0 = Math.hypot(lx0 - cx0, ly0 - cy0) - safeR;
            return {
                currentDist: currentDist,
                minDistanceM: Math.max(0, clear0),
                ttcSeconds: clear0 <= 0.05 ? 0 : null,
                conflictGroundXY: [x + cx0 * c0 - cy0 * s0, y + cx0 * s0 + cy0 * c0]
            };
        }

        // Second-order deceleration rollout factor for realistic MTTC/TTC transition
        var effA = a < 0 ? a * 0.72 : a;

        for (var i = 0; i <= steps; i += 1) {
            var t = i * dt;
            var dx = (px + pvx * t) - x;
            var dy = (py + pvy * t) - y;
            var cosP = Math.cos(psi);
            var sinP = Math.sin(psi);
            var lx = dx * cosP + dy * sinP;
            var ly = -dx * sinP + dy * cosP;
            var cx = clamp(lx, -halfL, halfL);
            var cy = clamp(ly, -halfW, halfW);
            var clearance = Math.hypot(lx - cx, ly - cy) - safeR;

            if (clearance < minClearance) {
                minClearance = clearance;
                bestX = x + (cx * cosP - cy * sinP + dx) * 0.5;
                bestY = y + (cx * sinP + cy * cosP + dy) * 0.5;
            }
            if (ttc === null && clearance <= 0.12) {
                ttc = t;
            }

            // Advance 2nd-order CTRV rollout step
            var nextV = Math.max(0, v + effA * dt);
            var avgV = 0.5 * (v + nextV);
            if (Math.abs(omega) > 0.01) {
                var nextPsi = psi + omega * dt;
                x += (avgV / omega) * (Math.sin(nextPsi) - Math.sin(psi));
                y += (avgV / omega) * (-Math.cos(nextPsi) + Math.cos(psi));
                psi = nextPsi;
            } else {
                x += avgV * Math.cos(psi) * dt;
                y += avgV * Math.sin(psi) * dt;
            }
            v = nextV;
        }

        return {
            currentDist: currentDist,
            minDistanceM: Math.max(0, minClearance),
            ttcSeconds: ttc,
            conflictGroundXY: [bestX, bestY]
        };
    }

    function predictCtrvArc(veh, horizonS, steps) {
        var pts = [{ x: veh.x, y: veh.y, t: 0 }];
        var count = steps || 15;
        var dt = (horizonS || 2.5) / count;
        var x = veh.x;
        var y = veh.y;
        var psi = veh.headingRad;
        var v = Math.hypot(veh.vx, veh.vy);
        var a = veh.accelMs2 || 0;
        var omega = veh.yawRate || 0;

        for (var i = 1; i <= count; i += 1) {
            var nextV = Math.max(0, v + a * dt);
            var avgV = 0.5 * (v + nextV);
            if (Math.abs(omega) > 0.01) {
                var nextPsi = psi + omega * dt;
                x += (avgV / omega) * (Math.sin(nextPsi) - Math.sin(psi));
                y += (avgV / omega) * (-Math.cos(nextPsi) + Math.cos(psi));
                psi = nextPsi;
            } else {
                x += avgV * Math.cos(psi) * dt;
                y += avgV * Math.sin(psi) * dt;
            }
            v = nextV;
            pts.push({ x: x, y: y, t: i * dt });
        }
        return pts;
    }

    var SCENARIO_CYCLE_S = 8.0;

    function isCyclistVruKind(item) {
        if (!item) return false;
        var k = String(item.category || item.kind || "").toLowerCase();
        return k === "bicycle" || k === "motorcycle";
    }

    function integrateCtrvState(tSec) {
        function phaseParams(tau) {
            if (tau < 0.50) {
                return { v: 5.4, a: 0.0, omega: 0.28 };
            }
            if (tau < 2.10) {
                var a1 = -5.4 / 1.60;
                var v1 = Math.max(0.0, 5.4 + a1 * (tau - 0.50));
                return { v: v1, a: a1, omega: 0.28 * (v1 / 5.4) };
            }
            if (tau < 4.95) {
                return { v: 0.0, a: 0.0, omega: 0.0 };
            }
            if (tau < 7.60) {
                var v2 = 2.2 * (tau - 4.95);
                return { v: v2, a: 2.2, omega: 0.14 * Math.min(1.0, v2 / 3.0) };
            }
            return { v: 5.83, a: 0.0, omega: 0.14 };
        }

        var x = 1.60;
        var y = -0.60;
        var psi = Math.PI * 0.5 + 0.08;
        var steps = Math.max(1, Math.ceil(tSec / 0.025));
        var dtStep = tSec > 0 ? (tSec / steps) : 0;
        for (var i = 0; i < steps; i += 1) {
            var tauMid = (i + 0.5) * dtStep;
            var pMid = phaseParams(tauMid);
            var psiMid = psi + pMid.omega * (0.5 * dtStep);
            x += pMid.v * Math.cos(psiMid) * dtStep;
            y += pMid.v * Math.sin(psiMid) * dtStep;
            psi += pMid.omega * dtStep;
        }
        var end = phaseParams(tSec);
        return { x: x, y: y, psi: psi, v: end.v, a: end.a, omega: end.omega };
    }

    function buildScenarioState(scenarioId, cycleTimeS, config) {
        var t = ((cycleTimeS % SCENARIO_CYCLE_S) + SCENARIO_CYCLE_S) % SCENARIO_CYCLE_S;
        var vehicles = [];
        var vrus = [];
        var parkingZones = [
            [[1.85, 2.2], [3.05, 2.2], [3.05, 6.2], [1.85, 6.2]]
        ];
        var stitchingOn = !(config && config.stitching === false);

        if (scenarioId === "cyclist_vru_cutin") {
            // Scenario 2: Cyclist blind-spot cut-in past parked car (continuous kinematics dx/dt=vx, dy/dt=vy)
            var suvX = 0.35;
            var suvY = -1.25;
            var suvVx = 0.0;
            var suvVy = 5.0;
            var suvAy = 0.0;

            if (t < 0.45) {
                suvVy = 5.0;
                suvY = -1.25 + 5.0 * t;
                suvAy = 0.0;
            } else if (t < 1.85) {
                var u2 = t - 0.45;
                var w2 = u2 / 1.40;
                suvAy = -5.0 / 1.40;
                suvVy = Math.max(0.0, 5.0 + suvAy * u2);
                suvY = 1.00 + 5.0 * u2 + 0.5 * suvAy * u2 * u2;
                suvX = 0.35 - 0.22 * (3 * w2 * w2 - 2 * w2 * w2 * w2);
                suvVx = (-0.22 / 1.40) * (6 * w2 - 6 * w2 * w2);
            } else if (t < 4.95) {
                suvX = 0.13;
                suvY = 4.50;
                suvVx = 0.0;
                suvVy = 0.0;
                suvAy = 0.0;
            } else if (t < 7.60) {
                var u3 = t - 4.95;
                var w3 = Math.min(1.0, u3 / 1.50);
                suvAy = 2.4;
                suvVy = suvAy * u3;
                suvY = 4.50 + 0.5 * suvAy * u3 * u3;
                suvX = 0.13 + 0.14 * (3 * w3 * w3 - 2 * w3 * w3 * w3);
                suvVx = w3 < 1.0 ? (0.14 / 1.50) * (6 * w3 - 6 * w3 * w3) : 0.0;
            } else {
                var u4 = t - 7.60;
                suvAy = 0.0;
                suvVy = 2.4 * 2.65; // 6.36 m/s
                suvY = 4.50 + 0.5 * 2.4 * 2.65 * 2.65 + suvVy * u4;
                suvX = 0.27;
                suvVx = 0.0;
            }

            var suvSpd = Math.hypot(suvVx, suvVy);
            var suvPsi = suvSpd > 0.05 ? Math.atan2(suvVy, suvVx) : Math.PI * 0.5;
            var suvYawRate = (t >= 0.45 && t < 1.85) ? 0.06 : 0.0;

            vehicles.push({
                trackId: 1,
                kind: "suv",
                x: suvX,
                y: suvY,
                vx: suvVx,
                vy: suvVy,
                ax: 0,
                ay: suvAy,
                accelMs2: suvAy,
                headingRad: suvPsi,
                yawRate: suvYawRate,
                lengthM: 4.7,
                widthM: 1.95,
                speedKmh: suvSpd * 3.6,
                inParkingZone: false
            });

            vehicles.push({
                trackId: 2,
                kind: "car",
                x: 2.42,
                y: 4.2,
                vx: 0,
                vy: 0,
                ax: 0,
                ay: 0,
                accelMs2: 0,
                headingRad: Math.PI * 0.5,
                yawRate: 0,
                lengthM: 4.4,
                widthM: 1.85,
                speedKmh: 0,
                inParkingZone: true
            });

            // Cyclist emerges from behind V-02 laterally across SUV lane
            var cycX, cycY, cycVx, cycVy;
            if (t < 5.20) {
                cycVx = -0.95;
                cycVy = 0.22;
                cycX = 2.35 + cycVx * t;
                cycY = 7.40 + cycVy * t;
            } else if (t < 5.60) {
                var cu = t - 5.20;
                var cAx = 0.95 / 0.40;
                var cAy = (0.85 - 0.22) / 0.40;
                cycVx = -0.95 + cAx * cu;
                cycVy = 0.22 + cAy * cu;
                cycX = 2.35 - 0.95 * 5.20 - 0.95 * cu + 0.5 * cAx * cu * cu;
                cycY = 7.40 + 0.22 * 5.20 + 0.22 * cu + 0.5 * cAy * cu * cu;
            } else {
                var cu2 = t - 5.60;
                cycVx = 0.0;
                cycVy = 0.85;
                cycX = 2.35 - 0.95 * 5.20 - 0.5 * 0.95 * 0.40;
                cycY = 7.40 + 0.22 * 5.20 + 0.5 * (0.22 + 0.85) * 0.40 + 0.85 * cu2;
            }
            var cycTrackId = (!stitchingOn && t >= 1.15) ? 19 : 12;

            vrus.push({
                trackId: cycTrackId,
                kind: "bicycle",
                category: "bicycle",
                x: cycX,
                y: cycY,
                vx: cycVx,
                vy: cycVy,
                radiusM: 0.50,
                lengthM: 1.8,
                widthM: 0.7,
                headingRad: Math.atan2(cycVy, cycVx)
            });

            var ped5Omega = Math.PI * 0.25;
            vrus.push({
                trackId: 5,
                kind: "pedestrian",
                category: "pedestrian",
                x: -2.75,
                y: 11.2 + 0.35 * Math.sin(t * ped5Omega),
                vx: 0,
                vy: 0.35 * ped5Omega * Math.cos(t * ped5Omega),
                radiusM: 0.40
            });
        } else if (scenarioId === "ctrv_curve_yield") {
            // Scenario 3: Turning vehicle with exact CTRV numerical integration (dx/dt=v cos psi, dy/dt=v sin psi, dpsi/dt=omega)
            var ctrv = integrateCtrvState(t);

            vehicles.push({
                trackId: 1,
                kind: "car",
                x: ctrv.x,
                y: ctrv.y,
                vx: Math.cos(ctrv.psi) * ctrv.v,
                vy: Math.sin(ctrv.psi) * ctrv.v,
                ax: 0,
                ay: ctrv.a,
                accelMs2: ctrv.a,
                headingRad: ctrv.psi,
                yawRate: ctrv.omega,
                lengthM: 4.5,
                widthM: 1.9,
                speedKmh: ctrv.v * 3.6,
                inParkingZone: false
            });

            // Pedestrian P-04 crosses right-to-left into the curved CTRV turn path at y=8.40m
            var pedX, pedVx;
            if (t < 5.10) {
                pedVx = -0.95;
                pedX = 1.95 + pedVx * t;
            } else if (t < 5.40) {
                var pu = t - 5.10;
                var pAx = 0.95 / 0.30;
                pedVx = -0.95 + pAx * pu;
                pedX = 1.95 - 0.95 * 5.10 - 0.95 * pu + 0.5 * pAx * pu * pu;
            } else {
                pedVx = 0.0;
                pedX = 1.95 - 0.95 * 5.10 - 0.5 * 0.95 * 0.30;
            }
            vrus.push({
                trackId: 4,
                kind: "pedestrian",
                category: "pedestrian",
                x: pedX,
                y: 8.40,
                vx: pedVx,
                vy: 0,
                radiusM: 0.42
            });
        } else {
            // Scenario 1 (default): Crosswalk emergency braking (30.6 km/h L1 -> L2 -> L3 -> full yield -> depart)
            var vY = -1.65;
            var vSpeedMs = 8.5; // 30.6 km/h (> 30 km/h L1)
            var vAccel = 0.0;

            if (t < 0.22) {
                vSpeedMs = 8.5;
                vAccel = 0.0;
                vY = -1.65 + 8.5 * t;
            } else if (t < 1.33) {
                var b1 = t - 0.22;
                vAccel = -8.5 / 1.11;
                vSpeedMs = Math.max(0.0, 8.5 + vAccel * b1);
                vY = 0.22 + 8.5 * b1 + 0.5 * vAccel * b1 * b1; // ends at y=4.9375m (front bumper=7.19m < 7.25m)
            } else if (t < 4.75) {
                vY = 4.9375;
                vSpeedMs = 0.0;
                vAccel = 0.0;
            } else if (t < 7.60) {
                var b3 = t - 4.75;
                vAccel = 2.4;
                vSpeedMs = vAccel * b3;
                vY = 4.9375 + 0.5 * vAccel * b3 * b3; // reaches y=14.6845m at v=6.84 m/s
            } else {
                var b4 = t - 7.60;
                vAccel = 0.0;
                vSpeedMs = 2.4 * 2.85;
                vY = 4.9375 + 0.5 * 2.4 * 2.85 * 2.85 + vSpeedMs * b4;
            }

            vehicles.push({
                trackId: 1,
                kind: "car",
                x: -0.65,
                y: vY,
                vx: 0,
                vy: vSpeedMs,
                ax: 0,
                ay: vAccel,
                accelMs2: vAccel,
                headingRad: Math.PI * 0.5,
                yawRate: 0,
                lengthM: 4.5,
                widthM: 1.9,
                speedKmh: vSpeedMs * 3.6,
                inParkingZone: false
            });

            // Pedestrian P-07 steps from right sidewalk onto crosswalk and exits left sidewalk cleanly
            var p7x, p7vx;
            if (t < 0.22) {
                p7vx = -0.35;
                p7x = 2.65 + p7vx * t;
            } else if (t < 0.42) {
                var pu1 = t - 0.22;
                var pax1 = (-1.10 - (-0.35)) / 0.20;
                p7vx = -0.35 + pax1 * pu1;
                p7x = 2.65 - 0.35 * 0.22 - 0.35 * pu1 + 0.5 * pax1 * pu1 * pu1;
            } else if (t < 4.95) {
                p7vx = -1.10;
                p7x = 2.428 + p7vx * (t - 0.42);
            } else if (t < 5.25) {
                var pu2 = t - 4.95;
                var pax2 = 1.10 / 0.30;
                p7vx = -1.10 + pax2 * pu2;
                p7x = 2.428 - 1.10 * (4.95 - 0.42) - 1.10 * pu2 + 0.5 * pax2 * pu2 * pu2;
            } else {
                p7vx = 0.0;
                p7x = 2.428 - 1.10 * (4.95 - 0.42) - 0.5 * 1.10 * 0.30;
            }

            vrus.push({
                trackId: 7,
                kind: "pedestrian",
                category: "pedestrian",
                x: p7x,
                y: 7.85,
                vx: p7vx,
                vy: 0,
                radiusM: 0.42
            });

            var p3Omega = Math.PI * 0.25;
            vrus.push({
                trackId: 3,
                kind: "pedestrian",
                category: "pedestrian",
                x: 2.75,
                y: 11.4 - 0.45 * Math.cos(t * p3Omega),
                vx: 0,
                vy: 0.45 * p3Omega * Math.sin(t * p3Omega),
                radiusM: 0.40
            });
        }

        return {
            bounds: { xMin: -3.2, xMax: 3.2, yMin: 0.0, yMax: 15.0 },
            parkingZones: parkingZones,
            vehicles: vehicles,
            vrus: vrus
        };
    }

    function CampusRoadTwinCanvas(canvasEl, options) {
        if (!(this instanceof CampusRoadTwinCanvas)) {
            return new CampusRoadTwinCanvas(canvasEl, options);
        }
        if (!canvasEl) {
            throw new Error("CampusRoadTwinCanvas requires a valid <canvas> element");
        }
        var opts = options || {};
        this._canvas = canvasEl;
        this._ctx = canvasEl.getContext("2d");
        this._onTelemetry = typeof opts.onTelemetry === "function" ? opts.onTelemetry : null;
        this._onRenderState = typeof opts.onRenderState === "function" ? opts.onRenderState : null;
        this._speedLimitKmh = toNum(opts.speedLimitKmh, 30.0);
        this._config = {
            speedModel: (opts.config && opts.config.speedModel) || "ctrv",
            conflictModel: (opts.config && opts.config.conflictModel) || "obb_ssm",
            stitching: opts.config && opts.config.stitching !== undefined ? Boolean(opts.config.stitching) : true,
            cyclistVru: opts.config && opts.config.cyclistVru !== undefined ? Boolean(opts.config.cyclistVru) : true
        };

        this._mode = "simulation";
        this._scenarioId = "crosswalk_braking";
        this._userPaused = false;
        this._offscreenPaused = false;
        this._destroyed = false;

        this._simTimeS = 0.72; // Start near L3 conflict approach peak if reduced motion or paused
        this._simFrameCount = 0;
        this._lastDrawTs = null;
        this._lastTelemetryTs = 0;
        this._rafId = null;
        this._bgTimerId = null;
        this._lastBgTs = null;

        // 0.5m PET spatial occupancy grid: key -> { vehId, vehTime, vruId, vruTime, petS }
        this._petCells = Object.create(null);
        this._replayPrevTracks = Object.create(null);
        this._lastReplayFrameIdx = -1;

        this._cssWidth = 0;
        this._cssHeight = 0;
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);

        this._probe = createColorProbe();
        this._palette = readThemePalette(this._probe);

        var self = this;
        this._motionTracker = createReducedMotionTracker(function (isReduced) {
            if (isReduced) {
                self._stopLoop();
                self._renderCurrent();
                self._emitSimTelemetry(true);
            } else {
                self._syncLoop();
            }
        });
        this._unobserveTheme = observeThemeChanges(function () {
            self.refreshTheme();
        });

        this._replayState = {
            frameIndex: 0,
            timestampS: 0,
            bounds: normalizeBounds(null),
            vehicles: [],
            vrus: [],
            conflicts: [],
            parkingZones: [],
            label: ""
        };

        this._currentRenderState = {
            bounds: normalizeBounds(null),
            parkingZones: [],
            vehicles: [],
            vrus: [],
            conflicts: [],
            label: "",
            simTimeS: 0,
            maxRisk: 0
        };

        if (typeof ResizeObserver === "function") {
            this._resizeObserver = new ResizeObserver(function () {
                self._measureCanvas();
                self._renderCurrent();
            });
            this._resizeObserver.observe(this._canvas);
        } else {
            this._resizeObserver = null;
        }

        if (typeof IntersectionObserver === "function") {
            this._intersectionObserver = new IntersectionObserver(function (entries) {
                var visible = entries && entries[0] ? entries[0].isIntersecting : true;
                var wasOffscreen = self._offscreenPaused;
                self._offscreenPaused = !visible;
                if (visible && wasOffscreen) {
                    self._measureCanvas();
                    self._renderCurrent();
                }
                self._syncLoop();
            }, { threshold: 0.05 });
            this._intersectionObserver.observe(this._canvas);
        } else {
            this._intersectionObserver = null;
        }

        this._measureCanvas();
        this._seedInitialPetTrail();
        this._stepSimulation(0, true);
        this._syncLoop();
    }

    CampusRoadTwinCanvas.prototype._measureCanvas = function () {
        var rect = this._canvas.getBoundingClientRect();
        var w = Math.max(0, rect.width || this._canvas.clientWidth || 0);
        var h = Math.max(0, rect.height || this._canvas.clientHeight || 0);
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);
        this._cssWidth = w;
        this._cssHeight = h;
        if (w >= 2 && h >= 2) {
            var pw = Math.round(w * this._dpr);
            var ph = Math.round(h * this._dpr);
            if (this._canvas.width !== pw || this._canvas.height !== ph) {
                this._canvas.width = pw;
                this._canvas.height = ph;
            }
        }
    };

    CampusRoadTwinCanvas.prototype._seedInitialPetTrail = function () {
        this._petCells = Object.create(null);
        // Seed recent swept-OBB history behind initial keyframe so PET grid is intelligible immediately
        var startT = Math.max(0, this._simTimeS - 2.0);
        var activeCyclistVru = !(this._config && this._config.cyclistVru === false);
        for (var t = startT; t <= this._simTimeS; t += 0.25) {
            var st = buildScenarioState(this._scenarioId, t, this._config);
            var activeVrus = activeCyclistVru
                ? st.vrus
                : st.vrus.filter(function (v) { return !isCyclistVruKind(v); });
            this._recordPetGrid(st.vehicles, activeVrus, t);
        }
    };

    CampusRoadTwinCanvas.prototype._recordPetGrid = function (vehicles, vrus, timeS) {
        var cellSize = 0.5;
        for (var vi = 0; vi < vehicles.length; vi += 1) {
            var v = vehicles[vi];
            if (v.inParkingZone || (v.speedKmh !== undefined && v.speedKmh < 0.4)) continue;
            var halfL = (v.lengthM || 4.5) * 0.5;
            var halfW = (v.widthM || 1.9) * 0.5;
            var cosP = Math.cos(v.headingRad || 0);
            var sinP = Math.sin(v.headingRad || 0);

            for (var lx = -halfL; lx <= halfL + 0.01; lx += cellSize) {
                for (var ly = -halfW; ly <= halfW + 0.01; ly += cellSize) {
                    var gx = v.x + lx * cosP - ly * sinP;
                    var gy = v.y + lx * sinP + ly * cosP;
                    var cx = Math.floor(gx / cellSize);
                    var cy = Math.floor(gy / cellSize);
                    var key = cx + "," + cy;
                    var existing = this._petCells[key];
                    if (!existing) {
                        this._petCells[key] = {
                            cx: cx,
                            cy: cy,
                            vehId: v.trackId,
                            vehTime: timeS,
                            vruId: null,
                            vruTime: -999,
                            petS: null
                        };
                    } else {
                        existing.vehId = v.trackId;
                        existing.vehTime = timeS;
                        if (existing.vruId !== null && timeS >= existing.vruTime && (timeS - existing.vruTime) <= 2.2) {
                            existing.petS = timeS - existing.vruTime;
                        }
                    }
                }
            }
        }

        for (var pi = 0; pi < vrus.length; pi += 1) {
            var p = vrus[pi];
            var pcx = Math.floor(p.x / cellSize);
            var pcy = Math.floor(p.y / cellSize);
            var pkey = pcx + "," + pcy;
            var cell = this._petCells[pkey];
            if (cell && cell.vehTime <= timeS && (timeS - cell.vehTime) <= 2.4) {
                cell.vruId = p.trackId;
                cell.vruTime = timeS;
                cell.petS = Math.max(0.05, timeS - cell.vehTime);
            }
        }

        // Prune cells older than 3.6s to keep map clean and bounded
        var keys = Object.keys(this._petCells);
        for (var k = 0; k < keys.length; k += 1) {
            var rec = this._petCells[keys[k]];
            if (timeS - rec.vehTime > 3.6) {
                delete this._petCells[keys[k]];
            }
        }
    };

    CampusRoadTwinCanvas.prototype._evaluateSimulation = function (simData, timeS) {
        var conflicts = [];
        var driverAlerts = [];
        var pedestrianAlerts = [];
        var vehMap = Object.create(null);
        var maxRisk = 0;
        var maxSpeed = 0;
        var minTtc = null;
        var cyclistVruEnabled = !(this._config && this._config.cyclistVru === false);
        var activeVrus = cyclistVruEnabled
            ? simData.vrus
            : simData.vrus.filter(function (item) { return !isCyclistVruKind(item); });

        this._recordPetGrid(simData.vehicles, activeVrus, timeS);

        // Find active PET overlaps per pair
        var pairPet = Object.create(null);
        var cellKeys = Object.keys(this._petCells);
        for (var c = 0; c < cellKeys.length; c += 1) {
            var cell = this._petCells[cellKeys[c]];
            if (cell.petS !== null && timeS - cell.vruTime <= 1.8) {
                var pk = cell.vehId + ":" + cell.vruId;
                if (pairPet[pk] === undefined || cell.petS < pairPet[pk]) {
                    pairPet[pk] = cell.petS;
                }
            }
        }

        for (var vi = 0; vi < simData.vehicles.length; vi += 1) {
            var veh = simData.vehicles[vi];
            if (veh.speedKmh > maxSpeed) maxSpeed = veh.speedKmh;
            if (veh.inParkingZone && veh.speedKmh <= 5.0) {
                veh.riskLevel = 0;
                continue;
            }

            var bestConflict = null;
            for (var pi = 0; pi < activeVrus.length; pi += 1) {
                var vru = activeVrus[pi];
                var res = computeCpaAndTtc(veh, vru, 3.5);
                var pk2 = veh.trackId + ":" + vru.trackId;
                var petS = pairPet[pk2] !== undefined ? pairPet[pk2] : null;
                if (res.currentDist > 16.0 && petS === null) continue;

                var level = 0;
                if ((res.ttcSeconds !== null && res.ttcSeconds <= 1.8) || (petS !== null && petS <= 1.0)) {
                    level = 3;
                } else if ((res.ttcSeconds !== null && res.ttcSeconds <= 3.5) || (petS !== null && petS <= 2.0)) {
                    level = 2;
                }

                if (level > 0) {
                    var cand = {
                        vehTrackId: veh.trackId,
                        pedTrackId: vru.trackId,
                        vehicleKind: veh.kind || "car",
                        targetKind: vru.kind || "pedestrian",
                        riskLevel: level,
                        vehSpeedKmh: veh.speedKmh,
                        currentDistanceM: res.currentDist,
                        minDistanceM: res.minDistanceM,
                        ttcSeconds: res.ttcSeconds,
                        petSeconds: petS,
                        conflictGroundXY: res.conflictGroundXY
                    };
                    if (!bestConflict ||
                        cand.riskLevel > bestConflict.riskLevel ||
                        (cand.riskLevel === bestConflict.riskLevel &&
                         (cand.ttcSeconds !== null ? cand.ttcSeconds : 99) < (bestConflict.ttcSeconds !== null ? bestConflict.ttcSeconds : 99))) {
                        bestConflict = cand;
                    }
                }
            }

            if (bestConflict) {
                veh.riskLevel = bestConflict.riskLevel;
                vehMap[veh.trackId] = bestConflict;
                conflicts.push(bestConflict);
                if (bestConflict.riskLevel > maxRisk) maxRisk = bestConflict.riskLevel;
                if (bestConflict.ttcSeconds !== null && (minTtc === null || bestConflict.ttcSeconds < minTtc)) {
                    minTtc = bestConflict.ttcSeconds;
                }

                var ttcTxt = bestConflict.ttcSeconds !== null ? ("TTC=" + bestConflict.ttcSeconds.toFixed(2) + "s") : "PET=" + bestConflict.petSeconds.toFixed(2) + "s";
                var spdTxt = bestConflict.vehSpeedKmh.toFixed(1) + " km/h";
                var targetLabel = (bestConflict.targetKind === "bicycle" || bestConflict.targetKind === "motorcycle") ? "骑行者" : "行人";
                var urgency = bestConflict.riskLevel === 3 ? "critical" : "caution";

                driverAlerts.push({
                    target: "driver",
                    role: "driver",
                    urgency: urgency,
                    risk_level: bestConflict.riskLevel,
                    risk_name: RISK_NAMES[bestConflict.riskLevel],
                    message: (bestConflict.riskLevel === 3 ? "紧急碰撞预警" : "注意预警") + "：车辆 #" + bestConflict.vehTrackId + " (" + spdTxt + ") 与" + targetLabel + " #" + bestConflict.pedTrackId + " 轨迹汇聚（" + ttcTxt + "）。",
                    advice: bestConflict.riskLevel === 3 ? "请立即紧急制动停车避让。" : "请提前备刹减速，注意观察前方弱势交通参与者动态。",
                    veh_track_id: bestConflict.vehTrackId,
                    ped_track_id: bestConflict.pedTrackId,
                    vehicle_kind: bestConflict.vehicleKind,
                    target_kind: bestConflict.targetKind,
                    veh_speed_kmh: Number(bestConflict.vehSpeedKmh.toFixed(1)),
                    current_distance_m: Number(bestConflict.currentDistanceM.toFixed(2)),
                    min_distance_m: Number(bestConflict.minDistanceM.toFixed(2)),
                    ttc_seconds: bestConflict.ttcSeconds !== null ? Number(bestConflict.ttcSeconds.toFixed(2)) : null,
                    pet_seconds: bestConflict.petSeconds !== null ? Number(bestConflict.petSeconds.toFixed(2)) : null
                });

                pedestrianAlerts.push({
                    target: "pedestrian",
                    role: "pedestrian",
                    urgency: urgency,
                    risk_level: bestConflict.riskLevel,
                    risk_name: RISK_NAMES[bestConflict.riskLevel],
                    message: (bestConflict.riskLevel === 3 ? "紧急避险预警" : "路旁提示") + "：" + targetLabel + " #" + bestConflict.pedTrackId + " 请注意车辆 #" + bestConflict.vehTrackId + " (" + spdTxt + ") 正在接近（" + ttcTxt + "）。",
                    advice: bestConflict.riskLevel === 3 ? "请立即止步并向路侧安全区域退让避险。" : "请停步观望或沿安全区域通行，避免盲目横穿。",
                    veh_track_id: bestConflict.vehTrackId,
                    ped_track_id: bestConflict.pedTrackId,
                    vehicle_kind: bestConflict.vehicleKind,
                    target_kind: bestConflict.targetKind,
                    veh_speed_kmh: Number(bestConflict.vehSpeedKmh.toFixed(1)),
                    current_distance_m: Number(bestConflict.currentDistanceM.toFixed(2)),
                    min_distance_m: Number(bestConflict.minDistanceM.toFixed(2)),
                    ttc_seconds: bestConflict.ttcSeconds !== null ? Number(bestConflict.ttcSeconds.toFixed(2)) : null,
                    pet_seconds: bestConflict.petSeconds !== null ? Number(bestConflict.petSeconds.toFixed(2)) : null
                });
            } else if (veh.speedKmh > this._speedLimitKmh) {
                veh.riskLevel = 1;
                if (1 > maxRisk) maxRisk = 1;
                driverAlerts.push({
                    target: "driver",
                    role: "driver",
                    urgency: "speeding",
                    risk_level: 1,
                    risk_name: RISK_NAMES[1],
                    message: "超速提示：车辆 #" + veh.trackId + " (" + (veh.kind || "motor_vehicle") + ") 当前车速 " + veh.speedKmh.toFixed(1) + " km/h，已超出校园限速。",
                    advice: "请立即减速慢行，将车速控制在校内限速以内。",
                    veh_track_id: veh.trackId,
                    ped_track_id: null,
                    vehicle_kind: veh.kind || "motor_vehicle",
                    target_kind: "pedestrian",
                    veh_speed_kmh: Number(veh.speedKmh.toFixed(1)),
                    current_distance_m: 0,
                    min_distance_m: 0,
                    ttc_seconds: null,
                    pet_seconds: null
                });
            } else {
                veh.riskLevel = 0;
            }
        }

        // Tag VRUs involved in conflicts for clean ring highlight
        for (var j = 0; j < simData.vrus.length; j += 1) {
            var item = simData.vrus[j];
            item.riskLevel = 0;
            for (var cIdx = 0; cIdx < conflicts.length; cIdx += 1) {
                if (conflicts[cIdx].pedTrackId === item.trackId && conflicts[cIdx].riskLevel > item.riskLevel) {
                    item.riskLevel = conflicts[cIdx].riskLevel;
                }
            }
        }

        this._latestSimSnapshot = {
            timestampS: Number(timeS.toFixed(2)),
            frameCount: this._simFrameCount,
            maxSpeedKmh: Number(maxSpeed.toFixed(1)),
            minTtcS: minTtc !== null ? Number(minTtc.toFixed(2)) : null,
            activeConflicts: driverAlerts.length,
            riskLevel: maxRisk,
            riskName: RISK_NAMES[maxRisk],
            driverAlerts: driverAlerts,
            pedestrianAlerts: pedestrianAlerts,
            vehicleCount: simData.vehicles.length,
            vruCount: simData.vrus.length,
            seriesPoint: {
                t: Number(timeS.toFixed(2)),
                speedKmh: Number(maxSpeed.toFixed(1)),
                ttcS: minTtc !== null ? Number(minTtc.toFixed(2)) : null,
                riskLevel: maxRisk
            }
        };

        this._currentRenderState = {
            bounds: simData.bounds,
            parkingZones: simData.parkingZones,
            vehicles: simData.vehicles,
            vrus: simData.vrus,
            conflicts: conflicts,
            label: this._scenarioTitle(this._scenarioId),
            simTimeS: timeS,
            maxRisk: maxRisk
        };
    };

    CampusRoadTwinCanvas.prototype._scenarioTitle = function (id) {
        if (id === "cyclist_vru_cutin") return "SIM · 骑行者盲区切入";
        if (id === "ctrv_curve_yield") return "SIM · 弯道 CTRV 轨迹会车";
        return "SIM · 斑马线横穿急刹";
    };

    CampusRoadTwinCanvas.prototype._stepSimulation = function (dt, forceEmit, skipRender) {
        if (this._mode !== "simulation") return;
        var prevMod = this._simTimeS % SCENARIO_CYCLE_S;
        this._simTimeS += dt;
        var nextMod = this._simTimeS % SCENARIO_CYCLE_S;
        if (nextMod < prevMod) {
            // Reset PET grid smoothly on scenario cycle restart
            this._petCells = Object.create(null);
        }
        this._simFrameCount += 1;
        var simData = buildScenarioState(this._scenarioId, this._simTimeS, this._config);
        this._evaluateSimulation(simData, this._simTimeS);
        if (!skipRender) {
            this._renderCurrent();
        }
        this._emitSimTelemetry(Boolean(forceEmit));
    };

    CampusRoadTwinCanvas.prototype._emitSimTelemetry = function (force) {
        if (!this._onTelemetry || !this._latestSimSnapshot || this._mode !== "simulation") {
            return;
        }
        var now = typeof performance !== "undefined" ? performance.now() : Date.now();
        if (force || (now - this._lastTelemetryTs >= 110)) {
            this._lastTelemetryTs = now;
            this._onTelemetry(this._latestSimSnapshot);
        }
    };

    CampusRoadTwinCanvas.prototype._stopVisibleLoop = function () {
        if (this._rafId !== null && typeof global.cancelAnimationFrame === "function") {
            global.cancelAnimationFrame(this._rafId);
            this._rafId = null;
        }
        this._lastDrawTs = null;
    };

    CampusRoadTwinCanvas.prototype._stopBgLoop = function () {
        if (this._bgTimerId !== null && typeof global.clearInterval === "function") {
            global.clearInterval(this._bgTimerId);
            this._bgTimerId = null;
        }
        this._lastBgTs = null;
    };

    CampusRoadTwinCanvas.prototype._stopLoop = function () {
        this._stopVisibleLoop();
        this._stopBgLoop();
    };

    CampusRoadTwinCanvas.prototype._syncLoop = function () {
        var simActive = !this._destroyed &&
            this._mode === "simulation" &&
            !this._userPaused &&
            !this._motionTracker.isReduced();

        if (!simActive) {
            this._stopLoop();
            return;
        }

        if (this._offscreenPaused) {
            // Keep physical simulation + 8Hz telemetry active while canvas is offscreen (e.g. Alerts tab)
            this._stopVisibleLoop();
            if (this._bgTimerId === null && typeof global.setInterval === "function") {
                var bgSelf = this;
                this._lastBgTs = typeof performance !== "undefined" ? performance.now() : Date.now();
                this._bgTimerId = global.setInterval(function () {
                    if (bgSelf._destroyed ||
                        bgSelf._mode !== "simulation" ||
                        bgSelf._userPaused ||
                        !bgSelf._offscreenPaused ||
                        bgSelf._motionTracker.isReduced()) {
                        bgSelf._stopBgLoop();
                        return;
                    }
                    var now = typeof performance !== "undefined" ? performance.now() : Date.now();
                    var dt = Math.min(0.25, Math.max(0.02, (now - (bgSelf._lastBgTs || now)) / 1000));
                    bgSelf._lastBgTs = now;
                    bgSelf._stepSimulation(dt, true, true);
                }, 125);
            }
            return;
        }

        this._stopBgLoop();
        if (this._rafId !== null || typeof global.requestAnimationFrame !== "function") {
            return;
        }

        var self = this;
        function frame(now) {
            if (self._destroyed ||
                self._mode !== "simulation" ||
                self._userPaused ||
                self._offscreenPaused ||
                self._motionTracker.isReduced()) {
                self._rafId = null;
                return;
            }
            if (self._lastDrawTs === null) {
                self._lastDrawTs = now;
            }
            var dt = Math.min(0.06, Math.max(0, (now - self._lastDrawTs) / 1000));
            self._lastDrawTs = now;
            self._stepSimulation(dt, false, false);
            self._rafId = global.requestAnimationFrame(frame);
        }

        this._rafId = global.requestAnimationFrame(frame);
    };

    CampusRoadTwinCanvas.prototype._renderCurrent = function () {
        if (!this._ctx || this._cssWidth < 2 || this._cssHeight < 2) return;
        var ctx = this._ctx;
        var pal = this._palette;
        var st = this._currentRenderState;
        var w = this._cssWidth;
        var h = this._cssHeight;

        ctx.save();
        ctx.scale(this._dpr, this._dpr);
        ctx.clearRect(0, 0, w, h);

        // Background shoulder
        ctx.fillStyle = pal.roadShoulder;
        ctx.fillRect(0, 0, w, h);

        var padL = 44;
        var padR = 20;
        var padT = 32;
        var padB = 26;
        var availW = Math.max(40, w - padL - padR);
        var availH = Math.max(40, h - padT - padB);

        var b = st.bounds;
        var spanX = Math.max(2, b.xMax - b.xMin);
        var spanY = Math.max(2, b.yMax - b.yMin);
        var scale = Math.min(availW / spanX, availH / spanY);
        var drawW = spanX * scale;
        var drawH = spanY * scale;
        var originX = padL + (availW - drawW) * 0.5;
        var originY = padT + (availH - drawH) * 0.5 + drawH; // yMin at bottom, yMax at top

        function toPx(wx, wy) {
            return {
                x: originX + (wx - b.xMin) * scale,
                y: originY - (wy - b.yMin) * scale
            };
        }

        // Roadbed polygon & calibrated 1:1 sidewalk shoulders
        var roadTopLeft = toPx(b.xMin, b.yMax);
        var roadBotRight = toPx(b.xMax, b.yMin);

        var leftSidewalkW = Math.max(0, roadTopLeft.x - padL);
        var rightSidewalkW = Math.max(0, (w - padR) - roadBotRight.x);
        var sidewalkFill = withAlpha(pal.bgCard, pal.isLight ? 0.56 : 0.40);
        var tileStroke = withAlpha(pal.hairline, pal.isLight ? 0.70 : 0.65);

        if (leftSidewalkW > 4) {
            ctx.fillStyle = sidewalkFill;
            ctx.fillRect(padL, roadTopLeft.y, leftSidewalkW, drawH);
        }
        if (rightSidewalkW > 4) {
            ctx.fillStyle = sidewalkFill;
            ctx.fillRect(roadBotRight.x, roadTopLeft.y, rightSidewalkW, drawH);
        }

        // Subtle architectural sidewalk paving joints (clipped strictly to left/right sidewalk bands)
        if (leftSidewalkW > 12 || rightSidewalkW > 12) {
            ctx.save();
            ctx.strokeStyle = tileStroke;
            ctx.lineWidth = 1;
            var slabStepM = 1.8;
            for (var sy = Math.ceil(b.yMin / slabStepM) * slabStepM; sy < b.yMax; sy += slabStepM) {
                var syPx = toPx(b.xMin, sy).y;
                if (leftSidewalkW > 12) {
                    ctx.beginPath();
                    ctx.moveTo(padL, syPx);
                    ctx.lineTo(roadTopLeft.x, syPx);
                    ctx.stroke();
                }
                if (rightSidewalkW > 12) {
                    ctx.beginPath();
                    ctx.moveTo(roadBotRight.x, syPx);
                    ctx.lineTo(w - padR, syPx);
                    ctx.stroke();
                }
            }
            var swColM = 2.0 * scale;
            if (leftSidewalkW > swColM) {
                for (var lx = roadTopLeft.x - swColM; lx > padL + 6; lx -= swColM) {
                    ctx.beginPath();
                    ctx.moveTo(lx, roadTopLeft.y);
                    ctx.lineTo(lx, roadBotRight.y);
                    ctx.stroke();
                }
            }
            if (rightSidewalkW > swColM) {
                for (var rx = roadBotRight.x + swColM; rx < w - padR - 6; rx += swColM) {
                    ctx.beginPath();
                    ctx.moveTo(rx, roadTopLeft.y);
                    ctx.lineTo(rx, roadBotRight.y);
                    ctx.stroke();
                }
            }
            ctx.restore();
        }

        ctx.fillStyle = pal.roadSurface;
        ctx.fillRect(roadTopLeft.x, roadTopLeft.y, roadBotRight.x - roadTopLeft.x, roadBotRight.y - roadTopLeft.y);

        // 1m subtle ruler grid + distance labels
        ctx.lineWidth = 1;
        ctx.font = "500 10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";

        var firstX = Math.ceil(b.xMin);
        for (var gx = firstX; gx <= Math.floor(b.xMax); gx += 1) {
            var px1 = toPx(gx, b.yMin);
            var px2 = toPx(gx, b.yMax);
            ctx.strokeStyle = pal.gridLine;
            ctx.beginPath();
            ctx.moveTo(px1.x, px1.y);
            ctx.lineTo(px2.x, px2.y);
            ctx.stroke();

            if (gx % 2 === 0 || spanX <= 7) {
                ctx.fillStyle = pal.textMuted;
                ctx.textAlign = "center";
                ctx.textBaseline = "top";
                ctx.fillText(gx + "m", px1.x, roadBotRight.y + 6);
            }
        }

        var firstY = Math.ceil(b.yMin);
        var yStep = spanY > 12 ? 2 : 1;
        for (var gy = firstY; gy <= Math.floor(b.yMax); gy += 1) {
            var py1 = toPx(b.xMin, gy);
            var py2 = toPx(b.xMax, gy);
            ctx.strokeStyle = pal.gridLine;
            ctx.beginPath();
            ctx.moveTo(py1.x, py1.y);
            ctx.lineTo(py2.x, py2.y);
            ctx.stroke();

            if (gy % yStep === 0 || gy === Math.floor(b.yMax)) {
                ctx.fillStyle = pal.textMuted;
                ctx.textAlign = "right";
                ctx.textBaseline = "middle";
                ctx.fillText(gy + "m", padL - 6, py1.y);
            }
        }

        // Crosswalk zebra stripes around y = 7.8m (plus subtle sidewalk curb ramps)
        if (7.8 >= b.yMin + 0.5 && 7.8 <= b.yMax - 0.5) {
            var cwLeft = b.xMin + 0.35;
            var cwRight = b.xMax - 0.35;
            var cwY0 = 7.25;
            var cwY1 = 8.45;
            var cwTopPx = toPx(b.xMin, cwY1).y;
            var cwBotPx = toPx(b.xMin, cwY0).y;
            if (leftSidewalkW > 12) {
                ctx.fillStyle = withAlpha(pal.textPrimary, pal.isLight ? 0.045 : 0.035);
                ctx.fillRect(Math.max(padL, roadTopLeft.x - 1.2 * scale), cwTopPx, Math.min(leftSidewalkW, 1.2 * scale), cwBotPx - cwTopPx);
            }
            if (rightSidewalkW > 12) {
                ctx.fillStyle = withAlpha(pal.textPrimary, pal.isLight ? 0.045 : 0.035);
                ctx.fillRect(roadBotRight.x, cwTopPx, Math.min(rightSidewalkW, 1.2 * scale), cwBotPx - cwTopPx);
            }
            ctx.fillStyle = withAlpha(pal.textPrimary, pal.isLight ? 0.14 : 0.11);
            for (var sx = cwLeft; sx < cwRight - 0.25; sx += 0.65) {
                var spA = toPx(sx, cwY1);
                var spB = toPx(Math.min(sx + 0.34, cwRight), cwY0);
                ctx.fillRect(spA.x, spA.y, spB.x - spA.x, spB.y - spA.y);
            }
        }

        // Road boundary curbs and center dashed lane divider
        var curbInset = 0.25;
        var curbL0 = toPx(b.xMin + curbInset, b.yMin);
        var curbL1 = toPx(b.xMin + curbInset, b.yMax);
        var curbR0 = toPx(b.xMax - curbInset, b.yMin);
        var curbR1 = toPx(b.xMax - curbInset, b.yMax);
        ctx.strokeStyle = pal.roadCurb;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(curbL0.x, curbL0.y);
        ctx.lineTo(curbL1.x, curbL1.y);
        ctx.moveTo(curbR0.x, curbR0.y);
        ctx.lineTo(curbR1.x, curbR1.y);
        ctx.stroke();

        var centerX = 0;
        if (centerX <= b.xMin + 0.8 || centerX >= b.xMax - 0.8) {
            centerX = 0.5 * (b.xMin + b.xMax);
        }
        var div0 = toPx(centerX, b.yMin);
        var div1 = toPx(centerX, b.yMax);
        ctx.save();
        ctx.setLineDash([8, 8]);
        ctx.strokeStyle = pal.roadMark;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(div0.x, div0.y);
        ctx.lineTo(div1.x, div1.y);
        ctx.stroke();
        ctx.restore();

        // Collision-free pill reservation tracking
        var placedBoxes = [];
        var obstacleBoxes = [];
        var headerLabel = st.label || (this._mode === "replay" ? "REPLAY" : "SIMULATION");
        ctx.font = "500 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        var headerW = ctx.measureText(headerLabel).width + 12;
        placedBoxes.push({ x: 10, y: 8, w: headerW, h: 18 });

        var safeArea = {
            minX: padL + 2,
            maxX: w - 8,
            minY: Math.max(32, roadTopLeft.y),
            maxY: Math.max(54, roadBotRight.y - 6)
        };

        // Precompute OBB footprint screen boxes so labels avoid sitting on top of cars/VRUs
        for (var oi = 0; oi < st.vehicles.length; oi += 1) {
            var ov = st.vehicles[oi];
            var ovPx = toPx(ov.x, ov.y);
            var oLen = Math.max(14, (ov.lengthM || 4.5) * scale);
            var oWid = Math.max(8, (ov.widthM || 1.9) * scale);
            var cosA = Math.abs(Math.cos(ov.headingRad || Math.PI * 0.5));
            var sinA = Math.abs(Math.sin(ov.headingRad || Math.PI * 0.5));
            var bW = oLen * cosA + oWid * sinA;
            var bH = oLen * sinA + oWid * cosA;
            obstacleBoxes.push({ x: ovPx.x - bW * 0.5, y: ovPx.y - bH * 0.5, w: bW, h: bH });
        }

        // Roadside parking exclusion zone polygon
        var zones = st.parkingZones || [];
        for (var zi = 0; zi < zones.length; zi += 1) {
            var poly = zones[zi];
            if (!Array.isArray(poly) || poly.length < 3) continue;
            var pMinX = Infinity, pMaxX = -Infinity, pMinY = Infinity, pMaxY = -Infinity;
            ctx.save();
            ctx.beginPath();
            for (var pi = 0; pi < poly.length; pi += 1) {
                var pPt = toPx(poly[pi][0], poly[pi][1]);
                if (pPt.x < pMinX) pMinX = pPt.x;
                if (pPt.x > pMaxX) pMaxX = pPt.x;
                if (pPt.y < pMinY) pMinY = pPt.y;
                if (pPt.y > pMaxY) pMaxY = pPt.y;
                if (pi === 0) ctx.moveTo(pPt.x, pPt.y);
                else ctx.lineTo(pPt.x, pPt.y);
            }
            ctx.closePath();
            ctx.fillStyle = withAlpha(pal.parkingZone, 0.08);
            ctx.fill();
            ctx.setLineDash([5, 4]);
            ctx.strokeStyle = withAlpha(pal.parkingZone, 0.72);
            ctx.lineWidth = 1.2;
            ctx.stroke();
            ctx.restore();

            ctx.font = "500 10px system-ui, -apple-system, sans-serif";
            var pzText = "禁停标定区";
            var pzW = ctx.measureText(pzText).width + 6;
            var pzH = 14;
            var pzCand = [
                { x: pMaxX - pzW - 2, y: pMinY - pzH - 3 },
                { x: pMinX + 2, y: pMinY - pzH - 3 },
                { x: pMaxX + 6, y: pMinY + 2 },
                { x: pMinX + 3, y: pMaxY - pzH - 2 }
            ];
            var pzPicked = chooseBestBox(pzCand, pzW, pzH, safeArea, placedBoxes, obstacleBoxes);
            ctx.fillStyle = withAlpha(pal.parkingZone, 0.94);
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(pzText, pzPicked.x + 2, pzPicked.y + pzH * 0.5);
            placedBoxes.push(pzPicked);
        }

        // Clip dynamic road overlays (PET grid, CTRV curves, TTC rings) strictly inside road network draw area
        var netClipX = padL;
        var netClipY = Math.max(30, roadTopLeft.y);
        var netClipW = Math.max(40, (w - padR) - padL);
        var netClipH = Math.max(20, roadBotRight.y - netClipY);

        ctx.save();
        ctx.beginPath();
        ctx.rect(netClipX, netClipY, netClipW, netClipH);
        ctx.clip();

        // 0.5m PET swept-OBB occupancy grid
        var refTime = st.simTimeS;
        var cellSize = 0.5;
        var cellKeys = Object.keys(this._petCells);
        for (var ci = 0; ci < cellKeys.length; ci += 1) {
            var cell = this._petCells[cellKeys[ci]];
            var age = refTime - cell.vehTime;
            if (age < 0 || age > 3.6) continue;
            var wx0 = cell.cx * cellSize;
            var wy0 = cell.cy * cellSize;
            if (wx0 + cellSize < b.xMin || wx0 > b.xMax || wy0 + cellSize < b.yMin || wy0 > b.yMax) continue;

            var cTopLeft = toPx(wx0, wy0 + cellSize);
            var cBotRight = toPx(wx0 + cellSize, wy0);
            var cw = cBotRight.x - cTopLeft.x;
            var ch = cBotRight.y - cTopLeft.y;
            var fade = clamp(1 - age / 3.6, 0.12, 1);

            if (cell.petS !== null && (refTime - cell.vruTime) <= 2.2) {
                var petUrgent = cell.petS <= 1.0;
                var petTint = petUrgent ? pal.critical : pal.accent;
                ctx.fillStyle = withAlpha(petTint, 0.24 * fade);
                ctx.fillRect(cTopLeft.x + 0.5, cTopLeft.y + 0.5, cw - 1, ch - 1);
                ctx.strokeStyle = withAlpha(petTint, 0.65 * fade);
                ctx.lineWidth = 1;
                ctx.strokeRect(cTopLeft.x + 0.5, cTopLeft.y + 0.5, cw - 1, ch - 1);
            } else {
                ctx.fillStyle = withAlpha(pal.accent, 0.08 * fade);
                ctx.fillRect(cTopLeft.x + 0.5, cTopLeft.y + 0.5, cw - 1, ch - 1);
            }
        }

        // Draw 2nd-order CTRV/Kalman prediction arcs beneath vehicles (clipped cleanly inside road network)
        for (var vi = 0; vi < st.vehicles.length; vi += 1) {
            var veh = st.vehicles[vi];
            var spdMs = Math.hypot(veh.vx || 0, veh.vy || 0);
            if (spdMs < 0.35 || veh.inParkingZone) continue;

            var arcPts = predictCtrvArc(veh, 2.5, 16);
            var arcColor = veh.riskLevel >= 1 ? riskColor(pal, veh.riskLevel) : pal.safe;
            ctx.save();
            ctx.setLineDash([5, 4]);
            ctx.strokeStyle = withAlpha(arcColor, 0.78);
            ctx.lineWidth = 1.6;
            ctx.beginPath();
            for (var ai = 0; ai < arcPts.length; ai += 1) {
                var aPx = toPx(arcPts[ai].x, arcPts[ai].y);
                if (ai === 0) ctx.moveTo(aPx.x, aPx.y);
                else ctx.lineTo(aPx.x, aPx.y);
            }
            ctx.stroke();
            ctx.restore();

            // Subtle 0.5s trajectory tick markers
            ctx.fillStyle = withAlpha(arcColor, 0.85);
            for (var ti = 3; ti < arcPts.length; ti += 3) {
                var tPx = toPx(arcPts[ti].x, arcPts[ti].y);
                ctx.beginPath();
                ctx.arc(tPx.x, tPx.y, 2, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Draw TTC convergence links & calm concentric arcs; defer pill drawing for priority collision avoidance
        var pendingConflictPills = [];
        for (var k = 0; k < st.conflicts.length; k += 1) {
            var cf = st.conflicts[k];
            if ((cf.riskLevel || 0) < 2) continue;
            var cfColor = riskColor(pal, cf.riskLevel);
            var vehObj = findById(st.vehicles, cf.vehTrackId);
            var vruObj = findById(st.vrus, cf.pedTrackId);

            if (vehObj && vruObj) {
                var pA = toPx(vehObj.x, vehObj.y);
                var pB = toPx(vruObj.x, vruObj.y);
                ctx.save();
                ctx.setLineDash([3, 3]);
                ctx.strokeStyle = withAlpha(cfColor, 0.75);
                ctx.lineWidth = 1.25;
                ctx.beginPath();
                ctx.moveTo(pA.x, pA.y);
                ctx.lineTo(pB.x, pB.y);
                ctx.stroke();
                ctx.restore();
            }

            var cxy = cf.conflictGroundXY;
            if ((!Array.isArray(cxy) || cxy.length < 2) && vehObj && vruObj) {
                cxy = [(vehObj.x + vruObj.x) * 0.5, (vehObj.y + vruObj.y) * 0.5];
            }
            if (Array.isArray(cxy) && cxy.length >= 2 && isFiniteNum(cxy[0]) && isFiniteNum(cxy[1])) {
                var cPx = toPx(cxy[0], cxy[1]);
                ctx.strokeStyle = withAlpha(cfColor, 0.65);
                ctx.lineWidth = 1.3;
                ctx.beginPath();
                ctx.arc(cPx.x, cPx.y, 10, 0, Math.PI * 2);
                ctx.stroke();

                ctx.strokeStyle = withAlpha(cfColor, 0.28);
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.arc(cPx.x, cPx.y, 17, 0, Math.PI * 2);
                ctx.stroke();

                var pillParts = [];
                if (isFiniteNum(cf.ttcSeconds)) pillParts.push("TTC " + cf.ttcSeconds.toFixed(1) + "s");
                if (isFiniteNum(cf.petSeconds)) pillParts.push("PET " + cf.petSeconds.toFixed(1) + "s");
                else if (isFiniteNum(cf.minDistanceM)) pillParts.push("Dmin " + cf.minDistanceM.toFixed(1) + "m");

                if (pillParts.length > 0) {
                    pendingConflictPills.push({
                        cx: cPx.x,
                        cy: cPx.y,
                        rx: 18,
                        ry: 18,
                        text: pillParts.join(" / "),
                        border: cfColor
                    });
                }
            }
        }

        // Draw rounded OBB vehicles (record pill tasks for collision-free layout)
        var pendingVehiclePills = [];
        for (var vi2 = 0; vi2 < st.vehicles.length; vi2 += 1) {
            var v = st.vehicles[vi2];
            var vPx = toPx(v.x, v.y);
            var vColor = v.inParkingZone
                ? pal.textMuted
                : (v.riskLevel >= 1 ? riskColor(pal, v.riskLevel) : pal.safe);
            var lenPx = Math.max(14, (v.lengthM || 4.5) * scale);
            var widPx = Math.max(8, (v.widthM || 1.9) * scale);
            var cornerPx = Math.min(lenPx, widPx) * 0.22;

            ctx.save();
            ctx.translate(vPx.x, vPx.y);
            ctx.rotate(-(v.headingRad || Math.PI * 0.5));

            drawRoundedRectPath(ctx, -lenPx * 0.5, -widPx * 0.5, lenPx, widPx, cornerPx);
            ctx.fillStyle = withAlpha(vColor, v.inParkingZone ? 0.14 : 0.20);
            ctx.fill();
            ctx.lineWidth = v.riskLevel >= 2 ? 2.0 : 1.5;
            ctx.strokeStyle = vColor;
            ctx.stroke();

            // Directional heading triangle / headlight notch at +halfLen nose
            var noseX = lenPx * 0.5;
            ctx.fillStyle = vColor;
            ctx.beginPath();
            ctx.moveTo(noseX - 2, 0);
            ctx.lineTo(noseX - 8, -widPx * 0.24);
            ctx.lineTo(noseX - 8, widPx * 0.24);
            ctx.closePath();
            ctx.fill();
            ctx.restore();

            var spdDisplay = isFiniteNum(v.speedKmh) ? v.speedKmh.toFixed(1) + "km/h" : "0.0km/h";
            var badgeText = "V-" + padId(v.trackId) + " " + spdDisplay;
            var cosV = Math.abs(Math.cos(v.headingRad || Math.PI * 0.5));
            var sinV = Math.abs(Math.sin(v.headingRad || Math.PI * 0.5));
            pendingVehiclePills.push({
                cx: vPx.x,
                cy: vPx.y,
                rx: (lenPx * cosV + widPx * sinV) * 0.5,
                ry: (lenPx * sinV + widPx * cosV) * 0.5,
                text: badgeText,
                border: withAlpha(vColor, 0.55)
            });
        }

        // Draw pedestrian / cyclist VRUs (record pill tasks for collision-free layout)
        var pendingVruPills = [];
        for (var pi2 = 0; pi2 < st.vrus.length; pi2 += 1) {
            var u = st.vrus[pi2];
            var uPx = toPx(u.x, u.y);
            var isCyclist = u.kind === "bicycle" || u.category === "bicycle" || u.kind === "motorcycle";
            var uColor = u.riskLevel >= 2 ? riskColor(pal, u.riskLevel) : (isCyclist ? pal.accent : pal.textPrimary);

            // Velocity vector
            var uvx = u.vx || 0;
            var uvy = u.vy || 0;
            var uSpd = Math.hypot(uvx, uvy);
            if (uSpd > 0.15) {
                var tip = toPx(u.x + uvx * 0.9, u.y + uvy * 0.9);
                ctx.strokeStyle = withAlpha(uColor, 0.85);
                ctx.lineWidth = 1.4;
                ctx.beginPath();
                ctx.moveTo(uPx.x, uPx.y);
                ctx.lineTo(tip.x, tip.y);
                ctx.stroke();
            }

            var vruR = Math.max(6, (u.radiusM || 0.42) * scale * 0.8);
            if (isCyclist) {
                var cLen = Math.max(12, 1.8 * scale);
                var cWid = Math.max(6, 0.72 * scale);
                vruR = Math.max(cLen, cWid) * 0.5;
                ctx.save();
                ctx.translate(uPx.x, uPx.y);
                ctx.rotate(-(u.headingRad !== undefined ? u.headingRad : Math.atan2(uvy, uvx || 1)));
                drawRoundedRectPath(ctx, -cLen * 0.5, -cWid * 0.5, cLen, cWid, cWid * 0.45);
                ctx.fillStyle = withAlpha(uColor, 0.25);
                ctx.fill();
                ctx.lineWidth = 1.6;
                ctx.strokeStyle = uColor;
                ctx.stroke();
                ctx.restore();
            } else {
                var rPx = Math.max(4.5, (u.radiusM || 0.42) * scale * 0.8);
                ctx.fillStyle = withAlpha(uColor, 0.24);
                ctx.beginPath();
                ctx.arc(uPx.x, uPx.y, rPx, 0, Math.PI * 2);
                ctx.fill();
                ctx.lineWidth = 1.6;
                ctx.strokeStyle = uColor;
                ctx.stroke();
            }

            var tagPrefix = isCyclist ? "B-" : "P-";
            pendingVruPills.push({
                cx: uPx.x,
                cy: uPx.y,
                rx: vruR,
                ry: vruR,
                text: tagPrefix + padId(u.trackId),
                border: withAlpha(uColor, 0.55)
            });
        }

        // End road network clip before placing crisp unclipped pills
        ctx.restore();

        // Top-left status pill
        drawPill(ctx, 10, 8, headerLabel, pal.bgCard, pal.hairline, pal.textSecondary, w);

        // Collision-free pill pass (priority: TTC/PET -> VRUs -> Vehicles)
        for (var cp = 0; cp < pendingConflictPills.length; cp += 1) {
            var cItem = pendingConflictPills[cp];
            placePill(ctx, cItem, cItem.text, pal.bgCard, cItem.border, pal.textPrimary, safeArea, placedBoxes, obstacleBoxes);
        }
        for (var vp = 0; vp < pendingVruPills.length; vp += 1) {
            var uItem = pendingVruPills[vp];
            placePill(ctx, uItem, uItem.text, pal.bgCard, uItem.border, pal.textPrimary, safeArea, placedBoxes, obstacleBoxes);
        }
        for (var qp = 0; qp < pendingVehiclePills.length; qp += 1) {
            var vItem = pendingVehiclePills[qp];
            placePill(ctx, vItem, vItem.text, pal.bgCard, vItem.border, pal.textPrimary, safeArea, placedBoxes, obstacleBoxes);
        }

        ctx.restore();
        if (this._onRenderState) {
            this._onRenderState(this._currentRenderState);
        }
    };

    CampusRoadTwinCanvas.prototype.getCurrentRenderState = function () {
        return this._currentRenderState;
    };

    function padId(n) {
        var s = String(n);
        return s.length < 2 ? "0" + s : s;
    }

    function findById(list, id) {
        if (id === null || id === undefined) return null;
        for (var i = 0; i < list.length; i += 1) {
            if (String(list[i].trackId) === String(id)) return list[i];
        }
        return null;
    }

    function boxOverlapArea(a, b, pad) {
        var p = pad || 0;
        var ox = Math.max(0, Math.min(a.x + a.w + p, b.x + b.w + p) - Math.max(a.x, b.x));
        var oy = Math.max(0, Math.min(a.y + a.h + p, b.y + b.h + p) - Math.max(a.y, b.y));
        return ox * oy;
    }

    function chooseBestBox(candidates, boxW, boxH, safeArea, placedBoxes, obstacleBoxes) {
        var minX = safeArea.minX;
        var maxX = Math.max(minX, safeArea.maxX - boxW);
        var minY = safeArea.minY;
        var maxY = Math.max(minY, safeArea.maxY - boxH);

        var expanded = [];
        var yNudges = [0, -22, 22, -44, 44];
        for (var n = 0; n < yNudges.length; n += 1) {
            for (var c = 0; c < candidates.length; c += 1) {
                var cx = clamp(candidates[c].x, minX, maxX);
                var cy = clamp(candidates[c].y + yNudges[n], minY, maxY);
                expanded.push({ x: cx, y: cy, w: boxW, h: boxH, nudge: Math.abs(yNudges[n]) + c * 2 });
            }
        }

        var best = expanded[0];
        var bestScore = Infinity;
        for (var i = 0; i < expanded.length; i += 1) {
            var cand = expanded[i];
            var pillHit = 0;
            for (var j = 0; j < placedBoxes.length; j += 1) {
                pillHit += boxOverlapArea(cand, placedBoxes[j], 4);
            }
            var bodyHit = 0;
            for (var k = 0; k < obstacleBoxes.length; k += 1) {
                bodyHit += boxOverlapArea(cand, obstacleBoxes[k], 2);
            }
            var score = pillHit * 20 + bodyHit * 2 + cand.nudge;
            if (score === 0) {
                return { x: cand.x, y: cand.y, w: boxW, h: boxH };
            }
            if (score < bestScore) {
                bestScore = score;
                best = cand;
            }
        }
        return { x: best.x, y: best.y, w: boxW, h: boxH };
    }

    function placePill(ctx, target, text, bg, border, fg, safeArea, placedBoxes, obstacleBoxes) {
        ctx.save();
        ctx.font = "500 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        var padX = 6;
        var h = 18;
        var boxW = ctx.measureText(text).width + padX * 2;
        var rx = target.rx || 8;
        var ry = target.ry || 8;

        var candidates = [
            { x: target.cx + rx + 8, y: target.cy - h * 0.5 },
            { x: target.cx - boxW * 0.5, y: target.cy - ry - h - 8 },
            { x: target.cx - rx - boxW - 8, y: target.cy - h * 0.5 },
            { x: target.cx + rx + 8, y: target.cy + ry + 6 },
            { x: target.cx - boxW * 0.5, y: target.cy + ry + 8 },
            { x: target.cx - rx - boxW - 8, y: target.cy - ry - h - 8 },
            { x: target.cx + rx + 8, y: target.cy - ry - h - 8 }
        ];

        var chosen = chooseBestBox(candidates, boxW, h, safeArea, placedBoxes, obstacleBoxes);
        placedBoxes.push(chosen);

        drawRoundedRectPath(ctx, chosen.x, chosen.y, chosen.w, chosen.h, 4);
        ctx.fillStyle = withAlpha(bg, 0.94);
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = border;
        ctx.stroke();

        ctx.fillStyle = fg;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(text, chosen.x + padX, chosen.y + chosen.h * 0.5 + 0.5);
        ctx.restore();
    }

    function drawRoundedRectPath(ctx, x, y, w, h, r) {
        var radius = Math.min(r, w * 0.5, h * 0.5);
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + w - radius, y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + radius);
        ctx.lineTo(x + w, y + h - radius);
        ctx.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
        ctx.lineTo(x + radius, y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
    }

    function drawPill(ctx, x, y, text, bg, border, fg, maxCanvasW) {
        ctx.save();
        ctx.font = "500 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        var padX = 6;
        var h = 18;
        var textW = ctx.measureText(text).width;
        var boxW = textW + padX * 2;
        var drawX = clamp(x, 6, Math.max(6, maxCanvasW - boxW - 6));
        var drawY = Math.max(6, y);

        drawRoundedRectPath(ctx, drawX, drawY, boxW, h, 4);
        ctx.fillStyle = withAlpha(bg, 0.92);
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = border;
        ctx.stroke();

        ctx.fillStyle = fg;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(text, drawX + padX, drawY + h * 0.5 + 0.5);
        ctx.restore();
    }

    CampusRoadTwinCanvas.prototype.setMode = function (mode) {
        if (this._destroyed) return;
        this._mode = mode === "replay" ? "replay" : "simulation";
        this._petCells = Object.create(null);
        this._replayPrevTracks = Object.create(null);
        this._lastReplayFrameIdx = -1;
        if (this._mode === "simulation") {
            this._seedInitialPetTrail();
            this._stepSimulation(0, true);
        } else {
            this._stopLoop();
            this._renderCurrent();
        }
        this._syncLoop();
    };

    CampusRoadTwinCanvas.prototype.setScenario = function (id) {
        if (this._destroyed) return;
        this._scenarioId = id || "crosswalk_braking";
        this._simTimeS = (this._userPaused || this._motionTracker.isReduced()) ? 0.72 : 0.25;
        this._seedInitialPetTrail();
        if (this._mode === "simulation") {
            this._stepSimulation(0, true);
        }
    };

    CampusRoadTwinCanvas.prototype.setConfig = function (config) {
        if (this._destroyed) return;
        var cfg = config || {};
        this._config = {
            speedModel: cfg.speedModel || this._config.speedModel,
            conflictModel: cfg.conflictModel || this._config.conflictModel,
            stitching: cfg.stitching !== undefined ? Boolean(cfg.stitching) : this._config.stitching,
            cyclistVru: cfg.cyclistVru !== undefined ? Boolean(cfg.cyclistVru) : this._config.cyclistVru
        };
        if (this._mode === "simulation") {
            this._seedInitialPetTrail();
            this._stepSimulation(0, true);
        }
    };

    CampusRoadTwinCanvas.prototype.setPaused = function (paused) {
        if (this._destroyed) return;
        this._userPaused = Boolean(paused);
        this._syncLoop();
        if (this._mode === "simulation") {
            this._emitSimTelemetry(true);
        }
    };

    CampusRoadTwinCanvas.prototype.isPaused = function () {
        return this._userPaused;
    };

    CampusRoadTwinCanvas.prototype.setReplayFrame = function (frameData) {
        if (this._destroyed || !frameData) return;
        var fd = frameData;
        var frameIdx = toNum(fd.frameIndex !== undefined ? fd.frameIndex : fd.frame_idx, 0);
        var ts = toNum(fd.timestampS !== undefined ? fd.timestampS : fd.timestamp_s, frameIdx * 0.1);

        if (frameIdx < this._lastReplayFrameIdx || Math.abs(frameIdx - this._lastReplayFrameIdx) > 3) {
            this._petCells = Object.create(null);
            this._replayPrevTracks = Object.create(null);
        }
        this._lastReplayFrameIdx = frameIdx;

        var rawVehicles = Array.isArray(fd.vehicles) ? fd.vehicles : [];
        var rawVrus = Array.isArray(fd.vrus) ? fd.vrus : [];
        var rawObjects = Array.isArray(fd.objects) ? fd.objects : (Array.isArray(fd.tracks) ? fd.tracks : []);
        for (var oi = 0; oi < rawObjects.length; oi += 1) {
            var obj = rawObjects[oi];
            var cat = String(obj.category || obj.kind || "").toLowerCase();
            if (cat === "pedestrian" || cat === "person" || cat === "bicycle" || cat === "motorcycle") {
                rawVrus.push(obj);
            } else {
                rawVehicles.push(obj);
            }
        }

        var self = this;
        var normVehicles = [];
        var normVrus = [];

        function normalizeTrack(item, isVeh) {
            var tid = item.trackId !== undefined ? item.trackId : (item.track_id !== undefined ? item.track_id : item.id);
            var x = toNum(item.x !== undefined ? item.x : item.x_m, NaN);
            var y = toNum(item.y !== undefined ? item.y : item.y_m, NaN);
            if (!Number.isFinite(x) || !Number.isFinite(y)) return null;

            var kind = String(item.kind || item.vehicle_kind || item.target_kind || item.category || (isVeh ? "car" : "pedestrian")).toLowerCase();
            var dims = VEHICLE_DIMS[kind] || (isVeh ? VEHICLE_DIMS.car : VEHICLE_DIMS.pedestrian);
            var spdKmh = toNum(item.speedKmh !== undefined ? item.speedKmh : item.speed_kmh, 0);
            var spdMs = toNum(item.speedMs !== undefined ? item.speedMs : item.speed_ms, spdKmh / 3.6);

            var vx = toNum(item.vx !== undefined ? item.vx : item.vx_ms, NaN);
            var vy = toNum(item.vy !== undefined ? item.vy : item.vy_ms, NaN);
            var headingDeg = toNum(item.headingDeg !== undefined ? item.headingDeg : item.heading_deg, NaN);

            var prev = self._replayPrevTracks[String(tid)];
            if ((!Number.isFinite(vx) || !Number.isFinite(vy)) && prev && ts > prev.ts) {
                var dt = ts - prev.ts;
                var estVx = (x - prev.x) / dt;
                var estVy = (y - prev.y) / dt;
                var mag = Math.hypot(estVx, estVy);
                if (mag > 0.15) {
                    vx = estVx;
                    vy = estVy;
                }
            }
            if (!Number.isFinite(vx) || !Number.isFinite(vy)) {
                if (Number.isFinite(headingDeg)) {
                    var hr = headingDeg * Math.PI / 180;
                    vx = Math.cos(hr) * spdMs;
                    vy = Math.sin(hr) * spdMs;
                } else {
                    vx = 0;
                    vy = isVeh ? spdMs : 0;
                }
            }
            var headingRad = Number.isFinite(headingDeg)
                ? headingDeg * Math.PI / 180
                : (Math.hypot(vx, vy) > 0.15 ? Math.atan2(vy, vx) : (prev ? prev.headingRad : Math.PI * 0.5));

            self._replayPrevTracks[String(tid)] = { x: x, y: y, ts: ts, headingRad: headingRad };

            return {
                trackId: tid,
                kind: kind,
                category: item.category || kind,
                x: x,
                y: y,
                vx: vx,
                vy: vy,
                accelMs2: toNum(item.accelMs2 !== undefined ? item.accelMs2 : item.ax_ms2, 0),
                headingRad: headingRad,
                yawRate: toNum(item.yawRate !== undefined ? item.yawRate : item.yaw_rate_rad_s, 0),
                lengthM: toNum(item.lengthM !== undefined ? item.lengthM : item.length_m, dims.lengthM),
                widthM: toNum(item.widthM !== undefined ? item.widthM : item.width_m, dims.widthM),
                radiusM: toNum(item.radiusM, 0.42),
                speedKmh: spdKmh,
                riskLevel: parseRiskLevel(item.riskLevel !== undefined ? item.riskLevel : item.risk_level),
                inParkingZone: Boolean(item.inParkingZone || item.in_parking_zone)
            };
        }

        for (var vi = 0; vi < rawVehicles.length; vi += 1) {
            var nv = normalizeTrack(rawVehicles[vi], true);
            if (nv) normVehicles.push(nv);
        }
        for (var pi = 0; pi < rawVrus.length; pi += 1) {
            var np = normalizeTrack(rawVrus[pi], false);
            if (np) normVrus.push(np);
        }

        var rawConflicts = Array.isArray(fd.conflicts) ? fd.conflicts : [];
        var normConflicts = [];
        for (var ci = 0; ci < rawConflicts.length; ci += 1) {
            var rc = rawConflicts[ci];
            var cObj = rc.assessment ? rc.assessment : rc;
            var rLvl = parseRiskLevel(cObj.riskLevel !== undefined ? cObj.riskLevel : cObj.risk_level);
            var vehTid = cObj.vehTrackId !== undefined ? cObj.vehTrackId : cObj.veh_track_id;
            var pedTid = cObj.pedTrackId !== undefined ? cObj.pedTrackId : cObj.ped_track_id;

            var matchedVeh = findById(normVehicles, vehTid);
            if (matchedVeh && rLvl > matchedVeh.riskLevel) matchedVeh.riskLevel = rLvl;
            var matchedVru = findById(normVrus, pedTid);
            if (matchedVru && rLvl > matchedVru.riskLevel) matchedVru.riskLevel = rLvl;

            normConflicts.push({
                vehTrackId: vehTid,
                pedTrackId: pedTid,
                riskLevel: rLvl,
                ttcSeconds: toNum(cObj.ttcSeconds !== undefined ? cObj.ttcSeconds : cObj.ttc_seconds, null),
                petSeconds: toNum(cObj.petSeconds !== undefined ? cObj.petSeconds : cObj.pet_seconds, null),
                minDistanceM: toNum(cObj.minDistanceM !== undefined ? cObj.minDistanceM : cObj.min_distance_m, null),
                conflictGroundXY: cObj.conflictGroundXY || cObj.conflict_ground_xy || null
            });
        }

        var bounds = normalizeBounds(fd.bounds);
        // Auto-expand bounds cleanly if no custom bounds passed and coordinates exceed default
        if (!fd.bounds) {
            var allPts = normVehicles.concat(normVrus);
            for (var bi = 0; bi < allPts.length; bi += 1) {
                if (allPts[bi].x - 1.0 < bounds.xMin) bounds.xMin = Math.floor(allPts[bi].x - 1.0);
                if (allPts[bi].x + 1.0 > bounds.xMax) bounds.xMax = Math.ceil(allPts[bi].x + 1.0);
                if (allPts[bi].y - 1.5 < bounds.yMin) bounds.yMin = Math.floor(allPts[bi].y - 1.5);
                if (allPts[bi].y + 1.5 > bounds.yMax) bounds.yMax = Math.ceil(allPts[bi].y + 1.5);
            }
        }

        this._recordPetGrid(normVehicles, normVrus, ts);

        this._currentRenderState = {
            bounds: bounds,
            parkingZones: Array.isArray(fd.parkingZones) ? fd.parkingZones : [],
            vehicles: normVehicles,
            vrus: normVrus,
            conflicts: normConflicts,
            label: fd.label || ("REPLAY · F" + padId(frameIdx) + " · " + ts.toFixed(1) + "s"),
            simTimeS: ts,
            maxRisk: 0
        };
        this._measureCanvas();
        this._renderCurrent();
    };

    CampusRoadTwinCanvas.prototype.refreshTheme = function () {
        if (this._destroyed) return;
        this._palette = readThemePalette(this._probe);
        this._measureCanvas();
        this._renderCurrent();
    };

    CampusRoadTwinCanvas.prototype.destroy = function () {
        if (this._destroyed) return;
        this._destroyed = true;
        this._stopLoop();
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }
        if (this._intersectionObserver) {
            this._intersectionObserver.disconnect();
            this._intersectionObserver = null;
        }
        this._unobserveTheme();
        this._motionTracker.destroy();
        if (this._probe && this._probe.parentNode) {
            this._probe.parentNode.removeChild(this._probe);
        }
    };

    /* ======================================================================
     * 4. SpeedTimelineChart (Dual-Band Speed & Conflict Risk Chart)
     * ====================================================================== */
    function SpeedTimelineChart(canvasEl, options) {
        if (!(this instanceof SpeedTimelineChart)) {
            return new SpeedTimelineChart(canvasEl, options);
        }
        if (!canvasEl) {
            throw new Error("SpeedTimelineChart requires a valid <canvas> element");
        }
        var opts = options || {};
        this._canvas = canvasEl;
        this._ctx = canvasEl.getContext("2d");
        this._speedLimitKmh = toNum(opts.speedLimitKmh, 30.0);
        this._onHover = typeof opts.onHover === "function" ? opts.onHover : null;
        this._onSelectFrame = typeof opts.onSelectFrame === "function" ? opts.onSelectFrame : null;
        this._points = [];
        this._activeFrameIndex = null;
        this._hoverIdx = -1;
        this._destroyed = false;

        this._cssWidth = 0;
        this._cssHeight = 0;
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);
        this._plotBox = null;

        this._probe = createColorProbe();
        this._palette = readThemePalette(this._probe);

        var self = this;
        this._unobserveTheme = observeThemeChanges(function () {
            self.refreshTheme();
        });

        this._onPointerMove = function (ev) {
            self._handlePointerMove(ev);
        };
        this._onPointerLeave = function () {
            if (self._hoverIdx !== -1) {
                self._hoverIdx = -1;
                if (self._onHover) self._onHover(null);
                self._render();
            }
        };
        this._onClick = function (ev) {
            self._handleClick(ev);
        };

        this._canvas.addEventListener("pointermove", this._onPointerMove);
        this._canvas.addEventListener("pointerleave", this._onPointerLeave);
        this._canvas.addEventListener("click", this._onClick);

        if (typeof ResizeObserver === "function") {
            this._resizeObserver = new ResizeObserver(function () {
                self._measure();
                self._render();
            });
            this._resizeObserver.observe(this._canvas);
        } else {
            this._resizeObserver = null;
        }

        this._measure();
        this._render();
    }

    SpeedTimelineChart.prototype._measure = function () {
        var rect = this._canvas.getBoundingClientRect();
        var w = Math.max(0, rect.width || this._canvas.clientWidth || 0);
        var h = Math.max(0, rect.height || this._canvas.clientHeight || 0);
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);
        this._cssWidth = w;
        this._cssHeight = h;
        if (w >= 2 && h >= 2) {
            var pw = Math.round(w * this._dpr);
            var ph = Math.round(h * this._dpr);
            if (this._canvas.width !== pw || this._canvas.height !== ph) {
                this._canvas.width = pw;
                this._canvas.height = ph;
            }
        }
    };

    SpeedTimelineChart.prototype._indexFromPointer = function (ev) {
        if (!this._plotBox || this._points.length === 0) return -1;
        var rect = this._canvas.getBoundingClientRect();
        var x = ev.clientX - rect.left;
        var box = this._plotBox;
        if (x < box.x - 10 || x > box.x + box.w + 10) return -1;
        if (this._points.length === 1) return 0;
        var ratio = clamp((x - box.x) / Math.max(1, box.w), 0, 1);
        return Math.round(ratio * (this._points.length - 1));
    };

    SpeedTimelineChart.prototype._handlePointerMove = function (ev) {
        var idx = this._indexFromPointer(ev);
        if (idx !== this._hoverIdx) {
            this._hoverIdx = idx;
            if (this._onHover) {
                this._onHover(idx >= 0 ? this._points[idx] : null);
            }
            this._render();
        }
    };

    SpeedTimelineChart.prototype._handleClick = function (ev) {
        var idx = this._indexFromPointer(ev);
        if (idx >= 0 && this._points[idx]) {
            this._activeFrameIndex = this._points[idx].frameIndex;
            this._render();
            if (this._onSelectFrame) {
                this._onSelectFrame(this._points[idx].frameIndex);
            }
        }
    };

    SpeedTimelineChart.prototype._render = function () {
        if (!this._ctx || this._cssWidth < 2 || this._cssHeight < 2) return;
        var ctx = this._ctx;
        var pal = this._palette;
        var w = this._cssWidth;
        var h = this._cssHeight;

        ctx.save();
        ctx.scale(this._dpr, this._dpr);
        ctx.clearRect(0, 0, w, h);

        ctx.fillStyle = pal.bgCard;
        ctx.fillRect(0, 0, w, h);

        var padL = 44;
        var padR = 16;
        var padT = 18;
        var padB = 24;
        var plotW = Math.max(30, w - padL - padR);
        var plotH = Math.max(40, h - padT - padB);
        this._plotBox = { x: padL, y: padT, w: plotW, h: plotH };

        if (this._points.length === 0) {
            ctx.fillStyle = pal.textMuted;
            ctx.font = "400 12px system-ui, -apple-system, sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("暂无时序轨迹数据", w * 0.5, h * 0.5);
            ctx.restore();
            return;
        }

        // Dual band geometry: top 64% speed line, bottom 28% conflict/TTC markers
        var topBandH = plotH * 0.64;
        var gapH = plotH * 0.08;
        var botBandY = padT + topBandH + gapH;
        var botBandH = plotH - topBandH - gapH;

        var maxSpd = this._speedLimitKmh * 1.25;
        for (var i = 0; i < this._points.length; i += 1) {
            if (this._points[i].speedKmh > maxSpd) maxSpd = this._points[i].speedKmh * 1.15;
        }
        maxSpd = Math.max(10, Math.ceil(maxSpd / 5) * 5);

        function xAt(idx, total) {
            if (total <= 1) return padL + plotW * 0.5;
            return padL + (idx / (total - 1)) * plotW;
        }
        function ySpeed(spd) {
            return padT + topBandH - clamp(spd / maxSpd, 0, 1) * topBandH;
        }

        // Horizontal Y ticks & speed limit dashed hairline
        ctx.font = "500 10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        var ticks = [0, Math.round(maxSpd * 0.5), maxSpd];
        for (var ti = 0; ti < ticks.length; ti += 1) {
            var ty = ySpeed(ticks[ti]);
            ctx.strokeStyle = pal.gridLine;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(padL, ty);
            ctx.lineTo(padL + plotW, ty);
            ctx.stroke();

            ctx.fillStyle = pal.textMuted;
            ctx.textAlign = "right";
            ctx.textBaseline = "middle";
            ctx.fillText(String(ticks[ti]), padL - 6, ty);
        }

        var limitY = ySpeed(this._speedLimitKmh);
        if (limitY >= padT - 4 && limitY <= padT + topBandH + 4) {
            ctx.save();
            ctx.setLineDash([5, 4]);
            ctx.strokeStyle = withAlpha(pal.accent, 0.78);
            ctx.lineWidth = 1.2;
            ctx.beginPath();
            ctx.moveTo(padL, limitY);
            ctx.lineTo(padL + plotW, limitY);
            ctx.stroke();
            ctx.restore();

            ctx.fillStyle = pal.accent;
            ctx.textAlign = "right";
            ctx.textBaseline = "bottom";
            ctx.fillText(this._speedLimitKmh.toFixed(0) + " km/h", padL + plotW, limitY - 3);
        }

        // Band divider hairline
        var divY = padT + topBandH + gapH * 0.5;
        ctx.strokeStyle = pal.hairline;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(padL, divY);
        ctx.lineTo(padL + plotW, divY);
        ctx.stroke();

        // Speed area fill & stroke
        var n = this._points.length;
        ctx.save();
        ctx.beginPath();
        for (var j = 0; j < n; j += 1) {
            var px = xAt(j, n);
            var py = ySpeed(this._points[j].speedKmh);
            if (j === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.strokeStyle = pal.textPrimary;
        ctx.lineWidth = 1.8;
        ctx.stroke();

        ctx.lineTo(xAt(n - 1, n), padT + topBandH);
        ctx.lineTo(xAt(0, n), padT + topBandH);
        ctx.closePath();
        var grad = ctx.createLinearGradient(0, padT, 0, padT + topBandH);
        grad.addColorStop(0, withAlpha(pal.textPrimary, 0.14));
        grad.addColorStop(1, withAlpha(pal.textPrimary, 0.0));
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.restore();

        // Bottom band: risk-colored markers + TTC severity stems
        ctx.fillStyle = pal.textMuted;
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText("RISK", padL - 6, botBandY + botBandH * 0.5);

        for (var m = 0; m < n; m += 1) {
            var pt = this._points[m];
            var mx = xAt(m, n);
            var rLvl = pt.riskLevel;
            var rColor = rLvl === 0 ? withAlpha(pal.textMuted, 0.35) : riskColor(pal, rLvl);

            var stemRatio = 0.18;
            if (rLvl >= 2) {
                stemRatio = pt.ttcS !== null ? clamp(1 - pt.ttcS / 4.0, 0.4, 0.95) : (rLvl === 3 ? 0.9 : 0.6);
            } else if (rLvl === 1) {
                stemRatio = 0.42;
            }
            var stemH = Math.max(3, botBandH * stemRatio);
            ctx.strokeStyle = rColor;
            ctx.lineWidth = rLvl >= 2 ? 2.6 : 1.5;
            ctx.beginPath();
            ctx.moveTo(mx, botBandY + botBandH);
            ctx.lineTo(mx, botBandY + botBandH - stemH);
            ctx.stroke();

            if (rLvl >= 2) {
                ctx.fillStyle = rColor;
                ctx.beginPath();
                ctx.arc(mx, botBandY + botBandH - stemH, 2.5, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // X axis time ticks
        var xTickIndices = n <= 2 ? [0, n - 1] : [0, Math.floor((n - 1) * 0.5), n - 1];
        for (var xi = 0; xi < xTickIndices.length; xi += 1) {
            var idx = xTickIndices[xi];
            if (idx < 0 || idx >= n) continue;
            var tx = xAt(idx, n);
            ctx.fillStyle = pal.textMuted;
            ctx.textAlign = idx === 0 ? "left" : (idx === n - 1 ? "right" : "center");
            ctx.textBaseline = "top";
            ctx.fillText(this._points[idx].t.toFixed(1) + "s", tx, padT + plotH + 6);
        }

        // Active frame cursor
        if (this._activeFrameIndex !== null) {
            for (var ai = 0; ai < n; ai += 1) {
                if (this._points[ai].frameIndex === this._activeFrameIndex) {
                    var ax = xAt(ai, n);
                    ctx.strokeStyle = pal.accent;
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.moveTo(ax, padT);
                    ctx.lineTo(ax, padT + plotH);
                    ctx.stroke();

                    ctx.fillStyle = pal.accent;
                    ctx.beginPath();
                    ctx.arc(ax, ySpeed(this._points[ai].speedKmh), 3.8, 0, Math.PI * 2);
                    ctx.fill();
                    break;
                }
            }
        }

        // Hover crosshair & compact telemetry readout
        if (this._hoverIdx >= 0 && this._hoverIdx < n) {
            var hp = this._points[this._hoverIdx];
            var hx = xAt(this._hoverIdx, n);
            ctx.save();
            ctx.setLineDash([3, 3]);
            ctx.strokeStyle = pal.textSecondary;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(hx, padT);
            ctx.lineTo(hx, padT + plotH);
            ctx.stroke();
            ctx.restore();

            ctx.fillStyle = pal.bgCard;
            ctx.strokeStyle = pal.textPrimary;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(hx, ySpeed(hp.speedKmh), 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();

            var ttcLabel = hp.ttcS !== null ? (" · TTC " + hp.ttcS.toFixed(2) + "s") : "";
            var tipText = "F" + hp.frameIndex + " · " + hp.t.toFixed(1) + "s · " + hp.speedKmh.toFixed(1) + "km/h" + ttcLabel + " · L" + hp.riskLevel;
            drawPill(ctx, hx + 8, padT + 2, tipText, pal.bgRaised, pal.hairline, pal.textPrimary, w);
        }

        ctx.restore();
    };

    SpeedTimelineChart.prototype.setData = function (points, speedLimitKmh) {
        if (this._destroyed) return;
        if (speedLimitKmh !== undefined) {
            this._speedLimitKmh = toNum(speedLimitKmh, this._speedLimitKmh);
        }
        var raw = Array.isArray(points) ? points : [];
        var normalized = [];
        for (var i = 0; i < raw.length; i += 1) {
            var p = raw[i];
            if (!p) continue;
            var frameIndex = toNum(p.frameIndex !== undefined ? p.frameIndex : p.frame_idx, i);
            var t = toNum(p.t !== undefined ? p.t : (p.timestampS !== undefined ? p.timestampS : p.timestamp_s), frameIndex * 0.1);
            var speedKmh = toNum(p.speedKmh !== undefined ? p.speedKmh : p.speed_kmh, 0);
            var ttcS = toNum(p.ttcS !== undefined ? p.ttcS : (p.ttc_s !== undefined ? p.ttc_s : p.ttc_seconds), null);
            var riskLevel = parseRiskLevel(p.riskLevel !== undefined ? p.riskLevel : p.risk_level);
            normalized.push({
                frameIndex: frameIndex,
                t: t,
                speedKmh: speedKmh,
                ttcS: ttcS,
                riskLevel: riskLevel
            });
        }
        this._points = normalized;
        this._hoverIdx = -1;
        this._measure();
        this._render();
    };

    SpeedTimelineChart.prototype.setActiveFrame = function (frameIndex) {
        if (this._destroyed) return;
        this._activeFrameIndex = frameIndex === null || frameIndex === undefined ? null : toNum(frameIndex, null);
        this._measure();
        this._render();
    };

    SpeedTimelineChart.prototype.refreshTheme = function () {
        if (this._destroyed) return;
        this._palette = readThemePalette(this._probe);
        this._measure();
        this._render();
    };

    SpeedTimelineChart.prototype.destroy = function () {
        if (this._destroyed) return;
        this._destroyed = true;
        this._canvas.removeEventListener("pointermove", this._onPointerMove);
        this._canvas.removeEventListener("pointerleave", this._onPointerLeave);
        this._canvas.removeEventListener("click", this._onClick);
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }
        this._unobserveTheme();
        if (this._probe && this._probe.parentNode) {
            this._probe.parentNode.removeChild(this._probe);
        }
    };

    /* ======================================================================
     * 5. CalibrationCanvas (Interactive Homography & Parking Zone Editor)
     * ====================================================================== */
    function roundCoord1(val) {
        return Math.round(val * 10) / 10;
    }

    function clonePointPair(pt) {
        if (!Array.isArray(pt) || pt.length < 2) return null;
        var u = toNum(pt[0], NaN);
        var v = toNum(pt[1], NaN);
        if (!Number.isFinite(u) || !Number.isFinite(v)) return null;
        return [u, v];
    }

    function clonePointList(list) {
        var out = [];
        if (!Array.isArray(list)) return out;
        for (var i = 0; i < list.length; i += 1) {
            var pair = clonePointPair(list[i]);
            if (pair) out.push(pair);
        }
        return out;
    }

    function clonePolygonList(polys) {
        var out = [];
        if (!Array.isArray(polys)) return out;
        for (var i = 0; i < polys.length; i += 1) {
            var poly = clonePointList(polys[i]);
            if (poly.length >= 3) out.push(poly);
        }
        return out;
    }

    function CalibrationCanvas(canvasEl, options) {
        if (!(this instanceof CalibrationCanvas)) {
            return new CalibrationCanvas(canvasEl, options);
        }
        if (!canvasEl) {
            throw new Error("CalibrationCanvas requires a valid <canvas> element");
        }
        var opts = options || {};
        this._canvas = canvasEl;
        this._ctx = canvasEl.getContext("2d");
        this._onChange = typeof opts.onChange === "function" ? opts.onChange : null;

        this._tool = "points";
        this._image = null;
        this._imageReady = false;
        this._imageWidth = 0;
        this._imageHeight = 0;
        this._loadToken = 0;

        this._imagePoints = [];
        this._parkingZonesImage = [];
        this._currentParkingDraft = [];

        this._dragState = null;
        this._hoverVertexIdx = -1;
        this._hoverPos = null;
        this._lastEmitTs = 0;
        this._emitTimer = null;
        this._destroyed = false;

        this._cssWidth = 0;
        this._cssHeight = 0;
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);

        this._probe = createColorProbe();
        this._palette = readThemePalette(this._probe);

        if (this._canvas.style && !this._canvas.style.touchAction) {
            this._canvas.style.touchAction = "none";
        }

        var self = this;
        this._motionTracker = createReducedMotionTracker(function () {
            if (!self._destroyed) {
                self._render();
            }
        });
        this._unobserveTheme = observeThemeChanges(function () {
            self.refreshTheme();
        });

        this._onPointerDown = function (ev) {
            self._handlePointerDown(ev);
        };
        this._onPointerMove = function (ev) {
            self._handlePointerMove(ev);
        };
        this._onPointerUp = function (ev) {
            self._handlePointerUp(ev);
        };
        this._onPointerCancel = function (ev) {
            self._handlePointerUp(ev);
        };
        this._onPointerLeave = function () {
            if (!self._dragState && (self._hoverVertexIdx !== -1 || self._hoverPos !== null)) {
                self._hoverVertexIdx = -1;
                self._hoverPos = null;
                self._updateCursor();
                self._render();
            }
        };

        this._canvas.addEventListener("pointerdown", this._onPointerDown);
        this._canvas.addEventListener("pointermove", this._onPointerMove);
        this._canvas.addEventListener("pointerup", this._onPointerUp);
        this._canvas.addEventListener("pointercancel", this._onPointerCancel);
        this._canvas.addEventListener("pointerleave", this._onPointerLeave);

        if (typeof ResizeObserver === "function") {
            this._resizeObserver = new ResizeObserver(function () {
                self._measure();
                self._render();
            });
            this._resizeObserver.observe(this._canvas);
        } else {
            this._resizeObserver = null;
        }

        this._measure();
        this._updateCursor();
        this._render();
    }

    CalibrationCanvas.prototype._measure = function () {
        var rect = this._canvas.getBoundingClientRect();
        var w = Math.max(0, rect.width || this._canvas.clientWidth || 0);
        var h = Math.max(0, rect.height || this._canvas.clientHeight || 0);
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);
        this._cssWidth = w;
        this._cssHeight = h;
        if (w >= 2 && h >= 2) {
            var pw = Math.round(w * this._dpr);
            var ph = Math.round(h * this._dpr);
            if (this._canvas.width !== pw || this._canvas.height !== ph) {
                this._canvas.width = pw;
                this._canvas.height = ph;
            }
        }
    };

    CalibrationCanvas.prototype._hasFrameDimensions = function () {
        return this._imageWidth > 0 && this._imageHeight > 0;
    };

    CalibrationCanvas.prototype._computeFitRect = function () {
        var w = this._cssWidth > 2
            ? this._cssWidth
            : (this._canvas.clientWidth || (this._canvas.width / (this._dpr || 1)) || 640);
        var h = this._cssHeight > 2
            ? this._cssHeight
            : (this._canvas.clientHeight || (this._canvas.height / (this._dpr || 1)) || 360);
        var pad = clamp(Math.floor(Math.min(w, h) * 0.045), 10, 18);
        var availW = Math.max(20, w - pad * 2);
        var availH = Math.max(20, h - pad * 2);
        var imgW = this._imageWidth > 0 ? this._imageWidth : 1280;
        var imgH = this._imageHeight > 0 ? this._imageHeight : 720;
        var scale = Math.min(availW / imgW, availH / imgH);
        var drawW = imgW * scale;
        var drawH = imgH * scale;
        return {
            x: pad + (availW - drawW) * 0.5,
            y: pad + (availH - drawH) * 0.5,
            w: drawW,
            h: drawH,
            scale: scale,
            imgW: imgW,
            imgH: imgH
        };
    };

    CalibrationCanvas.prototype.imageToCanvas = function (u, v) {
        var imgU = u;
        var imgV = v;
        if (Array.isArray(u)) {
            imgU = u[0];
            imgV = u[1];
        } else if (u && typeof u === "object") {
            imgU = u.u !== undefined ? u.u : u.x;
            imgV = u.v !== undefined ? u.v : u.y;
        }
        var fit = this._computeFitRect();
        return {
            x: fit.x + toNum(imgU, 0) * fit.scale,
            y: fit.y + toNum(imgV, 0) * fit.scale
        };
    };

    CalibrationCanvas.prototype.canvasToImage = function (cx, cy) {
        var px = cx;
        var py = cy;
        if (Array.isArray(cx)) {
            px = cx[0];
            py = cx[1];
        } else if (cx && typeof cx === "object") {
            px = cx.x;
            py = cx.y;
        }
        var fit = this._computeFitRect();
        var rawU = (toNum(px, fit.x) - fit.x) / Math.max(1e-6, fit.scale);
        var rawV = (toNum(py, fit.y) - fit.y) / Math.max(1e-6, fit.scale);
        return {
            u: clamp(roundCoord1(rawU), 0, fit.imgW),
            v: clamp(roundCoord1(rawV), 0, fit.imgH)
        };
    };

    CalibrationCanvas.prototype._pointerToCanvas = function (ev) {
        var rect = this._canvas.getBoundingClientRect();
        var x = ev.clientX - (rect.left || 0);
        var y = ev.clientY - (rect.top || 0);
        if (rect.width > 1 && this._cssWidth > 1 && Math.abs(rect.width - this._cssWidth) > 1) {
            x = x * (this._cssWidth / rect.width);
        }
        if (rect.height > 1 && this._cssHeight > 1 && Math.abs(rect.height - this._cssHeight) > 1) {
            y = y * (this._cssHeight / rect.height);
        }
        return { x: x, y: y };
    };

    CalibrationCanvas.prototype._activeVertexList = function () {
        return this._tool === "parking" ? this._currentParkingDraft : this._imagePoints;
    };

    CalibrationCanvas.prototype._findNearestVertex = function (cx, cy, list, maxDistPx) {
        var threshold = maxDistPx !== undefined ? maxDistPx : 14;
        var bestIdx = -1;
        var bestDist = Infinity;
        for (var i = 0; i < list.length; i += 1) {
            var pt = this.imageToCanvas(list[i][0], list[i][1]);
            var d = Math.hypot(pt.x - cx, pt.y - cy);
            if (d <= threshold && d < bestDist) {
                bestDist = d;
                bestIdx = i;
            }
        }
        return bestIdx;
    };

    CalibrationCanvas.prototype._updateCursor = function () {
        if (!this._canvas || !this._canvas.style) return;
        if (!this._hasFrameDimensions()) {
            this._canvas.style.cursor = "default";
        } else if (this._dragState) {
            this._canvas.style.cursor = "grabbing";
        } else if (this._hoverVertexIdx >= 0) {
            this._canvas.style.cursor = "grab";
        } else {
            this._canvas.style.cursor = "crosshair";
        }
    };

    CalibrationCanvas.prototype._emitChange = function (immediate) {
        if (this._destroyed || !this._onChange) return;
        var now = typeof performance !== "undefined" ? performance.now() : Date.now();
        if (immediate) {
            if (this._emitTimer !== null && typeof global.clearTimeout === "function") {
                global.clearTimeout(this._emitTimer);
                this._emitTimer = null;
            }
            this._lastEmitTs = now;
            this._onChange(this.getState());
            return;
        }
        var elapsed = now - this._lastEmitTs;
        if (elapsed >= 48) {
            if (this._emitTimer !== null && typeof global.clearTimeout === "function") {
                global.clearTimeout(this._emitTimer);
                this._emitTimer = null;
            }
            this._lastEmitTs = now;
            this._onChange(this.getState());
        } else if (this._emitTimer === null && typeof global.setTimeout === "function") {
            var self = this;
            var wait = Math.max(10, 48 - elapsed);
            this._emitTimer = global.setTimeout(function () {
                self._emitTimer = null;
                if (self._destroyed || !self._onChange) return;
                self._lastEmitTs = typeof performance !== "undefined" ? performance.now() : Date.now();
                self._onChange(self.getState());
            }, wait);
        }
    };

    CalibrationCanvas.prototype._handlePointerDown = function (ev) {
        if (this._destroyed) return;
        if (ev.button !== undefined && ev.button !== 0) return;
        if (!this._hasFrameDimensions()) return;

        var pos = this._pointerToCanvas(ev);
        this._hoverPos = pos;
        var list = this._activeVertexList();
        var hitIdx = this._findNearestVertex(pos.x, pos.y, list, 14);

        if (hitIdx >= 0) {
            this._dragState = {
                target: this._tool,
                index: hitIdx,
                pointerId: ev.pointerId,
                moved: false
            };
            this._hoverVertexIdx = hitIdx;
            if (ev.pointerId !== undefined && typeof this._canvas.setPointerCapture === "function") {
                try {
                    this._canvas.setPointerCapture(ev.pointerId);
                } catch (_err) {
                    // Ignore synthetic pointer capture errors
                }
            }
            this._updateCursor();
            this._render();
            return;
        }

        var fit = this._computeFitRect();
        var margin = 12;
        if (pos.x < fit.x - margin || pos.x > fit.x + fit.w + margin ||
            pos.y < fit.y - margin || pos.y > fit.y + fit.h + margin) {
            return;
        }

        var imgPt = this.canvasToImage(pos.x, pos.y);
        list.push([imgPt.u, imgPt.v]);
        this._hoverVertexIdx = list.length - 1;
        this._updateCursor();
        this._render();
        this._emitChange(true);
    };

    CalibrationCanvas.prototype._handlePointerMove = function (ev) {
        if (this._destroyed) return;
        if (!this._hasFrameDimensions()) return;

        var pos = this._pointerToCanvas(ev);
        this._hoverPos = pos;

        if (this._dragState) {
            if (ev.pointerId !== undefined &&
                this._dragState.pointerId !== undefined &&
                ev.pointerId !== this._dragState.pointerId) {
                return;
            }
            var list = this._dragState.target === "parking"
                ? this._currentParkingDraft
                : this._imagePoints;
            var idx = this._dragState.index;
            if (idx >= 0 && idx < list.length) {
                var imgPt = this.canvasToImage(pos.x, pos.y);
                var prev = list[idx];
                if (prev[0] !== imgPt.u || prev[1] !== imgPt.v) {
                    list[idx] = [imgPt.u, imgPt.v];
                    this._dragState.moved = true;
                    this._render();
                    this._emitChange(false);
                }
            }
            return;
        }

        var activeList = this._activeVertexList();
        var hitIdx = this._findNearestVertex(pos.x, pos.y, activeList, 14);
        var changed = hitIdx !== this._hoverVertexIdx;
        this._hoverVertexIdx = hitIdx;
        this._updateCursor();
        if (changed || activeList.length > 0) {
            this._render();
        }
    };

    CalibrationCanvas.prototype._handlePointerUp = function (ev) {
        if (this._destroyed || !this._dragState) return;
        if (ev.pointerId !== undefined &&
            this._dragState.pointerId !== undefined &&
            ev.pointerId !== this._dragState.pointerId) {
            return;
        }

        var wasMoved = this._dragState.moved;
        var ptrId = this._dragState.pointerId;
        this._dragState = null;

        if (ptrId !== undefined &&
            typeof this._canvas.hasPointerCapture === "function" &&
            typeof this._canvas.releasePointerCapture === "function") {
            try {
                if (this._canvas.hasPointerCapture(ptrId)) {
                    this._canvas.releasePointerCapture(ptrId);
                }
            } catch (_err) {
                // Ignore synthetic pointer release errors
            }
        }

        this._updateCursor();
        this._render();
        if (wasMoved || this._emitTimer !== null) {
            this._emitChange(true);
        }
    };

    CalibrationCanvas.prototype._drawCenteredBadge = function (ctx, cx, cy, text, bg, border, fg, canvasW, canvasH) {
        ctx.save();
        ctx.font = "500 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
        var padX = 6;
        var h = 18;
        var boxW = ctx.measureText(text).width + padX * 2;
        var x = clamp(cx - boxW * 0.5, 6, Math.max(6, canvasW - boxW - 6));
        var y = clamp(cy - h * 0.5, 6, Math.max(6, canvasH - h - 6));
        drawRoundedRectPath(ctx, x, y, boxW, h, 4);
        ctx.fillStyle = withAlpha(bg, 0.92);
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = border;
        ctx.stroke();
        ctx.fillStyle = fg;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(text, x + padX, y + h * 0.5 + 0.5);
        ctx.restore();
    };

    CalibrationCanvas.prototype._render = function () {
        if (!this._ctx || this._cssWidth < 2 || this._cssHeight < 2) return;
        var ctx = this._ctx;
        var pal = this._palette;
        var w = this._cssWidth;
        var h = this._cssHeight;

        ctx.save();
        ctx.scale(this._dpr, this._dpr);
        ctx.clearRect(0, 0, w, h);

        // Backdrop & subtle engineering dot pattern
        ctx.fillStyle = pal.roadShoulder;
        ctx.fillRect(0, 0, w, h);

        ctx.fillStyle = withAlpha(pal.textMuted, pal.isLight ? 0.14 : 0.12);
        var dotStep = 24;
        for (var dx = 12; dx < w; dx += dotStep) {
            for (var dy = 12; dy < h; dy += dotStep) {
                ctx.fillRect(dx, dy, 1, 1);
            }
        }

        var fit = this._computeFitRect();

        if (this._imageReady && this._image) {
            ctx.drawImage(this._image, fit.x, fit.y, fit.w, fit.h);
            ctx.strokeStyle = pal.hairline;
            ctx.lineWidth = 1;
            ctx.strokeRect(fit.x, fit.y, fit.w, fit.h);
        } else {
            // Crisp instrumented guide grid when no frame image is rendered
            ctx.fillStyle = pal.bgCard;
            ctx.fillRect(fit.x, fit.y, fit.w, fit.h);
            ctx.strokeStyle = pal.hairline;
            ctx.lineWidth = 1;
            ctx.strokeRect(fit.x, fit.y, fit.w, fit.h);

            var cols = 12;
            var rows = 8;
            ctx.strokeStyle = pal.gridLine;
            ctx.lineWidth = 1;
            for (var ci = 1; ci < cols; ci += 1) {
                var gx = fit.x + (fit.w * ci) / cols;
                ctx.beginPath();
                ctx.moveTo(gx, fit.y);
                ctx.lineTo(gx, fit.y + fit.h);
                ctx.stroke();
            }
            for (var ri = 1; ri < rows; ri += 1) {
                var gy = fit.y + (fit.h * ri) / rows;
                ctx.beginPath();
                ctx.moveTo(fit.x, gy);
                ctx.lineTo(fit.x + fit.w, gy);
                ctx.stroke();
            }

            // Center crosshair & corner calibration ticks
            var midX = fit.x + fit.w * 0.5;
            var midY = fit.y + fit.h * 0.5;
            ctx.strokeStyle = withAlpha(pal.roadMark, 0.85);
            ctx.beginPath();
            ctx.moveTo(midX - 24, midY);
            ctx.lineTo(midX + 24, midY);
            ctx.moveTo(midX, midY - 24);
            ctx.lineTo(midX, midY + 24);
            ctx.stroke();

            ctx.strokeStyle = withAlpha(pal.accent, 0.40);
            ctx.beginPath();
            ctx.arc(midX, midY, 16, 0, Math.PI * 2);
            ctx.stroke();

            var arm = 14;
            ctx.strokeStyle = withAlpha(pal.accent, 0.65);
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            // Top-left
            ctx.moveTo(fit.x, fit.y + arm);
            ctx.lineTo(fit.x, fit.y);
            ctx.lineTo(fit.x + arm, fit.y);
            // Top-right
            ctx.moveTo(fit.x + fit.w - arm, fit.y);
            ctx.lineTo(fit.x + fit.w, fit.y);
            ctx.lineTo(fit.x + fit.w, fit.y + arm);
            // Bottom-left
            ctx.moveTo(fit.x, fit.y + fit.h - arm);
            ctx.lineTo(fit.x, fit.y + fit.h);
            ctx.lineTo(fit.x + arm, fit.y + fit.h);
            // Bottom-right
            ctx.moveTo(fit.x + fit.w - arm, fit.y + fit.h);
            ctx.lineTo(fit.x + fit.w, fit.y + fit.h);
            ctx.lineTo(fit.x + fit.w, fit.y + fit.h - arm);
            ctx.stroke();

            if (!this._hasFrameDimensions()) {
                ctx.fillStyle = pal.textSecondary;
                ctx.font = "500 13px system-ui, -apple-system, sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("等待加载视频标定帧", midX, midY - 12);

                ctx.fillStyle = pal.textMuted;
                ctx.font = "400 11px system-ui, -apple-system, sans-serif";
                ctx.fillText("选择视频源并提取标定帧后，可点击设置 P1..PN 参考点与禁停区", midX, midY + 12);
            }

            ctx.fillStyle = pal.textMuted;
            ctx.font = "500 10px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
            ctx.textAlign = "right";
            ctx.textBaseline = "bottom";
            var specTag = this._hasFrameDimensions()
                ? (this._imageWidth + "x" + this._imageHeight + " px")
                : "CALIBRATION GRID · 16:9";
            ctx.fillText(specTag, fit.x + fit.w - 10, fit.y + fit.h - 8);
        }

        // 1. Committed roadside parking exclusion zones
        for (var zi = 0; zi < this._parkingZonesImage.length; zi += 1) {
            var poly = this._parkingZonesImage[zi];
            if (!Array.isArray(poly) || poly.length < 3) continue;
            var sumX = 0;
            var sumY = 0;
            var zPts = [];
            for (var zp = 0; zp < poly.length; zp += 1) {
                var cPt = this.imageToCanvas(poly[zp][0], poly[zp][1]);
                zPts.push(cPt);
                sumX += cPt.x;
                sumY += cPt.y;
            }
            ctx.save();
            ctx.beginPath();
            for (var k = 0; k < zPts.length; k += 1) {
                if (k === 0) ctx.moveTo(zPts[k].x, zPts[k].y);
                else ctx.lineTo(zPts[k].x, zPts[k].y);
            }
            ctx.closePath();
            ctx.fillStyle = withAlpha(pal.parkingZone, 0.18);
            ctx.fill();
            ctx.setLineDash([6, 4]);
            ctx.strokeStyle = withAlpha(pal.parkingZone, 0.92);
            ctx.lineWidth = 1.8;
            ctx.stroke();
            ctx.restore();

            for (var zv = 0; zv < zPts.length; zv += 1) {
                ctx.fillStyle = pal.bgCard;
                ctx.strokeStyle = pal.parkingZone;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(zPts[zv].x, zPts[zv].y, 3.2, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            }

            var zoneLabel = this._parkingZonesImage.length > 1
                ? ("禁停标定区 #" + (zi + 1))
                : "禁停标定区";
            this._drawCenteredBadge(
                ctx,
                sumX / zPts.length,
                sumY / zPts.length,
                zoneLabel,
                pal.bgCard,
                withAlpha(pal.parkingZone, 0.78),
                pal.parkingZone,
                w,
                h
            );
        }

        // 2. Active roadside parking exclusion draft polygon
        if (this._currentParkingDraft.length > 0) {
            var dPts = [];
            var dSumX = 0;
            var dSumY = 0;
            for (var di = 0; di < this._currentParkingDraft.length; di += 1) {
                var dPt = this.imageToCanvas(this._currentParkingDraft[di][0], this._currentParkingDraft[di][1]);
                dPts.push(dPt);
                dSumX += dPt.x;
                dSumY += dPt.y;
            }

            if (dPts.length >= 3) {
                ctx.save();
                ctx.beginPath();
                for (var df = 0; df < dPts.length; df += 1) {
                    if (df === 0) ctx.moveTo(dPts[df].x, dPts[df].y);
                    else ctx.lineTo(dPts[df].x, dPts[df].y);
                }
                ctx.closePath();
                ctx.fillStyle = withAlpha(pal.accent, 0.14);
                ctx.fill();
                ctx.setLineDash([3, 4]);
                ctx.strokeStyle = withAlpha(pal.accent, 0.55);
                ctx.lineWidth = 1.3;
                ctx.beginPath();
                ctx.moveTo(dPts[dPts.length - 1].x, dPts[dPts.length - 1].y);
                ctx.lineTo(dPts[0].x, dPts[0].y);
                ctx.stroke();
                ctx.restore();
            }

            if (dPts.length >= 2) {
                ctx.save();
                ctx.setLineDash([6, 4]);
                ctx.strokeStyle = pal.accent;
                ctx.lineWidth = 2.0;
                ctx.beginPath();
                for (var dl = 0; dl < dPts.length; dl += 1) {
                    if (dl === 0) ctx.moveTo(dPts[dl].x, dPts[dl].y);
                    else ctx.lineTo(dPts[dl].x, dPts[dl].y);
                }
                ctx.stroke();
                ctx.restore();
            }

            if (this._tool === "parking" && this._hoverPos && !this._dragState && this._hoverVertexIdx === -1) {
                var lastDraftPt = dPts[dPts.length - 1];
                ctx.save();
                ctx.setLineDash([3, 4]);
                ctx.strokeStyle = withAlpha(pal.accent, 0.45);
                ctx.lineWidth = 1.2;
                ctx.beginPath();
                ctx.moveTo(lastDraftPt.x, lastDraftPt.y);
                ctx.lineTo(this._hoverPos.x, this._hoverPos.y);
                ctx.stroke();
                ctx.restore();
            }

            for (var dv = 0; dv < dPts.length; dv += 1) {
                var draftActive = this._tool === "parking" &&
                    ((this._dragState && this._dragState.target === "parking" && this._dragState.index === dv) ||
                     this._hoverVertexIdx === dv);
                ctx.fillStyle = pal.bgCard;
                ctx.strokeStyle = pal.accent;
                ctx.lineWidth = draftActive ? 2.4 : 1.8;
                ctx.beginPath();
                ctx.arc(dPts[dv].x, dPts[dv].y, draftActive ? 6.5 : 4.5, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();
            }

            var draftLabelX = dPts.length >= 3 ? (dSumX / dPts.length) : dPts[0].x;
            var draftLabelY = dPts.length >= 3 ? (dSumY / dPts.length) : (dPts[0].y - 18);
            this._drawCenteredBadge(
                ctx,
                draftLabelX,
                draftLabelY,
                "禁停标定区",
                pal.bgCard,
                pal.accent,
                pal.accent,
                w,
                h
            );
        }

        // 3. Ground-correspondence calibration polygon & P1..PN vertices
        if (this._imagePoints.length > 0) {
            var cPts = [];
            for (var pi = 0; pi < this._imagePoints.length; pi += 1) {
                cPts.push(this.imageToCanvas(this._imagePoints[pi][0], this._imagePoints[pi][1]));
            }

            if (cPts.length >= 3) {
                ctx.save();
                ctx.beginPath();
                for (var pf = 0; pf < cPts.length; pf += 1) {
                    if (pf === 0) ctx.moveTo(cPts[pf].x, cPts[pf].y);
                    else ctx.lineTo(cPts[pf].x, cPts[pf].y);
                }
                ctx.closePath();
                ctx.fillStyle = withAlpha(pal.accent, 0.13);
                ctx.fill();
                ctx.setLineDash([6, 4]);
                ctx.strokeStyle = withAlpha(pal.accent, 0.90);
                ctx.lineWidth = 1.6;
                ctx.stroke();

                if (cPts.length === 4) {
                    ctx.setLineDash([3, 4]);
                    ctx.strokeStyle = withAlpha(pal.accent, 0.30);
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(cPts[0].x, cPts[0].y);
                    ctx.lineTo(cPts[2].x, cPts[2].y);
                    ctx.moveTo(cPts[1].x, cPts[1].y);
                    ctx.lineTo(cPts[3].x, cPts[3].y);
                    ctx.stroke();
                }
                ctx.restore();
            } else if (cPts.length === 2) {
                ctx.save();
                ctx.setLineDash([6, 4]);
                ctx.strokeStyle = withAlpha(pal.accent, 0.90);
                ctx.lineWidth = 1.6;
                ctx.beginPath();
                ctx.moveTo(cPts[0].x, cPts[0].y);
                ctx.lineTo(cPts[1].x, cPts[1].y);
                ctx.stroke();
                ctx.restore();
            }

            if (this._tool === "points" && this._hoverPos && !this._dragState && this._hoverVertexIdx === -1) {
                var lastPt = cPts[cPts.length - 1];
                ctx.save();
                ctx.setLineDash([3, 4]);
                ctx.strokeStyle = withAlpha(pal.accent, 0.42);
                ctx.lineWidth = 1.2;
                ctx.beginPath();
                ctx.moveTo(lastPt.x, lastPt.y);
                ctx.lineTo(this._hoverPos.x, this._hoverPos.y);
                ctx.stroke();
                ctx.restore();
            }

            for (var vi = 0; vi < cPts.length; vi += 1) {
                var pt = cPts[vi];
                var isActive = this._tool === "points" &&
                    ((this._dragState && this._dragState.target === "points" && this._dragState.index === vi) ||
                     this._hoverVertexIdx === vi);

                if (isActive) {
                    ctx.fillStyle = withAlpha(pal.accent, 0.22);
                    ctx.beginPath();
                    ctx.arc(pt.x, pt.y, 11, 0, Math.PI * 2);
                    ctx.fill();
                }

                ctx.fillStyle = pal.bgCard;
                ctx.strokeStyle = pal.accent;
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, isActive ? 7 : 5.5, 0, Math.PI * 2);
                ctx.fill();
                ctx.stroke();

                ctx.fillStyle = pal.accent;
                ctx.beginPath();
                ctx.arc(pt.x, pt.y, 2.2, 0, Math.PI * 2);
                ctx.fill();

                var badgeTxt = "P" + (vi + 1);
                ctx.save();
                ctx.font = "600 10.5px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
                var bPadX = 5;
                var bH = 17;
                var bW = ctx.measureText(badgeTxt).width + bPadX * 2;
                var bx = pt.x + 9;
                if (bx + bW > w - 6) bx = pt.x - bW - 9;
                bx = clamp(bx, 6, Math.max(6, w - bW - 6));
                var by = pt.y - bH - 6;
                if (by < 6) by = pt.y + 8;
                by = clamp(by, 6, Math.max(6, h - bH - 6));

                drawRoundedRectPath(ctx, bx, by, bW, bH, 4);
                ctx.fillStyle = withAlpha(pal.bgCard, 0.94);
                ctx.fill();
                ctx.lineWidth = 1;
                ctx.strokeStyle = isActive ? pal.accent : withAlpha(pal.accent, 0.72);
                ctx.stroke();

                ctx.fillStyle = pal.textPrimary;
                ctx.textAlign = "left";
                ctx.textBaseline = "middle";
                ctx.fillText(badgeTxt, bx + bPadX, by + bH * 0.5 + 0.5);
                ctx.restore();
            }
        }

        // Top-left compact HUD status pill when frame is loaded
        if (this._hasFrameDimensions() && w >= 220) {
            var hudText = this._tool === "parking"
                ? ("禁停区绘制 · 草稿 " + this._currentParkingDraft.length + " 点")
                : ("参考点标定 · 已选 " + this._imagePoints.length + " 点");
            hudText += " · " + this._imageWidth + "x" + this._imageHeight;
            drawPill(ctx, 10, 8, hudText, pal.bgCard, pal.hairline, pal.textSecondary, w);
        }

        ctx.restore();
    };

    CalibrationCanvas.prototype.loadImage = function (dataUrl, imgWidth, imgHeight) {
        if (this._destroyed) return;
        this._loadToken += 1;
        var token = this._loadToken;
        var reqW = toNum(imgWidth, 0);
        var reqH = toNum(imgHeight, 0);

        this._dragState = null;
        this._hoverVertexIdx = -1;

        if (!dataUrl || typeof dataUrl !== "string") {
            this._image = null;
            this._imageReady = false;
            this._imageWidth = reqW > 0 ? reqW : 0;
            this._imageHeight = reqH > 0 ? reqH : 0;
            this._measure();
            this._updateCursor();
            this._render();
            return;
        }

        this._image = null;
        this._imageReady = false;
        this._imageWidth = reqW > 0 ? reqW : 0;
        this._imageHeight = reqH > 0 ? reqH : 0;
        this._measure();
        this._updateCursor();
        this._render();

        if (typeof global.Image !== "function") {
            return;
        }

        var self = this;
        var img = new global.Image();
        img.onload = function () {
            if (self._destroyed || token !== self._loadToken) return;
            var natW = toNum(img.naturalWidth || img.width, 0);
            var natH = toNum(img.naturalHeight || img.height, 0);
            if (self._imageWidth <= 0 && natW > 0) self._imageWidth = natW;
            if (self._imageHeight <= 0 && natH > 0) self._imageHeight = natH;
            if (self._imageWidth <= 0) self._imageWidth = 1280;
            if (self._imageHeight <= 0) self._imageHeight = 720;
            self._image = img;
            self._imageReady = true;
            self._measure();
            self._updateCursor();
            self._render();
        };
        img.onerror = function () {
            if (self._destroyed || token !== self._loadToken) return;
            self._image = null;
            self._imageReady = false;
            self._render();
        };
        img.src = dataUrl;
    };

    CalibrationCanvas.prototype.setTool = function (tool) {
        if (this._destroyed) return;
        this._tool = tool === "parking" ? "parking" : "points";
        this._dragState = null;
        this._hoverVertexIdx = -1;
        this._updateCursor();
        this._render();
    };

    CalibrationCanvas.prototype.undo = function () {
        if (this._destroyed) return false;
        var list = this._activeVertexList();
        if (list.length === 0) return false;
        list.pop();
        this._dragState = null;
        this._hoverVertexIdx = -1;
        this._updateCursor();
        this._render();
        this._emitChange(true);
        return true;
    };

    CalibrationCanvas.prototype.finishParkingZone = function () {
        if (this._destroyed) return false;
        if (this._currentParkingDraft.length < 3) {
            return false;
        }
        this._parkingZonesImage.push(clonePointList(this._currentParkingDraft));
        this._currentParkingDraft = [];
        this._dragState = null;
        this._hoverVertexIdx = -1;
        this._updateCursor();
        this._render();
        this._emitChange(true);
        return true;
    };

    CalibrationCanvas.prototype.clearPoints = function () {
        if (this._destroyed) return;
        this._imagePoints = [];
        this._dragState = null;
        this._hoverVertexIdx = -1;
        this._updateCursor();
        this._render();
        this._emitChange(true);
    };

    CalibrationCanvas.prototype.clearParkingZones = function () {
        if (this._destroyed) return;
        this._parkingZonesImage = [];
        this._currentParkingDraft = [];
        this._dragState = null;
        this._hoverVertexIdx = -1;
        this._updateCursor();
        this._render();
        this._emitChange(true);
    };

    CalibrationCanvas.prototype.getState = function () {
        return {
            imagePoints: clonePointList(this._imagePoints),
            parkingZonesImage: clonePolygonList(this._parkingZonesImage),
            currentParkingDraft: clonePointList(this._currentParkingDraft),
            imageWidth: this._imageWidth,
            imageHeight: this._imageHeight
        };
    };

    CalibrationCanvas.prototype.setState = function (nextState) {
        if (this._destroyed || !nextState || typeof nextState !== "object") return;
        if (nextState.imagePoints !== undefined) {
            this._imagePoints = clonePointList(nextState.imagePoints);
        }
        if (nextState.parkingZonesImage !== undefined) {
            this._parkingZonesImage = clonePolygonList(nextState.parkingZonesImage);
        }
        if (nextState.currentParkingDraft !== undefined) {
            this._currentParkingDraft = clonePointList(nextState.currentParkingDraft);
        }
        if (isFiniteNum(nextState.imageWidth) && nextState.imageWidth > 0) {
            this._imageWidth = nextState.imageWidth;
        }
        if (isFiniteNum(nextState.imageHeight) && nextState.imageHeight > 0) {
            this._imageHeight = nextState.imageHeight;
        }
        this._dragState = null;
        this._hoverVertexIdx = -1;
        this._measure();
        this._updateCursor();
        this._render();
    };

    CalibrationCanvas.prototype.refreshTheme = function () {
        if (this._destroyed) return;
        this._palette = readThemePalette(this._probe);
        this._measure();
        this._render();
    };

    CalibrationCanvas.prototype.destroy = function () {
        if (this._destroyed) return;
        this._destroyed = true;
        this._loadToken += 1;
        this._image = null;
        if (this._emitTimer !== null && typeof global.clearTimeout === "function") {
            global.clearTimeout(this._emitTimer);
            this._emitTimer = null;
        }
        this._canvas.removeEventListener("pointerdown", this._onPointerDown);
        this._canvas.removeEventListener("pointermove", this._onPointerMove);
        this._canvas.removeEventListener("pointerup", this._onPointerUp);
        this._canvas.removeEventListener("pointercancel", this._onPointerCancel);
        this._canvas.removeEventListener("pointerleave", this._onPointerLeave);
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }
        this._unobserveTheme();
        this._motionTracker.destroy();
        if (this._probe && this._probe.parentNode) {
            this._probe.parentNode.removeChild(this._probe);
        }
    };

    /* ======================================================================
     * 6. AlertAudioSynth (Zero-Asset Web Audio L3 Instrumentation Chime)
     * ====================================================================== */
    function AlertAudioSynth() {
        if (!(this instanceof AlertAudioSynth)) {
            return new AlertAudioSynth();
        }
        this._enabled = false;
        this._ctx = null;
        this._lastChimeMs = -Infinity;
        this._lastRiskLevel = 0;
        this._destroyed = false;
    }

    AlertAudioSynth.prototype._ensureAudioContext = function () {
        if (this._destroyed) return null;
        if (!this._ctx) {
            var AudioCtx = global.AudioContext || global.webkitAudioContext;
            if (typeof AudioCtx !== "function") {
                return null;
            }
            try {
                this._ctx = new AudioCtx();
            } catch (_err) {
                this._ctx = null;
                return null;
            }
        }
        if (this._ctx && this._ctx.state === "suspended" && typeof this._ctx.resume === "function") {
            try {
                var p = this._ctx.resume();
                if (p && typeof p.catch === "function") {
                    p.catch(function () {});
                }
            } catch (_err2) {
                // Ignore resume failure
            }
        }
        return this._ctx;
    };

    AlertAudioSynth.prototype.setEnabled = function (enabled) {
        if (this._destroyed) return false;
        var next = Boolean(enabled);
        this._enabled = next;
        if (next) {
            this._ensureAudioContext();
        } else {
            this._lastRiskLevel = 0;
        }
        return this._enabled;
    };

    AlertAudioSynth.prototype.isEnabled = function () {
        return this._enabled;
    };

    AlertAudioSynth.prototype._playDualToneChime = function (ctx) {
        if (!ctx || ctx.state === "closed" ||
            typeof ctx.createOscillator !== "function" ||
            typeof ctx.createGain !== "function") {
            return false;
        }
        try {
            var t0 = typeof ctx.currentTime === "number" ? ctx.currentTime : 0;
            var notes = [
                { freq: 784.0, delay: 0.0, dur: 0.20, peak: 0.05 },
                { freq: 1046.5, delay: 0.095, dur: 0.26, peak: 0.055 }
            ];
            for (var i = 0; i < notes.length; i += 1) {
                var spec = notes[i];
                var osc = ctx.createOscillator();
                var gain = ctx.createGain();
                var startAt = t0 + spec.delay;
                var stopAt = startAt + spec.dur;

                osc.type = "sine";
                osc.frequency.setValueAtTime(spec.freq, startAt);

                gain.gain.setValueAtTime(0.0001, startAt);
                gain.gain.linearRampToValueAtTime(spec.peak, startAt + 0.014);
                gain.gain.exponentialRampToValueAtTime(0.0001, stopAt);

                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start(startAt);
                osc.stop(stopAt + 0.01);
            }
            return true;
        } catch (_err) {
            return false;
        }
    };

    AlertAudioSynth.prototype.triggerRisk = function (riskLevel) {
        if (this._destroyed) return false;
        if (!this._enabled) {
            this._lastRiskLevel = 0;
            return false;
        }

        var lvl = parseRiskLevel(riskLevel);
        var prevLvl = this._lastRiskLevel;
        this._lastRiskLevel = lvl;

        if (lvl < 3 || prevLvl >= 3) {
            return false;
        }

        var nowMs = Date.now();
        if (nowMs < this._lastChimeMs) {
            this._lastChimeMs = nowMs - 2500;
        }
        if (nowMs - this._lastChimeMs < 2500) {
            return false;
        }

        var ctx = this._ensureAudioContext();
        if (!ctx) return false;

        if (this._playDualToneChime(ctx)) {
            this._lastChimeMs = nowMs;
            return true;
        }
        return false;
    };

    AlertAudioSynth.prototype.destroy = function () {
        if (this._destroyed) return;
        this._destroyed = true;
        this._enabled = false;
        if (this._ctx && typeof this._ctx.close === "function" && this._ctx.state !== "closed") {
            try {
                var p = this._ctx.close();
                if (p && typeof p.catch === "function") {
                    p.catch(function () {});
                }
            } catch (_err) {
                // Ignore close failure
            }
        }
        this._ctx = null;
    };

    /* ======================================================================
     * 7. SimulatedCameraCanvas (Synchronized 3D Oblique Perspective Feed)
     * ====================================================================== */
    function SimulatedCameraCanvas(canvasEl, options) {
        if (!(this instanceof SimulatedCameraCanvas)) {
            return new SimulatedCameraCanvas(canvasEl, options);
        }
        if (!canvasEl) {
            throw new Error("SimulatedCameraCanvas requires a valid <canvas> element");
        }
        var opts = options || {};
        this._canvas = canvasEl;
        this._ctx = canvasEl.getContext("2d");
        this._destroyed = false;
        this._cssWidth = 0;
        this._cssHeight = 0;
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);
        this._lastState = opts.initialState || null;

        this._probe = createColorProbe();
        this._palette = readThemePalette(this._probe);

        var self = this;
        this._unobserveTheme = observeThemeChanges(function () {
            self.refreshTheme();
        });
        if (typeof ResizeObserver === "function") {
            this._resizeObserver = new ResizeObserver(function () {
                self._measure();
                self._render();
            });
            this._resizeObserver.observe(this._canvas);
        } else {
            this._resizeObserver = null;
        }
        this._measure();
        this._render();
    }

    SimulatedCameraCanvas.prototype._measure = function () {
        var rect = this._canvas.getBoundingClientRect();
        var cssW = Math.max(0, Math.round(rect.width || this._canvas.clientWidth || 0));
        var cssH = Math.max(0, Math.round(rect.height || this._canvas.clientHeight || 0));
        if (cssW < 2 || cssH < 2) {
            cssW = 540;
            cssH = 340;
        }
        this._dpr = Math.min(global.devicePixelRatio || 1, 2);
        var pxW = Math.round(cssW * this._dpr);
        var pxH = Math.round(cssH * this._dpr);
        if (this._canvas.width !== pxW || this._canvas.height !== pxH) {
            this._canvas.width = pxW;
            this._canvas.height = pxH;
        }
        this._cssWidth = cssW;
        this._cssHeight = cssH;
    };

    SimulatedCameraCanvas.prototype.renderState = function (state) {
        if (this._destroyed) return;
        this._lastState = state || null;
        this._render();
    };

    SimulatedCameraCanvas.prototype.refreshTheme = function () {
        if (this._destroyed) return;
        this._palette = readThemePalette(this._probe);
        this._measure();
        this._render();
    };

    SimulatedCameraCanvas.prototype._project3D = function (wx, wy, wz, cam) {
        var dx = wx - cam.x;
        var dy = wy - cam.y;
        var dz = wz - cam.z;
        var zCam = dy * cam.fwdY + dz * cam.fwdZ;
        if (zCam <= 0.45) return null;
        var xCam = dx;
        var yCam = dy * cam.upY + dz * cam.upZ;
        return {
            x: cam.cx + (xCam / zCam) * cam.focal,
            y: cam.cy - (yCam / zCam) * cam.focal,
            zCam: zCam
        };
    };

    SimulatedCameraCanvas.prototype._render = function () {
        if (!this._ctx || this._cssWidth < 2 || this._cssHeight < 2) return;
        var ctx = this._ctx;
        var pal = this._palette;
        var w = this._cssWidth;
        var h = this._cssHeight;
        var st = this._lastState || {
            bounds: { xMin: -3.2, xMax: 3.2, yMin: 0.0, yMax: 15.0 },
            parkingZones: [],
            vehicles: [],
            vrus: [],
            conflicts: [],
            maxRisk: 0,
            label: "SIM"
        };
        var b = st.bounds || { xMin: -3.2, xMax: 3.2, yMin: 0.0, yMax: 15.0 };

        // Oblique roadside CCTV camera calibrated looking along +Y across crosswalk
        var tgtY = 0.5 * (b.yMin + b.yMax);
        var camY = b.yMin - 6.2;
        var camZ = 5.1;
        var dY = tgtY - camY;
        var dZ = -camZ;
        var len = Math.hypot(dY, dZ) || 1;
        var fwdY = dY / len;
        var fwdZ = dZ / len;
        var cam = {
            x: 0.5 * (b.xMin + b.xMax),
            y: camY,
            z: camZ,
            fwdY: fwdY,
            fwdZ: fwdZ,
            upY: -fwdZ,
            upZ: fwdY,
            cx: w * 0.5,
            cy: h * 0.49,
            focal: Math.min(w * 0.86, h * 1.48)
        };

        var self = this;
        function p3(wx, wy, wz) {
            return self._project3D(wx, wy, wz || 0, cam);
        }

        function fillPolyPts(worldPts, fillColor, strokeColor, lineWidth, dashed) {
            var pts = [];
            for (var i = 0; i < worldPts.length; i += 1) {
                var pt = p3(worldPts[i][0], worldPts[i][1], worldPts[i][2] || 0);
                if (!pt) return;
                pts.push(pt);
            }
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(pts[0].x, pts[0].y);
            for (var j = 1; j < pts.length; j += 1) {
                ctx.lineTo(pts[j].x, pts[j].y);
            }
            ctx.closePath();
            if (fillColor) {
                ctx.fillStyle = fillColor;
                ctx.fill();
            }
            if (strokeColor) {
                if (dashed) ctx.setLineDash([5, 4]);
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = lineWidth || 1;
                ctx.stroke();
            }
            ctx.restore();
        }

        ctx.save();
        ctx.scale(this._dpr, this._dpr);
        ctx.clearRect(0, 0, w, h);

        // Background sky / campus horizon band + distant ground apron
        ctx.fillStyle = pal.isLight ? "#dde1e8" : "#101012";
        ctx.fillRect(0, 0, w, h);

        var horizonPt = p3(cam.x, b.yMax + 80, 0);
        var horizonY = horizonPt ? clamp(horizonPt.y, 18, h * 0.32) : h * 0.22;
        ctx.fillStyle = pal.roadShoulder;
        ctx.fillRect(0, horizonY, w, h - horizonY);

        // Crisp campus horizon curb + distant light pole silhouettes
        ctx.strokeStyle = pal.hairline;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, horizonY);
        ctx.lineTo(w, horizonY);
        ctx.stroke();

        var poleXs = [b.xMin - 0.6, b.xMax + 0.6];
        var poleYs = [5.5, 11.8];
        for (var pxi = 0; pxi < poleXs.length; pxi += 1) {
            for (var pyi = 0; pyi < poleYs.length; pyi += 1) {
                var pb = p3(poleXs[pxi], poleYs[pyi], 0);
                var ptTop = p3(poleXs[pxi], poleYs[pyi], 4.0);
                var ptArm = p3(poleXs[pxi] * 0.78, poleYs[pyi], 3.9);
                if (pb && ptTop && ptArm) {
                    ctx.strokeStyle = withAlpha(pal.textMuted, 0.34);
                    ctx.lineWidth = 1.6;
                    ctx.beginPath();
                    ctx.moveTo(pb.x, pb.y);
                    ctx.lineTo(ptTop.x, ptTop.y);
                    ctx.lineTo(ptArm.x, ptArm.y);
                    ctx.stroke();
                }
            }
        }

        // Far-extending asphalt carriageway & sidewalks
        var farY = b.yMax + 10.0;
        var nearY = Math.max(camY + 1.2, b.yMin - 1.5);
        fillPolyPts(
            [[b.xMin - 1.8, nearY], [b.xMax + 1.8, nearY], [b.xMax + 1.8, farY], [b.xMin - 1.8, farY]],
            withAlpha(pal.bgRaised, pal.isLight ? 0.85 : 0.7),
            null,
            0,
            false
        );
        fillPolyPts(
            [[b.xMin, nearY], [b.xMax, nearY], [b.xMax, farY], [b.xMin, farY]],
            pal.roadSurface,
            pal.roadCurb,
            1.4,
            false
        );

        // Perspective dashed center divider
        var cxMid = 0.5 * (b.xMin + b.xMax);
        ctx.save();
        ctx.strokeStyle = pal.roadMark;
        ctx.lineWidth = 1.4;
        for (var ly = nearY + 0.4; ly < farY - 0.8; ly += 1.8) {
            var d0 = p3(cxMid, ly, 0);
            var d1 = p3(cxMid, Math.min(farY, ly + 0.9), 0);
            if (d0 && d1) {
                ctx.beginPath();
                ctx.moveTo(d0.x, d0.y);
                ctx.lineTo(d1.x, d1.y);
                ctx.stroke();
            }
        }
        ctx.restore();

        // Crosswalk perspective zebra stripes (y = 7.25 .. 8.45)
        if (7.8 >= b.yMin + 0.4 && 7.8 <= b.yMax - 0.4) {
            var cwTint = withAlpha(pal.textPrimary, pal.isLight ? 0.16 : 0.12);
            for (var zx = b.xMin + 0.4; zx < b.xMax - 0.35; zx += 0.72) {
                fillPolyPts(
                    [[zx, 7.25], [Math.min(zx + 0.38, b.xMax - 0.3), 7.25], [Math.min(zx + 0.38, b.xMax - 0.3), 8.45], [zx, 8.45]],
                    cwTint,
                    null,
                    0,
                    false
                );
            }
        }

        // Calibrated 4-point homography ROI trapezoid overlay
        var roiPadX = 0.32;
        var roiY0 = b.yMin + 0.8;
        var roiY1 = b.yMax - 0.8;
        var roiWorld = [
            [b.xMin + roiPadX, roiY1, 0],
            [b.xMax - roiPadX, roiY1, 0],
            [b.xMax - roiPadX, roiY0, 0],
            [b.xMin + roiPadX, roiY0, 0]
        ];
        fillPolyPts(roiWorld, withAlpha(pal.accent, 0.045), withAlpha(pal.accent, 0.46), 1.2, false);
        for (var rci = 0; rci < roiWorld.length; rci += 1) {
            var rpt = p3(roiWorld[rci][0], roiWorld[rci][1], 0);
            if (rpt) {
                ctx.fillStyle = pal.accent;
                ctx.beginPath();
                ctx.arc(rpt.x, rpt.y, 2.5, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        // Roadside parking exclusion zone projected onto pavement
        var zones = st.parkingZones || [];
        for (var zi = 0; zi < zones.length; zi += 1) {
            var poly = zones[zi];
            if (!Array.isArray(poly) || poly.length < 3) continue;
            var worldPoly = poly.map(function (pt) { return [pt[0], pt[1], 0]; });
            fillPolyPts(worldPoly, withAlpha(pal.parkingZone, 0.10), withAlpha(pal.parkingZone, 0.72), 1.3, true);
        }

        // Future CTRV trajectory arcs projected onto pavement
        var vehicles = Array.isArray(st.vehicles) ? st.vehicles : [];
        for (var vi = 0; vi < vehicles.length; vi += 1) {
            var veh = vehicles[vi];
            var spdMs = Math.hypot(veh.vx || 0, veh.vy || 0);
            if (spdMs < 0.35 || veh.inParkingZone) continue;
            var arcPts = predictCtrvArc(veh, 2.5, 16);
            var arcColor = veh.riskLevel >= 1 ? riskColor(pal, veh.riskLevel) : pal.safe;
            ctx.save();
            ctx.setLineDash([5, 4]);
            ctx.strokeStyle = withAlpha(arcColor, 0.86);
            ctx.lineWidth = 1.7;
            ctx.beginPath();
            var startedArc = false;
            for (var ai = 0; ai < arcPts.length; ai += 1) {
                var aProj = p3(arcPts[ai].x, arcPts[ai].y, 0.02);
                if (!aProj) continue;
                if (!startedArc) {
                    ctx.moveTo(aProj.x, aProj.y);
                    startedArc = true;
                } else {
                    ctx.lineTo(aProj.x, aProj.y);
                }
            }
            if (startedArc) ctx.stroke();
            ctx.restore();
        }

        // Projected TTC convergence link & ground ring
        var vrus = Array.isArray(st.vrus) ? st.vrus : [];
        var conflicts = Array.isArray(st.conflicts) ? st.conflicts : [];
        for (var ci = 0; ci < conflicts.length; ci += 1) {
            var cf = conflicts[ci];
            if ((cf.riskLevel || 0) < 2) continue;
            var cfColor = riskColor(pal, cf.riskLevel);
            var vObj = findById(vehicles, cf.vehTrackId);
            var uObj = findById(vrus, cf.pedTrackId);
            if (vObj && uObj) {
                var vp = p3(vObj.x, vObj.y, 0.05);
                var up = p3(uObj.x, uObj.y, 0.05);
                if (vp && up) {
                    ctx.save();
                    ctx.setLineDash([4, 3]);
                    ctx.strokeStyle = withAlpha(cfColor, 0.85);
                    ctx.lineWidth = 1.4;
                    ctx.beginPath();
                    ctx.moveTo(vp.x, vp.y);
                    ctx.lineTo(up.x, up.y);
                    ctx.stroke();
                    ctx.restore();
                }
            }
            var cxy = cf.conflictGroundXY;
            if ((!Array.isArray(cxy) || cxy.length < 2) && vObj && uObj) {
                cxy = [(vObj.x + uObj.x) * 0.5, (vObj.y + uObj.y) * 0.5];
            }
            if (Array.isArray(cxy) && cxy.length >= 2) {
                var ringPts = [];
                for (var ang = 0; ang < 16; ang += 1) {
                    var theta = (ang / 16) * Math.PI * 2;
                    var rp = p3(cxy[0] + Math.cos(theta) * 0.65, cxy[1] + Math.sin(theta) * 0.65, 0.03);
                    if (rp) ringPts.push(rp);
                }
                if (ringPts.length >= 6) {
                    ctx.strokeStyle = cfColor;
                    ctx.lineWidth = 1.5;
                    ctx.beginPath();
                    ctx.moveTo(ringPts[0].x, ringPts[0].y);
                    for (var ri = 1; ri < ringPts.length; ri += 1) {
                        ctx.lineTo(ringPts[ri].x, ringPts[ri].y);
                    }
                    ctx.closePath();
                    ctx.stroke();
                }
            }
        }

        // Depth-sort actors far-to-near and project 3D boxes + 2D YOLO detection overlays
        var renderActors = [];
        for (var vIdx = 0; vIdx < vehicles.length; vIdx += 1) {
            renderActors.push({ kind: "veh", item: vehicles[vIdx], yDepth: vehicles[vIdx].y || 0 });
        }
        for (var uIdx = 0; uIdx < vrus.length; uIdx += 1) {
            renderActors.push({ kind: "vru", item: vrus[uIdx], yDepth: vrus[uIdx].y || 0 });
        }
        renderActors.sort(function (a, bObj) { return bObj.yDepth - a.yDepth; });

        var placedLabels = [];
        var safeRect = { minX: 6, minY: 30, maxX: w - 6, maxY: h - 6 };

        for (var ai2 = 0; ai2 < renderActors.length; ai2 += 1) {
            var entry = renderActors[ai2];
            if (entry.kind === "veh") {
                var v = entry.item;
                var vColor = v.inParkingZone
                    ? pal.textMuted
                    : (v.riskLevel >= 1 ? riskColor(pal, v.riskLevel) : pal.safe);
                var hl = (v.lengthM || 4.5) * 0.5;
                var hw = (v.widthM || 1.9) * 0.5;
                var vh = v.kind === "bus" ? 2.85 : (v.kind === "truck" ? 2.55 : 1.48);
                var hRad = v.headingRad !== undefined ? v.headingRad : Math.PI * 0.5;
                var cosH = Math.cos(hRad);
                var sinH = Math.sin(hRad);
                var cornersWorld = [
                    [v.x + hl * cosH - hw * sinH, v.y + hl * sinH + hw * cosH],
                    [v.x + hl * cosH + hw * sinH, v.y + hl * sinH - hw * cosH],
                    [v.x - hl * cosH + hw * sinH, v.y - hl * sinH - hw * cosH],
                    [v.x - hl * cosH - hw * sinH, v.y - hl * sinH + hw * cosH]
                ];
                var basePx = [];
                var topPx = [];
                var allValid = true;
                for (var ci2 = 0; ci2 < 4; ci2 += 1) {
                    var bp = p3(cornersWorld[ci2][0], cornersWorld[ci2][1], 0);
                    var tp = p3(cornersWorld[ci2][0], cornersWorld[ci2][1], vh);
                    if (!bp || !tp) {
                        allValid = false;
                        break;
                    }
                    basePx.push(bp);
                    topPx.push(tp);
                }
                if (!allValid) continue;

                // Subtle 3D wireframe cabin shell + roof tint
                ctx.save();
                ctx.fillStyle = withAlpha(vColor, v.inParkingZone ? 0.12 : 0.16);
                ctx.strokeStyle = withAlpha(vColor, 0.72);
                ctx.lineWidth = 1.1;
                ctx.beginPath();
                ctx.moveTo(topPx[0].x, topPx[0].y);
                for (var ti2 = 1; ti2 < 4; ti2 += 1) ctx.lineTo(topPx[ti2].x, topPx[ti2].y);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();

                for (var ei = 0; ei < 4; ei += 1) {
                    ctx.beginPath();
                    ctx.moveTo(basePx[ei].x, basePx[ei].y);
                    ctx.lineTo(topPx[ei].x, topPx[ei].y);
                    ctx.stroke();
                }
                ctx.restore();

                // Compute 2D YOLO detection bbox around projected 3D box
                var allBoxPts = basePx.concat(topPx);
                var minBx = Infinity;
                var minBy = Infinity;
                var maxBx = -Infinity;
                var maxBy = -Infinity;
                for (var bi = 0; bi < allBoxPts.length; bi += 1) {
                    if (allBoxPts[bi].x < minBx) minBx = allBoxPts[bi].x;
                    if (allBoxPts[bi].y < minBy) minBy = allBoxPts[bi].y;
                    if (allBoxPts[bi].x > maxBx) maxBx = allBoxPts[bi].x;
                    if (allBoxPts[bi].y > maxBy) maxBy = allBoxPts[bi].y;
                }
                ctx.strokeStyle = vColor;
                ctx.lineWidth = v.riskLevel >= 2 ? 2.0 : 1.5;
                ctx.strokeRect(minBx, minBy, maxBx - minBx, maxBy - minBy);

                // Ground anchor dot
                var anchorPt = p3(v.x, v.y, 0);
                if (anchorPt) {
                    ctx.fillStyle = pal.accent;
                    ctx.beginPath();
                    ctx.arc(anchorPt.x, anchorPt.y, 3, 0, Math.PI * 2);
                    ctx.fill();
                }

                var spdTxt = isFiniteNum(v.speedKmh) ? v.speedKmh.toFixed(1) + " km/h" : "0.0 km/h";
                var statusSuffix = v.riskLevel >= 3 ? " [ALERT]" : (v.riskLevel === 2 ? " [CAUTION]" : (v.riskLevel === 1 ? " [SPEEDING]" : ""));
                var vLabel = "V-" + padId(v.trackId) + " | " + spdTxt + statusSuffix;
                var vBg = v.riskLevel >= 3 ? pal.critical : pal.bgCard;
                var vFg = v.riskLevel >= 3 ? "#ffffff" : pal.textPrimary;
                placePill(
                    ctx,
                    { cx: (minBx + maxBx) * 0.5, cy: minBy - 10, rx: 0, ry: 2 },
                    vLabel,
                    vBg,
                    vColor,
                    vFg,
                    safeRect,
                    placedLabels,
                    []
                );
            } else {
                var u = entry.item;
                var isBike = u.kind === "bicycle" || u.category === "bicycle" || u.kind === "motorcycle";
                var uColor = u.riskLevel >= 2 ? riskColor(pal, u.riskLevel) : (isBike ? pal.accent : pal.safe);
                var uw = isBike ? 0.55 : 0.36;
                var uh = isBike ? 1.42 : 1.72;
                var uBaseL = p3(u.x - uw, u.y, 0);
                var uBaseR = p3(u.x + uw, u.y, 0);
                var uTopL = p3(u.x - uw, u.y, uh);
                var uTopR = p3(u.x + uw, u.y, uh);
                var uAnchor = p3(u.x, u.y, 0);
                if (!uBaseL || !uBaseR || !uTopL || !uTopR) continue;

                var bx1 = Math.min(uBaseL.x, uTopL.x);
                var bx2 = Math.max(uBaseR.x, uTopR.x);
                var by1 = Math.min(uTopL.y, uTopR.y);
                var by2 = Math.max(uBaseL.y, uBaseR.y);
                var bw = Math.max(14, bx2 - bx1);
                var bh = Math.max(24, by2 - by1);

                ctx.fillStyle = withAlpha(uColor, 0.14);
                ctx.fillRect(bx1, by1, bw, bh);
                ctx.strokeStyle = uColor;
                ctx.lineWidth = u.riskLevel >= 2 ? 1.8 : 1.4;
                ctx.strokeRect(bx1, by1, bw, bh);

                if (uAnchor) {
                    ctx.fillStyle = pal.accent;
                    ctx.beginPath();
                    ctx.arc(uAnchor.x, uAnchor.y, 2.5, 0, Math.PI * 2);
                    ctx.fill();
                }

                var uTag = (isBike ? "B-" : "P-") + padId(u.trackId);
                if (isBike && Math.hypot(u.vx || 0, u.vy || 0) > 0.2) {
                    uTag += " " + (Math.hypot(u.vx || 0, u.vy || 0) * 3.6).toFixed(1) + "km/h";
                }
                placePill(
                    ctx,
                    { cx: bx1 + bw * 0.5, cy: by1 - 10, rx: 0, ry: 2 },
                    uTag,
                    u.riskLevel >= 3 ? pal.critical : pal.bgCard,
                    uColor,
                    u.riskLevel >= 3 ? "#ffffff" : pal.textPrimary,
                    safeRect,
                    placedLabels,
                    []
                );
            }
        }

        // Top overlay status bar (clean corner pills)
        var camTitle = w < 410 ? "CAM-01 · 3D 透视仿真" : "CAM-01 · 3D 透视仿真视角 · 标定 6.0m x 15.0m";
        drawPill(ctx, 10, 8, camTitle, pal.bgCard, pal.hairline, pal.textSecondary, w);

        var maxRisk = st.maxRisk || 0;
        var riskTxt = maxRisk === 3 ? "L3 CRITICAL" : (maxRisk === 2 ? "L2 CAUTION" : (maxRisk === 1 ? "L1 SPEEDING" : "L0 SAFE"));
        var riskCol = riskColor(pal, maxRisk);
        drawPill(
            ctx,
            Math.max(10, w - 104),
            8,
            riskTxt,
            maxRisk === 3 ? pal.critical : pal.bgCard,
            riskCol,
            maxRisk === 3 ? "#ffffff" : riskCol,
            w
        );

        ctx.restore();
    };

    SimulatedCameraCanvas.prototype.destroy = function () {
        if (this._destroyed) return;
        this._destroyed = true;
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }
        this._unobserveTheme();
        if (this._probe && this._probe.parentNode) {
            this._probe.parentNode.removeChild(this._probe);
        }
    };

    /* ======================================================================
     * Public global export
     * ====================================================================== */
    global.CampusMotion = {
        RollingCounter: RollingCounter,
        AnimatedSignalPipeline: AnimatedSignalPipeline,
        CampusRoadTwinCanvas: CampusRoadTwinCanvas,
        SpeedTimelineChart: SpeedTimelineChart,
        CalibrationCanvas: CalibrationCanvas,
        AlertAudioSynth: AlertAudioSynth,
        SimulatedCameraCanvas: SimulatedCameraCanvas,
        _buildScenarioState: buildScenarioState
    };
})(typeof window !== "undefined" ? window : globalThis);
